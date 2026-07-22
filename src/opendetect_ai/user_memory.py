"""
用户长期记忆模块 —— OpenDetect_AI
在同一个 chat_history.db 里维护一张 user_profile 表，
记录用户跨会话的研究偏好、常用领域、感兴趣的论文方向。

表结构（按 user_id 隔离，多用户互不串画像）：
    user_profile (
        user_id    TEXT,             -- 用户标识，前端 localStorage 生成并随请求传入
        key        TEXT,             -- 偏好类型，如 'research_interests'
        value      TEXT,             -- JSON 序列化的内容
        updated_at TEXT,             -- ISO 时间戳
        PRIMARY KEY (user_id, key)
    )

三种"记忆"要分清（面试易被追问）：
    - Checkpointer(SqliteSaver)：工作流/会话状态，按 thread_id
    - user_profile（本模块）    ：跨会话用户偏好，按 user_id
    - Chroma                    ：论文知识库，全局共享

设计原则：
- 每次对话结束后异步提取偏好，不阻塞主流程
- 新会话开始时按 user_id 读取偏好，注入到 Supervisor prompt
- 提取失败静默处理，不影响正常对话
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


_profile_locks: dict[str, threading.Lock] = {}
_profile_locks_guard = threading.Lock()


def _profile_lock(user_id: str) -> threading.Lock:
    with _profile_locks_guard:
        return _profile_locks.setdefault(user_id, threading.Lock())


# ── 存储路径与表结构 ────────────────────────────────────────────
def _get_db_path() -> str:
    from opendetect_ai.env_utils import CHROMA_PERSIST_DIR
    import os
    return os.path.join(os.path.dirname(CHROMA_PERSIST_DIR), "chat_history.db")


def _create_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_profile (
            user_id    TEXT NOT NULL DEFAULT 'default',
            key        TEXT NOT NULL,
            value      TEXT NOT NULL,
            source     TEXT NOT NULL DEFAULT 'conversation',
            updated_at TEXT NOT NULL,
            PRIMARY KEY (user_id, key)
        )
    """)


def _ensure_table(conn: sqlite3.Connection) -> None:
    """建表；若检测到旧结构（只有 key 作主键、无 user_id 列）则平滑迁移到 'default'。"""
    cols = [r[1] for r in conn.execute("PRAGMA table_info(user_profile)").fetchall()]
    if cols and "user_id" not in cols:
        # 旧表迁移：历史画像全部归到 user_id='default'
        conn.execute("ALTER TABLE user_profile RENAME TO _user_profile_old")
        _create_table(conn)
        conn.execute(
            "INSERT OR IGNORE INTO user_profile (user_id, key, value, updated_at) "
            "SELECT 'default', key, value, updated_at FROM _user_profile_old"
        )
        conn.execute("DROP TABLE _user_profile_old")
    else:
        _create_table(conn)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(user_profile)").fetchall()]
    if "source" not in cols:
        conn.execute(
            "ALTER TABLE user_profile ADD COLUMN source TEXT NOT NULL DEFAULT 'conversation'"
        )
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_settings (
            user_id    TEXT PRIMARY KEY,
            enabled    INTEGER NOT NULL DEFAULT 1,
            ttl_days   INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        )
    """)
    conn.commit()


# ── 读写接口 ────────────────────────────────────────────────────
def load_user_profile(user_id: str = "default") -> dict:
    """
    按 user_id 读取用户画像，返回结构化字典。
    失败时返回空字典，不抛异常。
    """
    try:
        conn = sqlite3.connect(_get_db_path(), check_same_thread=False)
        _ensure_table(conn)
        settings = _get_memory_settings_conn(conn, user_id)
        if not settings["enabled"]:
            conn.close()
            return {}
        rows = conn.execute(
            "SELECT key, value, updated_at FROM user_profile WHERE user_id = ?", (user_id,)
        ).fetchall()
        conn.close()
        cutoff = None
        if settings["ttl_days"] > 0:
            cutoff = datetime.now(timezone.utc).timestamp() - settings["ttl_days"] * 86400
        profile = {}
        for key, value, updated_at in rows:
            if cutoff is not None and _parse_timestamp(updated_at) < cutoff:
                continue
            profile[key] = json.loads(value)
        return profile
    except Exception:
        return {}


def save_user_profile(user_id: str, updates: dict, source: str = "conversation") -> None:
    """
    保存/更新指定 user_id 的画像字段。
    updates 的每个 key-value 会独立 upsert，其他字段不受影响。
    """
    try:
        conn = sqlite3.connect(_get_db_path(), check_same_thread=False)
        _ensure_table(conn)
        if not _get_memory_settings_conn(conn, user_id)["enabled"]:
            conn.close()
            return
        now = datetime.now(timezone.utc).isoformat()
        for key, value in updates.items():
            conn.execute(
                "INSERT INTO user_profile (user_id, key, value, source, updated_at) VALUES (?, ?, ?, ?, ?)"
                " ON CONFLICT(user_id, key) DO UPDATE SET value=excluded.value,"
                " source=excluded.source, updated_at=excluded.updated_at",
                (user_id, key, json.dumps(value, ensure_ascii=False), source, now)
            )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[UserMemory] 保存失败（静默）: {e}")


def _parse_timestamp(value: str) -> float:
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except (TypeError, ValueError):
        return 0.0


def _get_memory_settings_conn(conn: sqlite3.Connection, user_id: str) -> dict:
    row = conn.execute(
        "SELECT enabled, ttl_days, updated_at FROM memory_settings WHERE user_id = ?",
        (user_id,),
    ).fetchone()
    if not row:
        return {"enabled": True, "ttl_days": 0, "updated_at": None}
    return {"enabled": bool(row[0]), "ttl_days": max(0, row[1]), "updated_at": row[2]}


def get_memory_settings(user_id: str = "default") -> dict:
    try:
        conn = sqlite3.connect(_get_db_path(), check_same_thread=False)
        _ensure_table(conn)
        settings = _get_memory_settings_conn(conn, user_id)
        conn.close()
        return settings
    except Exception:
        return {"enabled": True, "ttl_days": 0, "updated_at": None}


def set_memory_settings(
    user_id: str,
    *,
    enabled: bool | None = None,
    ttl_days: int | None = None,
) -> dict:
    """更新用户记忆开关和 TTL；ttl_days=0 表示不过期。"""
    conn = sqlite3.connect(_get_db_path(), check_same_thread=False)
    _ensure_table(conn)
    current = _get_memory_settings_conn(conn, user_id)
    next_enabled = current["enabled"] if enabled is None else bool(enabled)
    next_ttl = current["ttl_days"] if ttl_days is None else max(0, min(ttl_days, 3650))
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO memory_settings (user_id, enabled, ttl_days, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
          enabled=excluded.enabled, ttl_days=excluded.ttl_days, updated_at=excluded.updated_at
        """,
        (user_id, int(next_enabled), next_ttl, now),
    )
    conn.commit()
    conn.close()
    return {"enabled": next_enabled, "ttl_days": next_ttl, "updated_at": now}


def list_memory_entries(user_id: str = "default") -> list[dict]:
    """列出记忆值、来源与更新时间，供用户查看数据从哪里来。"""
    try:
        conn = sqlite3.connect(_get_db_path(), check_same_thread=False)
        _ensure_table(conn)
        rows = conn.execute(
            "SELECT key, value, source, updated_at FROM user_profile WHERE user_id = ?"
            " ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()
        conn.close()
        return [
            {"key": key, "value": json.loads(value), "source": source, "updated_at": updated_at}
            for key, value, source, updated_at in rows
        ]
    except Exception:
        return []



# ── 格式化为 prompt 可用的字符串 ──────────────────────────────
def format_profile_for_prompt(profile: dict) -> str:
    """
    把用户画像格式化为自然语言，供 Supervisor prompt 使用。
    如果画像为空，返回空字符串。
    """
    if not profile:
        return ""

    lines = []
    interests = profile.get("research_interests", [])
    if interests:
        lines.append(f"- 研究方向：{', '.join(interests)}")

    frequent_queries = profile.get("frequent_queries", [])
    if frequent_queries:
        lines.append(f"- 常搜话题：{', '.join(frequent_queries[:5])}")

    preferred_venues = profile.get("preferred_venues", [])
    if preferred_venues:
        lines.append(f"- 关注来源：{', '.join(preferred_venues)}")

    last_topics = profile.get("last_session_topics", [])
    if last_topics:
        lines.append(f"- 上次会话关注：{', '.join(last_topics)}")

    if not lines:
        return ""

    return "## 用户长期偏好（跨会话记忆）\n" + "\n".join(lines)


# ── LLM 提取偏好 ───────────────────────────────────────────────
_EXTRACT_PROMPT = """你是一个用户偏好分析助手。
请从以下对话历史中提取用户的研究偏好，输出 JSON，不要有任何其他内容。

## 对话历史
{conversation}

## 输出格式
{{
  "research_interests": ["领域1", "领域2"],      // 用户关注的研究方向（3个以内，中文）
  "frequent_queries":   ["话题1", "话题2"],      // 用户反复询问的具体话题（5个以内）
  "preferred_venues":   ["arxiv", "CVPR"],       // 用户提到或偏好的论文来源/会议
  "last_session_topics": ["话题1", "话题2"]      // 本次会话的核心话题（3个以内）
}}

注意：
- 只提取明确出现的信息，不要推断
- 如果某个字段没有相关信息，输出空列表 []
- research_interests 和 last_session_topics 使用中文
"""


def _extract_and_save_profile_unlocked(
    messages: list,
    llm_model: str,
    llm_base_url: str,
    llm_api_key: str,
    user_id: str = "default",
) -> None:
    """
    从对话历史中提取用户偏好并持久化到指定 user_id 的画像。
    在对话结束后调用，失败时静默处理。

    Args:
        messages: AgentState.messages 列表
        llm_model/base_url/api_key: LLM 配置
        user_id: 用户标识，画像按此隔离
    """
    if not messages:
        return
    if not get_memory_settings(user_id)["enabled"]:
        return

    # 只取面向用户的消息对，过滤路由日志
    _SKIP = ("Supervisor 决策:", "入库完成，新增", "找到 ", "没有待入库", "所有论文均已入库")
    conv_lines = []
    from langchain_core.messages import HumanMessage as HM, AIMessage as AM
    for msg in messages:
        if isinstance(msg, HM):
            conv_lines.append(f"用户：{msg.content.strip()}")
        elif isinstance(msg, AM):
            c = msg.content.strip()
            if c and not any(c.startswith(p) for p in _SKIP):
                conv_lines.append(f"助手：{c[:200]}")  # 截断避免超长

    if len(conv_lines) < 2:
        return  # 对话太短，不值得提取

    conversation = "\n".join(conv_lines)
    try:
        llm = ChatOpenAI(
            model=llm_model,
            base_url=llm_base_url,
            api_key=llm_api_key,
            temperature=0,
        )
        response = llm.invoke([HumanMessage(
            content=_EXTRACT_PROMPT.format(conversation=conversation)
        )])
        raw = response.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        extracted = json.loads(raw)

        # 合并而非覆盖：保留历史偏好，用新数据补充
        existing = load_user_profile(user_id)
        merged = {}
        for key in ("research_interests", "frequent_queries", "preferred_venues"):
            old_vals = existing.get(key, [])
            new_vals = extracted.get(key, [])
            # 去重合并，新数据优先，保留最多10条
            combined = list(dict.fromkeys(new_vals + old_vals))[:10]
            merged[key] = combined
        # last_session_topics 每次直接覆盖
        merged["last_session_topics"] = extracted.get("last_session_topics", [])

        save_user_profile(user_id, merged, source="conversation_extract")
        print(f"[UserMemory] 已更新用户偏好: {list(merged.keys())}")

    except Exception as e:
        print(f"[UserMemory] 提取失败（静默）: {e}")


def extract_and_save_profile(
    messages: list,
    llm_model: str,
    llm_base_url: str,
    llm_api_key: str,
    user_id: str = "default",
) -> None:
    """同一用户串行提取，避免并发 read-modify-write 覆盖偏好。"""
    with _profile_lock(user_id):
        _extract_and_save_profile_unlocked(
            messages, llm_model, llm_base_url, llm_api_key, user_id
        )
