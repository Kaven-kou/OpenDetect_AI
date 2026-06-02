"""
用户长期记忆模块 —— OpenDetect_AI
在同一个 chat_history.db 里维护一张 user_profile 表，
记录用户跨会话的研究偏好、常用领域、感兴趣的论文方向。

表结构：
    user_profile (
        key   TEXT PRIMARY KEY,   -- 偏好类型，如 'research_interests'
        value TEXT,               -- JSON 序列化的内容
        updated_at TEXT           -- ISO 时间戳
    )

设计原则：
- 每次对话结束后异步提取偏好，不阻塞主流程
- 新会话开始时读取偏好，注入到 Supervisor prompt
- 提取失败静默处理，不影响正常对话
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


# ── 存储路径与表结构 ────────────────────────────────────────────
def _get_db_path() -> str:
    from opendetect_ai.env_utils import CHROMA_PERSIST_DIR
    import os
    return os.path.join(os.path.dirname(CHROMA_PERSIST_DIR), "chat_history.db")


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_profile (
            key        TEXT PRIMARY KEY,
            value      TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.commit()


# ── 读写接口 ────────────────────────────────────────────────────
def load_user_profile() -> dict:
    """
    读取用户画像，返回结构化字典。
    失败时返回空字典，不抛异常。
    """
    try:
        conn = sqlite3.connect(_get_db_path(), check_same_thread=False)
        _ensure_table(conn)
        rows = conn.execute("SELECT key, value FROM user_profile").fetchall()
        conn.close()
        return {row[0]: json.loads(row[1]) for row in rows}
    except Exception:
        return {}


def save_user_profile(updates: dict) -> None:
    """
    保存/更新用户画像字段。
    updates 的每个 key-value 会独立 upsert，其他字段不受影响。
    """
    try:
        conn = sqlite3.connect(_get_db_path(), check_same_thread=False)
        _ensure_table(conn)
        now = datetime.utcnow().isoformat()
        for key, value in updates.items():
            conn.execute(
                "INSERT INTO user_profile (key, value, updated_at) VALUES (?, ?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (key, json.dumps(value, ensure_ascii=False), now)
            )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[UserMemory] 保存失败（静默）: {e}")


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


def extract_and_save_profile(
    messages: list,
    llm_model: str,
    llm_base_url: str,
    llm_api_key: str,
) -> None:
    """
    从对话历史中提取用户偏好并持久化。
    在对话结束后调用，失败时静默处理。

    Args:
        messages: AgentState.messages 列表
        llm_model/base_url/api_key: LLM 配置
    """
    if not messages:
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
        existing = load_user_profile()
        merged = {}
        for key in ("research_interests", "frequent_queries", "preferred_venues"):
            old_vals = existing.get(key, [])
            new_vals = extracted.get(key, [])
            # 去重合并，新数据优先，保留最多10条
            combined = list(dict.fromkeys(new_vals + old_vals))[:10]
            merged[key] = combined
        # last_session_topics 每次直接覆盖
        merged["last_session_topics"] = extracted.get("last_session_topics", [])

        save_user_profile(merged)
        print(f"[UserMemory] 已更新用户偏好: {list(merged.keys())}")

    except Exception as e:
        print(f"[UserMemory] 提取失败（静默）: {e}")