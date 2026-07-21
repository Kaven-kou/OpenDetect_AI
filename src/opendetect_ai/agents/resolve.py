"""
Resolve Query Agent —— OpenDetect_AI
每轮对话入口的「上游查询解析」节点：把省略/指代/确认式输入解析成自包含的 resolved_query，
让下游 Supervisor / Search / RAG 只面对干净、自包含的问题。

设计边界（刻意克制，不顺带重构路由）：
- 每轮只在入口执行一次：START → resolve_query → supervisor；子 Agent 回 Supervisor 不再经过这里。
- 只新增 resolved_query，绝不覆写 user_query（原始输入保留，供日志/评测/后续迁移 TaskSpec）。
- 普通、自包含问题直接透传，**不调用 LLM**；只有指代/追问才进 1 次改写。
- 有 pending_action 时的确认/拒绝用**确定性规则**判定，**不调用 LLM**。
成本红线：独立节点不能变成「每条消息固定多一次 LLM 调用」，故用两道确定性闸门把 LLM 挡在后面。
"""

from __future__ import annotations

import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from opendetect_ai.state import AgentState
from opendetect_ai.context_utils import build_context_str
from opendetect_ai.tools.progress import push_progress
from opendetect_ai.env_utils import (
    OPENDETECT_LLM_MODEL,
    OPENDETECT_LLM_BASE_URL,
    OPENDETECT_LLM_API_KEY,
)

# 供评测/测试核对「改写 LLM 调用次数」——确定性路径不应触碰它。
_llm_call_count = 0


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=OPENDETECT_LLM_MODEL,
        base_url=OPENDETECT_LLM_BASE_URL,
        api_key=OPENDETECT_LLM_API_KEY,
        temperature=0,
    )


# ── pending_action 生命周期辅助 ────────────────────────────────
# 系统在「库里没有相关论文、主动提议搜索入库」时写入；用户确认时消费、拒绝/新任务时清空。
_SEARCH_OFFER_MARKERS = ("要我去搜", "要我搜", "搜索并入库", "去搜一批", "搜一批")


def answer_offers_search(text: str) -> bool:
    """判断一段助手回复是否是在「提议去搜索入库」——据此写入 pending_action。"""
    return any(m in (text or "") for m in _SEARCH_OFFER_MARKERS)


def make_search_pending(question: str) -> dict:
    """构造一条待确认的搜索动作。query 写成显式搜索指令，确认后下游 Supervisor 天然路由到 search。"""
    return {"kind": "search", "query": f"帮我搜索并入库与「{question}」相关的论文"}


# ── 确定性闸门 ①：确认 / 拒绝词判定（不调用 LLM）───────────────
_AFFIRM = {
    "好", "好的", "好啊", "好呀", "好滴", "可以", "可以的", "行", "行的", "嗯", "嗯嗯",
    "嗯好", "是的", "对", "要", "好的谢谢", "麻烦你了", "麻烦了", "去吧", "搜吧", "都行",
    "需要", "需要的", "ok", "okay", "yes", "sure", "好的呀",
}
_REJECT = {
    "不用", "不用了", "不用啦", "算了", "不需要", "不需要了", "先不用", "先不用了",
    "不要", "不", "没必要", "不必", "no", "不用麻烦了",
}


def _norm(text: str) -> str:
    """归一化：去掉空白与常见标点，转小写，便于和确认/拒绝词表精确比对。"""
    return re.sub(r"[\s，。,.!！?？、~…\-—]+", "", (text or "")).lower()


# 语气/礼貌填充词：可与确认/拒绝核心词共同出现，但单独出现不算数。
_FILLER = {"的", "呀", "啊", "哦", "呢", "滴", "嘛", "哈", "喔", "吧", "啦", "了", "谢谢", "谢"}


def _consumes_all(n: str, core: set[str]) -> bool:
    """贪心：整句能否被『核心词 + 填充词』完全覆盖，且至少含一个核心词。"""
    toks = sorted(core | _FILLER, key=len, reverse=True)
    i, used_core = 0, False
    while i < len(n):
        for t in toks:
            if t and n.startswith(t, i):
                used_core = used_core or (t in core)
                i += len(t)
                break
        else:
            return False
    return used_core


def _confirm_verdict(text: str) -> str:
    """返回 'affirm' | 'reject' | 'other'。只有整句就是确认/拒绝词才算数，避免误吞新任务。"""
    n = _norm(text)
    if not n:
        return "other"
    if n in _REJECT or _consumes_all(n, _REJECT):
        return "reject"
    if n in _AFFIRM or _consumes_all(n, _AFFIRM):
        return "affirm"
    return "other"


# ── 确定性闸门 ②：是否含指代/追问，需要借上下文改写 ─────────────
_ANAPHORA_MARKERS = (
    "还有", "其他的", "其它的", "别的", "更多", "多找", "再找", "再来",
    "它", "它们", "这个", "那个", "这些", "那些", "上面", "前面", "刚才", "之前", "上一篇",
)


def _needs_rewrite(text: str) -> bool:
    """含指代/追问标记才需要借上下文改写；否则视为自包含，直接透传（0 LLM）。"""
    return any(m in (text or "") for m in _ANAPHORA_MARKERS)


def _rewrite_with_llm(raw: str, chat_context: str) -> str:
    """把含指代/追问的输入借上下文改写成自包含问题。1 次 LLM 调用；失败 fail-open 回原文。"""
    global _llm_call_count
    _llm_call_count += 1
    prompt = (
        "把用户这句含指代/省略的追问，结合对话上下文，改写成一句**自包含**的问题"
        "（不依赖上文也能看懂，保留原语言，只补全指代对象/话题）。只输出改写后的问题本身。\n\n"
        "## 对话上下文\n" + chat_context + "\n\n## 用户这句\n" + raw + "\n\n## 改写后"
    )
    try:
        resp = _get_llm().invoke([HumanMessage(content=prompt)])
        out = (resp.content or "").strip().strip('"').strip("'")
        return out or raw
    except Exception as exc:
        print(f"[Resolve] 改写失败，回退原文: {exc}")
        return raw


def resolve_node(state: AgentState) -> dict:
    """
    入口解析：产出 resolved_query（不覆写 user_query），并维护 pending_action 生命周期。
    路径优先级：pending 确认(确定性) → 拒绝/新任务清空 pending → 自包含透传 → 指代改写(1 次 LLM)。

    同时把本轮用户输入记为 HumanMessage 追加进 messages——此前全流程只追加 AIMessage，
    build_context_str 恒为空、短期记忆形同虚设（顺带修复的潜藏 bug）；有了它上下文才真正可用。
    """
    raw = state.get("user_query", "")
    _tid = state.get("thread_id", "default")
    pending = state.get("pending_action")
    # 每个返回分支都带上这条，保证本轮用户输入进入历史
    base: dict = {"messages": [HumanMessage(content=raw)]} if raw else {}

    # 1) 有待确认动作：用确定性规则判定确认/拒绝（不调用 LLM）
    if pending:
        verdict = _confirm_verdict(raw)
        if verdict == "affirm":
            resolved = pending.get("query", raw)
            push_progress(_tid, f"✅ 承接上一轮提议：{resolved[:40]}")
            print(f"[Resolve] 确认 pending_action → '{resolved}'")
            return {**base, "resolved_query": resolved, "pending_action": None}
        if verdict == "reject":
            print("[Resolve] 用户拒绝上一轮提议，清空 pending_action")
            return {**base, "resolved_query": raw, "pending_action": None}
        # 'other'：是新的明确任务，清空 pending，继续按普通输入处理
        base["pending_action"] = None

    # 2) 自包含问题直接透传，不花 LLM
    if not _needs_rewrite(raw):
        return {**base, "resolved_query": raw}

    # 3) 含指代/追问：借上下文改写。把当前这句也纳入再算上下文，
    #    build_context_str 会排除「最后一对未回答的当前轮」，恰好得到纯历史。
    msgs_with_current = state.get("messages", []) + [HumanMessage(content=raw)]
    chat_context = build_context_str(msgs_with_current)
    if not chat_context:                      # 没有历史可依，无从改写，透传（0 LLM）
        return {**base, "resolved_query": raw}

    resolved = _rewrite_with_llm(raw, chat_context)
    push_progress(_tid, f"✍️ 指代消解：{raw[:12]} → {resolved[:40]}")
    print(f"[Resolve] 改写 '{raw}' → '{resolved}'")
    return {**base, "resolved_query": resolved}
