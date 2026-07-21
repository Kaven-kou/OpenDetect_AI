"""
Resolve Query Agent —— OpenDetect_AI
每轮入口的「上游查询解析」：把省略/指代/确认式输入解析成自包含 resolved_query，
并维护 pending_action（搜索承接 kind:"search" / 澄清 kind:"clarification"）。

设计边界（刻意克制）：
- 每轮只在入口执行一次：START → resolve → (clarify | supervisor)；子 Agent 回流不经过这里。
- 只新增 resolved_query，绝不覆写 user_query；下游统一用 effective_query() 读取。
- 成本红线：普通问题 / 确认词 0 次 LLM；只有含指代/追问才花 1 次「引用解析」调用。
- 指代若在上下文里有 ≥2 个可 grounding 的候选 → 交 clarify 澄清，绝不瞎猜。
"""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage

from opendetect_ai.state import AgentState
from opendetect_ai.tools.progress import push_progress
from opendetect_ai.agents.clarify import (
    resolve_reference_llm,
    judge_reference,
    build_clarify_pending,
    fallback_pending,
    parse_clarification_selection,
)

# 供评测/测试核对「引用解析 LLM 调用次数」——确定性路径不应触碰它。
_llm_call_count = 0


# ── pending_action(search) 生命周期辅助 ────────────────────────
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
_FILLER = {"的", "呀", "啊", "哦", "呢", "滴", "嘛", "哈", "喔", "吧", "啦", "了", "谢谢", "谢"}


def _norm(text: str) -> str:
    """归一化：去掉空白与常见标点，转小写，便于和确认/拒绝词表精确比对。"""
    return re.sub(r"[\s，。,.!！?？、~…\-—]+", "", (text or "")).lower()


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


# ── 确定性闸门 ②：是否含指代/追问，需要借上下文解析 ───────────
_ANAPHORA_MARKERS = (
    "还有", "其他的", "其它的", "别的", "更多", "多找", "再找", "再来",
    "它", "它们", "这个", "那个", "这些", "那些", "上面", "前面", "刚才", "之前", "上一篇",
)


def _needs_rewrite(text: str) -> bool:
    """含指代/追问标记才需要借上下文解析；否则视为自包含，直接透传（0 LLM）。"""
    return any(m in (text or "") for m in _ANAPHORA_MARKERS)


def resolve_node(state: AgentState) -> dict:
    """
    入口解析。路径优先级：
      A) 进行中的澄清 → 确定性解析用户选择（select/clear/reprocess/reclarify/fallback）
      B) 搜索提议的确认（好啊/不用了）
      C) 自包含问题 → 透传（0 LLM）
      D) 含指代/追问 → 引用解析（1 LLM）；有 ≥2 个可 grounding 候选 → 澄清
    同时把本轮用户输入记为 HumanMessage（此前全流程只追加 AIMessage、短期记忆形同虚设）。
    """
    raw = state.get("user_query", "")
    _tid = state.get("thread_id", "default")
    pending = state.get("pending_action")
    base: dict = {"messages": [HumanMessage(content=raw)]} if raw else {}

    # ── A) 处理进行中的澄清 ────────────────────────────────────
    if pending and pending.get("kind") == "clarification":
        sel = parse_clarification_selection(raw, pending)
        if sel.action == "select":
            print(f"[Resolve] 澄清选择 → '{sel.resolved_query}'")
            return {**base, "resolved_query": sel.resolved_query or raw, "pending_action": None}
        if sel.action == "clear":
            print("[Resolve] 用户放弃澄清，清空 pending_action")
            return {**base, "resolved_query": raw, "pending_action": None}
        if sel.action == "reclarify":
            nxt = {**pending, "attempts": int(pending.get("attempts", 1)) + 1}
            push_progress(_tid, "❓ 回复不明确，再问一次")
            return {**base, "pending_action": nxt}          # 路由 → clarify 再问
        if sel.action == "fallback":
            print("[Resolve] 澄清达上限，走兜底并清空")
            return {**base, "pending_action": fallback_pending(pending.get("original_query", ""))}
        # reprocess：当作新任务，清空澄清后继续按普通输入处理
        pending = None
        base["pending_action"] = None

    # ── B) 搜索提议的确认（确定性，不调用 LLM）─────────────────
    if pending and pending.get("kind") == "search":
        verdict = _confirm_verdict(raw)
        if verdict == "affirm":
            resolved = pending.get("query", raw)
            push_progress(_tid, f"✅ 承接上一轮提议：{resolved[:40]}")
            print(f"[Resolve] 确认 pending_action → '{resolved}'")
            return {**base, "resolved_query": resolved, "pending_action": None}
        if verdict == "reject":
            print("[Resolve] 用户拒绝上一轮提议，清空 pending_action")
            return {**base, "resolved_query": raw, "pending_action": None}
        base["pending_action"] = None    # other → 清空，按普通输入继续

    # ── C) 自包含问题直接透传，不花 LLM ────────────────────────
    if not _needs_rewrite(raw):
        return {**base, "resolved_query": raw}

    # ── D) 含指代/追问：引用解析（1 次 LLM）；歧义则澄清 ────────
    msgs = state.get("messages", [])
    if not msgs:                          # 没有历史可依，无从解析，透传
        return {**base, "resolved_query": raw}

    global _llm_call_count
    _llm_call_count += 1
    result = resolve_reference_llm(raw, msgs)
    decision = judge_reference(raw, msgs, result)
    if decision:
        push_progress(_tid, "❓ 指代有多个可指对象，需澄清")
        print(f"[Resolve] 指代歧义 → 澄清（{len(decision.options)} 个候选）")
        return {**base, "pending_action": build_clarify_pending(decision, raw)}

    push_progress(_tid, f"✍️ 指代消解：{raw[:12]} → {result.resolved_query[:40]}")
    print(f"[Resolve] 改写 '{raw}' → '{result.resolved_query}'")
    return {**base, "resolved_query": result.resolved_query}
