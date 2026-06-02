"""
Supervisor Agent —— OpenDetect_AI
负责意图识别、动态路由、协调各子 Agent。
"""
 
from __future__ import annotations
 
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
 
from opendetect_ai.state import AgentState
from opendetect_ai.context_utils import build_context_str
from opendetect_ai.user_memory import load_user_profile, format_profile_for_prompt
from opendetect_ai.tools.progress import push_progress
from opendetect_ai.prompts import SUPERVISOR_PROMPT
from opendetect_ai.env_utils import (
    OPENDETECT_LLM_MODEL,
    OPENDETECT_LLM_BASE_URL,
    OPENDETECT_LLM_API_KEY,
)
 
 
# ── 初始化 LLM ─────────────────────────────────────────────────
def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=OPENDETECT_LLM_MODEL,
        base_url=OPENDETECT_LLM_BASE_URL,
        api_key=OPENDETECT_LLM_API_KEY,
        temperature=0,        # 路由决策不需要创造性，温度设为 0
    )
 
 
# ── 合法的路由目标 ─────────────────────────────────────────────
VALID_NEXT = {"search", "ingest", "rag", "report", "FINISH"}
 
 
def supervisor_node(state: AgentState) -> dict:
    _tid = state.get("thread_id", "default")
    if state.get("rag_answer"):
        reason = "RAG 已生成回答，任务完成。"
        print(f"[Supervisor] → FINISH  理由: {reason}")
        return {
            "next": "FINISH",
            "messages": [AIMessage(content=f"Supervisor 决策: FINISH — {reason}")],
        }
 
    if state.get("final_report"):
        reason = "Report 已生成最终报告，任务完成。"
        print(f"[Supervisor] → FINISH  理由: {reason}")
        return {
            "next": "FINISH",
            "messages": [AIMessage(content=f"Supervisor 决策: FINISH — {reason}")],
        }
 
    if state.get("error"):
        error_msg = state.get("error", "")
        ingested = state.get("ingested_count", 0) or 0
        is_fatal = ingested == 0 and not state.get("rag_answer") and not state.get("final_report")
        if is_fatal:
            reason = f"致命错误且无可用内容，终止工作流：{error_msg}"
            print(f"[Supervisor] → FINISH  理由: {reason}")
            push_progress(_tid, f"❌ 错误终止：{error_msg[:50]}")
            return {
                "next": "FINISH",
                "messages": [AIMessage(content=f"Supervisor 决策: FINISH — {reason}")],
            }
        print(f"[Supervisor] 非致命错误，继续调度: {error_msg}")
        push_progress(_tid, f"⚠️ 部分失败，继续调度：{error_msg[:50]}")
 
    # ── 计算真实待处理论文数 ───────────────────────────────────
    papers_to_ingest = state.get("papers_to_ingest", [])
    pending_count = len([p for p in papers_to_ingest if not p.ingested])
    failed_papers = state.get("failed_papers", [])
    failed_count  = len(failed_papers)
 
    chat_context = build_context_str(state.get("messages", []))
    ctx_display = f"\n{chat_context}\n" if chat_context else "（无历史记录）"

    # 读取用户长期偏好（跨会话记忆）
    user_profile = load_user_profile()
    profile_str = format_profile_for_prompt(user_profile) or "（暂无跨会话记忆）"

    prompt = SUPERVISOR_PROMPT.format(
        user_profile    = profile_str,
        chat_context    = ctx_display,
        user_query      = state.get("user_query", ""),
        search_count    = len(state.get("search_results", [])),
        ingested_count  = state.get("ingested_count", 0),
        pending_count   = pending_count,
        has_rag_answer  = bool(state.get("rag_answer", "")),
        error           = state.get("error", "无"),
        search_attempted = state.get("search_attempted", False),
        failed_count    = failed_count,
    )
 
    # ── 调用 LLM 做路由决策 ────────────────────────────────────
    llm = _get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()
 
    # ── 解析 JSON 输出 ─────────────────────────────────────────
    try:
        # 兼容 LLM 有时会包裹在 ```json ... ``` 里的情况
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        decision = json.loads(raw)
        next_agent    = decision.get("next", "FINISH")
        reason        = decision.get("reason", "")
        direct_answer = decision.get("reply", "")
    except (json.JSONDecodeError, IndexError):
        next_agent    = "FINISH"
        reason        = f"JSON 解析失败，原始输出: {raw}"
        direct_answer = ""
 
    # ── 合法性校验 ─────────────────────────────────────────────
    if next_agent not in VALID_NEXT:
        next_agent = "FINISH"
        reason     = f"非法路由目标 '{next_agent}'，已降级为 FINISH"
 
    # ── 闲聊兜底：LLM 有时把回复内容写进 reason 而不是 reply ──
    # 当路由到 FINISH 且 direct_answer 为空时，检查 reason 是否像回复内容
    if next_agent == "FINISH" and not direct_answer and reason:
        # reason 超过20字且不含"任务"/"完成"/"已"等路由语气词，视为对话回复
        routing_keywords = ("任务", "完成", "已", "操作", "执行", "工作流", "路由", "调用", "搜索已", "入库已")
        is_routing_msg = any(kw in reason for kw in routing_keywords)
        if not is_routing_msg and len(reason) > 15:
            direct_answer = reason
 
    print(f"[Supervisor] → {next_agent}  理由: {reason}")
    push_progress(_tid, f"🧭 路由决策：→ {next_agent}  {reason}")
    if direct_answer:
        print(f"[Supervisor] 闲聊回复: {direct_answer[:60]}...")
        push_progress(_tid, "💬 生成闲聊回复...")
 
    # 闲聊回复直接放入 messages，确保前端能从消息流里拿到
    msg_content = direct_answer if direct_answer else f"Supervisor 决策: {next_agent} — {reason}"
 
    return {
        "next":          next_agent,
        "direct_answer": direct_answer,
        "messages":      [AIMessage(content=msg_content)],
    }