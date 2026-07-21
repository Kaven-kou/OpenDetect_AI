"""
Supervisor Agent —— OpenDetect_AI
负责意图识别、动态路由、协调各子 Agent。
"""
 
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
 
from opendetect_ai.state import AgentState, effective_query
from opendetect_ai.context_utils import build_context_str
from opendetect_ai.user_memory import load_user_profile, format_profile_for_prompt
from opendetect_ai.agents.resolve import answer_offers_search, make_search_pending
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


# ── 结构化路由决策（用函数调用约束 LLM 输出，字段天然合法）──────
class RouteDecision(BaseModel):
    """Supervisor 的路由决策。用 with_structured_output 强约束，避免手撕 JSON。"""
    next: Literal["search", "ingest", "rag", "report", "FINISH"] = Field(
        description="下一步要调用的智能体；任务完成或闲聊时为 FINISH"
    )
    reason: str = Field(description="一句话说明路由原因")
    reply: str = Field(
        default="",
        description="仅当用户是闲聊/问身份/问能力时，在此生成中文友好回复（Markdown）；任务类请求留空字符串",
    )


def _route_with_llm(prompt: str) -> RouteDecision:
    """
    首选结构化输出（函数调用）拿到合法字段；若兼容端点偶发不支持工具调用，
    回退到「裸文本 + JSON 解析」，双保险保证路由永不崩。
    """
    llm = _get_llm()
    try:
        # DeepSeek 不支持 json_schema 的 response_format，但支持函数调用，
        # 因此显式指定 method="function_calling"。
        result = llm.with_structured_output(
            RouteDecision, method="function_calling"
        ).invoke([HumanMessage(content=prompt)])
        if isinstance(result, RouteDecision):
            return result
        if isinstance(result, dict):                      # 某些兼容端点返回 dict
            return RouteDecision(**result)
    except Exception as exc:
        print(f"[Supervisor] 结构化输出失败，回退 JSON 解析: {exc}")

    # ── 回退路径：解析裸 JSON ──────────────────────────────────
    try:
        raw = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        nxt = data.get("next", "FINISH")
        return RouteDecision(
            next=nxt if nxt in VALID_NEXT else "FINISH",
            reason=data.get("reason", ""),
            reply=data.get("reply", "") or "",
        )
    except Exception as exc:
        return RouteDecision(next="FINISH", reason=f"路由解析失败: {exc}", reply="")
 
 
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
 
    # 路由决策只需最近意图，用较短窗口（2 轮）省 token；RAG/Search/Report 才用完整 4 轮
    chat_context = build_context_str(state.get("messages", []), window=2)
    ctx_display = f"\n{chat_context}\n" if chat_context else "（无历史记录）"

    # 读取用户长期偏好（按 user_id 隔离的跨会话记忆）
    user_profile = load_user_profile(state.get("user_id", "default"))
    profile_str = format_profile_for_prompt(user_profile) or "（暂无跨会话记忆）"

    prompt = SUPERVISOR_PROMPT.format(
        user_profile    = profile_str,
        chat_context    = ctx_display,
        user_query      = effective_query(state),
        search_count    = len(state.get("search_results", [])),
        ingested_count  = state.get("ingested_count", 0),
        pending_count   = pending_count,
        has_rag_answer  = bool(state.get("rag_answer", "")),
        error           = state.get("error", "无"),
        search_attempted = state.get("search_attempted", False),
        failed_count    = failed_count,
    )
 
    # ── 结构化路由决策（字段由函数调用保证合法）────────────────
    decision = _route_with_llm(prompt)
    next_agent    = decision.next
    reason        = decision.reason
    direct_answer = decision.reply

    # Literal 已约束 next 合法；保留一次防御性兜底
    if next_agent not in VALID_NEXT:
        _illegal = next_agent
        next_agent = "FINISH"
        reason     = f"非法路由目标 '{_illegal}'，已降级为 FINISH"

    print(f"[Supervisor] → {next_agent}  理由: {reason}")
    push_progress(_tid, f"🧭 路由决策：→ {next_agent}  {reason}")
    if direct_answer:
        print(f"[Supervisor] 闲聊回复: {direct_answer[:60]}...")
        push_progress(_tid, "💬 生成闲聊回复...")

    # 闲聊回复直接放入 messages，确保前端能从消息流里拿到
    msg_content = direct_answer if direct_answer else f"Supervisor 决策: {next_agent} — {reason}"

    result = {
        "next":          next_agent,
        "direct_answer": direct_answer,
        "messages":      [AIMessage(content=msg_content)],
    }

    # 模式B：库里没有相关论文、主动提议搜索时（reply 含提议话术），写入 pending_action，
    # 下一轮用户「好啊」由 resolve 节点确定性承接为搜索——承接逻辑统一收在上游，不在这里改写 query。
    if answer_offers_search(direct_answer):
        result["pending_action"] = make_search_pending(effective_query(state))
        print(f"[Supervisor] 写入 pending_action(search): {effective_query(state)}")

    return result