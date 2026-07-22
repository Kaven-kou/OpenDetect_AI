"""
共享状态定义 —— OpenDetect_AI 多智能体工作流
所有 Agent 读写同一个 AgentState，通过 LangGraph 的 StateGraph 流转。
"""

from __future__ import annotations

from typing import Annotated, Any
from dataclasses import dataclass, field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


# ── 论文元数据结构 ─────────────────────────────────────────────
@dataclass
class PaperMeta:
    """单篇论文的元信息，由 Search Agent 填充，Ingest Agent 消费。"""
    title: str = ""
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    arxiv_id: str = ""
    pdf_url: str = ""
    published: str = ""
    ingested: bool = False          # Ingest Agent 处理完后标记为 True
    retry_count: int = 0            # 已重试次数，超过上限后不再重试



# ── 主状态类 ───────────────────────────────────────────────────
class AgentState(dict):
    """
    贯穿整个工作流的共享状态。
    LangGraph 的每个节点接收这个状态，处理后返回更新的字段。

    字段说明：
        messages      : 完整对话历史，使用 operator.add 追加（LangGraph 标准写法）
        user_query    : 用户原始输入
        next          : Supervisor 决定的下一个节点名称
        search_results: Search Agent 找到的论文列表
        papers_to_ingest: 待入向量库的论文列表
        ingested_count: 已成功入库的论文数量
        rag_context   : RAG Agent 从向量库召回的相关段落
        rag_answer    : RAG Agent 生成的最终回答
        final_report  : Report Agent 生成的综述或对比表
        error         : 任意节点出错时记录错误信息
    """

    # add_messages 支持按 message.id 覆盖。AnswerGuard 因此能用核验后的回答
    # 替换 RAG/Report 草稿，而不是让未通过核验的文本残留在会话历史中。
    messages:         Annotated[list[BaseMessage], add_messages]
    user_query:       str            # 用户本轮原始输入（永不覆写，供日志/评测/回溯）
    resolved_query:   str            # 上游 resolve 出的自包含检索问题；下游用 effective_query() 读取
    pending_action:   dict[str, Any] | None  # 系统提出的待确认动作 {"kind","query"}；确认时消费、拒绝/新任务时清空
    next:             str
    search_results:   list[PaperMeta]
    papers_to_ingest: list[PaperMeta]
    ingested_count:   int
    rag_context:      list[dict[str, Any]]
    rag_answer:       str
    verification:     dict[str, Any]  # AnswerGuard 结构化结果：状态、置信度、无支撑论断/引用
    final_report:     str
    error:            str
    search_attempted: bool
    local_pdf_path:   str
    failed_papers:    list[PaperMeta]
    direct_answer:    str            # Supervisor 针对闲聊/身份询问生成的直接回复
    thread_id:        str            # 当前会话 ID，用于进度推送队列隔离
    hitl:             bool           # 是否开启入库前人工确认（仅 Web 持久化会话置 True）
    user_id:          str            # 用户标识，长期记忆按此隔离（跨会话）


def effective_query(state: dict) -> str:
    """
    下游 Agent 统一入口：优先用上游 resolve_query 产出的自包含 query，回退到原始输入。
    不覆写 user_query——原始输入始终保留，便于日志、评测和后续迁移到 TaskSpec。
    """
    return state.get("resolved_query") or state.get("user_query", "")


def answer_message_id(state: dict) -> str:
    """为本轮论文回答生成稳定 ID，供 AnswerGuard 原位替换草稿。"""
    return f"answer:{state.get('thread_id', 'default')}:{len(state.get('messages', []))}"


# ── 初始状态工厂函数 ───────────────────────────────────────────
def create_initial_state(user_query: str) -> AgentState:
    """
    每次用户发起新请求时调用，返回一个干净的初始状态。
    用法：state = create_initial_state("CLIP-based OVD 有哪些主要方法？")
    """
    return AgentState(
        messages=[],
        user_query=user_query,
        resolved_query="",
        pending_action=None,
        next="supervisor",
        search_results=[],
        papers_to_ingest=[],
        ingested_count=0,
        rag_context=[],
        rag_answer="",
        verification={},
        final_report="",
        error="",
        local_pdf_path="",
        search_attempted=False,
        failed_papers=[],
        direct_answer="",
        thread_id="default",
        hitl=False,
        user_id="default",
    )
