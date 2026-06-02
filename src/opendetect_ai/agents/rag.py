"""
RAG Agent —— OpenDetect_AI
负责从向量库召回相关段落，结合上下文生成有文献依据的回答。
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

from opendetect_ai.state import AgentState
from opendetect_ai.context_utils import build_context_str
from opendetect_ai.tools.progress import push_progress
from opendetect_ai.prompts import RAG_PROMPT
from opendetect_ai.tools.rag_tool import retrieve_context
from opendetect_ai.env_utils import (
    OPENDETECT_LLM_MODEL,
    OPENDETECT_LLM_BASE_URL,
    OPENDETECT_LLM_API_KEY,
)


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=OPENDETECT_LLM_MODEL,
        base_url=OPENDETECT_LLM_BASE_URL,
        api_key=OPENDETECT_LLM_API_KEY,
        temperature=0.3,
    )


def _format_context(chunks: list[dict]) -> str:
    """把召回的文本块格式化成 prompt 里可读的字符串。"""
    if not chunks or "error" in chunks[0]:
        return "（向量库为空，无可用上下文）"

    parts = []
    for i, chunk in enumerate(chunks, 1):
        title     = chunk.get("title", "未知论文")
        arxiv_id  = chunk.get("arxiv_id", "")
        content   = chunk.get("content", "")
        source    = f"{title}（{arxiv_id}）" if arxiv_id else title
        parts.append(f"【片段 {i}】来源: {source}\n{content}")

    return "\n\n---\n\n".join(parts)


def rag_node(state: AgentState) -> dict:
    _tid = state.get("thread_id", "default")
    """
    RAG Agent 节点：
    1. 从向量库检索与问题相关的论文片段
    2. 拼装 prompt，调用 LLM 生成回答
    3. 把答案写入 state.rag_answer
    """
    user_query = state.get("user_query", "")

    # ── Step 1: 向量检索 ───────────────────────────────────────
    push_progress(_tid, f"🔎 向量检索：{user_query[:40]}...")
    print(f"[RAG] 检索问题: {user_query}")
    chunks = retrieve_context.invoke({
        "query": user_query,
        "k":     5,
    })

    # 检索失败（向量库为空）时提前返回
    if chunks and "error" in chunks[0]:
        error_msg = chunks[0]["error"]
        return {
            "rag_context": [],
            "rag_answer":  error_msg,
            "error":       error_msg,
            "messages": [AIMessage(content=error_msg)],
        }

    push_progress(_tid, f"📄 召回 {len(chunks)} 个相关段落，生成回答中...")
    print(f"[RAG] 召回 {len(chunks)} 个文本块")

    # ── Step 2: 格式化上下文 ───────────────────────────────────
    formatted_context = _format_context(chunks)

    # ── Step 3: 构造 prompt，调用 LLM ─────────────────────────
    chat_context = build_context_str(state.get("messages", []))
    prompt = RAG_PROMPT.format(
        rag_context=formatted_context,
        chat_context=chat_context or "（无历史记录）",
        user_query=user_query,
    )

    llm = _get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    answer = response.content.strip()

    print(f"[RAG] 生成回答（前100字）: {answer[:100]}...")

    return {
        "rag_context": chunks,
        "rag_answer":  answer,
        "error":       "",
        "messages":    [AIMessage(content=answer)],
    }