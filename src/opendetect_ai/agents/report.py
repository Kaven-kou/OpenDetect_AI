"""
Report Agent —— OpenDetect_AI
负责基于已入库论文生成结构化综述、对比表或研究脉络总结。
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

from opendetect_ai.state import AgentState, answer_message_id, effective_query
from opendetect_ai.context_utils import build_context_str
from opendetect_ai.tools.progress import push_progress
from opendetect_ai.prompts import REPORT_PROMPT
from opendetect_ai.tools.rag_tool import list_ingested_papers, retrieve_context
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
        temperature=0.4,
    )


def _format_papers_info(
    papers: list[dict],
    rag_chunks: list[dict],
) -> str:
    """
    把论文列表 + 召回的相关段落合并成 prompt 可用的字符串。
    先给 LLM 看论文全貌，再附上关键段落供引用。
    """
    lines = ["## 已入库论文列表\n"]
    for i, p in enumerate(papers, 1):
        arxiv_id = p.get("arxiv_id", "无")
        lines.append(
            f"{i}. {p.get('title', '未知标题')}\n"
            f"   arxiv: {arxiv_id} | 发表: {p.get('published', '未知')}\n"
        )

    if rag_chunks and "error" not in rag_chunks[0]:
        lines.append("\n## 相关论文段落（供综述引用）\n")
        for i, chunk in enumerate(rag_chunks, 1):
            title    = chunk.get("title", "未知")
            content  = chunk.get("content", "")
            element_type = chunk.get("element_type", "text")
            element_number = chunk.get("element_number", "")
            kind_label = {"table": "表格", "figure": "图片"}.get(element_type, "段落")
            if element_number:
                kind_label += f" {element_number}"
            lines.append(f"【{kind_label} {i}】{title}\n{content}\n")

    return "\n".join(lines)


def report_node(state: AgentState) -> dict:
    _tid = state.get("thread_id", "default")
    """
    Report Agent 节点：
    1. 从向量库获取已入库论文列表
    2. 针对用户问题召回相关段落作为写作素材
    3. 调用 LLM 生成结构化综述
    4. 把综述写入 state.final_report
    """
    user_query = effective_query(state)

    # ── Step 1: 获取已入库论文列表 ─────────────────────────────
    ingested = list_ingested_papers.invoke({})
    if ingested and "message" in ingested[0]:
        msg = "向量库为空，请先搜索并入库论文后再生成综述。"
        return {
            "final_report": msg,
            "error":        msg,
            "messages": [AIMessage(content=msg)],
        }

    push_progress(_tid, f"📋 读取已入库论文，共 {len(ingested)} 篇...")
    print(f"[Report] 已入库论文数: {len(ingested)}")

    # ── Step 2: 召回相关段落作为写作素材 ──────────────────────
    rag_chunks = retrieve_context.invoke({
        "query": user_query,
        "k":     8,          # 综述需要更多上下文，取 8 块
    })
    push_progress(_tid, f"🔎 召回 {len(rag_chunks)} 个段落，生成综述中...")
    print(f"[Report] 召回段落数: {len(rag_chunks)}")

    # ── Step 3: 拼装 prompt ────────────────────────────────────
    papers_info = _format_papers_info(ingested, rag_chunks)
    chat_context = build_context_str(state.get("messages", []))
    prompt = REPORT_PROMPT.format(
        papers_info=papers_info,
        chat_context=chat_context or "（无历史记录）",
        user_query=user_query,
    )

    # ── Step 4: 生成综述 ───────────────────────────────────────
    llm = _get_llm()
    # final_answer 标签：供 SSE 层筛选出「最终综述」的 token 做流式输出
    response = llm.invoke([HumanMessage(content=prompt)], config={"tags": ["final_answer"]})
    report = response.content.strip()

    push_progress(_tid, f"✅ 综述生成完成，共 {len(report)} 字")
    print(f"[Report] 综述生成完成，共 {len(report)} 字")

    return {
        "final_report": report,
        "rag_context":  rag_chunks,
        "error":        "",
        "messages":     [AIMessage(content=report, id=answer_message_id(state))],
    }
