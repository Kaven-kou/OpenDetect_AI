"""
Verifier Agent —— OpenDetect_AI
RAG 生成回答后的事实性校验关卡：检查回答的主要论断是否都能在检索片段中找到支撑，
不足时给回答加上警示（而非默默放行幻觉）。这是「生产级 RAG」的关键一环。

设计：
- 无检索内容却生成了回答 → 直接判 insufficient（不花 LLM）。
- 有检索内容 → 用 structured output 让 LLM 判是否 grounded，列出无支撑论断。
- 校验不通过只「附警示」，不删改用户已看到的流式正文（流式正文来自 rag 节点）。
- 任一步失败都放行（fail-open），校验器不该成为可用性瓶颈。
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from opendetect_ai.state import AgentState
from opendetect_ai.tools.progress import push_progress
from opendetect_ai.env_utils import (
    OPENDETECT_LLM_MODEL,
    OPENDETECT_LLM_BASE_URL,
    OPENDETECT_LLM_API_KEY,
    OPENDETECT_VERIFY,
)


class _Verdict(BaseModel):
    grounded: bool = Field(description="回答的主要论断是否都能在检索片段中找到支撑")
    unsupported_claims: list[str] = Field(
        default_factory=list, description="缺乏检索支撑的论断（最多 3 条，没有则空列表）"
    )


_VERIFY_PROMPT = """你是一个严格的事实核查员。判断下面的「回答」是否**完全由「检索片段」支撑**。

规则：
- 只依据检索片段判断，不要用你自己的知识补充。
- 若回答里有论断在检索片段中找不到依据，列进 unsupported_claims（最多 3 条）。
- grounded=true 当且仅当主要论断都能在片段中找到支撑。

## 检索片段
{context}

## 回答
{answer}"""


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=OPENDETECT_LLM_MODEL,
        base_url=OPENDETECT_LLM_BASE_URL,
        api_key=OPENDETECT_LLM_API_KEY,
        temperature=0,
    )


def _format_ctx(chunks: list[dict]) -> str:
    return "\n---\n".join(
        f"（{c.get('title','?')}）{c.get('content','')[:400]}" for c in chunks[:8]
    )


def verify_node(state: AgentState) -> dict:
    if not OPENDETECT_VERIFY:
        return {}

    answer = (state.get("rag_answer") or "").strip()
    chunks = state.get("rag_context") or []
    # 没有回答，或 rag 已因空库/错误提前返回 → 无需校验
    if not answer or (chunks and isinstance(chunks[0], dict) and "error" in chunks[0]):
        return {}

    _tid = state.get("thread_id", "default")
    push_progress(_tid, "🔎 校验回答的文献支撑...")

    # 无检索内容却生成了回答 → 直接提示
    if not chunks:
        caveat = "\n\n> ⚠️ 核验提示：本回答缺乏可核验的文献来源，请谨慎参考。"
        return {"rag_answer": answer + caveat}

    try:
        verdict = _get_llm().with_structured_output(
            _Verdict, method="function_calling"
        ).invoke([HumanMessage(content=_VERIFY_PROMPT.format(
            context=_format_ctx(chunks), answer=answer[:2000]))])
        if isinstance(verdict, dict):
            verdict = _Verdict(**verdict)
    except Exception as exc:
        print(f"[Verify] 校验失败，放行: {exc}")
        return {}   # fail-open

    if verdict.grounded:
        print("[Verify] ✓ 回答有文献支撑")
        return {}

    claims = verdict.unsupported_claims[:3]
    detail = ("：" + "；".join(claims)) if claims else "。"
    caveat = f"\n\n> ⚠️ 核验提示：以下论断未在检索文献中找到明确支撑，请谨慎对待{detail}"
    push_progress(_tid, "⚠️ 部分论断缺乏文献支撑，已附核验提示")
    print(f"[Verify] ⚠️ 未通过，无支撑论断: {claims}")
    return {"rag_answer": answer + caveat}
