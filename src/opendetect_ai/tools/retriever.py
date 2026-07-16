"""
检索管线 —— OpenDetect_AI

在朴素 similarity_search 之上叠加三层能力，专治「脏库召回跨领域噪音」：

    用户问题
        │
        ▼  ① Self-Query：LLM 结构化抽取「语义 query + 年份/作者/标题过滤条件」
    ┌───────────────┬───────────────┐
    ▼               ▼
  Dense(向量)     BM25(关键词)          ② Hybrid：稠密召专有名词弱、稀疏补精确匹配
    └──────┬────────┘
           ▼  RRF 融合（Reciprocal Rank Fusion）
      候选池(pool=30)
           ▼  元数据后置过滤（年份/作者/标题，作用于 self-query 抽出的条件）
           ▼  ③ Rerank + 噪音闸门：交叉相关性重排，丢弃跨领域段落
        top-k 结果

设计取舍见 README「检索管线」一节。所有 LLM 调用均带兜底，任一环节失败都
不会让检索崩溃，只是退化为更朴素的策略。
"""

from __future__ import annotations

import threading
from typing import Optional

import requests
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_community.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI

from opendetect_ai.env_utils import (
    OPENDETECT_LLM_MODEL,
    OPENDETECT_LLM_BASE_URL,
    OPENDETECT_LLM_API_KEY,
    OPENDETECT_EMBED_API_KEY,
    OPENDETECT_RETRIEVAL_POOL,
    OPENDETECT_SELF_QUERY,
    OPENDETECT_RERANK_BACKEND,
    OPENDETECT_RERANK_MODEL,
    OPENDETECT_RERANK_MIN_SCORE,
)
from opendetect_ai.tools import rag_tool


# 轻量 LLM 调用计数（供评估统计「每次检索花几次 LLM 调用」）。仅自增，评估侧读取/清零。
_llm_call_count = 0


def _bump_llm() -> None:
    global _llm_call_count
    _llm_call_count += 1


# ── LLM（自查询 + LLM 重排共用）────────────────────────────────
def _get_llm(temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        model=OPENDETECT_LLM_MODEL,
        base_url=OPENDETECT_LLM_BASE_URL,
        api_key=OPENDETECT_LLM_API_KEY,
        temperature=temperature,
    )


# ══════════════════════════════════════════════════════════════
# ① Self-Query：把自然语言里的过滤意图抽成结构化条件
# ══════════════════════════════════════════════════════════════
class RetrievalPlan(BaseModel):
    """从用户问题中抽取的检索计划。"""
    semantic_query: str = Field(description="用于向量/关键词检索的核心语义查询（去掉年份、作者等约束词）")
    year_min: Optional[int] = Field(default=None, description="发表年份下限，如“2023年后”→2023；无则 null")
    year_max: Optional[int] = Field(default=None, description="发表年份上限；无则 null")
    author: Optional[str] = Field(default=None, description="指定作者名（英文），无则 null")
    title_keyword: Optional[str] = Field(default=None, description="标题必须包含的关键词，无则 null")


_SELF_QUERY_PROMPT = """你是一个学术检索的查询解析器。从用户问题中抽取检索计划。

规则：
- semantic_query：保留问题的核心语义，去掉“2023年后”“XX作者的”这类约束词。
- 只有当用户**明确**提到年份/作者/标题约束时才填对应字段，否则一律 null。
- 不要臆测。例如“Swin Transformer 的改进”里没有年份约束，year_min/year_max 都为 null。

用户问题：{query}"""


def plan_query(query: str) -> RetrievalPlan:
    """LLM 结构化抽取检索计划；失败或关闭时退化为「纯语义、无过滤」。"""
    if not OPENDETECT_SELF_QUERY:
        return RetrievalPlan(semantic_query=query)
    try:
        _bump_llm()
        plan = _get_llm().with_structured_output(
            RetrievalPlan, method="function_calling"
        ).invoke([HumanMessage(content=_SELF_QUERY_PROMPT.format(query=query))])
        if isinstance(plan, dict):
            plan = RetrievalPlan(**plan)
        if not plan.semantic_query.strip():
            plan.semantic_query = query
        return plan
    except Exception as exc:
        print(f"[Retriever] self-query 失败，退化为纯语义检索: {exc}")
        return RetrievalPlan(semantic_query=query)


# ══════════════════════════════════════════════════════════════
# ② Hybrid：Dense + BM25，RRF 融合
# ══════════════════════════════════════════════════════════════
# BM25 索引随语料版本缓存，入库后自动重建
_bm25_cache: dict = {"version": -1, "retriever": None}
_bm25_lock = threading.Lock()


def _get_bm25(pool: int) -> Optional[BM25Retriever]:
    """按语料版本缓存 BM25 检索器；空库返回 None。"""
    version = rag_tool.get_corpus_version()
    with _bm25_lock:
        if _bm25_cache["version"] == version and _bm25_cache["retriever"] is not None:
            _bm25_cache["retriever"].k = pool
            return _bm25_cache["retriever"]
    docs = rag_tool.get_all_documents()
    if not docs:
        return None
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = pool
    with _bm25_lock:
        _bm25_cache["version"] = version
        _bm25_cache["retriever"] = retriever
    return retriever


def _doc_key(doc: Document) -> tuple:
    """段落唯一键：优先 arxiv_id，退回 title，再叠加 chunk 序号。"""
    meta = doc.metadata or {}
    base = meta.get("arxiv_id") or meta.get("title", "")
    return (base, meta.get("chunk_idx", 0))


def _rrf_fuse(ranked_lists: list[list[Document]], k_rrf: int = 60) -> list[Document]:
    """
    Reciprocal Rank Fusion：对多个排序列表按 1/(k_rrf + rank) 累加分数后重排。
    等价于 EnsembleRetriever 默认的融合方式，这里手写以便透明控制。
    """
    scores: dict = {}
    docs_by_key: dict = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            key = _doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k_rrf + rank)
            docs_by_key.setdefault(key, doc)
    ordered_keys = sorted(scores, key=lambda kk: scores[kk], reverse=True)
    return [docs_by_key[kk] for kk in ordered_keys]


def _hybrid_candidates(semantic_query: str, pool: int) -> list[Document]:
    """稠密 + 稀疏各召回 pool 条，RRF 融合成候选池。"""
    vectorstore = rag_tool._get_vectorstore()
    dense_docs = vectorstore.similarity_search(semantic_query, k=pool)

    bm25 = _get_bm25(pool)
    sparse_docs = bm25.invoke(semantic_query) if bm25 is not None else []

    if not sparse_docs:                 # 库里文档过少时 BM25 可能为空，退化为纯稠密
        return dense_docs
    return _rrf_fuse([dense_docs, sparse_docs])


# ══════════════════════════════════════════════════════════════
#   元数据后置过滤（作用于 self-query 抽出的条件）
# ══════════════════════════════════════════════════════════════
def _published_year(meta: dict) -> Optional[int]:
    published = str(meta.get("published", ""))[:4]
    return int(published) if published.isdigit() else None


def _apply_filters(docs: list[Document], plan: RetrievalPlan) -> list[Document]:
    """按 self-query 抽出的年份/作者/标题条件在 Python 端过滤（不依赖 Chroma filter 方言）。"""
    out = []
    for doc in docs:
        meta = doc.metadata or {}
        year = _published_year(meta)
        if plan.year_min and (year is None or year < plan.year_min):
            continue
        if plan.year_max and (year is None or year > plan.year_max):
            continue
        if plan.author and plan.author.lower() not in str(meta.get("authors", "")).lower():
            continue
        if plan.title_keyword and plan.title_keyword.lower() not in str(meta.get("title", "")).lower():
            continue
        out.append(doc)
    # 过滤后若为空，说明条件过严，放弃过滤以免「宁缺毋滥」丢光结果
    return out or docs


# ══════════════════════════════════════════════════════════════
# ③ Rerank + 噪音闸门
# ══════════════════════════════════════════════════════════════
class _RerankResult(BaseModel):
    relevant_indices: list[int] = Field(
        description="与问题真正相关的段落序号（从1开始），按相关性从高到低；跨领域/主题不符的段落不要包含"
    )


_LLM_RERANK_PROMPT = """给定用户问题和若干候选段落，请挑出**真正与问题相关**的段落，按相关性从高到低排序。

严格要求：
- 只保留主题真正匹配的段落；跨领域、答非所问的段落（例如问计算机视觉却是医学影像/天气预报）一律排除。
- 最多返回 {k} 个；宁可少返回，也不要塞入不相关段落。
- 输出段落的序号（从1开始）。

## 用户问题
{query}

## 候选段落
{passages}"""


def _rerank_llm(query: str, docs: list[Document], k: int) -> list[Document]:
    """用 LLM 做 listwise 重排，并借 prompt 指令天然过滤跨领域噪音。"""
    passages = "\n\n".join(
        f"[{i}] （{(d.metadata or {}).get('title','未知')}）{d.page_content[:400]}"
        for i, d in enumerate(docs, start=1)
    )
    try:
        result = _get_llm().with_structured_output(
            _RerankResult, method="function_calling"
        ).invoke([HumanMessage(content=_LLM_RERANK_PROMPT.format(
            k=k, query=query, passages=passages))])
        _bump_llm()
        if isinstance(result, dict):
            result = _RerankResult(**result)
        picked = [docs[i - 1] for i in result.relevant_indices if 1 <= i <= len(docs)]
        return picked[:k] if picked else docs[:k]
    except Exception as exc:
        print(f"[Retriever] LLM 重排失败，退回融合序: {exc}")
        return docs[:k]


def _rerank_dashscope(query: str, docs: list[Document], k: int) -> Optional[list[Document]]:
    """DashScope gte-rerank：复用 EMBED 的 Key；带噪音闸门。失败返回 None 交由上层兜底。"""
    url = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
    try:
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {OPENDETECT_EMBED_API_KEY}",
                     "Content-Type": "application/json"},
            json={
                "model": OPENDETECT_RERANK_MODEL,
                "input": {"query": query, "documents": [d.page_content[:800] for d in docs]},
                "parameters": {"return_documents": False, "top_n": max(k * 3, k)},
            },
            timeout=20,
        )
        resp.raise_for_status()
        results = resp.json().get("output", {}).get("results", [])
        picked = [
            docs[r["index"]]
            for r in results
            if r.get("relevance_score", 0) >= OPENDETECT_RERANK_MIN_SCORE and r["index"] < len(docs)
        ]
        return picked[:k]
    except Exception as exc:
        print(f"[Retriever] DashScope 重排失败，交由上层兜底: {exc}")
        return None


def _rerank(query: str, docs: list[Document], k: int) -> list[Document]:
    if not docs:
        return []
    backend = OPENDETECT_RERANK_BACKEND
    if backend == "none":
        return docs[:k]
    if backend == "dashscope":
        picked = _rerank_dashscope(query, docs, k)
        if picked is not None:
            return picked
        # DashScope 不可用时退回 LLM 重排
    return _rerank_llm(query, docs, k)


# ══════════════════════════════════════════════════════════════
#   公开接口
# ══════════════════════════════════════════════════════════════
def _to_dict(doc: Document) -> dict:
    meta = doc.metadata or {}
    return {
        "content":   doc.page_content,
        "title":     meta.get("title", ""),
        "arxiv_id":  meta.get("arxiv_id", ""),
        "published": meta.get("published", ""),
        "chunk_idx": meta.get("chunk_idx", 0),
    }


def retrieve(query: str, k: int = 5, pool: Optional[int] = None) -> list[dict]:
    """
    完整检索管线：self-query → hybrid(dense+BM25)+RRF → 元数据过滤 → rerank 去噪 → top-k。
    返回结构与旧 retrieve_context 一致；空库返回 [{"error": ...}]。
    """
    if rag_tool.vectorstore_is_empty():
        return [{"error": "向量库为空，请先使用 ingest_paper 工具入库论文"}]

    pool = pool or OPENDETECT_RETRIEVAL_POOL
    plan = plan_query(query)
    candidates = _hybrid_candidates(plan.semantic_query, pool)
    candidates = _apply_filters(candidates, plan)
    reranked = _rerank(plan.semantic_query, candidates, k)
    return [_to_dict(d) for d in reranked]


def retrieve_dense_only(query: str, k: int = 5) -> list[dict]:
    """基线：纯稠密 top-k（无 hybrid / self-query / rerank），供评估对比用。"""
    if rag_tool.vectorstore_is_empty():
        return [{"error": "向量库为空"}]
    docs = rag_tool._get_vectorstore().similarity_search(query, k=k)
    return [_to_dict(d) for d in docs]
