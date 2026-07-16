"""
RAG 工具 —— OpenDetect_AI
负责论文 PDF 下载、文本解析、向量入库、语义检索。
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fitz  # PyMuPDF
import requests
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from opendetect_ai.env_utils import (
    CHROMA_PERSIST_DIR,
    OPENDETECT_EMBED_API_KEY,
    OPENDETECT_EMBED_BASE_URL,
    OPENDETECT_EMBED_MODEL,
)


# ── 初始化 Embeddings 和向量库 ─────────────────────────────────
_vectorstore: Chroma | None = None
_vectorstore_lock = threading.Lock()

# 语料版本号：每次写入 +1。BM25 稀疏检索器据此判断缓存是否失效、需要重建。
_corpus_version = 0
_corpus_version_lock = threading.Lock()


def bump_corpus_version() -> None:
    """向量库发生写入后调用，使下游 BM25 缓存失效。"""
    global _corpus_version
    with _corpus_version_lock:
        _corpus_version += 1


def get_corpus_version() -> int:
    with _corpus_version_lock:
        return _corpus_version


def get_all_documents() -> list[Document]:
    """取出向量库中全部文档（供 BM25 稀疏检索构建索引）。空库返回 []。"""
    vectorstore = _get_vectorstore()
    result = vectorstore.get(include=["documents", "metadatas"])
    if not result or not result.get("ids"):
        return []
    docs = []
    for text, meta in zip(result.get("documents", []), result.get("metadatas", [])):
        docs.append(Document(page_content=text or "", metadata=meta or {}))
    return docs


def vectorstore_is_empty() -> bool:
    """用公开接口判断向量库是否为空。"""
    result = _get_vectorstore().get(limit=1)
    return not (result and result.get("ids"))


def _get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=OPENDETECT_EMBED_MODEL,
        base_url=OPENDETECT_EMBED_BASE_URL,
        api_key=OPENDETECT_EMBED_API_KEY,
        check_embedding_ctx_length=False,
        chunk_size=10,   # DashScope 单次最多 10 条（实测 25 即报 batch size invalid）；
                         # 从 5 提到 10 把入库时的 embedding 往返请求砍半
    )


def _get_vectorstore() -> Chroma:
    global _vectorstore

    if _vectorstore is None:
        with _vectorstore_lock:
            if _vectorstore is None:
                os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
                _vectorstore = Chroma(
                    collection_name="opendetect_papers",
                    embedding_function=_get_embeddings(),
                    persist_directory=CHROMA_PERSIST_DIR,
                )
    return _vectorstore


# ── 文本分块器 ─────────────────────────────────────────────────
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " "],
)

# 并行 embedding 的并发数：入库瓶颈是 embedding 的串行网络往返，
# 用线程池并发多个 batch 请求，把入库时间压下来。
_EMBED_WORKERS = 8


def _add_documents_fast(vectorstore: Chroma, documents: list[Document], ids: list[str]) -> None:
    """
    并行计算 embedding 后一次性写入向量库，替代 add_documents 的串行往返。
    langchain 的 add_documents 会按 chunk_size 逐批**串行**请求 embedding，
    一篇上百块的论文要几十次依次往返；这里把 batch（DashScope 上限 10 条）
    分组后用线程池并发请求，再用预计算向量批量 upsert，入库快数倍。
    """
    if not documents:
        return
    embeddings = _get_embeddings()
    texts = [d.page_content for d in documents]
    metas = [d.metadata for d in documents]
    groups = [texts[i:i + 10] for i in range(0, len(texts), 10)]  # DashScope 单批上限 10
    with ThreadPoolExecutor(max_workers=min(_EMBED_WORKERS, len(groups))) as ex:
        group_vecs = list(ex.map(embeddings.embed_documents, groups))  # 保序
    vectors = [v for gv in group_vecs for v in gv]
    # 用预计算向量批量写入（走底层 chroma collection，跳过其内部的串行 embedding）
    vectorstore._collection.upsert(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)


# ── 工具函数 ───────────────────────────────────────────────────
def _make_chunk_id(title: str, arxiv_id: str, chunk_idx: int) -> str:
    """
    生成确定性 chunk ID。
    相同论文的相同块永远是同一个 ID，保证幂等写入。
    优先用 arxiv_id，否则用标题的 md5 前 12 位。
    """
    base = arxiv_id if arxiv_id else hashlib.md5(title.encode()).hexdigest()[:12]
    return f"{base}__chunk_{chunk_idx}"


def _normalize_arxiv_id(arxiv_id: str) -> str:
    """去掉 arxiv 版本号，保证 2103.14030 和 2103.14030v2 命中同一篇。"""
    return re.sub(r"v\d+$", "", arxiv_id.strip())


def _paper_already_ingested(vectorstore: Chroma, arxiv_id: str, title: str) -> bool:
    """
    论文级别查重：检查该论文是否已在向量库中。
    优先用 arxiv_id 精确匹配，没有则退回标题匹配。
    使用公开的 vectorstore.get() 接口，不依赖私有 _collection。
    """
    arxiv_id = _normalize_arxiv_id(arxiv_id) if arxiv_id else ""
    try:
        if arxiv_id:
            result = vectorstore.get(where={"arxiv_id": arxiv_id})
        elif title:
            result = vectorstore.get(where={"title": title})
        else:
            return False
        return bool(result and result.get("ids"))
    except Exception:
        return False  # 查重失败不阻塞入库流程


def _download_pdf(pdf_url: str) -> str | None:
    """下载 PDF 到临时文件，返回临时文件路径，失败返回 None。"""
    try:
        resp = requests.get(pdf_url, timeout=30, stream=True)
        resp.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
            return f.name
    except Exception:
        return None


def _parse_pdf(file_path: str) -> str:
    """用 PyMuPDF 解析 PDF，返回全文文本。"""
    with fitz.open(file_path) as doc:
        return "\n".join(page.get_text() for page in doc)


# ── LangChain Tools ────────────────────────────────────────────
@tool
def ingest_paper(
    title: str,
    pdf_url: str,
    arxiv_id: str = "",
    authors: list[str] | None = None,
    published: str = "",
) -> dict:
    """
    下载论文 PDF，解析文本，分块后存入 Chroma 向量库。
    已存在的论文自动跳过，不会重复入库。

    Args:
        title:      论文标题
        pdf_url:    PDF 下载链接
        arxiv_id:   arxiv ID（可选）
        authors:    作者列表（可选）
        published:  发表日期（可选）

    Returns:
        {"status": "ok", "chunks": N} 或 {"status": "error", "message": "..."}
    """
    arxiv_id = _normalize_arxiv_id(arxiv_id) if arxiv_id else ""
    vectorstore = _get_vectorstore()

    # ── Step 1: 论文级别查重，已存在直接跳过 ──────────────────
    if _paper_already_ingested(vectorstore, arxiv_id, title):
        print(f"[Ingest] 已存在，跳过: {title}")
        return {"status": "ok", "chunks": 0, "skipped": True}

    # ── Step 2: 下载 PDF ───────────────────────────────────────
    file_path = _download_pdf(pdf_url)
    if not file_path:
        return {"status": "error", "message": f"PDF 下载失败: {pdf_url}"}

    try:
        # ── Step 3: 解析文本 ───────────────────────────────────
        raw_text = _parse_pdf(file_path)
        if not raw_text.strip():
            return {"status": "error", "message": f"PDF 文本为空: {title}"}

        # ── Step 4: 分块 + 构造 Document ──────────────────────
        text_chunks = _splitter.split_text(raw_text)
        documents, ids = [], []
        for i, chunk in enumerate(text_chunks):
            documents.append(Document(
                page_content=chunk,
                metadata={
                    "title":     title,
                    "arxiv_id":  arxiv_id,
                    "authors":   ", ".join(authors or []),
                    "published": published,
                    "chunk_idx": i,
                },
            ))
            ids.append(_make_chunk_id(title, arxiv_id, i))

        # ── Step 5: 用确定性 ID 写入，天然幂等 ────────────────
        # 即使因某种原因重复入库，相同 ID 只会覆盖写入，不产生垃圾数据
        _add_documents_fast(vectorstore, documents, ids)
        bump_corpus_version()   # 使 BM25 稀疏索引缓存失效
        return {"status": "ok", "chunks": len(documents)}

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@tool
def retrieve_context(query: str, k: int = 5) -> list[dict]:
    """
    从向量库中检索与问题最相关的论文片段。

    走完整检索管线：Self-Query → Hybrid(向量 + BM25) + RRF 融合 → 元数据过滤
    → Rerank 去噪 → top-k。相比朴素向量检索，能显著抑制脏库里的跨领域噪音。

    Args:
        query: 用户问题或检索关键词
        k:     返回最相关的段落数，默认 5
    """
    # 延迟导入：retriever 在模块级导入 rag_tool，这里延迟引用以避免循环导入
    from opendetect_ai.tools.retriever import retrieve
    return retrieve(query, k=k)


@tool
def list_ingested_papers() -> list[dict]:
    """列出向量库中已入库的所有论文（去重后）。"""
    vectorstore = _get_vectorstore()

    # 用公开接口，不用私有 _collection
    result = vectorstore.get(include=["metadatas"])
    if not result or not result.get("ids"):
        return [{"message": "向量库为空，尚未入库任何论文"}]

    seen, papers = set(), []
    for meta in result.get("metadatas", []):
        title = meta.get("title", "")
        if title not in seen:
            seen.add(title)
            papers.append({
                "title":     title,
                "arxiv_id":  meta.get("arxiv_id", ""),
                "published": meta.get("published", ""),
            })

    return papers

@tool
def ingest_local_pdf(
    file_path: str,
    title: str = "",
    authors: str = "",
    published: str = "",
) -> dict:
    """
    将本地 PDF 文件解析并存入 Chroma 向量库。

    Args:
        file_path:  本地 PDF 文件的绝对路径或相对路径
        title:      论文标题（可选，默认用文件名）
        authors:    作者（可选）
        published:  发表日期（可选）

    Returns:
        {"status": "ok", "chunks": N} 或 {"status": "error", "message": "..."}
    """

    # 规范化路径，自动处理正斜杠、反斜杠、相对路径
    path = Path(file_path.replace("\\", "/"))    # ← 加这一行
    
    if not path.exists():
        return {"status": "error", "message": f"文件不存在: {file_path}"}
    if path.suffix.lower() != ".pdf":
        return {"status": "error", "message": f"不是 PDF 文件: {file_path}"}

    # 没有提供标题就用文件名
    if not title:
        title = path.stem

    vectorstore = _get_vectorstore()

    # 去重检查（用标题）
    if _paper_already_ingested(vectorstore, "", title):
        return {"status": "ok", "chunks": 0, "skipped": True}

    try:
        raw_text = _parse_pdf(str(path))
        if not raw_text.strip():
            return {"status": "error", "message": f"PDF 文本为空: {file_path}"}

        text_chunks = _splitter.split_text(raw_text)
        documents, ids = [], []
        for i, chunk in enumerate(text_chunks):
            documents.append(Document(
                page_content=chunk,
                metadata={
                    "title":     title,
                    "arxiv_id":  "",
                    "authors":   authors,
                    "published": published,
                    "source":    "local",
                    "chunk_idx": i,
                },
            ))
            ids.append(_make_chunk_id(title, "", i))

        _add_documents_fast(vectorstore, documents, ids)
        bump_corpus_version()   # 使 BM25 稀疏索引缓存失效
        return {"status": "ok", "chunks": len(documents)}

    except Exception as e:
        return {"status": "error", "message": f"处理失败: {e}"}



# ── 导出给 Agent 使用的工具列表 ────────────────────────────────
RAG_TOOLS = [ingest_paper, retrieve_context, list_ingested_papers, ingest_local_pdf]
