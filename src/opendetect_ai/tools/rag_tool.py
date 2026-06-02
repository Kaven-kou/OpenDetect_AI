"""
RAG 工具 —— OpenDetect_AI
负责论文 PDF 下载、文本解析、向量入库、语义检索。
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
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
def _get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=OPENDETECT_EMBED_MODEL,
        base_url=OPENDETECT_EMBED_BASE_URL,
        api_key=OPENDETECT_EMBED_API_KEY,
        check_embedding_ctx_length=False,
        chunk_size=5,
    )


def _get_vectorstore() -> Chroma:
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    return Chroma(
        collection_name="opendetect_papers",
        embedding_function=_get_embeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
    )


# ── 文本分块器 ─────────────────────────────────────────────────
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "],
)


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
        vectorstore.add_documents(documents, ids=ids)
        return {"status": "ok", "chunks": len(documents)}

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@tool
def retrieve_context(query: str, k: int = 5) -> list[dict]:
    """
    从向量库中检索与问题最相关的论文片段。

    Args:
        query: 用户问题或检索关键词
        k:     返回最相关的段落数，默认 5
    """
    vectorstore = _get_vectorstore()

    # 用公开接口检查是否为空
    result = vectorstore.get(limit=1)
    if not result or not result.get("ids"):
        return [{"error": "向量库为空，请先使用 ingest_paper 工具入库论文"}]

    docs = vectorstore.similarity_search(query, k=k)
    return [
        {
            "content":   doc.page_content,
            "title":     doc.metadata.get("title", ""),
            "arxiv_id":  doc.metadata.get("arxiv_id", ""),
            "published": doc.metadata.get("published", ""),
            "chunk_idx": doc.metadata.get("chunk_idx", 0),
        }
        for doc in docs
    ]


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

        vectorstore.add_documents(documents, ids=ids)
        return {"status": "ok", "chunks": len(documents)}

    except Exception as e:
        return {"status": "error", "message": f"处理失败: {e}"}



# ── 导出给 Agent 使用的工具列表 ────────────────────────────────
RAG_TOOLS = [ingest_paper, retrieve_context, list_ingested_papers, ingest_local_pdf]
