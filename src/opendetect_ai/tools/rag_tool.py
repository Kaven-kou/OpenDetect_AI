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
from urllib.parse import urlparse

import fitz  # PyMuPDF
import chromadb
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
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
    OPENDETECT_MAX_PDF_MB,
    OPENDETECT_MAX_PDF_PAGES,
    OPENDETECT_PDF_ALLOWED_HOSTS,
)


# ── 初始化 Embeddings 和向量库 ─────────────────────────────────
_vectorstore: Chroma | None = None
_vectorstore_lock = threading.Lock()
_index_write_lock = threading.Lock()
_write_collection = None
_write_collection_lock = threading.Lock()

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

_PARSER_VERSION = 2
_EMBED_WORKERS = 8

_CAPTION_RE = re.compile(
    r"^\s*(?:(?P<figure>fig(?:ure)?\.?|图)|(?P<table>table|tab\.?|表))\s*"
    r"(?P<number>(?:[A-Za-z]?\d+(?:[.\-]\d+)*(?:[A-Za-z])?|[IVXLCDM]+))"
    r"(?=\s|[:.\-–—]|$)"
    r"\s*[:.\-–—]?\s*(?P<title>.*)$",
    re.IGNORECASE,
)

_PDF_SESSION = requests.Session()
_PDF_SESSION.mount(
    "https://",
    HTTPAdapter(max_retries=Retry(
        total=2,
        connect=2,
        read=2,
        status=2,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
    )),
)


def _add_documents_fast(vectorstore: Chroma, documents: list[Document], ids: list[str]) -> None:
    """并行生成 embedding，并通过 Chroma 官方 Collection API 批量写入。"""
    if not documents:
        return
    del vectorstore  # 保留参数以兼容既有调用点；写入由同目录的官方客户端完成。
    embeddings = _get_embeddings()
    texts = [document.page_content for document in documents]
    metadatas = [document.metadata for document in documents]
    groups = [texts[index:index + 10] for index in range(0, len(texts), 10)]
    with ThreadPoolExecutor(max_workers=min(_EMBED_WORKERS, len(groups))) as executor:
        grouped_vectors = list(executor.map(embeddings.embed_documents, groups))
    vectors = [vector for group in grouped_vectors for vector in group]
    _get_write_collection().upsert(
        ids=ids,
        embeddings=vectors,
        documents=texts,
        metadatas=metadatas,
    )


def _get_write_collection():
    global _write_collection
    if _write_collection is None:
        with _write_collection_lock:
            if _write_collection is None:
                client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
                _write_collection = client.get_or_create_collection("opendetect_papers")
    return _write_collection


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
        if not result or not result.get("ids"):
            return False
        metadatas = result.get("metadatas") or []
        return bool(metadatas) and all(
            int((meta or {}).get("parser_version", 0)) >= _PARSER_VERSION
            for meta in metadatas
        )
    except Exception:
        return False  # 查重失败不阻塞入库流程


def _existing_paper_ids(vectorstore: Chroma, arxiv_id: str, title: str) -> list[str]:
    """返回论文的现有块 ID，供解析器升级后清理不再使用的旧块。"""
    arxiv_id = _normalize_arxiv_id(arxiv_id) if arxiv_id else ""
    try:
        if arxiv_id:
            result = vectorstore.get(where={"arxiv_id": arxiv_id}, include=[])
        elif title:
            result = vectorstore.get(where={"title": title}, include=[])
        else:
            return []
        return list(result.get("ids") or [])
    except Exception:
        return []


def _delete_stale_chunks(vectorstore: Chroma, old_ids: list[str], new_ids: list[str]) -> None:
    """新块成功 upsert 后再删除多余旧块，避免解析失败破坏已有索引。"""
    stale = sorted(set(old_ids) - set(new_ids))
    if stale:
        vectorstore.delete(ids=stale)


def _download_pdf(pdf_url: str) -> tuple[str | None, str]:
    """受限下载 PDF，返回 ``(临时路径, 错误信息)``。"""
    tmp_path = ""
    resp = None
    max_bytes = OPENDETECT_MAX_PDF_MB * 1024 * 1024
    try:
        if not _is_allowed_pdf_url(pdf_url):
            return None, "PDF URL 不在允许的 HTTPS 域名白名单中"
        resp = _PDF_SESSION.get(pdf_url, timeout=(5, 30), stream=True)
        resp.raise_for_status()
        final_url = getattr(resp, "url", pdf_url) or pdf_url
        if not _is_allowed_pdf_url(final_url):
            return None, "PDF 重定向目标不在允许的 HTTPS 域名白名单中"
        content_length = resp.headers.get("Content-Length", "")
        if content_length.isdigit() and int(content_length) > max_bytes:
            return None, f"PDF 超过 {OPENDETECT_MAX_PDF_MB} MB 上限"

        total = 0
        header = b""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            tmp_path = f.name
            for chunk in resp.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                if len(header) < 5:
                    header = (header + chunk)[:5]
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(f"PDF 超过 {OPENDETECT_MAX_PDF_MB} MB 上限")
                f.write(chunk)
        if header != b"%PDF-":
            raise ValueError("响应内容不是有效的 PDF")
        return tmp_path, ""
    except Exception as exc:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        return None, str(exc)
    finally:
        if resp is not None:
            close = getattr(resp, "close", None)
            if callable(close):
                close()


def _is_allowed_pdf_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        return False
    host = parsed.hostname.lower().rstrip(".")
    return any(host == allowed or host.endswith(f".{allowed}") for allowed in OPENDETECT_PDF_ALLOWED_HOSTS)


def _parse_pdf_pages(file_path: str) -> list[tuple[int, str]]:
    """用 PyMuPDF 按页提取文本；``sort=True`` 按坐标排序以改善部分复杂布局。"""
    with fitz.open(file_path) as doc:
        if doc.needs_pass:
            raise ValueError("PDF 已加密，无法解析")
        if doc.page_count > OPENDETECT_MAX_PDF_PAGES:
            raise ValueError(f"PDF 超过 {OPENDETECT_MAX_PDF_PAGES} 页上限")
        return [
            (page.number + 1, page.get_text("text", sort=True).strip())
            for page in doc
        ]


def _normalise_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _parse_caption(text: str) -> dict | None:
    """识别英文/中文表图标题，并返回稳定的类型、编号和标题。"""
    normalised = _normalise_text(text)
    match = _CAPTION_RE.match(normalised)
    if not match:
        return None
    kind = "figure" if match.group("figure") else "table"
    number = match.group("number")
    prefix = "Figure" if kind == "figure" else "Table"
    return {
        "kind": kind,
        "number": number,
        "label": f"{prefix} {number}",
        "title": _normalise_text(match.group("title")),
        "caption": normalised,
    }


def _markdown_cell(value: object) -> str:
    return _normalise_text(value).replace("|", "\\|")


def _table_to_markdown(rows: list[list[object]]) -> str:
    """把 PyMuPDF 的二维表格结果转换为列数稳定的 Markdown。"""
    cleaned = [[_markdown_cell(cell) for cell in row] for row in rows if row]
    if not cleaned:
        return ""
    width = max(len(row) for row in cleaned)
    if width == 0:
        return ""
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]
    if not any(any(cell for cell in row) for row in cleaned):
        return ""
    header = cleaned[0]
    body = cleaned[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def _bbox_string(bbox: tuple | list | None) -> str:
    if not bbox:
        return ""
    return ",".join(f"{float(value):.1f}" for value in bbox[:4])


def _vertical_gap(first: tuple | list, second: tuple | list) -> float:
    if first[3] < second[1]:
        return float(second[1] - first[3])
    if second[3] < first[1]:
        return float(first[1] - second[3])
    return 0.0


def _layout_distance(first: tuple | list, second: tuple | list) -> float:
    first_center = (first[0] + first[2]) / 2
    second_center = (second[0] + second[2]) / 2
    return _vertical_gap(first, second) + abs(first_center - second_center) * 0.25


def _inside_any_region(bbox: tuple | list, regions: list[tuple]) -> bool:
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    return any(
        region[0] <= center_x <= region[2] and region[1] <= center_y <= region[3]
        for region in regions
    )


def _reference_pattern(kind: str, number: str) -> re.Pattern:
    names = r"(?:fig(?:ure)?\.?|图)" if kind == "figure" else r"(?:table|tab\.?|表)"
    return re.compile(
        rf"{names}\s*{re.escape(number)}(?=\s|[,.;:，。；：)\]]|$)",
        re.IGNORECASE,
    )


def _find_references(
    pages: list[dict],
    *,
    kind: str,
    number: str,
) -> list[dict]:
    """在全文中查找表/图号引用，排除标题自身并保留引用页码。"""
    pattern = _reference_pattern(kind, number)
    caption_texts = {
        _normalise_text(element.get("caption", "")).casefold()
        for page in pages
        for element in page["tables"] + page["figures"]
        if element["kind"] == kind and element["number"].casefold() == number.casefold()
    }
    caption_texts.discard("")
    references = []
    seen = set()
    for page in pages:
        parts = re.split(r"(?<=[.!?。！？])\s+|\n+", page["raw_text"])
        for part in parts:
            snippet = _normalise_text(part)
            if not snippet or not pattern.search(snippet):
                continue
            if snippet.casefold() in caption_texts:
                continue
            key = (page["page"], snippet.casefold())
            if key in seen:
                continue
            seen.add(key)
            references.append({"page": page["page"], "text": snippet[:400]})
            if len(references) >= 8:
                return references
    return references


def _format_references(references: list[dict]) -> str:
    if not references:
        return ""
    lines = ["\n正文引用："]
    lines.extend(f"- 第 {item['page']} 页：{item['text']}" for item in references)
    return "\n".join(lines)


def _parse_pdf_layout(file_path: str) -> list[dict]:
    """提取正文、Markdown 表格、图片位置和图注，随后建立跨页正文引用。"""
    pages: list[dict] = []
    with fitz.open(file_path) as doc:
        if doc.needs_pass:
            raise ValueError("PDF 已加密，无法解析")
        if doc.page_count > OPENDETECT_MAX_PDF_PAGES:
            raise ValueError(f"PDF 超过 {OPENDETECT_MAX_PDF_PAGES} 页上限")

        for page in doc:
            blocks = [block for block in page.get_text("blocks", sort=True) if block[6] == 0]
            captions = []
            for block in blocks:
                parsed = _parse_caption(block[4])
                if parsed:
                    captions.append({**parsed, "bbox": tuple(block[:4])})

            tables = []
            try:
                found_tables = page.find_tables().tables
            except Exception as exc:
                print(f"[PDF] 第 {page.number + 1} 页表格识别失败，退化为正文: {exc}")
                found_tables = []
            table_regions = [tuple(table.bbox) for table in found_tables]
            table_captions = [item for item in captions if item["kind"] == "table"]
            used_caption_ids: set[int] = set()
            for index, table in enumerate(found_tables, start=1):
                markdown = _table_to_markdown(table.extract())
                if not markdown:
                    continue
                candidates = [
                    (idx, caption)
                    for idx, caption in enumerate(table_captions)
                    if idx not in used_caption_ids
                ]
                nearest = min(
                    candidates,
                    key=lambda item: _layout_distance(table.bbox, item[1]["bbox"]),
                    default=None,
                )
                if nearest and _vertical_gap(table.bbox, nearest[1]["bbox"]) <= 160:
                    caption_idx, caption = nearest
                    used_caption_ids.add(caption_idx)
                else:
                    caption = {
                        "kind": "table",
                        "number": f"p{page.number + 1}-{index}",
                        "label": f"Table p{page.number + 1}-{index}",
                        "title": "",
                        "caption": "",
                    }
                tables.append({
                    **caption,
                    "bbox": tuple(table.bbox),
                    "markdown": markdown,
                    "caption_detected": bool(caption.get("caption")),
                })

            body_blocks = [
                _normalise_text(block[4])
                for block in blocks
                if not _inside_any_region(block[:4], table_regions)
            ]
            raw_text = page.get_text("text", sort=True).strip()
            images = [
                image
                for image in page.get_image_info(xrefs=True)
                if image.get("width", 0) >= 80
                and image.get("height", 0) >= 50
                and image["bbox"][2] - image["bbox"][0] >= 40
                and image["bbox"][3] - image["bbox"][1] >= 30
            ]
            figure_captions = [item for item in captions if item["kind"] == "figure"]
            figures = []
            used_images: set[int] = set()
            for caption in figure_captions:
                candidates = [
                    (idx, image)
                    for idx, image in enumerate(images)
                    if idx not in used_images
                ]
                nearest = min(
                    candidates,
                    key=lambda item: _layout_distance(item[1]["bbox"], caption["bbox"]),
                    default=None,
                )
                image = None
                if nearest and _vertical_gap(nearest[1]["bbox"], caption["bbox"]) <= 240:
                    image_idx, image = nearest
                    used_images.add(image_idx)
                figures.append({
                    **caption,
                    "bbox": tuple(image["bbox"]) if image else (),
                    "image_xref": int(image.get("xref", 0)) if image else 0,
                    "image_width": int(image.get("width", 0)) if image else 0,
                    "image_height": int(image.get("height", 0)) if image else 0,
                    "image_detected": image is not None,
                })

            pages.append({
                "page": page.number + 1,
                "text": "\n".join(filter(None, body_blocks)),
                "raw_text": raw_text,
                "tables": tables,
                "figures": figures,
            })

    for page in pages:
        for element in page["tables"] + page["figures"]:
            element["references"] = _find_references(
                pages,
                kind=element["kind"],
                number=element["number"],
            )
    return pages


def _build_pdf_documents(
    file_path: str,
    *,
    title: str,
    arxiv_id: str,
    authors: str,
    published: str,
    source: str,
) -> tuple[list[Document], list[str], int]:
    """按页构造正文、Markdown 表格和图片语义块，保留结构化关联元数据。"""
    pages = _parse_pdf_layout(file_path)
    documents: list[Document] = []
    ids: list[str] = []
    chunk_idx = 0
    base_metadata = {
        "title": title,
        "arxiv_id": arxiv_id,
        "authors": authors,
        "published": published,
        "source": source,
        "parser_version": _PARSER_VERSION,
    }
    for page in pages:
        page_number = page["page"]
        for page_chunk_idx, chunk in enumerate(_splitter.split_text(page["text"])):
            documents.append(Document(
                page_content=chunk,
                metadata={
                    **base_metadata,
                    "page": page_number,
                    "page_chunk_idx": page_chunk_idx,
                    "chunk_idx": chunk_idx,
                    "element_type": "text",
                    "element_number": "",
                    "element_title": "",
                    "reference_pages": "",
                },
            ))
            ids.append(_make_chunk_id(title, arxiv_id, chunk_idx))
            chunk_idx += 1

        for table in page["tables"]:
            references = table["references"]
            heading = table["label"] + (f": {table['title']}" if table["title"] else "")
            content = f"## {heading}\n\n{table['markdown']}{_format_references(references)}"
            documents.append(Document(
                page_content=content,
                metadata={
                    **base_metadata,
                    "page": page_number,
                    "page_chunk_idx": -1,
                    "chunk_idx": chunk_idx,
                    "element_type": "table",
                    "element_number": table["number"],
                    "element_title": table["title"],
                    "element_bbox": _bbox_string(table["bbox"]),
                    "caption_detected": table["caption_detected"],
                    "reference_pages": ",".join(str(ref["page"]) for ref in references),
                    "reference_count": len(references),
                },
            ))
            ids.append(_make_chunk_id(title, arxiv_id, chunk_idx))
            chunk_idx += 1

        for figure in page["figures"]:
            references = figure["references"]
            heading = figure["label"] + (f": {figure['title']}" if figure["title"] else "")
            image_status = "已关联页面图片" if figure["image_detected"] else "仅识别到图注（可能是矢量图）"
            content = (
                f"## {heading}\n\n图注：{figure['caption']}\n\n{image_status}"
                f"{_format_references(references)}"
            )
            documents.append(Document(
                page_content=content,
                metadata={
                    **base_metadata,
                    "page": page_number,
                    "page_chunk_idx": -1,
                    "chunk_idx": chunk_idx,
                    "element_type": "figure",
                    "element_number": figure["number"],
                    "element_title": figure["title"],
                    "element_bbox": _bbox_string(figure["bbox"]),
                    "image_xref": figure["image_xref"],
                    "image_width": figure["image_width"],
                    "image_height": figure["image_height"],
                    "image_detected": figure["image_detected"],
                    "reference_pages": ",".join(str(ref["page"]) for ref in references),
                    "reference_count": len(references),
                },
            ))
            ids.append(_make_chunk_id(title, arxiv_id, chunk_idx))
            chunk_idx += 1
    return documents, ids, len(pages)


def _document_counts(documents: list[Document]) -> dict[str, int]:
    counts = {"text": 0, "tables": 0, "figures": 0}
    for document in documents:
        kind = document.metadata.get("element_type", "text")
        if kind == "table":
            counts["tables"] += 1
        elif kind == "figure":
            counts["figures"] += 1
        else:
            counts["text"] += 1
    return counts


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
    with _index_write_lock:
        if _paper_already_ingested(vectorstore, arxiv_id, title):
            print(f"[Ingest] 已存在，跳过: {title}")
            return {"status": "ok", "chunks": 0, "skipped": True}

    # ── Step 2: 下载 PDF ───────────────────────────────────────
    file_path, download_error = _download_pdf(pdf_url)
    if not file_path:
        return {
            "status": "error",
            "message": f"PDF 下载失败: {download_error or pdf_url}",
        }

    try:
        # ── Step 3/4: 按页解析、分块并保留页码 ─────────────────
        documents, ids, page_count = _build_pdf_documents(
            file_path,
            title=title,
            arxiv_id=arxiv_id,
            authors=", ".join(authors or []),
            published=published,
            source=pdf_url,
        )
        if not documents:
            return {"status": "error", "message": f"PDF 文本为空: {title}"}

        # ── Step 5: 用确定性 ID 写入，天然幂等 ────────────────
        # 即使因某种原因重复入库，相同 ID 只会覆盖写入，不产生垃圾数据
        with _index_write_lock:
            if _paper_already_ingested(vectorstore, arxiv_id, title):
                return {"status": "ok", "chunks": 0, "skipped": True}
            old_ids = _existing_paper_ids(vectorstore, arxiv_id, title)
            _add_documents_fast(vectorstore, documents, ids)
            _delete_stale_chunks(vectorstore, old_ids, ids)
            bump_corpus_version()   # 使 BM25 稀疏索引缓存失效
        counts = _document_counts(documents)
        return {
            "status": "ok",
            "chunks": len(documents),
            "pages": page_count,
            "tables": counts["tables"],
            "figures": counts["figures"],
            "reindexed": bool(old_ids),
        }

    except Exception as exc:
        return {"status": "error", "message": f"PDF 处理失败: {exc}"}

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
    with _index_write_lock:
        if _paper_already_ingested(vectorstore, "", title):
            return {"status": "ok", "chunks": 0, "skipped": True}

    try:
        documents, ids, page_count = _build_pdf_documents(
            str(path),
            title=title,
            arxiv_id="",
            authors=authors,
            published=published,
            source="local",
        )
        if not documents:
            return {"status": "error", "message": f"PDF 文本为空: {file_path}"}

        with _index_write_lock:
            if _paper_already_ingested(vectorstore, "", title):
                return {"status": "ok", "chunks": 0, "skipped": True}
            old_ids = _existing_paper_ids(vectorstore, "", title)
            _add_documents_fast(vectorstore, documents, ids)
            _delete_stale_chunks(vectorstore, old_ids, ids)
            bump_corpus_version()   # 使 BM25 稀疏索引缓存失效
        counts = _document_counts(documents)
        return {
            "status": "ok",
            "chunks": len(documents),
            "pages": page_count,
            "tables": counts["tables"],
            "figures": counts["figures"],
            "reindexed": bool(old_ids),
        }

    except Exception as e:
        return {"status": "error", "message": f"处理失败: {e}"}



# ── 导出给 Agent 使用的工具列表 ────────────────────────────────
RAG_TOOLS = [ingest_paper, retrieve_context, list_ingested_papers, ingest_local_pdf]
