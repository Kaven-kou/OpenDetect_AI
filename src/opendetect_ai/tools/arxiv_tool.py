"""
学术论文搜索工具 —— OpenDetect_AI
使用 OpenAlex API（免费、无需 Key、国内可访问）
"""

from __future__ import annotations

import requests
from langchain_core.tools import tool


# ── OpenAlex API 配置 ──────────────────────────────────────────
OA_SEARCH_URL = "https://api.openalex.org/works"
OA_WORK_URL   = "https://api.openalex.org/works/{work_id}"   # ← 新增，用于精确获取
OA_HEADERS    = {"User-Agent": "OpenDetect-AI/1.0 (mailto:research@example.com)"}
OA_SELECT     = "display_name,authorships,abstract_inverted_index,publication_date,locations,open_access,doi,cited_by_count"


def _parse_oa_result(work: dict) -> dict:
    """把 OpenAlex work 对象转成项目统一格式。"""
    authors = [
        a.get("author", {}).get("display_name", "")
        for a in work.get("authorships", [])[:3]
    ]

    arxiv_id = ""
    for loc in work.get("locations", []):
        url = loc.get("landing_page_url", "") or ""
        if "arxiv.org" in url:
            arxiv_id = url.split("/abs/")[-1].split("v")[0]
            break

    if arxiv_id:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"   # 有 arxiv_id 永远用官方
    else:
        oa_info = work.get("open_access", {})
        pdf_url = oa_info.get("oa_url", "")

    abstract = _restore_abstract(work.get("abstract_inverted_index"))

    return {
        "title":     work.get("display_name", ""),
        "authors":   authors,
        "abstract":  abstract[:300] + "..." if len(abstract) > 300 else abstract,
        "arxiv_id":  arxiv_id,
        "pdf_url":   pdf_url,
        "published": (work.get("publication_date") or "")[:10],
        "doi":       work.get("doi", ""),
        "cited_by":  work.get("cited_by_count", 0),
    }


def _restore_abstract(inverted_index: dict | None) -> str:
    if not inverted_index:
        return ""
    tokens = [""] * (max(max(v) for v in inverted_index.values()) + 1)
    for word, positions in inverted_index.items():
        for pos in positions:
            tokens[pos] = word
    return " ".join(tokens)


@tool
def search_papers(query: str, max_results: int = 5) -> list[dict]:
    """
    在 OpenAlex 上搜索学术论文（覆盖 arxiv、ACM、IEEE、Nature 等来源）。

    Args:
        query:       搜索关键词，建议英文短句，例如 "vision transformer image classification"
        max_results: 最多返回几篇，默认 5，最大 10
    """
    max_results = min(max_results, 10)
    params = {
        "search":   query,
        "per-page": max_results,
        "select":   OA_SELECT,
        "sort":     "relevance_score:desc",
    }
    try:
        resp = requests.get(OA_SEARCH_URL, params=params, headers=OA_HEADERS, timeout=15)
        resp.raise_for_status()
        works = resp.json().get("results", [])
    except requests.RequestException as e:
        return [{"error": f"搜索请求失败: {e}"}]

    if not works:
        return [{"error": f"未找到与 '{query}' 相关的论文"}]
    return [_parse_oa_result(w) for w in works]


@tool
def get_paper_by_id(arxiv_id: str) -> dict:
    """根据 arxiv ID 精确获取论文，依次尝试多种方式。"""
    select = {"select": OA_SELECT}

    # 方式1: arxiv:ID 格式
    try:
        resp = requests.get(
            OA_WORK_URL.format(work_id=f"arxiv:{arxiv_id}"),
            params=select, headers=OA_HEADERS, timeout=15,
        )
        if resp.status_code == 200:
            return _parse_oa_result(resp.json())
    except requests.RequestException:
        pass

    # 方式2: DOI 格式（适合2022年后的论文）
    try:
        resp = requests.get(
            OA_WORK_URL.format(work_id=f"doi:10.48550/arXiv.{arxiv_id}"),
            params=select, headers=OA_HEADERS, timeout=15,
        )
        if resp.status_code == 200:
            return _parse_oa_result(resp.json())
    except requests.RequestException:
        pass

    # 方式3: https URL 过滤
    try:
        resp = requests.get(
            OA_SEARCH_URL,
            params={"filter": f"locations.landing_page_url:https://arxiv.org/abs/{arxiv_id}", "select": OA_SELECT},
            headers=OA_HEADERS, timeout=15,
        )
        works = resp.json().get("results", [])
        if works:
            return _parse_oa_result(works[0])
    except requests.RequestException:
        pass

    # 方式4: http URL 过滤（老论文常见）← 新增
    try:
        resp = requests.get(
            OA_SEARCH_URL,
            params={"filter": f"locations.landing_page_url:http://arxiv.org/abs/{arxiv_id}", "select": OA_SELECT},
            headers=OA_HEADERS, timeout=15,
        )
        works = resp.json().get("results", [])
        if works:
            return _parse_oa_result(works[0])
    except requests.RequestException:
        pass

    # 方式5: 不带协议的 URL 过滤 ← 新增
    try:
        resp = requests.get(
            OA_SEARCH_URL,
            params={"filter": f"locations.landing_page_url:arxiv.org/abs/{arxiv_id}", "select": OA_SELECT},
            headers=OA_HEADERS, timeout=15,
        )
        works = resp.json().get("results", [])
        if works:
            return _parse_oa_result(works[0])
    except requests.RequestException:
        pass

    # 方式6: 关键词搜索兜底（用 arxiv ID 当搜索词）← 新增
    try:
        resp = requests.get(
            OA_SEARCH_URL,
            params={"search": arxiv_id, "per-page": 3, "select": OA_SELECT},
            headers=OA_HEADERS, timeout=15,
        )
        works = resp.json().get("results", [])
        # 验证结果里确实有这个 arxiv ID，防止搜错
        for w in works:
            result = _parse_oa_result(w)
            if result.get("arxiv_id") == arxiv_id:
                return result
    except requests.RequestException:
        pass

    return {"error": f"六种方式均未找到 arxiv:{arxiv_id}"}


@tool
def get_paper_by_title(title: str) -> dict:
    """
    根据论文标题搜索单篇论文，返回最匹配的一篇。

    Args:
        title: 论文标题，例如 "An Image is Worth 16x16 Words"
    """
    params = {
        "search":   title,
        "per-page": 1,
        "select":   OA_SELECT,
    }
    try:
        resp = requests.get(OA_SEARCH_URL, params=params, headers=OA_HEADERS, timeout=15)
        resp.raise_for_status()
        works = resp.json().get("results", [])
    except requests.RequestException as e:
        return {"error": f"搜索失败: {e}"}

    if not works:
        return {"error": f"未找到论文: {title}"}
    return _parse_oa_result(works[0])


# get_paper_by_id / get_paper_by_title / search_papers 均可直接 import 使用
# 本文件不再导出 ARXIV_TOOLS 列表（Search Agent 通过 MCP 调用，不使用此列表）