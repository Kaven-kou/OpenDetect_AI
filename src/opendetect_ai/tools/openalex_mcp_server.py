"""
OpenAlex MCP Server —— OpenDetect_AI
将论文搜索功能封装为标准 MCP 工具，供任何支持 MCP 协议的客户端调用。
运行方式: python openalex_mcp_server.py
"""

from __future__ import annotations

import requests
from mcp.server.fastmcp import FastMCP

# ── MCP Server 实例 ────────────────────────────────────────────
mcp = FastMCP(name="openalex-search")

# ── OpenAlex API 配置 ──────────────────────────────────────────
OA_SEARCH_URL = "https://api.openalex.org/works"
OA_WORK_URL   = "https://api.openalex.org/works/{work_id}"
OA_HEADERS    = {"User-Agent": "OpenDetect-AI/1.0 (mailto:research@example.com)"}
OA_SELECT     = (
    "display_name,authorships,abstract_inverted_index,"
    "publication_date,locations,open_access,doi,cited_by_count"
)


# ── 内部解析函数 ───────────────────────────────────────────────
def _restore_abstract(inverted_index: dict | None) -> str:
    if not inverted_index:
        return ""
    tokens = [""] * (max(max(v) for v in inverted_index.values()) + 1)
    for word, positions in inverted_index.items():
        for pos in positions:
            tokens[pos] = word
    return " ".join(tokens)


def _parse(work: dict) -> dict:
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

    # ── PDF 链接：有 arxiv_id 优先用官方地址，不信任第三方镜像 ──
    if arxiv_id:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
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
        "cited_by":  work.get("cited_by_count", 0),
    }


# ── MCP 工具定义 ───────────────────────────────────────────────
@mcp.tool()
def search_papers(query: str, max_results: int = 5) -> list[dict]:
    """
    在 OpenAlex 上搜索学术论文。

    Args:
        query:       英文搜索短句，例如 "vision transformer image classification"
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
        return [{"error": f"搜索失败: {e}"}]
    return [_parse(w) for w in works] if works else [{"error": f"未找到: {query}"}]


@mcp.tool()
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
            return _parse(resp.json())
    except requests.RequestException:
        pass

    # 方式2: DOI 格式（适合2022年后的论文）
    try:
        resp = requests.get(
            OA_WORK_URL.format(work_id=f"doi:10.48550/arXiv.{arxiv_id}"),
            params=select, headers=OA_HEADERS, timeout=15,
        )
        if resp.status_code == 200:
            return _parse(resp.json())
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
            return _parse(works[0])
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
            return _parse(works[0])
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
            return _parse(works[0])
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
            result = _parse(w)
            if result.get("arxiv_id") == arxiv_id:
                return result
    except requests.RequestException:
        pass

    return {"error": f"六种方式均未找到 arxiv:{arxiv_id}"}


@mcp.tool()
def get_paper_by_title(title: str) -> dict:
    """
    根据论文标题搜索最匹配的一篇。

    Args:
        title: 例如 "Attention Is All You Need"
    """
    try:
        resp = requests.get(
            OA_SEARCH_URL,
            params={"search": title, "per-page": 1, "select": OA_SELECT},
            headers=OA_HEADERS, timeout=15,
        )
        resp.raise_for_status()
        works = resp.json().get("results", [])
    except requests.RequestException as e:
        return {"error": f"搜索失败: {e}"}
    return _parse(works[0]) if works else {"error": f"未找到: {title}"}


# ── 入口 ───────────────────────────────────────────────────────
if __name__ == "__main__":
    mcp.run(transport="stdio")