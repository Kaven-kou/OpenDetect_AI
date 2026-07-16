"""
Search Agent —— OpenDetect_AI
负责根据用户需求搜索学术论文，结果写入 state.search_results。
"""

from __future__ import annotations

import json                                          # ← 新增
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

from opendetect_ai.state import AgentState, PaperMeta
from opendetect_ai.context_utils import build_context_str
from opendetect_ai.tools.progress import push_progress
from opendetect_ai.tools.mcp_client import call_mcp_tool
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
        temperature=0,                  # ← 从 0.2 改成 0，意图识别需要确定性
    )


def _extract_query(user_query: str, llm: ChatOpenAI) -> str:
    """保留原有函数，作为 keyword 搜索的后备。"""
    lowered = user_query.lower()
    if "diffusion model" in lowered or "diffusion models" in lowered:
        return "denoising diffusion probabilistic models image generation"

    prompt = (
        "请从以下用户问题中提取最适合 AI / 深度学习论文搜索引擎检索的英文短语。\n"
        "要求：输出一个简洁的英文短句（不超过8个词），不要用逗号分隔关键词。\n"
        "例如：'ViT vision transformer image recognition' 而不是 'ViT, vision, transformer'\n"
        "如果术语存在跨学科歧义，请补充 AI 语境词，例如 diffusion model 应输出 'diffusion models deep learning image generation'。\n"
        "只输出这个短句，不要有任何其他内容。\n\n"
        f"用户问题：{user_query}"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip().strip('"').strip("'")


# ── 新增函数 ───────────────────────────────────────────────────
def _identify_search_intent(user_query: str, llm: ChatOpenAI, chat_context: str = "") -> dict:
    """
    用 LLM 自身的知识分析用户真正想搜什么，返回搜索策略。

    返回格式：
        {"type": "arxiv_id", "value": "2010.11929", "reason": "ViT原版论文"}
        {"type": "title",    "value": "Attention Is All You Need", "reason": "..."}
        {"type": "keyword",  "value": "diffusion model generation", "reason": "..."}
    """
    # 用 % 格式化插入变量，彻底避免 f-string / .format() 解析 chat_context 里的花括号
    # （chat_context 可能含 JSON 片段如 {"type":...}，会触发 KeyError）
    if chat_context:
        ctx_section = (
            "\n## 对话上下文（最近几轮对话，用于理解指代词和追问）\n"
            + chat_context
            + "\n\n**重要**：如果用户使用了\"它\"、\"这个\"、\"还有\"、\"其他的\"等指代词或追问，\n"
            "必须从上下文中提取当前会话的核心话题作为搜索方向，\n"
            "不要使用宽泛的 deep learning 等通用词。\n"
        )
    else:
        ctx_section = ""

    prompt_template = (
        "你是一个深度学习领域的论文专家，请分析用户的搜索意图并输出搜索策略。\n"
        "%s"
        "\n## 判断规则\n\n"
        "**情况A**：用户描述的是某篇特定著名论文（如\"ViT原版\"、\"首次提出XXX的论文\"）\n"
        "\u2192 用你的知识识别出该论文的 arxiv ID\n"
        '\u2192 输出: {"type": "arxiv_id", "value": "2010.11929", "reason": "ViT原版论文"}\n\n'
        "**情况B**：用户说出了大致标题\n"
        "\u2192 提取最准确的英文标题\n"
        '\u2192 输出: {"type": "title", "value": "Attention Is All You Need", "reason": "用户提到了标题"}\n\n'
        "**情况C**：用户想找某方向的多篇论文，或使用指代词追问更多论文\n"
        "\u2192 若有对话上下文，必须基于上下文话题搜索，不能用宽泛词\n"
        '\u2192 输出: {"type": "keyword", "value": "instance segmentation deep learning", "reason": "基于上下文"}\n\n'
        "**情况D**：用户提供了 arxiv 链接（如 https://arxiv.org/abs/2010.11929）\n"
        "\u2192 提取其中的 arxiv ID\n"
        '\u2192 输出: {"type": "arxiv_id", "value": "2010.11929", "reason": "用户提供了arxiv链接"}\n\n'
        "**情况E**：用户使用\"还有吗\"、\"其他的\"、\"更多\"等追问\n"
        "\u2192 从对话上下文中提取核心话题，输出该话题的关键词搜索\n"
        '\u2192 错误示例: {"type": "keyword", "value": "deep learning"}（太宽泛）\n'
        '\u2192 正确示例: {"type": "keyword", "value": "instance segmentation methods"}（基于上下文）\n\n'
        "## 常用著名论文 arxiv ID 参考\n"
        "ViT 2010.11929 | BERT 1810.04805 | GPT-3 2005.14165\n"
        "ResNet 1512.03385 | Transformer 1706.03762 | CLIP 2103.00020\n"
        "Swin Transformer 2103.14030 | DINO 2104.14294 | DINOv2 2304.07193\n"
        "DETR 2005.12872 | SAM 2304.02643 | GroundingDINO 2303.05499\n"
        "Stable Diffusion 2112.10752 | LLaMA 2302.13971 | LLaMA2 2307.09288\n\n"
        "只输出 JSON，不要有任何其他内容。\n\n"
        "用户需求：%s"
    )
    prompt = prompt_template % (ctx_section, user_query)

    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()
    try:
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception:
        return {"type": "keyword", "value": user_query, "reason": "解析失败，使用原始输入"}


def _parse_results(raw_results: list[dict] | dict | str) -> list[PaperMeta]:
    """把搜索工具返回值统一转成 PaperMeta 列表。"""
    if isinstance(raw_results, str):
        raw_results = _parse_arxiv_text_results(raw_results)

    if isinstance(raw_results, dict) and isinstance(raw_results.get("papers"), list):
        raw_results = raw_results["papers"]
    elif isinstance(raw_results, dict) and isinstance(raw_results.get("results"), list):
        raw_results = raw_results["results"]
    elif isinstance(raw_results, dict):   # ← 新增：兼容单篇返回
        raw_results = [raw_results]
    elif not isinstance(raw_results, list):
        return []

    papers = []
    for r in raw_results:
        if "error" in r:
            continue
        arxiv_id = _extract_arxiv_id(r)
        authors = r.get("authors", [])
        if isinstance(authors, str):
            authors = [a.strip() for a in re.split(r",|;", authors) if a.strip()]
        elif isinstance(authors, list):
            authors = [
                a.get("name", "") if isinstance(a, dict) else str(a)
                for a in authors
            ]

        papers.append(PaperMeta(
            title     = r.get("title", "") or r.get("name", ""),
            authors   = [a for a in authors if a],
            abstract  = r.get("abstract", "") or r.get("summary", ""),
            arxiv_id  = arxiv_id,
            pdf_url   = r.get("pdf_url", "") or r.get("pdfUrl", "") or (f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else ""),
            published = (r.get("published", "") or r.get("published_date", "") or r.get("updated", ""))[:10],
            ingested  = False,
        ))
    return papers


def _extract_arxiv_id(result: dict) -> str:
    for key in ("arxiv_id", "arxivId", "id"):
        value = result.get(key, "")
        if value:
            value = str(value)
            match = re.search(r"arxiv\.org/(?:abs|pdf)/([^/?#]+)", value)
            if match:
                return _normalize_arxiv_id(match.group(1).replace(".pdf", ""))
            return _normalize_arxiv_id(value.replace("arXiv:", "").strip())

    entry_id = result.get("entry_id", "") or result.get("url", "")
    match = re.search(r"arxiv\.org/(?:abs|pdf)/([^/?#]+)", str(entry_id))
    if match:
        return _normalize_arxiv_id(match.group(1).replace(".pdf", ""))

    return ""


def _normalize_arxiv_id(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id.strip())


def _parse_arxiv_text_results(text: str) -> list[dict]:
    """兼容远程 ArXiv MCP 返回纯文本搜索结果的情况。"""
    if not text.strip():
        return []

    blocks = re.split(r"\n\s*\n", text.strip())
    papers: list[dict] = []
    for block in blocks:
        title_match = re.search(r"(?:Title|标题)\s*:\s*(.+)", block, re.IGNORECASE)
        id_match = re.search(r"(?:arXiv(?:\s*ID)?|arxiv_id)\s*:\s*([^\s]+)", block, re.IGNORECASE)
        authors_match = re.search(r"(?:Authors|作者)\s*:\s*(.+)", block, re.IGNORECASE)
        published_match = re.search(r"(?:Published|发表|Date)\s*:\s*(.+)", block, re.IGNORECASE)
        abstract_match = re.search(
            r"(?:Summary|Abstract|摘要)\s*:\s*(.+)",
            block,
            re.IGNORECASE | re.DOTALL,
        )

        if not (title_match or id_match):
            continue

        arxiv_id = _normalize_arxiv_id(id_match.group(1).strip().rstrip(".")) if id_match else ""
        title = title_match.group(1).strip() if title_match else f"arXiv:{arxiv_id}"
        authors = []
        if authors_match:
            authors = [a.strip() for a in re.split(r",|;", authors_match.group(1)) if a.strip()]

        papers.append({
            "title": title,
            "authors": authors,
            "abstract": abstract_match.group(1).strip() if abstract_match else "",
            "arxiv_id": arxiv_id,
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else "",
            "published": published_match.group(1).strip()[:10] if published_match else "",
        })
    return papers


def _has_tool_error(raw_results: list[dict] | dict | str) -> bool:
    if isinstance(raw_results, dict):
        return "error" in raw_results
    if isinstance(raw_results, list):
        return bool(raw_results) and all(isinstance(r, dict) and "error" in r for r in raw_results)
    return False


def _call_openalex(search_type: str, search_value: str, user_query: str, llm: ChatOpenAI):
    if search_type == "arxiv_id":
        return call_mcp_tool("get_paper_by_id", {"arxiv_id": search_value}, server_name="openalex")
    if search_type == "title":
        return call_mcp_tool("get_paper_by_title", {"title": search_value}, server_name="openalex")

    search_query = _extract_query(user_query, llm)
    print(f"[Search] OpenAlex 关键词优化: {search_query}")
    return call_mcp_tool("search_papers", {
        "query":       search_query,
        "max_results": 5,
    }, server_name="openalex")


def search_node(state: AgentState) -> dict:
    user_query = state.get("user_query", "")
    llm = _get_llm()

    # ── Step 1: 识别搜索意图（新增）─────────────────────────────
    _tid = state.get("thread_id", "default")
    chat_context = build_context_str(state.get("messages", []))
    intent       = _identify_search_intent(user_query, llm, chat_context)
    search_type  = intent.get("type", "keyword")
    search_value = intent.get("value", user_query)
    push_progress(_tid, f"🔍 意图识别：{search_type} → {search_value}")
    print(f"[Search] 意图: {search_type} → '{search_value}'  ({intent.get('reason','')})")

    # ── Step 2: 搜索策略 ──────────────────────────────────────────
    # - arxiv_id / title（精确查询）：直接走 OpenAlex 精确接口，速度快且稳定
    #   若 OpenAlex 找不到再用 ArXiv MCP 补充（ArXiv MCP 对精确 ID 更权威）
    # - keyword（泛搜）：直接走 OpenAlex，避免 ArXiv MCP 限流带来的等待
    if search_type in {"arxiv_id", "title"}:
        # 精确查询：OpenAlex 优先
        push_progress(_tid, f"📡 精确查询 OpenAlex：{search_value}")
        print(f"[Search] OpenAlex 精确查询: {search_value}")
        raw_results = _call_openalex(search_type, search_value, user_query, llm)
        papers = _parse_results(raw_results)

        if not papers:
            # OpenAlex 找不到时用 ArXiv MCP 补充（对新论文或冷门论文更准）
            push_progress(_tid, "⚡ OpenAlex 未找到，尝试 ArXiv MCP...")
            print(f"[Search] OpenAlex 未找到，回退 ArXiv MCP: {search_value}")
            raw_results = call_mcp_tool("search_papers", {
                "query": search_value, "max_results": 1,
            }, server_name="arxiv")
            if not _has_tool_error(raw_results):
                papers = _parse_results(raw_results)
    else:
        # 泛搜：直接走 OpenAlex，稳定无限流问题
        search_query = _extract_query(user_query, llm)
        push_progress(_tid, f"📡 查询 OpenAlex：{search_query}")
        print(f"[Search] OpenAlex 泛搜: {search_query}")
        raw_results = call_mcp_tool("search_papers", {
            "query": search_query, "max_results": 5,
        }, server_name="openalex")
        papers = _parse_results(raw_results)

# ── Step 3 & 4: 解析结果 + 生成摘要 ──────────────────────

    if not papers:
        return {
            "search_results":   [],
            "papers_to_ingest": [],
            "error":            f"未找到与'{user_query}'相关的论文",
            "search_attempted": True,    # ← 加这一行
            "messages": [AIMessage(content="搜索未找到相关论文，请尝试换个描述方式。")],
        }

    summary_lines = [f"找到 {len(papers)} 篇相关论文：\n"]
    for i, p in enumerate(papers, 1):
        summary_lines.append(
            f"{i}. 【{p.published}】{p.title}\n"
            f"   作者: {', '.join(p.authors)}\n"
            f"   arxiv: {p.arxiv_id or '无'}\n"
        )
    summary = "\n".join(summary_lines)
    push_progress(_tid, f"✅ 搜索完成，找到 {len(papers)} 篇")
    print(f"[Search] {summary}")

    return {
        "search_results":   papers,
        "papers_to_ingest": papers,
        "error":            "",
        "search_attempted": True,    # ← 加这一行
        "messages": [AIMessage(content=summary)],
    }