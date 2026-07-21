"""
Search Agent —— OpenDetect_AI
负责根据用户需求搜索学术论文，结果写入 state.search_results。

语义理解分三段（rule / LLM 各归其位）：
  ① 确定性解析：用户明确给的 arXiv ID / 链接用正则识别，不进 LLM、零幻觉；
  ② LLM 结构化判断：只判断「精确标题 vs 话题关键词」这件需要判断的事，输出 SearchIntent；
  ③ 真正使用解析结果：泛搜用 intent.query（已含上下文/指代消解），而非拿原始输入再抽一遍。
ID 的事实来源是搜索后端（OpenAlex/arXiv），不是模型的参数记忆——所以不让模型回忆 arxiv ID。
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

from opendetect_ai.state import AgentState, PaperMeta, effective_query
from opendetect_ai.tools.progress import push_progress
from opendetect_ai.tools.mcp_client import call_mcp_tool
from opendetect_ai.agents.clarify import (
    judge_title_pool,
    judge_entity_conflict,
    build_clarify_pending,
    _title_score,
)
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
        temperature=0,                  # 意图识别需要确定性
    )


# ── ① 确定性解析：只从用户明确给出的 ID / 链接里提取 arXiv ID ─────
# 新式 ID：YYMM.number（月份 01-12），可带版本号 v2；
# 旧式 ID：category/7位数字，如 hep-th/9901001、cs.LG/0501001；
# URL：arxiv.org/abs|pdf/<id>。识别不到返回 ''，绝不从论文名称推断 ID。
_ARXIV_NEW = re.compile(r"\b\d{2}(?:0[1-9]|1[0-2])\.\d{4,5}(?:v\d+)?\b")
_ARXIV_OLD = re.compile(r"\b[a-z-]{2,}(?:\.[A-Za-z]{2})?/\d{7}(?:v\d+)?\b")
_ARXIV_URL = re.compile(r"arxiv\.org/(?:abs|pdf)/([^\s?#]+)", re.IGNORECASE)


def _extract_provided_arxiv_id(text: str) -> str:
    """
    只识别「用户明确提供」的 arXiv ID / 链接，返回规范化后的 ID；识别不到返回 ''。
    正则只做确定性识别，不承担「从标题猜 ID」——那交给搜索后端（API 才是 ID 的事实来源）。
    """
    if not text:
        return ""

    # 1) URL 形式（abs / pdf）优先
    m = _ARXIV_URL.search(text)
    if m:
        cand = m.group(1).split("?")[0].split("#")[0].replace(".pdf", "")
        return _normalize_arxiv_id(cand)

    # 2) 显式 "arXiv:xxxx" 前缀
    m = re.search(r"arxiv:\s*([^\s]+)", text, re.IGNORECASE)
    if m:
        cand = m.group(1).rstrip(".,;)")
        if _ARXIV_NEW.fullmatch(cand) or _ARXIV_OLD.fullmatch(cand):
            return _normalize_arxiv_id(cand)

    # 3) 裸的新式 / 旧式 ID
    m = _ARXIV_NEW.search(text)
    if m:
        return _normalize_arxiv_id(m.group(0))
    m = _ARXIV_OLD.search(text)
    if m:
        return _normalize_arxiv_id(m.group(0))
    return ""


# ── ③ 结构化搜索意图：只判断「精确标题 vs 话题关键词」这一件需要判断的事 ──
class SearchIntent(BaseModel):
    """搜索意图。用 with_structured_output 约束，杜绝手撕 JSON。不含 arxiv_id（由正则前置分流）。"""
    mode: Literal["exact_title", "topic"] = Field(
        description="exact_title=用户点名了某一篇具体论文；topic=想找某方向的多篇论文或用指代/追问延续话题"
    )
    query: str = Field(
        description="传给搜索后端的英文检索串：exact_title 填规范英文标题；topic 填英文关键词短语（≤8词，追问时须从上下文取当前话题）"
    )
    reason: str = Field(default="", description="一句话说明判断依据")


def _classify_search_intent(user_query: str, llm: ChatOpenAI) -> SearchIntent:
    """
    判断用户想「精确找某篇论文」还是「找某方向的多篇」，并给出要传给搜索后端的英文检索串。
    输入已是上游 resolve 出的**自包含** query，因此这里**不再读对话历史做指代消解**——
    只做「标题 vs 话题」这一件需要判断的事。不再让模型回忆 arxiv ID（由正则前置、后端校验）。
    失败时 fail-open 到 topic。
    """
    instructions = (
        "你是深度学习论文检索助手。判断用户的搜索意图，输出结构化结果。\n\n"
        "## 两种模式\n"
        "- exact_title：用户点名了某一篇具体论文（给了较完整标题，或\"XX 原版/首次提出 XX 的那篇\"这类唯一指向）\n"
        "  → query 填该论文最规范的英文标题，如 Attention Is All You Need\n"
        "- topic：用户想找某方向的多篇论文，或用追问/指代延续上一个话题\n"
        "  → query 填英文关键词短语（≤8 词）；跨学科歧义词补 AI 语境，"
        "如 diffusion model → denoising diffusion probabilistic models image generation\n\n"
        "## 约束\n"
        "- 不要输出 arxiv ID（用户给的 ID/链接已由上游正则识别，不会走到这里）\n"
        "- query 一律英文；只依据用户明确表达判断，别脑补\n"
    )

    prompt = instructions + "\n## 用户输入\n" + user_query

    try:
        result = llm.with_structured_output(
            SearchIntent, method="function_calling"
        ).invoke([HumanMessage(content=prompt)])
        if isinstance(result, SearchIntent):
            return result
        if isinstance(result, dict):
            return SearchIntent(**result)
    except Exception as exc:
        print(f"[Search] 结构化意图解析失败，回退 topic: {exc}")
    # fail-open：默认按话题搜，query 用原始输入（上游可能已把它改写成自包含 query）
    return SearchIntent(mode="topic", query=user_query, reason="解析失败，回退原始输入")


def _parse_results(raw_results: list[dict] | dict | str) -> list[PaperMeta]:
    """把搜索工具返回值统一转成 PaperMeta 列表。"""
    if isinstance(raw_results, str):
        raw_results = _parse_arxiv_text_results(raw_results)

    if isinstance(raw_results, dict) and isinstance(raw_results.get("papers"), list):
        raw_results = raw_results["papers"]
    elif isinstance(raw_results, dict) and isinstance(raw_results.get("results"), list):
        raw_results = raw_results["results"]
    elif isinstance(raw_results, dict):   # 兼容单篇返回
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


# 从「同时给了标题 + arXiv ID」的输入里剥出用户所述标题，用于 entity_conflict 判定
_CMD_WORDS = (
    "帮我", "请", "麻烦", "入库", "收录", "下载", "这篇", "那篇", "这个", "那个",
    "论文", "文章", "搜索", "搜一下", "找一下", "找", "一下", "把", "讲讲", "看看",
)


def _extract_stated_title(text: str, provided_id: str) -> str:
    """去掉 arXiv ID / URL / 指令词后，剩下的当作用户所述标题；太短则返回 ''（视为没给标题）。"""
    t = re.sub(r"https?://arxiv\.org/\S+", " ", text or "", flags=re.IGNORECASE)
    t = re.sub(r"arxiv:\s*\S+", " ", t, flags=re.IGNORECASE)
    t = t.replace(provided_id, " ")
    for w in _CMD_WORDS:
        t = t.replace(w, " ")
    t = re.sub(r"[\s，。、：:；;的]+", " ", t).strip()
    return t if len(t) >= 4 else ""


def _clarify_search_return(pending: dict) -> dict:
    """search 侧触发澄清：写入 pending_action、不产出待入库论文，交 clarify 节点渲染问题。"""
    return {
        "search_results":   [],
        "papers_to_ingest": [],
        "pending_action":   pending,
        "search_attempted": True,
        "error":            "",
    }


def _title_candidate_pool(query: str, _tid: str) -> tuple[list[PaperMeta], str]:
    """
    精确标题候选池：OpenAlex 取 5 篇；成功返回空时再问 ArXiv MCP。
    返回 (papers, status)，status ∈ {ok, empty, error}——供 judge_title_pool 区分
    「后端报错」与「成功返回空」（后者才可能是 exact_title_not_found）。
    """
    raw = call_mcp_tool("search_papers", {"query": query, "max_results": 5}, server_name="openalex")
    if _has_tool_error(raw):
        return [], "error"
    papers = _parse_results(raw)
    if papers:
        return papers, "ok"
    # OpenAlex 成功返回空 → 再问 ArXiv
    push_progress(_tid, "⚡ OpenAlex 无结果，尝试 ArXiv MCP...")
    raw2 = call_mcp_tool("search_papers", {"query": query, "max_results": 5}, server_name="arxiv")
    if not _has_tool_error(raw2):
        papers = _parse_results(raw2)
    return papers, ("ok" if papers else "empty")


def search_node(state: AgentState) -> dict:
    user_query = effective_query(state)   # 用上游 resolve 出的自包含 query
    _tid = state.get("thread_id", "default")
    llm = _get_llm()

    # ── Step 1: 确定性解析——用户明确给了 arXiv ID / 链接就直接精确检索（不进 LLM）──
    provided_id = _extract_provided_arxiv_id(user_query)
    if provided_id:
        push_progress(_tid, f"🔗 识别到 arXiv ID：{provided_id}")
        print(f"[Search] 正则命中 arXiv ID: {provided_id}")
        raw_results = call_mcp_tool(
            "get_paper_by_id", {"arxiv_id": provided_id}, server_name="openalex"
        )
        papers = _parse_results(raw_results)
        # entity_conflict：用户同时给了标题，且与按 ID 取回的标题明显不一致 → 澄清
        stated = _extract_stated_title(user_query, provided_id)
        if stated and papers:
            conflict = judge_entity_conflict(stated, papers[0].title)
            if conflict:
                push_progress(_tid, "❓ 标题与 arXiv ID 不一致，需澄清")
                print(f"[Search] entity_conflict: 「{stated}」 vs 「{papers[0].title}」")
                return _clarify_search_return(build_clarify_pending(conflict, user_query))
        if not papers:
            push_progress(_tid, "⚡ OpenAlex 未找到，尝试 ArXiv MCP...")
            raw_results = call_mcp_tool(
                "search_papers", {"query": provided_id, "max_results": 1}, server_name="arxiv"
            )
            if not _has_tool_error(raw_results):
                papers = _parse_results(raw_results)
    else:
        # ── Step 2: LLM 只判断「精确标题 vs 话题」，并给出检索串（不再回忆 ID / 不读历史）──
        intent = _classify_search_intent(user_query, llm)
        push_progress(_tid, f"🔍 意图识别：{intent.mode} → {intent.query}")
        print(f"[Search] 意图: {intent.mode} → '{intent.query}'  ({intent.reason})")

        if intent.mode == "exact_title":
            # 取候选池 → 判定：多个接近候选(multiple_papers) / 两后端皆空(not_found) → 澄清
            push_progress(_tid, f"📡 精确标题候选池 OpenAlex：{intent.query}")
            papers, status = _title_candidate_pool(intent.query, _tid)
            pool = [{"title": p.title, "arxiv_id": p.arxiv_id} for p in papers]
            decision = judge_title_pool(intent.query, pool, status)
            if decision:
                push_progress(_tid, f"❓ 精确标题需澄清：{decision.reason}")
                print(f"[Search] 精确标题澄清: {decision.reason}")
                return _clarify_search_return(build_clarify_pending(decision, intent.query))
            # 无需澄清：从候选池里挑最匹配的一篇入库
            if papers:
                papers = [max(papers, key=lambda p: _title_score(intent.query, p.title))]
        else:
            # ★ 关键修复：泛搜用结构化解析出的 intent.query（已含上下文/指代消解），
            #   而非拿原始 user_query 再抽一次——否则前一步对上下文的理解会被完全丢弃。
            push_progress(_tid, f"📡 查询 OpenAlex：{intent.query}")
            print(f"[Search] OpenAlex 泛搜: {intent.query}")
            raw_results = call_mcp_tool(
                "search_papers", {"query": intent.query, "max_results": 5}, server_name="openalex"
            )
            papers = _parse_results(raw_results)

    # ── Step 3 & 4: 解析结果 + 生成摘要 ──────────────────────
    if not papers:
        return {
            "search_results":   [],
            "papers_to_ingest": [],
            "error":            f"未找到与'{user_query}'相关的论文",
            "search_attempted": True,
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
        "search_attempted": True,
        "messages": [AIMessage(content=summary)],
    }
