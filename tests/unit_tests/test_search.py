"""单元测试：Search 层——确定性 arXiv 解析 + 结构化意图真正被使用（均不触网）。"""

import pytest

from opendetect_ai.agents import search as search_mod
from opendetect_ai.agents.search import _extract_provided_arxiv_id, SearchIntent


# ── ① 确定性 arXiv ID / URL 解析（覆盖新式/版本号/URL/旧式）────────
@pytest.mark.parametrize("text,expect", [
    ("2010.11929", "2010.11929"),
    ("2010.11929v2", "2010.11929"),
    ("https://arxiv.org/abs/2010.11929", "2010.11929"),
    ("https://arxiv.org/pdf/2010.11929.pdf", "2010.11929"),
    ("arXiv:1706.03762", "1706.03762"),
    ("hep-th/9901001", "hep-th/9901001"),
    ("cs.LG/0501001", "cs.LG/0501001"),
    ("帮我找 https://arxiv.org/abs/2103.14030v3 这篇", "2103.14030"),
])
def test_extract_provided_arxiv_id_hits(text, expect) -> None:
    assert _extract_provided_arxiv_id(text) == expect


@pytest.mark.parametrize("text", [
    "讲讲 LoRA 低秩适配",
    "比较 PPO 和 SAC",
    "2020年的目标检测综述",   # 年份不是 arxiv id，不能误判
    "",
])
def test_extract_provided_arxiv_id_no_false_positive(text) -> None:
    assert _extract_provided_arxiv_id(text) == ""


# ── ③ 结构化意图必须真正传给搜索后端 ──────────────────────────
# 回归：keyword 分支曾算出 search_value 却又拿原始 user_query 重抽一遍，
# 导致上下文/指代理解被丢弃。这里锁死「传给 search_papers 的就是 intent.query」。
def _capture_mcp(monkeypatch) -> list:
    calls: list = []

    def fake_mcp(tool, args, server_name=None):
        calls.append((tool, args, server_name))
        return []   # 空结果即可，只关心传入的 query

    monkeypatch.setattr(search_mod, "call_mcp_tool", fake_mcp)
    monkeypatch.setattr(search_mod, "_get_llm", lambda: None)   # 避免构造真实 LLM
    return calls


def test_topic_uses_intent_query_not_raw_input(monkeypatch) -> None:
    calls = _capture_mcp(monkeypatch)
    monkeypatch.setattr(
        search_mod, "_classify_search_intent",
        lambda uq, ctx, llm: SearchIntent(mode="topic", query="LoRA low-rank adaptation"),
    )
    search_mod.search_node({"user_query": "好啊", "messages": []})
    hits = [c for c in calls if c[0] == "search_papers"]
    assert hits, "topic 模式应调用 search_papers"
    assert hits[0][1]["query"] == "LoRA low-rank adaptation"   # 不是 "好啊"


def test_exact_title_uses_intent_query(monkeypatch) -> None:
    calls = _capture_mcp(monkeypatch)
    monkeypatch.setattr(
        search_mod, "_classify_search_intent",
        lambda uq, ctx, llm: SearchIntent(mode="exact_title", query="Attention Is All You Need"),
    )
    search_mod.search_node({"user_query": "找 Attention 那篇", "messages": []})
    hits = [c for c in calls if c[0] == "get_paper_by_title"]
    assert hits and hits[0][1]["title"] == "Attention Is All You Need"


def test_provided_id_skips_llm_classification(monkeypatch) -> None:
    """用户给了 arXiv ID → 走确定性精确检索，绝不调用 LLM 意图分类。"""
    calls = _capture_mcp(monkeypatch)

    def boom(*a, **k):
        raise AssertionError("给了明确 ID 不应再调用 LLM 分类")

    monkeypatch.setattr(search_mod, "_classify_search_intent", boom)
    search_mod.search_node({"user_query": "帮我入库 2010.11929", "messages": []})
    hits = [c for c in calls if c[0] == "get_paper_by_id"]
    assert hits and hits[0][1]["arxiv_id"] == "2010.11929"
