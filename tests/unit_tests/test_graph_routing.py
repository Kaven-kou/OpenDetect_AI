"""单元测试：Search 后确定性边界 + Verifier 校验（不触网的分支）。"""

from opendetect_ai.graph import route_after_search, build_graph
from opendetect_ai.state import PaperMeta
from opendetect_ai.agents import verify as verify_mod
from langgraph.pregel import Pregel


# ── 确定性边界 ────────────────────────────────────────────────
def test_route_after_search_to_ingest_when_pending() -> None:
    st = {"papers_to_ingest": [PaperMeta(title="A", ingested=False)]}
    assert route_after_search(st) == "ingest"


def test_route_after_search_to_supervisor_when_empty() -> None:
    assert route_after_search({"papers_to_ingest": []}) == "supervisor"


def test_route_after_search_to_supervisor_when_all_ingested() -> None:
    st = {"papers_to_ingest": [PaperMeta(title="A", ingested=True)]}
    assert route_after_search(st) == "supervisor"


def test_graph_has_verify_node() -> None:
    graph = build_graph()
    assert isinstance(graph, Pregel)
    assert "verify" in graph.nodes


# ── Verifier 确定性分支 ───────────────────────────────────────
def test_verify_noop_when_disabled(monkeypatch) -> None:
    monkeypatch.setattr(verify_mod, "OPENDETECT_VERIFY", False)
    out = verify_mod.verify_node({"rag_answer": "任意", "rag_context": []})
    assert out == {}


def test_verify_skips_empty_answer(monkeypatch) -> None:
    monkeypatch.setattr(verify_mod, "OPENDETECT_VERIFY", True)
    assert verify_mod.verify_node({"rag_answer": "", "rag_context": []}) == {}


def test_verify_skips_when_context_is_error(monkeypatch) -> None:
    monkeypatch.setattr(verify_mod, "OPENDETECT_VERIFY", True)
    st = {"rag_answer": "答案", "rag_context": [{"error": "向量库为空"}]}
    assert verify_mod.verify_node(st) == {}


def test_verify_adds_caveat_when_no_context(monkeypatch) -> None:
    """有回答但完全没有检索片段 → 附上「缺乏来源」警示（无需 LLM）。"""
    monkeypatch.setattr(verify_mod, "OPENDETECT_VERIFY", True)
    st = {"rag_answer": "ViT 很强", "rag_context": []}
    out = verify_mod.verify_node(st)
    assert "核验提示" in out["rag_answer"]
    assert out["rag_answer"].startswith("ViT 很强")
