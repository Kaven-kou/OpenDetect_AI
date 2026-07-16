"""单元测试：Search 后确定性边界 + Verifier 校验（不触网的分支）。"""

from opendetect_ai.graph import route_after_search, build_graph
from opendetect_ai.state import PaperMeta
from opendetect_ai.agents import verify as verify_mod
from opendetect_ai.agents import supervisor as sup_mod
from opendetect_ai.agents.supervisor import RouteDecision, supervisor_node
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


# ── Supervisor 承接上一轮意图：query rewriting ────────────────────
def _patch_supervisor_side_effects(monkeypatch) -> None:
    """屏蔽 supervisor_node 里会触库/触网的副作用，只测路由与改写逻辑。"""
    monkeypatch.setattr(sup_mod, "load_user_profile", lambda uid="default": {})
    monkeypatch.setattr(sup_mod, "format_profile_for_prompt", lambda p: "")
    monkeypatch.setattr(sup_mod, "push_progress", lambda *a, **k: None)


def test_supervisor_applies_rewritten_query(monkeypatch) -> None:
    """
    用户对上一轮"要我去搜索吗？"回复"好啊" → supervisor 用改写后的完整问题
    替换本轮无主题的 user_query，供下游 search/rag 检索（多轮意图承接的核心）。
    """
    _patch_supervisor_side_effects(monkeypatch)
    monkeypatch.setattr(
        sup_mod, "_route_with_llm",
        lambda prompt: RouteDecision(
            next="search", reason="确认搜索", reply="",
            rewritten_query="讲讲 LoRA 低秩适配",
        ),
    )
    out = supervisor_node({"user_query": "好啊", "messages": []})
    assert out["next"] == "search"
    assert out["user_query"] == "讲讲 LoRA 低秩适配"


def test_supervisor_no_rewrite_keeps_original_query(monkeypatch) -> None:
    """未改写（rewritten_query 为空）时，绝不覆盖 user_query。"""
    _patch_supervisor_side_effects(monkeypatch)
    monkeypatch.setattr(
        sup_mod, "_route_with_llm",
        lambda prompt: RouteDecision(next="rag", reason="知识问题", reply=""),
    )
    out = supervisor_node({"user_query": "讲讲LoRA", "messages": []})
    assert out["next"] == "rag"
    assert "user_query" not in out   # 未改写不写回，避免误覆盖


def test_supervisor_rewrite_ignored_when_finish(monkeypatch) -> None:
    """即便模型误填了 rewritten_query，路由到 FINISH 时也不应改写 user_query。"""
    _patch_supervisor_side_effects(monkeypatch)
    monkeypatch.setattr(
        sup_mod, "_route_with_llm",
        lambda prompt: RouteDecision(
            next="FINISH", reason="闲聊", reply="你好",
            rewritten_query="不该生效的改写",
        ),
    )
    out = supervisor_node({"user_query": "你好", "messages": []})
    assert out["next"] == "FINISH"
    assert "user_query" not in out
