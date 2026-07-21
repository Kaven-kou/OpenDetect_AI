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


# ── Supervisor 模式B 提议搜索时写入 pending_action（承接逻辑已移到 resolve 节点）──
def _patch_supervisor_side_effects(monkeypatch) -> None:
    """屏蔽 supervisor_node 里会触库/触网的副作用，只测路由与 pending_action 写入。"""
    monkeypatch.setattr(sup_mod, "load_user_profile", lambda uid="default": {})
    monkeypatch.setattr(sup_mod, "format_profile_for_prompt", lambda p: "")
    monkeypatch.setattr(sup_mod, "push_progress", lambda *a, **k: None)


def test_supervisor_sets_pending_on_search_offer(monkeypatch) -> None:
    """库空 + 知识问题 → 模式B 提议搜索时，写入 pending_action，供下一轮 resolve 承接。"""
    _patch_supervisor_side_effects(monkeypatch)
    monkeypatch.setattr(
        sup_mod, "_route_with_llm",
        lambda prompt: RouteDecision(
            next="FINISH", reason="库空，提议搜索",
            reply="我的文献库里还没有相关论文。要我去搜索并入库一批相关论文吗？",
        ),
    )
    out = supervisor_node({"user_query": "讲讲 LoRA", "messages": []})
    assert out["next"] == "FINISH"
    assert out["pending_action"] == {
        "kind": "search",
        "query": "帮我搜索并入库与「讲讲 LoRA」相关的论文",
    }


def test_supervisor_no_pending_when_not_offering(monkeypatch) -> None:
    """普通闲聊回复（不含搜索提议）→ 不写 pending_action。"""
    _patch_supervisor_side_effects(monkeypatch)
    monkeypatch.setattr(
        sup_mod, "_route_with_llm",
        lambda prompt: RouteDecision(next="FINISH", reason="打招呼", reply="你好呀！我是 OpenDetect AI"),
    )
    out = supervisor_node({"user_query": "你好", "messages": []})
    assert out["next"] == "FINISH"
    assert "pending_action" not in out


def test_supervisor_reads_resolved_query(monkeypatch) -> None:
    """supervisor 走 effective_query：有 resolved_query 时提议话术里带的是 resolved_query。"""
    _patch_supervisor_side_effects(monkeypatch)
    monkeypatch.setattr(
        sup_mod, "_route_with_llm",
        lambda prompt: RouteDecision(next="FINISH", reason="库空", reply="要我去搜一批相关论文吗？"),
    )
    out = supervisor_node({"user_query": "好啊", "resolved_query": "讲讲 LoRA 低秩适配", "messages": []})
    assert out["pending_action"]["query"] == "帮我搜索并入库与「讲讲 LoRA 低秩适配」相关的论文"

