"""单元测试：Search 后确定性边界 + Verifier 校验（不触网的分支）。"""

from opendetect_ai.graph import route_after_search, build_graph
from opendetect_ai.state import PaperMeta
from opendetect_ai.agents import verify as verify_mod
from opendetect_ai.agents import supervisor as sup_mod
from opendetect_ai.agents.supervisor import RouteDecision, supervisor_node
from langgraph.pregel import Pregel
from langchain_core.messages import AIMessage


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
    assert any(
        edge.source == "report" and edge.target == "verify"
        for edge in graph.get_graph().edges
    )


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


def test_verify_refuses_when_no_context(monkeypatch) -> None:
    """有回答但完全没有检索片段 → 拒答并生成可承接的搜索动作。"""
    monkeypatch.setattr(verify_mod, "OPENDETECT_VERIFY", True)
    st = {"user_query": "ViT 为什么有效", "rag_answer": "ViT 很强", "rag_context": []}
    out = verify_mod.verify_node(st)
    assert "足够的检索证据" in out["rag_answer"]
    assert out["verification"]["status"] == "insufficient_context"
    assert out["pending_action"]["kind"] == "search"


def test_citation_report_rejects_title_not_in_retrieval() -> None:
    chunks = [{"title": "Attention Is All You Need"}]
    report = verify_mod._citation_report(
        "结论。（来源：Imaginary Paper，第 3 页）", chunks
    )
    assert report["invalid_citations"] == ["Imaginary Paper，第 3 页"]


def test_verify_passes_grounded_answer_with_valid_citation(monkeypatch) -> None:
    class FakeRunnable:
        def invoke(self, _messages):
            return verify_mod._Verdict(
                grounded=True,
                sufficient_context=True,
                confidence="high",
                unsupported_claims=[],
                claim_evidence=[verify_mod._ClaimEvidence(
                    claim="Transformer 使用自注意力", evidence_ids=["E1"], supported=True,
                )],
            )

    class FakeLLM:
        def with_structured_output(self, *_args, **_kwargs):
            return FakeRunnable()

    monkeypatch.setattr(verify_mod, "OPENDETECT_VERIFY", True)
    monkeypatch.setattr(verify_mod, "_get_llm", lambda: FakeLLM())
    state = {
        "rag_answer": "Transformer 使用自注意力。（来源：Attention Is All You Need，第 2 页）",
        "rag_context": [{"title": "Attention Is All You Need", "content": "self-attention"}],
    }
    out = verify_mod.verify_node(state)
    assert out["verification"]["status"] == "passed"
    assert out["verification"]["confidence"] == "high"


def test_verify_refuses_when_llm_says_context_insufficient(monkeypatch) -> None:
    class FakeRunnable:
        def invoke(self, _messages):
            return verify_mod._Verdict(
                grounded=False,
                sufficient_context=False,
                confidence="low",
                unsupported_claims=["缺少实验数据"],
                claim_evidence=[verify_mod._ClaimEvidence(
                    claim="提升 20%", evidence_ids=[], supported=False,
                )],
            )

    class FakeLLM:
        def with_structured_output(self, *_args, **_kwargs):
            return FakeRunnable()

    monkeypatch.setattr(verify_mod, "OPENDETECT_VERIFY", True)
    monkeypatch.setattr(verify_mod, "_get_llm", lambda: FakeLLM())
    state = {
        "user_query": "具体提升多少",
        "rag_answer": "提升 20%。（来源：Paper A）",
        "rag_context": [{"title": "Paper A", "content": "method description"}],
    }
    out = verify_mod.verify_node(state)
    assert "不会用模型自身知识补写" in out["rag_answer"]
    assert out["verification"]["status"] == "insufficient_context"


def test_verify_marks_answer_when_verifier_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(verify_mod, "OPENDETECT_VERIFY", True)
    monkeypatch.setattr(
        verify_mod,
        "_get_llm",
        lambda: (_ for _ in ()).throw(RuntimeError("provider timeout")),
    )
    state = {
        "rag_answer": "结论。（来源：Paper A，第 1 页）",
        "rag_context": [{"title": "Paper A", "content": "evidence"}],
    }

    out = verify_mod.verify_node(state)
    assert "自动事实核验暂不可用" in out["rag_answer"]
    assert out["verification"]["status"] == "unavailable"
    assert out["verification"]["grounded"] is None


def test_verify_receives_query_and_full_evidence_and_replaces_draft(monkeypatch) -> None:
    captured = {}

    class FakeRunnable:
        def invoke(self, messages):
            captured["prompt"] = messages[0].content
            return verify_mod._Verdict(
                grounded=False,
                sufficient_context=False,
                confidence="low",
                unsupported_claims=["尾部结论"],
                claim_evidence=[verify_mod._ClaimEvidence(
                    claim="尾部结论", evidence_ids=[], supported=False,
                )],
            )

    class FakeLLM:
        def with_structured_output(self, *_args, **_kwargs):
            return FakeRunnable()

    monkeypatch.setattr(verify_mod, "OPENDETECT_VERIFY", True)
    monkeypatch.setattr(verify_mod, "_get_llm", lambda: FakeLLM())
    draft = "A" * 2100 + "尾部结论。（来源：Paper A，第 1 页）"
    evidence = "B" * 500 + "尾部证据"
    state = {
        "user_query": "完整问题是什么？",
        "rag_answer": draft,
        "rag_context": [{"title": "Paper A", "page": 1, "content": evidence}],
        "messages": [AIMessage(content=draft, id="answer-1")],
    }

    out = verify_mod.verify_node(state)

    assert "完整问题是什么？" in captured["prompt"]
    assert "尾部证据" in captured["prompt"]
    assert "尾部结论" in captured["prompt"]
    assert out["messages"][0].id == "answer-1"
    assert out["messages"][0].content == out["rag_answer"]


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
