"""单元测试：Ingest Agent 的失败重试机制（第一次失败 → 第二次成功）。

直接驱动 ingest_node（不经 Supervisor / 网络），用假的 ingest_paper 工具模拟
先失败后成功，验证：
- 第一次失败后，论文进入 failed_papers（可重试，retry_count=1）
- 第二次路由到 ingest 时，能把失败论文捞回来重试并成功
"""

from opendetect_ai.agents import ingest as ingest_mod
from opendetect_ai.state import create_initial_state, PaperMeta


class _FakeTool:
    """模拟 langchain @tool，可按调用序列返回不同结果。"""
    def __init__(self, results):
        self._results = list(results)
        self.calls = 0

    def invoke(self, _payload):
        self.calls += 1
        return self._results.pop(0) if self._results else {"status": "error", "message": "no more"}


def _paper():
    return PaperMeta(title="Paper X", pdf_url="http://x/p.pdf",
                     arxiv_id="1234.5678", retry_count=0)


def test_ingest_retry_first_fails_then_succeeds(monkeypatch) -> None:
    paper = _paper()

    # ── 第一轮：入库失败 ───────────────────────────────────────
    fail_tool = _FakeTool([{"status": "error", "message": "网络超时"}])
    monkeypatch.setattr(ingest_mod, "ingest_paper", fail_tool)

    st = create_initial_state("入库")
    st["papers_to_ingest"] = [paper]
    out1 = ingest_mod.ingest_node(st)

    assert fail_tool.calls == 1
    # 有 arxiv_id 且未超上限 → 应进入可重试列表
    failed = out1["failed_papers"]
    assert len(failed) == 1
    assert failed[0].retry_count == 1
    assert out1["ingested_count"] == 0

    # ── 第二轮：papers_to_ingest 已空，但 failed_papers 带着可重试论文；这次成功 ──
    ok_tool = _FakeTool([{"status": "ok", "chunks": 42}])
    monkeypatch.setattr(ingest_mod, "ingest_paper", ok_tool)

    st2 = create_initial_state("重新入库失败的")
    st2["papers_to_ingest"] = []            # 上一轮的都标记过了
    st2["failed_papers"] = failed           # ← 关键：把失败论文带进来
    out2 = ingest_mod.ingest_node(st2)

    assert ok_tool.calls == 1, "重试应真正再次调用入库（此前的 bug 是根本不会重试）"
    assert out2["ingested_count"] == 1, "第二次应入库成功"


def test_ingest_no_pending_and_no_failed_returns_done() -> None:
    """没有待入库也没有可重试论文时，直接返回已完成。"""
    st = create_initial_state("入库")
    st["papers_to_ingest"] = [PaperMeta(title="done", ingested=True)]
    out = ingest_mod.ingest_node(st)
    assert out["ingested_count"] == 0
    assert "没有待入库" in out["messages"][0].content


def test_confirmation_invalid_selection_fails_closed() -> None:
    """HITL 是审批边界：缺失或畸形选择不能退化成全量批准。"""
    papers = [PaperMeta(title="A"), PaperMeta(title="B")]
    kept = ingest_mod._apply_confirmation(papers, ["bad", 99])
    assert kept == []
    assert all(p.ingested for p in papers)


def test_confirmation_keeps_only_valid_indices() -> None:
    papers = [PaperMeta(title="A"), PaperMeta(title="B")]
    kept = ingest_mod._apply_confirmation(papers, {"selected": ["1", -1, 9]})
    assert [p.title for p in kept] == ["B"]
    assert papers[0].ingested is True
    assert papers[1].ingested is False


def test_ingest_recovers_official_pdf_url_from_arxiv_id(monkeypatch) -> None:
    paper = PaperMeta(title="Paper X", arxiv_id="1234.5678")
    class CaptureTool(_FakeTool):
        def __init__(self):
            super().__init__([{"status": "ok", "chunks": 1}])
            self.payloads = []

        def invoke(self, payload):
            self.payloads.append(payload)
            return super().invoke(payload)

    tool = CaptureTool()
    monkeypatch.setattr(ingest_mod, "ingest_paper", tool)
    state = create_initial_state("入库")
    state["papers_to_ingest"] = [paper]
    out = ingest_mod.ingest_node(state)

    assert out["ingested_count"] == 1
    assert tool.payloads[0]["pdf_url"] == "https://arxiv.org/pdf/1234.5678"
