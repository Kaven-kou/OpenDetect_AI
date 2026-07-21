"""
单元测试：resolve_node —— 上游查询解析（离线，确定性路径不触网）。
重点锁死成本红线：普通问题 / 确认词 → 0 次改写 LLM 调用；只有指代追问才 1 次。
"""

from langchain_core.messages import HumanMessage, AIMessage

from opendetect_ai.agents import resolve as resolve_mod
from opendetect_ai.agents.clarify import ResolveResult
from opendetect_ai.agents.resolve import (
    resolve_node,
    _confirm_verdict,
    _needs_rewrite,
    answer_offers_search,
    make_search_pending,
)


def _spy_rewrite(monkeypatch) -> list:
    """把引用解析 LLM 调用换成计数桩（返回改写、无歧义），避免触网并统计是否走了 LLM 路径。"""
    calls: list = []

    def fake_resolver(raw, msgs):
        calls.append((raw, msgs))
        return ResolveResult(resolved_query=f"REWRITTEN::{raw}", candidates=[])

    monkeypatch.setattr(resolve_mod, "resolve_reference_llm", fake_resolver)
    monkeypatch.setattr(resolve_mod, "judge_reference", lambda raw, msgs, res: None)
    return calls


# ── 确定性判定 ────────────────────────────────────────────────
def test_confirm_verdict() -> None:
    assert _confirm_verdict("好啊") == "affirm"
    assert _confirm_verdict("可以") == "affirm"
    assert _confirm_verdict("嗯，好的") == "affirm"       # 归一化去标点
    assert _confirm_verdict("不用了") == "reject"
    assert _confirm_verdict("算了") == "reject"
    assert _confirm_verdict("帮我找找别的方向") == "other"  # 新任务，不是纯确认


def test_needs_rewrite() -> None:
    assert _needs_rewrite("还有吗")
    assert _needs_rewrite("它和 CNN 比呢")
    assert not _needs_rewrite("PPO 和 SAC 有什么区别")     # 自包含
    assert not _needs_rewrite("讲讲扩散模型")


# ── 成本红线：改写 LLM 调用次数 ────────────────────────────────
def test_selfcontained_question_zero_llm(monkeypatch) -> None:
    """普通自包含问题 → 透传，0 次改写调用。"""
    calls = _spy_rewrite(monkeypatch)
    out = resolve_node({"user_query": "PPO 和 SAC 有什么区别", "messages": []})
    assert out["resolved_query"] == "PPO 和 SAC 有什么区别"
    assert len(calls) == 0


def test_pending_confirm_zero_llm(monkeypatch) -> None:
    """有 pending_action 且用户确认 → 确定性承接，0 次改写调用，并清空 pending。"""
    calls = _spy_rewrite(monkeypatch)
    state = {
        "user_query": "好啊",
        "messages": [],
        "pending_action": {"kind": "search", "query": "帮我搜索并入库与「讲讲 LoRA」相关的论文"},
    }
    out = resolve_node(state)
    assert out["resolved_query"] == "帮我搜索并入库与「讲讲 LoRA」相关的论文"
    assert out["pending_action"] is None
    assert len(calls) == 0


def test_pending_reject_clears_and_zero_llm(monkeypatch) -> None:
    calls = _spy_rewrite(monkeypatch)
    state = {"user_query": "不用了", "messages": [], "pending_action": {"kind": "search", "query": "X"}}
    out = resolve_node(state)
    assert out["resolved_query"] == "不用了"
    assert out["pending_action"] is None
    assert len(calls) == 0


def test_pending_new_task_clears_pending(monkeypatch) -> None:
    """有 pending 但用户给了新任务（other）→ 清空 pending，按普通输入处理。"""
    calls = _spy_rewrite(monkeypatch)
    state = {"user_query": "帮我找目标检测的论文", "messages": [],
             "pending_action": {"kind": "search", "query": "X"}}
    out = resolve_node(state)
    assert out["pending_action"] is None
    assert out["resolved_query"] == "帮我找目标检测的论文"   # 自包含，透传
    assert len(calls) == 0


def test_anaphora_triggers_one_llm(monkeypatch) -> None:
    """指代/追问 + 有上下文 → 恰好 1 次改写调用。"""
    calls = _spy_rewrite(monkeypatch)
    state = {
        "user_query": "还有吗",
        "messages": [HumanMessage("介绍一下 LoRA"), AIMessage("LoRA 是低秩适配……")],
    }
    out = resolve_node(state)
    assert out["resolved_query"] == "REWRITTEN::还有吗"
    assert len(calls) == 1


def test_anaphora_without_context_no_llm(monkeypatch) -> None:
    """指代但没有上下文 → 无从改写，透传，0 次调用。"""
    calls = _spy_rewrite(monkeypatch)
    out = resolve_node({"user_query": "还有吗", "messages": []})
    assert out["resolved_query"] == "还有吗"
    assert len(calls) == 0


# ── offer / pending 辅助 ──────────────────────────────────────
def test_answer_offers_search() -> None:
    assert answer_offers_search("要我去搜索并入库一批相关论文吗？")
    assert answer_offers_search("要我去搜一批吗？")
    assert not answer_offers_search("根据论文，ViT 的优势是……")


def test_make_search_pending() -> None:
    p = make_search_pending("讲讲 LoRA")
    assert p["kind"] == "search"
    assert "讲讲 LoRA" in p["query"]


# ── 澄清选择：resolve 里对 clarification pending 的确定性处理 ────
_CLARIFY_PENDING = {
    "kind": "clarification", "reason": "multiple_papers", "original_query": "找 BERT",
    "question": "哪一篇？", "attempts": 1,
    "options": [
        {"id": "1", "label": "BERT: Pre-training of Deep...", "resolved_query": "帮我入库 arXiv:1810.04805"},
        {"id": "2", "label": "RoBERTa: A Robustly Optimized...", "resolved_query": "帮我入库 arXiv:1907.11692"},
    ],
}


def test_resolve_clarify_select(monkeypatch) -> None:
    _spy_rewrite(monkeypatch)
    out = resolve_node({"user_query": "第2篇", "messages": [], "pending_action": dict(_CLARIFY_PENDING)})
    assert out["resolved_query"] == "帮我入库 arXiv:1907.11692"
    assert out["pending_action"] is None


def test_resolve_clarify_clear_on_reject(monkeypatch) -> None:
    _spy_rewrite(monkeypatch)
    out = resolve_node({"user_query": "都不是", "messages": [], "pending_action": dict(_CLARIFY_PENDING)})
    assert out["pending_action"] is None


def test_resolve_clarify_reclarify_increments(monkeypatch) -> None:
    _spy_rewrite(monkeypatch)
    out = resolve_node({"user_query": "不确定", "messages": [], "pending_action": dict(_CLARIFY_PENDING)})
    assert out["pending_action"]["kind"] == "clarification"
    assert out["pending_action"]["attempts"] == 2       # 再问一次


def test_resolve_clarify_fallback_when_maxed(monkeypatch) -> None:
    _spy_rewrite(monkeypatch)
    p = {**_CLARIFY_PENDING, "attempts": 2}
    out = resolve_node({"user_query": "不确定", "messages": [], "pending_action": p})
    assert out["pending_action"]["reason"] == "fallback"   # 达上限 → 兜底
    assert out["pending_action"]["options"] == []
