"""
Clarify 澄清判定 golden set —— OpenDetect_AI（阶段 ⑦，先评测后改 Graph）

覆盖四个判定面（前二含真实模型，后二纯确定性）：
  reference     ：resolve_reference_llm → judge_reference（候选须 grounding）——含真实模型
  title_pool    ：judge_title_pool（绝对下限 + 相对差距，去重，empty/error 区分）
  entity_conflict：judge_entity_conflict
  selection     ：parse_clarification_selection（序号/标题/ID/拒绝/新任务/越界/attempts 护栏）

验收偏精度：重点看「不该澄清用例的误触发率」——宁可少澄清，也别频繁打断。

运行：
    make clarify-eval
    uv run python -m opendetect_ai.eval.clarify_eval
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, AIMessage

from opendetect_ai.agents.clarify import (
    judge_title_pool,
    judge_entity_conflict,
    judge_reference,
    resolve_reference_llm,
    parse_clarification_selection,
)


@dataclass
class Case:
    kind: str                       # reference | title_pool | entity_conflict | selection
    name: str
    expect: Any                     # 判定面里期望的 reason(str)/None，或 selection 的 action
    # reference
    utterance: str = ""
    context: list[str] = field(default_factory=list)
    # title_pool
    query: str = ""
    pool: list[dict] = field(default_factory=list)
    status: str = "ok"
    # entity_conflict
    stated_title: str = ""
    fetched_title: str = ""
    # selection
    reply: str = ""
    pending: dict = field(default_factory=dict)
    expect_option: str = ""


_OPTS = [
    {"id": "1", "label": "BERT: Pre-training of Deep Bidirectional Transformers",
     "arxiv_id": "1810.04805", "resolved_query": "帮我入库 arXiv:1810.04805"},
    {"id": "2", "label": "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
     "arxiv_id": "1907.11692", "resolved_query": "帮我入库 arXiv:1907.11692"},
]
_PENDING1 = {"kind": "clarification", "reason": "multiple_papers", "options": _OPTS, "attempts": 1}
_PENDING2 = {"kind": "clarification", "reason": "multiple_papers", "options": _OPTS, "attempts": 2}


CASES: list[Case] = [
    # ── reference（真实模型 + grounding 校验）──
    Case("reference", "双实体+单数指代→澄清", "ambiguous_reference",
         utterance="它有什么优势？", context=["用户：讲讲 ViT", "助手：ViT 是视觉 Transformer……",
                                          "用户：讲讲 LoRA", "助手：LoRA 是低秩适配……"]),
    Case("reference", "单实体+单数指代→不澄清", None,
         utterance="它有什么优势？", context=["用户：讲讲 ViT", "助手：ViT 是视觉 Transformer……"]),
    Case("reference", "对比语境+单数指代→澄清", "ambiguous_reference",
         utterance="它更好吗？", context=["用户：比较 PPO 和 SAC", "助手：PPO 与 SAC 是两种策略优化算法……"]),
    Case("reference", "指代已自明→不澄清", None,
         utterance="它和 ViT 比呢？", context=["用户：讲讲 Swin Transformer", "助手：Swin 是层级式 ViT……"]),
    Case("reference", "复数指代→不澄清(改写成多对象)", None,
         utterance="它们有什么区别？", context=["用户：比较 PPO 和 SAC", "助手：PPO 与 SAC……"]),

    # ── title_pool（确定性）──
    Case("title_pool", "多篇接近候选→multiple_papers", "multiple_papers", query="BERT", pool=[
        {"title": "BERT: Pre-training of Deep Bidirectional Transformers", "arxiv_id": "1810.04805"},
        {"title": "BERT Rediscovers the Classical NLP Pipeline", "arxiv_id": "1905.05950"},
        {"title": "RoBERTa: A Robustly Optimized BERT Pretraining Approach", "arxiv_id": "1907.11692"},
    ]),
    Case("title_pool", "唯一精确命中→不澄清", None, query="Attention Is All You Need", pool=[
        {"title": "Attention Is All You Need", "arxiv_id": "1706.03762"},
    ]),
    Case("title_pool", "一强其余弱→不澄清(选强)", None, query="BERT", pool=[
        {"title": "BERT: Pre-training of Deep Bidirectional Transformers", "arxiv_id": "1810.04805"},
        {"title": "Deep Residual Learning for Image Recognition", "arxiv_id": "1512.03385"},
        {"title": "An Image is Worth 16x16 Words", "arxiv_id": "2010.11929"},
    ]),
    Case("title_pool", "后端成功返回空→not_found", "exact_title_not_found",
         query="一篇讲量子雷达深度学习XYZ的论文", pool=[], status="empty"),
    Case("title_pool", "同论文多版本/重复→去重后不澄清", None, query="Deep Residual Learning for Image Recognition",
         pool=[
             {"title": "Deep Residual Learning for Image Recognition", "arxiv_id": "1512.03385"},
             {"title": "Deep Residual Learning for Image Recognition", "arxiv_id": "1512.03385v2"},
             {"title": "Deep Residual Learning for Image Recognition", "arxiv_id": "1512.03385"},
         ]),
    Case("title_pool", "后端报错→None(不得伪装not_found)", None, query="BERT", pool=[], status="error"),

    # ── entity_conflict（确定性）──
    Case("entity_conflict", "标题与ID对应论文不一致→冲突", "entity_conflict",
         stated_title="Attention Is All You Need",
         fetched_title="LoRA: Low-Rank Adaptation of Large Language Models"),
    Case("entity_conflict", "标题与ID一致→不澄清", None,
         stated_title="Attention Is All You Need", fetched_title="Attention Is All You Need"),

    # ── selection（确定性）──
    Case("selection", "第2篇→选2", "select", reply="第2篇", pending=_PENDING1, expect_option="2"),
    Case("selection", "2→选2", "select", reply="2", pending=_PENDING1, expect_option="2"),
    Case("selection", "第一个→选1", "select", reply="第一个", pending=_PENDING1, expect_option="1"),
    Case("selection", "标题匹配→选2", "select", reply="RoBERTa 那篇", pending=_PENDING1, expect_option="2"),
    Case("selection", "arXiv ID 匹配→选2", "select", reply="1907.11692 这篇", pending=_PENDING1, expect_option="2"),
    Case("selection", "都不是→清空", "clear", reply="都不是", pending=_PENDING1),
    Case("selection", "算了→清空", "clear", reply="算了", pending=_PENDING1),
    Case("selection", "新任务→清空并转处理", "reprocess", reply="帮我找目标检测的论文", pending=_PENDING1),
    Case("selection", "含糊回复(attempts=1)→再澄清", "reclarify", reply="不确定诶", pending=_PENDING1),
    Case("selection", "含糊回复(attempts=2)→兜底清空", "fallback", reply="不确定诶", pending=_PENDING2),
    Case("selection", "越界序号(仅2项)→再澄清不默认", "reclarify", reply="第9篇", pending=_PENDING1),
]


def _messages(context: list[str]) -> list:
    msgs: list = []
    for line in context:
        if line.startswith("用户："):
            msgs.append(HumanMessage(line[3:]))
        elif line.startswith("助手："):
            msgs.append(AIMessage(line[3:]))
    return msgs


def _run_case(c: Case):
    """返回 (got_reason_or_action, ok)。"""
    if c.kind == "reference":
        msgs = _messages(c.context)
        res = resolve_reference_llm(c.utterance, msgs)
        d = judge_reference(c.utterance, msgs, res)
        got = d.reason if d else None
        return got, got == c.expect
    if c.kind == "title_pool":
        d = judge_title_pool(c.query, c.pool, c.status)
        got = d.reason if d else None
        return got, got == c.expect
    if c.kind == "entity_conflict":
        d = judge_entity_conflict(c.stated_title, c.fetched_title)
        got = d.reason if d else None
        return got, got == c.expect
    # selection
    sel = parse_clarification_selection(c.reply, c.pending)
    ok = sel.action == c.expect and (not c.expect_option or sel.option_id == c.expect_option)
    return f"{sel.action}({sel.option_id})" if sel.option_id else sel.action, ok


def main() -> None:
    print(f"\n{'=' * 76}\nClarify 澄清判定 golden set（{len(CASES)} 条）\n{'=' * 76}")
    should = should_ok = 0          # 应澄清用例（expect 是 reason 字符串）
    shouldnt = false_trigger = 0    # 不该澄清用例（reference/title_pool/entity 里 expect=None）
    sel_total = sel_ok = 0
    fails = []
    for c in CASES:
        got, ok = _run_case(c)
        mark = "✓" if ok else "✗"
        print(f"  {mark} [{c.kind:15}] {c.name:32} 期望={str(c.expect):18} 实得={got}")
        if not ok:
            fails.append(c)
        if c.kind == "selection":
            sel_total += 1
            sel_ok += int(ok)
        elif c.expect is None:
            shouldnt += 1
            if got is not None:      # 误触发：不该澄清却澄清了
                false_trigger += 1
        else:
            should += 1
            should_ok += int(ok)

    print("-" * 76)
    print("汇总（偏精度）：")
    if should:
        print(f"  应澄清命中率     : {should_ok}/{should} = {should_ok / should:.0%}")
    if shouldnt:
        print(f"  不该澄清误触发率 : {false_trigger}/{shouldnt} = {false_trigger / shouldnt:.0%}  ★越低越好")
    if sel_total:
        print(f"  选择解析准确率   : {sel_ok}/{sel_total} = {sel_ok / sel_total:.0%}")
    print(f"  整体通过         : {len(CASES) - len(fails)}/{len(CASES)}")
    if fails:
        print(f"\n失败明细（{len(fails)}）：")
        for c in fails:
            print(f"  ✗ [{c.kind}] {c.name}")
    else:
        print("\n✅ 全部通过——可作为接入 Graph（clarify 节点）前的基线。")
    print("=" * 76)


if __name__ == "__main__":
    main()
