"""
Search 语义链在线评测（golden set）—— OpenDetect_AI

评的是完整语义链 **resolve_query → SearchIntent**（真实配置模型），而非孤立的 SearchIntent：
- 自包含问题：resolve 透传（0 次改写 LLM），等价于单独测 SearchIntent（标准分类子集）。
- 指代/追问：resolve 借上下文改写（1 次 LLM）后再分类——这才是重构前后可比的完整链路。
- 确认式（配 pending_action）：resolve 确定性承接（0 次 LLM）成显式搜索指令，再分类。

和 tests/unit_tests/ 的分工：单测离线锁死「代码正确应用解析结果 + 成本红线」；本 set 在线证明
「模型真的理解」。作为「无评测不重构核心路由」的基线：动核心路由前先建基线，改 prompt 后回归对比。

运行：
    make intent-eval
    uv run python -m opendetect_ai.eval.intent_eval
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, AIMessage

from opendetect_ai.agents import resolve as resolve_mod
from opendetect_ai.agents.resolve import resolve_node
from opendetect_ai.agents.search import (
    _classify_search_intent,
    _extract_provided_arxiv_id,
    _get_llm,
)


@dataclass
class Case:
    utterance: str
    expect_mode: str | None                         # exact_title | topic | None(=正则直达 arXiv ID)
    context: list[str] = field(default_factory=list)  # 交替 用户/助手 文本
    pending: dict | None = None                     # 上一轮系统提议留下的待确认动作
    expect_arxiv_id: str | None = None
    includes: tuple[str, ...] = ()                  # 最终 intent.query 应（忽略大小写）包含
    excludes: tuple[str, ...] = ()                  # 不应包含（如泛词）


# 供确认类用例复用的一条 pending（模拟上一轮「讲讲 LoRA」→ 提议搜索）
_PENDING_LORA = {"kind": "search", "query": "帮我搜索并入库与「讲讲 LoRA 低秩适配」相关的论文"}
_PENDING_VIT = {"kind": "search", "query": "帮我搜索并入库与「视觉 Transformer」相关的论文"}


CASES: list[Case] = [
    # ── arXiv ID / URL：确定性，应正则直达，不进分类 ──
    Case("帮我入库 2010.11929", None, expect_arxiv_id="2010.11929"),
    Case("https://arxiv.org/abs/1706.03762 这篇", None, expect_arxiv_id="1706.03762"),
    Case("找 https://arxiv.org/pdf/2103.14030v2.pdf", None, expect_arxiv_id="2103.14030"),
    Case("arXiv:2005.14165 讲讲", None, expect_arxiv_id="2005.14165"),
    Case("hep-th/9901001", None, expect_arxiv_id="hep-th/9901001"),

    # ── exact_title（自包含，resolve 透传 → 标准分类子集）──
    Case("找 Attention Is All You Need 这篇", "exact_title", includes=("attention",)),
    Case("首次提出 ViT 的那篇原版论文", "exact_title", includes=("image",)),
    Case("何恺明的 ResNet 那篇", "exact_title", includes=("residual",)),
    Case("BERT 原论文", "exact_title", includes=("bert",)),
    Case("帮我找 CLIP 那篇原版", "exact_title", excludes=("deep learning",)),

    # ── topic（自包含，resolve 透传 → 标准分类子集）──
    Case("比较强化学习里的策略优化方法", "topic", includes=("policy",), excludes=("deep learning",)),
    Case("讲讲扩散模型", "topic", includes=("diffusion",)),
    Case("目标检测最近有哪些新方法", "topic", includes=("detection",)),
    Case("PPO 和 SAC 有什么区别", "topic", excludes=("deep learning",)),
    Case("找几篇对比学习的论文", "topic", includes=("contrastive",)),

    # ── 指代 / 追问（resolve 借上下文改写 → 完整链路，各 1 次 LLM）──
    Case("还有吗", "topic",
         context=["用户：介绍一下 LoRA", "助手：LoRA 是一种低秩适配微调方法……"],
         includes=("lora",), excludes=("deep learning",)),
    Case("其他的呢", "topic",
         context=["用户：找几篇实例分割的论文", "助手：已找到 Mask R-CNN……"],
         includes=("segmentation",), excludes=("deep learning",)),
    Case("多找几篇这个方向的", "topic",
         context=["用户：讲讲知识蒸馏 knowledge distillation", "助手：知识蒸馏是……"],
         includes=("distillation",)),

    # ── 确认式（配 pending_action，resolve 确定性承接 → 0 次 LLM）──
    Case("好啊", "topic", pending=_PENDING_LORA, includes=("lora",)),
    Case("可以", "topic", pending=_PENDING_VIT, includes=("transformer",)),

    # ── 跨学科歧义：应补 AI 语境 ──
    Case("讲讲 diffusion model", "topic", includes=("diffusion",)),
]


def _messages(context: list[str]) -> list:
    msgs: list = []
    for line in context:
        if line.startswith("用户："):
            msgs.append(HumanMessage(line[3:]))
        elif line.startswith("助手："):
            msgs.append(AIMessage(line[3:]))
    return msgs


def _fmt(s: str, width: int) -> str:
    """按显示宽度（中文算 2）左对齐截断。"""
    disp = sum(2 if ord(ch) > 0x2E7F else 1 for ch in s)
    if disp <= width:
        return s + " " * (width - disp)
    out, w = "", 0
    for ch in s:
        cw = 2 if ord(ch) > 0x2E7F else 1
        if w + cw > width - 1:
            break
        out += ch
        w += cw
    return out + "…" + " " * (width - w - 1)


def main() -> None:
    llm = _get_llm()
    resolve_mod._llm_call_count = 0
    id_total = id_ok = 0
    mode_total = mode_ok = 0
    kw_total = kw_ok = 0
    failures: list[tuple[Case, str]] = []

    print(f"\n{'=' * 82}")
    print("Search 语义链 golden set（resolve → SearchIntent，真实配置模型）")
    print(f"{'=' * 82}")
    print(f"{_fmt('utterance', 30)}{_fmt('期望', 12)}{_fmt('resolved→query', 36)}结果")
    print("-" * 82)

    for c in CASES:
        provided = _extract_provided_arxiv_id(c.utterance)

        # 确定性 arXiv 分支（不进语义链）
        if c.expect_arxiv_id is not None:
            id_total += 1
            ok = provided == c.expect_arxiv_id
            id_ok += int(ok)
            print(f"{_fmt(c.utterance, 30)}{_fmt('id:' + c.expect_arxiv_id, 12)}"
                  f"{_fmt('id:' + (provided or '∅'), 36)}{'✓' if ok else '✗'}")
            if not ok:
                failures.append((c, f"arXiv 解析：期望 {c.expect_arxiv_id}，实得 {provided or '∅'}"))
            continue

        false_id = bool(provided)   # 非 ID 用例，正则不应误触发

        # 完整语义链：resolve → SearchIntent
        state = {"user_query": c.utterance, "messages": _messages(c.context), "pending_action": c.pending}
        resolved = resolve_node(state).get("resolved_query") or c.utterance
        intent = _classify_search_intent(resolved, llm)

        mode_total += 1
        mode_hit = intent.mode == c.expect_mode
        mode_ok += int(mode_hit)

        q = intent.query.lower()
        kw_hit = all(k.lower() in q for k in c.includes) and all(k.lower() not in q for k in c.excludes)
        if c.includes or c.excludes:
            kw_total += 1
            kw_ok += int(kw_hit)

        case_ok = mode_hit and kw_hit and not false_id
        print(f"{_fmt(c.utterance, 30)}{_fmt(c.expect_mode or '-', 12)}"
              f"{_fmt(f'{intent.mode}:{intent.query}', 36)}{'✓' if case_ok else '✗'}")
        if not case_ok:
            det = []
            if false_id:
                det.append(f"正则误判出 ID {provided}")
            if not mode_hit:
                det.append(f"mode 期望 {c.expect_mode} 实得 {intent.mode}")
            if not kw_hit:
                det.append(f"query={intent.query!r} 不满足 includes={c.includes} excludes={c.excludes}")
            failures.append((c, "；".join(det)))

    print("-" * 82)
    print("汇总：")
    if id_total:
        print(f"  arXiv 解析准确率 : {id_ok}/{id_total} = {id_ok / id_total:.0%}")
    if mode_total:
        print(f"  mode 分类准确率  : {mode_ok}/{mode_total} = {mode_ok / mode_total:.0%}")
    if kw_total:
        print(f"  query 关键词准确率: {kw_ok}/{kw_total} = {kw_ok / kw_total:.0%}（含指代消解 / 消歧）")
    print(f"  整体通过         : {len(CASES) - len(failures)}/{len(CASES)}")
    print(f"  resolve 改写 LLM 调用: {resolve_mod._llm_call_count} 次"
          f"（应≈指代类用例数；自包含/确认类为 0）")

    if failures:
        print(f"\n失败明细（{len(failures)} 条，边界案例回归看这里）：")
        for c, why in failures:
            print(f"  ✗ {c.utterance!r}  —  {why}")
    else:
        print("\n✅ 全部通过。可作为进入阶段 2 后续（clarify / TaskSpec）的基线。")
    print("=" * 82)


if __name__ == "__main__":
    main()
