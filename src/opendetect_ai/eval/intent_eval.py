"""
Search 意图理解在线评测（golden set）—— OpenDetect_AI

和 tests/unit_tests/test_search.py 的分工：
- 单元测试（离线、mock 模型）：证明「代码会正确应用解析结果」——正则、Schema、分支、
  fallback、最终工具参数。确定性、进 `make test`。
- 本 golden set（在线、真实模型）：证明「模型真的理解」(utterance, context) -> SearchIntent，
  覆盖 标题 / 话题 / 指代追问 / 确认 / arXiv-ID 等边界。非确定性，手动 / make intent-eval 跑。

这是「无评测不重构核心路由」的地基：阶段 2（查询改写收归上游 / 统一 TaskSpec）动核心路由前，
先用它建立基线；之后每次改分类 prompt，都能立刻看出 mode / 指代消解准确率有没有掉。

运行：
    make intent-eval
    uv run python -m opendetect_ai.eval.intent_eval
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
    expect_arxiv_id: str | None = None
    includes: tuple[str, ...] = ()                  # query 应（忽略大小写）包含
    excludes: tuple[str, ...] = ()                  # query 不应包含（如泛词）


CASES: list[Case] = [
    # ── arXiv ID / URL：确定性，应正则直达，不进分类 ──
    Case("帮我入库 2010.11929", None, expect_arxiv_id="2010.11929"),
    Case("https://arxiv.org/abs/1706.03762 这篇", None, expect_arxiv_id="1706.03762"),
    Case("找 https://arxiv.org/pdf/2103.14030v2.pdf", None, expect_arxiv_id="2103.14030"),
    Case("arXiv:2005.14165 讲讲", None, expect_arxiv_id="2005.14165"),
    Case("hep-th/9901001", None, expect_arxiv_id="hep-th/9901001"),

    # ── exact_title：点名某一篇 ──
    Case("找 Attention Is All You Need 这篇", "exact_title", includes=("attention",)),
    Case("首次提出 ViT 的那篇原版论文", "exact_title", includes=("image",)),
    Case("何恺明的 ResNet 那篇", "exact_title", includes=("residual",)),
    Case("BERT 原论文", "exact_title", includes=("bert",)),
    Case("帮我找 CLIP 那篇原版", "exact_title", excludes=("deep learning",)),

    # ── topic：找方向 / 多篇 ──
    Case("比较强化学习里的策略优化方法", "topic", includes=("policy",), excludes=("deep learning",)),
    Case("讲讲扩散模型", "topic", includes=("diffusion",)),
    Case("目标检测最近有哪些新方法", "topic", includes=("detection",)),
    Case("PPO 和 SAC 有什么区别", "topic", excludes=("deep learning",)),
    Case("找几篇对比学习的论文", "topic", includes=("contrastive",)),

    # ── 指代 / 追问：必须从上下文取话题，不能退化为泛词 ──
    Case("还有吗", "topic",
         context=["用户：介绍一下 LoRA", "助手：LoRA 是一种低秩适配微调方法……"],
         includes=("lora",), excludes=("deep learning",)),
    Case("其他的呢", "topic",
         context=["用户：找几篇实例分割的论文", "助手：已找到 Mask R-CNN……"],
         includes=("segmentation",), excludes=("deep learning",)),
    Case("多找几篇这个方向的", "topic",
         context=["用户：讲讲知识蒸馏 knowledge distillation", "助手：知识蒸馏是……"],
         includes=("distillation",)),

    # ── 确认式（上游正常会改写；这里测分类器兜底能否从上下文取题）──
    Case("好啊", "topic",
         context=["用户：讲讲 LoRA", "助手：库里还没有，要我去搜索并入库一批吗？"],
         includes=("lora",)),
    Case("可以", "topic",
         context=["用户：找找视觉 Transformer 的论文", "助手：要我去搜吗？"],
         includes=("transformer",)),

    # ── 跨学科歧义：应补 AI 语境（保留旧 _extract_query 的价值）──
    Case("讲讲 diffusion model", "topic", includes=("diffusion",)),
]


def _fmt(s: str, width: int) -> str:
    """按显示宽度（中文算 2）左对齐截断。"""
    disp = sum(2 if ord(ch) > 0x2E7F else 1 for ch in s)
    if disp <= width:
        return s + " " * (width - disp)
    # 截断
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
    id_total = id_ok = 0
    mode_total = mode_ok = 0
    kw_total = kw_ok = 0
    failures: list[tuple[Case, str]] = []

    print(f"\n{'=' * 78}")
    print("Search 意图 golden set（真实配置模型）")
    print(f"{'=' * 78}")
    print(f"{_fmt('utterance', 34)}{_fmt('期望', 14)}{_fmt('实际', 20)}结果")
    print("-" * 78)

    for c in CASES:
        provided = _extract_provided_arxiv_id(c.utterance)

        # 确定性 arXiv 分支
        if c.expect_arxiv_id is not None:
            id_total += 1
            ok = provided == c.expect_arxiv_id
            id_ok += int(ok)
            print(f"{_fmt(c.utterance, 34)}{_fmt('id:' + c.expect_arxiv_id, 14)}"
                  f"{_fmt('id:' + (provided or '∅'), 20)}{'✓' if ok else '✗'}")
            if not ok:
                failures.append((c, f"arXiv 解析：期望 {c.expect_arxiv_id}，实得 {provided or '∅'}"))
            continue

        # 非 ID：正则不应误触发
        false_id = bool(provided)

        ctx = "\n".join(c.context)
        intent = _classify_search_intent(c.utterance, ctx, llm)

        mode_total += 1
        mode_hit = intent.mode == c.expect_mode
        mode_ok += int(mode_hit)

        q = intent.query.lower()
        kw_hit = all(k.lower() in q for k in c.includes) and all(k.lower() not in q for k in c.excludes)
        if c.includes or c.excludes:
            kw_total += 1
            kw_ok += int(kw_hit)

        case_ok = mode_hit and kw_hit and not false_id
        actual = f"{intent.mode}:{_fmt(intent.query, 0)}"
        print(f"{_fmt(c.utterance, 34)}{_fmt(c.expect_mode or '-', 14)}"
              f"{_fmt(intent.mode, 20)}{'✓' if case_ok else '✗'}")
        if not case_ok:
            det = []
            if false_id:
                det.append(f"正则误判出 ID {provided}")
            if not mode_hit:
                det.append(f"mode 期望 {c.expect_mode} 实得 {intent.mode}")
            if not kw_hit:
                det.append(f"query={intent.query!r} 不满足 includes={c.includes} excludes={c.excludes}")
            failures.append((c, "；".join(det)))
        _ = actual  # 保留 query 供调试（已在失败明细里展示）

    print("-" * 78)
    print("汇总：")
    if id_total:
        print(f"  arXiv 解析准确率 : {id_ok}/{id_total} = {id_ok / id_total:.0%}")
    if mode_total:
        print(f"  mode 分类准确率  : {mode_ok}/{mode_total} = {mode_ok / mode_total:.0%}")
    if kw_total:
        print(f"  query 关键词准确率: {kw_ok}/{kw_total} = {kw_ok / kw_total:.0%}（含指代消解 / 消歧）")
    print(f"  整体通过         : {len(CASES) - len(failures)}/{len(CASES)}")

    if failures:
        print(f"\n失败明细（{len(failures)} 条，边界案例回归看这里）：")
        for c, why in failures:
            print(f"  ✗ {c.utterance!r}  —  {why}")
    else:
        print("\n✅ 全部通过。可作为进入阶段 2（上游改写 / TaskSpec）的基线。")
    print("=" * 78)


if __name__ == "__main__":
    main()
