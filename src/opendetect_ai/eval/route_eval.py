"""
Supervisor 路由评测（route-eval）—— OpenDetect_AI（阶段 ⑦.5）

评的是 Supervisor 的核心职责：给定 (resolved_query, 工作流状态) → 该路由到哪个节点。
这是「要不要上 TaskSpec」的证据基础——只有当这里出现明显、稳定的错误路由，才值得设计
最小 TaskSpec；否则不为不存在的问题造一个「承担所有语义的大对象」。

记录三件事：路由准确率、LLM 调用次数、延迟（P50/P95）。真实使用中的错误路由样本应持续
追加进 CASES（(query, state) → expected_next），让这里成为路由行为的回归基线。

说明：Supervisor 对 has_rag_answer / final_report / error 是**代码里确定性短路**（不进 LLM），
本 eval 只覆盖真正交给 LLM 决策的路由空间——那才是会出错的地方。

运行：
    make route-eval
    uv run python -m opendetect_ai.eval.route_eval
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from opendetect_ai.prompts import SUPERVISOR_PROMPT
from opendetect_ai.agents.supervisor import _route_with_llm


@dataclass
class Case:
    name: str
    query: str                       # resolved_query（已是自包含）
    expected_next: str               # search | ingest | rag | report | FINISH
    ingested_count: int = 5
    search_count: int = 0
    pending_count: int = 0
    failed_count: int = 0
    search_attempted: bool = False
    chat_context: str = ""


CASES: list[Case] = [
    # ── 闲聊 / 身份（→ FINISH）──
    Case("打招呼", "你好", "FINISH"),
    Case("问能力", "你能帮我做什么？", "FINISH"),
    Case("道谢", "谢谢啦", "FINISH"),
    # ── 知识问题、库非空（→ rag）──
    Case("知识问题-是什么", "ViT 是什么？", "rag"),
    Case("知识问题-对比", "PPO 和 SAC 在策略优化上有什么区别？", "rag"),
    Case("知识问题-细节", "Swin Transformer 的 shifted window 是怎么工作的？", "rag"),
    Case("追问-更多同主题", "还有哪些关于视觉 Transformer 的内容？", "rag"),
    # ── 知识问题、库空（→ FINISH 模式B 提议搜索）──
    Case("库空+知识问题", "讲讲 LoRA 低秩适配", "FINISH", ingested_count=0),
    # ── 明确搜索（→ search）──
    Case("明确搜索-主题", "帮我搜索 diffusion model 的论文", "search"),
    Case("明确搜索-标题", "找一篇 Attention Is All You Need", "search"),
    Case("确认改写后的搜索指令", "帮我搜索并入库与「LoRA 低秩适配」相关的论文", "search"),
    Case("明确要更多论文", "帮我再找几篇强化学习方向的论文", "search"),
    # ── 状态驱动（→ ingest / report / FINISH）──
    Case("有待入库论文", "继续处理", "ingest", pending_count=3),
    Case("重试失败入库", "重新入库刚才失败的那几篇", "ingest", failed_count=2),
    Case("生成综述", "帮我生成一份目标检测方向的综述", "report"),
    Case("查看已入库", "有哪些已经入库的论文？", "FINISH"),
    # ── 防重复搜索（search_attempted=True，不应再 search）──
    Case("已搜过不再搜", "再搜搜看还有没有别的", "FINISH", search_attempted=True, search_count=0),
]


def _build_prompt(c: Case) -> str:
    ctx = f"\n{c.chat_context}\n" if c.chat_context else "（无历史记录）"
    return SUPERVISOR_PROMPT.format(
        user_profile="（暂无跨会话记忆）",
        chat_context=ctx,
        user_query=c.query,
        search_count=c.search_count,
        ingested_count=c.ingested_count,
        pending_count=c.pending_count,
        has_rag_answer=False,
        error="无",
        search_attempted=c.search_attempted,
        failed_count=c.failed_count,
    )


def _pctl(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    i = min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))
    return s[i]


def main() -> None:
    print(f"\n{'=' * 74}\nSupervisor 路由评测（{len(CASES)} 条，真实模型）\n{'=' * 74}")
    print(f"{'用例':<26}{'期望':<9}{'实得':<9}{'ms':>7}  结果")
    print("-" * 74)
    ok = 0
    lat: list[float] = []
    fails = []
    for c in CASES:
        prompt = _build_prompt(c)
        t0 = time.time()
        decision = _route_with_llm(prompt)
        dt = (time.time() - t0) * 1000
        lat.append(dt)
        hit = decision.next == c.expected_next
        ok += int(hit)
        if not hit:
            fails.append((c, decision.next, decision.reason))
        name = c.name if len(c.name) <= 24 else c.name[:23] + "…"
        print(f"{name:<26}{c.expected_next:<9}{decision.next:<9}{dt:>7.0f}  {'✓' if hit else '✗'}")

    print("-" * 74)
    print(f"路由准确率 : {ok}/{len(CASES)} = {ok / len(CASES):.0%}")
    print(f"LLM 调用   : {len(CASES)} 次（每用例 1 次结构化输出；偶发回退 JSON 解析 +1）")
    print(f"延迟       : P50 {_pctl(lat, 50):.0f}ms · P95 {_pctl(lat, 95):.0f}ms")
    if fails:
        print(f"\n错误路由（{len(fails)}）——真实使用中遇到的也追加到 CASES：")
        for c, got, reason in fails:
            print(f"  ✗ {c.query!r}  期望 {c.expected_next} 实得 {got}  （{reason[:40]}）")
    else:
        print("\n✅ 全部命中。当前 Supervisor 路由无明显问题——尚无证据支持引入 TaskSpec。")
    print("=" * 74)


if __name__ == "__main__":
    main()
