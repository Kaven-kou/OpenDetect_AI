"""
Clarify 判定逻辑 —— OpenDetect_AI（阶段 ⑦ MVP，先只做纯判定，暂不接入 Graph）

只覆盖「可可靠观测」的 4 类澄清信号，`low_relevance` 暂缓（检索层尚未透出可比 rerank 分数）：
  - ambiguous_reference   ：指代无法唯一映射（候选必须能在历史里 grounding 到原文证据）
  - multiple_papers       ：精确标题查询返回多个「绝对下限 + 相对差距」都满足的接近候选
  - entity_conflict       ：用户同时给标题 + arXiv ID，按 ID 取回的标题与所述明显不一致
  - exact_title_not_found ：两个后端都「成功返回空」（后端报错 != 没找到）

设计红线（评审确定）：
  · 候选可 grounding：只信「能在对应历史消息里找到 evidence 原文」的候选，不裸信模型枚举。
  · 多候选用「绝对下限 + 相对差距」，不接受纯相对阈值（两个都很差、但同样差 != 接近候选）。
  · attempts = 已展示的澄清问题次数；初次=1，第一次无效回复后展示第二次置 2，第二次仍无效才兜底清空。
阈值先给具名常量，之后用 golden set 校准。
"""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from opendetect_ai.tools.progress import push_progress
from opendetect_ai.env_utils import (
    OPENDETECT_LLM_MODEL,
    OPENDETECT_LLM_BASE_URL,
    OPENDETECT_LLM_API_KEY,
)

# ── 具名阈值（待 golden set 校准，勿散落成魔法数）──────────────
TITLE_MATCH_FLOOR = 0.55        # 标题匹配「绝对下限」：低于它不算合理候选
TITLE_AMBIGUITY_GAP = 0.15      # top1 与 top2 的「相对差距」上限：小于它才算接近、才澄清
ENTITY_CONFLICT_MAX_SIM = 0.45  # 所述标题 vs 按 ID 取回标题：低于它判为冲突
MAX_CLARIFY_ATTEMPTS = 2        # 同一问题最多展示两次澄清，之后兜底清空

ClarifyReason = Literal[
    "ambiguous_reference", "multiple_papers", "entity_conflict", "exact_title_not_found",
]


# ── 结构化模型 ────────────────────────────────────────────────
class ReferentCandidate(BaseModel):
    entity: str = Field(description="候选指代对象（实体名，如 ViT / LoRA）")
    evidence: str = Field(description="该候选在历史消息中的原文证据（逐字片段）")
    message_index: int = Field(description="evidence 所在历史消息的下标")


class ResolveResult(BaseModel):
    """resolve 阶段那唯一一次 LLM 调用的结构化输出：改写 + 候选枚举（供 grounding 校验）。"""
    resolved_query: str = Field(description="把指代改写成自包含问题；无法唯一确定时给最可能的一种")
    candidates: list[ReferentCandidate] = Field(
        default_factory=list,
        description="当前输入里的指代可能指向的多个历史实体；唯一时给 0 或 1 个",
    )


class ClarifyOption(BaseModel):
    id: str
    label: str
    resolved_query: str = ""     # 选中该项后要用的自包含 query；空表示需用户进一步给信息


class ClarifyDecision(BaseModel):
    reason: ClarifyReason
    question: str
    options: list[ClarifyOption] = Field(default_factory=list)


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=OPENDETECT_LLM_MODEL,
        base_url=OPENDETECT_LLM_BASE_URL,
        api_key=OPENDETECT_LLM_API_KEY,
        temperature=0,
    )


# ── 文本归一化 + 标题打分（确定性）────────────────────────────
def _norm(text: str) -> str:
    """小写 + Unicode NFKC + 去标点空白，用于稳健比较。"""
    t = unicodedata.normalize("NFKC", text or "").lower()
    return re.sub(r"[^\w一-鿿]+", " ", t).strip()


def _tokens(text: str) -> set[str]:
    return {w for w in _norm(text).split() if w}


def _title_score(query: str, title: str) -> float:
    """
    标题匹配分 = query 词元被 title 覆盖的比例（token coverage）。
    对「短查询 vs 长标题」比裸 difflib 稳健：'BERT' 完整命中长标题得 1.0，'RoBERTa' 得 0。
    """
    q = _tokens(query)
    if not q:
        return 0.0
    t = _tokens(title)
    return len(q & t) / len(q)


def _strip_arxiv_version(aid: str) -> str:
    return re.sub(r"v\d+$", "", (aid or "").strip())


def _dedup_candidates(pool: list[dict]) -> list[dict]:
    """按 归一化标题 / arXiv ID（去版本号）去重：同一论文的不同版本/重复来源算一个。"""
    seen: set[str] = set()
    out: list[dict] = []
    for p in pool:
        aid = _strip_arxiv_version(p.get("arxiv_id", ""))
        key = f"id:{aid}" if aid else f"title:{_norm(p.get('title', ''))}"
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


# ── search 侧判定：multiple_papers / exact_title_not_found ─────
def judge_title_pool(
    query: str,
    pool: list[dict],
    status: Literal["ok", "empty", "error"] = "ok",
) -> ClarifyDecision | None:
    """
    精确标题路径的候选池判定。返回澄清决策或 None（无需澄清 / 交给上层）。

    - status=="error"（后端报错/超时）→ 返回 None：这是 operational error，
      **绝不伪装成 exact_title_not_found**，由上层按错误处理。
    - status=="empty" 或去重后无候选达到下限 → exact_title_not_found。
    - top1、top2 都 >= 绝对下限 且 差距 <= 相对差距上限 → multiple_papers（接近候选）。
    - 一强一弱（top1 达标、差距大）→ None（直接选强候选，不打扰用户）。
    """
    if status == "error":
        return None

    deduped = _dedup_candidates(pool)
    scored = sorted(
        ((p, _title_score(query, p.get("title", ""))) for p in deduped),
        key=lambda x: x[1], reverse=True,
    )

    if status == "empty" or not scored or scored[0][1] < TITLE_MATCH_FLOOR:
        return ClarifyDecision(
            reason="exact_title_not_found",
            question=f"我在两个来源都没找到与「{query}」精确匹配的论文。你要换个更完整的标题，还是按这个主题搜一批相关论文？",
            options=[
                ClarifyOption(id="1", label="按主题搜索相关论文", resolved_query=f"帮我搜索并入库与「{query}」相关的论文"),
                ClarifyOption(id="2", label="我换个标题再说", resolved_query=""),
            ],
        )

    if len(scored) >= 2:
        top1, top2 = scored[0][1], scored[1][1]
        if top1 >= TITLE_MATCH_FLOOR and top2 >= TITLE_MATCH_FLOOR and (top1 - top2) <= TITLE_AMBIGUITY_GAP:
            close = [p for p, s in scored if s >= TITLE_MATCH_FLOOR and (top1 - s) <= TITLE_AMBIGUITY_GAP]
            return ClarifyDecision(
                reason="multiple_papers",
                question="有多篇标题都很接近，你指的是哪一篇？",
                options=[
                    ClarifyOption(
                        id=str(i + 1),
                        label=p.get("title", "?"),
                        resolved_query=(
                            f"帮我入库 arXiv:{p['arxiv_id']}" if p.get("arxiv_id")
                            else f"帮我入库《{p.get('title', '')}》"
                        ),
                    )
                    for i, p in enumerate(close)
                ],
            )

    return None   # 一强候选，直接用，不澄清


# ── search 侧判定：entity_conflict ────────────────────────────
def judge_entity_conflict(stated_title: str, fetched_title: str) -> ClarifyDecision | None:
    """用户给了标题 + arXiv ID：按 ID 取回的标题与所述标题相似度过低 → 冲突澄清。"""
    if not stated_title or not fetched_title:
        return None
    # 双向覆盖取较大，避免长短标题误判
    sim = max(_title_score(stated_title, fetched_title), _title_score(fetched_title, stated_title))
    if sim >= ENTITY_CONFLICT_MAX_SIM:
        return None
    return ClarifyDecision(
        reason="entity_conflict",
        question=f"你给的标题「{stated_title}」和这个 arXiv ID 对应的论文「{fetched_title}」看起来不是同一篇，你想要哪一个？",
        options=[
            ClarifyOption(id="1", label=f"按标题：{stated_title}", resolved_query=f"帮我入库《{stated_title}》"),
            ClarifyOption(id="2", label=f"按 arXiv ID：{fetched_title}", resolved_query=f"帮我入库《{fetched_title}》"),
        ],
    )


# ── resolve 侧判定：ambiguous_reference（候选必须可 grounding）──
# 只统计「单数」指代；复数「它们/他们」指向多对象时不澄清，交给改写成多对象问题。
_SINGULAR_REF = ("它", "他", "她", "这个", "那个", "这篇", "那篇", "这项", "那项", "此")
_PLURAL_REF = ("它们", "他们", "她们", "这些", "那些", "两者", "二者")


def _has_singular_reference(raw: str) -> bool:
    if any(p in raw for p in _PLURAL_REF):
        return False
    return any(s in raw for s in _SINGULAR_REF)


def _grounded(cand: ReferentCandidate, messages: list[BaseMessage]) -> bool:
    """候选的 evidence 必须能在其声称的历史消息里逐字（归一化后）找到，否则视为模型臆造。"""
    idx = cand.message_index
    if not (0 <= idx < len(messages)):
        return False
    ev = _norm(cand.evidence)
    return bool(ev) and ev in _norm(getattr(messages[idx], "content", "") or "")


def judge_reference(raw: str, messages: list[BaseMessage], result: ResolveResult) -> ClarifyDecision | None:
    """
    仅当：当前输入含未解析的单数指代 + 至少两个「去重且 grounding 通过」的候选 → 才澄清。
    否则返回 None（唯一候选/复数指代/证据对不上 → 不打扰用户，用 result.resolved_query 即可）。
    """
    if not _has_singular_reference(raw):
        return None

    grounded = [c for c in result.candidates if _grounded(c, messages)]
    # 按实体名归一化去重（别名/大小写视为同一实体）
    uniq: dict[str, ReferentCandidate] = {}
    for c in grounded:
        uniq.setdefault(_norm(c.entity), c)
    cands = list(uniq.values())
    if len(cands) < 2:
        return None

    return ClarifyDecision(
        reason="ambiguous_reference",
        question="你说的“它”指的是哪一个？",
        options=[
            ClarifyOption(
                id=str(i + 1), label=c.entity,
                resolved_query=re.sub("|".join(_SINGULAR_REF), c.entity, raw, count=1),
            )
            for i, c in enumerate(cands)
        ],
    )


def resolve_reference_llm(raw: str, messages: list[BaseMessage], llm: ChatOpenAI | None = None) -> ResolveResult:
    """
    resolve 阶段那唯一一次改写调用，扩成结构化「改写 + 候选枚举」。fail-open：异常时无候选、原样返回。
    历史消息带下标喂给模型，便于它给出可核对的 message_index。
    """
    llm = llm or _get_llm()
    indexed = "\n".join(
        f"[{i}] {'用户' if isinstance(m, HumanMessage) else '助手'}：{getattr(m, 'content', '')}"
        for i, m in enumerate(messages)
    )
    prompt = (
        "把用户这句含指代/省略的话改写成自包含问题；同时列出这个指代在对话历史中可能指向的实体，"
        "每个实体必须给出其在历史里的**逐字证据**和所在消息下标（message_index）。"
        "指代唯一时 candidates 给 0 或 1 个；能唯一确定就直接改写。\n\n"
        f"## 带下标的对话历史\n{indexed}\n\n## 用户这句\n{raw}"
    )
    try:
        return llm.with_structured_output(ResolveResult, method="function_calling").invoke(
            [HumanMessage(content=prompt)]
        )
    except Exception as exc:
        print(f"[Clarify] 指代解析失败，fail-open: {exc}")
        return ResolveResult(resolved_query=raw, candidates=[])


# ── 下一轮：澄清回复的确定性解析 ──────────────────────────────
_CN_NUM = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
_REJECT = ("都不是", "都不对", "不是这些", "算了", "取消", "不用了", "没有", "都不要")
_SELECT_FILLER = ("那一篇", "这一篇", "那篇", "这篇", "那个", "这个", "那本", "这本", "论文", "那篇论文")


def _strip_select_filler(reply: str) -> str:
    r = reply
    for f in _SELECT_FILLER:
        r = r.replace(f, "")
    return r.strip()


def _parse_ordinal(reply: str) -> int | None:
    # 用负向断言把「更长数字/小数」的一部分挡掉：避免命中 arXiv ID(如 1907.11692)里的数字
    m = re.search(
        r"(?:第|选|选择)?\s*(?<![\d.])(\d{1,2}|[一二两三四五六七八九十])(?![\d.])\s*(?:篇|个|条|项|号|本)?",
        reply,
    )
    if not m:
        return None
    g = m.group(1)
    return int(g) if g.isdigit() else _CN_NUM.get(g)


def _looks_like_new_task(reply: str) -> bool:
    """像一条新的完整任务（含检索动词或较长），而非对选项的含糊回应。"""
    return bool(re.search(r"搜索|搜一下|帮我找|找一下|找几篇|讲讲|入库|综述|列出|生成", reply)) or len(reply.strip()) >= 12


class Selection(BaseModel):
    action: Literal["select", "clear", "reprocess", "reclarify", "fallback"]
    resolved_query: str = ""
    option_id: str = ""


def parse_clarification_selection(reply: str, pending: dict) -> Selection:
    """
    解析用户对澄清问题的回复（确定性，0 LLM）。attempts 护栏在此收口：
    reclarify 前若已展示达上限，改为 fallback（给可操作兜底并清空）。
    """
    options = pending.get("options", []) or []
    attempts = int(pending.get("attempts", 1))
    reply_s = (reply or "").strip()

    # 1) 明确拒绝
    if any(k in reply_s for k in _REJECT):
        return Selection(action="clear")

    # 2) 序号选择（越界不默认、不崩，继续澄清）
    n = _parse_ordinal(reply_s)
    if n is not None:
        if 1 <= n <= len(options):
            opt = options[n - 1]
            return Selection(action="select", option_id=str(opt.get("id", n)),
                             resolved_query=opt.get("resolved_query", ""))
        return _reclarify_or_fallback(attempts)

    # 3) 标题 / arXiv ID 模糊匹配某个选项
    core = _strip_select_filler(reply_s)
    for opt in options:
        label = opt.get("label", "")
        if label and core and (_title_score(core, label) >= 0.6 or _norm(core) in _norm(label)):
            return Selection(action="select", option_id=str(opt.get("id", "")),
                             resolved_query=opt.get("resolved_query", ""))
        aid = _strip_arxiv_version(opt.get("arxiv_id", ""))
        if aid and aid in reply_s:
            return Selection(action="select", option_id=str(opt.get("id", "")),
                             resolved_query=opt.get("resolved_query", ""))

    # 4) 像新的完整任务 → 清空旧澄清并按新任务处理
    if _looks_like_new_task(reply_s):
        return Selection(action="reprocess", resolved_query=reply_s)

    # 5) 含糊回应 → 继续澄清（受 attempts 护栏约束）
    return _reclarify_or_fallback(attempts)


def _reclarify_or_fallback(attempts: int) -> Selection:
    if attempts >= MAX_CLARIFY_ATTEMPTS:
        return Selection(action="fallback")
    return Selection(action="reclarify")


# ── pending_action(clarification) 构造 / 渲染 / clarify 节点 ────
def build_clarify_pending(decision: ClarifyDecision, original_query: str, attempts: int = 1) -> dict:
    """把一次澄清判定落成可持久化的 pending_action（复用同一字段，kind 区分）。"""
    return {
        "kind": "clarification",
        "reason": decision.reason,
        "original_query": original_query,
        "question": decision.question,
        "options": [o.model_dump() for o in decision.options],
        "attempts": attempts,
    }


_FALLBACK_TEXT = "我不太确定你指的是哪一个 😅 你可以直接发论文标题或 arXiv ID，我就能定位；或者换个说法重新描述需求。"


def fallback_pending(original_query: str) -> dict:
    """澄清达上限后的兜底：无选项的终结式澄清，clarify 节点渲染后即清空。"""
    return {
        "kind": "clarification", "reason": "fallback", "original_query": original_query,
        "question": _FALLBACK_TEXT, "options": [], "attempts": MAX_CLARIFY_ATTEMPTS,
    }


def render_clarify(pending: dict) -> str:
    """把澄清 pending 渲染成面向用户的一段文本（问题 + 带序号的选项）。"""
    lines = [pending.get("question", "你指的是哪一个？")]
    for o in pending.get("options", []) or []:
        lines.append(f"{o.get('id', '?')}. {o.get('label', '')}")
    return "\n".join(lines)


def clarify_node(state: dict) -> dict:
    """
    渲染澄清问题并收束到 END（普通对话轮，不用 interrupt）。
    - 常规澄清（有选项）：保留 pending_action，等下一轮 resolve 确定性解析用户选择。
    - 兜底（无选项）：渲染后清空 pending_action，避免下一轮再对空选项解析。
    """
    _tid = state.get("thread_id", "default")
    pending = state.get("pending_action") or {}
    msg = render_clarify(pending)
    push_progress(_tid, f"❓ 需要澄清：{pending.get('reason', '')}")
    print(f"[Clarify] {pending.get('reason', '')} → 询问用户")

    out = {
        "direct_answer": msg,
        "messages": [AIMessage(content=msg)],
        "next": "FINISH",
    }
    if not pending.get("options"):     # 兜底式（无选项）→ 渲染后清空
        out["pending_action"] = None
    return out

