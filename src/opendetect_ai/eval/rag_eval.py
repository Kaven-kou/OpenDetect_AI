"""
RAG 检索评估 —— OpenDetect_AI

对比「朴素稠密检索(baseline)」与「Hybrid + Self-Query + Rerank(新管线)」在同一
受控基准语料上的检索质量，量化你在简历里能讲的那句「检索质量提升 X%」。

为什么自带受控语料（而非直接跑真实库）：
- 可复现：不依赖 arxiv 下载（正是被限流的痛点），任何机器结果一致。
- 有金标准：每个问题的正确论文 + 跨领域「噪音」论文都是已知的，指标才有意义。
- 噪音语料刻意混入医学 CLIP / 天气预报等，复刻真实脏库，专测「噪音拒绝」。

指标：
- Hit@1 / Hit@5 : top-k 中是否命中正确论文（召回是否正确）
- MRR           : 正确论文首次出现位置的倒数（排序质量）
- Recall@5      : 多 gold 问题中召回了多少篇正确论文
- Precision@5   : top-k 中属于正确论文的比例（越高越聚焦）
- nDCG@5        : 多 gold 的论文级排序质量
- Noise@5       : top-k 中属于跨领域噪音论文的比例（越低越干净）★核心
- CtxRelevant   : LLM 判定「检索内容是否足以回答问题」（0/1，可用 --no-judge 关闭）
- P50/P95、失败率、平均结果数、检索侧 LLM 调用数：质量之外的成本与稳定性

运行：
    make eval
    uv run python -m opendetect_ai.eval.rag_eval --no-judge   # 更快，跳过 LLM 判分
    uv run python -m opendetect_ai.eval.rag_eval --dataset data/eval/questions.jsonl

外部 JSONL 每行格式：
    {"q": "问题", "gold": ["arxiv_id_1", "arxiv_id_2"]}

传入 --dataset 时直接评估当前知识库，不再写入受控合成语料。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage


# ══════════════════════════════════════════════════════════════
#   受控基准语料（5 篇目标论文 + 4 篇跨领域噪音）
# ══════════════════════════════════════════════════════════════
_TARGETS = [
    {
        "title": "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
        "arxiv_id": "2010.11929", "published": "2020-10-22", "authors": "Dosovitskiy et al.",
        "chunks": [
            "We split an image into fixed-size patches, linearly embed each of them, add position "
            "embeddings, and feed the resulting sequence of vectors to a standard Transformer encoder. "
            "The Vision Transformer (ViT) treats image patches the same way tokens are treated in NLP.",
            "When pre-trained on large datasets and transferred to mid-sized image recognition benchmarks, "
            "ViT attains excellent results compared to state-of-the-art convolutional networks while "
            "requiring substantially fewer computational resources to train.",
        ],
    },
    {
        "title": "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows",
        "arxiv_id": "2103.14030", "published": "2021-03-25", "authors": "Liu et al.",
        "chunks": [
            "Swin Transformer computes self-attention within local non-overlapping windows. The shifted "
            "window scheme alternates the window partitioning between consecutive layers, introducing "
            "cross-window connections while keeping computation linear in image size.",
            "This hierarchical architecture has the flexibility to model at various scales, unlike ViT which "
            "produces feature maps of a single low resolution and has quadratic complexity to image size "
            "because it computes global self-attention.",
        ],
    },
    {
        "title": "DINOv2: Learning Robust Visual Features without Supervision",
        "arxiv_id": "2304.07193", "published": "2023-04-14", "authors": "Oquab et al.",
        "chunks": [
            "DINOv2 learns general-purpose visual features via self-supervised learning without any labels, "
            "using a student-teacher self-distillation objective over a large curated image dataset.",
            "The resulting frozen features work out of the box on classification, segmentation and depth "
            "estimation, rivaling weakly-supervised alternatives without fine-tuning.",
        ],
    },
    {
        "title": "Diffusion Models: A Comprehensive Survey of Methods and Applications",
        "arxiv_id": "2209.00796", "published": "2022-09-02", "authors": "Yang et al.",
        "chunks": [
            "In the forward diffusion process, a data sample is gradually corrupted by adding small amounts "
            "of Gaussian noise over many timesteps until it becomes pure noise. The model then learns the "
            "reverse denoising process to generate samples from noise.",
            "We categorize diffusion models into denoising diffusion probabilistic models, score-based "
            "generative models, and stochastic differential equation formulations.",
        ],
    },
    {
        "title": "Learning Transferable Visual Models From Natural Language Supervision",
        "arxiv_id": "2103.00020", "published": "2021-02-26", "authors": "Radford et al.",
        "chunks": [
            "CLIP is trained on 400 million natural image-text pairs collected from the internet using a "
            "contrastive objective that predicts which caption goes with which image, learning transferable "
            "visual representations from natural language supervision.",
            "At test time CLIP enables zero-shot transfer to downstream tasks by embedding the names of "
            "target classes as text prompts and matching them against image embeddings.",
        ],
    },
]

# 跨领域噪音（复刻真实脏库里混进来的那些）
_NOISE = [
    {
        "title": "PMC-CLIP: Contrastive Language-Image Pre-training using Biomedical Documents",
        "arxiv_id": "2303.07240", "published": "2023-03-13", "authors": "Lin et al.",
        "chunks": ["We pre-train a contrastive language-image model on biomedical figure-caption pairs "
                   "extracted from PubMed Central for medical image understanding."],
    },
    {
        "title": "Accurate medium-range global weather forecasting with 3D neural networks",
        "arxiv_id": "", "published": "2023-07-05", "authors": "Bi et al.",
        "chunks": ["A 3D Earth-specific transformer predicts atmospheric variables such as geopotential and "
                   "temperature for medium-range global weather forecasting, outperforming numerical methods."],
    },
    {
        "title": "AI-Assisted Pipeline for Dynamic Generation of Trustworthy Health Supplement Content",
        "arxiv_id": "", "published": "2018-10-11", "authors": "Anon.",
        "chunks": ["An AI-assisted content pipeline dynamically generates trustworthy marketing copy for "
                   "dietary health supplements at scale, with human review checkpoints."],
    },
    {
        "title": "Temporal networks",
        "arxiv_id": "", "published": "2012-03-06", "authors": "Holme and Saramaki",
        "chunks": ["Temporal networks are graphs whose edges are active only at certain points in time; we "
                   "review representations, metrics and dynamical processes on such time-varying graphs."],
    },
]

# 问题集：每个问题绑定正确论文的 arxiv_id（金标准，支持多 gold）
# 说明：这是「受控合成基准」（语料/标注均已知），用于可复现地对比检索策略；
# 生产中应替换为几十~上百条真实标注问题。多 gold 用于跨主题/综述型问题。
_QUESTIONS = [
    {"q": "ViT 是如何把图像切成 patch 输入 Transformer 的？",        "gold": ["2010.11929"]},
    {"q": "ViT 相比卷积网络有什么优势？",                            "gold": ["2010.11929"]},
    {"q": "Swin Transformer 的 shifted window 机制是什么？",         "gold": ["2103.14030"]},
    {"q": "shifted window 相比标准 ViT 的全局注意力有什么优势？",     "gold": ["2103.14030"]},
    {"q": "Swin 的层次化结构如何处理多尺度？",                       "gold": ["2103.14030"]},
    {"q": "DINOv2 如何在无监督下学习视觉特征？",                     "gold": ["2304.07193"]},
    {"q": "扩散模型的前向加噪过程是怎样的？",                        "gold": ["2209.00796"]},
    {"q": "扩散模型可以分成哪几类？",                                "gold": ["2209.00796"]},
    {"q": "CLIP 如何用自然图像和文本做对比学习预训练？",             "gold": ["2103.00020"]},
    {"q": "CLIP 如何实现零样本迁移？",                               "gold": ["2103.00020"]},
    # 跨主题 / 综述型（多 gold）
    {"q": "视觉 Transformer 有哪些代表性工作？",                     "gold": ["2010.11929", "2103.14030"]},
    {"q": "自监督与对比学习在视觉表征里怎么用？",                    "gold": ["2304.07193", "2103.00020"]},
]

_NOISE_KEYS = {p["title"] for p in _NOISE}
EVAL_DIR = "./data/eval_chroma"


# ══════════════════════════════════════════════════════════════
#   语料装载（指向独立 eval 集，幂等）
# ══════════════════════════════════════════════════════════════
def _seed_corpus() -> None:
    """把基准语料写入独立的 eval Chroma 集，不污染真实库。确定性 id → 可重复运行。"""
    from opendetect_ai.tools import rag_tool, retriever

    rag_tool.CHROMA_PERSIST_DIR = EVAL_DIR
    rag_tool._vectorstore = None                       # 重置单例，指向 eval 集
    retriever._bm25_cache = {"version": -1, "retriever": None}

    vs = rag_tool._get_vectorstore()
    docs, ids = [], []
    for paper in _TARGETS + _NOISE:
        for i, chunk in enumerate(paper["chunks"]):
            docs.append(Document(page_content=chunk, metadata={
                "title": paper["title"], "arxiv_id": paper["arxiv_id"],
                "authors": paper["authors"], "published": paper["published"], "chunk_idx": i,
            }))
            base = paper["arxiv_id"] or paper["title"]
            ids.append(f"eval::{base}__chunk_{i}")
    vs.add_documents(docs, ids=ids)
    rag_tool.bump_corpus_version()


# ══════════════════════════════════════════════════════════════
#   指标
# ══════════════════════════════════════════════════════════════
def _norm(a: str) -> str:
    import re
    return re.sub(r"v\d+$", "", (a or "").strip())


def _hit_at(results, gold: set, k) -> float:
    return 1.0 if any(_norm(r.get("arxiv_id")) in gold for r in results[:k]) else 0.0


def _mrr(results, gold: set) -> float:
    for idx, r in enumerate(results, start=1):
        if _norm(r.get("arxiv_id")) in gold:
            return 1.0 / idx
    return 0.0


def _precision_at(results, gold: set, k) -> float:
    top = results[:k]
    if not top:
        return 0.0
    return sum(1 for r in top if _norm(r.get("arxiv_id")) in gold) / len(top)


def _recall_at(results, gold: set, k) -> float:
    if not gold:
        return 0.0
    retrieved = {
        _norm(r.get("arxiv_id"))
        for r in results[:k]
        if _norm(r.get("arxiv_id")) in gold
    }
    return len(retrieved) / len(gold)


def _ndcg_at(results, gold: set, k) -> float:
    """论文级二值 nDCG@k：同一篇的多个 chunk 只算一次，避免 DCG 超过 IDCG。"""
    import math
    seen, rels = set(), []
    for r in results:
        pid = _norm(r.get("arxiv_id"))
        if pid in seen:
            continue
        seen.add(pid)
        rels.append(1.0 if pid in gold else 0.0)
        if len(rels) >= k:
            break
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def _noise_at(results, k) -> float:
    top = results[:k]
    if not top:
        return 0.0
    return sum(1 for r in top if r.get("title") in _NOISE_KEYS) / len(top)


def _judge_relevance(question: str, results: list[dict]) -> float:
    """LLM 判定检索到的内容是否足以回答问题（0/1）。"""
    from opendetect_ai.env_utils import (
        OPENDETECT_LLM_MODEL, OPENDETECT_LLM_BASE_URL, OPENDETECT_LLM_API_KEY)
    from langchain_openai import ChatOpenAI
    ctx = "\n---\n".join(r.get("content", "")[:300] for r in results[:5])
    prompt = (f"问题：{question}\n\n检索到的内容：\n{ctx}\n\n"
              "这些内容是否足以回答该问题？只回答 yes 或 no。")
    try:
        llm = ChatOpenAI(model=OPENDETECT_LLM_MODEL, base_url=OPENDETECT_LLM_BASE_URL,
                         api_key=OPENDETECT_LLM_API_KEY, temperature=0)
        ans = llm.invoke([HumanMessage(content=prompt)]).content.strip().lower()
        return 1.0 if "yes" in ans else 0.0
    except Exception:
        return 0.0


def _pctl(values: list[float], p: float) -> float:
    """简单百分位（线性插值），values 非空。"""
    if not values:
        return 0.0
    s = sorted(values)
    idx = (len(s) - 1) * p
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def _load_questions(path: str) -> list[dict]:
    """读取人工标注 JSONL；拒绝空问题和空 gold，避免产生虚高指标。"""
    questions = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"数据集第 {line_number} 行不是有效 JSON: {exc.msg}") from exc
            question = item.get("q") or item.get("question")
            gold = item.get("gold")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"数据集第 {line_number} 行缺少非空 q/question")
            if not isinstance(gold, list) or not gold or not all(isinstance(x, str) and x.strip() for x in gold):
                raise ValueError(f"数据集第 {line_number} 行 gold 必须是非空字符串数组")
            questions.append({"q": question.strip(), "gold": [_norm(x) for x in gold]})
    if not questions:
        raise ValueError("评测数据集为空")
    return questions


# ══════════════════════════════════════════════════════════════
#   评估主流程
# ══════════════════════════════════════════════════════════════
def _eval_method(retrieve_fn, k: int, judge: bool, questions: list[dict] | None = None) -> dict:
    import time
    from opendetect_ai.tools import retriever

    questions = questions or _QUESTIONS
    metric_k = max(1, k)
    agg = {
        "hit@1": 0.0,
        f"hit@{metric_k}": 0.0,
        "mrr": 0.0,
        f"prec@{metric_k}": 0.0,
        f"recall@{metric_k}": 0.0,
        f"ndcg@{metric_k}": 0.0,
        f"noise@{metric_k}": 0.0,
        "ctx": 0.0,
    }
    latencies, llm_calls = [], []
    failures = 0
    result_counts = []
    for item in questions:
        gold = {_norm(g) for g in item["gold"]}
        calls_before = retriever._llm_call_count
        t0 = time.perf_counter()
        try:
            results = retrieve_fn(item["q"], metric_k)
        except Exception:
            results = []
            failures += 1
        latencies.append((time.perf_counter() - t0) * 1000.0)   # ms
        llm_calls.append(retriever._llm_call_count - calls_before)

        error_results = [r for r in results if "error" in r]
        if error_results:
            failures += 1
        results = [r for r in results if "error" not in r]
        result_counts.append(len(results))
        agg["hit@1"] += _hit_at(results, gold, 1)
        agg[f"hit@{metric_k}"] += _hit_at(results, gold, metric_k)
        agg["mrr"] += _mrr(results, gold)
        agg[f"prec@{metric_k}"] += _precision_at(results, gold, metric_k)
        agg[f"recall@{metric_k}"] += _recall_at(results, gold, metric_k)
        agg[f"ndcg@{metric_k}"] += _ndcg_at(results, gold, metric_k)
        agg[f"noise@{metric_k}"] += _noise_at(results, metric_k)
        if judge:
            agg["ctx"] += _judge_relevance(item["q"], results)

    n = len(questions)
    out = {m: v / n for m, v in agg.items()}
    out["p50_ms"] = _pctl(latencies, 0.50)
    out["p95_ms"] = _pctl(latencies, 0.95)
    out["llm_calls"] = sum(llm_calls) / n     # 平均每次检索的 LLM 调用数
    out["failure_rate"] = failures / n
    out["result_count"] = sum(result_counts) / n
    out["question_count"] = n
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 检索评估：baseline vs 新管线")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--no-judge", action="store_true", help="跳过 LLM 判分，更快")
    parser.add_argument(
        "--dataset",
        help="人工标注 JSONL；设置后评估当前知识库，不装载仓库内受控语料",
    )
    args = parser.parse_args()

    from opendetect_ai.env_utils import validate_env
    validate_env()

    if args.dataset:
        questions = _load_questions(args.dataset)
        benchmark = f"人工标注集 {args.dataset}"
        print(f"→ 使用当前知识库评估 {benchmark}（{len(questions)} 题）...")
    else:
        questions = _QUESTIONS
        benchmark = "受控合成基准"
        print("→ 装载受控基准语料（5 篇目标 + 4 篇跨领域噪音）...")
        _seed_corpus()

    from opendetect_ai.tools.retriever import retrieve, retrieve_dense_only
    judge = not args.no_judge

    print("→ 评估 baseline（纯稠密 top-k）...")
    base = _eval_method(lambda q, k: retrieve_dense_only(q, k), args.k, judge, questions)
    print("→ 评估 新管线（Hybrid + Self-Query + Rerank）...")
    new = _eval_method(lambda q, k: retrieve(q, k), args.k, judge, questions)

    metric_k = max(1, args.k)
    metrics = [
        "hit@1", f"hit@{metric_k}", "mrr", f"prec@{metric_k}",
        f"recall@{metric_k}", f"ndcg@{metric_k}", f"noise@{metric_k}",
    ] + (["ctx"] if judge else [])
    metrics = list(dict.fromkeys(metrics))
    better_when_low = {f"noise@{metric_k}"}

    print(f"\n评估集：{len(questions)} 题（支持多 gold），{benchmark}；k={metric_k}")
    print("=" * 66)
    print(f"{'指标':<10}{'Baseline':>14}{'新管线':>14}{'Δ':>16}")
    print("-" * 66)
    for m in metrics:
        b, nw = base[m], new[m]
        delta = nw - b
        arrow = "↓好" if m in better_when_low else "↑好"
        sign = "+" if delta >= 0 else ""
        print(f"{m:<10}{b:>14.2f}{nw:>14.2f}{sign+format(delta,'.2f'):>14} {arrow}")
    print("-" * 66)
    # 成本 / 延迟（新管线的代价面，如实呈现）
    print(f"{'P50 延迟':<10}{base['p50_ms']:>12.0f}ms{new['p50_ms']:>12.0f}ms")
    print(f"{'P95 延迟':<10}{base['p95_ms']:>12.0f}ms{new['p95_ms']:>12.0f}ms")
    print(f"{'LLM/次检索':<10}{base['llm_calls']:>14.1f}{new['llm_calls']:>14.1f}")
    print(f"{'平均结果数':<10}{base['result_count']:>14.1f}{new['result_count']:>14.1f}")
    print(f"{'失败率':<10}{base['failure_rate']:>14.2%}{new['failure_rate']:>14.2%}")
    print("=" * 66)
    print("注：noise@5 越低越好；其余质量指标越高越好。新管线用更高延迟 / 额外 LLM")
    print("    调用（self-query + rerank）换取更干净的召回——这是要如实权衡的成本面。")


if __name__ == "__main__":
    main()
