import json

import pytest

from opendetect_ai.eval import rag_eval


def _result(arxiv_id: str) -> dict:
    return {"arxiv_id": arxiv_id, "title": arxiv_id, "content": "text"}


def test_recall_and_ndcg_support_multiple_gold_papers() -> None:
    results = [_result("paper-a"), _result("noise"), _result("paper-b")]
    gold = {"paper-a", "paper-b"}

    assert rag_eval._recall_at(results, gold, 2) == 0.5
    assert rag_eval._recall_at(results, gold, 3) == 1.0
    assert 0.0 < rag_eval._ndcg_at(results, gold, 3) < 1.0


def test_load_questions_accepts_q_alias_and_normalizes_versions(tmp_path) -> None:
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text(
        "\n".join([
            json.dumps({"q": "first", "gold": ["2103.14030v2"]}),
            json.dumps({"question": "second", "gold": ["paper-b"]}),
        ]),
        encoding="utf-8",
    )

    assert rag_eval._load_questions(str(dataset)) == [
        {"q": "first", "gold": ["2103.14030"]},
        {"q": "second", "gold": ["paper-b"]},
    ]


def test_load_questions_rejects_missing_gold(tmp_path) -> None:
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text('{"q": "missing labels"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="gold"):
        rag_eval._load_questions(str(dataset))


def test_eval_method_records_retrieval_failures(monkeypatch) -> None:
    from opendetect_ai.tools import retriever

    monkeypatch.setattr(retriever, "_llm_call_count", 0)
    metrics = rag_eval._eval_method(
        lambda question, k: (_ for _ in ()).throw(RuntimeError("timeout")),
        k=5,
        judge=False,
        questions=[{"q": "question", "gold": ["paper-a"]}],
    )

    assert metrics["failure_rate"] == 1.0
    assert metrics["result_count"] == 0.0
    assert metrics["recall@5"] == 0.0
