from datetime import datetime, timedelta, timezone

import pytest

from opendetect_ai import approval


def _point_db(monkeypatch, tmp_path):
    db = str(tmp_path / "approval.db")
    monkeypatch.setattr(approval, "_get_db_path", lambda: db)
    return db


def test_approval_is_audited_and_replay_is_idempotent(monkeypatch, tmp_path) -> None:
    _point_db(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(approval, "interrupt", lambda payload: calls.append(payload) or [0])

    kwargs = {
        "action": "confirm_ingest",
        "payload": {"papers": [{"idx": 0, "title": "A"}]},
        "reason": "modify knowledge base",
        "thread_id": "thread-a",
        "user_id": "user-a",
        "idempotency_key": "turn-1",
    }
    assert approval.approval_required(**kwargs) == [0]
    assert approval.approval_required(**kwargs) == [0]
    assert len(calls) == 1
    assert calls[0]["thread_id"] == "thread-a"
    assert calls[0]["user_id"] == "user-a"

    rows = approval.list_approvals("user-a")
    assert len(rows) == 1
    assert rows[0]["status"] == "approved"
    assert rows[0]["decision"] == [0]


def test_expired_approval_becomes_rejected(monkeypatch, tmp_path) -> None:
    _point_db(monkeypatch, tmp_path)
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    request = {
        "approval_id": "expired-id",
        "thread_id": "thread-a",
        "user_id": "user-a",
        "action": "export_report",
        "reason": "external side effect",
        "payload": {},
        "created_at": (past - timedelta(minutes=1)).isoformat(),
        "expires_at": past.isoformat(),
    }
    approval._record_pending(request)
    assert approval.list_approvals("user-a")[0]["status"] == "expired"
    assert approval._resolve_approval("expired-id", "all") == "none"
    assert approval.list_approvals("user-a")[0]["status"] == "expired"


def test_approval_fails_closed_when_audit_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(approval, "_connect", lambda: (_ for _ in ()).throw(OSError("db down")))
    monkeypatch.setattr(
        approval,
        "interrupt",
        lambda payload: pytest.fail("audit unavailable must not reach interrupt"),
    )

    with pytest.raises(RuntimeError, match="审批审计不可用"):
        approval.approval_required(
            action="export_report",
            payload={"format": "pdf"},
            reason="external side effect",
            thread_id="thread-a",
            user_id="user-a",
            idempotency_key="turn-1",
        )


def test_explicit_user_action_is_recorded_as_approved(monkeypatch, tmp_path) -> None:
    _point_db(monkeypatch, tmp_path)
    approval.record_explicit_approval(
        action="upload_pdf",
        payload={"title": "Paper", "sha256": "abc"},
        reason="user selected a local file",
        thread_id="thread-a",
        user_id="user-a",
        idempotency_key="abc",
    )

    rows = approval.list_approvals("user-a")
    assert len(rows) == 1
    assert rows[0]["action"] == "upload_pdf"
    assert rows[0]["status"] == "approved"
