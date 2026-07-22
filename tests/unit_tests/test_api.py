import asyncio
import io
from types import SimpleNamespace

import api
import pytest
from fastapi import HTTPException
from starlette.datastructures import Headers, UploadFile

from opendetect_ai import graph


def test_non_stream_chat_forwards_user_id(monkeypatch) -> None:
    received = {}

    def fake_chat(query, thread_id, user_id, hitl):
        received.update(query=query, thread_id=thread_id, user_id=user_id, hitl=hitl)
        return {"direct_answer": "ok"}

    monkeypatch.setattr(graph, "chat", fake_chat)
    request = api.ChatRequest(query="hello", thread_id="thread-a", user_id="user-a")
    response = asyncio.run(api.chat_endpoint(request))

    assert response["answer"] == "ok"
    assert received == {
        "query": "hello", "thread_id": "thread-a", "user_id": "user-a", "hitl": True,
    }


def test_non_stream_chat_returns_pending_approval(monkeypatch) -> None:
    def fake_chat(*_args):
        return {"_interrupts": [{"action": "confirm_ingest", "approval_id": "a1"}]}

    monkeypatch.setattr(graph, "chat", fake_chat)
    request = api.ChatRequest(query="search", thread_id="thread-a", user_id="user-a")
    response = asyncio.run(api.chat_endpoint(request))

    assert response == {"action": "confirm_ingest", "approval_id": "a1", "type": "interrupt"}


def test_extract_answer_exposes_verification_result() -> None:
    result = api._extract_answer({
        "rag_answer": "grounded answer",
        "verification": {"status": "passed", "confidence": "high"},
    })
    assert result["verification"] == {"status": "passed", "confidence": "high"}


def test_extract_report_exposes_verification_result() -> None:
    result = api._extract_answer({
        "final_report": "verified report",
        "verification": {"status": "passed", "output_kind": "final_report"},
    })
    assert result["type"] == "report"
    assert result["verification"]["status"] == "passed"


async def _body_text(response) -> str:
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
    return "".join(chunks)


def test_resume_rejects_when_no_approval_is_pending(monkeypatch) -> None:
    monkeypatch.setattr(graph, "_get_chat_graph", lambda: object())
    monkeypatch.setattr(api, "_pending_interrupts", lambda chat_graph, config: [])

    async def run():
        response = await api.chat_resume_endpoint(
            api.ResumeRequest(thread_id="thread-a", user_id="user-a", selection="all")
        )
        return await _body_text(response)

    body = asyncio.run(run())
    assert "没有待处理审批" in body


def test_memory_settings_endpoint_updates_current_user(monkeypatch, tmp_path) -> None:
    from opendetect_ai import user_memory

    monkeypatch.setattr(user_memory, "_get_db_path", lambda: str(tmp_path / "memory.db"))
    response = asyncio.run(api.update_memory_settings(
        api.MemorySettingsRequest(user_id="alice", enabled=False, ttl_days=30)
    ))

    assert response["status"] == "ok"
    assert response["settings"]["enabled"] is False
    assert response["settings"]["ttl_days"] == 30
    assert user_memory.get_memory_settings("bob")["enabled"] is True


def test_approvals_endpoint_only_returns_requested_user(monkeypatch, tmp_path) -> None:
    from opendetect_ai import approval

    monkeypatch.setattr(approval, "_get_db_path", lambda: str(tmp_path / "approval.db"))
    now = approval._utcnow()
    for user_id in ("alice", "bob"):
        approval._record_pending({
            "approval_id": f"approval-{user_id}",
            "thread_id": f"thread-{user_id}",
            "user_id": user_id,
            "action": "export_report",
            "reason": "external side effect",
            "payload": {"owner": user_id},
            "created_at": now.isoformat(),
            "expires_at": (now + approval.timedelta(minutes=5)).isoformat(),
        })

    response = asyncio.run(api.get_approvals(user_id="alice"))
    assert response["total"] == 1
    assert response["approvals"][0]["payload"] == {"owner": "alice"}


def test_upload_pdf_rejects_wrong_mime_type() -> None:
    upload = UploadFile(
        file=io.BytesIO(b"%PDF-fake"),
        filename="paper.pdf",
        headers=Headers({"content-type": "text/plain"}),
    )
    with pytest.raises(HTTPException, match="MIME") as exc_info:
        asyncio.run(api.upload_pdf(upload))
    assert exc_info.value.status_code == 400


def test_sse_never_emits_unverified_model_draft(monkeypatch) -> None:
    class Chunk:
        content = "UNVERIFIED-DRAFT"

    class FakeGraph:
        def stream(self, *_args, **_kwargs):
            yield "messages", (Chunk(), {"tags": ["final_answer"]})
            yield "values", {
                "rag_answer": "VERIFIED-ANSWER",
                "verification": {"status": "passed"},
                "messages": [],
                "user_id": "user-a",
            }

        def get_state(self, _config):
            return SimpleNamespace(tasks=[])

    monkeypatch.setattr(graph, "spawn_profile_extraction", lambda *_args: None)

    async def run():
        return "".join([
            chunk async for chunk in api._sse_run(
                FakeGraph(), {"configurable": {"thread_id": "thread-a"}}, {}, "thread-a"
            )
        ])

    body = asyncio.run(run())
    assert "UNVERIFIED-DRAFT" not in body
    assert "VERIFIED-ANSWER" in body
    assert '"verified": true' in body
