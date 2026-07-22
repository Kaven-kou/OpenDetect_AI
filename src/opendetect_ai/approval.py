"""通用 Human-in-the-Loop 审批原语：可恢复、可过期、可审计、幂等。"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any

from langgraph.types import interrupt

from opendetect_ai.env_utils import CHROMA_PERSIST_DIR, OPENDETECT_APPROVAL_TTL_SECONDS


_TERMINAL = {"approved", "rejected", "expired"}


def _get_db_path() -> str:
    return os.path.join(os.path.dirname(CHROMA_PERSIST_DIR), "chat_history.db")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _connect() -> sqlite3.Connection:
    db_path = _get_db_path()
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=5, check_same_thread=False)
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS approval_audit (
            approval_id TEXT PRIMARY KEY,
            thread_id   TEXT NOT NULL,
            user_id     TEXT NOT NULL,
            action      TEXT NOT NULL,
            reason      TEXT NOT NULL,
            payload     TEXT NOT NULL,
            decision    TEXT,
            status      TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            expires_at  TEXT NOT NULL,
            resolved_at TEXT
        )
    """)
    conn.commit()
    return conn


def _approval_id(
    action: str,
    payload: dict,
    thread_id: str,
    user_id: str,
    idempotency_key: str,
) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    raw = "\x1f".join((thread_id, user_id, action, idempotency_key, canonical))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _load_approval(approval_id: str) -> dict | None:
    try:
        conn = _connect()
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM approval_audit WHERE approval_id = ?", (approval_id,)
        ).fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception as exc:
        print(f"[Approval] 读取审计记录失败: {exc}")
        return None


def _record_pending(request: dict) -> dict:
    existing = _load_approval(request["approval_id"])
    if existing:
        return existing
    try:
        conn = _connect()
        conn.execute(
            """
            INSERT OR IGNORE INTO approval_audit
              (approval_id, thread_id, user_id, action, reason, payload,
               decision, status, created_at, expires_at, resolved_at)
            VALUES (?, ?, ?, ?, ?, ?, NULL, 'pending', ?, ?, NULL)
            """,
            (
                request["approval_id"], request["thread_id"], request["user_id"],
                request["action"], request["reason"],
                json.dumps(request["payload"], ensure_ascii=False),
                request["created_at"], request["expires_at"],
            ),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        print(f"[Approval] 写入待审批记录失败: {exc}")
    stored = _load_approval(request["approval_id"])
    if not stored:
        raise RuntimeError("审批审计不可用，已阻止高风险操作")
    return stored


def _decision_status(decision: Any) -> str:
    if decision == "all":
        return "approved"
    if isinstance(decision, dict):
        decision = decision.get("selected", [])
    if isinstance(decision, (list, tuple, set)) and decision:
        return "approved"
    return "rejected"


def _resolve_approval(approval_id: str, decision: Any) -> Any:
    """原子地完成审批；已完成时返回第一次决定，保证 resume 幂等。"""
    existing = _load_approval(approval_id)
    if existing and existing.get("status") in _TERMINAL:
        stored = existing.get("decision")
        return json.loads(stored) if stored is not None else "none"

    now = _utcnow()
    expired = bool(existing) and now >= datetime.fromisoformat(existing["expires_at"])
    final_decision = "none" if expired else decision
    status = "expired" if expired else _decision_status(final_decision)
    try:
        conn = _connect()
        cursor = conn.execute(
            """
            UPDATE approval_audit
               SET decision = ?, status = ?, resolved_at = ?
             WHERE approval_id = ? AND status = 'pending'
            """,
            (
                json.dumps(final_decision, ensure_ascii=False), status,
                now.isoformat(), approval_id,
            ),
        )
        conn.commit()
        if cursor.rowcount == 0:
            row = conn.execute(
                "SELECT decision FROM approval_audit WHERE approval_id = ?",
                (approval_id,),
            ).fetchone()
            conn.close()
            if row and row[0] is not None:
                return json.loads(row[0])
            return "none"
        conn.close()
    except Exception as exc:
        print(f"[Approval] 更新审批结果失败: {exc}")
        return "none"
    return final_decision


def approval_required(
    *,
    action: str,
    payload: dict,
    reason: str,
    thread_id: str,
    user_id: str,
    idempotency_key: str,
    ttl_seconds: int = OPENDETECT_APPROVAL_TTL_SECONDS,
) -> Any:
    """暂停工作流等待审批；重放时复用第一次终态决定。"""
    approval_id = _approval_id(action, payload, thread_id, user_id, idempotency_key)
    existing = _load_approval(approval_id)
    if existing and existing.get("status") in _TERMINAL:
        stored = existing.get("decision")
        return json.loads(stored) if stored is not None else "none"

    now = _utcnow()
    request = {
        "approval_id": approval_id,
        "thread_id": thread_id,
        "user_id": user_id,
        "action": action,
        "reason": reason,
        "payload": payload,
        "created_at": now.isoformat(),
        "expires_at": (now + timedelta(seconds=max(1, ttl_seconds))).isoformat(),
    }
    stored = _record_pending(request)
    interrupt_payload = {
        "approval_id": approval_id,
        "thread_id": thread_id,
        "user_id": user_id,
        "action": action,
        "reason": reason,
        "created_at": stored["created_at"],
        "expires_at": stored["expires_at"],
        **payload,
    }
    decision = interrupt(interrupt_payload)
    return _resolve_approval(approval_id, decision)


def record_explicit_approval(
    *,
    action: str,
    payload: dict,
    reason: str,
    thread_id: str,
    user_id: str,
    idempotency_key: str,
) -> dict:
    """审计用户直接发起的动作；动作本身即人工批准，无需额外 interrupt。"""
    approval_id = _approval_id(action, payload, thread_id, user_id, idempotency_key)
    now = _utcnow()
    request = {
        "approval_id": approval_id,
        "thread_id": thread_id,
        "user_id": user_id,
        "action": action,
        "reason": reason,
        "payload": payload,
        "created_at": now.isoformat(),
        "expires_at": (now + timedelta(seconds=OPENDETECT_APPROVAL_TTL_SECONDS)).isoformat(),
    }
    _record_pending(request)
    _resolve_approval(approval_id, "all")
    return _load_approval(approval_id) or request


def list_approvals(user_id: str, limit: int = 50) -> list[dict]:
    """返回指定用户最近的审批审计记录，不暴露其他用户数据。"""
    try:
        conn = _connect()
        now = _utcnow().isoformat()
        conn.execute(
            """
            UPDATE approval_audit
               SET decision = '"none"', status = 'expired', resolved_at = ?
             WHERE status = 'pending' AND expires_at <= ?
            """,
            (now, now),
        )
        conn.commit()
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT approval_id, thread_id, action, reason, payload, decision,
                   status, created_at, expires_at, resolved_at
              FROM approval_audit
             WHERE user_id = ?
             ORDER BY created_at DESC
             LIMIT ?
            """,
            (user_id, max(1, min(limit, 200))),
        ).fetchall()
        conn.close()
        out = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item["payload"])
            if item["decision"] is not None:
                item["decision"] = json.loads(item["decision"])
            out.append(item)
        return out
    except Exception as exc:
        print(f"[Approval] 查询审计记录失败: {exc}")
        return []
