"""
OpenDetect AI — FastAPI 后端包装层
将 LangGraph 工作流封装为 HTTP API，供前端调用。

运行方式（在项目根目录执行）:
    pip install fastapi uvicorn python-multipart
    uvicorn api:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field
from opendetect_ai.tools.progress import drain_queue, cleanup_queue
from opendetect_ai.env_utils import OPENDETECT_CORS_ORIGINS, OPENDETECT_MAX_PDF_MB

app = FastAPI(title="OpenDetect AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=OPENDETECT_CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 请求模型 ───────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=20_000)
    thread_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")
    user_id: str = Field(default="default", min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")


class ResumeRequest(BaseModel):
    """HITL 恢复请求：用户对「入库确认」的选择。"""
    thread_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")
    selection: Any = None   # list[int]（保留的序号）| "all" | "none"
    user_id: str = Field(default="default", min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")


class MemorySettingsRequest(BaseModel):
    user_id: str = Field(default="default", min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")
    enabled: bool | None = None
    ttl_days: int | None = None


# ── 各节点的流式进度提示文本 ──────────────────────────────────
_STEP_MESSAGES = {
    "search": "🔍 正在搜索论文...",
    "ingest": "📚 正在入库论文...",
    "rag":    "💬 正在检索并生成回答...",
    "report": "📝 正在生成综述报告...",
}

# Supervisor 路由日志的特征前缀，用于过滤
_ROUTING_PREFIX = "Supervisor 决策:"


def _is_routing_message(content: str) -> bool:
    """判断一条消息是否是 Supervisor 路由日志，而非面向用户的内容。"""
    if not content:
        return True
    return content.startswith(_ROUTING_PREFIX)


def _extract_answer(accumulated: dict) -> dict:
    """
    从流式事件累积的 state 中提取面向用户的回复。

    优先级：
    1. rag_answer  — RAG 问答结果
    2. final_report — 综述报告
    3. direct_answer — Supervisor 的闲聊/引导回复
    4. messages 里最后一条非路由消息 — 入库/搜索完成通知
    5. 兜底文案
    """
    rag_answer    = accumulated.get("rag_answer", "") or ""
    final_report  = accumulated.get("final_report", "") or ""
    direct_answer = accumulated.get("direct_answer", "") or ""
    error         = accumulated.get("error", "") or ""
    ingested      = accumulated.get("ingested_count", 0) or 0
    messages      = accumulated.get("messages", []) or []

    if rag_answer:
        return {
            "answer": rag_answer,
            "type": "rag",
            "ingested_count": ingested,
            "error": "",
            "verification": accumulated.get("verification") or {},
        }

    if final_report:
        return {
            "answer": final_report,
            "type": "report",
            "ingested_count": ingested,
            "error": "",
            "verification": accumulated.get("verification") or {},
        }

    if direct_answer:
        return {"answer": direct_answer, "type": "info", "ingested_count": ingested, "error": ""}

    # 从消息列表里逆序找最后一条有实质内容的消息
    last_content = ""
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):     # 跳过用户 HumanMessage，绝不把用户输入当答案回显
            continue
        content = msg.content if hasattr(msg, "content") else str(msg)
        if content and not _is_routing_message(content):
            last_content = content
            break

    if error and not last_content:
        return {"answer": error, "type": "error", "ingested_count": ingested, "error": error}

    if ingested > 0:
        answer_type = "ingest"
    elif last_content and ("找到" in last_content or "篇相关论文" in last_content):
        answer_type = "search"
    else:
        answer_type = "info"

    return {
        "answer":         last_content or "操作已完成。",
        "type":           answer_type,
        "ingested_count": ingested,
        "error":          error,
    }


# ── API 端点 ───────────────────────────────────────────────────

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """向 Agent 发送消息，返回最终回答（非流式）。"""
    try:
        from opendetect_ai.graph import chat as graph_chat
        state = await asyncio.to_thread(
            graph_chat, req.query, req.thread_id, req.user_id, True
        )
        interrupts = state.get("_interrupts", []) if isinstance(state, dict) else []
        if interrupts:
            return {**interrupts[0], "type": "interrupt"}
        return _extract_answer(state if isinstance(state, dict) else {})
    except EnvironmentError as e:
        return JSONResponse(status_code=500,
            content={"answer": f"环境配置错误: {e}", "type": "error", "error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500,
            content={"answer": f"处理失败: {e}", "type": "error", "error": str(e)})


@app.get("/api/threads")
async def get_threads():
    try:
        from opendetect_ai.graph import list_threads
        threads = await asyncio.to_thread(list_threads)
        return {"threads": threads or []}
    except Exception as e:
        return {"threads": [], "error": str(e)}


@app.get("/api/papers")
async def get_papers():
    try:
        from opendetect_ai.tools.rag_tool import list_ingested_papers
        papers = await asyncio.to_thread(list_ingested_papers.invoke, {})
        if papers and isinstance(papers[0], dict) and "message" in papers[0]:
            return {"papers": [], "total": 0}
        return {"papers": papers, "total": len(papers)}
    except Exception as e:
        return {"papers": [], "total": 0, "error": str(e)}


@app.post("/api/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    title: str      = Form(""),
    authors: str    = Form(""),
    published: str  = Form(""),
    thread_id: str  = Form("upload"),
    user_id: str    = Form("default"),
):
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只支持 PDF 文件")
    allowed_mime = {"application/pdf", "application/x-pdf", "application/octet-stream", ""}
    if (file.content_type or "").lower() not in allowed_mime:
        raise HTTPException(status_code=400, detail="上传文件的 MIME 类型不是 PDF")
    max_bytes = OPENDETECT_MAX_PDF_MB * 1024 * 1024
    tmp_path = ""
    try:
        total = 0
        header = b""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp_path = tmp.name
            while chunk := await file.read(1024 * 1024):
                if len(header) < 5:
                    header = (header + chunk)[:5]
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"PDF 不能超过 {OPENDETECT_MAX_PDF_MB} MB",
                    )
                tmp.write(chunk)
        if header != b"%PDF-":
            raise HTTPException(status_code=400, detail="文件内容不是有效的 PDF")

        from opendetect_ai.tools.rag_tool import ingest_local_pdf
        from opendetect_ai.approval import record_explicit_approval
        effective_title = title.strip() or Path(filename).stem
        file_digest = hashlib.sha256(Path(tmp_path).read_bytes()).hexdigest()
        await asyncio.to_thread(
            record_explicit_approval,
            action="upload_pdf",
            payload={"title": effective_title, "filename": filename, "sha256": file_digest},
            reason="用户主动选择并上传 PDF，将修改共享论文知识库。",
            thread_id=thread_id,
            user_id=user_id,
            idempotency_key=file_digest,
        )
        result = await asyncio.to_thread(ingest_local_pdf.invoke, {
            "file_path": tmp_path, "title": effective_title,
            "authors": authors, "published": published,
        })
        return {
            "status":  result.get("status", "error"),
            "chunks":  result.get("chunks", 0),
            "pages":   result.get("pages", 0),
            "tables":  result.get("tables", 0),
            "figures": result.get("figures", 0),
            "reindexed": result.get("reindexed", False),
            "skipped": result.get("skipped", False),
            "title":   effective_title,
            "message": result.get("message", ""),
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _pending_interrupts(chat_graph, config) -> list[dict]:
    """检查该会话是否停在 interrupt 处，返回中断 payload 列表（HITL 确认卡片数据）。"""
    snap = chat_graph.get_state(config)
    out = []
    for task in getattr(snap, "tasks", []) or []:
        for intr in (getattr(task, "interrupts", None) or []):
            val = getattr(intr, "value", None)
            if isinstance(val, dict):
                out.append(val)
    return out


async def _sse_run(chat_graph, config, graph_input, thread_id):
    """
    驱动 graph.stream，把 values / messages / interrupt / done 转成 SSE 事件串。
    首轮（input=state）与 HITL 恢复（input=Command(resume=...)）共用同一套逻辑。
    """
    q: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _run_stream():
        try:
            last_state = {}
            for stream_mode, payload in chat_graph.stream(
                graph_input, config=config, stream_mode=["values", "messages"]
            ):
                if stream_mode == "values":
                    last_state = payload
                    loop.call_soon_threadsafe(q.put_nowait, ("snapshot", payload))
                elif stream_mode == "messages":
                    # AnswerGuard 在生成完成后才运行。这里故意不向客户端发送模型草稿，
                    # 避免未经核验的数字/结论被展示或复制。
                    continue
            loop.call_soon_threadsafe(q.put_nowait, ("done", last_state))
        except Exception as exc:
            loop.call_soon_threadsafe(q.put_nowait, ("error", str(exc)))

    stream_task = asyncio.create_task(asyncio.to_thread(_run_stream))
    last_snapshot: dict = {}
    prev_nodes: set = set()
    prev_ingested_count = None   # None=首帧只做基线，避免库已非空时误报「入库中」

    try:
        while True:
            kind, data = await q.get()

            if kind == "snapshot":
                last_snapshot = data if isinstance(data, dict) else {}
                for think_msg in drain_queue(thread_id):
                    yield _sse({"type": "think", "message": think_msg})

                cur_ingested = last_snapshot.get("ingested_count", 0) or 0
                if last_snapshot.get("rag_answer") and "rag" not in prev_nodes:
                    prev_nodes.add("rag")
                    yield _sse({"type": "step", "agent": "rag", "message": _STEP_MESSAGES["rag"]})
                elif last_snapshot.get("final_report") and "report" not in prev_nodes:
                    prev_nodes.add("report")
                    yield _sse({"type": "step", "agent": "report", "message": _STEP_MESSAGES["report"]})
                elif prev_ingested_count is not None and cur_ingested > prev_ingested_count and "ingest" not in prev_nodes:
                    prev_nodes.add("ingest")
                    yield _sse({"type": "step", "agent": "ingest", "message": _STEP_MESSAGES["ingest"]})
                elif last_snapshot.get("search_results") and "search" not in prev_nodes:
                    prev_nodes.add("search")
                    yield _sse({"type": "step", "agent": "search", "message": _STEP_MESSAGES["search"]})
                prev_ingested_count = cur_ingested

            elif kind == "token":
                yield _sse({"type": "token", "text": data})

            elif kind == "done":
                last_snapshot = data if isinstance(data, dict) else last_snapshot
                break

            else:  # error
                yield _sse({"type": "error", "answer": str(data)})
                await stream_task
                return
    except asyncio.CancelledError:
        # to_thread 中的同步 graph.stream 无法被取消；等它安全落盘后再让外层释放会话锁。
        await asyncio.shield(stream_task)
        raise

    await stream_task
    for think_msg in drain_queue(thread_id):
        yield _sse({"type": "think", "message": think_msg})

    # ── HITL：若图停在 interrupt 处，推送确认卡片并暂停（等待 /resume）──
    interrupts = await asyncio.to_thread(_pending_interrupts, chat_graph, config)
    if interrupts:
        # 注意 payload 里带有自己的 action 字段，不能让它覆盖 SSE 的 type
        yield _sse({**interrupts[0], "type": "interrupt"})
        cleanup_queue(thread_id)
        return

    cleanup_queue(thread_id)

    from opendetect_ai.graph import spawn_profile_extraction
    answer = _extract_answer(last_snapshot)
    spawn_profile_extraction(last_snapshot.get("messages", []),
                             last_snapshot.get("user_id", "default"))
    if answer.get("type") in {"rag", "report"} and answer.get("answer"):
        verified_text = answer["answer"]
        for start in range(0, len(verified_text), 96):
            yield _sse({"type": "token", "text": verified_text[start:start + 96], "verified": True})
    done_payload = {"type": "done", "msg_type": answer.get("type", "info"),
                    **{k: v for k, v in answer.items() if k != "type"}}
    yield _sse(done_payload)


@app.post("/api/chat/stream")
async def chat_stream_endpoint(req: ChatRequest):
    """向 Agent 发送消息，以 SSE 流式返回进度 / token / 中断 / 最终回答。"""

    async def generate():
        execution_lock = None
        lock_acquired = False
        try:
            from opendetect_ai.graph import _get_chat_graph, build_turn_input, get_thread_lock
            from opendetect_ai.env_utils import validate_env
            validate_env()

            chat_graph = _get_chat_graph()
            execution_lock = get_thread_lock(req.thread_id)
            await asyncio.to_thread(execution_lock.acquire)
            lock_acquired = True
            config = {
                "configurable": {"thread_id": req.thread_id, "hitl": True},
                "recursion_limit": 20,
            }
            input_state = await asyncio.to_thread(
                build_turn_input, chat_graph, config, req.query, req.thread_id, req.user_id
            )
            input_state["hitl"] = True   # 仅 Web 路径开启 HITL（CLI run/chat 不开）
            async for chunk in _sse_run(chat_graph, config, input_state, req.thread_id):
                yield chunk

        except EnvironmentError as exc:
            yield _sse({"type": "error", "answer": f"环境配置错误: {exc}"})
        except Exception as exc:
            yield _sse({"type": "error", "answer": f"处理失败: {exc}"})
        finally:
            if execution_lock is not None and lock_acquired:
                execution_lock.release()

    return StreamingResponse(generate(), media_type="text/event-stream", headers=SSE_HEADERS)


@app.post("/api/chat/resume")
async def chat_resume_endpoint(req: ResumeRequest):
    """HITL 恢复：用户确认入库选择后，从 interrupt 处继续执行并继续流式。"""

    async def generate():
        execution_lock = None
        lock_acquired = False
        try:
            from opendetect_ai.graph import _get_chat_graph, get_thread_lock
            from langgraph.types import Command
            chat_graph = _get_chat_graph()
            execution_lock = get_thread_lock(req.thread_id)
            await asyncio.to_thread(execution_lock.acquire)
            lock_acquired = True
            config = {
                "configurable": {"thread_id": req.thread_id, "hitl": True},
                "recursion_limit": 20,
            }
            pending = await asyncio.to_thread(_pending_interrupts, chat_graph, config)
            if not pending:
                yield _sse({"type": "error", "answer": "该会话没有待处理审批，可能已完成或已恢复。"})
                return
            pending_user = pending[0].get("user_id")
            if pending_user and pending_user != req.user_id:
                yield _sse({"type": "error", "answer": "当前用户无权恢复该审批。"})
                return
            async for chunk in _sse_run(
                chat_graph, config, Command(resume=req.selection), req.thread_id
            ):
                yield chunk
        except Exception as exc:
            yield _sse({"type": "error", "answer": f"处理失败: {exc}"})
        finally:
            if execution_lock is not None and lock_acquired:
                execution_lock.release()

    return StreamingResponse(generate(), media_type="text/event-stream", headers=SSE_HEADERS)


@app.get("/api/approvals")
async def get_approvals(user_id: str = "default", limit: int = 50):
    """查询当前用户的审批审计记录。"""
    from opendetect_ai.approval import list_approvals
    approvals = await asyncio.to_thread(list_approvals, user_id, limit)
    return {"approvals": approvals, "total": len(approvals)}


# ── 用户长期记忆 API ──────────────────────────────────────────
@app.get("/api/user-profile")
async def get_user_profile(user_id: str = "default"):
    """获取指定用户的长期偏好记忆。"""
    try:
        from opendetect_ai.user_memory import (
            get_memory_settings,
            list_memory_entries,
            load_user_profile,
        )
        profile = await asyncio.to_thread(load_user_profile, user_id)
        settings = await asyncio.to_thread(get_memory_settings, user_id)
        entries = await asyncio.to_thread(list_memory_entries, user_id)
        return {"profile": profile, "entries": entries, "settings": settings, "empty": not profile}
    except Exception as e:
        return {"profile": {}, "empty": True, "error": str(e)}


@app.delete("/api/user-profile")
async def clear_user_profile(user_id: str = "default"):
    """清除指定用户的长期偏好记忆（重置画像）。"""
    try:
        from opendetect_ai.user_memory import _get_db_path, _ensure_table
        import sqlite3
        def _clear():
            conn = sqlite3.connect(_get_db_path(), check_same_thread=False)
            _ensure_table(conn)
            conn.execute("DELETE FROM user_profile WHERE user_id = ?", (user_id,))
            conn.commit()
            conn.close()
        await asyncio.to_thread(_clear)
        return {"status": "ok", "message": "用户记忆已清除"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.patch("/api/user-profile/settings")
async def update_memory_settings(req: MemorySettingsRequest):
    """启用/关闭长期记忆，并设置可选 TTL。"""
    try:
        from opendetect_ai.user_memory import set_memory_settings
        settings = await asyncio.to_thread(
            set_memory_settings,
            req.user_id,
            enabled=req.enabled,
            ttl_days=req.ttl_days,
        )
        return {"status": "ok", "settings": settings}
    except Exception as exc:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})


# ── 静态前端 ───────────────────────────────────────────────────
@app.get("/favicon.ico")
async def favicon():
    favicon_path = Path(__file__).parent / "frontend" / "favicon.ico"
    if favicon_path.exists():
        return FileResponse(str(favicon_path))
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#b8621a"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>'
    from fastapi.responses import Response
    return Response(content=svg, media_type="image/svg+xml")


@app.get("/")
async def serve_frontend():
    index = Path(__file__).parent / "frontend" / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse(status_code=404,
        content={"error": "前端文件未找到，请确保 frontend/index.html 存在。"})
