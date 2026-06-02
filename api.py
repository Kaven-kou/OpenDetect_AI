"""
OpenDetect AI — FastAPI 后端包装层
将 LangGraph 工作流封装为 HTTP API，供前端调用。

运行方式（在项目根目录执行）:
    pip install fastapi uvicorn python-multipart
    uvicorn api:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from opendetect_ai.tools.progress import drain_queue, cleanup_queue

app = FastAPI(title="OpenDetect AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 请求模型 ───────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str
    thread_id: str


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
        return {"answer": rag_answer, "type": "rag", "ingested_count": ingested, "error": ""}

    if final_report:
        return {"answer": final_report, "type": "report", "ingested_count": ingested, "error": ""}

    if direct_answer:
        return {"answer": direct_answer, "type": "info", "ingested_count": ingested, "error": ""}

    # 从消息列表里逆序找最后一条有实质内容的消息
    last_content = ""
    for msg in reversed(messages):
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
        state = await asyncio.to_thread(graph_chat, req.query, req.thread_id)
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
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只支持 PDF 文件")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    try:
        from opendetect_ai.tools.rag_tool import ingest_local_pdf
        effective_title = title.strip() or Path(file.filename).stem
        result = await asyncio.to_thread(ingest_local_pdf.invoke, {
            "file_path": tmp_path, "title": effective_title,
            "authors": authors, "published": published,
        })
        return {
            "status":  result.get("status", "error"),
            "chunks":  result.get("chunks", 0),
            "skipped": result.get("skipped", False),
            "title":   effective_title,
            "message": result.get("message", ""),
        }
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/api/chat/stream")
async def chat_stream_endpoint(req: ChatRequest):
    """向 Agent 发送消息，以 SSE 流式返回进度事件和最终回答。"""

    async def generate():
        try:
            from opendetect_ai.graph import _get_chat_graph
            from opendetect_ai.state import create_initial_state
            from opendetect_ai.tools.rag_tool import list_ingested_papers
            from opendetect_ai.env_utils import validate_env
            validate_env()

            chat_graph = _get_chat_graph()
            config = {
                "configurable": {"thread_id": req.thread_id},
                "recursion_limit": 20,
            }

            def _build_input():
                current = chat_graph.get_state(config)
                if current.values:
                    prev_failed = current.values.get("failed_papers", [])
                    return {
                        "user_query":       req.query,
                        "next":             "supervisor",
                        "search_attempted": False,
                        "rag_answer":       "",
                        "final_report":     "",
                        "direct_answer":    "",
                        "error":            "",
                        "search_results":   [],
                        "papers_to_ingest": [],
                        "failed_papers":    prev_failed,
                        "thread_id":        req.thread_id,
                    }
                existing = list_ingested_papers.invoke({})
                already_ingested = 0 if (existing and "message" in existing[0]) else len(existing)
                state = create_initial_state(req.query)
                state["ingested_count"] = already_ingested
                state["direct_answer"] = ""
                state["thread_id"] = req.thread_id
                return state

            input_state = await asyncio.to_thread(_build_input)

            # Queue 桥接同步 stream 与异步 SSE
            q: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_running_loop()

            # stream(mode="values") 每步返回执行该节点后的完整 state 快照
            # 比 mode="updates"（默认）更可靠：直接拿最后一个完整快照即可
            def _run_stream():
                try:
                    last_state = {}
                    for state_snapshot in chat_graph.stream(
                        input_state, config=config, stream_mode="values"
                    ):
                        last_state = state_snapshot
                        # 同时把节点名传出去用于进度提示
                        loop.call_soon_threadsafe(q.put_nowait, ("snapshot", state_snapshot))
                    loop.call_soon_threadsafe(q.put_nowait, ("done", last_state))
                except Exception as exc:
                    loop.call_soon_threadsafe(q.put_nowait, ("error", str(exc)))

            stream_task = asyncio.create_task(asyncio.to_thread(_run_stream))

            last_snapshot: dict = {}
            prev_nodes: set = set()
            prev_ingested_count: int = 0  # 追踪 ingested_count 的变化，只有真正增加才说明 ingest 节点执行了

            while True:
                kind, data = await q.get()

                if kind == "snapshot":
                    last_snapshot = data if isinstance(data, dict) else {}
                    # 轮询进度队列，把 Agent 推送的进度消息发给前端
                    for think_msg in drain_queue(req.thread_id):
                        yield f"data: {json.dumps({'type': 'think', 'message': think_msg}, ensure_ascii=False)}\n\n"

                    cur_ingested = last_snapshot.get("ingested_count", 0) or 0

                    # 节点推断：只根据本轮真实发生的变化触发，避免历史数据误触发
                    if last_snapshot.get("rag_answer") and "rag" not in prev_nodes:
                        prev_nodes.add("rag")
                        yield f"data: {json.dumps({'type': 'step', 'agent': 'rag', 'message': _STEP_MESSAGES['rag']}, ensure_ascii=False)}\n\n"
                    elif last_snapshot.get("final_report") and "report" not in prev_nodes:
                        prev_nodes.add("report")
                        yield f"data: {json.dumps({'type': 'step', 'agent': 'report', 'message': _STEP_MESSAGES['report']}, ensure_ascii=False)}\n\n"
                    elif (cur_ingested > prev_ingested_count and "ingest" not in prev_nodes):
                        # 只有 ingested_count 本轮真正增加了，才推送入库提示
                        prev_nodes.add("ingest")
                        yield f"data: {json.dumps({'type': 'step', 'agent': 'ingest', 'message': _STEP_MESSAGES['ingest']}, ensure_ascii=False)}\n\n"
                    elif last_snapshot.get("search_results") and "search" not in prev_nodes:
                        prev_nodes.add("search")
                        yield f"data: {json.dumps({'type': 'step', 'agent': 'search', 'message': _STEP_MESSAGES['search']}, ensure_ascii=False)}\n\n"

                    prev_ingested_count = cur_ingested  # 更新基准值

                elif kind == "done":
                    # data 是最后一个完整 state，直接用
                    last_snapshot = data if isinstance(data, dict) else last_snapshot
                    break

                else:  # error
                    yield f"data: {json.dumps({'type': 'error', 'answer': str(data)}, ensure_ascii=False)}\n\n"
                    return

            await stream_task
            # drain 最后可能残留的进度消息
            for think_msg in drain_queue(req.thread_id):
                yield f"data: {json.dumps({'type': 'think', 'message': think_msg}, ensure_ascii=False)}\n\n"
            cleanup_queue(req.thread_id)

            answer = _extract_answer(last_snapshot)

            # 对话结束后异步提取用户偏好（不阻塞 SSE 响应）
            try:
                from opendetect_ai.user_memory import extract_and_save_profile
                from opendetect_ai.env_utils import OPENDETECT_LLM_MODEL, OPENDETECT_LLM_BASE_URL, OPENDETECT_LLM_API_KEY
                import threading
                threading.Thread(
                    target=extract_and_save_profile,
                    args=(last_snapshot.get("messages", []),
                          OPENDETECT_LLM_MODEL, OPENDETECT_LLM_BASE_URL, OPENDETECT_LLM_API_KEY),
                    daemon=True,
                ).start()
            except Exception:
                pass

            # answer 里已有 type 字段（'rag'/'info' 等），用 msg_type 区分消息类型
            # 避免与 SSE 协议的 type:'done' 冲突
            done_payload = {"type": "done", "msg_type": answer.get("type", "info"), **{k: v for k, v in answer.items() if k != "type"}}
            yield f"data: {json.dumps(done_payload, ensure_ascii=False)}\n\n"

        except EnvironmentError as exc:
            yield f"data: {json.dumps({'type': 'error', 'answer': f'环境配置错误: {exc}'}, ensure_ascii=False)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'answer': f'处理失败: {exc}'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── 用户长期记忆 API ──────────────────────────────────────────
@app.get("/api/user-profile")
async def get_user_profile():
    """获取用户长期偏好记忆。"""
    try:
        from opendetect_ai.user_memory import load_user_profile
        profile = await asyncio.to_thread(load_user_profile)
        return {"profile": profile, "empty": not profile}
    except Exception as e:
        return {"profile": {}, "empty": True, "error": str(e)}


@app.delete("/api/user-profile")
async def clear_user_profile():
    """清除用户长期偏好记忆（重置画像）。"""
    try:
        from opendetect_ai.user_memory import _get_db_path
        import sqlite3
        def _clear():
            conn = sqlite3.connect(_get_db_path(), check_same_thread=False)
            conn.execute("DELETE FROM user_profile")
            conn.commit()
            conn.close()
        await asyncio.to_thread(_clear)
        return {"status": "ok", "message": "用户记忆已清除"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


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