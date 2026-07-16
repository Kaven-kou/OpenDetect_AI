"""
LangGraph 主工作流 —— OpenDetect_AI
把 Supervisor + 四个子 Agent 串成完整的多智能体图。
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END
import sqlite3         # ← 新增
from langgraph.checkpoint.sqlite import SqliteSaver          # ← 新增

from opendetect_ai.tools.rag_tool import list_ingested_papers
from opendetect_ai.state import AgentState, create_initial_state
from opendetect_ai.agents.supervisor import supervisor_node
from opendetect_ai.agents.search import search_node
from opendetect_ai.agents.ingest import ingest_node
from opendetect_ai.agents.rag import rag_node
from opendetect_ai.agents.report import report_node
from opendetect_ai.agents.verify import verify_node
from opendetect_ai.env_utils import validate_env, CHROMA_PERSIST_DIR, OPENDETECT_LLM_MODEL, OPENDETECT_LLM_BASE_URL, OPENDETECT_LLM_API_KEY  

import os
_DB_PATH = os.path.join(os.path.dirname(CHROMA_PERSIST_DIR), "chat_history.db")

# ── 路由函数：读取 state.next，返回下一个节点名 ────────────────
def route(state: AgentState) -> str:
    """Supervisor 决策后，根据 state.next 跳转到对应节点。"""
    return state.get("next", "FINISH")


def route_after_search(state: AgentState) -> str:
    """
    Search 完成后的确定性边界：有待入库论文就直连 ingest，省掉一次 Supervisor LLM 调用；
    没搜到（或都处理过了）才回 Supervisor 收尾。
    —— LLM 只负责真正需要理解意图的决策，确定性转移交给状态机。
    """
    pending = [p for p in state.get("papers_to_ingest", []) if not p.ingested]
    return "ingest" if pending else "supervisor"


# ── 构建 LangGraph ─────────────────────────────────────────────
def build_graph(checkpointer=None) -> StateGraph:
    """
    构建并编译多智能体工作流图。

    图结构：
        __start__
            ↓
        supervisor  ←──────────────┐
            ↓ (条件路由)            │
      ┌─────┴──────┬─────┬────────┐│
    search       ingest  rag    report
      └────────────┘      │       │
            ↓             ↓       ↓
        supervisor       END     END
            ↓ (next == FINISH)
           END
    """
    builder = StateGraph(AgentState)

    # ── 注册节点 ───────────────────────────────────────────────
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("search",     search_node)
    builder.add_node("ingest",     ingest_node)
    builder.add_node("rag",        rag_node)
    builder.add_node("report",     report_node)
    builder.add_node("verify",     verify_node)

    # ── 入口：从 supervisor 开始 ───────────────────────────────
    builder.set_entry_point("supervisor")

    # ── 条件路由：supervisor → 各子 Agent ─────────────────────
    builder.add_conditional_edges(
        "supervisor",
        route,
        {
            "search":  "search",
            "ingest":  "ingest",
            "rag":     "rag",
            "report":  "report",
            "FINISH":  END,
        },
    )

    # ── Search 后确定性边界：有待入库论文直连 ingest，否则回 supervisor ──
    builder.add_conditional_edges(
        "search",
        route_after_search,
        {"ingest": "ingest", "supervisor": "supervisor"},
    )
    # Ingest 完成后回 supervisor，由它按用户意图决定 rag / report / FINISH
    builder.add_edge("ingest",  "supervisor")

    # RAG 生成回答后经 Verifier 校验事实性再结束；Report 直接结束。
    builder.add_edge("rag",     "verify")
    builder.add_edge("verify",  END)
    builder.add_edge("report",  END)

    return builder.compile(checkpointer=checkpointer)        # ← 传入 checkpointer

# ── 单轮图（无记忆，原有 run() 用）────────────────────────────
graph = build_graph()


# ── 多轮图（SQLite 持久化）─────────────────────────────────────

_chat_graph = None   # ← 全局单例

def _get_chat_graph():
    """
    获取带 SQLite Checkpointer 的持久化图实例（单例）。
    用直接传 sqlite3 连接的方式，避免 from_conn_string 的上下文管理器问题。
    """
    global _chat_graph
    if _chat_graph is None:
        os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
        conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        _chat_graph = build_graph(checkpointer=checkpointer)
    return _chat_graph


def _print_output(final_state: dict) -> None:
    """统一打印最终输出。"""
    print(f"\n{'='*50}")
    print("工作流完成，最终输出：")
    print(f"{'='*50}")
    if final_state.get("rag_answer"):
        print(f"\n[RAG 回答]\n{final_state['rag_answer']}")
    if final_state.get("final_report"):
        print(f"\n[综述报告]\n{final_state['final_report']}")
    if final_state.get("error"):
        print(f"\n[错误信息]\n{final_state['error']}")
    if (final_state.get("ingested_count", 0) > 0
            and not final_state.get("rag_answer")
            and not final_state.get("final_report")):
        papers = list_ingested_papers.invoke({})
        if papers and "message" not in papers[0]:
            print("\n[已入库论文列表]")
            for i, p in enumerate(papers, 1):
                print(f"  {i}. {p.get('title')} ({p.get('published')}) arxiv:{p.get('arxiv_id') or '无'}")


# ── 共享辅助：避免 run() / chat() / api.py 三处重复 ────────────
def _live_ingested_count() -> int:
    """实时从向量库读取已入库论文数（去重后）。空库返回 0。"""
    existing = list_ingested_papers.invoke({})
    return 0 if (existing and "message" in existing[0]) else len(existing)


def build_turn_input(chat_graph, config: dict, user_query: str, thread_id: str,
                     user_id: str = "default") -> dict:
    """
    构造「一轮对话」的输入 state（供多轮 chat 与 Web SSE 共用）。

    - 新会话：走 create_initial_state 初始化全部字段。
    - 续接会话：复用 checkpoint 中的 failed_papers，重置本轮临时字段；
      **两个分支都实时从向量库刷新 ingested_count**，避免跨会话入库后计数偏低
      （原 api.py 续接分支漏设该字段，会退回 checkpoint 旧值 —— 已修复）。
    """
    current = chat_graph.get_state(config)
    already_ingested = _live_ingested_count()

    if current.values:
        return {
            "user_query":       user_query,
            "next":             "supervisor",
            "search_attempted": False,
            "rag_answer":       "",
            "final_report":     "",
            "direct_answer":    "",
            "error":            "",
            "search_results":   [],
            "papers_to_ingest": [],
            "failed_papers":    current.values.get("failed_papers", []),
            "ingested_count":   already_ingested,
            "thread_id":        thread_id,
            "user_id":          user_id,
        }

    state = create_initial_state(user_query)
    state["ingested_count"] = already_ingested
    state["direct_answer"] = ""
    state["thread_id"] = thread_id
    state["user_id"] = user_id
    return state


def spawn_profile_extraction(messages: list, user_id: str = "default") -> None:
    """对话结束后在后台线程异步提取用户长期偏好（按 user_id 隔离），失败静默、不阻塞主流程。"""
    if not messages:
        return
    try:
        from opendetect_ai.user_memory import extract_and_save_profile
        import threading
        threading.Thread(
            target=extract_and_save_profile,
            args=(messages, OPENDETECT_LLM_MODEL, OPENDETECT_LLM_BASE_URL,
                  OPENDETECT_LLM_API_KEY, user_id),
            daemon=True,
        ).start()
    except Exception:
        pass


def run(user_query: str) -> dict:
    """单轮运行，每次从空白状态开始（无持久化，原有接口语义保持不变）。"""
    validate_env()
    already_ingested = _live_ingested_count()
    initial_state = create_initial_state(user_query)
    initial_state["ingested_count"] = already_ingested
    initial_state["thread_id"] = "run_default"  # 单轮模式用固定 thread_id

    print(f"\n{'='*50}")
    print(f"用户问题: {user_query}")
    print(f"向量库已有论文: {already_ingested} 篇")
    print(f"{'='*50}\n")

    final_state = graph.invoke(initial_state, config={"recursion_limit": 20})
    _print_output(final_state)

    spawn_profile_extraction(final_state.get("messages", []), final_state.get("user_id", "default"))
    return final_state


def chat(user_query: str, thread_id: str = "default", user_id: str = "default") -> dict:
    """
    多轮对话入口。同一 thread_id 内的对话保留完整历史状态。

    Args:
        user_query: 本轮用户输入
        thread_id:  会话 ID，相同 ID 的对话共享状态（默认 "default"）

    用法：
        # 第一轮：搜索并入库
        chat('帮我搜索 ViT 相关论文', thread_id='session_1')
        # 第二轮：直接基于已入库内容提问，不重新搜索
        chat('ViT 和 CNN 相比有什么优势？', thread_id='session_1')
        # 第三轮：继续追问
        chat('Swin Transformer 做了哪些改进？', thread_id='session_1')
    """
    validate_env()
    chat_graph = _get_chat_graph()
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 20,
    }

    # ── 构造本轮输入（新会话 / 续接会话，共享逻辑）─────────────
    current = chat_graph.get_state(config)
    input_state = build_turn_input(chat_graph, config, user_query, thread_id, user_id)
    already_ingested = input_state["ingested_count"]

    print(f"\n{'='*50}")
    if current.values:
        print(f"[会话 {thread_id}] 第 {len(current.values.get('messages', [])) // 2 + 1} 轮对话（向量库实时计数: {already_ingested} 篇）")
    else:
        print(f"[会话 {thread_id}] 新会话开始")
    print(f"用户问题: {user_query}")
    print(f"向量库已有论文: {already_ingested} 篇")
    print(f"{'='*50}\n")

    final_state = chat_graph.invoke(input_state, config=config)
    _print_output(final_state)

    spawn_profile_extraction(final_state.get("messages", []), user_id)
    return final_state


def list_threads() -> list[str]:
    """列出所有存在历史记录的会话 ID。"""
    chat_graph = _get_chat_graph()
    try:
        # SqliteSaver 支持列出所有 checkpoint
        checkpointer = chat_graph.checkpointer
        threads = list({
            item.config["configurable"]["thread_id"]
            for item in checkpointer.list(None)
        })
        return threads
    except Exception:
        return []