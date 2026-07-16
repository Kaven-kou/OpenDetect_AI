"""单元测试：图能编译、状态工厂、纯函数工具（不依赖网络 / API Key）。"""

from langgraph.pregel import Pregel

from opendetect_ai.graph import build_graph, route
from opendetect_ai.state import create_initial_state, AgentState
from opendetect_ai.agents.supervisor import VALID_NEXT
from opendetect_ai.context_utils import build_context_str
from langchain_core.messages import HumanMessage, AIMessage


def test_graph_compiles() -> None:
    """主图应能无 checkpointer 编译成 Pregel 可执行图。"""
    graph = build_graph()
    assert isinstance(graph, Pregel)


def test_route_reads_next_field() -> None:
    """route() 直接读取 state.next，缺省回退 FINISH。"""
    assert route({"next": "search"}) == "search"
    assert route({}) == "FINISH"


def test_initial_state_has_expected_fields() -> None:
    """初始状态工厂应返回干净、字段齐全的 AgentState。"""
    state = create_initial_state("帮我搜索 ViT 论文")
    assert isinstance(state, AgentState)
    assert state["user_query"] == "帮我搜索 ViT 论文"
    assert state["next"] == "supervisor"
    assert state["search_attempted"] is False
    for key in ("messages", "search_results", "papers_to_ingest", "failed_papers"):
        assert state[key] == []
    assert state["ingested_count"] == 0


def test_supervisor_valid_next_targets() -> None:
    """路由白名单应覆盖全部子 Agent 与 FINISH。"""
    assert VALID_NEXT == {"search", "ingest", "rag", "report", "FINISH"}


def test_build_context_str_extracts_recent_turns() -> None:
    """短期记忆：应提取已完成的历史轮，且不含当前未回答轮与路由日志。"""
    messages = [
        HumanMessage(content="帮我搜索 ViT 的论文"),
        AIMessage(content="Supervisor 决策: search — 需要搜索"),   # 路由日志，应过滤
        AIMessage(content="已找到并入库《An Image is Worth 16x16 Words》"),
        HumanMessage(content="它和 CNN 相比有什么优势？"),         # 当前轮，未回答
    ]
    ctx = build_context_str(messages)
    assert "帮我搜索 ViT 的论文" in ctx
    assert "已找到并入库" in ctx
    assert "Supervisor 决策" not in ctx        # 路由日志被过滤
    assert "它和 CNN 相比" not in ctx          # 当前未回答轮不进上下文


def test_build_context_str_empty() -> None:
    assert build_context_str([]) == ""
