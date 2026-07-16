"""集成测试：真实驱动 OpenDetect 主图。

需要有效的 LLM Key 才运行；未配置时自动跳过（不会让 CI 变红）。
用一句纯闲聊（"你好"）触发 Supervisor → FINISH 的最短路径，
只消耗 1 次 LLM 调用，不触发搜索 / 网络重活。
"""

import os

import pytest

from opendetect_ai.graph import build_graph
from opendetect_ai.state import create_initial_state

if not os.getenv("OPENDETECT_LLM_API_KEY") and not os.getenv("DEEPSEEK_API_KEY"):
    pytest.skip(
        "Set OPENDETECT_LLM_API_KEY to run integration tests.",
        allow_module_level=True,
    )


def test_chitchat_routes_to_finish() -> None:
    """闲聊问候应被 Supervisor 识别为一般对话并收束到 FINISH。"""
    graph = build_graph()
    state = create_initial_state("你好，你是谁？")
    result = graph.invoke(state, config={"recursion_limit": 10})

    # 闲聊不应触发搜索 / 入库
    assert result.get("search_attempted") is False
    assert result.get("ingested_count", 0) == 0
    # 应产出一条面向用户的回复（direct_answer 或 messages 尾条）
    answer = result.get("direct_answer") or (
        result["messages"][-1].content if result.get("messages") else ""
    )
    assert answer.strip()
