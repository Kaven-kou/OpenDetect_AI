"""
MCP Client —— OpenDetect_AI
通过 langchain-mcp-adapters 连接远程 ArXiv MCP 与本地 OpenAlex MCP，
把 MCP 工具转成 LangChain Tool 供 Search Agent 调用。
"""

import asyncio
import json
import sys
from typing import Any
from pathlib import Path
from langchain_core.tools import ToolException
from langchain_mcp_adapters.client import MultiServerMCPClient

from opendetect_ai.env_utils import OPENDETECT_ARXIV_MCP_URL

_SERVER_SCRIPT = str(Path(__file__).parent / "openalex_mcp_server.py")

_MCP_CONFIG = {
    "arxiv": {
        "url": OPENDETECT_ARXIV_MCP_URL,
        "transport": "streamable_http",
    },
    "openalex": {
        "command":   sys.executable,
        "args":      [_SERVER_SCRIPT],
        "transport": "stdio",
    }
}


def _parse_mcp_response(raw: Any):
    """
    解析 MCP 工具返回值。
    MCP 返回格式: [{"type": "text", "text": "{...json...}", "id": "..."}]
    需要提取 text 字段并反序列化。
    """
    if isinstance(raw, tuple) and raw:
        raw = raw[0]

    # 已经是 dict/list（非 MCP 格式）直接返回
    if isinstance(raw, (dict, list)) and not (
        isinstance(raw, list) and raw and isinstance(raw[0], dict) and "type" in raw[0]
    ):
        return raw

    # MCP 标准格式：合并 text 块；如果是 JSON 则反序列化。
    if isinstance(raw, list):
        texts = []
        for item in raw:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text", ""))
        if texts:
            parsed_items = []
            all_json = True
            for text in texts:
                try:
                    parsed_items.append(json.loads(text))
                except json.JSONDecodeError:
                    all_json = False
                    break
            if all_json:
                return parsed_items[0] if len(parsed_items) == 1 else parsed_items

            text = "\n".join(t for t in texts if t)
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text

    return raw


# ── 工具列表缓存（按 server_name 缓存，避免每次重建 client）──────
_tools_cache: dict[str | None, list] = {}
_cache_lock = __import__("threading").Lock()


async def _get_tools_cached(server_name: str | None = None) -> list:
    """
    带缓存的工具列表获取。
    首次调用时建立连接并缓存，后续直接复用。
    MCP stdio server 进程在首次连接时启动，后续不重启。
    """
    cache_key = server_name
    with _cache_lock:
        if cache_key in _tools_cache:
            return _tools_cache[cache_key]

    client = MultiServerMCPClient(_MCP_CONFIG)
    tools = await client.get_tools(server_name=server_name)

    with _cache_lock:
        _tools_cache[cache_key] = tools
    return tools


def invalidate_tools_cache(server_name: str | None = None) -> None:
    """连接异常时手动清除缓存，下次调用会重新建立连接。"""
    with _cache_lock:
        if server_name is None:
            _tools_cache.clear()
        else:
            _tools_cache.pop(server_name, None)


async def _call_tool_async(tool_name: str, tool_input: dict, server_name: str | None = None):
    try:
        tools = await _get_tools_cached(server_name)
    except Exception as exc:
        # 获取工具列表失败时清除缓存，下次重试
        invalidate_tools_cache(server_name)
        return {"error": f"MCP 连接失败: {exc}"}

    tool = next((t for t in tools if t.name == tool_name), None)
    if tool is None:
        scope = f"'{server_name}' " if server_name else ""
        return {"error": f"MCP {scope}工具 '{tool_name}' 不存在，可用: {[t.name for t in tools]}"}

    try:
        raw = await tool.ainvoke(tool_input)
    except ToolException as exc:
        return {"error": str(exc)}
    except Exception as exc:
        # 调用失败可能是连接断开，清除缓存让下次重连
        invalidate_tools_cache(server_name)
        return {"error": f"MCP 工具调用失败: {exc}"}

    return _parse_mcp_response(raw)


async def _list_tools_async(server_name: str | None = None) -> list[str]:
    tools = await _get_tools_cached(server_name)
    return [t.name for t in tools]


def call_mcp_tool(tool_name: str, tool_input: dict, server_name: str | None = None):
    """同步包装器，供 Search Agent 直接调用。工具列表带缓存，复用已有连接。"""
    return asyncio.run(_call_tool_async(tool_name, tool_input, server_name=server_name))


def list_mcp_tools(server_name: str | None = None) -> list[str]:
    """列出 MCP 工具，便于调试远程/本地 server 是否接通。"""
    return asyncio.run(_list_tools_async(server_name=server_name))