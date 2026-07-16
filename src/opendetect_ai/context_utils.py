"""
对话上下文工具 —— OpenDetect_AI
提取最近 N 轮有效对话，格式化为可注入 prompt 的字符串。
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, AIMessage

# 滑动窗口大小：保留最近几轮「用户问 + 助手答」
CONTEXT_WINDOW = 4

# 过滤掉 Supervisor 路由日志和进度消息，只保留面向用户的内容
_SKIP_PREFIXES = (
    "Supervisor 决策:",
    "入库完成，新增",
    "找到 ",
    "没有待入库",
    "所有论文均已入库",
)


def _is_user_facing(content: str) -> bool:
    """判断一条 AIMessage 是否是面向用户的实质内容（非路由日志）。"""
    if not content or not content.strip():
        return False
    return not any(content.startswith(p) for p in _SKIP_PREFIXES)


def build_context_str(messages: list, window: int = CONTEXT_WINDOW) -> str:
    """
    从 AgentState.messages 里提取最近 window 轮有效对话，
    返回可直接插入 prompt 的字符串。

    格式：
        用户：帮我搜索 ViT 的论文
        助手：已找到并入库论文《An Image is Worth 16x16 Words》...
        用户：它和 CNN 相比有什么优势？
        助手：根据论文内容，ViT 的主要优势是...

    如果没有历史记录，返回空字符串。
    """
    if not messages:
        return ""

    # 只取 HumanMessage 和面向用户的 AIMessage，忽略路由日志
    pairs: list[tuple[str, str]] = []  # [(user_msg, assistant_msg), ...]
    i = 0
    while i < len(messages):
        msg = messages[i]
        if isinstance(msg, HumanMessage):
            user_text = msg.content.strip()
            # 找紧跟在后面的第一条面向用户的 AIMessage
            ai_text = ""
            j = i + 1
            while j < len(messages):
                next_msg = messages[j]
                if isinstance(next_msg, HumanMessage):
                    break  # 下一轮用户消息，停止
                if isinstance(next_msg, AIMessage):
                    content = next_msg.content.strip()
                    if _is_user_facing(content):
                        ai_text = content
                        break
                j += 1
            if user_text:
                pairs.append((user_text, ai_text))
        i += 1

    # 只保留最近 window 轮，去掉当前轮（最后一对，尚未回答）
    recent = pairs[-(window + 1):-1] if len(pairs) > 1 else []
    if not recent:
        return ""

    lines = []
    for user_text, ai_text in recent:
        lines.append(f"用户：{user_text}")
        if ai_text:
            # 截断过长的回答，避免 context 太长撑爆 token
            truncated = ai_text[:300] + "..." if len(ai_text) > 300 else ai_text
            lines.append(f"助手：{truncated}")

    return "\n".join(lines)