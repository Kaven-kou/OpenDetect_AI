# OpenDetect_AI — 智能学术文献研究助手

一个面向 AI 领域研究者的智能文献管理与问答系统，基于 **LangGraph 多智能体工作流**驱动，集成 **MCP 协议**、**RAG 检索增强生成**、**短期 + 长期记忆机制**，将论文搜索、入库、问答、综述生成串联成一套完整的研究辅助流水线。

---

## 核心功能

- **智能搜索**：理解模糊描述（如"首次提出 ViT 的论文"），自动映射到精确论文或 arxiv ID；精确查询优先走 **OpenAlex**，泛搜直接走 OpenAlex 避免 ArXiv 限流，找不到时再用 **ArXiv MCP** 补充
- **多方式入库**：支持自动搜索入库、arxiv 链接入库、本地 PDF 手动入库三种方式，自动去重防止重复入库；付费墙论文自动识别放弃，防止重试死循环
- **RAG 问答**：基于已入库论文回答技术问题，每条结论标注来源论文，支持指代词理解（"它"、"这篇"）
- **综述生成**：对已入库的一批论文生成结构化综述和方法对比表，适合写 Related Work
- **库存查询**：随时列出向量库中已入库的所有论文
- **短期记忆**：滑动窗口保留最近 4 轮对话，注入所有 Agent 的 prompt，支持多轮追问
- **长期记忆**：跨会话持久化用户研究偏好（方向、话题、来源），新会话自动读取并注入 Supervisor
- **实时进度**：SSE 流式推送各 Agent 执行进度，前端可折叠"思考过程"面板

---

## 架构概览

```
用户输入
    ↓
Supervisor Agent（意图识别 · 动态路由 · 状态管理）
    ↓ 条件路由（含长期记忆 + 短期上下文注入）
┌──────────┬──────────┬──────────┬──────────┐
│  Search  │  Ingest  │   RAG    │  Report  │
│  Agent   │  Agent   │  Agent   │  Agent   │
└────┬─────┴────┬─────┴────┬─────┴────┬─────┘
     │          │          │          │
     ▼          ▼          ▼          ▼
 OpenAlex   Chroma     DeepSeek   DeepSeek
 优先，     向量库      LLM        LLM
 ArXiv补充
     │          │          │          │
     └──────┬───┘          └────┬─────┘
            ▼                   ▼
      回到 Supervisor           END
            ▼（next == FINISH）
           END → 异步提取用户偏好 → SQLite
```

---

## 技术栈

| 模块 | 技术选型 |
|---|---|
| 多智能体工作流 | LangGraph（StateGraph + Supervisor 模式） |
| 链调用 / Prompt 管理 | LangChain |
| 外部工具调用 | MCP 协议 + langchain-mcp-adapters |
| MCP Server | 本地 OpenAlex MCP（FastMCP stdio）+ 远程 ArXiv MCP（ModelScope streamable HTTP，精确查询补充） |
| RAG 存储与召回 | Chroma 向量数据库（本地持久化） |
| Embeddings | 阿里云 DashScope text-embedding-v4 |
| LLM | DeepSeek（兼容 OpenAI 接口） |
| PDF 解析 | PyMuPDF |
| 短期记忆 | 滑动窗口（最近 4 轮），注入 Supervisor / RAG / Search / Report |
| 长期记忆 | SQLite user_profile 表，跨会话持久化用户研究偏好 |
| 进度推送 | SSE（Server-Sent Events）+ 线程安全进度队列 |
| 前端 | 单页 HTML，支持 Markdown 渲染、思考面板折叠、PDF 拖拽上传 |

---

## 项目结构

```
OpenDetect_AI/
├── api.py                        # FastAPI 后端入口，SSE 流式接口
├── frontend/
│   └── index.html                # 前端页面（Markdown 渲染 + 思考面板）
├── src/opendetect_ai/
│   ├── graph.py                  # LangGraph 主图，工作流入口
│   ├── state.py                  # AgentState 共享状态定义
│   ├── prompts.py                # 各 Agent 的 Prompt 模板
│   ├── env_utils.py              # 环境变量加载
│   ├── context_utils.py          # 短期记忆：滑动窗口上下文提取
│   ├── user_memory.py            # 长期记忆：用户偏好读写与 LLM 提取
│   ├── agents/
│   │   ├── supervisor.py         # Supervisor：意图识别与动态路由
│   │   ├── search.py             # Search Agent：论文搜索 + 意图映射
│   │   ├── ingest.py             # Ingest Agent：PDF 下载解析入库
│   │   ├── rag.py                # RAG Agent：语义检索 + 问答生成
│   │   └── report.py             # Report Agent：综述与对比表生成
│   └── tools/
│       ├── arxiv_tool.py         # OpenAlex 搜索工具（LangChain Tool）
│       ├── rag_tool.py           # Chroma 向量库工具
│       ├── openalex_mcp_server.py # 本地 OpenAlex MCP Server（FastMCP）
│       ├── mcp_client.py         # MCP Client（带工具列表缓存）
│       └── progress.py           # SSE 进度队列（线程安全）
├── data/
│   ├── chroma_db/                # 向量库持久化目录（自动创建）
│   └── chat_history.db           # SQLite：对话历史 + 用户长期偏好
├── .env                          # 环境变量（不提交 Git）
├── .env.example                  # 环境变量模板
├── pyproject.toml
└── langgraph.json
```

---

## 快速开始

### 1. 克隆并安装依赖

```bash
git clone <your-repo-url>
cd OpenDetect_AI
uv sync
```

> 本项目依赖已写入 `pyproject.toml`，通常只需要执行 `uv sync`。如果你修改了依赖，再用 `uv add ...`。

### 2. 配置环境变量

复制模板并填入 Key（注意：`.env` 中不要写中文注释，否则 Windows 下 `langgraph dev` 可能报编码错误）：

```powershell
Copy-Item .env.example .env
```

`.env` 关键字段：

```dotenv
OPENDETECT_LLM_MODEL="deepseek-chat"
OPENDETECT_LLM_BASE_URL="https://api.deepseek.com"
OPENDETECT_LLM_API_KEY="your-deepseek-key"

OPENDETECT_EMBED_MODEL="text-embedding-v4"
OPENDETECT_EMBED_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
OPENDETECT_EMBED_API_KEY="your-dashscope-key"

# LangSmith 追踪（可选，关闭时设为 false，无需填 API Key）
LANGCHAIN_TRACING_V2="false"
LANGCHAIN_PROJECT="OpenDetect_AI"
LANGCHAIN_API_KEY="your-langsmith-key"

CHROMA_PERSIST_DIR="./data/chroma_db"

OPENDETECT_ARXIV_MCP_URL="https://mcp.api-inference.modelscope.net/f26e1fc45ee54a/mcp"
```

### 3. 启动服务

```powershell
pip install fastapi uvicorn python-multipart
uvicorn api:app --reload --host 0.0.0.0 --port 8000
# 浏览器打开 http://localhost:8000
```

### 4. 验证各组件

```powershell
# 验证 MCP 工具连通性
uv run python -c @'
from opendetect_ai.tools.mcp_client import list_mcp_tools
print("openalex:", list_mcp_tools("openalex"))
print("arxiv:", list_mcp_tools("arxiv"))
'@

# 验证向量库工具
uv run python -c @'
from opendetect_ai.tools.rag_tool import RAG_TOOLS
print([t.name for t in RAG_TOOLS])
'@

# 验证主图能编译
uv run python -c @'
from opendetect_ai.graph import build_graph
print(type(build_graph()).__name__)
'@
```

---

## 运行指令

以下命令默认在项目根目录执行。

### 1. 搜索并自动入库

系统会先搜索论文，再下载 PDF、解析、分块并写入 Chroma。已存在的论文会自动跳过。付费墙论文（无 arxiv_id）自动放弃，不会死循环重试。

```powershell
# 精确搜索：著名论文 / 明确标题 / arxiv ID，返回 1 条
uv run python -c @'
from opendetect_ai.graph import run
run("帮我搜索首次提出 ViT 的论文")
'@

# 宽泛搜索：研究方向，返回 5 条
uv run python -c @'
from opendetect_ai.graph import run
run("帮我找几篇 instance segmentation 相关论文")
'@
```

### 2. arxiv 链接直接入库

```powershell
uv run python -c @'
from opendetect_ai.graph import run
run("帮我入库这篇论文 https://arxiv.org/abs/1706.03762")
'@
```

### 3. 本地 PDF 手动入库

```powershell
uv run python -c @'
from opendetect_ai.tools.rag_tool import ingest_local_pdf

r = ingest_local_pdf.invoke({
    "file_path": "F:/papers/2303.13076v1.pdf",
    "title": "CORA: Adapting CLIP for Open-Vocabulary Detection",
    "authors": "Yuzhong Ma et al.",
    "published": "2023-03-23",
})
print(r)  # {"status": "ok", "chunks": 113}
'@
```

### 4. 查询已入库论文

```powershell
uv run python -c @'
from opendetect_ai.tools.rag_tool import list_ingested_papers

papers = list_ingested_papers.invoke({})
if papers and "message" in papers[0]:
    print("向量库为空")
else:
    print(f"已入库论文共 {len(papers)} 篇：\n")
    for i, p in enumerate(papers, 1):
        arxiv = p.get("arxiv_id") or "无"
        print("{}. {}".format(i, p.get("title")))
        print("   发表: {}  arxiv: {}\n".format(p.get("published"), arxiv))
'@
```

### 5. RAG 技术问答

```powershell
uv run python -c @'
from opendetect_ai.graph import run
run("Swin Transformer 相比 ViT 做了哪些改进？")
'@
```

### 6. 生成综述报告

```powershell
uv run python -c @'
from opendetect_ai.graph import run
run("帮我生成一份已有论文的综述报告")
'@
```

### 7. 多轮对话（支持指代词追问）

同一个 `thread_id` 会复用 LangGraph Checkpointer 保存的会话状态，短期记忆（最近 4 轮）自动注入：

```powershell
uv run python -c @'
from opendetect_ai.graph import chat

chat("帮我搜索 Swin Transformer 的论文", thread_id="demo_session")
chat("它相比 ViT 做了哪些改进？", thread_id="demo_session")   # "它" 自动理解为 Swin Transformer
chat("在目标检测任务上表现如何？", thread_id="demo_session")
'@
```

### 8. 查看 / 清除长期记忆

```powershell
# 查看跨会话用户偏好（HTTP API）
curl http://localhost:8000/api/user-profile

# 清除长期记忆（重置画像）
curl -X DELETE http://localhost:8000/api/user-profile
```

或直接用 Python：

```powershell
uv run python -c @'
from opendetect_ai.user_memory import load_user_profile
print(load_user_profile())
'@
```

### 9. 查看历史会话 ID

```powershell
uv run python -c @'
from opendetect_ai.graph import list_threads
print(list_threads())
'@
```

### 10. LangGraph 可视化调试

```powershell
uv run langgraph dev
```

启动后在浏览器中打开 LangGraph Studio 页面，选择 `agent` 图进行调试。

---

## 常见问题

### Windows 终端中文输出乱码

```powershell
$env:PYTHONIOENCODING="utf-8"
```

### LangSmith 连接失败导致启动报错

将 `.env` 中的追踪关闭即可，`LANGSMITH_API_KEY` 不再是必填项：

```dotenv
LANGCHAIN_TRACING_V2="false"
```

### `uv` 缓存目录异常

```powershell
$env:UV_CACHE_DIR="$PWD\.uv-cache"
uv run python -c @'
print("uv cache ok")
'@
```

### OpenAlex 搜索结果相关性差

OpenAlex 覆盖广但对 AI 细分方向的关键词匹配不如 ArXiv 精准。遇到"instance segmentation 但搜到混凝土检测"之类的问题，建议换更具体的英文关键词，或直接提供 arxiv ID / 论文标题。

### PaperMeta 反序列化警告

```
Deserializing unregistered type opendetect_ai.state.PaperMeta
```

这是 LangGraph checkpoint 的兼容性提示，不影响功能。如需消除，在 `.env` 中添加：

```dotenv
LANGGRAPH_STRICT_MSGPACK=false
```

---

## 子 Agent 详解

### Supervisor Agent
工作流调度中枢。每轮执行时：
1. 读取**长期记忆**（用户跨会话研究偏好）
2. 读取**短期记忆**（最近 4 轮对话上下文）
3. 结合当前状态（搜索数、待入库数、是否已回答等）用 LLM 决策下一步路由

内置防死循环机制：`search_attempted` 确保搜索失败时不再重试；`rag_answer` / `final_report` 生成后直接收束到 FINISH；入库失败超过 2 次的论文永久放弃。

### Search Agent
三阶段工作：
1. **意图识别**：用 LLM + 对话上下文判断是特定著名论文（映射 arxiv ID）、标题（精确搜索）还是研究方向（关键词泛搜）；追问时（"还有吗"、"其他的"）从上下文提取核心话题，不会退化为 `deep learning` 等泛词
2. **搜索策略**：精确查询走 OpenAlex 精确接口，找不到再用 ArXiv MCP 补充；泛搜直接走 OpenAlex，完全规避 ArXiv 限流等待
3. **结果数量**：精确搜索返回 1 条，泛搜返回 5 条

### Ingest Agent
- 网络论文：Search Agent 元数据 → 下载 PDF → PyMuPDF 解析 → 分块向量化 → 存入 Chroma
- 本地 PDF：直接读取本地路径 → 解析 → 入库
- 去重：确定性 chunk ID（`arxiv_id__chunk_N`）幂等写入
- 重试保护：无 arxiv_id 的付费墙论文直接放弃；有 arxiv_id 的论文最多重试 2 次

### RAG Agent
用户问题向量化 → Chroma 召回 top-5 相关段落 → 注入对话上下文（理解指代词）→ DeepSeek 生成带引用的回答。

### Report Agent
召回 top-8 相关段落 + 已入库论文列表 + 对话上下文 → 生成包含背景、方法分类、对比表、趋势分析的结构化综述。

---

## 记忆机制说明

| 类型 | 实现 | 范围 | 内容 |
|---|---|---|---|
| 短期记忆 | 滑动窗口（4轮）| 会话内 | 最近对话问答对，理解指代词和追问 |
| 长期记忆（对话历史）| LangGraph SQLite Checkpointer | 同 thread_id 内 | 完整 messages 列表，多轮连贯 |
| 长期记忆（用户偏好）| SQLite user_profile 表 | 跨会话 | 研究方向、常搜话题、关注来源 |
| 向量知识库 | Chroma | 永久 | 已入库论文全文 embedding |

每次对话结束后，系统在后台线程异步提取本轮对话的研究偏好并合并到 `user_profile`，不阻塞响应。

---

## 入库方式对比

| 方式 | 触发方法 | 适用场景 |
|---|---|---|
| 自动搜索入库 | `run("帮我搜索 ViT 论文")` | 在线检索 + 自动下载 |
| arxiv 链接入库 | `run("帮我入库 https://arxiv.org/abs/xxxx")` | 已知具体论文链接 |
| 本地 PDF 入库 | `ingest_local_pdf.invoke({...})` 或前端拖拽 | 已下载到本地的 PDF |

---

## MCP 集成说明

本项目接入两个 MCP Server：

- **本地 OpenAlex MCP**：项目内置的 `tools/openalex_mcp_server.py`，基于 FastMCP，stdio 传输。**主要搜索后端**，稳定无限流，覆盖绝大多数 arxiv 论文
- **远程 ArXiv MCP**：ModelScope streamable HTTP 部署，地址来自 `OPENDETECT_ARXIV_MCP_URL`。仅在 OpenAlex 精确查询找不到时作为补充

远程 ArXiv MCP 配置：

```json
{
  "mcpServers": {
    "arxiv-mcp-server": {
      "type": "streamable_http",
      "url": "https://mcp.api-inference.modelscope.net/f26e1fc45ee54a/mcp"
    }
  }
}
```

本地 OpenAlex MCP 暴露三个工具：

| 工具 | 功能 |
|---|---|
| `search_papers` | 关键词搜索，返回多篇论文 |
| `get_paper_by_id` | 按 arxiv ID 精确获取单篇（6 种方式逐一尝试） |
| `get_paper_by_title` | 按标题搜索最匹配的一篇 |

MCP 客户端带**工具列表缓存**，首次连接后复用，不重复建立进程，降低调用开销。

---

## API 端点

| 端点 | 方法 | 说明 |
|---|---|---|
| `/api/chat/stream` | POST | SSE 流式对话（主接口） |
| `/api/chat` | POST | 非流式对话 |
| `/api/papers` | GET | 列出已入库论文 |
| `/api/threads` | GET | 列出历史会话 ID |
| `/api/upload-pdf` | POST | 上传本地 PDF 入库 |
| `/api/user-profile` | GET | 查看用户长期偏好 |
| `/api/user-profile` | DELETE | 清除用户长期偏好 |

---

## 参考文档

- LangGraph 文档：https://langchain-ai.github.io/langgraph/
- LangChain 文档：https://docs.langchain.com
- MCP 协议规范：https://modelcontextprotocol.io
- OpenAlex API：https://docs.openalex.org

---

## 已完成 / 后续计划

**已完成**
- [x] arxiv 链接直接入库
- [x] 本地 PDF 手动入库（前端拖拽 + API）
- [x] 查询已入库论文列表
- [x] LangGraph Checkpointer 支持多轮对话历史
- [x] 短期记忆：滑动窗口上下文注入（4轮）
- [x] 长期记忆：跨会话用户研究偏好持久化
- [x] SSE 实时进度推送 + 前端思考面板
- [x] OpenAlex 优先搜索策略（规避 ArXiv 限流）
- [x] 付费墙论文识别，防止重试死循环
- [x] MCP 工具列表缓存，降低连接开销
- [x] FastAPI Web 服务 + 单页前端

**后续计划**
- [ ] `langgraph dev` 可视化调试界面完善
- [ ] 异步化工作流，提升并发性能
- [ ] 向量库可视化查询界面
- [ ] LLM token 流式输出（逐字打字效果）