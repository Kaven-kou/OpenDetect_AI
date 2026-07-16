# OpenDetect_AI — 智能学术文献研究助手

一个面向 AI 领域研究者的智能文献管理与问答系统，基于 **LangGraph 多智能体工作流**驱动，集成 **MCP 协议**、**RAG 检索增强生成**、**短期 + 长期记忆机制**，将论文搜索、入库、问答、综述生成串联成一套完整的研究辅助流水线。

---

## 核心功能

- **智能搜索**：理解模糊描述（如"首次提出 ViT 的论文"），自动映射到精确论文或 arxiv ID；精确查询优先走 **OpenAlex**，泛搜直接走 OpenAlex 避免 ArXiv 限流，找不到时再用 **ArXiv MCP** 补充
- **多方式入库**：支持自动搜索入库、arxiv 链接入库、本地 PDF 手动入库三种方式，自动去重防止重复入库；付费墙论文自动识别放弃，防止重试死循环
- **RAG 问答**：基于已入库论文回答技术问题，每条结论标注来源，支持指代词理解（"它"、"这篇"）；检索走 **Hybrid（向量 + BM25）+ Self-Query + Rerank** 三级管线，显著抑制脏库跨领域噪音（详见「检索管线」一节）
- **综述生成**：对已入库的一批论文生成结构化综述和方法对比表，适合写 Related Work
- **库存查询**：随时列出向量库中已入库的所有论文
- **短期记忆**：滑动窗口保留最近 4 轮对话，注入所有 Agent 的 prompt，支持多轮追问
- **长期记忆**：**按 `user_id` 隔离**跨会话持久化用户研究偏好（方向、话题、来源），新会话自动读取并注入 Supervisor；与 Checkpointer（会话状态）、Chroma（论文库）三类记忆分别建模
- **事实性校验（Verifier）**：RAG 生成回答后经校验节点检查论断是否有检索来源支撑，不足时附核验提示，抑制幻觉
- **Human-in-the-Loop**：搜索到论文后、入库前用 LangGraph `interrupt` 中断，让用户勾选确认要入库的论文，再 `resume` 继续（既保留人类监督，又把不相关论文挡在库外）
- **Token 级流式**：RAG / 综述的最终回答通过 `stream(stream_mode=["values","messages"])` 逐字推送，前端打字机式渲染，不再让用户干等一大段
- **实时进度**：SSE 流式推送各 Agent 执行进度，前端可折叠"思考过程"面板
- **检索质量评估**：内置 `make eval`，在受控基准语料上量化 baseline vs 新管线（Hit@k / MRR / Precision@k / Noise@k / 上下文相关性）

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

> Web 会话下，Ingest 前会插入 **Human-in-the-Loop** 确认关卡（`interrupt` → 用户勾选 → `resume`）；RAG 回答经 **Verifier** 校验事实性后结束，最终答案 **token 级流式**推送。Search 有结果时**直连 Ingest**（确定性边界，省一次 Supervisor LLM）。

---

## 技术栈

| 模块 | 技术选型 |
|---|---|
| 多智能体工作流 | LangGraph（StateGraph + Supervisor 模式 + search→ingest 确定性边界） |
| 路由决策 | LLM `with_structured_output`（函数调用，Pydantic 约束，杜绝手撕 JSON） |
| 事实性校验 | Verifier 节点（rag→verify→END），检索来源支撑校验 + 不足降级 |
| 链调用 / Prompt 管理 | LangChain（prompt 静态前缀 / 动态后缀，前缀缓存友好） |
| 外部工具调用 | MCP 协议 + langchain-mcp-adapters |
| MCP Server | 本地 OpenAlex MCP（FastMCP stdio）+ 远程 ArXiv MCP（ModelScope streamable HTTP，精确查询补充） |
| RAG 存储与召回 | Chroma 向量库 + BM25（rank_bm25）稀疏检索 |
| 检索管线 | Self-Query（结构化过滤）→ Hybrid(dense+BM25) + RRF 融合 → Rerank（LLM / DashScope gte-rerank）+ 噪音闸门 |
| 检索评估 | 自建 LLM-as-judge + 检索指标（Hit@k / MRR / Precision@k / Noise@k），`make eval` |
| Embeddings | 阿里云 DashScope text-embedding-v4 |
| LLM | DeepSeek（兼容 OpenAI 接口） |
| PDF 解析 | PyMuPDF |
| 短期记忆 | 滑动窗口（最近 4 轮），注入 Supervisor / RAG / Search / Report |
| 长期记忆 | SQLite user_profile 表（按 `user_id` 隔离），跨会话持久化用户研究偏好 |
| 人机协作 | LangGraph `interrupt` / `Command(resume)` —— 入库前人工确认关卡 |
| 流式输出 | LangGraph `stream(["values","messages"])` —— 节点进度 + 最终答案 token 双流，SSE 推送 |
| 进度推送 | SSE（Server-Sent Events）+ 线程安全进度队列 |
| 前端 | 单页 HTML，支持 Markdown 渲染、思考面板折叠、PDF 拖拽上传、流式打字机、入库确认卡片 |

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
│   │   ├── supervisor.py         # Supervisor：结构化路由决策（with_structured_output）
│   │   ├── search.py             # Search Agent：论文搜索 + 意图映射
│   │   ├── ingest.py             # Ingest Agent：PDF 下载解析入库 + HITL 入库确认
│   │   ├── rag.py                # RAG Agent：语义检索 + 问答生成（token 流式）
│   │   ├── report.py             # Report Agent：综述与对比表生成
│   │   └── verify.py             # Verifier：RAG 回答事实性校验 + 不足降级
│   ├── tools/
│   │   ├── arxiv_tool.py         # OpenAlex 搜索工具（LangChain Tool）
│   │   ├── rag_tool.py           # Chroma 向量库工具 + 语料版本管理
│   │   ├── retriever.py          # 检索管线：Self-Query + Hybrid(RRF) + Rerank 去噪
│   │   ├── openalex_mcp_server.py # 本地 OpenAlex MCP Server（FastMCP）
│   │   ├── mcp_client.py         # MCP Client（带工具列表缓存）
│   │   └── progress.py           # SSE 进度队列（线程安全）
│   └── eval/
│       └── rag_eval.py           # RAG 检索评估：baseline vs 新管线（make eval）
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

> 检索管线（`OPENDETECT_RETRIEVAL_POOL` / `OPENDETECT_SELF_QUERY` / `OPENDETECT_RERANK_BACKEND` …）与 `OPENDETECT_HITL` 等均有合理默认值，通常无需配置；完整清单见 `.env.example`。

### 3. 启动服务

```powershell
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

### 5. 跑测试与检索评估

```bash
make test              # 单元测试（无需网络 / Key）
make integration-tests # 集成测试（需 LLM Key，跑一条最短路径）
make eval              # RAG 检索评估：baseline vs 新管线（受控基准语料）
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

路由决策用 `llm.with_structured_output(RouteDecision, method="function_calling")` 约束为 Pydantic 结构（`next` 字段是 `Literal` 白名单），从根上杜绝手撕 ```json``` 的脆弱解析；DeepSeek 偶发不支持工具调用时自动回退到 JSON 解析，双保险。

**路由理念（文献研究助手，答案有出处）**：
- 打招呼 / 问身份 / 问能力 → 自我介绍模板；
- 任何知识 / 技术 / 概念问题（"XX 是什么"、"XX 与 YY 的区别"）→ **一律走 RAG**，由文献支撑作答；库里没有相关论文时，RAG 会**如实说明并主动提议去搜索入库**，而不是用模型自身知识编造（严格"答案有出处"）；
- 只有库还空着时，才由 Supervisor 直接提示"暂无相关论文，要我去搜一批吗？"。

内置防死循环机制：`search_attempted` 确保搜索失败时不再重试；`rag_answer` / `final_report` 生成后直接收束到 FINISH；入库失败超过 2 次的论文永久放弃。

### Search Agent
三阶段工作：
1. **意图识别**：用 LLM + 对话上下文判断是特定著名论文（映射 arxiv ID）、标题（精确搜索）还是研究方向（关键词泛搜）；追问时（"还有吗"、"其他的"）从上下文提取核心话题，不会退化为 `deep learning` 等泛词
2. **搜索策略**：精确查询走 OpenAlex 精确接口，找不到再用 ArXiv MCP 补充；泛搜直接走 OpenAlex，完全规避 ArXiv 限流等待
3. **结果数量**：精确搜索返回 1 条，泛搜返回 5 条

### Ingest Agent
- **HITL 入库确认**：Web 会话下，搜索到论文后先用 `interrupt` 中断，把论文列表抛给前端让用户勾选；`resume` 返回选择后，仅入库勾选的论文，未选中的标记为已处理（避免重试）
- 网络论文：Search Agent 元数据 → 下载 PDF → PyMuPDF 解析 → 递归字符分块（`chunk_size=800`）→ **并行 embedding**（线程池并发多批，实测 ≈4.6×）→ 存入 Chroma
- 本地 PDF：直接读取本地路径 → 解析 → 入库（用户主动上传，无需确认）
- 去重：确定性 chunk ID（`arxiv_id__chunk_N`）幂等写入；写入后 `bump_corpus_version()` 使 BM25 索引缓存失效
- 重试保护：无 arxiv_id 的付费墙论文直接放弃；有 arxiv_id 的论文最多重试 2 次

### RAG Agent
用户问题 → **检索管线**（Self-Query → Hybrid(dense+BM25)+RRF → Rerank 去噪，见下节）召回 top-5 段落 → 注入对话上下文（理解指代词）→ DeepSeek 生成带引用的回答，答案 token 打上 `final_answer` 标签逐字流式推送。**严格"答案有出处"**：只依据检索片段作答，检索不到相关内容时不编造，而是明确告知"库里还没有能回答的论文，要我去搜索入库吗？"。

### Report Agent
检索管线召回 top-8 段落 + 已入库论文列表 + 对话上下文 → 生成包含背景、方法分类、对比表、趋势分析的结构化综述（同样 token 流式）。

---

## 检索管线（RAG 升级）

朴素 `similarity_search(k=5)` 在「脏库」上会捞出大量跨领域噪音（例如问 CLIP 却召回医学 PMC-CLIP）。本项目在向量检索之上叠加三层能力：

```
用户问题
    │
    ▼  ① Self-Query：LLM 结构化抽取「语义 query + 年份/作者/标题过滤条件」
┌───────────────┬───────────────┐
▼               ▼
Dense(向量)     BM25(关键词)          ② Hybrid：稠密召专有名词/缩写弱，稀疏补精确匹配
└──────┬────────┘
       ▼  RRF 融合（Reciprocal Rank Fusion，1/(60+rank) 累加）
  候选池(pool=30)
       ▼  元数据后置过滤（年份/作者/标题，在 Python 端做，不依赖 Chroma filter 方言）
       ▼  ③ Rerank + 噪音闸门：交叉相关性重排，丢弃跨领域段落
    top-k 结果
```

- **Self-Query**：用 `with_structured_output` 把"2023年后的 XX 论文"解析成 `year_min=2023` + 纯语义 query，避免约束词污染向量检索。
- **Hybrid + RRF**：Dense（DashScope embedding）+ BM25（rank_bm25）各召回一批，用 RRF 融合。BM25 索引随语料版本缓存，入库后自动重建。
- **Rerank + 噪音闸门**：默认 **LLM listwise 重排**（复用 DeepSeek，零额外依赖，prompt 显式指令"跨领域段落一律排除"）；可切 DashScope `gte-rerank`（复用 embedding Key，带相关性阈值 `OPENDETECT_RERANK_MIN_SCORE`）。任一环节失败都优雅退化，不会让检索崩。

**评估结果**（`make eval`，受控合成基准：5 篇目标论文 + 4 篇跨领域噪音，12 个问题含多 gold）：

| 指标 | Baseline（纯 dense） | 新管线 | 说明 |
|---|---|---|---|
| Precision@5 ↑ | 0.45 | **0.98** | top-5 聚焦到正确论文的比例（核心提升）|
| Noise@5 ↓ | 0.03 | **0.00** | top-5 里跨领域噪音占比 |
| nDCG@5 / Hit@k / MRR | 1.00 | 1.00 | 小语料上已饱和，区分度看 Precision/Noise |
| 上下文相关性 ↑ | 0.83 | **1.00** | LLM 判定检索内容是否足以回答 |
| P50 / P95 延迟 | ~180 / 186 ms | ~2.6 / 2.8 s | ★成本面：新管线更慢 |
| LLM 调用 / 次检索 | 0 | 2 | ★成本面：self-query + rerank |

> 说明：baseline 能召回正确论文但 top-5 里塞了一半不相关内容；新管线返回紧凑、无跨领域噪音的结果，**代价是每次检索多 2 次 LLM 调用、延迟升到秒级**——`make eval` 如实打印这一成本面。评估为自建 LLM-as-judge + 检索指标（Hit@k / MRR / Precision@k / nDCG@k / Noise@k），不依赖 RAGAS 等重依赖；`--no-judge` 可跳过 LLM 判分更快跑。这是**受控合成基准**（可复现），生产中应替换为几十~上百条真实标注问题。

---

## Human-in-the-Loop（入库确认）

Web 会话下，工作流在 **search 出结果、ingest 入库前**插入人工确认关卡：

```
search 找到论文 → ingest 节点 interrupt(论文列表) ──暂停──▶ 前端渲染勾选卡片
                                                              │ 用户勾选
        继续入库选中的论文 ◀──resume(选中序号)── /api/chat/resume
```

- 用 LangGraph `interrupt()` 抛出待确认论文，SSE 推 `type:"interrupt"` 事件；前端渲染带复选框的确认卡片。
- 用户确认后 `POST /api/chat/resume`，后端 `Command(resume=选中序号)` 从中断点继续；未选中的论文标记为已处理，不再重试。
- 状态用 checkpointer（SQLite）持久化，中断可跨请求恢复。开关：`OPENDETECT_HITL`（CLI 的 `run()` / `chat()` 不触发，仅 Web 生效）。

这一关同时缓解了脏库问题——用户可以把搜错的、跨领域的论文直接挡在库外。

---

## 记忆机制说明

三种"记忆"分别建模，概念清晰、互不混淆：

| 类型 | 实现 | 范围键 | 内容 |
|---|---|---|---|
| 短期记忆（上下文）| 滑动窗口（4 轮；Supervisor 用 2 轮）| 会话内 | 最近对话问答对，理解指代词和追问 |
| 会话状态持久化 | LangGraph SqliteSaver（Checkpointer）| `thread_id` | 完整工作流状态 + messages，多轮连贯、中断可恢复 |
| 长期用户偏好 | SQLite user_profile 表 | `user_id` | 研究方向、常搜话题、关注来源（**按用户隔离**）|
| 论文知识库 | Chroma 向量库 | 全局 | 已入库论文全文 embedding |

> 注意区分：Checkpointer 存的是「会话状态」（按 thread_id），user_profile 才是「跨会话的长期用户偏好」（按 user_id）——两者常被混为一谈，面试时要说清。

每次对话结束后，系统在后台线程按 `user_id` 异步提取本轮研究偏好并合并到 `user_profile`，不阻塞响应。`user_id` 由前端在 localStorage 生成并随请求传入，多用户互不串画像。

---

## Verifier 事实性校验

RAG 生成回答后，工作流经一个 `verify` 节点（`rag → verify → END`）做事实性把关，抑制"检索没支撑却硬答"的幻觉：

- 无检索内容却生成了回答 → 直接附「缺乏来源」提示（不花 LLM）。
- 有检索内容 → 用 `with_structured_output` 让 LLM 判回答是否 grounded、列出无支撑论断；不通过则给回答**附核验提示**（不删改已流式给用户的正文）。
- fail-open：校验器任一步失败都放行，不成为可用性瓶颈。开关 `OPENDETECT_VERIFY`。

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
| `/api/chat/stream` | POST | SSE 流式对话（主接口，含进度 / token / 中断事件） |
| `/api/chat/resume` | POST | HITL 恢复：提交入库确认选择后继续流式 |
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
- [x] **检索管线升级**：Hybrid(dense+BM25)+RRF + Self-Query + Rerank 去噪
- [x] **RAG 检索评估**：`make eval`，baseline vs 新管线量化对比
- [x] **Human-in-the-Loop**：入库前 `interrupt`/`resume` 人工确认
- [x] **Token 级流式输出**：最终答案逐字打字机渲染
- [x] **Supervisor 结构化路由**：`with_structured_output` 替代手撕 JSON
- [x] **确定性边界**：search 有结果直连 ingest，省一次 Supervisor LLM
- [x] **Verifier 事实性校验**：RAG 回答检索来源支撑校验 + 不足降级
- [x] **上下文工程**：prompt 静态前缀 / 动态后缀（前缀缓存友好）+ 按 Agent 裁剪窗口
- [x] **用户级长期记忆**：user_profile 按 `user_id` 隔离（含旧表平滑迁移）
- [x] **RAG 检索评估**：`make eval`，Precision@k / nDCG@k / Noise@k + P50/P95 延迟 / LLM 调用成本
- [x] 真实单元 + 集成测试（替换模板残留用例，含入库失败重试、用户隔离、检索纯函数）

**后续计划**
- [ ] 异步化工作流（节点 `ainvoke` + MCP 全异步），提升并发
- [ ] 长期记忆升级为 LangGraph `BaseStore` + 语义召回
- [ ] 交叉编码器（bge-reranker）本地重排后端
- [ ] 评估集扩充到几十~上百条真实标注问题（当前为受控合成基准）
- [ ] 安全边界：PDF 大小/域名白名单、CORS 收敛、生产用 Postgres Checkpointer