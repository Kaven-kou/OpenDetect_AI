# OpenDetect_AI

OpenDetect_AI 是一个面向 AI 研究者的文献检索与问答系统。它使用 LangGraph 组织论文搜索、PDF 入库、RAG 问答和综述生成，并通过 MCP 接入 OpenAlex 与 ArXiv。

当前主流程可用；主动澄清（Clarify）已接入 LangGraph 控制流：指代歧义、精确标题多候选、标题与 arXiv ID 冲突、精确标题两后端皆空时，系统会先反问澄清，下一轮确定性解析用户选择。

## 核心能力

- **论文搜索**：支持 arXiv ID、链接、明确标题和研究主题。
- **多种入库方式**：搜索后入库、arXiv 链接入库、本地 PDF 上传。
- **文献问答**：只基于已入库论文作答，并标注论文来源。
- **综述生成**：基于论文库生成研究背景、方法分类、对比表和趋势总结。
- **多轮理解**：将“好啊”“还有吗”“它呢”等输入解析为自包含查询。
- **检索增强**：Self-Query、Dense + BM25、RRF 和 Rerank 组成完整检索管线。
- **人工确认**：Web 流程在论文入库前暂停，让用户选择需要入库的论文。
- **评测闭环**：提供检索、搜索意图和澄清判定三套评测。

## 当前架构

```text
用户输入
   |
   v
Resolve                 每轮只执行一次
   |                    生成 resolved_query，维护 pending_action
   |                    指代有多个可指对象 --> Clarify --> END
   v
Supervisor              结构化路由
   |
   +----------+-----------+----------+
   |          |           |          |
   v          v           v          v
 Search     Ingest       RAG       Report
   |          |           |          |
OpenAlex   PDF/Chroma  Retriever   Retriever
ArXiv MCP    |           |          |
   |          |           v          v
   |          |        Verifier     END
   |          |           |
   |          +-> Supervisor v
   |                        END
   +-- 多候选/标题冲突/两后端皆空 --> Clarify --> END
   +-- 有待入库论文 --> Ingest
   +-- 否则 --> Supervisor
```

确定性状态转移由代码控制。例如 Search 找到待入库论文后直接进入 Ingest；只有需要理解自然语言意图时才调用 LLM。Web 会话在 Ingest 前通过 `interrupt` 暂停，并由 `/api/chat/resume` 恢复。**Clarify 是普通对话轮**（非 `interrupt`）：反问后收束到 END，下一轮 Resolve 用确定性规则解析用户的选择（序号/标题/arXiv ID/放弃/新任务），最多连续澄清两次后给可操作兜底。

### 语义理解边界

系统没有把所有判断都塞进一个 Prompt：

1. Resolve 在入口处理省略、指代和上一轮待确认动作。
2. 用户明确提供的 arXiv ID 或 URL 由正则解析，不进入 LLM。
3. SearchIntent 只判断 `exact_title` 或 `topic`，并生成后端查询串。
4. arXiv ID 的真实性由 OpenAlex/ArXiv 返回结果确认，不让模型凭记忆生成。
5. 下游统一读取 `effective_query(state)`，保留原始 `user_query` 便于回溯。

## 技术栈

| 模块 | 实现 |
|---|---|
| 工作流 | LangGraph `StateGraph` + SQLite Checkpointer |
| LLM 编排 | LangChain + OpenAI 兼容接口 |
| 结构化输出 | Pydantic + `with_structured_output(function_calling)` |
| 搜索工具 | 本地 OpenAlex MCP + 远程 ArXiv MCP |
| 向量库 | Chroma |
| Embedding | DashScope `text-embedding-v4` |
| 稀疏检索 | BM25 |
| PDF 解析 | PyMuPDF |
| API | FastAPI + SSE |
| 前端 | 单页 HTML |

## 项目结构

```text
OpenDetect_AI/
├── api.py
├── frontend/
│   └── index.html
├── src/opendetect_ai/
│   ├── graph.py
│   ├── state.py
│   ├── prompts.py
│   ├── context_utils.py
│   ├── user_memory.py
│   ├── agents/
│   │   ├── resolve.py
│   │   ├── supervisor.py
│   │   ├── search.py
│   │   ├── ingest.py
│   │   ├── rag.py
│   │   ├── report.py
│   │   ├── verify.py
│   │   └── clarify.py       # Clarify：澄清判定 + clarify 节点（已接入 Graph）
│   ├── tools/
│   │   ├── mcp_client.py
│   │   ├── openalex_mcp_server.py
│   │   ├── rag_tool.py
│   │   ├── retriever.py
│   │   └── progress.py
│   └── eval/
│       ├── rag_eval.py
│       ├── intent_eval.py
│       └── clarify_eval.py
├── tests/
├── data/
├── Makefile
├── pyproject.toml
└── langgraph.json
```

## 快速开始

### 1. 安装依赖

项目要求 Python 3.13+ 和 `uv`。

```bash
git clone <your-repo-url>
cd OpenDetect_AI
uv sync
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

必填配置：

```dotenv
OPENDETECT_LLM_MODEL="deepseek-chat"
OPENDETECT_LLM_BASE_URL="https://api.deepseek.com"
OPENDETECT_LLM_API_KEY="your-deepseek-api-key"

OPENDETECT_EMBED_MODEL="text-embedding-v4"
OPENDETECT_EMBED_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
OPENDETECT_EMBED_API_KEY="your-dashscope-api-key"

CHROMA_PERSIST_DIR="./data/chroma_db"
```

检索、HITL、Verifier 和 LangSmith 开关均有默认值，完整配置见 [.env.example](.env.example)。

### 3. 启动 Web 服务

```bash
uv run uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

浏览器打开 `http://localhost:8000`。

### 4. 运行测试和评测

```bash
make test               # 单元测试，不访问外部服务
make integration-tests  # 最短集成链路，需要模型配置
make lint               # Ruff
make eval               # RAG 检索评测
make intent-eval        # Resolve -> SearchIntent 在线评测
make clarify-eval       # Clarify 判定评测，尚不代表已接入 Graph
```

## 使用方式

### Web

Web 页面支持：

- SSE 流式对话和执行进度；
- RAG/Report token 级输出；
- 搜索结果入库前勾选确认；
- 本地 PDF 拖拽上传；
- 多轮会话和历史线程。

### Python

单轮调用：

```python
from opendetect_ai.graph import run

run("帮我搜索首次提出 ViT 的论文")
run("Swin Transformer 相比 ViT 做了哪些改进？")
run("帮我生成一份已有论文的综述报告")
```

多轮调用：

```python
from opendetect_ai.graph import chat

chat("讲讲 LoRA", thread_id="demo", user_id="user-1")
chat("好啊", thread_id="demo", user_id="user-1")
chat("还有吗", thread_id="demo", user_id="user-1")
```

相同 `thread_id` 复用会话 checkpoint；`user_id` 用于隔离跨会话用户偏好。

本地 PDF 入库：

```python
from opendetect_ai.tools.rag_tool import ingest_local_pdf

result = ingest_local_pdf.invoke({
    "file_path": "papers/example.pdf",
    "title": "Example Paper",
    "authors": "A. Researcher et al.",
    "published": "2026-01-01",
})
print(result)
```

## Agent 职责

### Resolve

Resolve 是每轮入口，只运行一次：

- 普通自包含问题直接透传，不调用 LLM；
- 有 `pending_action` 时，用确定性规则处理确认或拒绝；
- 只有含指代或省略标记的输入才调用一次 LLM 改写；
- 将本轮用户输入记录为 `HumanMessage`；
- 输出 `resolved_query`，不覆盖原始输入。

### Supervisor

Supervisor 结合当前状态、短期上下文和用户偏好输出结构化 `RouteDecision`。`next` 受 Literal 白名单约束，只能进入 `search`、`ingest`、`rag`、`report` 或 `FINISH`。

产品策略是“答案有出处”：知识问题进入 RAG；文献不足时明确说明并提出搜索建议，不用模型参数知识补写答案。

### Search

Search 分三步：

1. 正则识别用户明确提供的现代/旧式 arXiv ID 和 URL；
2. `SearchIntent` 判断精确标题或主题搜索；
3. 将解析结果直接传给 OpenAlex，必要时用 ArXiv MCP 补充。

精确查询从 OpenAlex 取候选池（`search_papers` 五篇，成功空结果时再问 ArXiv），主题查询默认返回五篇。精确标题若有多个接近候选、或两后端都成功返回空、或与用户明确给的 arXiv ID 冲突，会转 Clarify 反问而不是猜一篇。

### Ingest

Ingest 下载并解析 PDF，按块生成 embedding 后写入 Chroma。主要保护包括：

- Web 入库前 HITL 确认；
- 确定性 chunk ID，重复写入幂等；
- 入库后使 BM25 缓存失效；
- 无 arXiv ID 的不可下载论文不反复重试；
- 可重试论文有次数上限。

### RAG、Report 和 Verifier

RAG 召回 top-k 论文片段并生成带来源的回答；Report 使用更多上下文生成结构化综述。RAG 回答随后进入 Verifier：无检索内容时直接附提示，有检索内容时检查论断是否得到来源支撑。

Verifier 采用 fail-open 策略，校验失败不会阻塞主回答。开关为 `OPENDETECT_VERIFY`。

## RAG 检索管线

```text
自然语言问题
   |
   v
Self-Query                 语义查询 + 年份/作者/标题过滤
   |
   +------------+
   |            |
   v            v
Dense          BM25
   |            |
   +-----RRF----+
         |
         v
元数据过滤 -> Rerank/噪音过滤 -> top-k
```

- **Self-Query**：将显式约束抽成结构化字段，避免约束词污染语义查询。
- **Hybrid**：Dense 负责语义召回，BM25 补充缩写、论文名和专有词匹配。
- **RRF**：按倒数排名融合，不要求两种检索分数同尺度。
- **Rerank**：支持 LLM listwise、DashScope `gte-rerank` 或关闭重排。
- **降级策略**：任一增强环节失败时退化为更简单的检索路径。

受控合成评测通过 `make eval` 运行。当前结果表明新管线显著提高 Precision、降低跨领域噪音，但会增加两次 LLM 调用并带来秒级延迟。该结果只代表仓库内的小型基准，不等同于生产效果。

## 状态与记忆

| 类型 | 存储 | 隔离键 | 用途 |
|---|---|---|---|
| 短期上下文 | `messages` 滑动窗口 | `thread_id` | 指代消解和多轮问答 |
| 工作流状态 | LangGraph SQLite Checkpointer | `thread_id` | 节点状态、HITL 和跨进程恢复 |
| 用户偏好 | SQLite `user_profile` | `user_id` | 跨会话研究兴趣 |
| 论文知识库 | Chroma | 当前为全局库 | 文献检索 |

`pending_action` 保存系统等待用户确认的动作。当前主流程使用 `kind: "search"`；Clarify 设计复用同一字段的 `kind: "clarification"`，避免维护两份悬挂状态。

## 评测

| 命令 | 目标 | 状态 |
|---|---|---|
| `make test` | 确定性逻辑和节点单元测试 | 已接入 CI 式本地检查 |
| `make integration-tests` | 最短在线链路 | 需要模型和外部服务 |
| `make eval` | Dense baseline 与完整检索管线对比 | 已完成 |
| `make intent-eval` | arXiv 解析、Resolve、SearchIntent | 21 条 golden |
| `make clarify-eval` | 澄清信号和选择解析 | 24 条 golden，Graph 接入基线 |

评测集规模较小，主要用于防止重构回归。生产化前需要替换为更多真实查询、真实论文库和人工标注。

## Clarify 主动澄清

`clarify.py` 的判定与选择解析已注册为 `clarify` 图节点，并接入 Resolve/Search 的条件边。当前覆盖：

- `ambiguous_reference`：上下文中存在多个**可 grounding**（`evidence + message_index` 可核对）的指代对象；
- `multiple_papers`：候选池去重后多个标题同时达到**绝对下限**且分差很小（`top1,top2 ≥ FLOOR ∧ top1−top2 ≤ GAP`）；
- `entity_conflict`：用户给出的标题与明确 arXiv ID 返回标题冲突；
- `exact_title_not_found`：两个后端都**成功返回空**结果（后端报错 ≠ 没找到）。

触发后 `clarify` 节点把带序号的问题作为本轮答案返回并收束到 END；下一轮 Resolve 以确定性规则解析回复（序号/标题/arXiv ID → 选中；放弃 → 清空；新任务 → 转处理；越界不默认第一项），`attempts` 最多连续澄清两次后给可操作兜底。`low_relevance` 暂缓——检索层还没有统一输出可比较的重排分数，不临时拍阈值。判定逻辑有 24 条 golden（`make clarify-eval`）持续回归。

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/chat` | 非流式对话 |
| POST | `/api/chat/stream` | SSE 流式对话 |
| POST | `/api/chat/resume` | 恢复 HITL 入库流程 |
| GET | `/api/threads` | 历史会话 ID |
| GET | `/api/papers` | 已入库论文 |
| POST | `/api/upload-pdf` | 上传本地 PDF |
| GET | `/api/user-profile` | 读取用户偏好 |
| DELETE | `/api/user-profile` | 清除用户偏好 |

## 已知限制

- Chroma 论文库当前是全局共享的，没有按用户隔离。
- SQLite Checkpointer 适合本地开发，不适合作为高并发生产存储。
- `PaperMeta` 作为自定义类型写入旧 checkpoint 时会触发 LangGraph forward-compat 警告；后续应改存普通 dict 或显式注册类型。
- Clarify 已接入图，但尚未做「跨进程重启」级别的端到端验收（已有单测 + 真实模型 smoke 覆盖判定与路由）。
- RAG 和意图评测仍是小型受控数据集。
- PDF 下载速度受 arXiv 限流和来源站点影响。

## 后续计划

- 为 Clarify 补「跨进程重启」端到端验收（复用 pending_action 持久化，与主流程一致）；
- 为检索结果保留统一相关性分数，再评估 `low_relevance` 澄清；
- 将 `PaperMeta` checkpoint 数据迁移为稳定的可序列化结构；
- 扩充真实标注意图和检索评测集；
- 异步化节点与 MCP 调用；
- 生产环境改用持久化服务和更严格的安全策略。

架构决策、故障案例和面试讲法见 [面试.md](面试.md)。
