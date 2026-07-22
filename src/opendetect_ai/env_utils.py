"""Environment variable loader for OpenDetect_AI."""

import os
from dotenv import load_dotenv
 
load_dotenv(override=True)
 
# ── 通用 Keys ──────────────────────────────────────────
ALI_API_KEY        = os.getenv("ALI_API_KEY", "")
ALI_BASE_URL       = os.getenv("ALI_BASE_URL", "")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL    = os.getenv("OPENAI_BASE_URL", "")
ZHIPUAI_API_KEY    = os.getenv("ZHIPUAI_API_KEY", "")
DEEPSEEK_API_KEY   = os.getenv("DEEPSEEK_API_KEY", "")
HF_TOKEN           = os.getenv("HF_TOKEN", "")
LANGSMITH_API_KEY  = os.getenv("LANGSMITH_API_KEY", "")
 
# ── LLM：DeepSeek ──────────────────────────────────────
OPENDETECT_LLM_MODEL    = os.getenv("OPENDETECT_LLM_MODEL", "deepseek-v4-pro")
OPENDETECT_LLM_BASE_URL = os.getenv("OPENDETECT_LLM_BASE_URL", "https://api.deepseek.com")
OPENDETECT_LLM_API_KEY  = os.getenv("OPENDETECT_LLM_API_KEY", DEEPSEEK_API_KEY)
 
# ── Embeddings：阿里 DashScope ─────────────────────────
OPENDETECT_EMBED_MODEL    = os.getenv("OPENDETECT_EMBED_MODEL", "text-embedding-v4")
OPENDETECT_EMBED_BASE_URL = os.getenv("OPENDETECT_EMBED_BASE_URL", ALI_BASE_URL)
OPENDETECT_EMBED_API_KEY  = os.getenv("OPENDETECT_EMBED_API_KEY", ALI_API_KEY)
 
# ── 向量数据库 ─────────────────────────────────────────
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")

# ── 检索管线（Hybrid + Self-Query + Rerank）────────────
# 粗召回候选池大小（dense + BM25 融合后、rerank 之前）
OPENDETECT_RETRIEVAL_POOL = int(os.getenv("OPENDETECT_RETRIEVAL_POOL", "30"))
# 是否启用自查询（LLM 抽取年份/作者/标题过滤条件），"false" 关闭
OPENDETECT_SELF_QUERY = os.getenv("OPENDETECT_SELF_QUERY", "true").lower() == "true"
# 重排后端：llm（默认，复用 DeepSeek，零额外依赖）| dashscope（gte-rerank）| none
OPENDETECT_RERANK_BACKEND = os.getenv("OPENDETECT_RERANK_BACKEND", "llm").lower()
# dashscope 重排模型（仅 backend=dashscope 时使用，复用 EMBED 的 DashScope Key）
OPENDETECT_RERANK_MODEL = os.getenv("OPENDETECT_RERANK_MODEL", "gte-rerank-v2")
# 噪音闸门：rerank 相关性分低于该阈值的段落直接丢弃（跨领域脏数据）
OPENDETECT_RERANK_MIN_SCORE = float(os.getenv("OPENDETECT_RERANK_MIN_SCORE", "0.3"))

# ── Human-in-the-Loop ─────────────────────────────────
# 入库前是否插入人工确认关卡（仅 Web/持久化会话生效），"false" 关闭
OPENDETECT_HITL = os.getenv("OPENDETECT_HITL", "true").lower() == "true"
OPENDETECT_APPROVAL_TTL_SECONDS = max(
    60, int(os.getenv("OPENDETECT_APPROVAL_TTL_SECONDS", "1800"))
)

# PDF 入库安全上限：同时约束远程下载与本地上传，避免超大文件耗尽内存/磁盘。
OPENDETECT_MAX_PDF_MB = max(1, int(os.getenv("OPENDETECT_MAX_PDF_MB", "50")))
OPENDETECT_MAX_PDF_PAGES = max(1, int(os.getenv("OPENDETECT_MAX_PDF_PAGES", "200")))
OPENDETECT_PDF_ALLOWED_HOSTS = {
    host.strip().lower()
    for host in os.getenv(
        "OPENDETECT_PDF_ALLOWED_HOSTS",
        "arxiv.org,export.arxiv.org,openaccess.thecvf.com,aclanthology.org",
    ).split(",")
    if host.strip()
}

# Web 默认只允许本地开发来源；生产环境通过逗号分隔显式配置。
OPENDETECT_CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "OPENDETECT_CORS_ORIGINS",
        "http://localhost:8000,http://127.0.0.1:8000",
    ).split(",")
    if origin.strip()
]

# ── Verifier（RAG 回答事实性校验）─────────────────────
# RAG 生成后是否校验回答有无检索来源支撑、不足时降级提示，"false" 关闭
OPENDETECT_VERIFY = os.getenv("OPENDETECT_VERIFY", "true").lower() == "true"
 
# ── MCP Server ─────────────────────────────────────────
OPENDETECT_ARXIV_MCP_URL = os.getenv(
    "OPENDETECT_ARXIV_MCP_URL",
    "https://mcp.api-inference.modelscope.net/f26e1fc45ee54a/mcp",
)
 
# ── LangSmith 追踪 ─────────────────────────────────────
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_PROJECT    = os.getenv("LANGCHAIN_PROJECT", "OpenDetect_AI")
LANGCHAIN_API_KEY    = os.getenv("LANGCHAIN_API_KEY", LANGSMITH_API_KEY)
 
# ── 启动检查 ───────────────────────────────────────────
# LangSmith 为可选项（LANGCHAIN_TRACING_V2=false 时不需要）
_REQUIRED = {
    "OPENDETECT_LLM_API_KEY":   OPENDETECT_LLM_API_KEY,
    "OPENDETECT_EMBED_API_KEY": OPENDETECT_EMBED_API_KEY,
}

def validate_env() -> None:
    """项目启动时调用，检查必要环境变量是否齐全。"""
    missing = [k for k, v in _REQUIRED.items() if not v]
    if missing:
        raise EnvironmentError(
            f"缺少必要的环境变量: {', '.join(missing)}\n"
            "请检查 .env 文件是否配置正确。"
        )
