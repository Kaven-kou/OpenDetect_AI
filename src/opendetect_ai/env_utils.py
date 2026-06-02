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
 
# ── MCP Server ─────────────────────────────────────────
OPENDETECT_ARXIV_MCP_URL = os.getenv(
    "OPENDETECT_ARXIV_MCP_URL",
    "https://mcp.api-inference.modelscope.net/f26e1fc45ee54a/mcp",
)
 
# ── LangSmith 追踪 ─────────────────────────────────────
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
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
 