import os

# ==========================================
# 📂 1. 路径与环境配置
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = BASE_DIR
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
CHROMA_DB_DIR = os.path.join(PROJECT_ROOT, "data", "chroma_legal_db")
BM25_INDEX_PATH = os.path.join(PROJECT_ROOT, "data", "bm25_index.pkl")
UPLOAD_TEMP_DIR = os.path.join(PROJECT_ROOT, "data", "temp_upload")
DOC_HASH_PATH = os.path.join(PROJECT_ROOT, "data", "doc_hashes.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_TEMP_DIR, exist_ok=True)

# ==========================================
# 🔍 2. LangSmith 监控
# ==========================================
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = "AI_Lawyer_Multi_Agent"

# ==========================================
# 💻 3. 硬件
# ==========================================
DEVICE = "cpu"  # 有GPU改成 cuda

# ==========================================
# 🧠 4. 模型（使用 ModelScope 国内镜像）
# ==========================================
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

# 使用 ModelScope 国内镜像
USE_MODELSCOPE = True

# ==========================================
# 🔍 5. 检索
# ==========================================
RETRIEVE_K = 150
RERANK_K = 100
RETRIEVAL_RELEVANCE_THRESHOLD = 0.6
MAX_ADAPTIVE_RETRIEVAL = 2
MAX_RETRIEVAL_RETRIES = 2
FALLBACK_RETRIEVE_K = 200     # 兜底时扩大召回
FALLBACK_RERANK_K = 200

# ==========================================
# 🤖 6. LLM（免费API）
# ==========================================
# 日常使用：智谱AI GLM-4-Flash
ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY", "")
ZHIPUAI_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
ZHIPUAI_MODEL = "glm-4-flash"
ZHIPUAI_TEMP = 0.2

# 深层推理：DeepSeek-V1
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_TEMP = 0.1

# 最大重写次数
MAX_REWRITE_LOOPS = 2

# ==========================================
# 🧭 7. 路由（规则优先）
# ==========================================
ENABLE_ROUTER = True

# ==========================================
# 🧠 8. 记忆模块（摘要记忆）
# ==========================================
MEMORY_ENABLED = True
MAX_HISTORY_TURNS = 3
SUMMARY_MAX_LENGTH = 100

# ==========================================
# 📤 9. 上传去重
# ==========================================
ENABLE_DEDUP = True

# ==========================================
# 📊 10. 检索质量控制
# ==========================================
RETRIEVAL_QUALITY_THRESHOLD = 0.7
CONFIDENCE_THRESHOLD = 0.65

# ==========================================
# 💾 11. Redis记忆
# ==========================================
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
