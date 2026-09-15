"""Legal Flow 全局配置"""
import os

# ==========================================
# 📂 1. 路径与环境配置
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根（legal_flow/ 的上级）
PROJECT_ROOT = BASE_DIR
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
CHROMA_DB_DIR = os.path.join(PROJECT_ROOT, "data", "chroma_legal_db")
BM25_INDEX_PATH = os.path.join(PROJECT_ROOT, "data", "bm25_index.pkl")
UPLOAD_TEMP_DIR = os.path.join(PROJECT_ROOT, "data", "temp_upload")
DOC_HASH_PATH = os.path.join(PROJECT_ROOT, "data", "doc_hashes.json")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

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

# ==========================================
# 🤖 6. LLM
# ==========================================
# 日常使用：阿里云百炼 Qwen（OpenAI 兼容模式）
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_MODEL = "qwen3.7-flash"
DASHSCOPE_TEMP = 0.2

# 备用：DeepSeek
DEEPSEEK_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4.1-flash"
DEEPSEEK_TEMP = 0.1

# ==========================================
# 📊 7. 检索质量控制 / 审查
# ==========================================
CONFIDENCE_THRESHOLD = 0.65

# ==========================================
# 💾 8. Redis记忆
# ==========================================
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# ==========================================
# 🧩 9. v2 扩展：多智能体调度（Harness Runtime）
# ==========================================
# —— 有界重规划（Bounded Replan）——
MAX_REPLANS = int(os.getenv("MAX_REPLANS", "2"))

# —— 并行研究 ——
MAX_PARALLEL_TASKS = int(os.getenv("MAX_PARALLEL_TASKS", "3"))
TASK_TIMEOUT_SECONDS = int(os.getenv("TASK_TIMEOUT_SECONDS", "120"))
MAX_RESEARCH_WAVES = int(os.getenv("MAX_RESEARCH_WAVES", "8"))  # 研究波次上限（防死循环）

# —— 研究任务检索参数（比 v1 更聚焦：每个任务只取 Top 少量证据）——
RESEARCH_RETRIEVE_K = int(os.getenv("RESEARCH_RETRIEVE_K", "100"))
RESEARCH_RERANK_K = int(os.getenv("RESEARCH_RERANK_K", "8"))

# —— 自适应召回（检索质量不足时扩大召回重试）——
ADAPTIVE_RECALL_THRESHOLD = float(os.getenv(
    "ADAPTIVE_RECALL_THRESHOLD", str(RETRIEVAL_RELEVANCE_THRESHOLD)))  # 0.6
MAX_RECALL_RETRIES = int(os.getenv("MAX_RECALL_RETRIES", "2"))

# —— Blackboard（案件共享状态）——
BLACKBOARD_TTL = int(os.getenv("BLACKBOARD_TTL", str(24 * 3600)))  # 案件状态保留时长（秒）
