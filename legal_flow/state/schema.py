"""工作流状态与核心数据结构定义。

职责划分：
- CaseState（LangGraph 状态）：只承载路由决策需要的最小字段
- Task / Finding：研究产物，持久化在 Blackboard（state/blackboard.py）
"""
from typing import TypedDict, List


class CaseState(TypedDict, total=False):
    """LangGraph 工作流状态（路由用最小字段；研究产物存 Blackboard）"""
    case_id: str
    session_id: str
    user_query: str
    case_type: str
    issues: List[str]
    research_waves: int          # 研究波次计数（并行调度轮数）
    replan_count: int            # 重规划计数（有界）
    missing_info: List[str]      # Reviewer 指出的缺失信息 → Planner 据此补任务
    verdict: str                 # PASS / REPLAN / ESCALATE
    review_feedback: str
    draft: str
    answer: str
    confidence: float
    conflicts: List[dict]
    error: str


# 任务类型 → 执行方式（researcher.py 按此分发）
TASK_TYPES = {
    "facts":     "案件事实整理（LLM 结构化提取）",
    "statute":   "法源检索（混合检索：BM25 + 向量 + RRF + CrossEncoder 重排）",
    "precedent": "类案研究（模型知识，待接入真实判例源后升级为检索验证）",
    "element":   "法律要件分析（检索 + 要件归纳）",
}

# 证据来源三层标签（可信度从高到低）
SOURCE_TIERS = ["法条原文", "检索验证", "模型知识"]
