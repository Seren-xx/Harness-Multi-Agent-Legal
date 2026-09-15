"""Legal Flow ——多智能体法律分析系统（Planner / Researcher / Analyst / Reviewer）。

本包自包含：检索（runtime/retrieval.py）、对话记忆（state/conversation_memory.py）、
配置（config.py）均复制自 legacy/（v1 原始实现，算法与行为保持一致），
在其上构建"规划 → 并行研究 → 分析 → 审查仲裁"的动态任务协作架构；
"""
