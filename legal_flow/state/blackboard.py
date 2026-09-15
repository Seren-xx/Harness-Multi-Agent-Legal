"""案件 Blackboard：多智能体共享案件状态。

与对话记忆（memory_manager.py）的分工：
- 对话记忆：用户说过什么（Redis List + 摘要压缩）
- Blackboard：案件研究到什么程度（facts / tasks / findings / conflicts / decisions）

实现：Redis String 存整个案件快照（JSON），带 TTL；
Redis 不可用时自动降级为进程内字典（与 memory_manager 同样的降级策略）。
"""
import json
import time
from typing import Optional

import redis

from legal_flow import config

_local_store: dict = {}  # Redis 不可用时的兜底存储


class Blackboard:
    """单个案件的共享工作区。

    - Planner 写 tasks
    - Researcher 写 findings / 更新 task 状态
    - Reviewer 写 conflicts / decisions
    - 任何人可读全量快照（审计 / 前端任务看板）
    """

    KEY_PREFIX = "legal_flow:case"

    def __init__(self, case_id: str, redis_url: Optional[str] = None):
        self.case_id = case_id
        try:
            self.client = redis.from_url(
                redis_url or config.REDIS_URL, decode_responses=True
            )
            self.client.ping()
            self.use_redis = True
        except Exception:
            self.use_redis = False

    @property
    def _key(self) -> str:
        return f"{self.KEY_PREFIX}:{self.case_id}"

    # ---------- 读写 ----------
    def snapshot(self) -> dict:
        """读取案件快照（不存在时返回空结构）"""
        if self.use_redis:
            raw = self.client.get(self._key)
            return json.loads(raw) if raw else self._empty()
        return _local_store.get(self._key, self._empty())

    def update(self, **fields) -> dict:
        """合并写入并返回最新快照"""
        snap = self.snapshot()
        snap.update(fields)
        snap["case_id"] = self.case_id
        snap["updated_at"] = time.time()
        if self.use_redis:
            self.client.set(
                self._key,
                json.dumps(snap, ensure_ascii=False),
                ex=config.BLACKBOARD_TTL,
            )
        else:
            _local_store[self._key] = snap
        return snap

    @staticmethod
    def _empty() -> dict:
        return {
            "case_id": "",
            "facts": "",
            "case_type": "",
            "issues": [],
            "tasks": [],
            "findings": [],
            "conflicts": [],
            "decisions": [],
            "missing_info": [],
            "created_at": time.time(),
            "updated_at": time.time(),
        }

    # ---------- 便捷方法 ----------
    def add_finding(self, finding: dict):
        snap = self.snapshot()
        snap["findings"].append(finding)
        self.update(findings=snap["findings"])

    def add_conflicts(self, conflicts: list):
        if not conflicts:
            return
        snap = self.snapshot()
        snap["conflicts"].extend(conflicts)
        self.update(conflicts=snap["conflicts"])

    def add_decision(self, decision: dict):
        snap = self.snapshot()
        snap["decisions"].append(decision)
        self.update(decisions=snap["decisions"])
