"""任务图（Task DAG）：Planner 产出的任务依赖与调度状态。

纯逻辑模块：不依赖 LLM / 检索 / Redis，可独立单元测试（tests/test_task_graph.py）。
"""
from dataclasses import dataclass, field, asdict
from typing import List

VALID_TYPES = {"facts", "statute", "precedent", "element"}

PENDING, RUNNING, DONE, FAILED = "pending", "running", "done", "failed"


@dataclass
class Task:
    task_id: str
    type: str                      # facts / statute / precedent / element
    description: str
    depends_on: List[str] = field(default_factory=list)
    status: str = PENDING
    assignee: str = "researcher"   # 预留 Agent Handoff：任务可动态改派
    result: str = ""
    retry_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def to_task(d: dict) -> Task:
    """字典 → Task（容忍多余字段，例如 LLM 输出的额外键）"""
    return Task(**{k: v for k, v in d.items() if k in Task.__dataclass_fields__})


class TaskGraph:
    """任务依赖图：就绪调度 / 状态流转 / 追加任务（Bounded Replan 用）"""

    def __init__(self, tasks: List[Task]):
        self.tasks: dict = {t.task_id: t for t in tasks}
        self._validate()

    def _validate(self):
        # 1) 类型合法 + 依赖存在
        for t in self.tasks.values():
            if t.type not in VALID_TYPES:
                raise ValueError(f"未知任务类型: {t.type}（{t.task_id}）")
            for dep in t.depends_on:
                if dep not in self.tasks:
                    raise ValueError(f"任务 {t.task_id} 依赖不存在的任务: {dep}")
        # 2) 环检测（Kahn：反复摘除依赖已满足的任务，摘不干净即有环）
        remaining = {tid: set(t.depends_on) for tid, t in self.tasks.items()}
        while remaining:
            ready = [tid for tid, deps in remaining.items() if not (deps & remaining.keys())]
            if not ready:
                raise ValueError(f"任务依赖存在环: {sorted(remaining)}")
            for tid in ready:
                del remaining[tid]

    @classmethod
    def from_dicts(cls, dicts: list) -> "TaskGraph":
        return cls([to_task(d) for d in dicts])

    def to_dicts(self) -> list:
        return [t.to_dict() for t in self.tasks.values()]

    def ready(self) -> List[Task]:
        """就绪任务：pending 且所有依赖已完成（这些任务可以并行执行）"""
        return [
            t for t in self.tasks.values()
            if t.status == PENDING
            and all(self.tasks[d].status == DONE for d in t.depends_on)
        ]

    def pending_count(self) -> int:
        return sum(1 for t in self.tasks.values() if t.status == PENDING)

    def mark(self, task_id: str, status: str, result: str = ""):
        t = self.tasks[task_id]
        t.status = status
        if result:
            t.result = result

    def fail_deadlocked(self):
        """级联跳过：依赖含 FAILED 的 pending 任务直接记为 FAILED。

        否则这些任务永远不就绪但一直 pending，研究波次会空转耗尽上限。
        反复扫描直到不再有变化（处理失败链：A失败→B跳过→C也跳过）。
        """
        changed = True
        while changed:
            changed = False
            for t in self.tasks.values():
                if t.status != PENDING:
                    continue
                if any(self.tasks[d].status == FAILED for d in t.depends_on):
                    self.mark(t.task_id, FAILED, "前置任务失败，已跳过")
                    changed = True

    def add(self, task: Task):
        """追加任务（Reviewer 裁决 REPLAN 后，Planner 补充研究任务）"""
        task.depends_on = [d for d in task.depends_on if d in self.tasks]  # 只保留存在的依赖
        self.tasks[task.task_id] = task
        self._validate()
