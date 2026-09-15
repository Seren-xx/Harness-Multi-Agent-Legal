"""Planner Agent：用户问题 → 案件类型 / 法律争点 / 任务 DAG。

流程不再写死，由 Planner 动态分工，Researcher 再按任务依赖并行执行。
"""
from legal_flow.runtime.llm import call_llm_json
from legal_flow.orchestration.task_graph import Task, VALID_TYPES, to_task
from legal_flow.state.blackboard import Blackboard

PLANNER_PROMPT = """你是法律案件规划师。请分析用户问题并制定研究计划。

要求：
1. 识别案件类型（如：知识产权侵权 / 消费维权 / 劳动争议）与需要解决的法律争点。
2. 制定 3~6 个研究任务，任务类型只能是：
   - facts     案件事实整理
   - statute   法源检索（找适用的法条）
   - precedent 类案研究
   - element   法律要件分析（必须依赖 facts 和至少一个 statute 任务）
3. facts / statute / precedent 之间不设依赖（可并行）；element 依赖前置任务。
4. 每个任务用一句话 description 说明研究目标。

只输出 JSON，格式：
{"case_type": "...", "issues": ["争点1", "..."],
 "tasks": [{"task_id": "T1", "type": "facts", "description": "...", "depends_on": []}]}"""


def default_plan(query: str, offset: int = 0) -> dict:
    """兜底计划：LLM 规划失败时使用（保证系统不因规划失败而中断）"""
    return {
        "case_type": "一般法律咨询",
        "issues": [query[:50]],
        "tasks": [
            Task(f"T{offset+1}", "facts", "整理案件事实与争议焦点"),
            Task(f"T{offset+2}", "statute", "检索与问题直接相关的法条"),
            Task(f"T{offset+3}", "precedent", "研究类似案件的裁判要点"),
            Task(f"T{offset+4}", "element", "归纳法律适用要件并对照本案",
                 depends_on=[f"T{offset+1}", f"T{offset+2}"]),
        ],
    }


def _clean(tasks: list) -> list:
    """清洗 LLM 输出的任务：过滤非法类型、修剪不存在的依赖"""
    ids = {t.get("task_id") for t in tasks}
    cleaned = []
    for t in tasks:
        task = to_task(t)
        if task.type not in VALID_TYPES:
            continue
        task.depends_on = [d for d in task.depends_on if d in ids and d != task.task_id]
        cleaned.append(task)
    return cleaned


def plan_case(query: str) -> dict:
    """LLM 规划；输出非法或解析失败时回退到兜底计划"""
    try:
        plan = call_llm_json(
            [{"role": "user", "content": f"用户问题：{query}\n\n{PLANNER_PROMPT}"}],
            model="dashscope", temperature=0.2,
        )
        tasks = _clean(plan.get("tasks", []))
        if not tasks:
            raise ValueError("空任务列表")
        return {
            "case_type": plan.get("case_type") or "一般法律咨询",
            "issues": plan.get("issues", []),
            "tasks": tasks,
        }
    except Exception:
        return default_plan(query)


def planner_node(state: dict) -> dict:
    """LangGraph 节点：首轮规划；REPLAN 轮针对缺失信息补充任务"""
    case_id = state["case_id"]
    bb = Blackboard(case_id)
    snap = bb.snapshot()
    missing = [m for m in state.get("missing_info", []) if m]

    if snap["tasks"]:  # REPLAN：保留已完成任务，针对缺失信息补充研究任务
        offset = len(snap["tasks"])
        new_tasks = [
            Task(f"T{offset + i + 1}", "statute", f"补充检索：{info}")
            for i, info in enumerate(missing)
        ]
        bb.update(tasks=snap["tasks"] + [t.to_dict() for t in new_tasks],
                  missing_info=missing)
        print(f"📋 [Planner] 重规划：补充 {len(new_tasks)} 个任务")
        return {}

    plan = plan_case(state["user_query"])
    bb.update(case_type=plan["case_type"],
              issues=plan["issues"],
              tasks=[t.to_dict() for t in plan["tasks"]])
    print(f"📋 [Planner] 案件类型: {plan['case_type']}，任务数: {len(plan['tasks'])}")
    return {"case_type": plan["case_type"], "issues": plan["issues"]}
