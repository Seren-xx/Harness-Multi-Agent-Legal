"""Researcher Agent：按 Task DAG 并行执行就绪任务，产出 Finding 写入 Blackboard。

调度方式：每轮（wave）取所有就绪任务，线程池并行执行；
无依赖的任务（facts / statute / precedent）天然并行，element 等依赖完成后执行。
"""
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

from legal_flow import config
from legal_flow.runtime import tools
from legal_flow.runtime.llm import call_llm
from legal_flow.orchestration.task_graph import TaskGraph, DONE, FAILED
from legal_flow.state.blackboard import Blackboard

FACTS_PROMPT = """你是法律助理。请从用户陈述中提取结构化案件事实，只输出：
1. 当事人 / 主体
2. 关键行为与时间线
3. 争议焦点（一句话一条）
4. 尚不明确、需要向当事人确认的信息
不得编造事实中没有的内容。"""

PRECEDENT_PROMPT = """你是资深律师。请基于你的法律知识，概括与下列问题类似的典型案例裁判要点（3 条以内），
每条注明"案型"与"裁判倾向"。注意：这是模型知识而非真实判例检索，请在输出开头标注〔模型知识〕。"""

ELEMENT_PROMPT = """你是法律分析专家。基于以下法条证据，归纳本案的适用法律要件（构成要件列表），
并逐项对照案件事实说明"满足 / 不满足 / 待确认"。"""


def _finding(task, content: str, tier: str, laws: list = None) -> dict:
    """构造标准化研究发现"""
    return {
        "finding_id": uuid.uuid4().hex[:8],
        "task_id": task.task_id,
        "type": task.type,
        "description": task.description,
        "content": content,
        "source_tier": tier,  # 法条原文 / 检索验证 / 模型知识 / 当事人陈述
        "laws": laws or [],
        "created_at": time.time(),
    }


def execute_task(task, query: str) -> tuple:
    """执行单个研究任务，返回 (task_id, status, finding)"""
    try:
        if task.type == "facts":
            content = call_llm([
                {"role": "system", "content": FACTS_PROMPT},
                {"role": "user", "content": query},
            ], temperature=0.1)
            return task.task_id, DONE, _finding(task, content, "当事人陈述")

        if task.type == "statute":
            r = tools.search_articles(task.description or query)
            if not r.success:
                raise RuntimeError(r.error)
            return task.task_id, DONE, _finding(task, r.data, "法条原文", laws=r.laws)

        if task.type == "precedent":
            content = call_llm([
                {"role": "system", "content": PRECEDENT_PROMPT},
                {"role": "user", "content": query},
            ], temperature=0.2)
            return task.task_id, DONE, _finding(task, content, "模型知识")

        if task.type == "element":
            r = tools.search_articles(f"{task.description} {query}")
            if not r.success:
                raise RuntimeError(r.error)
            content = call_llm([
                {"role": "system", "content": ELEMENT_PROMPT},
                {"role": "user", "content": f"问题：{query}\n\n法条证据：\n{r.data}"},
            ], temperature=0.1)
            return task.task_id, DONE, _finding(task, content, "检索验证", laws=r.laws)

        raise ValueError(f"未知任务类型: {task.type}")
    except Exception as e:
        return task.task_id, FAILED, _finding(task, f"任务失败: {e}", "不可用")


def research_node(state: dict) -> dict:
    """LangGraph 节点：执行一波（wave）就绪任务。

    线程池并行 + 整波超时控制；任务级失败不阻断流程（Analyst/Reviewer 会处理缺口）。
    """
    case_id = state["case_id"]
    bb = Blackboard(case_id)
    graph = TaskGraph.from_dicts(bb.snapshot()["tasks"])

    ready = graph.ready()
    if not ready:
        return {}

    tools.warmup()  # 预加载检索模型，避免并行线程竞争初始化

    print(f"🔬 [Researcher] 第 {state.get('research_waves', 0) + 1} 轮并行执行 "
          f"{len(ready)} 个任务: {[t.task_id for t in ready]}")

    deadline = time.time() + config.TASK_TIMEOUT_SECONDS
    results = []
    with ThreadPoolExecutor(max_workers=config.MAX_PARALLEL_TASKS) as ex:
        futures = {ex.submit(execute_task, t, state["user_query"]): t for t in ready}
        for fut, task in futures.items():
            try:
                results.append(fut.result(timeout=max(0, deadline - time.time())))
            except FutureTimeout:
                results.append((task.task_id, FAILED, _finding(task, "任务超时", "不可用")))
            except Exception as e:
                results.append((task.task_id, FAILED, _finding(task, f"任务失败: {e}", "不可用")))

    for task_id, status, finding in results:
        graph.mark(task_id, status)
        bb.add_finding(finding)

    bb.update(tasks=graph.to_dicts())
    return {"research_waves": state.get("research_waves", 0) + 1}
