"""工作流组装：Planner → Research（并行波次）→ Analyst → Reviewer → Finalize。

              ┌──────────── REPLAN（有界，默认 ≤ 2 次）───────────┐
              ↓                                                  │
   Planner → Research ⇄（多轮波次，直到任务完成）→ Analyst → Reviewer → Finalize → END
                                                  │ PASS / ESCALATE        ↑
                                                  └────────────────────────┘
"""
import sys
import time
import uuid
from pathlib import Path

# 允许直接运行本文件（python legal_flow/orchestration/workflow.py）：
# 把项目根目录加入模块搜索路径，使 legal_flow 包可导入
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langgraph.graph import StateGraph, END

from legal_flow import config
from legal_flow.agents.planner import planner_node
from legal_flow.agents.researcher import research_node
from legal_flow.agents.analyst import analyst_node
from legal_flow.agents.reviewer import reviewer_node
from legal_flow.orchestration.task_graph import TaskGraph
from legal_flow.state.blackboard import Blackboard
from legal_flow.state.schema import CaseState
from legal_flow.runtime.llm import call_llm
from legal_flow.state.conversation_memory import memory_manager


def _is_greeting(query: str) -> bool:
    """问候语识别（复用 v1 规则：直接回复，不进工作流）"""
    greetings = ["你好", "您好", "hello", "hi", "嗨", "早上好", "晚上好", "下午好", "谢谢", "感谢"]
    return any(kw in query.lower() for kw in greetings) and len(query) < 20


def _pending_tasks(case_id: str) -> int:
    return TaskGraph.from_dicts(Blackboard(case_id).snapshot()["tasks"]).pending_count()


def after_research(state: dict) -> str:
    """研究波次结束：仍有 pending → 下一轮；全部完成（或波次超限）→ 进入分析"""
    if _pending_tasks(state["case_id"]) == 0:
        return "analyst"
    if state.get("research_waves", 0) >= config.MAX_RESEARCH_WAVES:
        return "analyst"
    return "research"


def after_review(state: dict) -> str:
    return {"PASS": "finalize", "REPLAN": "planner", "ESCALATE": "finalize"}[state["verdict"]]


def finalize_node(state: dict) -> dict:
    """收尾：PASS 直接定稿；ESCALATE 附加未决事项与免责声明；写对话记忆。"""
    verdict = state.get("verdict", "ESCALATE")
    answer = state.get("draft", "")

    if verdict == "ESCALATE":
        lines = ["", "---", "⚠️ **本意见未通过最终审查，仅供初步参考**："]
        if state.get("review_feedback"):
            lines.append(f"- 审查意见：{state['review_feedback'][:300]}")
        for c in state.get("conflicts", []):
            lines.append(f"- 冲突：{c.get('issue', c)} → {c.get('suggestion', '')}")
        for m in state.get("missing_info", []):
            lines.append(f"- 待补充：{m}")
        lines.append("- 建议咨询执业律师获取正式意见。")
        answer += "\n".join(lines)

    # 对话记忆（与案件 Blackboard 分离：对话记忆管"用户说过什么"）
    session_id = state.get("session_id") or state["case_id"]
    memory_manager.add_message(session_id, "user", state["user_query"])
    memory_manager.add_message(session_id, "assistant", answer)
    if memory_manager.get_message_count(session_id) > 8:
        memory_manager.compress_history(session_id, call_llm)

    Blackboard(state["case_id"]).add_decision(
        {"agent": "finalize", "verdict": verdict, "at": time.time()})
    return {"answer": answer}


def build_workflow():
    g = StateGraph(CaseState)
    g.add_node("planner", planner_node)
    g.add_node("research", research_node)
    g.add_node("analyst", analyst_node)
    g.add_node("reviewer", reviewer_node)
    g.add_node("finalize", finalize_node)

    g.set_entry_point("planner")
    g.add_edge("planner", "research")
    g.add_conditional_edges("research", after_research,
                            {"research": "research", "analyst": "analyst"})
    g.add_edge("analyst", "reviewer")
    g.add_conditional_edges("reviewer", after_review,
                            {"finalize": "finalize", "planner": "planner"})
    g.add_edge("finalize", END)
    return g.compile()


legal_workflow = build_workflow()


def run_case(query: str, session_id: str = None) -> dict:
    """处理一次法律咨询，返回最终状态（answer / verdict / case_id ...）"""
    query = query.strip()
    if _is_greeting(query):
        return {"answer": "您好！我是 AI 法律助手，可以帮您分析案件、检索法条、评估法律风险。请描述您的问题。",
                "verdict": "PASS", "case_id": "", "case_type": "问候"}
    case_id = uuid.uuid4().hex[:12]
    return legal_workflow.invoke({
        "case_id": case_id,
        "session_id": session_id or case_id,
        "user_query": query,
    })


def _print_result(result: dict):
    """终端结果展示：裁决摘要 + 任务看板（Agent 协作过程可视化）"""
    print("\n" + "=" * 60)
    print(f"裁决: {result.get('verdict')}　|　置信度: {result.get('confidence') or 0:.2f}"
          f"　|　重规划: {result.get('replan_count', 0)} 次")
    print("=" * 60)

    case_id = result.get("case_id")
    if case_id:
        snap = Blackboard(case_id).snapshot()
        print("🗂 任务看板:")
        for t in snap.get("tasks", []):
            icon = {"pending": "⏳", "running": "🔄", "done": "✅", "failed": "❌"}.get(t["status"], "❓")
            deps = f"（依赖: {','.join(t['depends_on'])}）" if t["depends_on"] else "（并行）"
            print(f"  {icon} {t['task_id']} [{t['type']}] {t['description']} {deps}")
        if snap.get("conflicts"):
            print("⚔️ 冲突记录:")
            for c in snap["conflicts"]:
                print(f"  ⚠️ {c.get('issue', c)} → {c.get('suggestion', '')}")
        print("-" * 60)
    print(result.get("answer", ""))


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # 单次模式：python legal_flow/orchestration/workflow.py "你的问题"
        _print_result(run_case(" ".join(sys.argv[1:])))
    else:
        # 交互模式：连续提问，exit 退出
        print("⚖️ AI 虚拟律所 · 终端模式（输入 exit 退出）")
        while True:
            try:
                query = input("\n🧑 请输入法律问题: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not query or query.lower() in ("exit", "quit", "q"):
                break
            _print_result(run_case(query))
