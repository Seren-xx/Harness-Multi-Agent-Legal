"""Analyst Agent（法律分析）：读取 Blackboard 全部研究发现，
按 IRAC（事实 → 规则 → 要件 → 结论）产出法律分析草稿。
不再是"拿证据直接写答案"，是先综合多个研究任务的发现，建立完整的法律推理链。
"""
from legal_flow.runtime.llm import call_llm
from legal_flow.state.blackboard import Blackboard

ANALYST_PROMPT = """你是一位顶尖的中国执业律师。团队的研究员已完成多项研究，请基于【研究发现】撰写法律分析意见。

要求：
1. 按 IRAC 结构组织：案件事实 → 适用规则（引用法条原文）→ 要件分析 → 结论。
2. 每个结论后标注来源层级，只能是以下三种之一：
   〔法条原文〕结论直接来自检索到的条文
   〔检索验证〕结论由检索证据推理得出
   〔模型知识〕结论来自模型内部知识，未经检索验证
3. 只引用【研究发现】中出现的法条，不得编造；证据不足的要点，明确写入"待补充信息"。
4. 末尾附"风险提示"：指出当事人可能面临的不利因素。"""


def format_findings(findings: list) -> str:
    """把研究发现格式化为可供下游 Agent / 审查官阅读的文本"""
    return "\n\n".join(
        f"### [{f['source_tier']}] {f['description']}（任务 {f['task_id']}）\n{f['content']}"
        for f in findings
    ) or "（暂无研究发现）"


def analyst_node(state: dict) -> dict:
    case_id = state["case_id"]
    snap = Blackboard(case_id).snapshot()

    draft = call_llm(
        [
            {"role": "system", "content": ANALYST_PROMPT},
            {"role": "user",
             "content": f"用户问题：{state['user_query']}\n\n【研究发现】\n{format_findings(snap['findings'])}"},
        ],
        model="dashscope", temperature=0.1,
    )
    print(f"⚖️ [Analyst] 完成法律分析草稿（基于 {len(snap['findings'])} 项研究发现）")
    return {"draft": draft}
