"""Reviewer / Arbiter Agent：结构化审查 + 冲突仲裁。

- 三态裁决而非 PASS/FAIL：PASS / REPLAN / ESCALATE
- 冲突仲裁（Arbiter）：确定性冲突检测（法条版本冲突）+ LLM 语义冲突判断
"""
import time

from legal_flow import config
from legal_flow.runtime.llm import call_llm_json
from legal_flow.agents.analyst import format_findings
from legal_flow.state.blackboard import Blackboard

REVIEW_PROMPT = """【只读审查模式】你是合规审查官兼冲突仲裁人。只能评估，不得改写证据。

请审查以下【律师分析草稿】与【研究发现】：
1. 事实是否与法条证据一致；引用是否准确、有无编造。
2. 推理链是否完整（事实 → 规则 → 要件 → 结论）。
3. 多个研究发现之间是否存在矛盾（同一问题不同结论、适用条件冲突）。
4. 是否存在关键信息缺失（导致结论不可靠）。

只输出 JSON：
{"pass": true 或 false,
 "confidence": 0.0~1.0,
 "feedback": "具体意见",
 "missing_info": ["缺失的关键信息"],
 "conflicts": [{"between": "任务A vs 任务B", "issue": "矛盾点", "suggestion": "采纳哪方 / 需补充什么"}]}"""


def detect_version_conflicts(findings: list) -> list:
    """确定性冲突检测：同一部法律命中多个版本（不同修订日期）→ 法条版本冲突"""
    law_dates = {}
    for f in findings:
        for law in f.get("laws", []):
            if law.get("date"):
                law_dates.setdefault(law["name"], set()).add(law["date"])
    return [
        {"type": "法条版本冲突", "law": name, "versions": sorted(dates),
         "issue": f"检索同时命中《{name}》的多个版本",
         "suggestion": "以最新修订版本为准并复核条文"}
        for name, dates in law_dates.items() if len(dates) > 1
    ]


def reviewer_node(state: dict) -> dict:
    case_id = state["case_id"]
    bb = Blackboard(case_id)
    snap = bb.snapshot()

    # 1) 确定性冲突检测（法条版本冲突）
    conflicts = detect_version_conflicts(snap["findings"])

    # 2) LLM 结构化审查（只读）+ 语义冲突判断
    review = call_llm_json(
        [{"role": "user",
          "content": f"{REVIEW_PROMPT}\n\n【律师分析草稿】\n{state.get('draft', '')}"
                     f"\n\n【研究发现】\n{format_findings(snap['findings'])}"}],
        model="deepseek", temperature=0,
    )
    conflicts.extend(review.get("conflicts", []))
    bb.add_conflicts(conflicts)

    confidence = float(review.get("confidence", 0) or 0)
    passed = bool(review.get("pass")) and confidence >= config.CONFIDENCE_THRESHOLD
    missing = [m for m in review.get("missing_info", []) if m]
    replans = state.get("replan_count", 0)

    # 三态裁决：
    #   PASS     审查通过 → 定稿
    #   REPLAN   有明确缺失且未超预算 → 回 Planner 补任务（有界重规划）
    #   ESCALATE 无法通过且无法/无需再规划 → 带风险提示升级输出
    if passed:
        verdict = "PASS"
    elif missing and replans < config.MAX_REPLANS:
        verdict = "REPLAN"
    else:
        verdict = "ESCALATE"

    bb.add_decision({"agent": "reviewer", "verdict": verdict, "confidence": confidence,
                     "feedback": review.get("feedback", ""), "at": time.time()})
    print(f"🔍 [Reviewer] 裁决: {verdict}（置信度 {confidence:.2f}，冲突 {len(conflicts)} 项）")

    return {
        "verdict": verdict,
        "review_feedback": review.get("feedback", ""),
        "missing_info": missing,
        "conflicts": conflicts,
        "confidence": confidence,
        "replan_count": replans + (1 if verdict == "REPLAN" else 0),
    }
