"""v2 可视化工作台（单页）：问答 + Agent 任务看板 + 研究发现溯源 + 冲突裁决。

运行：streamlit run legal_flow/ui/app.py
说明：进程内直接调用工作流 run_case()，无需启动任何后端服务。
"""
import sys
from pathlib import Path

# streamlit 以脚本目录为工作路径，这里把项目根目录加入 sys.path 以便导入 legal_flow
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from legal_flow.orchestration.workflow import run_case
from legal_flow.state.blackboard import Blackboard

st.set_page_config(page_title="AI 虚拟律所 · 多智能体工作台", page_icon="⚖️", layout="wide")
st.title("⚖️ AI 虚拟律所 · 多智能体工作台")
st.caption("Planner 规划 → Researcher 并行研究 → Analyst 法律分析 → Reviewer 审查仲裁")

# 会话内对话历史（进程内状态；跨会话记忆由 Redis 对话记忆负责）
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("请描述您的法律问题（例如：网购商品是假货，如何索赔？）")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("多智能体协作处理中：Planner → Research → Analyst → Reviewer ..."):
            try:
                r = run_case(query)
            except Exception as e:
                st.error(f"处理失败: {e}")
                st.stop()
        answer = r.get("answer", "")
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        c1, c2, c3 = st.columns(3)
        c1.metric("裁决", r.get("verdict"))
        c2.metric("置信度", f"{r.get('confidence') or 0:.2f}")
        c3.metric("重规划次数", r.get("replan_count", 0))

    # —— Agent 执行追溯（直接读 Blackboard 快照）——
    if r.get("case_id"):
        case = Blackboard(r["case_id"]).snapshot()

        tab_tasks, tab_findings, tab_conflicts = st.tabs(
            ["🗂 任务看板", "📚 研究发现（溯源）", "⚔️ 冲突与裁决"])

        with tab_tasks:
            st.caption(f"案件类型：{case.get('case_type', '')}　|　争点：{'；'.join(case.get('issues', []))}")
            for t in case.get("tasks", []):
                icon = {"pending": "⏳", "running": "🔄", "done": "✅", "failed": "❌"}.get(t["status"], "❓")
                deps = f"（依赖: {', '.join(t['depends_on'])}）" if t["depends_on"] else "（可并行）"
                st.write(f"{icon} **{t['task_id']}** `{t['type']}` {t['description']} {deps}")

        with tab_findings:
            for f in case.get("findings", []):
                tier = f.get("source_tier", "")
                with st.expander(f"[{tier}] {f['description']}（{f['task_id']}）"):
                    laws = "、".join(l["name"] for l in f.get("laws", [])) or "—"
                    st.markdown(f"**命中法律**：{laws}")
                    st.text(f["content"][:2000])

        with tab_conflicts:
            for c in case.get("conflicts", []):
                st.warning(f"{c.get('issue', c)}　→　{c.get('suggestion', '')}")
            for d in case.get("decisions", []):
                st.write(f"- `{d.get('agent')}` → **{d.get('verdict')}**")
            if not case.get("conflicts") and not case.get("decisions"):
                st.info("无冲突记录")
