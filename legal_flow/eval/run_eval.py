"""评测脚本：引用命中率 / 要点覆盖率 / 裁决通过率 / 延迟 / 重规划次数。

运行（需配置 LLM API Key 且已构建知识库）：
    python -m legal_flow.eval.run_eval

输出：legal_flow/eval/eval_outputs.json（评测结果可提交仓库，作为效果证据）
"""
import json
import time
from pathlib import Path

from legal_flow.orchestration.workflow import run_case
from legal_flow.state.blackboard import Blackboard

HERE = Path(__file__).parent
OUT = HERE / "eval_outputs.json"


def load_eval_set() -> list:
    with open(HERE / "eval_set.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def cited_laws(case_id: str) -> set:
    """从案件 Blackboard 中提取研究阶段实际命中的法律名"""
    if not case_id:
        return set()
    snap = Blackboard(case_id).snapshot()
    return {law["name"] for f in snap["findings"] for law in f.get("laws", [])}


def main():
    rows = []
    for item in load_eval_set():
        t0 = time.time()
        try:
            result = run_case(item["query"])
            answer = result.get("answer", "")
            laws = cited_laws(result.get("case_id", ""))
            hit = sorted(set(item["expected_laws"]) & laws)
            kw = [k for k in item["expected_keywords"] if k in answer]
            rows.append({
                **item,
                "status": "ok",
                "verdict": result.get("verdict"),
                "law_hit": hit,
                "law_hit_rate": round(len(hit) / len(item["expected_laws"]), 3),
                "keyword_hit": kw,
                "keyword_coverage": round(len(kw) / len(item["expected_keywords"]), 3),
                "replans": result.get("replan_count", 0),
                "latency_s": round(time.time() - t0, 1),
            })
        except Exception as e:
            rows.append({**item, "status": "error", "error": str(e)[:200],
                         "latency_s": round(time.time() - t0, 1)})
        print(f"[{item['id']}] {rows[-1]['status']} "
              f"law_hit={rows[-1].get('law_hit', [])} {rows[-1]['latency_s']}s")

    ok = [r for r in rows if r["status"] == "ok"]
    summary = {
        "total": len(rows),
        "ok": len(ok),
        "citation_hit_rate": round(sum(r["law_hit_rate"] for r in ok) / len(ok), 3) if ok else None,
        "keyword_coverage": round(sum(r["keyword_coverage"] for r in ok) / len(ok), 3) if ok else None,
        "pass_rate": round(sum(1 for r in ok if r["verdict"] == "PASS") / len(ok), 3) if ok else None,
        "avg_latency_s": round(sum(r["latency_s"] for r in ok) / len(ok), 1) if ok else None,
    }
    OUT.write_text(
        json.dumps({"summary": summary, "results": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n===== 评测汇总 =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"结果已写入 {OUT}")


if __name__ == "__main__":
    main()
