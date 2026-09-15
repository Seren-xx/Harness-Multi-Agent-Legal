"""法律研究工具：包装混合检索实现（BM25 + 向量 + RRF + CrossEncoder 重排）。

检索算法本体在 legal_flow/runtime/retrieval.py；
本文件只做 Harness 层封装：ToolResult 规范化 + tool_retry + 自适应召回。
"""
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from legal_flow import config

_retrieval = None  # 懒加载的检索模块


def _get_retrieval():
    """懒加载检索模块（首次调用才加载 torch / modelscope，避免拖慢服务启动）"""
    global _retrieval
    if _retrieval is None:
        from legal_flow.runtime import retrieval as r
        _retrieval = r
    return _retrieval


def warmup():
    """预加载向量库 / BM25 / 重排模型。

    在并行调度前调用一次，避免多个线程同时触发模型初始化产生竞争。
    """
    r = _get_retrieval()
    r.get_vector_store()
    r.build_or_load_bm25()
    r.get_reranker()


@dataclass
class ToolResult:
    """工具调用统一封装（Harness：规范化 + 可观测）"""
    tool_name: str
    success: bool
    data: str = ""
    error: str = ""
    laws: list = field(default_factory=list)  # 命中的法律（名称 + 版本日期）


def _parse_laws(evidence: str) -> list:
    """从检索证据中解析命中的法律名与版本（来源文件名形如 中华人民共和国著作权法_20201111.md）"""
    laws = []
    for src in re.findall(r"\(([^)]+\.md)\)", evidence):
        m = re.match(r"(.+?)_(\d{8})", src)
        item = {"name": m.group(1) if m else src.replace(".md", ""),
                "date": m.group(2) if m else ""}
        if item not in laws:
            laws.append(item)
    return laws


def tool_retry(max_retries: int = 2):
    """工具重试装饰器（指数退避；失败返回失败的 ToolResult，不向上抛异常）"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    data = func(*args, **kwargs)
                    return ToolResult(func.__name__, True, data=data, laws=_parse_laws(data))
                except Exception as e:
                    if attempt < max_retries:
                        time.sleep(2 ** attempt)
                    else:
                        return ToolResult(func.__name__, False, error=str(e)[:200])
        return wrapper
    return decorator


def _avg_relevance(evidence: str) -> float:
    """从证据文本解析平均相关性分数（v1 检索输出内嵌"相关性:0.xx"标记）"""
    scores = [float(s) for s in re.findall(r"相关性:(\d+(?:\.\d+)?)", evidence)]
    return sum(scores) / len(scores) if scores else 0.0


@tool_retry(max_retries=1)
def search_articles(query: str,
                    retrieve_k: Optional[int] = None,
                    rerank_k: Optional[int] = None) -> str:
    """法条检索（自适应召回）：混合检索 + RRF 融合 + CrossEncoder 重排（复用 v1 实现）。

    首轮用聚焦参数；Top 证据平均相关性低于阈值（或结果为空）时，
    扩大召回范围并追加引导词重试，最多 MAX_RECALL_RETRIES 次（默认 2）；
    重试耗尽返回历次最优结果，仍无可用结果才抛出异常（交由 @tool_retry 兜底）。
    """
    r = _get_retrieval()
    k = retrieve_k or config.RESEARCH_RETRIEVE_K
    rk = rerank_k or config.RESEARCH_RERANK_K
    best, best_quality = "", 0.0

    for attempt in range(config.MAX_RECALL_RETRIES + 1):
        # 首轮用原查询；重试轮追加引导词提升关键词匹配
        q = query if attempt == 0 else f"{query} 关键 法条"
        evidence = r.execute_legal_search(q, retrieve_k=k, rerank_k=rk)

        quality = 0.0 if "未检索到" in evidence else _avg_relevance(evidence)
        if quality > best_quality:
            best, best_quality = evidence, quality

        if quality >= config.ADAPTIVE_RECALL_THRESHOLD and len(evidence.strip()) >= 100:
            return evidence

        if attempt < config.MAX_RECALL_RETRIES:
            k, rk = int(k * 1.5), rk * 2  # 扩大召回：召回量×1.5、重排保留×2
            print(f"⚠️ [自适应召回] 相关性 {quality:.2f} < {config.ADAPTIVE_RECALL_THRESHOLD}，"
                  f"扩大召回重试 ({attempt + 1}/{config.MAX_RECALL_RETRIES})")

    # 重试耗尽：有可用结果则返回历次最优（弱证据交由 Analyst/Reviewer 判断），否则失败
    if len(best.strip()) >= 100 and "未检索到" not in best:
        return best
    raise ValueError(f"检索质量不足: {query}")
