import os
import torch
import numpy as np
from typing import List, Tuple

os.environ["MODELSCOPE_CACHE"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

from modelscope import AutoTokenizer, AutoModelForSequenceClassification

import config
from step03_build_vector_db_ChromaDB import get_vector_store, build_or_load_bm25


class ModelScopeCrossEncoder:
    """使用 ModelScope 的CrossEncoder 重排模型"""
    
    def __init__(self, model_name=config.RERANKER_MODEL_NAME, device=config.DEVICE):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True)
        self.model.to(device)
        self.model.eval()
    
    def predict(self, pairs: List[List[str]]) -> np.ndarray:
        """预测文本对的相关性分数"""
        scores = []
        for query, passage in pairs:
            inputs = self.tokenizer(
                query, passage, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                score = outputs.logits.squeeze().cpu().numpy()
                scores.append(score)
        
        return np.array(scores)


_reranker = None


def get_reranker():
    """获取重排模型"""
    global _reranker
    if _reranker is None:
        _reranker = ModelScopeCrossEncoder(
            model_name=config.RERANKER_MODEL_NAME,
            device=config.DEVICE
        )
    return _reranker


def hybrid_search(query: str, k=10, rerank_k=5):
    """混合检索（BM25 + 向量）+ RRF融合 + CrossEncoder重排"""

    vector_store = get_vector_store()
    bm25 = build_or_load_bm25()
    reranker = get_reranker()

    vec_docs = vector_store.similarity_search(query, k=k)
    bm25_docs = bm25.invoke(query)

    def rrf(docs, weight=60):
        """RRF融合算法"""
        score = {}
        for i, d in enumerate(docs):
            score[d.page_content] = score.get(d.page_content, 0) + 1 / (i + 1 + weight)
        return score

    scores = rrf(vec_docs)
    for i, d in enumerate(bm25_docs):
        scores[d.page_content] = scores.get(d.page_content, 0) + 1 / (i + 1 + 60)

    all_docs = {d.page_content: d for d in vec_docs + bm25_docs}

    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    fused_docs = [all_docs[k] for k, _ in merged]

    if not fused_docs:
        return [], []

    pairs = [[query, d.page_content] for d in fused_docs]
    rerank_scores = reranker.predict(pairs)

    # 将分数归一化到 0~1 之间
    if len(rerank_scores) > 0:
        min_score = rerank_scores.min()
        max_score = rerank_scores.max()
        if max_score > min_score:
            normalized_scores = (rerank_scores - min_score) / (max_score - min_score)
        else:
            normalized_scores = np.ones_like(rerank_scores) * 0.5
    else:
        normalized_scores = np.array([])

    ranked = [
        (d, s) for d, s in sorted(
            zip(fused_docs, normalized_scores),
            key=lambda x: x[1],
            reverse=True
        )
    ]

    top_docs = [d for d, s in ranked[:rerank_k]]
    top_scores = [float(s) for d, s in ranked[:rerank_k]]
    avg_score = sum(top_scores) / len(top_scores) if top_scores else 0

    return top_docs, top_scores, avg_score


def execute_legal_search(query: str, retrieve_k=300, rerank_k=100):
    """执行法律检索，返回格式化的证据文本"""
    print(f"🔍 检索查询: {query}")

    docs, scores, avg_score = hybrid_search(query, k=retrieve_k, rerank_k=rerank_k)

    if not docs:
        return "未检索到相关法律依据。"

    evidence_parts = []
    for i, (doc, score) in enumerate(zip(docs, scores)):
        source = doc.metadata.get("source", "未知")
        evidence_parts.append(f"[证据{i+1}] 相关性:{score:.3f} ({source})\n{doc.page_content}")

    result = "\n\n".join(evidence_parts)
    print(f"✅ 检索到 {len(docs)} 条证据，平均相关性: {avg_score:.2f}")

    return result
