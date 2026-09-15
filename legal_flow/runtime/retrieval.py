"""混合检索运行时：BM25 + 向量 + RRF 融合 + CrossEncoder 重排。
1. import config → from legal_flow import config
2. MODELSCOPE_CACHE 指向项目根 models/
"""
import os
import pickle

import numpy as np
import jieba
import torch

from legal_flow import config

# MODELSCOPE_CACHE 必须在导入 modelscope 前设置，模型才会下载/加载到项目根 models/
os.environ["MODELSCOPE_CACHE"] = os.path.join(config.PROJECT_ROOT, "models")

from modelscope import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

jieba.setLogLevel(jieba.logging.INFO)


class ModelScopeEmbeddings(Embeddings):
    """使用 ModelScope 的嵌入模型"""

    def __init__(self, model_name=config.EMBEDDING_MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()

    def _embed_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
        return embeddings.squeeze().numpy().tolist()

    def embed_documents(self, texts):
        return [self._embed_text(t) for t in texts]

    def embed_query(self, text):
        return self._embed_text(text)


_vector_store = None
_bm25_retriever = None


def get_vector_store():
    """获取向量数据库"""
    global _vector_store

    if _vector_store is None:
        embeddings = ModelScopeEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)

        _vector_store = Chroma(
            collection_name="legal_kb",
            embedding_function=embeddings,
            persist_directory=config.CHROMA_DB_DIR
        )

    return _vector_store


def jieba_preprocess(text: str):
    """jieba分词预处理"""
    return jieba.lcut(text)


def build_or_load_bm25():
    """构建或加载BM25索引"""
    global _bm25_retriever

    bm25_path = config.BM25_INDEX_PATH

    if os.path.exists(bm25_path):
        with open(bm25_path, "rb") as f:
            _bm25_retriever = pickle.load(f)
        print("✅ BM25加载成功")
        return _bm25_retriever

    print("🔨 构建BM25索引...")

    vector_store = get_vector_store()
    data = vector_store.get(include=["documents", "metadatas"])

    docs = [
        Document(page_content=d, metadata=m or {})
        for d, m in zip(data["documents"], data["metadatas"])
    ]

    _bm25_retriever = BM25Retriever.from_documents(
        docs,
        preprocess_func=jieba_preprocess
    )

    _bm25_retriever.k = config.RETRIEVE_K

    with open(bm25_path, "wb") as f:
        pickle.dump(_bm25_retriever, f)

    print("✅ BM25构建完成")
    return _bm25_retriever


class ModelScopeCrossEncoder:
    """使用 ModelScope 的CrossEncoder 重排模型"""

    def __init__(self, model_name=config.RERANKER_MODEL_NAME, device=config.DEVICE):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True)
        self.model.to(device)
        self.model.eval()

    def predict(self, pairs):
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
