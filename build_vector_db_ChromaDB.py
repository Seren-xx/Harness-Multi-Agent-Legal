import os
import json
import hashlib
import pickle
import jieba
from tqdm import tqdm

os.environ["MODELSCOPE_CACHE"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

from modelscope import AutoModel, AutoTokenizer
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

import config
from step02_hierarchical_chunking import chunk_document

jieba.setLogLevel(jieba.logging.INFO)


class ModelScopeEmbeddings(Embeddings):
    """使用 ModelScope 的嵌入模型"""

    def __init__(self, model_name=config.EMBEDDING_MODEL_NAME):
        import torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()
        self.torch = torch

    def _embed_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with self.torch.no_grad():
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


def rebuild_bm25():
    """重建BM25索引"""
    global _bm25_retriever
    vector_store = get_vector_store()
    data = vector_store.get(include=["documents", "metadatas"])
    docs = [Document(page_content=d, metadata=m or {}) for d, m in zip(data["documents"], data["metadatas"])]
    _bm25_retriever = BM25Retriever.from_documents(docs, preprocess_func=jieba_preprocess)
    _bm25_retriever.k = config.RETRIEVE_K
    with open(config.BM25_INDEX_PATH, "wb") as f:
        pickle.dump(_bm25_retriever, f)
    print("🔄 BM25索引已更新")


def get_file_hash(file_path: str) -> str:
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def load_doc_hashes():
    if os.path.exists(config.DOC_HASH_PATH):
        with open(config.DOC_HASH_PATH, 'r') as f:
            return set(json.load(f))
    return set()


def save_doc_hash(hash_value: str):
    hashes = load_doc_hashes()
    hashes.add(hash_value)
    with open(config.DOC_HASH_PATH, 'w') as f:
        json.dump(list(hashes), f)


def add_document_to_db(docx_path: str) -> tuple:
    """添加文档到知识库（带去重）"""
    if config.ENABLE_DEDUP:
        file_hash = get_file_hash(docx_path)
        if file_hash in load_doc_hashes():
            return False, "文档已存在，跳过添加"
    
    import mammoth
    with open(docx_path, "rb") as f:
        result = mammoth.convert_to_markdown(f)
        raw_md = result.value.strip()
    
    if not raw_md:
        return False, "文档无内容"
    
    md_name = os.path.basename(docx_path).replace('.docx', '.md')
    md_path = os.path.join(config.PROCESSED_DATA_DIR, md_name)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(raw_md)
    
    chunks = chunk_document(md_path)
    if not chunks:
        return False, "文档切分后无有效内容"
    
    vector_store = get_vector_store()
    docs = [Document(page_content=c, metadata={"source": md_name}) for c in chunks]
    ids = [hashlib.md5(c.encode()).hexdigest() for c in chunks]
    vector_store.add_documents(docs, ids=ids)
    
    if config.ENABLE_DEDUP:
        save_doc_hash(file_hash)
    
    rebuild_bm25()
    return True, f"成功添加 {md_name}，新增 {len(chunks)} 个知识片段"


def build_full_db():
    """构建完整向量库"""
    vector_store = get_vector_store()

    md_files = [
        f for f in os.listdir(config.PROCESSED_DATA_DIR)
        if f.endswith(".md")
    ]

    if not md_files:
        print("⚠️ 未找到Markdown文件，请先运行step01和step02")
        return

    total = 0

    for f in tqdm(md_files):
        path = os.path.join(config.PROCESSED_DATA_DIR, f)

        chunks = chunk_document(path)

        docs = [
            Document(page_content=c, metadata={"source": f})
            for c in chunks
        ]

        ids = [
            hashlib.md5(c.encode()).hexdigest()
            for c in chunks
        ]

        vector_store.add_documents(docs, ids=ids)
        total += len(docs)

    print(f"✅ 向量库构建完成：{total} chunks")

    build_or_load_bm25()


if __name__ == "__main__":
    build_full_db()