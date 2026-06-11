import os
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import PROCESSED_DATA_DIR


def normalize_structure(text: str) -> str:
    """将非标准法律格式统一成 #章 ##条结构"""
    text = re.sub(r'(第[一二三四五六七八九十百零]+章)', r'\n# \1', text)
    text = re.sub(r'(第[一二三四五六七八九十百零]+条)', r'\n## \1', text)
    text = re.sub(r'<a.*?>', '', text)
    return text


def chunk_document(file_path: str):
    """按章-条 + 语义窗口切分"""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    text = normalize_structure(text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", "。", "；", "，"]
    )

    chunks = splitter.split_text(text)
    chunks = [c.strip() for c in chunks if len(c.strip()) > 5]

    return chunks


def run_chunking():
    """批量处理入口"""
    print("=" * 50)
    print("✂️ 结构化分块开始")

    files = [f for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith(".md")]

    if not files:
        print("⚠️ 未找到Markdown文件，请先运行step01")
        return

    total_chunks = 0

    for file in files:
        path = os.path.join(PROCESSED_DATA_DIR, file)
        chunks = chunk_document(path)
        total_chunks += len(chunks)
        print(f"📄 {file} -> {len(chunks)} chunks")

    print(f"\n✅ 总计：{total_chunks} chunks")


if __name__ == "__main__":
    run_chunking()