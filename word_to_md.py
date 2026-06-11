import os
import re
import mammoth
from tqdm import tqdm
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def clean_text(text: str) -> str:
    """清理Markdown文本中的噪声"""
    if not text:
        return ""

    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.replace('\u3000', ' ').replace('\u200b', '')
    text = re.sub(r'[ \t]+', ' ', text)

    return text.strip()


def parse_document(file_path: str) -> str:
    """将Word文档转换为Markdown"""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".docx":
        with open(file_path, "rb") as f:
            result = mammoth.convert_to_markdown(f)
            text = result.value
    else:
        raise ValueError(f"不支持的文件类型: {ext}")

    return clean_text(text)


def run_etl():
    """ETL主流程：文档 -> Markdown文本"""
    print("=" * 50)
    print("🚀 文档解析与结构化预处理开始")

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    files = [f for f in os.listdir(RAW_DATA_DIR) if f.lower().endswith(".docx")]

    if not files:
        print("⚠️ 未找到.docx文件")
        return

    success = 0

    for file in tqdm(files):
        try:
            src_path = os.path.join(RAW_DATA_DIR, file)
            name = os.path.splitext(file)[0]
            out_path = os.path.join(PROCESSED_DATA_DIR, f"{name}.md")

            text = parse_document(src_path)

            if text:
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(text)
                success += 1

        except Exception as e:
            print(f"\n❌ 处理失败 {file}: {e}")

    print(f"\n✅ 完成：{success}/{len(files)}")


if __name__ == "__main__":
    run_etl()