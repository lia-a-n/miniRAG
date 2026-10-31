import os
import re
import glob
from openai import OpenAI
import json

# 读取md文档，切分、向量化
doc_path = r"D:\XXXX\ScenicAgent\backend\kb"  # md文档的所在路径，此处仅为演示，实际使用时请根据情况修改

# 长段落按句子划分
def split_sentences(text):
    text = text.replace("\n", "")
    segs = re.split(r"([。！？!?；;])", text)
    sentences = []
    for i in range(0, len(segs)-1, 2):
        sentences.append((segs[i] + segs[i+1]).strip())
    if segs[-1].strip():
        sentences.append(segs[-1].strip())
    return [s for s in sentences if s]

def _split_long_unit(unit, max_chars):
    # 单段落超长： 先按句划分，再贪心拼回max_chars以内：
    if len(unit) <= max_chars:
        return [unit]
    pieces, cur = [], ""
    for s in split_sentences(unit):
        if len(s) > max_chars:  # 单句仍超长，按字符硬切分
            for i in range(0, len(s), max_chars):
                pieces.append(s[i:i + max_chars])
            continue
        if cur and len(cur) + len(s) > max_chars:
            pieces.append(cur)
            cur = s
        else:
            cur = s if not cur else cur + s
    if cur:
        pieces.append(cur)
    return pieces


# Step 1：切分
# chunk_text()，按固定长度切分并带 overlap。
def chunk_text(text, max_chars=400, overlap=80):
    """
    按固定长度切分并带 overlap。
    """
    # 归一化换行，按空行分段
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 按段落划分
    para_list = [p.strip() for p in text.split("\n\n") if p.strip()]

    # 所有段落展开成一个单元序列（超长的先拆分成句块），每个<=max_chars
    units_list = []
    for para in para_list:
        units_list.extend(_split_long_unit(para, max_chars))

    # 统一累积：接近上限就收块，新块带重叠滑窗
    chunks_list = []
    current_bar = ""
    for u in units_list:
        if current_bar and len(current_bar) + len(u) > max_chars:
            chunks_list.append(current_bar)
            current_bar = current_bar[-overlap:] + "\n" + u if overlap else u
        else:
            current_bar = u if not current_bar else current_bar + "\n" + u
    if current_bar:
        chunks_list.append(current_bar)
    return chunks_list

chunks_list = []
for file in glob.glob(os.path.join(doc_path, "*.md")):
    with open(file, encoding="utf-8") as f:
        chunks = chunk_text(f.read())
        chunks_list.extend(chunks)
    print("块数：", len(chunks))
    for i, c in enumerate(chunks):
        print(f"[{i}]{len(c)}字 | {c[:30]}...\n")

lengths = sorted(len(c) for c in chunks_list)
print("块数:", len(chunks_list))
print("最短/最长:", lengths[0], lengths[-1])
print("长度分布:", lengths)


# step2：向量化
client = OpenAI(
    api_key = os.getenv("DASHSCOPE_API_KEY"),
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

vector_list = []
for chunk in chunks_list:
    res =client.embeddings.create(
        model="text-embedding-v3",
        input=chunk,
        encoding_format="float"
    )
    vector_list.append(res.data[0].embedding)
    # print(res.model_dump_json())
    # print(res.data[0].embedding)
data = [{"text": c, "vec": v} for c, v in zip(chunks_list, vector_list)]
json.dump(data, open("index.json", "w", encoding="utf-8"), ensure_ascii=False)
