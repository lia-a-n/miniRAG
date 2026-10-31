import json
import numpy as np
from openai import OpenAI
import os

# 检索（余弦相似度）与答案生成（调用LLM）

# 读取索引
def load_index(path="index.json"):
    data = json.load(open(path, encoding="utf-8"))
    chunks = [d["text"]for d in data]
    doc_vecs = np.stack([np.array(d["vec"], dtype=np.float32) for d in data])
    return chunks, doc_vecs

# embedding 同build_index.py中的向量化
def embed(text):
    client = OpenAI(
            api_key = os.getenv("DASHSCOPE_API_KEY"),
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    res =client.embeddings.create(
        model="text-embedding-v3",
        input=text,
        encoding_format="float"
    )
    return res.data[0].embedding

# 归一化： 归一化后，余弦相似度退化为点积
def normalize(v):
    v = np.array(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-8)

# 检索
def retrieve(query, chunks, doc_vecs, top_k=3):
    query_vec = normalize(embed(query))
    similarity = np.dot(doc_vecs, query_vec)
    top_idxs = np.argsort(similarity)[-top_k:][::-1]
    return [(i, chunks[i], float(similarity[i]))  for i in top_idxs]

# 拼接prompt、附加引用来源
def build_prompt(query, hits):
    sources = ""
    for i, (idx, chunk, sim) in enumerate(hits, 1):
        sources += f"[文档{i}]（来源块：#{idx}）\n{chunk}\n\n"
    prompt = f"""
        你是景区智能客服助手。请仅根据参考资料回答，并在句中标注引用的文档编号（如[文档1])。\n
        若资料中无相关信息，请明确告知无法回答，严禁编造。\n\n
        【参考资料】\n
        {sources}\n
        【用户问题】\n
        {query}
    """
    return prompt

# 生成（调用LLM）
def generate(prompt):
    client = OpenAI(
            api_key = os.getenv("DASHSCOPE_API_KEY"),
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    res = client.chat.completions.create(
        model="qwen-flash",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.5
    )
    return res.choices[0].message.content


# 运行
if __name__ == "__main__":
    chunks, doc_vecs = load_index(path="index.json")
    doc_vecs_normalized = []
    for vec in doc_vecs:
        vec_normalized = normalize(vec)
        doc_vecs_normalized.append(vec_normalized)
    while(True):
        query = input("请输入问题：")
        hits = retrieve(query, chunks, doc_vecs_normalized, top_k=3)
        print("\n检索到的块：\n")
        for i, (idx, chunk, sim) in enumerate(hits, 1):
            print(f"[文档{i}]（来源块：#{idx}）\n{chunk}\n相似度：{sim:.4f}\n")
        prompt = build_prompt(query, hits)
        answer = generate(prompt)
        print("\n回答：\n", answer)
