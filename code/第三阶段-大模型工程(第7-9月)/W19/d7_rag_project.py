"""
W19-D7 RAG完整项目 (RAG Project)
==================================
完整RAG项目: 个人知识库, 多格式文档支持,
混合检索+重排序, 对话式问答, Web界面概念
"""

import sys
import re
import math
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
from collections import Counter, defaultdict, OrderedDict

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W19-D7 RAG完整项目 (RAG Project)")
print("=" * 60)

# ============================================================
# 1. 项目概述
# ============================================================
print("\n--- 1. 项目概述: 个人知识库系统 ---")
print("""
  功能:
    1) 多格式文档导入 (TXT/PDF模拟/Markdown)
    2) 自动文本切分和索引
    3) 混合检索 (BM25 + 向量) + RRF融合
    4) Cross-Encoder重排序
    5) 对话式问答 (支持追问)
    6) 来源追溯
""")


# ============================================================
# 2. 核心组件
# ============================================================

# --- 嵌入模型 ---
class EmbeddingModel:
    """文本嵌入模型 (模拟)"""

    def __init__(self, dim=64):
        self.dim = dim
        self.keyword_vectors = {
            "机器学习": [1, .8, .3, .1], "深度学习": [.9, .7, .5, .2],
            "神经网络": [.8, .6, .4, .3], "自然语言": [.3, .2, 1, .8],
            "Transformer": [.5, .4, .7, .6], "RAG": [.3, .4, .5, .7],
            "模型": [.4, .5, .4, .4], "训练": [.7, .5, .2, .1],
            "Python": [.7, .5, .1, .2], "向量": [.4, .3, .5, .8],
            "检索": [.5, .3, .4, .6], "生成": [.6, .4, .3, .5],
            "微调": [.6, .5, .3, .2], "LoRA": [.5, .4, .3, .2],
            "注意力": [.5, .4, .6, .5], "量化": [.3, .2, .3, .4],
        }

    def embed(self, text: str) -> np.ndarray:
        rng = np.random.RandomState(abs(hash(text)) % (2**31))
        base = rng.randn(self.dim) * 0.3
        semantic = np.zeros(self.dim)
        for kw, vec in self.keyword_vectors.items():
            if kw in text:
                semantic[:4] += np.array(vec)
        combined = base + semantic * 0.7
        return combined / (np.linalg.norm(combined) + 1e-10)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        return np.array([self.embed(t) for t in texts])


# --- BM25检索 ---
class BM25Retriever:
    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b

    def fit(self, docs: List[str]):
        self.docs = docs
        self.tokenized = [re.findall(r'[一-鿿]+|[a-zA-Z]+', d.lower()) for d in docs]
        self.n = len(docs)
        self.avgdl = np.mean([len(t) for t in self.tokenized]) if self.tokenized else 1
        self.df = Counter()
        for tokens in self.tokenized:
            for t in set(tokens):
                self.df[t] += 1

    def search(self, query: str, top_k=10) -> List[Tuple[int, float]]:
        q_tokens = re.findall(r'[一-鿿]+|[a-zA-Z]+', query.lower())
        scores = []
        for i, doc_tokens in enumerate(self.tokenized):
            tf = Counter(doc_tokens)
            score = 0
            for t in q_tokens:
                if t in tf:
                    idf = math.log((self.n - self.df.get(t, 0) + 0.5) / (self.df.get(t, 0) + 0.5) + 1)
                    f = tf[t]
                    score += idf * f * 2.5 / (f + 1.5 * (0.25 + 0.75 * len(doc_tokens) / self.avgdl))
            scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# --- 向量检索 ---
class VectorRetriever:
    def __init__(self, embedding_model: EmbeddingModel):
        self.model = embedding_model
        self.vectors = None
        self.docs = []

    def fit(self, docs: List[str]):
        self.docs = docs
        self.vectors = self.model.embed_batch(docs)

    def search(self, query: str, top_k=10) -> List[Tuple[int, float]]:
        q_vec = self.model.embed(query)
        sims = np.dot(self.vectors, q_vec)
        top_indices = np.argsort(sims)[-top_k:][::-1]
        return [(int(i), float(sims[i])) for i in top_indices]


# --- RRF融合 ---
def rrf_fuse(result_lists: List[List[Tuple]], k=60) -> List[Tuple[int, float]]:
    rrf_scores = defaultdict(float)
    for results in result_lists:
        for rank, (doc_id, _) in enumerate(results):
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


# --- 重排序器 ---
class CrossEncoderReranker:
    """模拟Cross-Encoder重排序"""

    def rerank(self, query: str, docs: List[str], top_k=5) -> List[Tuple[int, float]]:
        scores = []
        for i, doc in enumerate(docs):
            q_words = set(re.findall(r'[一-鿿]+|[a-zA-Z]+', query.lower()))
            d_words = set(re.findall(r'[一-鿿]+|[a-zA-Z]+', doc.lower()))
            overlap = len(q_words & d_words)
            score = overlap / max(len(q_words), 1) + np.random.RandomState(i).rand() * 0.2
            scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ============================================================
# 3. 个人知识库系统
# ============================================================
print("\n--- 3. 个人知识库系统实现 ---")


class PersonalKnowledgeBase:
    """完整个人知识库系统"""

    def __init__(self, chunk_size=200, chunk_overlap=40):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = EmbeddingModel(dim=64)
        self.bm25 = BM25Retriever()
        self.vector_retriever = VectorRetriever(self.embedding_model)
        self.reranker = CrossEncoderReranker()

        self.chunks = []
        self.sources = {}
        self.conversation_history = []

    def _split_text(self, text: str, source: str) -> List[Dict]:
        """文本切分"""
        paragraphs = re.split(r'\n\n+', text)
        chunks = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            # 按句子组切分
            sentences = re.split(r'(?<=[。！？.!?])\s*', para)
            current = ""
            for sent in sentences:
                if len(current) + len(sent) <= self.chunk_size:
                    current += sent
                else:
                    if current.strip():
                        chunks.append({
                            'text': current.strip(),
                            'source': source,
                            'length': len(current.strip()),
                        })
                    current = sent
            if current.strip():
                chunks.append({
                    'text': current.strip(),
                    'source': source,
                    'length': len(current.strip()),
                })
        return chunks

    def add_document(self, text: str, source: str):
        """添加文档"""
        new_chunks = self._split_text(text, source)
        self.chunks.extend(new_chunks)
        self.sources[source] = len(new_chunks)
        self._rebuild_index()
        print(f"  添加文档 '{source}': {len(new_chunks)} 个chunks")

    def _rebuild_index(self):
        """重建索引"""
        texts = [c['text'] for c in self.chunks]
        self.bm25.fit(texts)
        self.vector_retriever.fit(texts)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """混合检索 + 重排序"""
        # Stage 1: BM25 + 向量检索
        bm25_results = self.bm25.search(query, top_k=10)
        dense_results = self.vector_retriever.search(query, top_k=10)

        # Stage 2: RRF融合
        fused = rrf_fuse([bm25_results, dense_results], k=60)
        fused_top = fused[:10]

        # Stage 3: 重排序
        candidate_docs = [self.chunks[doc_id]['text'] for doc_id, _ in fused_top]
        reranked = self.reranker.rerank(query, candidate_docs, top_k=top_k)

        # 组装结果
        results = []
        for local_idx, score in reranked:
            doc_id = fused_top[local_idx][0]
            chunk = self.chunks[doc_id]
            results.append({
                'text': chunk['text'],
                'source': chunk['source'],
                'score': score,
                'chunk_id': doc_id,
            })

        return results

    def ask(self, question: str, top_k: int = 5) -> Dict:
        """问答"""
        # 记录对话历史
        self.conversation_history.append({'role': 'user', 'content': question})

        # 检索
        results = self.search(question, top_k=top_k)

        # 构建上下文
        context_parts = []
        for i, r in enumerate(results):
            context_parts.append(f"[来源{i+1}: {r['source']}]\n{r['text']}")

        context = "\n\n---\n\n".join(context_parts)

        # 模拟生成答案
        answer = self._generate_answer(question, results)

        self.conversation_history.append({'role': 'assistant', 'content': answer})

        return {
            'question': question,
            'answer': answer,
            'sources': [{'text': r['text'][:50] + '...', 'source': r['source'], 'score': r['score']} for r in results],
            'context_length': len(context),
        }

    def _generate_answer(self, question: str, results: List[Dict]) -> str:
        """模拟LLM生成答案"""
        if not results:
            return "抱歉，没有找到相关信息。"

        best = results[0]
        return (f"根据知识库中的资料(来源: {best['source']}):\n"
                f"{best['text']}\n\n"
                f"以上信息回答了'{question}'的相关内容。")

    def get_stats(self) -> Dict:
        return {
            'total_chunks': len(self.chunks),
            'total_sources': len(self.sources),
            'sources': self.sources,
            'conversation_turns': len(self.conversation_history) // 2,
        }


# ============================================================
# 4. 构建知识库
# ============================================================
print("\n--- 4. 构建知识库 ---")

kb = PersonalKnowledgeBase(chunk_size=150, chunk_overlap=30)

# 添加多格式文档
documents = {
    "AI基础.txt": """机器学习是人工智能的核心子领域，通过算法从数据中自动学习模式。
监督学习使用标注数据训练分类器，如线性回归、决策树和支持向量机。
无监督学习发现数据中的隐藏结构，如K-means聚类和PCA降维。
强化学习通过与环境交互获得奖励来学习最优策略。""",

    "深度学习.md": """# 深度学习概述

## 基本概念
深度学习使用多层神经网络进行特征学习。反向传播算法是训练神经网络的核心。

## 主要架构
- CNN(卷积神经网络): 用于图像处理，如ResNet、VGG
- RNN(循环神经网络): 用于序列数据，如LSTM、GRU
- Transformer: 基于注意力机制，如BERT、GPT

## 训练技巧
学习率调度、批量归一化、Dropout正则化是常用的训练优化方法。""",

    "RAG技术.pdf": """检索增强生成(RAG)是一种结合信息检索和文本生成的技术。

RAG系统架构:
1. 文档处理: 加载文档、文本切分、向量化
2. 向量存储: 使用向量数据库存储嵌入
3. 检索阶段: BM25+向量混合检索，RRF融合
4. 重排序: Cross-Encoder对候选文档重排
5. 生成阶段: LLM基于检索上下文生成答案

RAG优势: 减少幻觉、实时更新知识、可追溯来源。""",

    "Python工具.txt": """Python是AI开发最常用的编程语言。

数据处理: NumPy(数组计算), Pandas(数据分析), Matplotlib(可视化)
机器学习: Scikit-learn(传统ML), XGBoost(梯度提升)
深度学习: PyTorch(研究), TensorFlow(工业), JAX(高性能)
预训练模型: HuggingFace Transformers, LangChain
向量数据库: FAISS, ChromaDB, Milvus""",
}

for source, content in documents.items():
    kb.add_document(content, source)

stats = kb.get_stats()
print(f"\n  知识库统计:")
for k, v in stats.items():
    print(f"    {k}: {v}")

# ============================================================
# 5. 对话式问答
# ============================================================
print("\n--- 5. 对话式问答 ---")

questions = [
    "什么是机器学习？有哪些主要类型？",
    "深度学习的主要架构有哪些？",
    "RAG系统的架构是怎样的？",
    "Python在AI开发中有哪些常用工具？",
    "什么是Transformer架构？",
]

for q in questions:
    result = kb.ask(q, top_k=3)
    print(f"\n  Q: {result['question']}")
    print(f"  A: {result['answer'][:100]}...")
    print(f"  来源: {len(result['sources'])}个文档片段")
    for s in result['sources']:
        print(f"    [{s['source']}] score={s['score']:.3f}: {s['text']}")

# ============================================================
# 6. 可视化
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 子图1: 系统架构图
ax1 = axes[0]
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('个人知识库系统架构', fontsize=14, fontweight='bold')

# 文档输入
input_items = [(1.5, 9, "TXT"), (3.5, 9, "PDF"), (5.5, 9, "MD"), (7.5, 9, "HTML")]
for x, y, label in input_items:
    ax1.add_patch(plt.Rectangle((x - 0.6, y - 0.3), 1.2, 0.6,
                                  facecolor='#3498db', edgecolor='black', alpha=0.4))
    ax1.text(x, y, label, ha='center', va='center', fontsize=9, fontweight='bold')

# 处理流程
flow_items = [
    (5, 7.5, "文本切分 + 向量化", '#2ecc71'),
    (2.5, 6.0, "BM25索引", '#e74c3c'),
    (7.5, 6.0, "向量索引", '#3498db'),
    (5, 4.5, "RRF融合 + 重排序", '#9b59b6'),
    (5, 3.0, "LLM生成 + 来源追溯", '#e67e22'),
    (5, 1.5, "对话式问答界面", '#1abc9c'),
]

for x, y, text, color in flow_items:
    ax1.add_patch(plt.Rectangle((x - 2, y - 0.4), 4.0, 0.8,
                                  facecolor=color, edgecolor='black', alpha=0.3, linewidth=2))
    ax1.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')

# 箭头
arrows = [
    ((3, 8.7), (5, 7.9)), ((5, 7.1), (2.5, 6.4)), ((5, 7.1), (7.5, 6.4)),
    ((2.5, 5.6), (5, 4.9)), ((7.5, 5.6), (5, 4.9)),
    ((5, 4.1), (5, 3.4)), ((5, 2.6), (5, 1.9)),
]
for start, end in arrows:
    ax1.annotate('', xy=end, xytext=start,
                 arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# 子图2: 各来源检索命中分布
ax2 = axes[1]
sources = list(documents.keys())
short_names = [s.split('.')[0] for s in sources]

# 模拟各来源被检索到的频率
hits_per_source = {
    "AI基础.txt": 8,
    "深度学习.md": 12,
    "RAG技术.pdf": 10,
    "Python工具.txt": 6,
}

names = list(hits_per_source.keys())
short = [n.split('.')[0] for n in names]
hits = list(hits_per_source.values())
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

bars = ax2.bar(range(len(names)), hits, color=colors, edgecolor='black', alpha=0.85)
ax2.set_xticks(range(len(names)))
ax2.set_xticklabels(short, fontsize=10)
ax2.set_ylabel('被检索次数', fontsize=12)
ax2.set_title('各文档被检索频率', fontsize=14, fontweight='bold')
for bar, val in zip(bars, hits):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             str(val), ha='center', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W19/rag_project.png', dpi=150, bbox_inches='tight')
print("\n  图表已保存: rag_project.png")
plt.close()

# ============================================================
# 7. Web界面概念
# ============================================================
print("\n--- 7. Web界面概念 ---")
print("""
  可选的Web框架:
    1) Streamlit:   最简单, 适合数据应用
    2) Gradio:      适合模型Demo
    3) FastAPI + React: 适合生产环境
    4) Chainlit:    专为对话AI设计

  Streamlit示例:
    import streamlit as st
    st.title("个人知识库")
    question = st.text_input("请输入问题:")
    if question:
        result = kb.ask(question)
        st.write(result['answer'])
        with st.expander("查看来源"):
            for s in result['sources']:
                st.write(f"[{s['source']}] {s['text']}")

  Gradio示例:
    import gradio as gr
    def answer(question):
        result = kb.ask(question)
        return result['answer']
    gr.Interface(fn=answer, inputs="text", outputs="text").launch()
""")

# ============================================================
# 8. 项目扩展方向
# ============================================================
print("\n--- 8. 项目扩展方向 ---")
print("""
  1) 多模态RAG: 支持图片/表格检索
  2) 知识图谱: 结构化知识存储和推理
  3) Agent模式: 自适应检索和推理
  4) 用户反馈: 收集反馈持续优化
  5) 权限控制: 不同用户访问不同知识库
  6) 增量更新: 实时添加新文档
  7) 多语言: 支持中英日韩等多语言
  8) 部署优化: Docker容器化, GPU加速
""")

print("\n" + "=" * 60)
print("W19-D7 完成! 本节实现了一个完整的个人知识库RAG系统")
print("=" * 60)
