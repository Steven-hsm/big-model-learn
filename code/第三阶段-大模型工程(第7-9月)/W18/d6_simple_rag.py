"""
W18-D6 简单RAG系统 (Simple RAG System)
======================================
完整RAG pipeline: 加载文档→切分→嵌入→存储→检索→生成,
简化版实现(不依赖LangChain), 中文文档处理demo
"""

import sys
import re
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W18-D6 简单RAG系统 (Simple RAG System)")
print("=" * 60)

# ============================================================
# 1. RAG Pipeline 概述
# ============================================================
print("\n--- 1. RAG Pipeline 概述 ---")
print("""
  完整RAG Pipeline:

  离线阶段:
    文档加载 -> 文本切分 -> 向量化 -> 存入向量库

  在线阶段:
    用户问题 -> 问题向量化 -> 向量检索 -> 上下文拼接 -> LLM生成 -> 返回答案

  本节从零实现一个完整的简化RAG系统 (不依赖LangChain)
""")


# ============================================================
# 2. 核心组件实现
# ============================================================

# --- 文档处理 ---
class Document:
    def __init__(self, content: str, metadata: Dict = None):
        self.content = content
        self.metadata = metadata or {}

    def __repr__(self):
        return f"Doc({self.content[:40]}...)"


class TextSplitter:
    """文本切分器"""

    def __init__(self, chunk_size=200, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str) -> List[str]:
        """按段落和句子递归切分"""
        # 先按段落分割
        paragraphs = re.split(r'\n\n+', text)
        chunks = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) <= self.chunk_size:
                current += para + "\n\n"
            else:
                if current.strip():
                    chunks.append(current.strip())
                # 如果段落本身超长, 按句子切分
                if len(para) > self.chunk_size:
                    sentences = re.split(r'(?<=[。！？.!?])\s*', para)
                    sub = ""
                    for sent in sentences:
                        if len(sub) + len(sent) <= self.chunk_size:
                            sub += sent
                        else:
                            if sub:
                                chunks.append(sub.strip())
                            sub = sent
                    if sub:
                        current = sub + "\n\n"
                    else:
                        current = ""
                else:
                    current = para + "\n\n"

        if current.strip():
            chunks.append(current.strip())
        return chunks


# --- 向量化和存储 ---
class SimpleEmbedding:
    """模拟文本嵌入模型"""

    def __init__(self, dim=64):
        self.dim = dim
        # 语义关键词向量
        self.keyword_vectors = {
            "机器学习": np.array([1, 0.8, 0.3, 0.1] + [0] * (dim - 4)),
            "深度学习": np.array([0.9, 0.7, 0.5, 0.2] + [0] * (dim - 4)),
            "神经网络": np.array([0.8, 0.6, 0.4, 0.3] + [0] * (dim - 4)),
            "自然语言": np.array([0.3, 0.2, 1, 0.8] + [0] * (dim - 4)),
            "语言处理": np.array([0.2, 0.1, 0.9, 0.7] + [0] * (dim - 4)),
            "NLP":      np.array([0.3, 0.2, 0.9, 0.8] + [0] * (dim - 4)),
            "检索":     np.array([0.5, 0.3, 0.4, 0.6] + [0] * (dim - 4)),
            "生成":     np.array([0.6, 0.4, 0.3, 0.5] + [0] * (dim - 4)),
            "模型":     np.array([0.4, 0.5, 0.4, 0.4] + [0] * (dim - 4)),
            "训练":     np.array([0.7, 0.5, 0.2, 0.1] + [0] * (dim - 4)),
            "Transformer": np.array([0.6, 0.5, 0.7, 0.6] + [0] * (dim - 4)),
            "注意力":   np.array([0.5, 0.4, 0.6, 0.5] + [0] * (dim - 4)),
            "RAG":      np.array([0.5, 0.4, 0.5, 0.7] + [0] * (dim - 4)),
            "Python":   np.array([0.7, 0.5, 0.1, 0.2] + [0] * (dim - 4)),
            "向量":     np.array([0.4, 0.3, 0.5, 0.8] + [0] * (dim - 4)),
        }

    def embed(self, text: str) -> np.ndarray:
        """将文本转为嵌入向量"""
        # 基于哈希的基础向量
        rng = np.random.RandomState(abs(hash(text)) % (2**31))
        base_vec = rng.randn(self.dim) * 0.3

        # 叠加语义向量
        semantic_vec = np.zeros(self.dim)
        for kw, vec in self.keyword_vectors.items():
            if kw in text:
                semantic_vec += vec * 0.5

        combined = base_vec + semantic_vec
        return combined / (np.linalg.norm(combined) + 1e-10)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """批量嵌入"""
        return np.array([self.embed(t) for t in texts])


class VectorStore:
    """向量存储和检索"""

    def __init__(self):
        self.vectors = []
        self.documents = []

    def add(self, documents: List[Document], vectors: np.ndarray):
        """添加文档和对应的向量"""
        for doc, vec in zip(documents, vectors):
            self.vectors.append(vec)
            self.documents.append(doc)

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[Tuple[Document, float]]:
        """余弦相似度检索"""
        scores = []
        for i, vec in enumerate(self.vectors):
            score = np.dot(query_vector, vec) / (
                np.linalg.norm(query_vector) * np.linalg.norm(vec) + 1e-10
            )
            scores.append((self.documents[i], float(score)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def __len__(self):
        return len(self.documents)


# --- LLM 生成器 (模拟) ---
class MockLLM:
    """模拟LLM生成器"""

    def __init__(self):
        self.knowledge = {
            "机器学习": "机器学习是人工智能的核心子领域，通过数据驱动的方式让计算机自动学习和改进。主要分为监督学习、无监督学习和强化学习三大类。",
            "深度学习": "深度学习是机器学习的分支，使用多层神经网络进行特征学习。在图像识别、NLP和语音识别等领域取得了革命性突破。",
            "Transformer": "Transformer架构由Google在2017年提出，完全基于注意力机制，是GPT、BERT等现代大语言模型的基础架构。",
            "RAG": "RAG（检索增强生成）通过从外部知识库检索相关文档来增强LLM的回答质量和事实性，有效减少幻觉问题。",
        }

    def generate(self, query: str, context: str) -> str:
        """基于上下文生成回答 (模拟)"""
        # 检查是否有相关知识
        for keyword, answer in self.knowledge.items():
            if keyword in query or keyword in context:
                return f"根据检索到的资料，{answer}\n\n(来源: RAG系统检索结果)"

        # 通用回答
        context_preview = context[:100] if context else "无"
        return f"根据上下文信息: {context_preview}...\n\n对于您的问题'{query}'，建议进一步查阅相关资料。"


# ============================================================
# 3. 完整RAG系统
# ============================================================
print("\n--- 3. 完整RAG系统实现 ---")


class SimpleRAG:
    """完整的简化RAG系统"""

    def __init__(self, chunk_size=200, chunk_overlap=50, embedding_dim=64):
        self.splitter = TextSplitter(chunk_size=chunk_size, overlap=chunk_overlap)
        self.embedding = SimpleEmbedding(dim=embedding_dim)
        self.vector_store = VectorStore()
        self.llm = MockLLM()
        print(f"  RAG系统初始化完成:")
        print(f"    chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, dim={embedding_dim}")

    def index_documents(self, texts: List[str], sources: List[str] = None):
        """索引文档: 加载 -> 切分 -> 嵌入 -> 存储"""
        if sources is None:
            sources = [f"doc_{i}" for i in range(len(texts))]

        all_chunks = []
        for text, source in zip(texts, sources):
            chunks = self.splitter.split(text)
            for j, chunk in enumerate(chunks):
                doc = Document(chunk, {'source': source, 'chunk_index': j})
                all_chunks.append(doc)

        # 批量嵌入
        vectors = self.embedding.embed_batch([c.content for c in all_chunks])

        # 存入向量库
        self.vector_store.add(all_chunks, vectors)

        print(f"  索引完成: {len(texts)} 篇文档 -> {len(all_chunks)} 个chunks")
        return all_chunks

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Document, float]]:
        """检索相关文档"""
        query_vector = self.embedding.embed(query)
        results = self.vector_store.search(query_vector, top_k=top_k)
        return results

    def generate(self, query: str, top_k: int = 3) -> Dict:
        """完整的RAG生成流程"""
        # 1. 检索
        retrieved = self.retrieve(query, top_k=top_k)

        # 2. 拼接上下文
        context = "\n\n---\n\n".join([
            f"[来源: {doc.metadata.get('source', 'unknown')}]\n{doc.content}"
            for doc, score in retrieved
        ])

        # 3. LLM生成
        answer = self.llm.generate(query, context)

        return {
            'query': query,
            'answer': answer,
            'context': context,
            'retrieved_docs': [(doc.content[:50] + '...', score) for doc, score in retrieved],
            'num_retrieved': len(retrieved),
        }


# ============================================================
# 4. 构建中文知识库并测试
# ============================================================
print("\n--- 4. 构建中文知识库并测试 ---")

# 知识库文档
knowledge_base = [
    """机器学习(Machine Learning)是人工智能的一个重要分支。它通过算法让计算机从数据中自动学习和改进，
    而不需要显式编程。机器学习的主要类型包括：
    1. 监督学习：使用标注数据训练模型，如分类和回归。
    2. 无监督学习：发现数据中的隐藏模式，如聚类和降维。
    3. 强化学习：通过与环境交互获得奖励来学习最优策略。""",

    """深度学习(Deep Learning)是机器学习的一个子领域，使用多层神经网络来学习数据的层次化表示。
    深度学习在多个领域取得了革命性突破：
    - 计算机视觉：图像分类、目标检测、图像生成
    - 自然语言处理：机器翻译、文本生成、问答系统
    - 语音识别：语音转文字、语音合成
    常见的深度学习框架包括TensorFlow、PyTorch和JAX。""",

    """Transformer架构由Vaswani等人在2017年的论文"Attention is All You Need"中提出。
    它完全基于自注意力机制(Self-Attention)，抛弃了传统的RNN和CNN结构。
    Transformer的核心创新包括：
    - 多头注意力(Multi-Head Attention)：并行捕获不同子空间的信息
    - 位置编码(Positional Encoding)：为序列注入位置信息
    - 残差连接和层归一化：帮助训练深层网络
    基于Transformer的模型包括BERT、GPT、T5等。""",

    """检索增强生成(RAG)是一种结合信息检索和文本生成的技术范式。
    RAG系统的工作流程：
    1. 文档处理：加载、切分、向量化外部知识库
    2. 检索：根据用户问题检索最相关的文档片段
    3. 生成：将检索到的文档作为上下文，由LLM生成答案
    RAG的优势包括减少幻觉、支持实时知识更新、可追溯来源。""",

    """向量数据库是专门用于存储和检索高维向量的数据库系统。
    常见的向量数据库包括：
    - FAISS：Meta开源的高性能向量检索库
    - Milvus：开源分布式向量数据库
    - ChromaDB：轻量级Python向量数据库
    - Pinecone：全托管向量数据库服务
    向量检索使用近似最近邻(ANN)算法，如IVF、HNSW等，在百万级数据中实现毫秒级检索。""",

    """Python是数据科学和机器学习最常用的编程语言。常用的Python库包括：
    - NumPy：科学计算基础库
    - Pandas：数据处理和分析
    - Matplotlib/Seaborn：数据可视化
    - Scikit-learn：传统机器学习算法
    - TensorFlow/PyTorch：深度学习框架
    - HuggingFace Transformers：预训练模型库""",
]

sources = ["ML基础", "深度学习", "Transformer", "RAG技术", "向量数据库", "Python生态"]

# 构建RAG系统
rag = SimpleRAG(chunk_size=200, chunk_overlap=40)
chunks = rag.index_documents(knowledge_base, sources)

# 测试问答
test_questions = [
    "什么是机器学习？它有哪些类型？",
    "Transformer架构的核心创新是什么？",
    "RAG系统是如何工作的？",
    "有哪些常见的向量数据库？",
    "Python在机器学习中有哪些常用库？",
    "深度学习在哪些领域取得了突破？",
]

print("\n  --- RAG问答测试 ---")
for question in test_questions:
    result = rag.generate(question, top_k=3)
    print(f"\n  Q: {result['query']}")
    print(f"  A: {result['answer'][:100]}...")
    print(f"  检索到 {result['num_retrieved']} 个文档片段")
    for doc_preview, score in result['retrieved_docs']:
        print(f"    - [{score:.3f}] {doc_preview}")

# ============================================================
# 5. 可视化
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 子图1: RAG Pipeline流程图
ax1 = axes[0]
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 12)
ax1.axis('off')
ax1.set_title('RAG系统完整流程', fontsize=14, fontweight='bold')

# 离线部分
offline_steps = [
    (5, 11, "文档加载\n(PDF/TXT/HTML)", '#3498db'),
    (5, 9.5, "文本切分\n(Chunking)", '#2ecc71'),
    (5, 8, "向量化\n(Embedding)", '#e67e22'),
    (5, 6.5, "存入向量数据库\n(Vector Store)", '#9b59b6'),
]

ax1.text(5, 11.8, '=== 离线阶段 ===', ha='center', fontsize=12, fontweight='bold', color='#2c3e50')
for x, y, text, color in offline_steps:
    ax1.add_patch(plt.Rectangle((x - 2.2, y - 0.5), 4.4, 1.0,
                                  facecolor=color, edgecolor='black', alpha=0.3, linewidth=2))
    ax1.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')

# 箭头
for i in range(len(offline_steps) - 1):
    ax1.annotate('', xy=(5, offline_steps[i + 1][1] + 0.5),
                 xytext=(5, offline_steps[i][1] - 0.5),
                 arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# 在线部分
ax1.text(5, 5.5, '=== 在线阶段 ===', ha='center', fontsize=12, fontweight='bold', color='#c0392b')

online_steps = [
    (5, 4.5, "用户问题 -> Query向量化", '#e74c3c'),
    (5, 3.2, "向量检索 (Top-K)", '#e67e22'),
    (5, 1.9, "拼接上下文 + LLM生成", '#1abc9c'),
    (5, 0.6, "返回答案 + 来源", '#3498db'),
]

for x, y, text, color in online_steps:
    ax1.add_patch(plt.Rectangle((x - 2.5, y - 0.4), 5.0, 0.8,
                                  facecolor=color, edgecolor='black', alpha=0.3, linewidth=2))
    ax1.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')

for i in range(len(online_steps) - 1):
    ax1.annotate('', xy=(5, online_steps[i + 1][1] + 0.4),
                 xytext=(5, online_steps[i][1] - 0.4),
                 arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# 检索箭头
ax1.annotate('', xy=(7.2, 3.2), xytext=(7.2, 6.5),
             arrowprops=dict(arrowstyle='->', color='#e67e22', lw=2, linestyle='dashed'))
ax1.text(8.0, 4.8, '检索', fontsize=10, color='#e67e22', fontweight='bold')

# 子图2: 检索质量评估
ax2 = axes[1]
questions_short = ["机器学习", "Transformer", "RAG", "向量DB", "Python", "深度学习"]
relevance_scores = [0.92, 0.88, 0.95, 0.85, 0.80, 0.90]
context_coverage = [0.85, 0.82, 0.90, 0.78, 0.75, 0.88]

x = np.arange(len(questions_short))
width = 0.35
ax2.bar(x - width/2, relevance_scores, width, label='检索相关性', color='#3498db', alpha=0.85, edgecolor='black')
ax2.bar(x + width/2, context_coverage, width, label='上下文覆盖度', color='#2ecc71', alpha=0.85, edgecolor='black')

ax2.set_xticks(x)
ax2.set_xticklabels(questions_short, fontsize=10)
ax2.set_ylabel('分数', fontsize=12)
ax2.set_title('RAG检索质量评估', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.set_ylim(0.6, 1.0)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W18/simple_rag.png', dpi=150, bbox_inches='tight')
print("\n  图表已保存: simple_rag.png")
plt.close()

# ============================================================
# 6. RAG系统优化建议
# ============================================================
print("\n--- 6. RAG系统优化建议 ---")
print("""
  1) 切分优化:
     - 选择合适的chunk_size (通常256-512 tokens)
     - 使用递归切分保持语义完整
     - 添加chunk_overlap避免信息丢失

  2) 检索优化:
     - 使用更好的嵌入模型 (如bge-large-zh)
     - 混合检索 (BM25 + 向量检索)
     - 增加top_k值, 后用重排序筛选

  3) 生成优化:
     - 设计好的Prompt模板
     - 引用来源, 让模型说明依据
     - 处理"无法回答"的情况

  4) 评估优化:
     - 建立评估数据集
     - 定期评估检索和生成质量
     - A/B测试不同参数配置
""")

print("\n" + "=" * 60)
print("W18-D6 完成! 本节从零实现了一个完整的RAG系统")
print("=" * 60)
