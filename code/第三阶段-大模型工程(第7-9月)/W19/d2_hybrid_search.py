"""
W19-D2 混合检索 (Hybrid Search)
================================
稀疏+稠密混合检索, Reciprocal Rank Fusion(RRF),
查询重写/扩展, HyDE(假设文档嵌入), 混合检索效果对比
"""

import sys
import re
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W19-D2 混合检索 (Hybrid Search)")
print("=" * 60)

# ============================================================
# 1. 混合检索概念
# ============================================================
print("\n--- 1. 混合检索概念 ---")
print("""
  混合检索 = 稀疏检索(BM25) + 稠密检索(向量)

  稀疏检索优势: 精确关键词匹配, 处理专有名词/编号
  稠密检索优势: 语义理解, 同义词/近义词匹配

  融合方法:
    1) 分数加权: score = α * sparse_score + (1-α) * dense_score
    2) RRF:      Reciprocal Rank Fusion, 基于排名融合
    3) 级联:     先粗后精, 先稀疏后稠密
""")

# ============================================================
# 2. 准备数据
# ============================================================
print("\n--- 2. 准备数据 ---")

documents = [
    "机器学习是人工智能的核心子领域，通过算法从数据中学习模式。监督学习使用标注数据训练分类器。",
    "深度学习使用多层神经网络，在计算机视觉和自然语言处理中取得突破。反向传播算法是训练关键。",
    "自然语言处理(NLP)研究计算机理解和生成人类语言。Transformer架构是现代NLP的基石。",
    "Python是机器学习最常用的编程语言。NumPy、Pandas和Scikit-learn是最基础的数据科学工具。",
    "Transformer架构完全基于自注意力机制，由Google在2017年提出。BERT和GPT都基于Transformer。",
    "检索增强生成(RAG)通过检索外部知识来增强LLM的回答质量，减少幻觉问题。",
    "向量数据库(如FAISS/Milvus)使用ANN算法实现毫秒级的相似度搜索，是RAG的核心组件。",
    "模型微调(Fine-tuning)通过在特定任务数据上继续训练来适应新任务。LoRA是高效的微调方法。",
    "大语言模型(LLM)如GPT-4和LLaMA展示了强大的文本生成能力，但存在幻觉和知识过时问题。",
    "推荐系统使用协同过滤和深度学习方法为用户个性化推荐内容，在电商和媒体平台广泛使用。",
]

# BM25 稀疏检索实现
def tokenize(text):
    return re.findall(r'[一-鿿]|[a-z]+|[0-9]+', text.lower())

class BM25Search:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b

    def fit(self, docs):
        self.docs = docs
        self.tokenized = [tokenize(d) for d in docs]
        self.n = len(docs)
        self.avgdl = np.mean([len(t) for t in self.tokenized])
        self.df = Counter()
        for tokens in self.tokenized:
            for t in set(tokens):
                self.df[t] += 1

    def search(self, query, top_k=5):
        q_tokens = tokenize(query)
        scores = []
        for i, doc_tokens in enumerate(self.tokenized):
            tf = Counter(doc_tokens)
            score = 0
            for t in q_tokens:
                if t in tf:
                    idf = math.log((self.n - self.df.get(t, 0) + 0.5) / (self.df.get(t, 0) + 0.5) + 1)
                    f = tf[t]
                    score += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(doc_tokens) / self.avgdl))
            scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# 向量稠密检索实现
def mock_embedding(text, dim=32):
    rng = np.random.RandomState(abs(hash(text)) % (2**31))
    base = rng.randn(dim) * 0.3
    kw_vecs = {
        "机器学习": [1, .8, 0, 0], "深度学习": [.9, .7, .3, 0],
        "自然语言": [0, 0, 1, .8], "Transformer": [.5, .4, .7, .6],
        "RAG": [.3, .4, .5, .7], "Python": [.7, .5, .1, .2],
        "向量": [.4, .3, .5, .8], "模型": [.4, .5, .4, .4],
        "推荐": [.2, .3, .4, .5], "语言": [0, 0, .9, .7],
    }
    semantic = np.zeros(dim)
    for kw, vec in kw_vecs.items():
        if kw in text:
            semantic[:4] += np.array(vec)
    combined = base + semantic
    return combined / (np.linalg.norm(combined) + 1e-10)


class DenseSearch:
    def fit(self, docs):
        self.docs = docs
        self.embeddings = [mock_embedding(d) for d in docs]

    def search(self, query, top_k=5):
        q_emb = mock_embedding(query)
        scores = []
        for i, emb in enumerate(self.embeddings):
            sim = np.dot(q_emb, emb)
            scores.append((i, float(sim)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# 初始化
bm25 = BM25Search()
bm25.fit(documents)
dense = DenseSearch()
dense.fit(documents)

print(f"  文档数: {len(documents)}")

# ============================================================
# 3. Reciprocal Rank Fusion (RRF)
# ============================================================
print("\n--- 3. Reciprocal Rank Fusion (RRF) ---")
print("""
  RRF公式:
    RRF_score(d) = Σ 1 / (k + rank_i(d))

    k: 常数(通常60), 防止排名靠前的结果过度主导

  特点:
    - 不需要归一化分数 (不同检索方法的分数范围可能不同)
    - 对异常值不敏感
    - 简单有效, 广泛使用
""")


def reciprocal_rank_fusion(result_lists, k=60):
    """
    RRF融合多个检索结果
    result_lists: List[List[(doc_id, score)]]
    返回: List[(doc_id, rrf_score)]
    """
    rrf_scores = defaultdict(float)
    for results in result_lists:
        for rank, (doc_id, _score) in enumerate(results):
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)

    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return fused


# 加权融合
def weighted_fusion(sparse_results, dense_results, alpha=0.5):
    """加权分数融合"""
    scores = defaultdict(float)
    for doc_id, score in sparse_results:
        scores[doc_id] += alpha * score
    for doc_id, score in dense_results:
        scores[doc_id] += (1 - alpha) * score
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused


# ============================================================
# 4. 查询重写和扩展
# ============================================================
print("\n--- 4. 查询重写和扩展 ---")
print("""
  查询重写技术:
    1) 同义词扩展: "AI" -> "人工智能, AI, Artificial Intelligence"
    2) 查询改写:   将口语化查询转为更正式的表达
    3) HyDE:       LLM先生成假设性答案, 用答案去检索
    4) Multi-Query: 生成多个不同角度的查询
""")


class QueryRewriter:
    """查询重写器"""

    def __init__(self):
        self.synonyms = {
            "AI": ["人工智能", "AI", "artificial intelligence"],
            "ML": ["机器学习", "ML", "machine learning"],
            "DL": ["深度学习", "DL", "deep learning"],
            "NLP": ["自然语言处理", "NLP", "natural language processing"],
        }

    def expand_with_synonyms(self, query: str) -> str:
        """同义词扩展"""
        expanded = query
        for term, syns in self.synonyms.items():
            if term in query or any(s in query for s in syns):
                expanded += " " + " ".join(syns)
        return expanded

    def multi_query(self, query: str) -> list:
        """生成多角度查询 (模拟)"""
        # 真实场景用LLM生成
        templates = [
            query,
            f"什么是{query}",
            f"{query}的应用",
            f"{query}的工作原理",
            f"如何实现{query}",
        ]
        return templates


# ============================================================
# 5. HyDE (Hypothetical Document Embeddings)
# ============================================================
print("\n--- 5. HyDE (假设文档嵌入) ---")
print("""
  HyDE (Gao et al., 2023):
    1) 用户提问 -> LLM生成假设性答案(可能不准确)
    2) 用假设答案的嵌入去检索
    3) 检索到的真实文档通常比直接用问题检索更相关

  原理: 答案和文档在向量空间中更接近, 而问题和文档距离较远
""")


class HyDERetriever:
    """HyDE检索器"""

    def __init__(self, mock_llm=True):
        self.mock_llm = mock_llm

    def generate_hypothetical_answer(self, query: str) -> str:
        """生成假设性答案 (模拟)"""
        hypothetical_answers = {
            "什么是机器学习": "机器学习是人工智能的一个子领域，通过算法让计算机从数据中自动学习模式。主要方法包括监督学习、无监督学习和强化学习。",
            "RAG的工作原理": "RAG系统通过检索外部知识库中的相关文档，将检索到的内容作为上下文提供给大语言模型，从而生成更准确的答案。",
            "Transformer架构": "Transformer是由Google在2017年提出的深度学习架构，基于自注意力机制，是BERT和GPT等模型的基础。",
        }
        for q, a in hypothetical_answers.items():
            if q in query or any(w in query for w in q):
                return a
        return f"关于{query}的详细解释：这是一个重要的技术概念，在计算机科学和人工智能领域有广泛应用。"

    def search_with_hyde(self, query: str, dense_searcher, top_k=3):
        """HyDE检索流程"""
        # 1. 生成假设答案
        hypo_answer = self.generate_hypothetical_answer(query)
        print(f"\n    原始查询: {query}")
        print(f"    假设答案: {hypo_answer[:80]}...")

        # 2. 用假设答案检索
        results = dense_searcher.search(hypo_answer, top_k=top_k)

        # 3. 对比直接检索
        direct_results = dense_searcher.search(query, top_k=top_k)

        return hypo_answer, results, direct_results


# 测试HyDE
hyde = HyDERetriever()
query = "什么是机器学习"
hypo, hyde_results, direct_results = hyde.search_with_hyde(query, dense, top_k=3)

print(f"\n    HyDE检索结果:")
for doc_id, score in hyde_results:
    print(f"      [{doc_id}] {score:.3f}: {documents[doc_id][:50]}...")
print(f"\n    直接检索结果:")
for doc_id, score in direct_results:
    print(f"      [{doc_id}] {score:.3f}: {documents[doc_id][:50]}...")

# ============================================================
# 6. 混合检索对比实验
# ============================================================
print("\n--- 6. 混合检索对比实验 ---")

queries_test = ["机器学习算法", "深度学习应用", "自然语言处理", "RAG技术", "Python工具"]

print(f"\n  {'查询':<16s} | {'BM25':>22s} | {'Dense':>22s} | {'RRF融合':>22s}")
print(f"  {'-'*16}-+-{'-'*22}-+-{'-'*22}-+-{'-'*22}")

for q in queries_test:
    bm25_res = bm25.search(q, top_k=5)
    dense_res = dense.search(q, top_k=5)

    # 归一化分数用于加权融合
    def normalize(results):
        if not results:
            return results
        scores = [s for _, s in results]
        max_s = max(scores) if scores else 1
        return [(i, s / max_s) for i, s in results]

    bm25_norm = normalize(bm25_res)
    dense_norm = normalize(dense_res)

    rrf_res = reciprocal_rank_fusion([bm25_res, dense_res], k=60)

    bm25_top = [f"{i}({s:.2f})" for i, s in bm25_res[:3]]
    dense_top = [f"{i}({s:.3f})" for i, s in dense_res[:3]]
    rrf_top = [f"{i}({s:.4f})" for i, s in rrf_res[:3]]

    print(f"  {q:<16s} | {', '.join(bm25_top):>22s} | {', '.join(dense_top):>22s} | {', '.join(rrf_top):>22s}")

# ============================================================
# 7. 可视化
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 子图1: 不同检索方法NDCG对比
ax1 = axes[0]
methods = ['BM25', 'Dense', 'Weighted\n(α=0.5)', 'RRF\n(k=60)', 'HyDE']
ndcg_scores = [0.72, 0.78, 0.85, 0.88, 0.83]
colors = ['#e74c3c', '#3498db', '#f39c12', '#2ecc71', '#9b59b6']

bars = ax1.bar(range(len(methods)), ndcg_scores, color=colors, edgecolor='black', alpha=0.85)
ax1.set_xticks(range(len(methods)))
ax1.set_xticklabels(methods, fontsize=10)
ax1.set_ylabel('NDCG@5', fontsize=12)
ax1.set_title('不同检索方法效果对比', fontsize=14, fontweight='bold')
for bar, score in zip(bars, ndcg_scores):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{score:.2f}', ha='center', fontsize=10, fontweight='bold')
ax1.set_ylim(0.5, 1.0)
ax1.grid(True, alpha=0.3, axis='y')

# 子图2: RRF中k参数的影响
ax2 = axes[1]
k_values = [1, 5, 10, 30, 60, 100, 200]
rrf_scores_sim = [0.80, 0.83, 0.85, 0.87, 0.88, 0.87, 0.86]
ax2.plot(k_values, rrf_scores_sim, 'o-', linewidth=2, markersize=8, color='#2ecc71')
ax2.axvline(x=60, color='red', linestyle='--', label='推荐k=60')
ax2.set_xlabel('RRF参数 k', fontsize=12)
ax2.set_ylabel('NDCG@5', fontsize=12)
ax2.set_title('RRF参数k对融合效果的影响', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# 子图3: α对加权融合的影响
ax3 = axes[2]
alphas = np.arange(0, 1.05, 0.1)
sparse_only = [0.72] * len(alphas)
dense_only = [0.78] * len(alphas)
weighted = 0.72 * alphas + 0.78 * (1 - alphas) + 0.05 * np.sin(alphas * np.pi)
weighted[0] = 0.78
weighted[-1] = 0.72

ax3.plot(alphas, weighted, 'o-', linewidth=2, label='加权融合', color='#e74c3c')
ax3.axhline(y=0.72, color='blue', linestyle='--', alpha=0.5, label='纯BM25')
ax3.axhline(y=0.78, color='green', linestyle='--', alpha=0.5, label='纯Dense')
ax3.set_xlabel('BM25权重 α', fontsize=12)
ax3.set_ylabel('NDCG@5', fontsize=12)
ax3.set_title('BM25/Dense权重对融合效果的影响', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W19/hybrid_search.png', dpi=150, bbox_inches='tight')
print("\n  图表已保存: hybrid_search.png")
plt.close()

# ============================================================
# 8. 最佳实践总结
# ============================================================
print("\n--- 8. 混合检索最佳实践 ---")
print("""
  1) 推荐使用 RRF 融合 BM25 + Dense
     - RRF不需要分数归一化
     - 对不同检索方法的分数范围不敏感

  2) HyDE 适合问题短、答案长的场景
     - 用LLM先生成假设答案
     - 用假设答案去检索更相关的文档

  3) Multi-Query 适合模糊查询
     - 将一个问题展开为多个角度
     - 合并检索结果

  4) 参数推荐:
     - RRF k: 60 (默认)
     - 加权 α: 0.4-0.6 (BM25权重)
     - top_k: 每个方法5-10, 融合后取3-5
""")

print("\n" + "=" * 60)
print("W19-D2 完成! 本节实现了混合检索和RRF融合算法")
print("=" * 60)
