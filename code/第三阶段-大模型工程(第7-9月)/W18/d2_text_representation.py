"""
W18-D2 文本表示与检索 (Text Representation & Retrieval)
=======================================================
TF-IDF实现, BM25算法实现, 文本相似度计算, 稀疏检索 vs 稠密检索对比
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
print("W18-D2 文本表示与检索 (Text Representation & Retrieval)")
print("=" * 60)

# ============================================================
# 1. 文本预处理工具
# ============================================================
print("\n--- 1. 文本预处理工具 ---")


def tokenize(text):
    """简单的中英文混合分词"""
    text = text.lower()
    # 提取中文单字和英文单词
    tokens = re.findall(r'[一-鿿]|[a-z]+|[0-9]+', text)
    return tokens


# 示例文档
documents = [
    "机器学习是人工智能的一个子领域，通过数据训练模型",
    "深度学习使用多层神经网络进行特征学习",
    "自然语言处理是人工智能的重要方向",
    "机器学习包括监督学习、无监督学习和强化学习",
    "深度学习在计算机视觉和自然语言处理中取得突破",
    "Python是机器学习和深度学习最常用的编程语言",
    "Transformer架构是现代自然语言处理的基础",
    "检索增强生成RAG结合了检索和生成的优势",
]

queries = [
    "什么是机器学习",
    "深度学习应用",
    "自然语言处理",
]

print(f"  文档数量: {len(documents)}")
print(f"  查询数量: {len(queries)}")

# 预处理
tokenized_docs = [tokenize(doc) for doc in documents]
tokenized_queries = [tokenize(q) for q in queries]

for i, (doc, tokens) in enumerate(zip(documents, tokenized_docs)):
    print(f"  Doc{i}: {doc[:40]}...")
    print(f"        Tokens: {tokens[:10]}")

# ============================================================
# 2. TF-IDF 实现
# ============================================================
print("\n--- 2. TF-IDF 实现 ---")
print("""
  TF-IDF (Term Frequency - Inverse Document Frequency)

  TF(t, d) = 词t在文档d中出现的次数 / 文档d的总词数
  IDF(t) = log(总文档数 / 包含词t的文档数)
  TF-IDF(t, d) = TF(t, d) * IDF(t)

  思想: 在特定文档中频繁出现, 但在所有文档中稀少的词更重要
""")


class TFIDF:
    """TF-IDF 向量化和检索"""

    def __init__(self):
        self.vocab = {}
        self.idf = {}
        self.doc_vectors = []

    def fit(self, tokenized_docs):
        """构建词表和计算IDF"""
        # 构建词表
        vocab = set()
        for tokens in tokenized_docs:
            vocab.update(tokens)
        self.vocab = {word: idx for idx, word in enumerate(sorted(vocab))}
        vocab_size = len(self.vocab)

        # 计算IDF
        doc_freq = Counter()
        for tokens in tokenized_docs:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_freq[token] += 1

        n_docs = len(tokenized_docs)
        self.idf = {}
        for word in self.vocab:
            df = doc_freq.get(word, 0)
            self.idf[word] = math.log((n_docs + 1) / (df + 1)) + 1  # 平滑

        # 计算文档TF-IDF向量
        self.doc_vectors = []
        for tokens in tokenized_docs:
            vec = np.zeros(vocab_size)
            tf = Counter(tokens)
            total = len(tokens)
            for word, count in tf.items():
                if word in self.vocab:
                    idx = self.vocab[word]
                    vec[idx] = (count / total) * self.idf.get(word, 0)
            # L2归一化
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            self.doc_vectors.append(vec)

        print(f"  词表大小: {vocab_size}")
        print(f"  文档向量维度: {vocab_size}")
        return self

    def transform(self, tokenized_query):
        """将查询转换为TF-IDF向量"""
        vec = np.zeros(len(self.vocab))
        tf = Counter(tokenized_query)
        total = len(tokenized_query)
        for word, count in tf.items():
            if word in self.vocab:
                idx = self.vocab[word]
                vec[idx] = (count / total) * self.idf.get(word, 0)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def search(self, tokenized_query, top_k=3):
        """检索最相关的文档"""
        query_vec = self.transform(tokenized_query)
        scores = []
        for i, doc_vec in enumerate(self.doc_vectors):
            score = np.dot(query_vec, doc_vec)  # 余弦相似度(已归一化)
            scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# 训练TF-IDF
tfidf = TFIDF()
tfidf.fit(tokenized_docs)

# 检索
print("\n  --- TF-IDF 检索结果 ---")
for query, tok_query in zip(queries, tokenized_queries):
    print(f"\n  查询: '{query}'")
    results = tfidf.search(tok_query, top_k=3)
    for doc_id, score in results:
        print(f"    [Doc{doc_id}] 相似度={score:.4f}: {documents[doc_id][:50]}...")

# ============================================================
# 3. BM25 算法实现
# ============================================================
print("\n--- 3. BM25 算法实现 ---")
print("""
  BM25 = Best Match 25 (Okapi BM25)

  BM25(q, d) = Σ IDF(qi) * (f(qi,d) * (k1+1)) / (f(qi,d) + k1 * (1 - b + b * |d|/avgdl))

  参数:
    k1 = 1.5    (词频饱和参数, 控制TF的增益递减)
    b  = 0.75   (长度归一化参数, 0=不考虑长度, 1=完全归一化)
    f(qi, d)    (词qi在文档d中的频率)
    |d|         (文档d的长度)
    avgdl       (所有文档的平均长度)
""")


class BM25:
    """BM25 检索算法"""

    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.doc_freq = {}
        self.doc_lens = []
        self.avgdl = 0
        self.n_docs = 0
        self.tokenized_docs = []

    def fit(self, tokenized_docs):
        """构建BM25索引"""
        self.tokenized_docs = tokenized_docs
        self.n_docs = len(tokenized_docs)

        # 文档频率
        self.doc_freq = Counter()
        for tokens in tokenized_docs:
            for token in set(tokens):
                self.doc_freq[token] += 1

        # 文档长度
        self.doc_lens = [len(tokens) for tokens in tokenized_docs]
        self.avgdl = np.mean(self.doc_lens) if self.doc_lens else 1

        print(f"  BM25参数: k1={self.k1}, b={self.b}")
        print(f"  平均文档长度: {self.avgdl:.1f}")
        return self

    def _idf(self, term):
        """计算IDF (BM25变体)"""
        df = self.doc_freq.get(term, 0)
        return math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)

    def score(self, query_tokens, doc_idx):
        """计算BM25分数"""
        doc_tokens = self.tokenized_docs[doc_idx]
        doc_len = self.doc_lens[doc_idx]
        tf = Counter(doc_tokens)

        score = 0.0
        for term in query_tokens:
            if term in tf:
                f = tf[term]
                idf = self._idf(term)
                numerator = f * (self.k1 + 1)
                denominator = f + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += idf * numerator / denominator
        return score

    def search(self, query_tokens, top_k=3):
        """BM25检索"""
        scores = []
        for i in range(self.n_docs):
            s = self.score(query_tokens, i)
            scores.append((i, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# 训练BM25
bm25 = BM25(k1=1.5, b=0.75)
bm25.fit(tokenized_docs)

# 检索
print("\n  --- BM25 检索结果 ---")
for query, tok_query in zip(queries, tokenized_queries):
    print(f"\n  查询: '{query}'")
    results = bm25.search(tok_query, top_k=3)
    for doc_id, score in results:
        print(f"    [Doc{doc_id}] BM25={score:.4f}: {documents[doc_id][:50]}...")

# ============================================================
# 4. 文本相似度计算
# ============================================================
print("\n--- 4. 文本相似度计算 ---")
print("""
  常见相似度计算方法:

  1) 余弦相似度 (Cosine Similarity)
     cos(A, B) = A·B / (|A| * |B|)
     范围: [-1, 1], 越大越相似

  2) Jaccard相似度
     J(A, B) = |A ∩ B| / |A ∪ B|
     范围: [0, 1]

  3) 欧氏距离 (Euclidean Distance)
     d(A, B) = sqrt(Σ(a_i - b_i)^2)
     越小越相似

  4) 点积 (Dot Product)
     A·B = Σ a_i * b_i
     未归一化时范围不固定
""")


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


def jaccard_similarity(set_a, set_b):
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / (union + 1e-10)


# 计算文档间相似度矩阵
print("\n  文档间余弦相似度 (TF-IDF):")
n = len(documents)
sim_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        sim_matrix[i][j] = cosine_similarity(tfidf.doc_vectors[i], tfidf.doc_vectors[j])

# 打印部分
for i in range(min(5, n)):
    sims = [f"{sim_matrix[i][j]:.2f}" for j in range(min(5, n))]
    print(f"  Doc{i}: {sims}")

# ============================================================
# 5. 稀疏检索 vs 稠密检索对比
# ============================================================
print("\n--- 5. 稀疏检索 vs 稠密检索对比 ---")
print("""
  +------------------+------------------------+------------------------+
  |                  |    稀疏检索(Sparse)     |    稠密检索(Dense)      |
  +------------------+------------------------+------------------------+
  | 代表算法         | TF-IDF, BM25           | Embedding + ANN        |
  | 表示方式         | 高维稀疏向量            | 低维稠密向量            |
  | 词表依赖         | 依赖                    | 不依赖                  |
  | 语义理解         | 弱 (关键词匹配)         | 强 (语义相似)           |
  | 精确匹配         | 强                      | 弱                      |
  | 存储开销         | 较大 (稀疏矩阵)         | 较小 (固定维度向量)      |
  | 查询速度         | 快 (倒排索引)           | 快 (ANN索引)            |
  | 同义词处理       | 差                      | 好                      |
  | OOV处理          | 差                      | 好                      |
  | 典型维度         | 词表大小 (30K-100K)    | 128-1536               |
  +------------------+------------------------+------------------------+

  最佳实践: 混合检索 (Sparse + Dense) 结合两者优势
""")

# ============================================================
# 6. 可视化
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 子图1: TF-IDF检索热力图
ax1 = axes[0]
im1 = ax1.imshow(sim_matrix[:6, :6], cmap='YlOrRd', aspect='auto')
ax1.set_xticks(range(6))
ax1.set_yticks(range(6))
ax1.set_xticklabels([f'D{i}' for i in range(6)], fontsize=9)
ax1.set_yticklabels([f'D{i}' for i in range(6)], fontsize=9)
ax1.set_title('TF-IDF 文档相似度矩阵', fontsize=14, fontweight='bold')
for i in range(6):
    for j in range(6):
        ax1.text(j, i, f'{sim_matrix[i,j]:.2f}', ha='center', va='center', fontsize=8)
plt.colorbar(im1, ax=ax1, shrink=0.8)

# 子图2: BM25 vs TF-IDF 检索分数对比
ax2 = axes[1]
query_idx = 0
bm25_scores = [bm25.score(tokenized_queries[query_idx], i) for i in range(len(documents))]
tfidf_scores = [np.dot(tfidf.transform(tokenized_queries[query_idx]), dv)
                for dv in tfidf.doc_vectors]

x = np.arange(len(documents))
width = 0.35
ax2.bar(x - width/2, [max(0, s) for s in bm25_scores], width, label='BM25', color='#3498db', alpha=0.85)
ax2.bar(x + width/2, [max(0, s) for s in tfidf_scores], width, label='TF-IDF', color='#e74c3c', alpha=0.85)
ax2.set_xticks(x)
ax2.set_xticklabels([f'D{i}' for i in range(len(documents))], fontsize=9)
ax2.set_ylabel('检索分数', fontsize=12)
ax2.set_title(f'BM25 vs TF-IDF (查询: "{queries[query_idx]}")', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# 子图3: 稀疏 vs 稠密 向量可视化
ax3 = axes[2]
np.random.seed(42)
# 稀疏向量: 大部分为0
sparse_vec = np.zeros(100)
sparse_vec[np.random.choice(100, 8)] = np.random.uniform(0.5, 2.0, 8)
# 稠密向量: 每个维度都有值
dense_vec = np.random.randn(100) * 0.5

ax3.plot(range(100), sparse_vec, 'o-', label='稀疏向量 (TF-IDF)', markersize=3, alpha=0.7)
ax3.plot(range(100), dense_vec, '-', label='稠密向量 (Embedding)', alpha=0.5)
ax3.set_xlabel('维度', fontsize=12)
ax3.set_ylabel('值', fontsize=12)
ax3.set_title('稀疏向量 vs 稠密向量', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.axhline(y=0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W18/text_representation.png', dpi=150, bbox_inches='tight')
print("\n  图表已保存: text_representation.png")
plt.close()

# ============================================================
# 7. 检索效果对比总结
# ============================================================
print("\n--- 7. 检索效果对比总结 ---")
print("\n  所有查询结果对比:")
print(f"  {'查询':<20s} {'BM25 Top1':>10s} {'BM25 Top2':>10s} {'TF-IDF Top1':>12s} {'TF-IDF Top2':>12s}")
print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*12} {'-'*12}")
for query, tok_query in zip(queries, tokenized_queries):
    bm25_res = bm25.search(tok_query, top_k=2)
    tfidf_res = tfidf.search(tok_query, top_k=2)
    b1 = f"D{bm25_res[0][0]}({bm25_res[0][1]:.2f})"
    b2 = f"D{bm25_res[1][0]}({bm25_res[1][1]:.2f})"
    t1 = f"D{tfidf_res[0][0]}({tfidf_res[0][1]:.3f})"
    t2 = f"D{tfidf_res[1][0]}({tfidf_res[1][1]:.3f})"
    print(f"  {query:<20s} {b1:>10s} {b2:>10s} {t1:>12s} {t2:>12s}")

print("\n" + "=" * 60)
print("W18-D2 完成! 本节实现了TF-IDF和BM25检索算法")
print("=" * 60)
