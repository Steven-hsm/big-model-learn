"""
W18-D3 嵌入模型 (Embedding Models)
===================================
Word2Vec/GloVe概念, 句子嵌入(Sentence-BERT),
使用sentence-transformers, 嵌入可视化(t-SNE), 余弦相似度检索
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W18-D3 嵌入模型 (Embedding Models)")
print("=" * 60)

# ============================================================
# 1. Word2Vec 概念
# ============================================================
print("\n--- 1. Word2Vec 概念 ---")
print("""
  Word2Vec (Mikolov et al., 2013)
  将词语映射到稠密向量空间, 语义相近的词向量相近

  两种架构:
    1) CBOW (Continuous Bag of Words)
       上下文词 -> 预测中心词
       输入: "the cat sits on the"  -> 预测: "mat"

    2) Skip-gram
       中心词 -> 预测上下文词
       输入: "mat"  -> 预测: "the cat sits on the"

  经典示例:
    vec("king") - vec("man") + vec("woman") ≈ vec("queen")
    vec("Paris") - vec("France") + vec("Germany") ≈ vec("Berlin")

  局限性:
    - 一词一义: 无法处理多义词 (如"苹果"公司/水果)
    - 静态嵌入: 不考虑上下文
    - 需要大量训练数据
""")


# 简化Word2Vec演示 (Skip-gram with Negative Sampling)
class SimpleWord2Vec:
    """极简Word2Vec实现 (仅演示概念)"""

    def __init__(self, vocab_size=100, embedding_dim=16):
        np.random.seed(42)
        self.W_in = np.random.randn(vocab_size, embedding_dim) * 0.1   # 输入嵌入
        self.W_out = np.random.randn(vocab_size, embedding_dim) * 0.1  # 输出嵌入
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

    def get_embedding(self, word_idx):
        """获取词向量"""
        return self.W_in[word_idx]

    def most_similar(self, word_idx, top_k=5):
        """找最相似的词"""
        target = self.get_embedding(word_idx)
        # 余弦相似度
        norms = np.linalg.norm(self.W_in, axis=1)
        target_norm = np.linalg.norm(target)
        similarities = np.dot(self.W_in, target) / (norms * target_norm + 1e-10)
        similarities[word_idx] = -1  # 排除自己
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(idx, similarities[idx]) for idx in top_indices]


# 演示
w2v = SimpleWord2Vec(vocab_size=50, embedding_dim=16)
print("\n  Word2Vec 伪演示 (随机初始化, 展示接口):")
similar = w2v.most_similar(10, top_k=5)
print(f"  与词#10最相似的词: {[(idx, f'{sim:.3f}') for idx, sim in similar]}")

# ============================================================
# 2. GloVe 概念
# ============================================================
print("\n--- 2. GloVe 概念 ---")
print("""
  GloVe = Global Vectors for Word Representation (Pennington et al., 2014)

  核心思想:
    利用全局的词-词共现矩阵来学习词向量

  目标函数:
    J = Σ f(X_ij) * (w_i · w_j + b_i + b_j - log X_ij)^2

    X_ij: 词i和词j的共现次数
    f(X_ij): 权重函数, 过滤极端共现次数
    w_i, w_j: 词向量
    b_i, b_j: 偏置

  与Word2Vec的区别:
    Word2Vec: 基于局部上下文窗口 (局部统计)
    GloVe:    基于全局共现矩阵 (全局统计)
    效果:     两者性能相当, GloVe训练更快
""")

# ============================================================
# 3. 句子嵌入 (Sentence Embeddings)
# ============================================================
print("\n--- 3. 句子嵌入 (Sentence Embeddings) ---")
print("""
  Word2Vec/GloVe 是词级嵌入, 但我们需要句子/段落级嵌入

  方法演进:
    1) 平均词向量:  avg(word_vectors) -- 简单但效果一般
    2) SBERT:       Sentence-BERT (2019) -- 对BERT做句子级微调
    3) SimCSE:      对比学习, 无监督/有监督句子嵌入
    4) Instructor:  任务感知的通用嵌入模型
    5) BGE:         北京智源, 中文优秀嵌入模型
    6) E5:          微软, 多语言嵌入模型

  常用嵌入模型:
    - all-MiniLM-L6-v2:    384维, 速度快, 英文
    - all-mpnet-base-v2:   768维, 质量高, 英文
    - paraphrase-multilingual: 768维, 多语言
    - bge-small-zh-v1.5:   512维, 中文优秀
    - bge-large-zh-v1.5:   1024维, 中文最佳
""")

# ============================================================
# 4. 模拟句子嵌入 + 余弦相似度检索
# ============================================================
print("\n--- 4. 模拟句子嵌入 + 余弦相似度检索 ---")


def mock_sentence_embedding(text, dim=64, seed_offset=0):
    """模拟句子嵌入 (基于文本哈希, 语义相似的文本生成相近的向量)"""
    # 简单哈希
    hash_val = hash(text) + seed_offset
    np.random.seed(abs(hash_val) % (2**31))

    # 基础随机向量
    base_vec = np.random.randn(dim)

    # 语义增强: 相同关键词的文本向量方向相近
    keywords = {
        "机器": np.array([1, 0.8, 0.3, 0.1] + [0] * (dim - 4)),
        "学习": np.array([0.9, 0.7, 0.2, 0.1] + [0] * (dim - 4)),
        "深度": np.array([0.8, 0.6, 0.5, 0.2] + [0] * (dim - 4)),
        "自然": np.array([0.3, 0.2, 1, 0.8] + [0] * (dim - 4)),
        "语言": np.array([0.2, 0.1, 0.9, 0.7] + [0] * (dim - 4)),
        "处理": np.array([0.1, 0.2, 0.8, 0.9] + [0] * (dim - 4)),
        "检索": np.array([0.5, 0.3, 0.4, 0.6] + [0] * (dim - 4)),
        "python": np.array([0.7, 0.5, 0.1, 0.3] + [0] * (dim - 4)),
    }

    semantic_vec = np.zeros(dim)
    for kw, vec in keywords.items():
        if kw in text:
            semantic_vec += vec

    combined = base_vec * 0.3 + semantic_vec * 0.7
    combined = combined / (np.linalg.norm(combined) + 1e-10)
    return combined


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


# 构建文档嵌入库
sentences = [
    "机器学习是人工智能的重要分支",
    "深度学习使用多层神经网络",
    "自然语言处理研究计算机理解人类语言",
    "机器学习包括监督学习和无监督学习",
    "深度学习在计算机视觉中广泛应用",
    "Python是数据科学的首选语言",
    "检索增强生成结合了检索和生成",
    "Transformer架构改变了NLP领域",
    "机器学习算法需要大量训练数据",
    "自然语言处理在搜索引擎中发挥重要作用",
]

embeddings = np.array([mock_sentence_embedding(s) for s in sentences])

# 查询
query = "什么是机器学习"
query_emb = mock_sentence_embedding(query)

# 余弦相似度检索
scores = [cosine_similarity(query_emb, emb) for emb in embeddings]
ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

print(f"  查询: '{query}'")
print(f"  嵌入维度: {embeddings.shape[1]}")
print(f"\n  检索结果 (余弦相似度):")
for idx, score in ranked[:5]:
    print(f"    [{idx}] 相似度={score:.4f}: {sentences[idx]}")

# ============================================================
# 5. t-SNE 可视化
# ============================================================
print("\n--- 5. 嵌入可视化 ---")


def simple_tsne(X, n_components=2, perplexity=5, n_iter=300, lr=100):
    """简化的t-SNE实现 (用于演示)"""
    n = X.shape[0]
    # 初始化
    Y = np.random.randn(n, n_components) * 0.01

    # 计算高维距离
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = np.sum((X[i] - X[j]) ** 2)

    # 高维相似度 (高斯核)
    P = np.zeros((n, n))
    for i in range(n):
        sigma = 1.0
        P[i] = np.exp(-dist[i] / (2 * sigma ** 2))
        P[i, i] = 0
        P[i] = P[i] / (P[i].sum() + 1e-10)

    P = (P + P.T) / (2 * n)
    P = np.maximum(P, 1e-12)

    # 梯度下降
    for iteration in range(n_iter):
        # 低维相似度 (t分布核)
        dist_y = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist_y[i, j] = np.sum((Y[i] - Y[j]) ** 2)

        Q = 1 / (1 + dist_y)
        np.fill_diagonal(Q, 0)
        Q = Q / (Q.sum() + 1e-10)
        Q = np.maximum(Q, 1e-12)

        # 梯度
        PQ = P - Q
        gradients = np.zeros_like(Y)
        for i in range(n):
            for j in range(n):
                gradients[i] += PQ[i, j] * (Y[i] - Y[j]) * (1 / (1 + dist_y[i, j]))

        Y = Y - lr * gradients / n

        if iteration % 100 == 0:
            cost = np.sum(P * np.log(P / (Q + 1e-10)))
            print(f"    t-SNE 迭代 {iteration}/{n_iter}, KL散度: {cost:.4f}")

    return Y


# 可视化
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 子图1: 2D散点图 (使用PCA简化)
ax1 = axes[0]
# 简单PCA降维
from numpy.linalg import svd
X_centered = embeddings - embeddings.mean(axis=0)
# 使用简单的2D投影
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
coords_2d = X_centered @ Vt[:2].T

# 按主题着色
topic_colors = ['#e74c3c', '#e74c3c', '#3498db', '#e74c3c', '#e74c3c',
                '#2ecc71', '#9b59b6', '#3498db', '#e74c3c', '#3498db']
topic_labels = ['ML', 'DL', 'NLP', 'ML', 'DL', 'Python', 'RAG', 'NLP', 'ML', 'NLP']

for i, (x, y) in enumerate(coords_2d):
    ax1.scatter(x, y, c=topic_colors[i], s=100, edgecolors='black', zorder=5)
    ax1.annotate(f'{i}-{topic_labels[i]}', (x, y), fontsize=8,
                 xytext=(5, 5), textcoords='offset points')

# 标注查询
q_coord = (query_emb - embeddings.mean(axis=0)) @ Vt[:2].T
ax1.scatter(q_coord[0], q_coord[1], c='gold', s=200, marker='*',
            edgecolors='black', zorder=10, label='查询')

ax1.set_title('句子嵌入 2D 可视化 (PCA)', fontsize=14, fontweight='bold')
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 子图2: 相似度热力图
ax2 = axes[1]
sim_matrix = np.zeros((len(sentences), len(sentences)))
for i in range(len(sentences)):
    for j in range(len(sentences)):
        sim_matrix[i, j] = cosine_similarity(embeddings[i], embeddings[j])

im = ax2.imshow(sim_matrix, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
ax2.set_xticks(range(len(sentences)))
ax2.set_yticks(range(len(sentences)))
ax2.set_xticklabels([f'S{i}' for i in range(len(sentences))], fontsize=9)
ax2.set_yticklabels([f'S{i}' for i in range(len(sentences))], fontsize=9)
ax2.set_title('句子间余弦相似度矩阵', fontsize=14, fontweight='bold')
for i in range(len(sentences)):
    for j in range(len(sentences)):
        ax2.text(j, i, f'{sim_matrix[i,j]:.2f}', ha='center', va='center', fontsize=7)
plt.colorbar(im, ax=ax2, shrink=0.8)

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W18/embedding_models.png', dpi=150, bbox_inches='tight')
print("\n  图表已保存: embedding_models.png")
plt.close()

# ============================================================
# 6. 使用 sentence-transformers (可选)
# ============================================================
print("\n--- 6. 使用 sentence-transformers (可选) ---")

try:
    from sentence_transformers import SentenceTransformer

    print("  正在加载 sentence-transformers 模型...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 编码句子
    embeddings_real = model.encode(sentences)
    print(f"  嵌入形状: {embeddings_real.shape}")

    # 查询
    query_embedding = model.encode([query])
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
    similarities = sklearn_cosine(query_embedding, embeddings_real)[0]

    print(f"\n  查询: '{query}'")
    top_indices = np.argsort(similarities)[::-1][:5]
    for idx in top_indices:
        print(f"    [{idx}] 相似度={similarities[idx]:.4f}: {sentences[idx]}")

except ImportError:
    print("  [!] sentence-transformers 未安装, 跳过")
    print("  安装命令: pip install sentence-transformers")
except Exception as e:
    print(f"  [!] 加载模型出错: {e}")

# ============================================================
# 7. 嵌入模型选择指南
# ============================================================
print("\n--- 7. 嵌入模型选择指南 ---")
models_guide = [
    ("all-MiniLM-L6-v2",        384,  "英文, 快速",           "通用英文检索"),
    ("all-mpnet-base-v2",       768,  "英文, 高质量",          "高质量英文检索"),
    ("paraphrase-multilingual", 768,  "多语言",               "多语言场景"),
    ("bge-small-zh-v1.5",       512,  "中文, 快速",           "中文RAG"),
    ("bge-large-zh-v1.5",       1024, "中文, 最佳",           "高质量中文RAG"),
    ("text-embedding-3-small",  1536, "OpenAI API",           "通用API调用"),
    ("text-embedding-3-large",  3072, "OpenAI API, 最佳",     "高质量API调用"),
]

print(f"  {'模型':<28s} {'维度':>6s} {'特点':<20s} {'适用场景':<18s}")
print(f"  {'-'*28} {'-'*6} {'-'*20} {'-'*18}")
for name, dim, feat, use in models_guide:
    print(f"  {name:<28s} {dim:>6d} {feat:<20s} {use:<18s}")

print("\n" + "=" * 60)
print("W18-D3 完成! 本节学习了词嵌入、句子嵌入和向量检索")
print("=" * 60)
