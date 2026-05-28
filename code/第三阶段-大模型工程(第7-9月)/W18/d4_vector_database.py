"""
W18-D4 向量数据库 (Vector Database)
====================================
向量数据库概念, FAISS索引构建(Flat/IVF/HNSW),
增删查改操作, ANN近似最近邻搜索
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import time

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W18-D4 向量数据库 (Vector Database)")
print("=" * 60)

# ============================================================
# 1. 向量数据库概念
# ============================================================
print("\n--- 1. 向量数据库概念 ---")
print("""
  向量数据库专门用于存储、索引和检索高维向量

  核心功能:
    1) 存储: 高效存储大量向量及其元数据
    2) 索引: 构建加速检索的数据结构
    3) 检索: 快速找到与查询向量最相似的K个向量 (ANN)
    4) CRUD: 支持增删改查操作

  常见向量数据库:
    +----------------+--------+----------+--------------------------+
    |     名称       | 开源   |  类型    |        特点              |
    +----------------+--------+----------+--------------------------+
    | FAISS          | 是     | 库       | Meta, 高性能, 灵活       |
    | ChromaDB       | 是     | 数据库   | 轻量, Python友好        |
    | Milvus         | 是     | 数据库   | 分布式, 生产级           |
    | Pinecone       | 否     | 云服务   | 全托管, 简单易用         |
    | Weaviate       | 是     | 数据库   | 支持混合检索             |
    | Qdrant         | 是     | 数据库   | Rust实现, 高性能         |
    | pgvector       | 是     | PG扩展   | PostgreSQL扩展           |
    +----------------+--------+----------+--------------------------+
""")

# ============================================================
# 2. 纯NumPy实现向量检索
# ============================================================
print("\n--- 2. 纯NumPy实现向量检索 ---")


class SimpleVectorDB:
    """纯NumPy实现的简单向量数据库"""

    def __init__(self, dimension=64):
        self.dimension = dimension
        self.vectors = []
        self.metadata = []

    def add(self, vector, metadata=None):
        """添加向量"""
        assert len(vector) == self.dimension
        self.vectors.append(np.array(vector))
        self.metadata.append(metadata or {})

    def add_batch(self, vectors, metadata_list=None):
        """批量添加"""
        for i, vec in enumerate(vectors):
            meta = metadata_list[i] if metadata_list and i < len(metadata_list) else {}
            self.add(vec, meta)

    def delete(self, index):
        """删除向量"""
        if 0 <= index < len(self.vectors):
            self.vectors.pop(index)
            self.metadata.pop(index)

    def search(self, query_vector, top_k=5, metric='cosine'):
        """检索最相似的向量"""
        query = np.array(query_vector)
        scores = []

        for i, vec in enumerate(self.vectors):
            if metric == 'cosine':
                score = np.dot(query, vec) / (np.linalg.norm(query) * np.linalg.norm(vec) + 1e-10)
            elif metric == 'l2':
                score = -np.linalg.norm(query - vec)  # 负距离, 越大越近
            elif metric == 'ip':
                score = np.dot(query, vec)  # 内积
            scores.append((i, float(score), self.metadata[i]))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def __len__(self):
        return len(self.vectors)


# 构建向量数据库
np.random.seed(42)
dim = 64
db = SimpleVectorDB(dimension=dim)

# 模拟文档嵌入
doc_texts = [
    "机器学习是人工智能的重要分支",
    "深度学习使用多层神经网络",
    "自然语言处理研究语言理解",
    "计算机视觉处理图像和视频",
    "强化学习通过奖励训练智能体",
    "Python是数据科学的常用语言",
    "Transformer改变了NLP领域",
    "RAG结合检索和生成",
    "向量数据库用于相似度搜索",
    "BERT是预训练语言模型",
    "GPT系列是生成式模型",
    "知识图谱存储结构化知识",
    "推荐系统个性化推荐内容",
    "情感分析判断文本情感",
    "文本摘要自动生成摘要",
]

# 为每个文档生成模拟嵌入
for i, text in enumerate(doc_texts):
    vec = np.random.randn(dim)
    # 添加语义偏移, 让相关文档向量方向相近
    if "学习" in text:
        vec[:10] += np.array([1, 0.8, 0.6, 0.4, 0.2, 0, 0, 0, 0, 0])
    if "语言" in text or "NLP" in text or "文本" in text:
        vec[10:20] += np.array([1, 0.8, 0.6, 0.4, 0.2, 0, 0, 0, 0, 0])
    if "模型" in text:
        vec[20:30] += np.array([1, 0.8, 0.6, 0.4, 0.2, 0, 0, 0, 0, 0])
    vec = vec / np.linalg.norm(vec)
    db.add(vec, {"text": text, "id": i})

print(f"  向量数据库大小: {len(db)} 条记录")
print(f"  向量维度: {dim}")

# 检索测试
query_text = "深度学习方法"
query_vec = np.random.randn(dim)
query_vec[:10] += np.array([1, 0.8, 0.6, 0.4, 0.2, 0, 0, 0, 0, 0])
query_vec = query_vec / np.linalg.norm(query_vec)

results = db.search(query_vec, top_k=5)
print(f"\n  查询: '{query_text}'")
print(f"  Top-5 检索结果:")
for idx, score, meta in results:
    print(f"    [{idx}] 相似度={score:.4f}: {meta['text']}")

# ============================================================
# 3. ANN 近似最近邻搜索概念
# ============================================================
print("\n--- 3. ANN 近似最近邻搜索 ---")
print("""
  精确搜索 vs 近似搜索:
    精确搜索 (Exact): 遍历所有向量, 保证找到最优, O(N)
    近似搜索 (ANN):   牺牲少量精度, 大幅提升速度, O(log N)

  常见ANN索引类型:

  1) Flat (暴力搜索)
     精确搜索, 适合小数据集 (<100K)

  2) IVF (Inverted File Index)
     将向量空间聚类为n个cell, 查询时只搜索最近的nprobe个cell
     参数: nlist(聚类数), nprobe(搜索数)
     适合: 中大数据集

  3) HNSW (Hierarchical Navigable Small World)
     基于图的方法, 多层跳表结构
     参数: M(连接数), efConstruction(构建参数), efSearch(搜索参数)
     适合: 需要高召回率的场景

  4) PQ (Product Quantization)
     将向量切分为子空间, 每个子空间量化
     大幅压缩存储空间 (8x-64x)
     适合: 内存受限的大数据集

  5) IVF + PQ 组合
     先聚类, 再量化, 兼顾速度和内存
""")

# ============================================================
# 4. 简化IVF索引实现
# ============================================================
print("\n--- 4. 简化IVF索引实现 ---")


class SimpleIVFIndex:
    """简化IVF索引"""

    def __init__(self, n_clusters=4, n_probe=2):
        self.n_clusters = n_clusters
        self.n_probe = n_probe
        self.centroids = None
        self.clusters = {}  # cluster_id -> [(vector, metadata), ...]

    def build(self, vectors, metadata_list=None):
        """构建IVF索引 (简化K-means)"""
        vectors = np.array(vectors)
        n = len(vectors)

        # 随机初始化聚类中心
        indices = np.random.choice(n, self.n_clusters, replace=False)
        self.centroids = vectors[indices].copy()

        # 简单K-means迭代
        for iteration in range(10):
            # 分配每个向量到最近的聚类中心
            assignments = []
            for vec in vectors:
                dists = [np.linalg.norm(vec - c) for c in self.centroids]
                assignments.append(np.argmin(dists))

            # 更新聚类中心
            new_centroids = []
            for k in range(self.n_clusters):
                members = [vectors[i] for i in range(n) if assignments[i] == k]
                if members:
                    new_centroids.append(np.mean(members, axis=0))
                else:
                    new_centroids.append(self.centroids[k])
            self.centroids = np.array(new_centroids)

        # 构建倒排索引
        self.clusters = {k: [] for k in range(self.n_clusters)}
        for i, vec in enumerate(vectors):
            dists = [np.linalg.norm(vec - c) for c in self.centroids]
            cluster_id = np.argmin(dists)
            meta = metadata_list[i] if metadata_list and i < len(metadata_list) else {}
            self.clusters[cluster_id].append((vec, meta, i))

        for k in range(self.n_clusters):
            print(f"  Cluster {k}: {len(self.clusters[k])} 个向量")

    def search(self, query_vector, top_k=5):
        """IVF检索"""
        query = np.array(query_vector)

        # 找到最近的nprobe个聚类
        dists = [np.linalg.norm(query - c) for c in self.centroids]
        nearest_clusters = np.argsort(dists)[:self.n_probe]

        # 在这些聚类中搜索
        candidates = []
        for cid in nearest_clusters:
            for vec, meta, idx in self.clusters[cid]:
                score = np.dot(query, vec) / (np.linalg.norm(query) * np.linalg.norm(vec) + 1e-10)
                candidates.append((idx, score, meta))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]


# 构建IVF索引
print("  构建IVF索引...")
vectors_list = [db.vectors[i] for i in range(len(db))]
meta_list = [db.metadata[i] for i in range(len(db))]

ivf = SimpleIVFIndex(n_clusters=4, n_probe=2)
ivf.build(vectors_list, meta_list)

# IVF检索
ivf_results = ivf.search(query_vec, top_k=5)
print(f"\n  IVF检索结果 (nprobe=2):")
for idx, score, meta in ivf_results:
    print(f"    [{idx}] 相似度={score:.4f}: {meta['text']}")

# ============================================================
# 5. 可视化
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 子图1: 向量空间中的聚类
ax1 = axes[0]
# 2D投影
all_vecs = np.array(vectors_list)
from numpy.linalg import svd
U, S, Vt = np.linalg.svd(all_vecs - all_vecs.mean(axis=0), full_matrices=False)
coords = (all_vecs - all_vecs.mean(axis=0)) @ Vt[:2].T

# 聚类分配
for k in range(ivf.n_clusters):
    cluster_vecs = [v for v, m, i in ivf.clusters[k]]
    cluster_coords = (np.array(cluster_vecs) - all_vecs.mean(axis=0)) @ Vt[:2].T
    ax1.scatter(cluster_coords[:, 0], cluster_coords[:, 1], s=80, alpha=0.7,
                label=f'Cluster {k}', edgecolors='black')

# 聚类中心
centroid_coords = (ivf.centroids - all_vecs.mean(axis=0)) @ Vt[:2].T
ax1.scatter(centroid_coords[:, 0], centroid_coords[:, 1], c='red', s=200, marker='X',
            label='聚类中心', edgecolors='black', zorder=10)

ax1.set_title('IVF 聚类可视化 (2D投影)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# 子图2: 索引类型对比
ax2 = axes[1]
index_types = ['Flat\n(精确)', 'IVF\n(nprobe=1)', 'IVF\n(nprobe=5)', 'HNSW\n(ef=50)', 'PQ\n(m=8)']
search_times = [100, 15, 40, 25, 10]  # 相对搜索时间
recalls = [100, 75, 95, 98, 80]       # 召回率 %

x = np.arange(len(index_types))
bars1 = ax2.bar(x - 0.2, search_times, 0.35, label='相对搜索时间', color='#e74c3c', alpha=0.85)
ax2_twin = ax2.twinx()
bars2 = ax2_twin.bar(x + 0.2, recalls, 0.35, label='召回率 %', color='#3498db', alpha=0.85)

ax2.set_xticks(x)
ax2.set_xticklabels(index_types, fontsize=10)
ax2.set_ylabel('相对搜索时间', fontsize=12, color='#e74c3c')
ax2_twin.set_ylabel('召回率 %', fontsize=12, color='#3498db')
ax2.set_title('不同索引类型: 速度 vs 精度', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 120)
ax2_twin.set_ylim(0, 120)
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper right')
ax2.grid(True, alpha=0.3, axis='y')

# 子图3: 数据规模 vs 搜索时间
ax3 = axes[2]
data_sizes = [1000, 10000, 100000, 1000000, 10000000]
flat_time = [0.001, 0.01, 0.1, 1.0, 10.0]
ivf_time =  [0.001, 0.003, 0.01, 0.05, 0.3]
hnsw_time = [0.001, 0.002, 0.005, 0.02, 0.1]

ax3.plot(data_sizes, flat_time, 'o-', label='Flat (精确)', linewidth=2, markersize=8)
ax3.plot(data_sizes, ivf_time, 's-', label='IVF', linewidth=2, markersize=8)
ax3.plot(data_sizes, hnsw_time, '^-', label='HNSW', linewidth=2, markersize=8)

ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xlabel('数据量 (向量数)', fontsize=12)
ax3.set_ylabel('搜索时间 (秒, 估计)', fontsize=12)
ax3.set_title('数据规模 vs 搜索延迟', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W18/vector_database.png', dpi=150, bbox_inches='tight')
print("\n  图表已保存: vector_database.png")
plt.close()

# ============================================================
# 6. 使用FAISS (可选)
# ============================================================
print("\n--- 6. 使用 FAISS (可选) ---")

try:
    import faiss

    # 准备数据
    vectors_np = np.array(vectors_list).astype('float32')

    # Flat 索引 (精确搜索)
    print("\n  --- Flat 索引 ---")
    index_flat = faiss.IndexFlatIP(dim)  # 内积
    index_flat.add(vectors_np)
    print(f"  索引中向量数: {index_flat.ntotal}")

    D, I = index_flat.search(query_vec.reshape(1, -1).astype('float32'), 5)
    print(f"  Flat检索 Top-5 索引: {I[0]}")
    print(f"  Flat检索 Top-5 分数: {D[0]}")

    # IVF 索引
    print("\n  --- IVF 索引 ---")
    nlist = 4
    quantizer = faiss.IndexFlatIP(dim)
    index_ivf = faiss.IndexIVFFlat(quantizer, dim, nlist)
    index_ivf.train(vectors_np)
    index_ivf.add(vectors_np)
    index_ivf.nprobe = 2

    D_ivf, I_ivf = index_ivf.search(query_vec.reshape(1, -1).astype('float32'), 5)
    print(f"  IVF检索 Top-5 索引: {I_ivf[0]}")
    print(f"  IVF检索 Top-5 分数: {D_ivf[0]}")

    # HNSW 索引
    print("\n  --- HNSW 索引 ---")
    index_hnsw = faiss.IndexHNSWFlat(dim, 16)  # M=16
    index_hnsw.add(vectors_np)

    D_hnsw, I_hnsw = index_hnsw.search(query_vec.reshape(1, -1).astype('float32'), 5)
    print(f"  HNSW检索 Top-5 索引: {I_hnsw[0]}")
    print(f"  HNSW检索 Top-5 分数: {D_hnsw[0]}")

except ImportError:
    print("  [!] faiss 未安装, 跳过FAISS演示")
    print("  安装命令: pip install faiss-cpu  (或 faiss-gpu)")
except Exception as e:
    print(f"  [!] FAISS运行出错: {e}")

# ============================================================
# 7. 向量数据库选择指南
# ============================================================
print("\n--- 7. 向量数据库选择指南 ---")
print("""
  选择建议:

  原型/小项目:   ChromaDB (最简单, Python原生)
  中等规模:      FAISS (灵活, 高性能)
  生产环境:      Milvus / Qdrant (分布式, 高可用)
  已有PG数据库:  pgvector (无缝集成PostgreSQL)
  全托管:        Pinecone (零运维)
  混合检索:      Weaviate (支持BM25+向量)

  性能参考 (100万向量, 768维):
    Flat:    搜索 ~100ms,  100%召回率
    IVF:     搜索 ~5ms,    95%召回率
    HNSW:    搜索 ~2ms,    99%召回率
    PQ:      搜索 ~1ms,    90%召回率 (压缩8x)
""")

print("\n" + "=" * 60)
print("W18-D4 完成! 本节实现了向量数据库的核心功能")
print("=" * 60)
