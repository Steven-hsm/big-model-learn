"""
W19-D6 RAG优化 (RAG Optimization)
==================================
缓存策略(语义缓存), 索引优化(量化/压缩),
批量检索, 流式生成, 性能优化技巧总结
"""

import sys
import re
import time
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from typing import List, Dict, Optional, Tuple

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W19-D6 RAG优化 (RAG Optimization)")
print("=" * 60)

# ============================================================
# 1. RAG性能瓶颈分析
# ============================================================
print("\n--- 1. RAG性能瓶颈分析 ---")
print("""
  RAG系统各阶段耗时分布 (典型):
    文本切分:       5%   (离线)
    向量化:         15%  (离线)
    查询向量化:     10%  (在线)
    向量检索:       20%  (在线)
    LLM生成:       50%  (在线)
    其他:           5%   (在线)

  优化方向:
    1) 缓存:     减少重复计算
    2) 索引优化: 加速向量检索
    3) 量化压缩: 减少内存占用
    4) 批处理:   提高吞吐量
    5) 流式生成: 改善用户体验
""")

# ============================================================
# 2. 语义缓存
# ============================================================
print("\n--- 2. 语义缓存 (Semantic Cache) ---")
print("""
  语义缓存: 相似问题的答案可以复用

  工作原理:
    1) 新查询 -> 向量化
    2) 与缓存中的查询比较相似度
    3) 相似度 > 阈值 -> 返回缓存的答案
    4) 相似度 < 阈值 -> 正常检索+生成, 缓存结果

  优势:
    - 相同/相似问题秒级响应
    - 大幅减少LLM调用成本
    - 特别适合FAQ类场景
""")


class SemanticCache:
    """语义缓存实现"""

    def __init__(self, similarity_threshold=0.92, max_size=100):
        self.threshold = similarity_threshold
        self.max_size = max_size
        self.cache = OrderedDict()  # query_hash -> (embedding, answer, metadata)
        self.stats = {'hits': 0, 'misses': 0}

    def _mock_embedding(self, text: str, dim: int = 32) -> np.ndarray:
        """模拟嵌入"""
        rng = np.random.RandomState(abs(hash(text)) % (2**31))
        base = rng.randn(dim) * 0.3
        kw_vecs = {
            "机器学习": [1, .8, 0, 0], "深度学习": [.9, .7, .3, 0],
            "RAG": [.3, .4, .5, .7], "Python": [.7, .5, .1, .2],
        }
        semantic = np.zeros(dim)
        for kw, vec in kw_vecs.items():
            if kw in text:
                semantic[:4] += np.array(vec)
        combined = base + semantic
        return combined / (np.linalg.norm(combined) + 1e-10)

    def _cosine_sim(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

    def get(self, query: str) -> Optional[Tuple[str, Dict]]:
        """查找缓存"""
        query_emb = self._mock_embedding(query)

        best_sim = 0
        best_key = None
        for key, (emb, answer, meta) in self.cache.items():
            sim = self._cosine_sim(query_emb, emb)
            if sim > best_sim:
                best_sim = sim
                best_key = key

        if best_sim >= self.threshold and best_key is not None:
            self.stats['hits'] += 1
            # 移到最新
            self.cache.move_to_end(best_key)
            return self.cache[best_key][1], {**self.cache[best_key][2], 'similarity': best_sim}

        self.stats['misses'] += 1
        return None

    def set(self, query: str, answer: str, metadata: Dict = None):
        """存入缓存"""
        query_emb = self._mock_embedding(query)
        key = hashlib.md5(query.encode()).hexdigest()

        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)  # LRU淘汰

        self.cache[key] = (query_emb, answer, metadata or {})
        self.cache.move_to_end(key)

    def get_stats(self) -> Dict:
        total = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total if total > 0 else 0
        return {
            'cache_size': len(self.cache),
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'hit_rate': hit_rate,
        }


# 测试语义缓存
cache = SemanticCache(similarity_threshold=0.90)

# 模拟查询
queries_and_answers = [
    ("什么是机器学习", "机器学习是AI的子领域，通过算法从数据中学习。"),
    ("深度学习的原理", "深度学习使用多层神经网络进行特征学习。"),
    ("RAG是什么", "RAG是检索增强生成技术，通过检索增强LLM回答。"),
    ("Python常用库", "NumPy, Pandas, PyTorch, TensorFlow等。"),
    ("什么是机器学习方法", "机器学习方法包括监督学习、无监督学习和强化学习。"),
    ("深度学习怎么学", "建议从神经网络基础开始，然后学习CNN和Transformer。"),
]

print("\n  语义缓存测试:")
for query, answer in queries_and_answers:
    cached = cache.get(query)
    if cached:
        print(f"  [命中] '{query}' -> 相似度={cached[1]['similarity']:.3f}")
    else:
        cache.set(query, answer)
        print(f"  [缓存] '{query}' -> 存入缓存")

# 再次查询相似问题
print("\n  重复查询测试:")
similar_queries = ["什么是机器学习", "机器学习的概念是什么", "深度学习原理"]
for q in similar_queries:
    cached = cache.get(q)
    if cached:
        print(f"  [命中] '{q}' -> 相似度={cached[1]['similarity']:.3f}")
    else:
        print(f"  [未命中] '{q}'")

stats = cache.get_stats()
print(f"\n  缓存统计: 命中率={stats['hit_rate']:.2%}, 命中={stats['hits']}, 未命中={stats['misses']}")

# ============================================================
# 3. 索引优化
# ============================================================
print("\n--- 3. 索引优化 ---")
print("""
  向量索引优化方法:

  1) 量化 (Quantization)
     - PQ (Product Quantization): 向量切分为子空间, 每个子空间用少量中心点表示
     - 压缩比: 8x-64x, 精度损失: 2-5%
     - 适合: 内存受限, 大规模数据

  2) 标量量化
     - float32 -> int8
     - 压缩比: 4x, 精度损失: <1%
     - 适合: 快速压缩

  3) 索引选择
     - < 100K:     Flat (精确)
     - 100K - 1M:  IVF + PQ
     - > 1M:       HNSW 或 IVF + HNSW

  4) 分区策略
     - 按时间分区: 方便增量更新
     - 按主题分区: 提高检索精度
""")


class QuantizedVectorIndex:
    """简化PQ量化索引"""

    def __init__(self, n_subvectors=4, n_centroids=16):
        self.n_subvectors = n_subvectors
        self.n_centroids = n_centroids
        self.codebooks = []
        self.codes = []

    def fit(self, vectors: np.ndarray):
        """训练量化码本"""
        n, dim = vectors.shape
        sub_dim = dim // self.n_subvectors

        self.codebooks = []
        for i in range(self.n_subvectors):
            sub_vectors = vectors[:, i * sub_dim:(i + 1) * sub_dim]
            # 简化K-means: 随机选择中心点
            indices = np.random.choice(n, min(self.n_centroids, n), replace=False)
            self.codebooks.append(sub_vectors[indices])

        # 编码
        self.codes = self._encode(vectors)
        self.original_shape = vectors.shape

        # 计算压缩率
        original_bytes = n * dim * 4  # float32
        compressed_bytes = n * self.n_subvectors * 1  # uint8 code
        compression_ratio = original_bytes / compressed_bytes
        print(f"  原始大小: {original_bytes / 1024:.1f} KB")
        print(f"  压缩后: {compressed_bytes / 1024:.1f} KB")
        print(f"  压缩比: {compression_ratio:.1f}x")

    def _encode(self, vectors: np.ndarray) -> np.ndarray:
        """将向量编码为PQ码"""
        n, dim = vectors.shape
        sub_dim = dim // self.n_subvectors
        codes = np.zeros((n, self.n_subvectors), dtype=np.int8)

        for i in range(self.n_subvectors):
            sub_vectors = vectors[:, i * sub_dim:(i + 1) * sub_dim]
            centroids = self.codebooks[i]
            # 找最近的中心点
            for j in range(n):
                dists = np.linalg.norm(sub_vectors[j] - centroids, axis=1)
                codes[j, i] = np.argmin(dists)

        return codes

    def search(self, query: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        """量化搜索 (非对称距离计算)"""
        dim = len(query)
        sub_dim = dim // self.n_subvectors

        # 预计算查询到各码字的距离表
        dist_tables = []
        for i in range(self.n_subvectors):
            sub_query = query[i * sub_dim:(i + 1) * sub_dim]
            dists = np.linalg.norm(self.codebooks[i] - sub_query, axis=1) ** 2
            dist_tables.append(dists)

        # 计算每个向量的近似距离
        scores = []
        for j in range(len(self.codes)):
            dist = sum(dist_tables[i][self.codes[j, i]] for i in range(self.n_subvectors))
            scores.append((j, dist))

        scores.sort(key=lambda x: x[1])
        return scores[:top_k]


# 测试量化索引
np.random.seed(42)
n_vectors = 1000
dim = 32
vectors = np.random.randn(n_vectors, dim).astype(np.float32)

print("\n  PQ量化索引测试:")
pq_index = QuantizedVectorIndex(n_subvectors=4, n_centroids=16)
pq_index.fit(vectors)

query = np.random.randn(dim).astype(np.float32)
results = pq_index.search(query, top_k=5)
print(f"\n  Top-5 检索结果 (PQ):")
for idx, dist in results:
    print(f"    [{idx}] 距离={dist:.4f}")

# 对比精确搜索
exact_dists = [(i, np.linalg.norm(vectors[i] - query)) for i in range(n_vectors)]
exact_dists.sort(key=lambda x: x[1])
print(f"\n  Top-5 检索结果 (精确):")
for idx, dist in exact_dists[:5]:
    print(f"    [{idx}] 距离={dist:.4f}")

# ============================================================
# 4. 批量检索
# ============================================================
print("\n--- 4. 批量检索 ---")
print("""
  批量检索: 同时处理多个查询, 提高吞吐量

  技巧:
    1) 批量向量化: 一次编码多个查询 (GPU并行)
    2) 批量检索:   复用索引加载, 减少IO
    3) 异步处理:   查询和生成并行
""")


def batch_search(query_embeddings: np.ndarray, doc_embeddings: np.ndarray, top_k: int = 5) -> List[List[Tuple]]:
    """批量检索"""
    # 矩阵乘法: (n_queries, dim) @ (dim, n_docs) -> (n_queries, n_docs)
    sim_matrix = query_embeddings @ doc_embeddings.T

    results = []
    for i in range(len(query_embeddings)):
        top_indices = np.argsort(sim_matrix[i])[-top_k:][::-1]
        results.append([(int(idx), float(sim_matrix[i, idx])) for idx in top_indices])
    return results


# 测试批量检索
n_queries = 10
query_embs = np.random.randn(n_queries, dim)
doc_embs = np.random.randn(100, dim)
query_embs = query_embs / np.linalg.norm(query_embs, axis=1, keepdims=True)
doc_embs = doc_embs / np.linalg.norm(doc_embs, axis=1, keepdims=True)

start = time.time()
batch_results = batch_search(query_embs, doc_embs, top_k=5)
elapsed = time.time() - start
print(f"  批量检索: {n_queries}个查询, 耗时{elapsed*1000:.2f}ms")
print(f"  每个查询平均: {elapsed/n_queries*1000:.2f}ms")

# ============================================================
# 5. 流式生成
# ============================================================
print("\n--- 5. 流式生成 ---")
print("""
  流式生成 (Streaming):
    逐token返回生成结果, 不需要等全部生成完

  实现方式:
    1) OpenAI API: stream=True 参数
    2) HuggingFace: TextIteratorStreamer
    3) 自定义: 逐token yield

  优势:
    - 首token延迟低 (用户更快看到输出)
    - 更好的用户体验 (打字机效果)
    - 可随时中断生成
""")


def simulate_streaming(text: str, chunk_size: int = 3):
    """模拟流式生成"""
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]


stream_text = "检索增强生成(RAG)是一种结合信息检索和文本生成的技术，通过检索外部知识来增强回答质量。"
print("  模拟流式输出: ", end="")
for chunk in simulate_streaming(stream_text, chunk_size=4):
    print(chunk, end="", flush=True)
    time.sleep(0.05)
print()

# ============================================================
# 6. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 子图1: 缓存命中率
ax1 = axes[0, 0]
request_counts = [10, 50, 100, 200, 500, 1000]
hit_rates_no_cache = [0] * len(request_counts)
hit_rates_with_cache = [0, 15, 35, 55, 70, 80]  # 随请求增多缓存命中增加

ax1.plot(request_counts, hit_rates_with_cache, 'o-', linewidth=2, label='有语义缓存', color='#2ecc71')
ax1.plot(request_counts, hit_rates_no_cache, '--', linewidth=2, label='无缓存', color='#e74c3c')
ax1.set_xlabel('请求数', fontsize=12)
ax1.set_ylabel('缓存命中率 (%)', fontsize=12)
ax1.set_title('语义缓存命中率增长趋势', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 子图2: 量化精度损失
ax2 = axes[0, 1]
quant_methods = ['float32\n(原始)', 'float16', 'int8\n(标量)', 'PQ-8\n(4x压缩)', 'PQ-4\n(8x压缩)']
memory_usage = [100, 50, 25, 25, 12.5]
accuracy = [100, 99.5, 99, 97, 93]

x = np.arange(len(quant_methods))
bars = ax2.bar(x, memory_usage, color=['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6'],
               edgecolor='black', alpha=0.85)
ax2_twin = ax2.twinx()
ax2_twin.plot(x, accuracy, 'r^-', linewidth=2, markersize=10, label='检索精度%')
ax2_twin.set_ylabel('检索精度 (%)', fontsize=12, color='red')

for bar, mem in zip(bars, memory_usage):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{mem}%', ha='center', fontsize=9)

ax2.set_xticks(x)
ax2.set_xticklabels(quant_methods, fontsize=10)
ax2.set_ylabel('内存占用 (% of float32)', fontsize=12)
ax2.set_title('量化方法: 内存 vs 精度', fontsize=14, fontweight='bold')
ax2_twin.legend(fontsize=10, loc='center right')
ax2.grid(True, alpha=0.3, axis='y')

# 子图3: 各阶段延迟优化
ax3 = axes[1, 0]
stages = ['查询\n向量化', '向量\n检索', '重排序', 'LLM\n生成', '总延迟']
latency_before = [50, 30, 100, 2000, 2180]
latency_after = [10, 15, 40, 1500, 1565]  # 优化后

x = np.arange(len(stages))
width = 0.35
ax3.bar(x - width/2, latency_before, width, label='优化前', color='#e74c3c', alpha=0.85, edgecolor='black')
ax3.bar(x + width/2, latency_after, width, label='优化后', color='#2ecc71', alpha=0.85, edgecolor='black')
ax3.set_xticks(x)
ax3.set_xticklabels(stages, fontsize=10)
ax3.set_ylabel('延迟 (ms)', fontsize=12)
ax3.set_title('RAG各阶段延迟优化', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')

# 子图4: 流式vs非流式用户体验
ax4 = axes[1, 1]
time_points = np.arange(0, 5, 0.1)

# 非流式: 全部在最后一次性返回
non_stream_output = np.zeros_like(time_points)
non_stream_output[time_points >= 3.0] = 1.0

# 流式: 逐步返回
stream_output = np.clip((time_points - 0.5) / 3.0, 0, 1)

ax4.plot(time_points, non_stream_output, linewidth=3, label='非流式: 等3秒后全部返回', color='#e74c3c')
ax4.plot(time_points, stream_output, linewidth=3, label='流式: 0.5秒开始逐步返回', color='#2ecc71')
ax4.fill_between(time_points, stream_output, alpha=0.1, color='#2ecc71')
ax4.set_xlabel('时间 (秒)', fontsize=12)
ax4.set_ylabel('已显示内容比例', fontsize=12)
ax4.set_title('流式 vs 非流式 用户体验', fontsize=14, fontweight='bold')
ax4.legend(fontsize=10, loc='lower right')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W19/rag_optimization.png', dpi=150, bbox_inches='tight')
print("\n  图表已保存: rag_optimization.png")
plt.close()

# ============================================================
# 7. 优化技巧总结
# ============================================================
print("\n--- 7. RAG优化技巧总结 ---")
tips = [
    ("语义缓存",     "相似查询复用缓存, 减少LLM调用",             "延迟降低60-80%"),
    ("向量量化",     "PQ/int8量化, 减少内存占用",                  "内存减少4-8x"),
    ("批量检索",     "多个查询并行处理",                            "吞吐量提升5-10x"),
    ("流式生成",     "逐token返回, 改善用户体验",                  "首token延迟降低80%"),
    ("异步管道",     "检索和生成并行, 流水线处理",                   "总延迟降低30-40%"),
    ("预计算缓存",   "预计算热门查询的嵌入和检索结果",              "热门查询秒级响应"),
    ("索引优化",     "根据数据规模选择合适索引类型",                 "检索延迟降低50-90%"),
    ("连接池",       "复用数据库和API连接",                          "减少连接开销"),
]

print(f"  {'技巧':<12s} {'描述':<36s} {'效果':<22s}")
print(f"  {'-'*12} {'-'*36} {'-'*22}")
for tip, desc, effect in tips:
    print(f"  {tip:<12s} {desc:<36s} {effect:<22s}")

print("\n" + "=" * 60)
print("W19-D6 完成! 本节学习了RAG系统的各种优化技术")
print("=" * 60)
