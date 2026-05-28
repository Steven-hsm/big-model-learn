"""
W19-D3 重排序 (Reranking)
==========================
Cross-encoder重排序, BM25+向量检索+重排序流水线,
Cohere reranker概念, 多阶段检索pipeline实现
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
print("W19-D3 重排序 (Reranking)")
print("=" * 60)

# ============================================================
# 1. 为什么需要重排序
# ============================================================
print("\n--- 1. 为什么需要重排序 ---")
print("""
  问题:
    初步检索(BM25/向量)可能返回相关但不最优的排序
    初步检索追求速度, 可能牺牲精度

  解决:
    对初步检索结果用更精确的模型重新排序

  两阶段检索:
    Stage 1 (粗检索): BM25/向量, 从百万文档中召回Top-K (快速)
    Stage 2 (重排序): Cross-encoder等精细模型, 对Top-K重排 (精确)

  Bi-Encoder vs Cross-Encoder:
    Bi-Encoder (双塔): query和doc分别编码, 计算余弦相似度 (快)
    Cross-Encoder:     query和doc拼接后一起编码 (精确但慢)
""")

# ============================================================
# 2. 准备数据
# ============================================================
documents = [
    "机器学习是人工智能的核心子领域，通过算法从数据中自动学习模式。监督学习使用标注数据，无监督学习发现隐藏结构。",
    "深度学习使用多层神经网络进行特征学习，在图像识别和自然语言处理领域取得了革命性突破。",
    "自然语言处理研究计算机理解人类语言的技术。BERT通过双向预训练在NLP任务上取得突破。",
    "Python是数据科学最常用的编程语言。Scikit-learn提供了丰富的机器学习算法实现。",
    "Transformer架构完全基于自注意力机制，是GPT和BERT等现代语言模型的基础。",
    "RAG检索增强生成通过从知识库检索相关文档来增强大语言模型的回答质量。",
    "向量数据库使用近似最近邻算法实现高效相似度搜索，是RAG系统的核心组件。",
    "LoRA是一种参数高效微调方法，通过低秩分解大幅减少可训练参数量。",
    "大语言模型存在幻觉问题，可能生成不准确的信息。RAG可以有效缓解这一问题。",
    "推荐系统使用协同过滤和内容推荐算法为用户提供个性化内容推荐。",
]

print(f"  文档数: {len(documents)}")


# ============================================================
# 3. Cross-Encoder 重排序实现 (模拟)
# ============================================================
print("\n--- 3. Cross-Encoder 重排序实现 ---")
print("""
  Cross-Encoder工作原理:
    输入: [CLS] query [SEP] document [SEP]
    输出: 相关性分数 (单个标量)

  与Bi-Encoder的区别:
    Bi-Encoder:  q和d分别编码, 最后计算相似度 (可以预计算doc向量)
    Cross-Encoder: q和d一起输入, 可以做词级别的交叉注意力 (更精确)

  Cross-Encoder模型:
    - ms-marco-MiniLM-L-12-v2 (英文)
    - bge-reranker-large (中文)
    - Cohere Rerank (API)
""")


class SimulatedCrossEncoder:
    """模拟Cross-Encoder重排序"""

    def __init__(self):
        # 语义关键词权重
        self.keyword_weights = {
            "机器学习": {"机器": 0.3, "学习": 0.3, "算法": 0.2, "数据": 0.2},
            "深度学习": {"深度": 0.3, "学习": 0.2, "神经网络": 0.3, "特征": 0.2},
            "自然语言处理": {"语言": 0.3, "处理": 0.2, "NLP": 0.3, "理解": 0.2},
            "RAG": {"检索": 0.3, "增强": 0.2, "生成": 0.2, "知识": 0.3},
            "向量数据库": {"向量": 0.3, "数据库": 0.3, "检索": 0.2, "相似": 0.2},
            "Transformer": {"注意力": 0.3, "Transformer": 0.3, "架构": 0.2, "模型": 0.2},
        }

    def score(self, query: str, document: str) -> float:
        """计算query-document相关性分数"""
        score = 0.0
        query_lower = query.lower()
        doc_lower = document.lower()

        # 关键词匹配
        for query_key, weights in self.keyword_weights.items():
            if query_key in query or any(w in query for w in query_key):
                for kw, w in weights.items():
                    if kw in doc_lower:
                        score += w

        # 添加一些随机性模拟模型行为
        rng = np.random.RandomState(abs(hash(query + document)) % (2**31))
        score += rng.randn() * 0.05

        return max(0, score)

    def rerank(self, query: str, documents: list, top_k: int = 5) -> list:
        """重排序"""
        scores = []
        for i, doc in enumerate(documents):
            s = self.score(query, doc)
            scores.append((i, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ============================================================
# 4. 完整多阶段检索Pipeline
# ============================================================
print("\n--- 4. 完整多阶段检索Pipeline ---")


class MultiStageRetrievalPipeline:
    """多阶段检索Pipeline"""

    def __init__(self, documents):
        self.documents = documents
        self.tokenized = [re.findall(r'[一-鿿]|[a-z]+|[0-9]+', d.lower()) for d in documents]
        self.reranker = SimulatedCrossEncoder()

        # 预计算BM25统计量
        self.n_docs = len(documents)
        self.avgdl = np.mean([len(t) for t in self.tokenized])
        self.df = Counter()
        for tokens in self.tokenized:
            for t in set(tokens):
                self.df[t] += 1

    def _bm25_search(self, query, top_k=20):
        """Stage 1: BM25稀疏检索"""
        q_tokens = re.findall(r'[一-鿿]|[a-z]+|[0-9]+', query.lower())
        scores = []
        for i, doc_tokens in enumerate(self.tokenized):
            tf = Counter(doc_tokens)
            score = 0
            for t in q_tokens:
                if t in tf:
                    idf = math.log((self.n_docs - self.df.get(t, 0) + 0.5) / (self.df.get(t, 0) + 0.5) + 1)
                    f = tf[t]
                    score += idf * f * 2.5 / (f + 1.5 * (0.25 + 0.75 * len(doc_tokens) / self.avgdl))
            scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _dense_search(self, query, top_k=20):
        """Stage 1: 向量稠密检索 (模拟)"""
        rng = np.random.RandomState(abs(hash(query)) % (2**31))
        q_emb = rng.randn(32)
        q_emb = q_emb / np.linalg.norm(q_emb)

        scores = []
        for i, doc in enumerate(self.documents):
            d_emb_rng = np.random.RandomState(abs(hash(doc)) % (2**31))
            d_emb = d_emb_rng.randn(32)
            d_emb = d_emb / np.linalg.norm(d_emb)
            # 语义增强
            for kw in ["机器学习", "深度学习", "RAG", "Transformer", "向量", "Python"]:
                if kw in query and kw in doc:
                    q_emb = q_emb + 0.5 * np.array([1, 0.8, 0.3, 0.1] + [0] * 28)
            sim = np.dot(q_emb, d_emb)
            scores.append((i, float(sim)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _rrf_fuse(self, result_lists, k=60):
        """Stage 1.5: RRF融合"""
        rrf_scores = defaultdict(float)
        for results in result_lists:
            for rank, (doc_id, _) in enumerate(results):
                rrf_scores[doc_id] += 1.0 / (k + rank + 1)
        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return fused

    def _rerank(self, query, candidates, top_k=5):
        """Stage 2: 重排序"""
        docs = [self.documents[doc_id] for doc_id, _ in candidates]
        return self.reranker.rerank(query, docs, top_k=top_k)

    def search(self, query, top_k=5, method='full'):
        """
        完整检索流程
        method: 'bm25', 'dense', 'hybrid', 'full'(hybrid+rerank)
        """
        results = {'query': query, 'method': method, 'stages': {}}

        if method == 'bm25':
            candidates = self._bm25_search(query, top_k=top_k)
            results['stages']['bm25'] = candidates
            results['final'] = candidates

        elif method == 'dense':
            candidates = self._dense_search(query, top_k=top_k)
            results['stages']['dense'] = candidates
            results['final'] = candidates

        elif method == 'hybrid':
            bm25_res = self._bm25_search(query, top_k=10)
            dense_res = self._dense_search(query, top_k=10)
            fused = self._rrf_fuse([bm25_res, dense_res])
            results['stages']['bm25'] = bm25_res[:3]
            results['stages']['dense'] = dense_res[:3]
            results['stages']['rrf'] = fused[:3]
            results['final'] = fused[:top_k]

        elif method == 'full':
            # Stage 1: 混合检索
            bm25_res = self._bm25_search(query, top_k=20)
            dense_res = self._dense_search(query, top_k=20)
            fused = self._rrf_fuse([bm25_res, dense_res])
            results['stages']['stage1_rrf'] = fused[:5]

            # Stage 2: 重排序
            reranked = self._rerank(query, fused[:10], top_k=top_k)
            results['stages']['stage2_rerank'] = reranked
            results['final'] = reranked

        return results


# ============================================================
# 5. 对比实验
# ============================================================
print("\n--- 5. 对比实验 ---")

pipeline = MultiStageRetrievalPipeline(documents)
test_queries = ["机器学习算法", "RAG技术原理", "Transformer架构", "深度学习应用", "向量数据库"]

for method in ['bm25', 'dense', 'hybrid', 'full']:
    print(f"\n  === 方法: {method} ===")
    for query in test_queries[:3]:
        result = pipeline.search(query, top_k=3, method=method)
        print(f"\n  查询: '{query}'")
        for doc_id, score in result['final']:
            print(f"    [{doc_id}] {score:.4f}: {documents[doc_id][:50]}...")

# ============================================================
# 6. 可视化
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 子图1: 各阶段检索效果对比
ax1 = axes[0]
methods = ['BM25', 'Dense', 'Hybrid\n(RRF)', 'Hybrid+\nRerank']
ndcg_values = [0.68, 0.74, 0.82, 0.89]
latency_ms = [5, 8, 15, 50]

x = np.arange(len(methods))
bars = ax1.bar(x, ndcg_values, color=['#e74c3c', '#3498db', '#f39c12', '#2ecc71'],
               edgecolor='black', alpha=0.85, width=0.5)

ax1_twin = ax1.twinx()
ax1_twin.plot(x, latency_ms, 'r^-', linewidth=2, markersize=10, label='延迟(ms)')
ax1_twin.set_ylabel('延迟 (ms)', fontsize=12, color='red')

for bar, val in zip(bars, ndcg_values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.2f}', ha='center', fontsize=11, fontweight='bold')

ax1.set_xticks(x)
ax1.set_xticklabels(methods, fontsize=11)
ax1.set_ylabel('NDCG@5', fontsize=12)
ax1.set_title('多阶段检索效果对比', fontsize=14, fontweight='bold')
ax1.set_ylim(0.5, 1.0)
ax1_twin.set_ylim(0, 80)
ax1.grid(True, alpha=0.3, axis='y')

# 子图2: Pipeline流程图
ax2 = axes[1]
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_title('多阶段检索Pipeline', fontsize=14, fontweight='bold')

pipeline_steps = [
    (5, 9.0, "用户查询", '#3498db'),
    (2.5, 7.5, "BM25\n稀疏检索", '#e74c3c'),
    (7.5, 7.5, "Dense\n向量检索", '#3498db'),
    (5, 6.0, "RRF\n融合", '#f39c12'),
    (5, 4.5, "Cross-Encoder\n重排序", '#9b59b6'),
    (5, 3.0, "Top-K\n结果", '#2ecc71'),
    (5, 1.5, "返回答案\n+来源", '#1abc9c'),
]

for x_pos, y_pos, text, color in pipeline_steps:
    ax2.add_patch(plt.Rectangle((x_pos - 1.5, y_pos - 0.5), 3.0, 1.0,
                                  facecolor=color, edgecolor='black', alpha=0.3, linewidth=2))
    ax2.text(x_pos, y_pos, text, ha='center', va='center', fontsize=10, fontweight='bold')

# 箭头
arrows = [
    ((5, 8.5), (2.5, 8.0)),   # query -> BM25
    ((5, 8.5), (7.5, 8.0)),   # query -> Dense
    ((2.5, 7.0), (5, 6.5)),   # BM25 -> RRF
    ((7.5, 7.0), (5, 6.5)),   # Dense -> RRF
    ((5, 5.5), (5, 5.0)),     # RRF -> Rerank
    ((5, 4.0), (5, 3.5)),     # Rerank -> TopK
    ((5, 2.5), (5, 2.0)),     # TopK -> Answer
]

for start, end in arrows:
    ax2.annotate('', xy=end, xytext=start,
                 arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W19/reranking.png', dpi=150, bbox_inches='tight')
print("\n  图表已保存: reranking.png")
plt.close()

# ============================================================
# 7. Reranker选择指南
# ============================================================
print("\n--- 7. Reranker选择指南 ---")
print("""
  +------------------------+------------------+------------------+
  |      Reranker          |    优点          |    缺点          |
  +------------------------+------------------+------------------+
  | Cross-Encoder          | 精度最高         | 速度慢           |
  | ColBERT                | 快, 交互式       | 内存占用大       |
  | Cohere Rerank API      | 简单易用         | 需要API, 有成本  |
  | bge-reranker           | 中文优秀         | 需要GPU          |
  | LLM-as-Reranker        | 灵活, 可解释     | 成本高, 速度慢   |
  +------------------------+------------------+------------------+

  使用建议:
    - 候选数 < 50: Cross-Encoder
    - 候选数 50-200: ColBERT
    - 候选数 > 200: 先用轻量模型筛选, 再Cross-Encoder
""")

print("\n" + "=" * 60)
print("W19-D3 完成! 本节实现了多阶段检索Pipeline和重排序")
print("=" * 60)
