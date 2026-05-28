"""
W30-D1 BM25+向量混合检索
========================
实现BM25关键词检索与向量语义检索的混合策略，
使用RRF(Reciprocal Rank Fusion)进行结果融合，
并对比不同检索方式的质量。

核心概念:
- BM25: 基于词频的经典信息检索算法
- 向量检索: 基于语义相似度的检索
- RRF: 将多个排序列表融合为统一排序的方法
"""

import numpy as np
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

# ============================================================
# 1. BM25 检索器
# ============================================================

class SimpleBM25:
    """简化版BM25检索器

    BM25公式:
    score(D, Q) = sum( IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D|/avgdl)) )

    其中:
    - f(qi, D): 词qi在文档D中的频率
    - |D|: 文档D的长度
    - avgdl: 所有文档的平均长度
    - k1, b: 调节参数
    """

    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.doc_freqs = []       # 每篇文档的词频
        self.df = defaultdict(int)  # 文档频率
        self.doc_len = []         # 文档长度
        self.avgdl = 0            # 平均文档长度
        self.N = 0                # 文档总数

    def _tokenize(self, text):
        """简单分词: 按空格和标点切分, 转小写"""
        import re
        return re.findall(r'\w+', text.lower())

    def fit(self, corpus):
        """构建索引"""
        self.corpus = corpus
        self.N = len(corpus)

        for doc in corpus:
            tokens = self._tokenize(doc)
            self.doc_len.append(len(tokens))

            # 统计每篇文档的词频
            tf = defaultdict(int)
            for token in tokens:
                tf[token] += 1
            self.doc_freqs.append(tf)

            # 统计文档频率(包含该词的文档数)
            for token in set(tokens):
                self.df[token] += 1

        self.avgdl = np.mean(self.doc_len) if self.doc_len else 1
        print(f"[BM25] 索引构建完成: {self.N}篇文档, 平均长度={self.avgdl:.1f}")

    def _idf(self, token):
        """计算逆文档频率 IDF"""
        df = self.df.get(token, 0)
        return np.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def query(self, query_text, top_k=5):
        """对查询进行检索, 返回 (文档索引, 分数) 列表"""
        query_tokens = self._tokenize(query_text)
        scores = []

        for idx, doc_tf in enumerate(self.doc_freqs):
            score = 0.0
            for token in query_tokens:
                if token not in doc_tf:
                    continue
                tf = doc_tf[token]
                idf = self._idf(token)
                # BM25核心公式
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * self.doc_len[idx] / self.avgdl)
                score += idf * numerator / denominator
            scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ============================================================
# 2. 向量检索器 (基于余弦相似度)
# ============================================================

class SimpleVectorRetriever:
    """简化版向量检索器

    使用TF-IDF向量作为文档表示, 计算余弦相似度。
    实际生产中应使用预训练模型(如BGE/E5)的embedding。
    """

    def __init__(self):
        self.corpus = []
        self.vectors = None
        self.vocab = {}

    def _tokenize(self, text):
        import re
        return re.findall(r'\w+', text.lower())

    def fit(self, corpus):
        """构建TF-IDF向量"""
        self.corpus = corpus
        N = len(corpus)

        # 构建词汇表
        all_tokens = set()
        doc_tokens_list = []
        for doc in corpus:
            tokens = self._tokenize(doc)
            doc_tokens_list.append(tokens)
            all_tokens.update(tokens)

        self.vocab = {word: idx for idx, word in enumerate(sorted(all_tokens))}
        vocab_size = len(self.vocab)

        # 计算IDF
        df = defaultdict(int)
        for tokens in doc_tokens_list:
            for token in set(tokens):
                df[token] += 1

        idf = {}
        for word, idx in self.vocab.items():
            idf[word] = np.log((N + 1) / (df[word] + 1)) + 1

        # 构建TF-IDF向量
        self.vectors = np.zeros((N, vocab_size))
        for doc_idx, tokens in enumerate(doc_tokens_list):
            tf = defaultdict(int)
            for t in tokens:
                tf[t] += 1
            for word, count in tf.items():
                self.vectors[doc_idx, self.vocab[word]] = count * idf[word]

        # L2归一化
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.vectors = self.vectors / norms

        print(f"[向量检索] 索引构建完成: {N}篇文档, 词汇表大小={vocab_size}")

    def query(self, query_text, top_k=5):
        """对查询进行向量检索"""
        tokens = self._tokenize(query_text)
        query_vec = np.zeros(len(self.vocab))

        tf = defaultdict(int)
        for t in tokens:
            tf[t] += 1
        for word, count in tf.items():
            if word in self.vocab:
                query_vec[self.vocab[word]] = count

        # 归一化查询向量
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        # 计算余弦相似度
        similarities = self.vectors @ query_vec
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [(idx, similarities[idx]) for idx in top_indices]


# ============================================================
# 3. RRF 融合
# ============================================================

def reciprocal_rank_fusion(result_lists, k=60):
    """RRF(Reciprocal Rank Fusion) 多路检索结果融合

    公式: RRF_score(d) = sum(1 / (k + rank_i(d)))

    参数:
        result_lists: 多个检索结果列表, 每个是 [(doc_idx, score), ...]
        k: 常数(默认60), 防止排名靠前的结果权重过大

    返回:
        融合后的排序结果 [(doc_idx, rrf_score), ...]
    """
    rrf_scores = defaultdict(float)

    for result_list in result_lists:
        for rank, (doc_idx, _) in enumerate(result_list, start=1):
            rrf_scores[doc_idx] += 1.0 / (k + rank)

    # 按RRF分数排序
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results


# ============================================================
# 4. 混合检索 Pipeline
# ============================================================

class HybridRetriever:
    """混合检索器: BM25 + 向量检索 + RRF融合"""

    def __init__(self, bm25_weight=0.5, vector_weight=0.5, rrf_k=60):
        self.bm25 = SimpleBM25()
        self.vector = SimpleVectorRetriever()
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.rrf_k = rrf_k
        self.corpus = []

    def fit(self, corpus):
        """构建双路索引"""
        self.corpus = corpus
        self.bm25.fit(corpus)
        self.vector.fit(corpus)

    def query(self, query_text, top_k=5):
        """混合检索"""
        # 双路检索
        bm25_results = self.bm25.query(query_text, top_k=top_k * 2)
        vector_results = self.vector.query(query_text, top_k=top_k * 2)

        # RRF融合
        fused = reciprocal_rank_fusion(
            [bm25_results, vector_results],
            k=self.rrf_k
        )

        return fused[:top_k]

    def compare(self, query_text, top_k=5):
        """对比三种检索方式的结果"""
        bm25_results = self.bm25.query(query_text, top_k=top_k)
        vector_results = self.vector.query(query_text, top_k=top_k)
        hybrid_results = self.query(query_text, top_k=top_k)

        return {
            'bm25': bm25_results,
            'vector': vector_results,
            'hybrid': hybrid_results
        }


# ============================================================
# 5. 检索质量对比
# ============================================================

def compare_retrieval_methods(retriever, queries, relevance_dict):
    """对比不同检索方法的召回率

    参数:
        retriever: HybridRetriever实例
        queries: 查询列表
        relevance_dict: {query: [relevant_doc_indices]}
    """
    methods = ['bm25', 'vector', 'hybrid']
    recall_at_k = {m: [] for m in methods}

    for query in queries:
        comparison = retriever.compare(query, top_k=5)
        relevant = set(relevance_dict.get(query, []))

        if not relevant:
            continue

        for method in methods:
            retrieved = set(idx for idx, _ in comparison[method])
            # Recall@5 = |relevant ∩ retrieved| / |relevant|
            recall = len(relevant & retrieved) / len(relevant)
            recall_at_k[method].append(recall)

    # 计算平均召回率
    avg_recall = {}
    for method in methods:
        recalls = recall_at_k[method]
        avg_recall[method] = np.mean(recalls) if recalls else 0.0

    return avg_recall


# ============================================================
# 主程序: 演示混合检索
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("W30-D1 BM25+向量混合检索")
    print("=" * 60)

    # --- 示例文档库 ---
    corpus = [
        "Python是一种广泛使用的高级编程语言, 支持面向对象和函数式编程",
        "机器学习是人工智能的一个分支, 通过数据训练模型来进行预测",
        "深度学习使用神经网络来学习数据的复杂表示",
        "自然语言处理(NLP)研究如何让计算机理解和生成人类语言",
        "向量数据库用于存储和检索高维向量, 广泛应用于RAG系统",
        "RAG(检索增强生成)结合了信息检索和文本生成的优势",
        "Transformer架构是现代NLP模型的基础, 支持并行计算",
        "大语言模型(LLM)通过海量文本数据训练, 具有强大的生成能力",
        "Python的NumPy库提供了高效的多维数组运算支持",
        "语义搜索通过理解查询的含义来找到最相关的结果",
    ]

    # --- 构建混合检索器 ---
    retriever = HybridRetriever()
    retriever.fit(corpus)

    # --- 测试查询 ---
    queries = [
        "什么是RAG?",
        "Python编程",
        "深度学习和神经网络",
    ]

    for query in queries:
        print(f"\n{'='*50}")
        print(f"查询: {query}")
        print(f"{'='*50}")

        comparison = retriever.compare(query, top_k=3)

        for method, results in comparison.items():
            method_names = {'bm25': 'BM25关键词检索', 'vector': '向量语义检索', 'hybrid': '混合检索(RRF)'}
            print(f"\n[{method_names[method]}]")
            for rank, (doc_idx, score) in enumerate(results, 1):
                doc_preview = corpus[doc_idx][:40] + "..."
                print(f"  #{rank} (分数={score:.4f}): {doc_preview}")

    # --- 检索质量对比 ---
    print(f"\n{'='*60}")
    print("检索质量对比 (Recall@5)")
    print(f"{'='*60}")

    relevance_dict = {
        "什么是RAG?": [5, 4, 9],
        "Python编程": [0, 8],
        "深度学习和神经网络": [2, 6],
    }

    avg_recall = compare_retrieval_methods(retriever, queries, relevance_dict)
    for method, recall in avg_recall.items():
        method_names = {'bm25': 'BM25', 'vector': '向量', 'hybrid': '混合'}
        print(f"  {method_names[method]}: Recall@5 = {recall:.4f}")

    # --- 可视化 ---
    if HAS_PLT:
        fig, ax = plt.subplots(figsize=(8, 5))
        methods = ['BM25', '向量检索', '混合检索']
        recalls = [avg_recall['bm25'], avg_recall['vector'], avg_recall['hybrid']]
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        bars = ax.bar(methods, recalls, color=colors, edgecolor='white', linewidth=1.5)

        for bar, val in zip(bars, recalls):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.2%}', ha='center', fontsize=12, fontweight='bold')

        ax.set_ylabel('Recall@5', fontsize=12)
        ax.set_title('不同检索方式的召回率对比', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig('D:/code/big-model-learn/code/q_01/W30/d1_hybrid_retrieval.png', dpi=150)
        print("\n图表已保存为 d1_hybrid_retrieval.png")
        plt.close()

    print("\n完成!")
