"""
W30-D2 重排序(Reranker)
========================
实现检索结果的二次排序, 包括:
- Cross-encoder概念解释
- 简化重排序实现
- 完整的检索+重排序Pipeline

核心思想:
检索器(Bi-encoder)先快速召回候选, 再用重排序器(Cross-encoder)精确排序。
"""

import numpy as np
from collections import defaultdict
import time

try:
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    HAS_PLT = True
except ImportError:
    HAS_PLT = False


# ============================================================
# 1. Cross-encoder 概念
# ============================================================

def explain_cross_encoder():
    """解释Cross-encoder vs Bi-encoder的区别"""
    print("=" * 60)
    print("Cross-encoder vs Bi-encoder")
    print("=" * 60)
    print("""
Bi-encoder (双塔模型, 用于检索阶段):
  Query --> Encoder --> 向量Q --\
                                 --> 余弦相似度 --> 快速排序
  Doc   --> Encoder --> 向量D --/

  优点: 可以预计算文档向量, 检索速度快
  缺点: Query和Doc没有交互, 精度有限

Cross-encoder (交叉编码器, 用于重排序阶段):
  [Query + Doc] --> Encoder --> 相关性分数

  优点: Query和Doc充分交互, 精度高
  缺点: 不能预计算, 每对(Q,D)都要过模型, 速度慢

典型流程:
  粗检索(Bi-encoder, top-100) --> 精排序(Cross-encoder, top-10)
""")


# ============================================================
# 2. 简化检索器(从d1复用)
# ============================================================

class SimpleBM25:
    """简化BM25检索器"""

    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.doc_freqs = []
        self.df = defaultdict(int)
        self.doc_len = []
        self.avgdl = 0
        self.N = 0

    def _tokenize(self, text):
        import re
        return re.findall(r'\w+', text.lower())

    def fit(self, corpus):
        self.corpus = corpus
        self.N = len(corpus)
        for doc in corpus:
            tokens = self._tokenize(doc)
            self.doc_len.append(len(tokens))
            tf = defaultdict(int)
            for token in tokens:
                tf[token] += 1
            self.doc_freqs.append(tf)
            for token in set(tokens):
                self.df[token] += 1
        self.avgdl = np.mean(self.doc_len) if self.doc_len else 1

    def _idf(self, token):
        df = self.df.get(token, 0)
        return np.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def query(self, query_text, top_k=10):
        query_tokens = self._tokenize(query_text)
        scores = []
        for idx, doc_tf in enumerate(self.doc_freqs):
            score = 0.0
            for token in query_tokens:
                if token not in doc_tf:
                    continue
                tf = doc_tf[token]
                idf = self._idf(token)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * self.doc_len[idx] / self.avgdl)
                score += idf * numerator / denominator
            scores.append((idx, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ============================================================
# 3. 简化重排序器
# ============================================================

class SimpleReranker:
    """简化重排序器 - 模拟Cross-encoder的行为

    在实际项目中, 应使用:
    - sentence-transformers 的 cross-encoder
    - Cohere Rerank API
    - BGE-Reranker模型

    这里使用基于特征工程的简化实现来演示概念。
    """

    def __init__(self):
        self.weights = {
            'exact_match': 3.0,      # 完全匹配加分
            'partial_match': 1.0,     # 部分匹配加分
            'query_coverage': 2.0,    # 查询覆盖率
            'doc_coverage': 1.5,      # 文档覆盖率
            'position_bonus': 1.0,    # 位置加分(关键词出现在文档开头)
            'length_penalty': 0.5,    # 长度惩罚(过长文档)
        }

    def _tokenize(self, text):
        import re
        return re.findall(r'\w+', text.lower())

    def _compute_features(self, query, document):
        """计算query-document对的特征"""
        q_tokens = set(self._tokenize(query))
        d_tokens = self._tokenize(document)
        d_token_set = set(d_tokens)

        # 1. 完全匹配
        exact_matches = q_tokens & d_token_set
        exact_match_score = len(exact_matches) / max(len(q_tokens), 1)

        # 2. 部分匹配(子串匹配)
        partial_matches = 0
        for qt in q_tokens:
            for dt in d_token_set:
                if qt in dt or dt in qt:
                    partial_matches += 1
        partial_match_score = partial_matches / max(len(q_tokens), 1)

        # 3. 查询覆盖率
        query_coverage = len(exact_matches) / max(len(q_tokens), 1)

        # 4. 文档覆盖率
        doc_coverage = len(exact_matches) / max(len(d_token_set), 1)

        # 5. 位置加分(匹配词出现在文档前部)
        position_scores = []
        for qt in exact_matches:
            if qt in d_tokens:
                pos = d_tokens.index(qt) / max(len(d_tokens), 1)
                position_scores.append(1.0 - pos)
        position_bonus = np.mean(position_scores) if position_scores else 0

        # 6. 长度惩罚
        length_penalty = -len(d_tokens) / 100.0

        return {
            'exact_match': exact_match_score,
            'partial_match': partial_match_score,
            'query_coverage': query_coverage,
            'doc_coverage': doc_coverage,
            'position_bonus': position_bonus,
            'length_penalty': length_penalty,
        }

    def rerank(self, query, documents, top_k=None):
        """对候选文档进行重排序

        参数:
            query: 查询文本
            documents: 候选文档列表 [(index, text), ...]
            top_k: 返回前K个结果

        返回:
            重排序后的结果 [(index, score), ...]
        """
        results = []
        for idx, doc_text in documents:
            features = self._compute_features(query, doc_text)
            # 加权求和得到最终分数
            score = sum(
                self.weights[feat] * value
                for feat, value in features.items()
            )
            results.append((idx, score, features))

        # 按分数排序
        results.sort(key=lambda x: x[1], reverse=True)

        if top_k:
            results = results[:top_k]

        return results


# ============================================================
# 4. 检索+重排序 Pipeline
# ============================================================

class RetrievalRerankPipeline:
    """完整的检索+重排序Pipeline

    流程:
    1. 使用BM25进行粗检索(top_n)
    2. 使用Reranker进行精排序(top_k)
    3. 返回最终结果
    """

    def __init__(self, retriever_top_n=20, reranker_top_k=5):
        self.retriever = SimpleBM25()
        self.reranker = SimpleReranker()
        self.retriever_top_n = retriever_top_n
        self.reranker_top_k = reranker_top_k
        self.corpus = []

    def fit(self, corpus):
        """构建索引"""
        self.corpus = corpus
        self.retriever.fit(corpus)

    def query(self, query_text, top_k=None):
        """执行检索+重排序"""
        if top_k is None:
            top_k = self.reranker_top_k

        # 第一阶段: 粗检索
        start_time = time.time()
        retrieval_results = self.retriever.query(query_text, top_k=self.retriever_top_n)
        retrieval_time = time.time() - start_time

        # 第二阶段: 精排序(重排序)
        start_time = time.time()
        candidates = [(idx, self.corpus[idx]) for idx, _ in retrieval_results]
        reranked = self.reranker.rerank(query_text, candidates, top_k=top_k)
        rerank_time = time.time() - start_time

        return {
            'results': [(idx, score) for idx, score, _ in reranked],
            'features': [(idx, features) for idx, _, features in reranked],
            'retrieval_time': retrieval_time,
            'rerank_time': rerank_time,
            'total_candidates': len(retrieval_results),
        }

    def compare_with_without_rerank(self, query_text, top_k=5):
        """对比有/无重排序的结果差异"""
        # 无重排序(纯BM25)
        bm25_results = self.retriever.query(query_text, top_k=top_k)

        # 有重排序
        pipeline_results = self.query(query_text, top_k=top_k)

        return {
            'bm25_only': bm25_results,
            'with_rerank': pipeline_results['results'],
            'features': pipeline_results['features'],
            'time': {
                'retrieval': pipeline_results['retrieval_time'],
                'rerank': pipeline_results['rerank_time'],
            }
        }


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    explain_cross_encoder()

    # --- 示例文档 ---
    corpus = [
        "Python是一种广泛使用的高级编程语言, 由Guido van Rossum创建",
        "机器学习使用统计方法让计算机从数据中学习, 无需显式编程",
        "深度学习是机器学习的子集, 使用多层神经网络处理复杂模式",
        "自然语言处理让计算机能够理解、解释和生成人类语言",
        "RAG系统通过检索相关文档来增强大语言模型的生成质量",
        "向量数据库如Milvus和Pinecone专为高维向量的存储和检索设计",
        "Transformer架构引入了自注意力机制, 成为NLP领域的里程碑",
        "大语言模型通过在海量文本上训练, 获得了广泛的知识和推理能力",
        "Python的pandas库是数据分析和处理的强大工具",
        "信息检索系统的核心目标是快速准确地找到用户需要的信息",
        "语义搜索使用向量嵌入来理解查询和文档的含义, 而非仅仅匹配关键词",
        "重排序技术可以对初步检索结果进行精细化排序, 提高最终结果质量",
    ]

    # --- 构建Pipeline ---
    pipeline = RetrievalRerankPipeline(retriever_top_n=10, reranker_top_k=5)
    pipeline.fit(corpus)

    # --- 测试查询 ---
    queries = ["什么是RAG检索增强生成?", "Python编程语言的特点"]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"查询: {query}")
        print(f"{'='*60}")

        comparison = pipeline.compare_with_without_rerank(query, top_k=5)

        print(f"\n[BM25原始排序]")
        for rank, (idx, score) in enumerate(comparison['bm25_only'], 1):
            print(f"  #{rank} (分数={score:.4f}): {corpus[idx][:50]}...")

        print(f"\n[重排序后]")
        for rank, (idx, score) in enumerate(comparison['with_rerank'], 1):
            print(f"  #{rank} (分数={score:.4f}): {corpus[idx][:50]}...")

        print(f"\n耗时: 检索={comparison['time']['retrieval']*1000:.1f}ms, "
              f"重排序={comparison['time']['rerank']*1000:.1f}ms")

        # 显示重排序特征
        print(f"\n[重排序特征详情]")
        for idx, features in comparison['features'][:3]:
            print(f"  文档{idx}: {corpus[idx][:30]}...")
            for feat, val in features.items():
                print(f"    {feat}: {val:.4f}")

    # --- 可视化排序变化 ---
    if HAS_PLT:
        query = queries[0]
        comparison = pipeline.compare_with_without_rerank(query, top_k=5)

        bm25_order = [idx for idx, _ in comparison['bm25_only']]
        rerank_order = [idx for idx, _ in comparison['with_rerank']]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # BM25排序
        bm25_scores = [score for _, score in comparison['bm25_only']]
        labels = [f'Doc{idx}' for idx in bm25_order]
        ax1.barh(range(len(bm25_order)), bm25_scores, color='#3498db')
        ax1.set_yticks(range(len(bm25_order)))
        ax1.set_yticklabels(labels)
        ax1.set_title('BM25原始排序', fontsize=13, fontweight='bold')
        ax1.set_xlabel('BM25分数')
        ax1.invert_yaxis()

        # 重排序后
        rerank_scores = [score for _, score in comparison['with_rerank']]
        labels = [f'Doc{idx}' for idx in rerank_order]
        ax2.barh(range(len(rerank_order)), rerank_scores, color='#e74c3c')
        ax2.set_yticks(range(len(rerank_order)))
        ax2.set_yticklabels(labels)
        ax2.set_title('重排序后', fontsize=13, fontweight='bold')
        ax2.set_xlabel('重排序分数')
        ax2.invert_yaxis()

        plt.suptitle(f'查询: "{query}"', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('D:/code/big-model-learn/code/q_01/W30/d2_reranker.png', dpi=150)
        print("\n图表已保存为 d2_reranker.png")
        plt.close()

    print("\n完成!")
