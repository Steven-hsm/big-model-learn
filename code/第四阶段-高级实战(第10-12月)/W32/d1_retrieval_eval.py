"""
W32-D1 ��索评估
================
实现RAG系统检索质量的评估指标, 包括:
- Recall@K (召回率)
- Precision@K (精确率)
- MRR (Mean Reciprocal Rank)
- NDCG (Normalized Discounted Cumulative Gain)

评估是持续优化系统的基础。
"""

import math
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass

try:
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

import numpy as np


# ============================================================
# 1. 评估指标实现
# ============================================================

def recall_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """Recall@K: 召回率

    含义: 在前K个检索结果中, 找到了多少比例的相关文档。
    公式: |retrieved[:k] ∩ relevant| / |relevant|

    参数:
        retrieved: 检索返回的文档ID列表(按相关性排序)
        relevant: 实际相关的文档ID集合
        k: 截断位置
    """
    if not relevant:
        return 0.0
    retrieved_set = set(retrieved[:k])
    hits = len(retrieved_set & relevant)
    return hits / len(relevant)


def precision_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """Precision@K: 精确率

    含义: 在前K个检索结果中, 有多少比例是真正相关的。
    公式: |retrieved[:k] ∩ relevant| / k
    """
    if k == 0:
        return 0.0
    retrieved_set = set(retrieved[:k])
    hits = len(retrieved_set & relevant)
    return hits / k


def mean_reciprocal_rank(results: List[List[int]],
                          relevant_sets: List[Set[int]]) -> float:
    """MRR: 平均倒数排名

    含义: 第一个相关文档的排名倒数的平均值。
    公式: (1/Q) * sum(1/rank_i)
    其中 rank_i 是第i个查询的第一个相关文档的排名。
    """
    rr_sum = 0.0
    for retrieved, relevant in zip(results, relevant_sets):
        rr = 0.0
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                rr = 1.0 / rank
                break
        rr_sum += rr
    return rr_sum / len(results) if results else 0.0


def ndcg_at_k(retrieved: List[int], relevant: Set[int],
               relevance_scores: Dict[int, float] = None, k: int = None) -> float:
    """NDCG@K: 归一化折扣累积增益

    含义: 衡量排序质量, 相关文档排在前面会得到更高分数。
    公式: DCG@K / IDCG@K

    DCG@K = sum(rel_i / log2(i+1)) for i=1..K
    IDCG@K = 理想排序的DCG@K
    """
    if k is not None:
        retrieved = retrieved[:k]

    if relevance_scores is None:
        # 二值相关性: relevant=1, not_relevant=0
        relevance_scores = {doc_id: 1.0 for doc_id in relevant}

    # DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved, start=1):
        rel = relevance_scores.get(doc_id, 0.0)
        dcg += rel / math.log2(i + 1)

    # IDCG (理想排序)
    ideal_rels = sorted([relevance_scores.get(doc_id, 0.0) for doc_id in relevant],
                         reverse=True)
    if k is not None:
        ideal_rels = ideal_rels[:k]

    idcg = 0.0
    for i, rel in enumerate(ideal_rels, start=1):
        idcg += rel / math.log2(i + 1)

    return dcg / idcg if idcg > 0 else 0.0


def average_precision(retrieved: List[int], relevant: Set[int]) -> float:
    """AP: 平均精确率

    用于计算MAP(Mean Average Precision)
    """
    if not relevant:
        return 0.0

    precisions = []
    hits = 0
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            hits += 1
            precisions.append(hits / i)

    return sum(precisions) / len(relevant) if precisions else 0.0


def mean_average_precision(results: List[List[int]],
                            relevant_sets: List[Set[int]]) -> float:
    """MAP: 平均精确率的均值"""
    aps = [average_precision(r, rel) for r, rel in zip(results, relevant_sets)]
    return sum(aps) / len(aps) if aps else 0.0


# ============================================================
# 2. 评估数据集
# ============================================================

@dataclass
class EvalQuery:
    """评估查询"""
    query_id: str
    query_text: str
    relevant_docs: Set[int]               # 相关文档ID
    relevance_scores: Dict[int, float] = None  # 文档ID -> 相关度分数


@dataclass
class RetrievalResult:
    """检索结果"""
    query_id: str
    retrieved_docs: List[int]             # 检索返回的文档ID(有序)


class RetrievalEvaluator:
    """检索评估器"""

    def __init__(self):
        self.queries: List[EvalQuery] = []
        self.results: List[RetrievalResult] = []

    def add_eval_query(self, query_id: str, query_text: str,
                        relevant_docs: List[int],
                        relevance_scores: Dict[int, float] = None):
        self.queries.append(EvalQuery(
            query_id=query_id,
            query_text=query_text,
            relevant_docs=set(relevant_docs),
            relevance_scores=relevance_scores,
        ))

    def add_retrieval_result(self, query_id: str, retrieved_docs: List[int]):
        self.results.append(RetrievalResult(
            query_id=query_id,
            retrieved_docs=retrieved_docs,
        ))

    def evaluate(self, k_values: List[int] = None) -> Dict:
        """运行完整评估"""
        if k_values is None:
            k_values = [1, 3, 5, 10, 20]

        # 匹配查询和结果
        query_map = {q.query_id: q for q in self.queries}
        result_map = {r.query_id: r for r in self.results}

        metrics = {
            'recall': {k: [] for k in k_values},
            'precision': {k: [] for k in k_values},
            'ndcg': {k: [] for k in k_values},
            'mrr': [],
            'map': [],
        }

        for query_id in query_map:
            if query_id not in result_map:
                continue

            query = query_map[query_id]
            result = result_map[query_id]
            retrieved = result.retrieved_docs
            relevant = query.relevant_docs
            rel_scores = query.relevance_scores

            for k in k_values:
                metrics['recall'][k].append(recall_at_k(retrieved, relevant, k))
                metrics['precision'][k].append(precision_at_k(retrieved, relevant, k))
                metrics['ndcg'][k].append(ndcg_at_k(retrieved, relevant, rel_scores, k))

            metrics['mrr'].append(1.0 / next(
                (i+1 for i, d in enumerate(retrieved) if d in relevant), len(retrieved)+1))
            metrics['map'].append(average_precision(retrieved, relevant))

        # 计算平均值
        summary = {}
        for k in k_values:
            summary[f'Recall@{k}'] = np.mean(metrics['recall'][k]) if metrics['recall'][k] else 0
            summary[f'Precision@{k}'] = np.mean(metrics['precision'][k]) if metrics['precision'][k] else 0
            summary[f'NDCG@{k}'] = np.mean(metrics['ndcg'][k]) if metrics['ndcg'][k] else 0

        summary['MRR'] = np.mean(metrics['mrr']) if metrics['mrr'] else 0
        summary['MAP'] = np.mean(metrics['map']) if metrics['map'] else 0

        return summary

    def print_report(self, k_values: List[int] = None):
        """打印评估报告"""
        results = self.evaluate(k_values)
        print(f"\n{'='*60}")
        print("检索评估报告")
        print(f"{'='*60}")
        print(f"评估查询数: {len(self.queries)}")
        print()
        for metric, value in results.items():
            print(f"  {metric:15s}: {value:.4f}")


# ============================================================
# 3. 模拟检索与评估
# ============================================================

def simulate_retrieval_evaluation():
    """模拟检索评估"""
    # 构建评估数据集
    evaluator = RetrievalEvaluator()

    # 定义评估查询和标准答案
    eval_data = [
        ("q1", "什么是RAG?", [0, 4, 7], {0: 3.0, 4: 2.0, 7: 1.0}),
        ("q2", "Python编程", [3, 8], {3: 3.0, 8: 2.0}),
        ("q3", "深度学习", [2, 6, 9], {2: 3.0, 6: 2.0, 9: 1.0}),
        ("q4", "向量检索原理", [1, 4, 5], {1: 3.0, 4: 2.0, 5: 1.0}),
        ("q5", "如何部署AI系统", [3, 7, 9], {3: 2.0, 7: 3.0, 9: 1.0}),
    ]

    for qid, text, rel_docs, scores in eval_data:
        evaluator.add_eval_query(qid, text, rel_docs, scores)

    # 模拟不同方法的检索结果
    import random
    random.seed(42)

    methods = {
        'BM25': [
            [0, 4, 7, 2, 5],     # q1: 好的排序
            [3, 8, 1, 5, 6],     # q2
            [2, 6, 9, 0, 3],     # q3
            [1, 4, 5, 0, 7],     # q4
            [7, 3, 5, 9, 1],     # q5
        ],
        '向量检索': [
            [0, 4, 5, 7, 2],     # q1: 漏了一个
            [3, 1, 8, 5, 6],     # q2
            [2, 6, 0, 9, 3],     # q3: 排序不够好
            [1, 4, 0, 5, 7],     # q4
            [7, 5, 3, 1, 9],     # q5: 漏了一个
        ],
        '混合检索': [
            [0, 4, 7, 5, 2],     # q1: 最佳
            [3, 8, 1, 6, 5],     # q2: 最佳
            [2, 6, 9, 0, 3],     # q3: 最佳
            [1, 4, 5, 7, 0],     # q4: 最佳
            [7, 3, 9, 5, 1],     # q5: 最佳
        ],
    }

    all_results = {}
    for method_name, method_results in methods.items():
        evaluator.results = []
        for i, (qid, _, _, _) in enumerate(eval_data):
            evaluator.add_retrieval_result(qid, method_results[i])
        all_results[method_name] = evaluator.evaluate()

    return all_results


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("W32-D1 检索评估")
    print("=" * 60)

    # --- 1. 指标演示 ---
    print("\n--- 1. 单指标演示 ---")
    retrieved = [1, 3, 5, 7, 9, 2, 4, 6, 8, 10]
    relevant = {1, 5, 9}

    for k in [1, 3, 5, 10]:
        r = recall_at_k(retrieved, relevant, k)
        p = precision_at_k(retrieved, relevant, k)
        n = ndcg_at_k(retrieved, relevant, k=k)
        print(f"  K={k:2d}: Recall={r:.3f}, Precision={p:.3f}, NDCG={n:.3f}")

    # --- 2. 完整评估 ---
    print(f"\n{'='*60}")
    print("--- 2. 多方法对比评估 ---")
    print(f"{'='*60}")

    results = simulate_retrieval_evaluation()

    for method, metrics in results.items():
        print(f"\n[{method}]")
        for metric, value in metrics.items():
            print(f"  {metric:15s}: {value:.4f}")

    # --- 可视化 ---
    if HAS_PLT:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        methods = list(results.keys())
        k_values = [1, 3, 5, 10]

        # Recall@K
        ax = axes[0]
        for i, method in enumerate(methods):
            recalls = [results[method][f'Recall@{k}'] for k in k_values]
            ax.plot(k_values, recalls, 'o-', label=method, linewidth=2)
        ax.set_xlabel('K')
        ax.set_ylabel('Recall')
        ax.set_title('Recall@K对比', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Precision@K
        ax = axes[1]
        for method in methods:
            precisions = [results[method][f'Precision@{k}'] for k in k_values]
            ax.plot(k_values, precisions, 'o-', label=method, linewidth=2)
        ax.set_xlabel('K')
        ax.set_ylabel('Precision')
        ax.set_title('Precision@K对比', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 综合指标
        ax = axes[2]
        metrics_names = ['MRR', 'MAP', 'NDCG@5']
        x = range(len(metrics_names))
        width = 0.25
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        for i, (method, color) in enumerate(zip(methods, colors)):
            values = [results[method].get(m, 0) for m in metrics_names]
            ax.bar([pos + i * width for pos in x], values, width,
                    label=method, color=color)
        ax.set_xticks([pos + width for pos in x])
        ax.set_xticklabels(metrics_names)
        ax.set_ylabel('分数')
        ax.set_title('综合指标对比', fontweight='bold')
        ax.legend()

        plt.tight_layout()
        plt.savefig('D:/code/big-model-learn/code/q_01/W32/d1_retrieval_eval.png', dpi=150)
        print("\n图表已保存为 d1_retrieval_eval.png")
        plt.close()

    print("\n完成!")
