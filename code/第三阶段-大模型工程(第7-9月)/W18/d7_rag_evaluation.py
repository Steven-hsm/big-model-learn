"""
W18-D7 RAG评估 (RAG Evaluation)
=================================
RAG评估指标(命中率/MRR/NDCG/faithfulness), 检索质量评估,
生成质量评估, 不同chunk_size和top_k的影响, 评估报告
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W18-D7 RAG评估 (RAG Evaluation)")
print("=" * 60)

# ============================================================
# 1. RAG评估概述
# ============================================================
print("\n--- 1. RAG评估概述 ---")
print("""
  RAG系统需要从两个维度评估:
    1) 检索质量: 检索到的文档是否相关?
    2) 生成质量: 生成的答案是否准确、完整、忠实?

  RAGAS框架 (Retrieval Augmented Generation Assessment):
    - Faithfulness (忠实度): 答案是否基于检索到的上下文
    - Answer Relevance (答案相关性): 答案是否回应了问题
    - Context Precision (上下文精确率): 检索到的文档是否相关
    - Context Recall (上下文召回率): 需要的信息是否被检索到
""")

# ============================================================
# 2. 检索评估指标
# ============================================================
print("\n--- 2. 检索评估指标 ---")
print("""
  检索评估的核心指标:

  1) 命中率 (Hit Rate / Recall@K)
     = 至少有一个相关文档出现在Top-K中的查询比例

  2) MRR (Mean Reciprocal Rank)
     = 1/|Q| * Σ 1/rank_i
     第一个相关文档排名越靠前, MRR越高

  3) NDCG (Normalized Discounted Cumulative Gain)
     考虑文档排序位置, 相关文档排前面得分更高
     NDCG = DCG / IDCG
     DCG = Σ (2^rel_i - 1) / log2(rank_i + 1)

  4) Precision@K
     Top-K结果中相关文档的比例

  5) MAP (Mean Average Precision)
     所有查询的Average Precision的均值
""")


def hit_rate(relevance_lists: List[List[int]], k: int = 5) -> float:
    """
    命中率: 至少有一个相关文档在Top-K中
    relevance_lists: 每个查询的相关性列表 (1=相关, 0=不相关)
    """
    hits = 0
    for rels in relevance_lists:
        if sum(rels[:k]) > 0:
            hits += 1
    return hits / len(relevance_lists) if relevance_lists else 0


def mrr(relevance_lists: List[List[int]]) -> float:
    """Mean Reciprocal Rank"""
    rr_sum = 0
    for rels in relevance_lists:
        for i, rel in enumerate(rels):
            if rel == 1:
                rr_sum += 1 / (i + 1)
                break
    return rr_sum / len(relevance_lists) if relevance_lists else 0


def ndcg(relevance_lists: List[List[int]], k: int = None) -> float:
    """Normalized Discounted Cumulative Gain"""
    ndcg_sum = 0
    for rels in relevance_lists:
        if k is not None:
            rels = rels[:k]
        # DCG
        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(rels))
        # IDCG (理想排序)
        ideal = sorted(rels, reverse=True)
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal))
        ndcg_sum += dcg / idcg if idcg > 0 else 0
    return ndcg_sum / len(relevance_lists) if relevance_lists else 0


def precision_at_k(relevance_lists: List[List[int]], k: int = 5) -> float:
    """Precision@K"""
    p_sum = 0
    for rels in relevance_lists:
        p_sum += sum(rels[:k]) / k
    return p_sum / len(relevance_lists) if relevance_lists else 0


def average_precision(relevance_list: List[int]) -> float:
    """Average Precision for a single query"""
    relevant = 0
    ap = 0
    for i, rel in enumerate(relevance_list):
        if rel == 1:
            relevant += 1
            ap += relevant / (i + 1)
    return ap / relevant if relevant > 0 else 0


def mean_average_precision(relevance_lists: List[List[int]]) -> float:
    """MAP"""
    return np.mean([average_precision(rels) for rels in relevance_lists])


# ============================================================
# 3. 生成质量评估
# ============================================================
print("\n--- 3. 生成质量评估 ---")
print("""
  生成质量的评估维度:

  1) Faithfulness (忠实度)
     答案中的每个陈述是否能从检索到的上下文中找到支持
     = 可支持的陈述数 / 总陈述数

  2) Answer Relevance (答案相关性)
     答案是否回应了原始问题
     可通过: 人工评分 / LLM-as-judge / 自动指标

  3) Completeness (完整性)
     答案是否涵盖了问题的所有方面

  4) Conciseness (简洁性)
     答案是否简洁, 没有不必要的信息
""")


def faithfulness_score(answer_claims: List[str], context: str) -> float:
    """
    模拟忠实度评估
    answer_claims: 答案中的声明列表
    context: 检索到的上下文
    """
    supported = 0
    for claim in answer_claims:
        # 简单的关键词匹配 (真实场景用LLM判断)
        claim_words = set(claim.lower().split())
        context_words = set(context.lower().split())
        overlap = claim_words & context_words
        if len(overlap) >= len(claim_words) * 0.3:
            supported += 1
    return supported / len(answer_claims) if answer_claims else 0


# ============================================================
# 4. 模拟RAG评估实验
# ============================================================
print("\n--- 4. 模拟RAG评估实验 ---")

# 模拟评估数据
np.random.seed(42)

# 10个查询, 每个查询检索10个文档
n_queries = 10
n_docs = 10

# 生成模拟检索结果 (1=相关, 0=不相关)
ground_truth = {
    "什么是机器学习":     [1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    "深度学习有哪些应用":  [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    "Transformer是什么":  [1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    "RAG的工作原理":      [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    "向量数据库有哪些":   [1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    "Python ML库":       [1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    "自然语言处理技术":   [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    "神经网络训练方法":   [1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    "推荐系统算法":       [1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    "计算机视觉模型":     [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
}

queries = list(ground_truth.keys())
relevance_lists = list(ground_truth.values())

# 计算所有指标
print("\n  --- 检索质量指标 ---")
for k in [1, 3, 5, 10]:
    hr = hit_rate(relevance_lists, k=k)
    ndcg_val = ndcg(relevance_lists, k=k)
    prec = precision_at_k(relevance_lists, k=k)
    print(f"  K={k:>2d}: Hit Rate={hr:.3f}, NDCG@{k}={ndcg_val:.3f}, Precision@{k}={prec:.3f}")

mrr_val = mrr(relevance_lists)
map_val = mean_average_precision(relevance_lists)
print(f"\n  MRR = {mrr_val:.3f}")
print(f"  MAP = {map_val:.3f}")

# 逐查询详情
print("\n  --- 逐查询详情 ---")
print(f"  {'查询':<20s} {'HR@3':>6s} {'HR@5':>6s} {'NDCG@5':>8s} {'AP':>6s}")
print(f"  {'-'*20} {'-'*6} {'-'*6} {'-'*8} {'-'*6}")
for query, rels in zip(queries, relevance_lists):
    hr3 = 1 if sum(rels[:3]) > 0 else 0
    hr5 = 1 if sum(rels[:5]) > 0 else 0
    ndcg5 = ndcg([rels], k=5)
    ap = average_precision(rels)
    print(f"  {query:<20s} {hr3:>6d} {hr5:>6d} {ndcg5:>8.3f} {ap:>6.3f}")

# ============================================================
# 5. 不同参数影响实验
# ============================================================
print("\n--- 5. 不同参数影响实验 ---")

# 模拟不同chunk_size对检索质量的影响
chunk_sizes = [50, 100, 200, 300, 500, 800, 1000]

# 模拟数据: chunk_size对检索指标的影响
np.random.seed(42)
# 太小: 上下文不足; 太大: 噪声多; 中间最合适
hit_by_chunk = [0.60, 0.75, 0.90, 0.92, 0.88, 0.82, 0.75]
ndcg_by_chunk = [0.55, 0.70, 0.85, 0.88, 0.83, 0.78, 0.70]
faithfulness_by_chunk = [0.50, 0.65, 0.80, 0.85, 0.82, 0.78, 0.72]

# 模拟不同top_k对检索质量的影响
top_k_values = [1, 2, 3, 5, 7, 10, 15, 20]
recall_by_topk = [0.30, 0.50, 0.65, 0.80, 0.88, 0.93, 0.96, 0.98]
precision_by_topk = [0.90, 0.80, 0.70, 0.55, 0.45, 0.35, 0.28, 0.22]

print(f"\n  chunk_size 影响:")
print(f"  {'chunk_size':>10s} {'Hit Rate':>10s} {'NDCG':>8s} {'忠实度':>8s}")
print(f"  {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
for cs, hr, ndcg_val, fa in zip(chunk_sizes, hit_by_chunk, ndcg_by_chunk, faithfulness_by_chunk):
    print(f"  {cs:>10d} {hr:>10.2f} {ndcg_val:>8.2f} {fa:>8.2f}")

print(f"\n  top_k 影响:")
print(f"  {'top_k':>6s} {'Recall':>8s} {'Precision':>10s}")
print(f"  {'-'*6} {'-'*8} {'-'*10}")
for tk, rec, prec in zip(top_k_values, recall_by_topk, precision_by_topk):
    print(f"  {tk:>6d} {rec:>8.2f} {prec:>10.2f}")

# ============================================================
# 6. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 子图1: chunk_size影响
ax1 = axes[0, 0]
ax1.plot(chunk_sizes, hit_by_chunk, 'o-', label='Hit Rate', linewidth=2, color='#e74c3c')
ax1.plot(chunk_sizes, ndcg_by_chunk, 's-', label='NDCG', linewidth=2, color='#3498db')
ax1.plot(chunk_sizes, faithfulness_by_chunk, '^-', label='忠实度', linewidth=2, color='#2ecc71')
ax1.axvline(x=300, color='gray', linestyle='--', alpha=0.5, label='最优区域')
ax1.set_xlabel('Chunk Size (字符数)', fontsize=12)
ax1.set_ylabel('分数', fontsize=12)
ax1.set_title('Chunk Size 对检索质量的影响', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 子图2: top_k影响
ax2 = axes[0, 1]
ax2.plot(top_k_values, recall_by_topk, 'o-', label='Recall', linewidth=2, color='#e74c3c')
ax2.plot(top_k_values, precision_by_topk, 's-', label='Precision', linewidth=2, color='#3498db')
ax2.fill_between(top_k_values, recall_by_topk, precision_by_topk, alpha=0.1, color='gray')
ax2.axvline(x=5, color='gray', linestyle='--', alpha=0.5, label='推荐值')
ax2.set_xlabel('Top-K', fontsize=12)
ax2.set_ylabel('分数', fontsize=12)
ax2.set_title('Top-K 对 Precision/Recall 的影响', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# 子图3: 逐查询HR@5
ax3 = axes[1, 0]
query_short = [q[:8] for q in queries]
hr5_per_query = [1 if sum(rels[:5]) > 0 else 0 for rels in relevance_lists]
ndcg5_per_query = [ndcg([rels], k=5) for rels in relevance_lists]
ap_per_query = [average_precision(rels) for rels in relevance_lists]

x = np.arange(len(queries))
width = 0.25
ax3.bar(x - width, hr5_per_query, width, label='HR@5', color='#e74c3c', alpha=0.85, edgecolor='black')
ax3.bar(x, [n * 0.8 for n in ndcg5_per_query], width, label='NDCG@5', color='#3498db', alpha=0.85, edgecolor='black')
ax3.bar(x + width, ap_per_query, width, label='AP', color='#2ecc71', alpha=0.85, edgecolor='black')

ax3.set_xticks(x)
ax3.set_xticklabels(query_short, fontsize=9, rotation=30)
ax3.set_ylabel('分数', fontsize=12)
ax3.set_title('逐查询检索质量', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')

# 子图4: 评估指标总览雷达图
ax4 = axes[1, 1]
categories = ['Hit Rate\n@5', 'NDCG\n@5', 'Precision\n@5', 'MRR', 'MAP', 'Faithfulness']
values = [
    hit_rate(relevance_lists, k=5),
    ndcg(relevance_lists, k=5),
    precision_at_k(relevance_lists, k=5),
    mrr_val,
    map_val,
    0.82,  # 模拟忠实度
]
values.append(values[0])  # 闭合

angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles.append(angles[0])

ax4 = fig.add_subplot(2, 2, 4, polar=True)
ax4.fill(angles, values, alpha=0.25, color='#3498db')
ax4.plot(angles, values, 'o-', color='#3498db', linewidth=2)
ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(categories, fontsize=10)
ax4.set_ylim(0, 1)
ax4.set_title('RAG系统评估总览', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W18/rag_evaluation.png', dpi=150, bbox_inches='tight')
print("\n  图表已保存: rag_evaluation.png")
plt.close()

# ============================================================
# 7. 评估报告
# ============================================================
print("\n--- 7. 评估报告 ---")

report = f"""
  ================================================
  RAG系统评估报告
  ================================================

  评估数据集: {len(queries)} 个查询
  每个查询检索文档数: {n_docs}

  --- 检索质量 ---
  Hit Rate @3:  {hit_rate(relevance_lists, k=3):.3f}
  Hit Rate @5:  {hit_rate(relevance_lists, k=5):.3f}
  Hit Rate @10: {hit_rate(relevance_lists, k=10):.3f}
  NDCG @5:      {ndcg(relevance_lists, k=5):.3f}
  NDCG @10:     {ndcg(relevance_lists, k=10):.3f}
  Precision @5: {precision_at_k(relevance_lists, k=5):.3f}
  MRR:          {mrr_val:.3f}
  MAP:          {map_val:.3f}

  --- 生成质量 (模拟) ---
  Faithfulness:     0.82
  Answer Relevance: 0.88
  Completeness:     0.78

  --- 建议 ---
  1. 最优chunk_size: 200-400字符
  2. 推荐top_k: 3-5
  3. 可考虑添加重排序提升NDCG
  4. 部分查询(如'推荐系统')检索质量偏低, 需优化嵌入模型
  ================================================
"""
print(report)

print("=" * 60)
print("W18-D7 完成! 本节实现了RAG系统的完整评估框架")
print("=" * 60)
