### Day 1（周一）���LLM评估指标概览
# 困惑度(Perplexity)、BLEU、ROUGE、准确率指标的实现与可视化

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 困惑度 (Perplexity) 计算
# ============================================================
def compute_perplexity(log_probs):
    """计算困惑度: PPL = exp(-1/N * sum(log P(w_i)))"""
    avg_neg_log_prob = -np.mean(log_probs)
    perplexity = np.exp(avg_neg_log_prob)
    return perplexity


# 模拟不同模型在每个token上的对数概率
np.random.seed(42)
model_a_log_probs = np.random.normal(-2.5, 0.5, 100)  # 较好的模型
model_b_log_probs = np.random.normal(-3.5, 0.8, 100)  # 较差的模型
model_c_log_probs = np.random.normal(-2.0, 0.3, 100)  # 最好的模型

ppl_a = compute_perplexity(model_a_log_probs)
ppl_b = compute_perplexity(model_b_log_probs)
ppl_c = compute_perplexity(model_c_log_probs)

print("=" * 60)
print("1. 困惑度 (Perplexity) 评估")
print("=" * 60)
print(f"  模型A (一般): PPL = {ppl_a:.2f}")
print(f"  模型B (较差): PPL = {ppl_b:.2f}")
print(f"  模型C (最好): PPL = {ppl_c:.2f}")
print(f"  → 困惑度越低, 模型对数据的拟合越好\n")


# ============================================================
# 2. BLEU 分数计算
# ============================================================
def compute_bleu(reference, candidate, max_n=4):
    """计算BLEU分数 (简化版, 无brevity penalty的精确计算)"""
    from collections import Counter

    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()

    # Brevity Penalty
    bp = 1.0
    if len(cand_tokens) < len(ref_tokens):
        bp = np.exp(1 - len(ref_tokens) / max(len(cand_tokens), 1))

    # N-gram precision
    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(
            tuple(ref_tokens[i:i + n]) for i in range(len(ref_tokens) - n + 1)
        )
        cand_ngrams = Counter(
            tuple(cand_tokens[i:i + n]) for i in range(len(cand_tokens) - n + 1)
        )

        clipped = sum(
            min(count, ref_ngrams[ngram])
            for ngram, count in cand_ngrams.items()
        )
        total = sum(cand_ngrams.values())

        precision = clipped / total if total > 0 else 0
        precisions.append(precision)

    # 几何平均
    if min(precisions) == 0:
        return 0.0

    log_avg = np.mean([np.log(p) for p in precisions])
    bleu = bp * np.exp(log_avg)
    return bleu


print("=" * 60)
print("2. BLEU 分数计算")
print("=" * 60)

references = "The cat is on the mat"
candidates = [
    "The cat is on the mat",       # 完全匹配
    "The cat is sitting on mat",    # 部分匹配
    "A dog plays in the garden",    # 完全不同
]

for cand in candidates:
    score = compute_bleu(references, cand)
    print(f"  Reference: '{references}'")
    print(f"  Candidate: '{cand}'")
    print(f"  BLEU-4:    {score:.4f}\n")


# ============================================================
# 3. ROUGE 分数计算
# ============================================================
def compute_rouge(reference, candidate):
    """计算ROUGE-1, ROUGE-2, ROUGE-L"""
    ref_tokens = set(reference.lower().split())
    cand_tokens = candidate.lower().split()

    # ROUGE-1 (unigram)
    overlap_1 = sum(1 for t in cand_tokens if t in ref_tokens)
    rouge_1_precision = overlap_1 / len(cand_tokens) if cand_tokens else 0
    rouge_1_recall = overlap_1 / len(ref_tokens) if ref_tokens else 0
    rouge_1_f1 = (
        2 * rouge_1_precision * rouge_1_recall
        / (rouge_1_precision + rouge_1_recall)
        if (rouge_1_precision + rouge_1_recall) > 0
        else 0
    )

    # ROUGE-2 (bigram)
    ref_bigrams = set(
        tuple(reference.lower().split()[i:i + 2])
        for i in range(len(reference.split()) - 1)
    )
    cand_bigrams = [
        tuple(candidate.lower().split()[i:i + 2])
        for i in range(len(candidate.split()) - 1)
    ]
    overlap_2 = sum(1 for b in cand_bigrams if b in ref_bigrams)
    rouge_2_precision = overlap_2 / len(cand_bigrams) if cand_bigrams else 0
    rouge_2_recall = overlap_2 / len(ref_bigrams) if ref_bigrams else 0
    rouge_2_f1 = (
        2 * rouge_2_precision * rouge_2_recall
        / (rouge_2_precision + rouge_2_recall)
        if (rouge_2_precision + rouge_2_recall) > 0
        else 0
    )

    # ROUGE-L (最长公共子序列)
    ref_words = reference.lower().split()
    cand_words = candidate.lower().split()
    m, n = len(ref_words), len(cand_words)

    # LCS动态规划
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i - 1] == cand_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]
    rouge_l_precision = lcs_len / n if n > 0 else 0
    rouge_l_recall = lcs_len / m if m > 0 else 0
    rouge_l_f1 = (
        2 * rouge_l_precision * rouge_l_recall
        / (rouge_l_precision + rouge_l_recall)
        if (rouge_l_precision + rouge_l_recall) > 0
        else 0
    )

    return {
        'rouge-1': {'precision': rouge_1_precision, 'recall': rouge_1_recall, 'f1': rouge_1_f1},
        'rouge-2': {'precision': rouge_2_precision, 'recall': rouge_2_recall, 'f1': rouge_2_f1},
        'rouge-L': {'precision': rouge_l_precision, 'recall': rouge_l_recall, 'f1': rouge_l_f1},
    }


print("=" * 60)
print("3. ROUGE 分数计算")
print("=" * 60)

ref_text = "机器学习是人工智能的一个分支, 它使计算机能够从数据中学习"
cand_text = "机器学习是AI的重要领域, 让计算机可以从数据学习模式"

rouge_scores = compute_rouge(ref_text, cand_text)
for metric, scores in rouge_scores.items():
    print(f"  {metric}:")
    print(f"    Precision: {scores['precision']:.4f}")
    print(f"    Recall:    {scores['recall']:.4f}")
    print(f"    F1:        {scores['f1']:.4f}")
print()


# ============================================================
# 4. 准确率类指标
# ============================================================
def compute_accuracy_metrics(predictions, labels):
    """计算分类准确率指标: Accuracy, Precision, Recall, F1"""
    tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)
    tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)

    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion': {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn},
    }


print("=" * 60)
print("4. 分类准确率指标")
print("=" * 60)

# 模拟模型对问题的正确/错误预测
np.random.seed(42)
y_true = np.random.randint(0, 2, 200)
y_pred = (y_true * 0.8 + np.random.random(200) * 0.4).astype(int)

metrics = compute_accuracy_metrics(y_pred.tolist(), y_true.tolist())
print(f"  Accuracy:  {metrics['accuracy']:.4f}")
print(f"  Precision: {metrics['precision']:.4f}")
print(f"  Recall:    {metrics['recall']:.4f}")
print(f"  F1 Score:  {metrics['f1']:.4f}")
print(f"  混淆矩阵: TP={metrics['confusion']['TP']}, "
      f"FP={metrics['confusion']['FP']}, "
      f"FN={metrics['confusion']['FN']}, "
      f"TN={metrics['confusion']['TN']}\n")


# ============================================================
# 5. 可视化: 评估指标对比
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 5.1 困惑度对比
ax1 = axes[0, 0]
models = ['模型A\n(一般)', '模型B\n(较差)', '模型C\n(最好)']
ppls = [ppl_a, ppl_b, ppl_c]
colors = ['#FFA500', '#FF4444', '#4CAF50']
bars = ax1.bar(models, ppls, color=colors, edgecolor='black')
ax1.set_ylabel('Perplexity')
ax1.set_title('模型困惑度对比 (越低越好)')
for bar, val in zip(bars, ppls):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f'{val:.1f}', ha='center', fontsize=11)

# 5.2 BLEU分数随n-gram变化
ax2 = axes[0, 1]
ref = "the cat is on the mat and the dog is in the garden"
cand = "the cat is sitting on the mat and a dog plays in garden"
n_values = [1, 2, 3, 4]
bleu_ns = [compute_bleu(ref, cand, max_n=n) for n in n_values]
ax2.plot(n_values, bleu_ns, 'bo-', linewidth=2, markersize=10)
ax2.set_xlabel('BLEU-N')
ax2.set_ylabel('Score')
ax2.set_title('BLEU分数随N-gram变化')
ax2.set_xticks(n_values)
ax2.set_ylim(0, 1.05)
for i, val in enumerate(bleu_ns):
    ax2.annotate(f'{val:.3f}', (n_values[i], val),
                 textcoords="offset points", xytext=(0, 10), ha='center')

# 5.3 ROUGE指标雷达图 (用柱状图代替)
ax3 = axes[1, 0]
metric_names = ['R1-P', 'R1-R', 'R1-F1', 'R2-P', 'R2-R', 'R2-F1', 'RL-P', 'RL-R', 'RL-F1']
metric_values = []
for key in ['rouge-1', 'rouge-2', 'rouge-L']:
    for sub in ['precision', 'recall', 'f1']:
        metric_values.append(rouge_scores[key][sub])
bar_colors = ['#2196F3'] * 3 + ['#FF9800'] * 3 + ['#9C27B0'] * 3
ax3.bar(range(len(metric_names)), metric_values, color=bar_colors, edgecolor='black')
ax3.set_xticks(range(len(metric_names)))
ax3.set_xticklabels(metric_names, fontsize=8)
ax3.set_ylabel('Score')
ax3.set_title('ROUGE各指标分数')
ax3.set_ylim(0, 1.05)

# 5.4 准确率指标可视化
ax4 = axes[1, 1]
acc_names = ['Accuracy', 'Precision', 'Recall', 'F1']
acc_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']]
bars = ax4.bar(acc_names, acc_values, color=['#4CAF50', '#2196F3', '#FF9800', '#E91E63'],
               edgecolor='black')
ax4.set_ylim(0, 1.1)
ax4.set_ylabel('Score')
ax4.set_title('分类评估指标')
for bar, val in zip(bars, acc_values):
    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
             f'{val:.3f}', ha='center', fontsize=11)

plt.suptitle('W21-D1: LLM评估指标概览', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W21/d1_eval_metrics.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存!")
