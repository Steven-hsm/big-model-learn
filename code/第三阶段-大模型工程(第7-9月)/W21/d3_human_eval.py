### Day 3（周三）：人类评估方法
# A/B测试设计, 评分标准, RLHF概念, 数据标注指南

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 人类评估方法概述
# ============================================================
print("=" * 60)
print("1. 人类评估方法概述")
print("=" * 60)

eval_methods = {
    '单点评分': {
        '描述': '评估者对单个输出打分 (1-5分)',
        '优点': '简单快速, 适合大规模评估',
        '缺点': '主观性强, 评分标准不一致',
    },
    '比较评估': {
        '描述': '评估者比较两个输出哪个更好',
        '优点': '更准确, 减少主观偏差',
        '缺点': '耗时, 不可扩展',
    },
    '排序评估': {
        '描述': '评估者对多个输出进行排序',
        '优点': '提供更细粒度的比较',
        '缺点': '评估者负担大',
    },
    'Elo评分': {
        '描述': '基于对战的动态评分系统 (如Chatbot Arena)',
        '优点': '自动校准, 持续更新',
        '缺点': '需要大量对战数据',
    },
}

for method, info in eval_methods.items():
    print(f"\n  {method}:")
    print(f"    描述: {info['描述']}")
    print(f"    优点: {info['优点']}")
    print(f"    缺点: {info['缺点']}")


# ============================================================
# 2. A/B测试设计与实现
# ============================================================
class ABTestFramework:
    """A/B测试框架"""

    def __init__(self, model_a_name, model_b_name):
        self.model_a = model_a_name
        self.model_b = model_b_name
        self.comparisons = []

    def add_comparison(self, prompt, response_a, response_b, winner, evaluator_id):
        """添加一次比较结果"""
        self.comparisons.append({
            'prompt': prompt,
            'response_a': response_a,
            'response_b': response_b,
            'winner': winner,  # 'A', 'B', 'tie'
            'evaluator': evaluator_id,
        })

    def compute_statistics(self):
        """计算A/B测试统计量"""
        total = len(self.comparisons)
        wins_a = sum(1 for c in self.comparisons if c['winner'] == 'A')
        wins_b = sum(1 for c in self.comparisons if c['winner'] == 'B')
        ties = sum(1 for c in self.comparisons if c['winner'] == 'tie')

        win_rate_a = wins_a / total if total > 0 else 0
        win_rate_b = wins_b / total if total > 0 else 0
        tie_rate = ties / total if total > 0 else 0

        # 简化版统计显著性检验 (正态近似)
        # H0: 两个模型胜率相同 (p=0.5)
        p_hat = wins_a / total if total > 0 else 0.5
        se = np.sqrt(p_hat * (1 - p_hat) / total) if total > 0 else 1
        z_score = (p_hat - 0.5) / se if se > 0 else 0
        p_value = 2 * (1 - self._norm_cdf(abs(z_score)))

        return {
            'total': total,
            'wins_a': wins_a,
            'wins_b': wins_b,
            'ties': ties,
            'win_rate_a': win_rate_a,
            'win_rate_b': win_rate_b,
            'tie_rate': tie_rate,
            'z_score': z_score,
            'p_value': p_value,
            'significant': p_value < 0.05,
        }

    @staticmethod
    def _norm_cdf(x):
        """标准正态分布CDF近似"""
        return 0.5 * (1 + np.math.erf(x / np.sqrt(2)))


print("\n" + "=" * 60)
print("2. A/B测试模拟")
print("=" * 60)

# 模拟A/B测试
np.random.seed(42)
ab_test = ABTestFramework("Model-A (GPT-4)", "Model-B (Claude)")

sample_prompts = [
    "解释量子计算的基本原理",
    "用Python实现快速排序",
    "分析《红楼梦》的主题思想",
    "推导线性回归的损失函数",
    "设计一个电商推荐系统",
    "比较中美教育体系的差异",
    "解释区块链的工作原理",
    "写一首关于春天的诗",
]

evaluators = [f"评估者{i}" for i in range(1, 6)]

for prompt in sample_prompts:
    for evaluator in evaluators:
        # Model-A有55%的概率获胜 (模拟略好的模型)
        rand = np.random.random()
        if rand < 0.45:
            winner = 'A'
        elif rand < 0.80:
            winner = 'B'
        else:
            winner = 'tie'
        ab_test.add_comparison(prompt, "Response-A", "Response-B", winner, evaluator)

stats = ab_test.compute_statistics()
print(f"\n  总比较次数: {stats['total']}")
print(f"  {ab_test.model_a}: {stats['wins_a']}胜 ({stats['win_rate_a']:.1%})")
print(f"  {ab_test.model_b}: {stats['wins_b']}胜 ({stats['win_rate_b']:.1%})")
print(f"  平局: {stats['ties']}次 ({stats['tie_rate']:.1%})")
print(f"  Z-score: {stats['z_score']:.3f}")
print(f"  P-value: {stats['p_value']:.4f}")
print(f"  统计显著 (p<0.05): {'是' if stats['significant'] else '否'}")


# ============================================================
# 3. 评分标准制定
# ============================================================
class ScoringRubric:
    """评分标准框架"""

    def __init__(self):
        self.dimensions = {}

    def add_dimension(self, name, description, weight, levels):
        """添加评分维度"""
        self.dimensions[name] = {
            'description': description,
            'weight': weight,
            'levels': levels,  # {分数: 描述}
        }

    def score_response(self, scores):
        """根据各维度得分计算加权总分"""
        total_weight = sum(d['weight'] for d in self.dimensions.values())
        weighted_sum = 0
        for dim, score in scores.items():
            if dim in self.dimensions:
                weighted_sum += score * self.dimensions[dim]['weight']
        return weighted_sum / total_weight if total_weight > 0 else 0


print("\n" + "=" * 60)
print("3. 评分标准制定")
print("=" * 60)

rubric = ScoringRubric()
rubric.add_dimension('准确性', '回答是否事实正确', weight=3, levels={
    5: '完全正确', 4: '基本正确,有小错误', 3: '部分正确',
    2: '多处错误', 1: '严重错误',
})
rubric.add_dimension('完整性', '回答是否全面', weight=2, levels={
    5: '非常全面', 4: '较全面', 3: '基本涵盖要点',
    2: '遗漏较多', 1: '严重缺失',
})
rubric.add_dimension('清晰度', '表达是否清晰易懂', weight=2, levels={
    5: '非常清晰', 4: '较清晰', 3: '基本清晰',
    2: '表达混乱', 1: '无法理解',
})
rubric.add_dimension('安全性', '是否包含有害内容', weight=3, levels={
    5: '完全安全', 4: '基本安全', 3: '有轻微风险',
    2: '有明显问题', 1: '严重安全问题',
})

print("\n  评分维度:")
for dim, info in rubric.dimensions.items():
    print(f"    {dim} (权重={info['weight']}): {info['description']}")

# 模拟多个模型的评分
model_scores = {
    'GPT-4': {'准确性': 4.5, '完整性': 4.2, '清晰度': 4.3, '安全性': 4.8},
    'Claude': {'准确性': 4.3, '完整性': 4.5, '清晰度': 4.4, '安全性': 4.7},
    'LLaMA-2': {'准确性': 3.8, '完整性': 3.5, '清晰度': 3.9, '安全性': 3.6},
}

print("\n  模型综合评分:")
for model, scores in model_scores.items():
    total = rubric.score_response(scores)
    print(f"    {model}: {total:.2f}/5.0")
    for dim, score in scores.items():
        print(f"      {dim}: {score}")


# ============================================================
# 4. RLHF概念介绍
# ============================================================
print("\n" + "=" * 60)
print("4. RLHF (人类反馈强化学习) 概念")
print("=" * 60)

rlhf_steps = [
    ("Step 1: SFT (监督微调)", "用高质量对话数据微调预训练模型"),
    ("Step 2: RM (奖励模型)", "收集人类偏好数据, 训练奖励模型"),
    ("Step 3: PPO (强化学习)", "用奖励模型指导模型优化, 防止奖励作弊"),
]

print("\n  RLHF三步流程:")
for step, desc in rlhf_steps:
    print(f"    {step}: {desc}")

print("\n  RLHF中的评估:")
print("    - 偏好数据质量 → 直接影响奖励模型的效果")
print("    - 标注者间一致性 (Inter-annotator Agreement)")
print("    - 奖励模型准确率 → 能否准确预测人类偏好")
print("    - KL散度 → 确保模型不会偏离原始分布太远")


# ============================================================
# 5. 标注者一致性分析
# ============================================================
def compute_fleiss_kappa(ratings_matrix):
    """计算Fleiss' Kappa (多标注者一致性)"""
    n = len(ratings_matrix)        # 样本数
    k = len(ratings_matrix[0])     # 类别数
    N = sum(ratings_matrix[0])     # 每个样本的标注者数

    # P_i: 每个样本的一致性
    P_i = []
    for row in ratings_matrix:
        P_i.append((sum(x ** 2 for x in row) - N) / (N * (N - 1)))

    P_bar = np.mean(P_i)

    # p_j: 每个类别的边际比例
    p_j = []
    for j in range(k):
        p_j.append(sum(row[j] for row in ratings_matrix) / (n * N))

    P_e = sum(p ** 2 for p in p_j)

    kappa = (P_bar - P_e) / (1 - P_e) if (1 - P_e) > 0 else 0
    return kappa


print("\n" + "=" * 60)
print("5. 标注者一致性分析")
print("=" * 60)

# 模拟5个标注者对10个样本的偏好标注 (A/B/Tie)
np.random.seed(42)
ratings = []
for _ in range(20):
    row = [0, 0, 0]  # [选A的次数, 选B的次数, 选Tie的次数]
    for _ in range(5):
        r = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
        row[r] += 1
    ratings.append(row)

kappa = compute_fleiss_kappa(ratings)
print(f"  Fleiss' Kappa: {kappa:.4f}")
print(f"  一致性水平: ", end="")
if kappa < 0.2:
    print("轻微一致")
elif kappa < 0.4:
    print("一般一致")
elif kappa < 0.6:
    print("中等一致")
elif kappa < 0.8:
    print("高度一致")
else:
    print("几乎完全一致")


# ============================================================
# 6. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 6.1 A/B测试结果
ax1 = axes[0, 0]
labels = [ab_test.model_a, ab_test.model_b, '平局']
sizes = [stats['wins_a'], stats['wins_b'], stats['ties']]
colors = ['#4CAF50', '#2196F3', '#FF9800']
ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90,
        textprops={'fontsize': 9})
ax1.set_title('A/B测试胜负分布')

# 6.2 模型多维度评分
ax2 = axes[0, 1]
dims = list(rubric.dimensions.keys())
x = np.arange(len(dims))
width = 0.25
for i, (model, scores) in enumerate(model_scores.items()):
    values = [scores[d] for d in dims]
    ax2.bar(x + i * width, values, width, label=model, edgecolor='black')
ax2.set_ylabel('评分')
ax2.set_title('模型多维度评分对比')
ax2.set_xticks(x + width)
ax2.set_xticklabels(dims)
ax2.legend()
ax2.set_ylim(0, 5.5)

# 6.3 标注者一致性分布
ax3 = axes[1, 0]
agreement_levels = []
for row in ratings:
    max_agree = max(row)
    agreement_levels.append(max_agree / sum(row))
ax3.hist(agreement_levels, bins=10, color='#9C27B0', edgecolor='black', alpha=0.7)
ax3.axvline(x=np.mean(agreement_levels), color='red', linestyle='--',
            label=f'平均一致率={np.mean(agreement_levels):.2f}')
ax3.set_xlabel('一致率')
ax3.set_ylabel('样本数')
ax3.set_title('标注者一致率分布')
ax3.legend()

# 6.4 RLHF训练过程模拟
ax4 = axes[1, 1]
steps = np.arange(0, 100)
reward = 2.0 + 0.8 * (1 - np.exp(-steps / 30)) + np.random.normal(0, 0.05, 100)
kl_div = 0.01 * steps * np.exp(-steps / 80) + np.random.normal(0, 0.002, 100)
kl_div = np.maximum(kl_div, 0)

ax4_twin = ax4.twinx()
l1 = ax4.plot(steps, reward, 'g-', linewidth=2, label='奖励分数')
l2 = ax4_twin.plot(steps, kl_div, 'r-', linewidth=2, label='KL散度')
ax4.set_xlabel('训练步数')
ax4.set_ylabel('奖励分数', color='green')
ax4_twin.set_ylabel('KL散度', color='red')
ax4.set_title('RLHF训练过程模拟')

lines = l1 + l2
labels = [l.get_label() for l in lines]
ax4.legend(lines, labels, loc='center right')

plt.suptitle('W21-D3: 人类评估方法', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W21/d3_human_eval.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存!")
