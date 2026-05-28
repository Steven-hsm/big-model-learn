### Day 5（周五）：DPO (Direct Preference Optimization)
# Bradley-Terry模型, DPO vs PPO对比, 简化DPO训练实现

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. DPO原理详解
# ============================================================
print("=" * 60)
print("1. DPO (Direct Preference Optimization) 原理")
print("=" * 60)

print("""
  核心思想: 绕过奖励模型, 直接用偏好数据优化策略模型

  传统RLHF流程:
    偏好数据 → 训练奖励模型 → PPO优化策略模型

  DPO流程:
    偏好数据 → 直接优化策略模型 (一步到位!)

  Bradley-Terry模型:
    P(y_w > y_l | x) = sigmoid(r(x, y_w) - r(x, y_l))

  DPO损失函数:
    L_DPO = -E[log sigmoid(beta * (log pi(y_w|x)/pi_ref(y_w|x)
                                    - log pi(y_l|x)/pi_ref(y_l|x)))]

  其中:
    y_w = chosen (人类偏好的回答)
    y_l = rejected (人类不喜欢的回答)
    pi = 当前策略模型
    pi_ref = 参考模型 (SFT模型)
    beta = 温度参数
""")


# ============================================================
# 2. Bradley-Terry模型实现
# ============================================================
class BradleyTerryModel:
    """Bradley-Terry偏好模型"""

    def __init__(self, num_items):
        self.num_items = num_items
        self.scores = np.random.randn(num_items) * 0.1  # 每个项目的潜在分数

    def preference_probability(self, i, j):
        """计算项目i优于项目j的概率"""
        return 1.0 / (1.0 + np.exp(-(self.scores[i] - self.scores[j])))

    def train(self, comparisons, lr=0.01, epochs=100):
        """从比较数据中学习分数"""
        losses = []
        for epoch in range(epochs):
            total_loss = 0
            for winner, loser in comparisons:
                prob = self.preference_probability(winner, loser)
                loss = -np.log(prob + 1e-8)

                # 梯度
                grad = -(1 - prob)
                self.scores[winner] -= lr * grad
                self.scores[loser] += lr * grad

                total_loss += loss
            losses.append(total_loss / len(comparisons))
        return losses


print("=" * 60)
print("2. Bradley-Terry模型训练")
print("=" * 60)

np.random.seed(42)
bt_model = BradleyTerryModel(num_items=5)
item_names = ['回答A', '回答B', '回答C', '回答D', '回答E']

# 模拟比较数据: (winner_index, loser_index)
# 假设真实排序: E > D > C > B > A
comparisons = []
for _ in range(100):
    i, j = np.random.choice(5, 2, replace=False)
    # 质量越高的越可能获胜
    true_quality = [0.2, 0.4, 0.6, 0.8, 1.0]
    if np.random.random() < true_quality[i] / (true_quality[i] + true_quality[j]):
        comparisons.append((i, j))
    else:
        comparisons.append((j, i))

losses = bt_model.train(comparisons, lr=0.05, epochs=50)

print(f"  训练完成! 最终损失: {losses[-1]:.4f}")
print(f"  学到的分数 (越高越好):")
for name, score in sorted(zip(item_names, bt_model.scores), key=lambda x: -x[1]):
    print(f"    {name}: {score:.4f}")

print(f"\n  偏好概率矩阵:")
print(f"    {'':>8}", end='')
for name in item_names:
    print(f"  {name:>8}", end='')
print()
for i, name_i in enumerate(item_names):
    print(f"    {name_i:>8}", end='')
    for j in range(5):
        prob = bt_model.preference_probability(i, j)
        print(f"  {prob:>8.3f}", end='')
    print()


# ============================================================
# 3. 简化版DPO训练实现
# ============================================================
class SimpleDPO:
    """简化版DPO训练器"""

    def __init__(self, num_params=10, beta=0.1):
        self.num_params = num_params
        self.beta = beta
        # 当前策略模型参数
        self.policy_params = np.random.randn(num_params) * 0.1
        # 参考模型参数 (冻结)
        self.ref_params = self.policy_params.copy()
        self.history = {'loss': [], 'chosen_reward': [], 'rejected_reward': [], 'margin': []}

    def log_ratio(self, response_features, is_policy=True):
        """计算 log(pi(y|x) / pi_ref(y|x)) 的简化版"""
        params = self.policy_params if is_policy else self.ref_params
        log_prob = -np.sum((params - response_features) ** 2) / 2
        ref_log_prob = -np.sum((self.ref_params - response_features) ** 2) / 2
        return log_prob - ref_log_prob

    def compute_loss(self, chosen_feat, rejected_feat):
        """计算DPO损失"""
        log_ratio_chosen = self.log_ratio(chosen_feat, True)
        log_ratio_rejected = self.log_ratio(rejected_feat, True)

        margin = self.beta * (log_ratio_chosen - log_ratio_rejected)
        loss = -np.log(1 / (1 + np.exp(-margin)) + 1e-8)
        return loss, margin

    def train_step(self, chosen_feat, rejected_feat, lr=0.01):
        """DPO训练一步"""
        loss, margin = self.compute_loss(chosen_feat, rejected_feat)

        # 简化梯度: 增加chosen的log_ratio, 减少rejected的log_ratio
        sigmoid_val = 1 / (1 + np.exp(-margin))

        # 梯度方向
        grad = (1 - sigmoid_val) * self.beta
        self.policy_params += lr * grad * (chosen_feat - rejected_feat)

        # 记录
        chosen_r = np.dot(self.policy_params, chosen_feat)
        rejected_r = np.dot(self.policy_params, rejected_feat)
        self.history['loss'].append(loss)
        self.history['chosen_reward'].append(chosen_r)
        self.history['rejected_reward'].append(rejected_r)
        self.history['margin'].append(chosen_r - rejected_r)

        return loss

    def train(self, dataset, epochs=5, lr=0.01):
        """完整训练"""
        for epoch in range(epochs):
            epoch_losses = []
            for chosen_feat, rejected_feat in dataset:
                loss = self.train_step(chosen_feat, rejected_feat, lr)
                epoch_losses.append(loss)
            print(f"  Epoch {epoch + 1}/{epochs}, 平均损失: {np.mean(epoch_losses):.4f}")


print("\n" + "=" * 60)
print("3. 简化版DPO训练")
print("=" * 60)

np.random.seed(42)
dpo = SimpleDPO(num_params=8, beta=0.1)

# 创建偏好数据集
dataset = []
for _ in range(100):
    chosen = np.random.uniform(0.5, 1.0, 8)
    rejected = np.random.uniform(0.0, 0.5, 8)
    dataset.append((chosen, rejected))

print("  开始训练...")
dpo.train(dataset, epochs=10, lr=0.005)

print(f"\n  训练结果:")
print(f"    最终损失: {dpo.history['loss'][-1]:.4f}")
print(f"    Chosen奖励: {dpo.history['chosen_reward'][-1]:.4f}")
print(f"    Rejected奖励: {dpo.history['rejected_reward'][-1]:.4f}")
print(f"    奖励差距: {dpo.history['margin'][-1]:.4f}")


# ============================================================
# 4. DPO vs PPO 对比
# ============================================================
print("\n" + "=" * 60)
print("4. DPO vs PPO 对比分析")
print("=" * 60)

comparison = {
    '维度': ['训练复杂度', '是否需要奖励模型', '训练稳定性', '样本效率',
             '计算成本', '实现难度', 'KL控制', '适用场景'],
    'PPO': ['高 (需要4个模型)', '是', '较低 (超参敏感)', '较低',
            '高', '高', '显式 (KL惩罚)', '大规模部署'],
    'DPO': ['低 (只需2个模型)', '否', '较高', '较高',
            '低', '低', '隐式 (beta参数)', '快速迭代'],
}

print(f"  {'维度':<12} {'PPO':<20} {'DPO':<20}")
print(f"  {'-' * 52}")
for i in range(len(comparison['维度'])):
    print(f"  {comparison['维度'][i]:<12} {comparison['PPO'][i]:<20} {comparison['DPO'][i]:<20}")


# ============================================================
# 5. Beta参数影响分析
# ============================================================
print("\n" + "=" * 60)
print("5. Beta参数 (温度) 影响分析")
print("=" * 60)

beta_values = [0.01, 0.05, 0.1, 0.5, 1.0]
beta_results = {}

for beta in beta_values:
    np.random.seed(42)
    dpo_beta = SimpleDPO(num_params=8, beta=beta)
    for chosen, rejected in dataset[:50]:
        dpo_beta.train_step(chosen, rejected, lr=0.005)
    beta_results[beta] = {
        'final_loss': dpo_beta.history['loss'][-1],
        'margin': dpo_beta.history['margin'][-1],
        'losses': dpo_beta.history['loss'],
    }
    print(f"  Beta={beta:.2f}: 最终损失={beta_results[beta]['final_loss']:.4f}, "
          f"奖励差距={beta_results[beta]['margin']:.4f}")


# ============================================================
# 6. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 6.1 Bradley-Terry训练损失
ax1 = axes[0, 0]
ax1.plot(losses, 'b-', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('平均损失')
ax1.set_title('Bradley-Terry模型训练损失')
ax1.grid(True, alpha=0.3)

# 6.2 DPO训练过程
ax2 = axes[0, 1]
steps = range(len(dpo.history['loss']))
ax2.plot(steps, dpo.history['chosen_reward'], 'g-', alpha=0.6, label='Chosen奖励')
ax2.plot(steps, dpo.history['rejected_reward'], 'r-', alpha=0.6, label='Rejected奖励')
# 平滑
window = 20
if len(dpo.history['margin']) > window:
    smooth_margin = [np.mean(dpo.history['margin'][i:i + window])
                     for i in range(0, len(dpo.history['margin']) - window)]
    ax2_twin = ax2.twinx()
    ax2_twin.plot(range(len(smooth_margin)), smooth_margin, 'k--', linewidth=2,
                  label='奖励差距 (平滑)')
    ax2_twin.set_ylabel('奖励差距', color='black')
    ax2_twin.legend(loc='lower right')
ax2.set_xlabel('训练步数')
ax2.set_ylabel('奖励值')
ax2.set_title('DPO训练过程')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

# 6.3 Beta参数影响
ax3 = axes[1, 0]
betas = list(beta_results.keys())
final_losses = [beta_results[b]['final_loss'] for b in betas]
margins = [beta_results[b]['margin'] for b in betas]
ax3_twin = ax3.twinx()
l1 = ax3.plot([str(b) for b in betas], final_losses, 'bo-', linewidth=2, label='最终损失')
l2 = ax3_twin.plot([str(b) for b in betas], margins, 'rs-', linewidth=2, label='奖励差距')
ax3.set_xlabel('Beta值')
ax3.set_ylabel('损失', color='blue')
ax3_twin.set_ylabel('奖励差距', color='red')
ax3.set_title('Beta参数对DPO的影响')
lines = l1 + l2
ax3.legend(lines, [l.get_label() for l in lines])

# 6.4 DPO vs PPO对比雷达图
ax4 = axes[1, 1]
categories = ['训练稳定性', '实现简单', '样本效率', '计算效率', 'KL控制', '适用性']
num_cats = len(categories)
angles = np.linspace(0, 2 * np.pi, num_cats, endpoint=False).tolist()
angles += angles[:1]

ppo_scores = [3, 2, 3, 2, 4, 4]
dpo_scores = [4, 5, 4, 5, 3, 4]
ppo_scores += ppo_scores[:1]
dpo_scores += dpo_scores[:1]

ax4.plot(angles, ppo_scores, 'ro-', linewidth=2, label='PPO')
ax4.fill(angles, ppo_scores, alpha=0.15, color='red')
ax4.plot(angles, dpo_scores, 'bo-', linewidth=2, label='DPO')
ax4.fill(angles, dpo_scores, alpha=0.15, color='blue')
ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(categories)
ax4.set_ylim(0, 5.5)
ax4.set_title('DPO vs PPO 对比 (分数越高越好)')
ax4.legend()

plt.suptitle('W21-D5: DPO (Direct Preference Optimization)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W21/d5_dpo.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存!")
