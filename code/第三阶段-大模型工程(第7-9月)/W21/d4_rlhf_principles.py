### Day 4（周四）：RLHF原理与实现
# SFT→RM→PPO三步流程, 奖励模型训练, PPO算法, KL散度惩罚

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. RLHF三步流程详解
# ============================================================
print("=" * 60)
print("1. RLHF三步流程详解")
print("=" * 60)

print("""
  Step 1: 监督微调 (SFT - Supervised Fine-Tuning)
    - 输入: 高质量指令-回答对
    - 目标: 让预训练模型学会遵循指令
    - 损失: 标准交叉熵损失 L = -sum(y * log(p))

  Step 2: 奖励模型训练 (RM - Reward Model)
    - 输入: 人类偏好对比数据 (chosen > rejected)
    - 目标: 学习人类偏好, 输出标量奖励值
    - 损失: 对比损失 L = -log(sigmoid(r_chosen - r_rejected))

  Step 3: PPO强化学习优化
    - 输入: SFT模型 + 奖励模型
    - 目标: 最大化奖励, 同时保持与SFT模型接近
    - 目标函数: maximize E[R(x,y)] - beta * KL(pi || pi_ref)
""")


# ============================================================
# 2. 简化版奖励模型实现
# ============================================================
class SimpleRewardModel:
    """简化版奖励模型: 基于特征的线性奖励函数"""

    def __init__(self, num_features=6):
        self.num_features = num_features
        # 特征: [相关性, 流畅性, 信息量, 安全性, 有帮助, 准确性]
        self.feature_names = ['相关性', '流畅性', '信息量', '安全性', '有帮助', '准确性']
        self.weights = np.random.randn(num_features) * 0.1
        self.bias = 0.0

    def extract_features(self, response):
        """模拟特征提取 (实际中需要用另一个模型)"""
        # 这里用随机值模拟特征
        features = np.random.rand(self.num_features)
        return features

    def predict(self, features):
        """预测奖励值"""
        return np.dot(self.weights, features) + self.bias

    def train(self, chosen_features, rejected_features, lr=0.01, epochs=100):
        """训练奖励模型 (使用对比损失)"""
        losses = []
        for epoch in range(epochs):
            # 对比损失: -log(sigmoid(r_chosen - r_rejected))
            r_chosen = self.predict(chosen_features)
            r_rejected = self.predict(rejected_features)

            diff = r_chosen - r_rejected
            loss = -np.log(1 / (1 + np.exp(-diff)) + 1e-8)

            # 梯度计算
            sigmoid_val = 1 / (1 + np.exp(-diff))
            grad = -(1 - sigmoid_val)

            # 更新权重
            grad_w = grad * (chosen_features - rejected_features)
            self.weights -= lr * grad_w
            self.bias -= lr * grad

            losses.append(loss)

        return losses


print("=" * 60)
print("2. 简化版奖励模型训练")
print("=" * 60)

np.random.seed(42)
reward_model = SimpleRewardModel()

# 模拟偏好数据: chosen回答的特征普遍比rejected好
num_pairs = 50
chosen_data = []
rejected_data = []
for _ in range(num_pairs):
    chosen = np.random.uniform(0.6, 1.0, 6)     # 较好的回答
    rejected = np.random.uniform(0.1, 0.5, 6)    # 较差的回答
    chosen_data.append(chosen)
    rejected_data.append(rejected)

# 训练
all_losses = []
for i in range(num_pairs):
    losses = reward_model.train(chosen_data[i], rejected_data[i], lr=0.02, epochs=10)
    all_losses.extend(losses)

print(f"  训练完成! 最终损失: {all_losses[-1]:.4f}")
print(f"  学到的权重:")
for name, w in zip(reward_model.feature_names, reward_model.weights):
    print(f"    {name}: {w:.4f}")

# 测试奖励模型
test_good = np.array([0.9, 0.8, 0.85, 0.95, 0.9, 0.88])
test_bad = np.array([0.2, 0.3, 0.15, 0.4, 0.25, 0.1])
print(f"\n  测试:")
print(f"    好回答奖励: {reward_model.predict(test_good):.4f}")
print(f"    差回答奖励: {reward_model.predict(test_bad):.4f}")


# ============================================================
# 3. PPO算法原理
# ============================================================
class SimplePPO:
    """简化版PPO算法"""

    def __init__(self, initial_params, ref_params, reward_fn,
                 clip_ratio=0.2, kl_coeff=0.1):
        self.params = initial_params.copy()
        self.ref_params = ref_params.copy()
        self.reward_fn = reward_fn
        self.clip_ratio = clip_ratio
        self.kl_coeff = kl_coeff
        self.history = {'reward': [], 'kl': [], 'objective': []}

    def compute_kl_divergence(self):
        """计算KL散度 (简化: 参数空间的L2距离)"""
        return np.sum((self.params - self.ref_params) ** 2) / 2

    def generate_response(self):
        """模拟生成回答 (参数→输出)"""
        quality = np.mean(self.params) + np.random.normal(0, 0.1)
        return quality

    def compute_ratio(self, new_params):
        """计算重要性采样比率 (简化)"""
        old_prob = np.exp(-np.sum((self.params - 0.5) ** 2))
        new_prob = np.exp(-np.sum((new_params - 0.5) ** 2))
        return new_prob / (old_prob + 1e-8)

    def step(self):
        """PPO更新一步"""
        # 生成回答并获取奖励
        quality = self.generate_response()
        reward = self.reward_fn(quality)

        # 计算KL惩罚
        kl = self.compute_kl_divergence()

        # 总目标: reward - beta * KL
        objective = reward - self.kl_coeff * kl

        # PPO更新: 带clip的梯度更新
        gradient = np.random.randn(len(self.params)) * 0.1
        new_params = self.params + 0.01 * gradient

        # Clip: 限制更新幅度
        diff = new_params - self.params
        clip_mask = np.abs(diff) > self.clip_ratio
        diff[clip_mask] = np.sign(diff[clip_mask]) * self.clip_ratio
        self.params = self.params + diff

        self.history['reward'].append(reward)
        self.history['kl'].append(kl)
        self.history['objective'].append(objective)

        return reward, kl, objective


print("\n" + "=" * 60)
print("3. PPO强化学习模拟")
print("=" * 60)

np.random.seed(42)
initial_params = np.random.randn(10) * 0.1
ref_params = initial_params.copy()


def reward_function(quality):
    """模拟奖励函数"""
    return quality * 2.0


ppo = SimplePPO(initial_params, ref_params, reward_function,
                clip_ratio=0.2, kl_coeff=0.05)

for step in range(200):
    r, kl, obj = ppo.step()

print(f"  训练完成 (200步)!")
print(f"  初始奖励: {ppo.history['reward'][0]:.4f}")
print(f"  最终奖励: {ppo.history['reward'][-1]:.4f}")
print(f"  初始KL散度: {ppo.history['kl'][0]:.4f}")
print(f"  最终KL散度: {ppo.history['kl'][-1]:.4f}")


# ============================================================
# 4. KL散度惩罚分析
# ============================================================
print("\n" + "=" * 60)
print("4. KL散度惩罚分析")
print("=" * 60)

# 不同KL系数的效果对比
kl_coeffs = [0.01, 0.05, 0.1, 0.5]
kl_comparison = {}

for coeff in kl_coeffs:
    np.random.seed(42)
    ppo_test = SimplePPO(
        np.random.randn(10) * 0.1,
        np.random.randn(10) * 0.1,
        reward_function,
        kl_coeff=coeff,
    )
    for _ in range(200):
        ppo_test.step()
    kl_comparison[coeff] = {
        'rewards': ppo_test.history['reward'],
        'kl': ppo_test.history['kl'],
        'final_reward': ppo_test.history['reward'][-1],
        'final_kl': ppo_test.history['kl'][-1],
    }
    print(f"  KL系数={coeff:.2f}: 最终奖励={kl_comparison[coeff]['final_reward']:.4f}, "
          f"最终KL={kl_comparison[coeff]['final_kl']:.4f}")

print("\n  → KL系数越大, 模型越保守 (奖励低但KL小)")
print("  → KL系数越小, 模型越激进 (奖励高但可能偏离原始分布)")


# ============================================================
# 5. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 5.1 奖励模型训练损失
ax1 = axes[0, 0]
window = 50
smoothed = [np.mean(all_losses[i:i + window])
            for i in range(0, len(all_losses) - window, window)]
ax1.plot(smoothed, 'b-', linewidth=2)
ax1.set_xlabel('训练步数 (窗口平均)')
ax1.set_ylabel('对比损失')
ax1.set_title('奖励模型训练损失曲线')
ax1.grid(True, alpha=0.3)

# 5.2 PPO训练过程
ax2 = axes[0, 1]
steps = range(len(ppo.history['reward']))
ax2.plot(steps, ppo.history['reward'], 'g-', alpha=0.5, linewidth=1)
window = 20
smoothed_reward = [np.mean(ppo.history['reward'][i:i + window])
                   for i in range(0, len(ppo.history['reward']) - window)]
ax2.plot(range(len(smoothed_reward)), smoothed_reward, 'g-', linewidth=2,
         label='奖励 (平滑)')
ax2.set_xlabel('训练步数')
ax2.set_ylabel('奖励')
ax2.set_title('PPO训练奖励曲线')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 5.3 KL散度随训练变化
ax3 = axes[1, 0]
for coeff in kl_coeffs:
    kl_data = kl_comparison[coeff]['kl']
    smoothed_kl = [np.mean(kl_data[i:i + 20])
                   for i in range(0, len(kl_data) - 20)]
    ax3.plot(smoothed_kl, label=f'KL系数={coeff}', linewidth=2)
ax3.set_xlabel('训练步数')
ax3.set_ylabel('KL散度')
ax3.set_title('不同KL系数下的KL散度变化')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 5.4 KL系数 vs 最终奖励/KL
ax4 = axes[1, 1]
final_rewards = [kl_comparison[c]['final_reward'] for c in kl_coeffs]
final_kls = [kl_comparison[c]['final_kl'] for c in kl_coeffs]
ax4_twin = ax4.twinx()
l1 = ax4.bar(range(len(kl_coeffs)), final_rewards, width=0.4, color='green',
             alpha=0.7, label='最终奖励', edgecolor='black')
l2 = ax4_twin.bar([x + 0.4 for x in range(len(kl_coeffs))], final_kls, width=0.4,
                   color='red', alpha=0.7, label='最终KL散度', edgecolor='black')
ax4.set_xlabel('KL系数')
ax4.set_ylabel('奖励', color='green')
ax4_twin.set_ylabel('KL散度', color='red')
ax4.set_title('KL系数对最终结果的影响')
ax4.set_xticks([x + 0.2 for x in range(len(kl_coeffs))])
ax4.set_xticklabels([str(c) for c in kl_coeffs])

lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_twin.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2)

plt.suptitle('W21-D4: RLHF原理与实现', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W21/d4_rlhf_principles.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存!")
