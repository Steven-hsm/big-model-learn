"""
W27-D2 数据漂移检测
====================
数据漂移检测(PSI), 概念漂移, 特征分布变化, KS检验, 漂移检测实现
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W27-D2 数据漂移检测")
print("=" * 60)

# ============================================================
# 1. 漂移类型概述
# ============================================================
print("\n--- 1. 漂移类型概述 ---")
print("""
  漂移类型:
  ┌──────────────┬───────────────────────────────────────┐
  │ 类型         │ 描述                                  │
  ├──────────────┼───────────────────────────────────────┤
  │ 数据漂移     │ P(X) 变化 - 输入特征分布变化          │
  │ (Covariate)  │ 例: 用户年龄分布从年轻变为中年        │
  ├──────────────┼───────────────────────────────────────┤
  │ 概念漂移     │ P(Y|X) 变化 - 特征与标签的关系变化    │
  │ (Concept)    │ 例: 购买行为模式随季节变化            │
  ├──────────────┼───────────────────────────────────────┤
  │ 标签漂移     │ P(Y) 变化 - 目标分布变化              │
  │ (Prior)      │ 例: 欺诈比例突然上升                  │
  └──────────────┴───────────────────────────────────────┘

  检测方法:
    - PSI (Population Stability Index): 分布稳定性
    - KS检验: 连续变量分布差异
    - 卡方检验: 分类变量分布差异
    - Jensen-Shannon散度: 概率分布距离
    - ADWIN: 自适应窗口概念漂移检测
""")

# ============================================================
# 2. PSI (Population Stability Index)
# ============================================================
print("\n--- 2. PSI 计算 ---")


class PSIDetector:
    """PSI漂移检测器"""

    @staticmethod
    def calculate(reference, current, bins=10):
        """计算PSI"""
        all_data = np.concatenate([reference, current])
        bin_edges = np.percentile(all_data, np.linspace(0, 100, bins + 1))
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        ref_hist = np.histogram(reference, bins=bin_edges)[0] / len(reference)
        cur_hist = np.histogram(current, bins=bin_edges)[0] / len(current)

        ref_hist = np.clip(ref_hist, 1e-6, None)
        cur_hist = np.clip(cur_hist, 1e-6, None)

        psi = np.sum((cur_hist - ref_hist) * np.log(cur_hist / ref_hist))

        if psi < 0.1:
            level = '无显著变化'
        elif psi < 0.25:
            level = '轻微漂移 - 需关注'
        else:
            level = '显著漂移 - 需行动'

        return {
            'psi': psi,
            'level': level,
            'ref_distribution': ref_hist,
            'cur_distribution': cur_hist,
        }

    @staticmethod
    def calculate_for_all_features(ref_df, cur_df, features, bins=10):
        """计算所有特征的PSI"""
        results = {}
        for feat in features:
            result = PSIDetector.calculate(ref_df[feat].values, cur_df[feat].values, bins)
            results[feat] = result
        return results


# ============================================================
# 3. KS检验
# ============================================================
print("\n--- 3. KS检验 ---")


class KSDetector:
    """KS检验漂移检测器"""

    @staticmethod
    def test(reference, current, alpha=0.05):
        """KS检验"""
        stat, p_value = scipy_stats.ks_2samp(reference, current)
        return {
            'statistic': stat,
            'p_value': p_value,
            'drift_detected': p_value < alpha,
            'alpha': alpha,
        }

    @staticmethod
    def test_all_features(ref_df, cur_df, features, alpha=0.05):
        results = {}
        for feat in features:
            results[feat] = KSDetector.test(ref_df[feat].values, cur_df[feat].values, alpha)
        return results


# ============================================================
# 4. 概念漂移检测
# ============================================================
print("\n--- 4. 概念漂移检测 ---")


class ConceptDriftDetector:
    """概念漂移检测 (基于性能监控)"""

    def __init__(self, window_size=10, threshold=2.0):
        self.window_size = window_size
        self.threshold = threshold
        self.scores = []
        self.drift_points = []

    def add_score(self, score):
        """添加新的性能分数"""
        self.scores.append(score)

        if len(self.scores) >= 2 * self.window_size:
            # 比较最近窗口和之前窗口
            recent = self.scores[-self.window_size:]
            previous = self.scores[-2 * self.window_size:-self.window_size]

            recent_mean = np.mean(recent)
            prev_mean = np.mean(previous)
            prev_std = np.std(previous) + 1e-10

            z_score = (recent_mean - prev_mean) / prev_std

            if abs(z_score) > self.threshold:
                self.drift_points.append({
                    'index': len(self.scores),
                    'z_score': z_score,
                    'recent_mean': recent_mean,
                    'previous_mean': prev_mean,
                })
                return True, z_score

        return False, 0

    def detect_ddm(self, min_instances=30, warning_level=2.0, drift_level=3.0):
        """DDM (Drift Detection Method)"""
        warnings = []
        drifts = []

        for i in range(min_instances, len(self.scores)):
            window = self.scores[:i+1]
            p = np.mean(window)
            s = np.std(window) + 1e-10

            if p + s * warning_level < p:
                warnings.append(i)
            if p + s * drift_level < p:
                drifts.append(i)

        return warnings, drifts


# ============================================================
# 5. 综合漂移检测实验
# ============================================================
print("\n--- 5. 综合漂移检测实验 ---")

# 生成基准数据
np.random.seed(42)
n = 1000
ref_df = pd.DataFrame({
    'age': np.random.normal(35, 10, n),
    'income': np.random.normal(50000, 15000, n),
    'score': np.random.uniform(0, 100, n),
    'tenure': np.random.exponential(3, n),
    'balance': np.random.normal(10000, 3000, n),
})

# 模拟不同漂移场景
scenarios = {
    '无漂移': pd.DataFrame({
        'age': np.random.normal(35, 10, n),
        'income': np.random.normal(50000, 15000, n),
        'score': np.random.uniform(0, 100, n),
        'tenure': np.random.exponential(3, n),
        'balance': np.random.normal(10000, 3000, n),
    }),
    '轻微漂移': pd.DataFrame({
        'age': np.random.normal(37, 11, n),  # 均值和方差微调
        'income': np.random.normal(52000, 16000, n),
        'score': np.random.uniform(5, 105, n),
        'tenure': np.random.exponential(3.2, n),
        'balance': np.random.normal(10500, 3100, n),
    }),
    '显著漂移': pd.DataFrame({
        'age': np.random.normal(45, 12, n),  # 大幅偏移
        'income': np.random.normal(70000, 20000, n),
        'score': np.random.uniform(20, 100, n),
        'tenure': np.random.exponential(5, n),
        'balance': np.random.normal(15000, 5000, n),
    }),
}

features = ['age', 'income', 'score', 'tenure', 'balance']

print("\n  PSI检测结果:")
print(f"  {'场景':<12} {'age':<12} {'income':<12} {'score':<12} {'tenure':<12} {'balance':<12}")
print("  " + "-" * 72)

psi_results_all = {}
for scenario_name, cur_df in scenarios.items():
    psi_results = PSIDetector.calculate_for_all_features(ref_df, cur_df, features)
    psi_results_all[scenario_name] = psi_results
    row = f"  {scenario_name:<12}"
    for feat in features:
        psi_val = psi_results[feat]['psi']
        row += f" {psi_val:<12.4f}"
    print(row)

print("\n  KS检验结果:")
print(f"  {'场景':<12} {'age':<12} {'income':<12} {'score':<12} {'tenure':<12} {'balance':<12}")
print("  " + "-" * 72)

for scenario_name, cur_df in scenarios.items():
    ks_results = KSDetector.test_all_features(ref_df, cur_df, features)
    row = f"  {scenario_name:<12}"
    for feat in features:
        p_val = ks_results[feat]['p_value']
        marker = '*' if p_val < 0.05 else ' '
        row += f" {p_val:<10.4f}{marker}"
    print(row)

# 概念漂移检测
print("\n  概念漂移检测:")
cd_detector = ConceptDriftDetector(window_size=5, threshold=2.0)

# 模拟性能分数: 先稳定后下降
np.random.seed(42)
stable_scores = [0.95 + np.random.randn() * 0.01 for _ in range(20)]
declining_scores = [0.95 - i * 0.008 + np.random.randn() * 0.01 for i in range(20)]
all_scores = stable_scores + declining_scores

drift_detected_at = []
for i, score in enumerate(all_scores):
    is_drift, z = cd_detector.add_score(score)
    if is_drift:
        drift_detected_at.append(i)
        print(f"    概念漂移检测于 index={i}, z_score={z:.3f}")

if not drift_detected_at:
    print("    未检测到概念漂移")

# ============================================================
# 6. 可视化
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 第一行: 三种漂移场景的分布对比
scenarios_list = ['无漂移', '轻微漂移', '显著漂移']
for col, scenario in enumerate(scenarios_list):
    ax = axes[0, col]
    feat = 'income'
    ax.hist(ref_df[feat], bins=30, alpha=0.5, density=True, label='参考', color='#2196F3')
    ax.hist(scenarios[scenario][feat], bins=30, alpha=0.5, density=True, label='当前', color='#F44336')
    psi_val = psi_results_all[scenario][feat]['psi']
    ax.set_title(f'{scenario}\n(income PSI={psi_val:.3f})')
    ax.legend(fontsize=8)
    if col == 0:
        ax.set_ylabel('密度')

# 第二行左: PSI热力图
ax = axes[1, 0]
psi_matrix = np.zeros((len(scenarios_list), len(features)))
for i, scenario in enumerate(scenarios_list):
    for j, feat in enumerate(features):
        psi_matrix[i, j] = psi_results_all[scenario][feat]['psi']

im = ax.imshow(psi_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=0.3)
ax.set_xticks(range(len(features)))
ax.set_yticks(range(len(scenarios_list)))
ax.set_xticklabels(features, fontsize=9)
ax.set_yticklabels(scenarios_list)
ax.set_title('PSI漂移热力图')
for i in range(len(scenarios_list)):
    for j in range(len(features)):
        color = 'white' if psi_matrix[i, j] > 0.15 else 'black'
        ax.text(j, i, f'{psi_matrix[i,j]:.3f}', ha='center', va='center', fontsize=9, color=color)
plt.colorbar(im, ax=ax, label='PSI值')

# 第二行中: 概念漂移检测
ax = axes[1, 1]
ax.plot(all_scores, 'b-', label='模型准确率')
ax.axvline(20, color='orange', linestyle='--', label='漂移开始点')
for dp in drift_detected_at:
    ax.axvline(dp, color='red', linestyle='--', alpha=0.7)
    ax.text(dp, min(all_scores), f'检测到\n漂移@{dp}', ha='center', fontsize=8, color='red')

# 滑动平均
window = 5
moving_avg = pd.Series(all_scores).rolling(window).mean()
ax.plot(moving_avg, 'r-', linewidth=2, label=f'{window}期移动平均')
ax.set_xlabel('时间')
ax.set_ylabel('准确率')
ax.set_title('概念漂移检测')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 第二行右: 特征PSI雷达图
ax = axes[1, 2]
for scenario in scenarios_list:
    psi_vals = [psi_results_all[scenario][f]['psi'] for f in features]
    ax.plot(features, psi_vals, 'o-', label=scenario, markersize=6)
ax.axhline(0.1, color='green', linestyle='--', alpha=0.5, label='轻微漂移阈值')
ax.axhline(0.25, color='red', linestyle='--', alpha=0.5, label='显著漂移阈值')
ax.set_ylabel('PSI')
ax.set_title('各特征PSI对比')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.suptitle('数据漂移检测分析', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W27/d2_data_drift.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存: d2_data_drift.png")

print("\n完成! 数据漂移检测要点:")
print("  1. PSI: 通用漂移检测, PSI<0.1稳定, >0.25显著漂移")
print("  2. KS检验: 连续变量分布差异的统计检验")
print("  3. 概念漂移: 监控模型性能的突变")
print("  4. 多特征监控: 对所有关键特征进行漂移检测")
print("  5. 自动化: 定期运行漂移检测, 异常时触发告警")
