"""
W27-D4 模型测试策略
====================
模型测试策略, 预测质量测试, 延迟测试, 公平性测试, 鲁棒性测试
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W27-D4 模型测试策略")
print("=" * 60)

# ============================================================
# 1. 模型测试策略概述
# ============================================================
print("\n--- 1. 模型测试策略概述 ---")
print("""
  ML模型测试金字塔:

            ┌───────────┐
            │ E2E 测试  │  端到端: 完整推理流程
            ├───────────┤
            │ 集成测试  │  组件交互: Pipeline+模型+API
            ├───────────┤
            │ 专项测试  │  公平性/鲁棒性/安全性
            ├───────────┤
            │ 性能测试  │  延迟/吞吐/资源/稳定性
            ├───────────┤
            │ 质量测试  │  准确率/F1/AUC/回归测试
            └───────────┘

  测试类别:
    1. 预测质量: 准确率达标, 不劣于基线, 边界case
    2. 延迟性能: P50/P95/P99延迟, 吞吐量
    3. 公平性:   不同群体性能差异
    4. 鲁棒性:   噪声/异常/对抗样本下的表现
    5. 一致性:   相同输入得到相同输出
""")

# ============================================================
# 2. 准备数据
# ============================================================
print("\n--- 2. 准备数据 ---")

data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_s, y_train)

print(f"  训练集: {X_train_s.shape}, 测试集: {X_test_s.shape}")

# ============================================================
# 3. 预测质量测试
# ============================================================
print("\n--- 3. 预测质量测试 ---")


class PredictionQualityTests:
    """预测质量测试"""

    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = model.predict(X_test)
        self.results = []

    def test_accuracy_threshold(self, threshold=0.90):
        """准确率阈值测试"""
        acc = accuracy_score(self.y_test, self.y_pred)
        passed = acc >= threshold
        self.results.append(('准确率阈值', passed, f'{acc:.4f} >= {threshold}'))
        return passed

    def test_f1_threshold(self, threshold=0.85):
        """F1阈值测试"""
        f1 = f1_score(self.y_test, self.y_pred)
        passed = f1 >= threshold
        self.results.append(('F1阈值', passed, f'{f1:.4f} >= {threshold}'))
        return passed

    def test_class_balance(self, max_ratio=5.0):
        """类别平衡测试"""
        cm = confusion_matrix(self.y_test, self.y_pred)
        # 每类的召回率
        recalls = cm.diagonal() / cm.sum(axis=1)
        min_recall = recalls.min()
        max_recall = recalls.max()
        passed = max_recall / (min_recall + 1e-10) < max_ratio
        self.results.append(('类别平衡', passed,
                             f'召回率比={max_recall/(min_recall+1e-10):.2f}'))
        return passed

    def test_no_degradation(self, baseline_model, max_drop=0.05):
        """回归测试: 不劣于基线模型"""
        baseline_pred = baseline_model.predict(self.X_test)
        baseline_acc = accuracy_score(self.y_test, baseline_pred)
        current_acc = accuracy_score(self.y_test, self.y_pred)
        drop = baseline_acc - current_acc
        passed = drop <= max_drop
        self.results.append(('回归测试', passed,
                             f'基线={baseline_acc:.4f}, 当前={current_acc:.4f}, 差距={drop:+.4f}'))
        return passed

    def test_edge_cases(self, edge_indices=None):
        """边界case测试"""
        if edge_indices is None:
            # 选择概率最不确定的样本
            proba = self.model.predict_proba(self.X_test)
            uncertainty = 1 - np.max(proba, axis=1)
            edge_indices = np.argsort(uncertainty)[-20:]

        edge_correct = (self.y_pred[edge_indices] == self.y_test[edge_indices]).mean()
        passed = edge_correct >= 0.5  # 不确定性高的样本至少50%正确
        self.results.append(('边界case', passed,
                             f'高不确定性样本准确率={edge_correct:.2%}'))
        return passed

    def run_all(self):
        self.results = []
        self.test_accuracy_threshold()
        self.test_f1_threshold()
        self.test_class_balance()
        baseline = LogisticRegression(max_iter=1000, random_state=42)
        baseline.fit(self.X_test, self.y_test)  # 简化: 用测试集做基线
        self.test_no_degradation(baseline)
        self.test_edge_cases()
        return self.report()

    def report(self):
        print(f"\n  预测质量测试:")
        for name, passed, detail in self.results:
            status = 'PASS' if passed else 'FAIL'
            print(f"    [{status}] {name}: {detail}")
        return all(r[1] for r in self.results)


quality_tests = PredictionQualityTests(model, X_test_s, y_test)
quality_tests.run_all()

# ============================================================
# 4. 延迟测试
# ============================================================
print("\n--- 4. 延迟测试 ---")


class LatencyTests:
    """延迟测试"""

    def __init__(self, model, X_test):
        self.model = model
        self.X_test = X_test
        self.results = []

    def measure_single_latency(self, n=1000):
        """单次推理延迟"""
        latencies = []
        for i in range(n):
            sample = self.X_test[i % len(self.X_test):i % len(self.X_test) + 1]
            t0 = time.time()
            self.model.predict(sample)
            latencies.append((time.time() - t0) * 1000)

        stats = {
            'mean': np.mean(latencies),
            'p50': np.percentile(latencies, 50),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99),
            'max': np.max(latencies),
        }

        p99_passed = stats['p99'] < 100  # P99 < 100ms
        self.results.append(('单次推理P99', p99_passed,
                             f'{stats["p99"]:.2f}ms (< 100ms)'))
        return stats, latencies

    def measure_batch_latency(self, batch_sizes=[1, 10, 100, 500]):
        """批量推理延迟"""
        stats = {}
        for bs in batch_sizes:
            batch = self.X_test[:bs]
            t0 = time.time()
            self.model.predict(batch)
            elapsed = (time.time() - t0) * 1000
            per_sample = elapsed / bs
            stats[bs] = {'total_ms': elapsed, 'per_sample_ms': per_sample}

            passed = per_sample < 10  # 每样本<10ms
            self.results.append((f'批量{bs}延迟', passed,
                                 f'{per_sample:.3f}ms/样本'))

        return stats

    def measure_throughput(self, duration_sec=1):
        """吞吐量测试"""
        count = 0
        t0 = time.time()
        while time.time() - t0 < duration_sec:
            idx = count % len(self.X_test)
            self.model.predict(self.X_test[idx:idx+1])
            count += 1

        throughput = count / duration_sec
        passed = throughput >= 100  # 至少100 QPS
        self.results.append(('吞吐量', passed,
                             f'{throughput:.0f} QPS (>= 100)'))
        return throughput

    def run_all(self):
        self.results = []
        stats, latencies = self.measure_single_latency(1000)
        self.measure_batch_latency()
        self.measure_throughput(0.5)

        print(f"\n  延迟测试:")
        for name, passed, detail in self.results:
            status = 'PASS' if passed else 'FAIL'
            print(f"    [{status}] {name}: {detail}")

        return latencies


latency_tests = LatencyTests(model, X_test_s)
latencies = latency_tests.run_all()

# ============================================================
# 5. 公平性测试
# ============================================================
print("\n--- 5. 公平性测试 ---")


class FairnessTests:
    """公平性测试"""

    def __init__(self, model, X_test, y_test, sensitive_features=None):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = model.predict(X_test)
        self.sensitive_features = sensitive_features or {}
        self.results = []

    def add_sensitive_feature(self, name, values):
        """添加敏感特征 (如性别/种族/年龄组)"""
        self.sensitive_features[name] = values

    def test_demographic_parity(self, max_diff=0.1):
        """人口统计平价测试"""
        for feat_name, values in self.sensitive_features.items():
            groups = np.unique(values)
            positive_rates = {}
            for g in groups:
                mask = values == g
                if mask.sum() > 0:
                    positive_rates[g] = self.y_pred[mask].mean()

            rates = list(positive_rates.values())
            max_rate_diff = max(rates) - min(rates)
            passed = max_rate_diff <= max_diff
            self.results.append((f'人口统计平价({feat_name})', passed,
                                 f'最大差异={max_rate_diff:.4f} (阈值={max_diff})'))
        return self.results

    def test_equalized_odds(self, max_diff=0.1):
        """平等机会测试"""
        for feat_name, values in self.sensitive_features.items():
            groups = np.unique(values)
            accuracies = {}
            for g in groups:
                mask = values == g
                if mask.sum() > 0:
                    accuracies[g] = accuracy_score(
                        self.y_test[mask], self.y_pred[mask]
                    )

            accs = list(accuracies.values())
            max_acc_diff = max(accs) - min(accs)
            passed = max_acc_diff <= max_diff
            self.results.append((f'平等机会({feat_name})', passed,
                                 f'最大准确率差异={max_acc_diff:.4f}'))
        return self.results

    def run_all(self):
        self.results = []

        # 模拟敏感特征
        np.random.seed(42)
        age_group = pd.cut(
            [data.data[i, 0] for i in range(len(self.y_test))],  # 用第一个特征模拟
            bins=[0, np.percentile(data.data[:, 0], 33),
                  np.percentile(data.data[:, 0], 66), 100],
            labels=['低', '中', '高']
        ).astype(str)
        # 用随机值模拟(因为breast_cancer没有真正的敏感特征)
        simulated_group = np.random.choice(['A', 'B'], len(self.y_test))
        self.add_sensitive_feature('simulated_group', simulated_group)

        self.test_demographic_parity(max_diff=0.15)
        self.test_equalized_odds(max_diff=0.15)

        print(f"\n  公平性测试:")
        for name, passed, detail in self.results:
            status = 'PASS' if passed else 'FAIL'
            print(f"    [{status}] {name}: {detail}")


fairness_tests = FairnessTests(model, X_test_s, y_test)
fairness_tests.run_all()

# ============================================================
# 6. 鲁棒性测试
# ============================================================
print("\n--- 6. 鲁棒性测试 ---")


class RobustnessTests:
    """鲁棒性测试"""

    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.baseline_acc = accuracy_score(y_test, model.predict(X_test))
        self.results = []

    def test_noise_robustness(self, noise_levels=[0.01, 0.05, 0.1, 0.2]):
        """噪声鲁棒性"""
        for noise in noise_levels:
            X_noisy = self.X_test + np.random.randn(*self.X_test.shape) * noise
            y_pred_noisy = self.model.predict(X_noisy)
            acc = accuracy_score(self.y_test, y_pred_noisy)
            drop = self.baseline_acc - acc
            passed = drop < 0.1
            self.results.append((f'噪声{noise:.0%}', passed,
                                 f'acc={acc:.4f}, 下降={drop:.4f}'))

    def test_missing_features(self, drop_ratios=[0.1, 0.2, 0.3]):
        """缺失特征鲁棒性"""
        for ratio in drop_ratios:
            X_missing = self.X_test.copy()
            n_features = X_missing.shape[1]
            n_drop = int(n_features * ratio)
            drop_cols = np.random.choice(n_features, n_drop, replace=False)
            X_missing[:, drop_cols] = 0  # 用0填充

            y_pred = self.model.predict(X_missing)
            acc = accuracy_score(self.y_test, y_pred)
            drop = self.baseline_acc - acc
            passed = drop < 0.15
            self.results.append((f'缺失{ratio:.0%}特征', passed,
                                 f'acc={acc:.4f}, 下降={drop:.4f}'))

    def test_feature_permutation(self):
        """特征排列测试"""
        # 随机排列一个特征
        for i in range(min(5, self.X_test.shape[1])):
            X_perm = self.X_test.copy()
            X_perm[:, i] = np.random.permutation(X_perm[:, i])
            y_pred = self.model.predict(X_perm)
            acc = accuracy_score(self.y_test, y_pred)
            drop = self.baseline_acc - acc
            self.results.append((f'排列特征{i}', drop > 0,
                                 f'acc={acc:.4f}, 影响={drop:+.4f}'))

    def test_determinism(self, n_runs=10):
        """确定性测试"""
        predictions = []
        for _ in range(n_runs):
            pred = self.model.predict(self.X_test)
            predictions.append(pred)

        all_same = all(np.array_equal(predictions[0], p) for p in predictions)
        self.results.append(('确定性', all_same,
                             f'{n_runs}次预测{"全部一致" if all_same else "不一致"}'))

    def run_all(self):
        self.results = []
        self.test_noise_robustness()
        self.test_missing_features()
        self.test_feature_permutation()
        self.test_determinism()

        print(f"\n  鲁棒性测试 (基线准确率={self.baseline_acc:.4f}):")
        for name, passed, detail in self.results:
            status = 'PASS' if passed else 'FAIL'
            print(f"    [{status}] {name}: {detail}")


robustness_tests = RobustnessTests(model, X_test_s, y_test)
robustness_tests.run_all()

# ============================================================
# 7. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 左上: 延迟分布
ax = axes[0, 0]
ax.hist(latencies, bins=50, color='#2196F3', alpha=0.7, edgecolor='white')
ax.axvline(np.percentile(latencies, 50), color='green', linestyle='--', label='P50')
ax.axvline(np.percentile(latencies, 95), color='orange', linestyle='--', label='P95')
ax.axvline(np.percentile(latencies, 99), color='red', linestyle='--', label='P99')
ax.set_xlabel('延迟 (ms)')
ax.set_ylabel('频次')
ax.set_title('推理延迟分布')
ax.legend()

# 右上: 噪声鲁棒性
ax = axes[0, 1]
noise_results = [(n, d) for n, p, d in robustness_tests.results if n.startswith('噪声')]
if noise_results:
    labels = [n for n, d in noise_results]
    drops = [float(d.split('下降=')[1]) for n, d in noise_results]
    colors = ['#4CAF50' if d < 0.1 else '#F44336' for d in drops]
    ax.bar(labels, drops, color=colors)
    ax.axhline(0.1, color='red', linestyle='--', label='可接受阈值')
    ax.set_ylabel('准确率下降')
    ax.set_title('噪声鲁棒性测试')
    ax.legend()
    ax.tick_params(axis='x', rotation=45)

# 左下: 所有测试汇总
ax = axes[1, 0]
all_results = (quality_tests.results + latency_tests.results +
               fairness_tests.results + robustness_tests.results)
test_types = []
for name, passed, _ in all_results:
    if any(k in name for k in ['准确率', 'F1', '平衡', '回归', '边界']):
        test_types.append('质量')
    elif any(k in name for k in ['延迟', '吞吐', 'P99']):
        test_types.append('延迟')
    elif any(k in name for k in ['平价', '机会']):
        test_types.append('公平性')
    else:
        test_types.append('鲁棒性')

from collections import Counter
type_counts = Counter()
type_passed = Counter()
for (name, passed, _), t in zip(all_results, test_types):
    type_counts[t] += 1
    if passed:
        type_passed[t] += 1

categories = list(type_counts.keys())
passed_counts = [type_passed[c] for c in categories]
failed_counts = [type_counts[c] - type_passed[c] for c in categories]
x = np.arange(len(categories))
ax.bar(x, passed_counts, label='通过', color='#4CAF50')
ax.bar(x, failed_counts, bottom=passed_counts, label='失败', color='#F44336')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylabel('测试数')
ax.set_title('测试结果汇总')
ax.legend()

# 右下: 测试通过率
ax = axes[1, 1]
total_tests = len(all_results)
total_passed = sum(1 for _, p, _ in all_results if p)
pass_rate = total_passed / total_tests if total_tests > 0 else 0

ax.pie([total_passed, total_tests - total_passed],
       labels=[f'通过 ({total_passed})', f'失败 ({total_tests - total_passed})'],
       colors=['#4CAF50', '#F44336'],
       autopct='%1.1f%%', startangle=90)
ax.set_title(f'总测试通过率: {pass_rate:.1%}')

plt.suptitle('模型测试策略报告', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W27/d4_model_testing.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存: d4_model_testing.png")

print("\n完成! 模型测试策略要点:")
print("  1. 预测质量: 准确率/F1阈值, 回归测试, 边界case")
print("  2. 延迟性能: P50/P95/P99, 吞吐量, 批量延迟")
print("  3. 公平性: 人口统计平价, 平等机会")
print("  4. 鲁棒性: 噪声/缺失/排列/确定性")
print("  5. 持续测试: 每次模型更新都运行完整测试")
