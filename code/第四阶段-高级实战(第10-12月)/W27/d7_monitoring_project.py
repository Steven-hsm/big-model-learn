"""
W27-D7 监控项目
================
构建完整的模型监控系统, 漂移检测+告警+自动重训练Pipeline
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from scipy import stats as scipy_stats

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W27-D7 监控项目: 完整模型监控系统")
print("=" * 60)

# ============================================================
# 项目概述
# ============================================================
print("""
项目: 构建完整的模型监控系统

  系统架构:
  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │ 数据采集 │->│ 漂移检测 │->│ 告警系统 │->│ 自动重训 │
  └──────────┘  └──────────┘  └──────────┘  └──────────┘
       │              │              │              │
   特征分布      PSI/KS检验     规则引擎       新模型训练
   预测记录      概念漂移       多级告警       测试+发布
""")

# ============================================================
# 1. 核心监控组件
# ============================================================

class DriftDetector:
    """漂移检测器"""

    @staticmethod
    def psi(reference, current, bins=10):
        all_data = np.concatenate([reference, current])
        bin_edges = np.percentile(all_data, np.linspace(0, 100, bins + 1))
        bin_edges[0], bin_edges[-1] = -np.inf, np.inf
        ref_hist = np.clip(np.histogram(reference, bins=bin_edges)[0] / len(reference), 1e-6, None)
        cur_hist = np.clip(np.histogram(current, bins=bin_edges)[0] / len(current), 1e-6, None)
        return float(np.sum((cur_hist - ref_hist) * np.log(cur_hist / ref_hist)))

    @staticmethod
    def ks_test(reference, current):
        stat, p = scipy_stats.ks_2samp(reference, current)
        return {'statistic': stat, 'p_value': p, 'significant': p < 0.05}


class AlertSystem:
    """告警系统"""

    def __init__(self, rules):
        self.rules = rules  # list of (name, check_fn, level)
        self.alert_history = []
        self.active_alerts = []

    def evaluate(self, context):
        """评估所有规则"""
        new_alerts = []
        for name, check_fn, level in self.rules:
            triggered, message = check_fn(context)
            if triggered:
                alert = {
                    'rule': name, 'level': level,
                    'message': message, 'timestamp': datetime.now().isoformat(),
                }
                new_alerts.append(alert)
                self.alert_history.append(alert)

        self.active_alerts = new_alerts
        return new_alerts

    def suppress(self, cooldown_minutes=30):
        """告警抑制"""
        if not self.alert_history:
            return
        cutoff = datetime.now() - timedelta(minutes=cooldown_minutes)
        # 简化: 只保留最近告警
        recent = [a for a in self.alert_history
                  if a['timestamp'] > cutoff.isoformat()]
        self.alert_history = recent


class AutoRetrainer:
    """自动重训练器"""

    def __init__(self, retrain_threshold=0.85):
        self.retrain_threshold = retrain_threshold
        self.retrain_history = []

    def should_retrain(self, metrics_history):
        """判断是否需要重训练"""
        if len(metrics_history) < 5:
            return False
        recent = metrics_history[-5:]
        avg_acc = np.mean([m['accuracy'] for m in recent])
        declining = all(recent[i]['accuracy'] >= recent[i+1]['accuracy']
                        for i in range(len(recent)-1))
        return avg_acc < self.retrain_threshold and declining

    def retrain(self, X_train, y_train, config=None):
        """执行重训练"""
        config = config or {'n_estimators': 150, 'max_depth': 12}
        model = RandomForestClassifier(random_state=42, **config)
        model.fit(X_train, y_train)

        record = {
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'model_type': 'RandomForest',
        }
        self.retrain_history.append(record)
        return model


# ============================================================
# 2. 完整监控系统
# ============================================================


class ModelMonitoringSystem:
    """完整模型监控系统"""

    def __init__(self, model, X_ref, feature_names, thresholds=None):
        self.model = model
        self.X_ref = X_ref
        self.feature_names = feature_names
        self.thresholds = thresholds or {
            'accuracy_min': 0.85,
            'psi_max': 0.2,
            'f1_min': 0.80,
            'latency_p99_max': 50,
        }

        self.drift_detector = DriftDetector()
        self.auto_retrainer = AutoRetrainer(self.thresholds['accuracy_min'])

        # 告警规则
        self.alert_system = AlertSystem([
            ('准确率下降', self._check_accuracy, 'WARNING'),
            ('数据漂移', self._check_drift, 'WARNING'),
            ('严重性能下降', self._check_critical, 'CRITICAL'),
        ])

        self.daily_metrics = []
        self.daily_drift = []
        self.model_versions = [{'version': 1, 'model': model, 'deployed_at': 'Day 0'}]

    def _check_accuracy(self, context):
        acc = context.get('accuracy', 1.0)
        if acc < self.thresholds['accuracy_min']:
            return True, f'准确率 {acc:.4f} < {self.thresholds["accuracy_min"]}'
        return False, ''

    def _check_drift(self, context):
        max_psi = context.get('max_psi', 0)
        if max_psi > self.thresholds['psi_max']:
            return True, f'最大PSI {max_psi:.4f} > {self.thresholds["psi_max"]}'
        return False, ''

    def _check_critical(self, context):
        acc = context.get('accuracy', 1.0)
        if acc < self.thresholds['accuracy_min'] - 0.1:
            return True, f'严重: 准确率 {acc:.4f}'
        return False, ''

    def daily_check(self, day, X_current, y_true):
        """每日检查"""
        # 1. 性能评估
        y_pred = self.model.predict(X_current)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')

        # 2. 漂移检测
        drift_results = {}
        for i, feat in enumerate(self.feature_names):
            psi_val = self.drift_detector.psi(self.X_ref[:, i], X_current[:, i])
            drift_results[feat] = psi_val

        max_psi = max(drift_results.values())

        # 3. 延迟模拟
        t0 = time.time()
        for _ in range(100):
            self.model.predict(X_current[:1])
        avg_latency = (time.time() - t0) / 100 * 1000

        # 记录指标
        metrics = {
            'day': day,
            'accuracy': acc,
            'f1_score': f1,
            'max_psi': max_psi,
            'avg_latency_ms': avg_latency,
            'drift_details': drift_results,
            'timestamp': datetime.now().isoformat(),
        }
        self.daily_metrics.append(metrics)

        # 4. 告警评估
        context = {'accuracy': acc, 'max_psi': max_psi, 'f1_score': f1}
        alerts = self.alert_system.evaluate(context)

        # 5. 检查是否需要重训练
        retrained = False
        if self.auto_retrainer.should_retrain(self.daily_metrics):
            new_model = self.auto_retrainer.retrain(
                X_current, y_true,
                {'n_estimators': 150, 'max_depth': 12}
            )
            new_acc = accuracy_score(y_true, new_model.predict(X_current))
            if new_acc > acc:
                self.model = new_model
                version = len(self.model_versions) + 1
                self.model_versions.append({
                    'version': version,
                    'model': new_model,
                    'deployed_at': f'Day {day}',
                    'accuracy': new_acc,
                })
                retrained = True

        return {
            'metrics': metrics,
            'alerts': alerts,
            'retrained': retrained,
        }


# ============================================================
# 3. 运行监控系统 (模拟60天)
# ============================================================
print("\n--- 3. 运行监控系统 (模拟60天) ---")

# 准备数据
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 训练初始模型
initial_model = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
initial_model.fit(X_train_s, y_train)

# 创建监控系统
monitoring = ModelMonitoringSystem(
    initial_model, X_train_s, data.feature_names[:5],  # 监控前5个特征
    thresholds={'accuracy_min': 0.90, 'psi_max': 0.15, 'f1_min': 0.85}
)

print(f"  初始模型准确率: {accuracy_score(y_test, initial_model.predict(X_test_s)):.4f}")

# 模拟60天
np.random.seed(42)
retrain_events = []
alert_events = []

for day in range(60):
    # 模拟数据分布逐渐漂移
    drift_factor = day * 0.005  # 逐渐增加漂移
    X_simulated = X_test_s + np.random.randn(*X_test_s.shape) * drift_factor

    # 随着漂移增加, 引入预测错误
    y_pred = monitoring.model.predict(X_simulated)
    flip_prob = min(0.002 * day, 0.12)
    flip_mask = np.random.random(len(y_pred)) < flip_prob
    y_simulated = y_test.copy()
    # 模拟标签也受漂移影响
    y_simulated[flip_mask] = 1 - y_simulated[flip_mask]

    result = monitoring.daily_check(day, X_simulated, y_simulated)

    if result['alerts']:
        for alert in result['alerts']:
            alert_events.append({'day': day, **alert})

    if result['retrained']:
        retrain_events.append({'day': day, 'accuracy': result['metrics']['accuracy']})
        print(f"  Day {day:2d}: [RETRAIN] 准确率={result['metrics']['accuracy']:.4f}")

    if day % 10 == 0:
        m = result['metrics']
        alert_str = f" [{len(result['alerts'])} alerts]" if result['alerts'] else ""
        print(f"  Day {day:2d}: acc={m['accuracy']:.4f}, psi={m['max_psi']:.4f}, "
              f"latency={m['avg_latency_ms']:.2f}ms{alert_str}")

# ============================================================
# 4. 监控报告
# ============================================================
print("\n--- 4. 监控报告 ---")

all_accs = [m['accuracy'] for m in monitoring.daily_metrics]
all_psis = [m['max_psi'] for m in monitoring.daily_metrics]

print(f"\n  === 60天监控报告 ===")
print(f"  平均准确率: {np.mean(all_accs):.4f}")
print(f"  最低准确率: {np.min(all_accs):.4f} (Day {np.argmin(all_accs)})")
print(f"  最大PSI: {np.max(all_psis):.4f}")
print(f"  告警总数: {len(alert_events)}")
print(f"  重训练次数: {len(retrain_events)}")
print(f"  模型版本数: {len(monitoring.model_versions)}")

# 告警统计
if alert_events:
    alert_by_level = {}
    for a in alert_events:
        level = a['level']
        alert_by_level[level] = alert_by_level.get(level, 0) + 1
    print(f"  告警级别分布: {alert_by_level}")

# ============================================================
# 5. 可视化
# ============================================================
fig, axes = plt.subplots(3, 2, figsize=(14, 14))

days = [m['day'] for m in monitoring.daily_metrics]

# 1. 准确率趋势
ax = axes[0, 0]
accs = [m['accuracy'] for m in monitoring.daily_metrics]
ax.plot(days, accs, 'b-', linewidth=1.5)
ax.axhline(monitoring.thresholds['accuracy_min'], color='red', linestyle='--',
           label=f'阈值 ({monitoring.thresholds["accuracy_min"]})')
for re in retrain_events:
    ax.axvline(re['day'], color='green', linestyle='--', alpha=0.7)
    ax.text(re['day'], min(accs), 'R', ha='center', color='green', fontsize=10, fontweight='bold')
ax.set_xlabel('天数')
ax.set_ylabel('准确率')
ax.set_title('模型准确率趋势 (R=重训练)')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. 漂移PSI趋势
ax = axes[0, 1]
psis = [m['max_psi'] for m in monitoring.daily_metrics]
ax.plot(days, psis, 'r-', linewidth=1.5)
ax.axhline(monitoring.thresholds['psi_max'], color='orange', linestyle='--',
           label=f'PSI阈值 ({monitoring.thresholds["psi_max"]})')
ax.fill_between(days, monitoring.thresholds['psi_max'], max(psis) * 1.1,
                where=[p > monitoring.thresholds['psi_max'] for p in psis],
                alpha=0.2, color='red', label='漂移区域')
ax.set_xlabel('天数')
ax.set_ylabel('最大PSI')
ax.set_title('数据漂移趋势')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. F1趋势
ax = axes[1, 0]
f1s = [m['f1_score'] for m in monitoring.daily_metrics]
ax.plot(days, f1s, 'g-', linewidth=1.5)
ax.axhline(monitoring.thresholds['f1_min'], color='red', linestyle='--',
           label=f'F1阈值 ({monitoring.thresholds["f1_min"]})')
ax.set_xlabel('天数')
ax.set_ylabel('F1 Score')
ax.set_title('F1 Score趋势')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. 延迟趋势
ax = axes[1, 1]
latencies = [m['avg_latency_ms'] for m in monitoring.daily_metrics]
ax.plot(days, latencies, 'purple', linewidth=1.5)
ax.set_xlabel('天数')
ax.set_ylabel('延迟 (ms)')
ax.set_title('推理延迟趋势')
ax.grid(True, alpha=0.3)

# 5. 告警时间线
ax = axes[2, 0]
if alert_events:
    alert_days = [a['day'] for a in alert_events]
    alert_levels = [1 if a['level'] == 'WARNING' else 2 for a in alert_events]
    colors = ['#FF9800' if l == 1 else '#F44336' for l in alert_levels]
    ax.scatter(alert_days, alert_levels, c=colors, s=30, alpha=0.6)
    ax.set_yticks([1, 2])
    ax.set_yticklabels(['WARNING', 'CRITICAL'])
    ax.set_xlabel('天数')
    ax.set_title(f'告警时间线 (共{len(alert_events)}条)')
    ax.grid(True, alpha=0.3)
else:
    ax.text(0.5, 0.5, '无告警', ha='center', va='center', transform=ax.transAxes)
    ax.set_title('告警时间线')

# 6. 模型版本时间线
ax = axes[2, 1]
for v in monitoring.model_versions:
    dep_day = int(v['deployed_at'].replace('Day ', '')) if 'Day' in v['deployed_at'] else 0
    ax.barh(v['version'], 60 - dep_day, left=dep_day,
            color='#4CAF50' if v['version'] == len(monitoring.model_versions) else '#BBDEFB',
            edgecolor='#333', height=0.5)
    label = f"v{v['version']} ({v['deployed_at']}"
    if 'accuracy' in v:
        label += f", acc={v['accuracy']:.3f}"
    label += ")"
    ax.text(dep_day + 1, v['version'], label, va='center', fontsize=8)

ax.set_xlabel('天数')
ax.set_ylabel('版本')
ax.set_title('模型版本时间线')
ax.set_yticks([v['version'] for v in monitoring.model_versions])

plt.suptitle('模型监控系统 - 60天运行报告', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W27/d7_monitoring_project.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存: d7_monitoring_project.png")

print("\n" + "=" * 60)
print("监控系统项目总结:")
print(f"  1. 监控天数: 60天")
print(f"  2. 告警次数: {len(alert_events)}")
print(f"  3. 重训练次数: {len(retrain_events)}")
print(f"  4. 模型版本: {len(monitoring.model_versions)}")
print(f"  5. 最终准确率: {all_accs[-1]:.4f}")
print("=" * 60)
