"""
W27-D1 模型监控
================
模型监控维度(性能/数据/漂移), 监控指标定义, 告警策略, 监控仪表盘设计
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import load_breast_cancer

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W27-D1 模型监控")
print("=" * 60)

# ============================================================
# 1. 模型监控概述
# ============================================================
print("\n--- 1. 模型监控概述 ---")
print("""
模型监控是MLOps中确保模型持续有效运行的关键环节:

  监控维度:
  ┌─────────────┬──────────────────────────────────────┐
  │ 维度        │ 具体指标                             │
  ├─────────────┼──────────────────────────────────────┤
  │ 模型性能    │ 准确率, F1, AUC, 损失值             │
  │ 数据质量    │ 空值率, 分布变化, 异常值比例         │
  │ 数据漂移    │ PSI, KS统计量, 特征分布变化          │
  │ 概念漂移    │ 标签分布变化, 性能趋势               │
  │ 运行性能    │ 延迟(P50/P95/P99), 吞吐量, 错误率   │
  │ 资源使用    │ CPU, 内存, GPU利用率                 │
  │ 公平性      │ 不同群体的性能差异                   │
  └─────────────┴──────────────────────────────────────┘

  监控挑战:
    - 延迟标签: 真实标签需要时间才能获取
    - 基线漂移: 什么是"正常"行为?
    - 告警疲劳: 太多告警导致忽视
""")

# ============================================================
# 2. 监控指标定义
# ============================================================
print("\n--- 2. 监控指标定义 ---")


class ModelMonitor:
    """模型监控器"""

    def __init__(self, model_name, thresholds=None):
        self.model_name = model_name
        self.thresholds = thresholds or {
            'accuracy_min': 0.85,
            'f1_min': 0.80,
            'drift_psi_max': 0.2,
            'latency_p99_max': 200,  # ms
            'error_rate_max': 0.01,
            'null_rate_max': 0.05,
        }
        self.metrics_history = []
        self.alerts = []

    def compute_metrics(self, y_true, y_pred, predictions_proba=None, latency_ms=None):
        """计算监控指标"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'sample_count': len(y_true),
            'positive_rate': y_pred.mean(),
        }

        if latency_ms is not None:
            metrics['latency_p50'] = np.percentile(latency_ms, 50)
            metrics['latency_p95'] = np.percentile(latency_ms, 95)
            metrics['latency_p99'] = np.percentile(latency_ms, 99)

        self.metrics_history.append(metrics)
        return metrics

    def check_thresholds(self, metrics):
        """检查阈值并生成告警"""
        alerts = []

        if metrics['accuracy'] < self.thresholds['accuracy_min']:
            alerts.append({
                'level': 'CRITICAL',
                'metric': 'accuracy',
                'value': metrics['accuracy'],
                'threshold': self.thresholds['accuracy_min'],
                'message': f"准确率 {metrics['accuracy']:.4f} 低于阈值 {self.thresholds['accuracy_min']}",
            })

        if metrics['f1_score'] < self.thresholds['f1_min']:
            alerts.append({
                'level': 'WARNING',
                'metric': 'f1_score',
                'value': metrics['f1_score'],
                'threshold': self.thresholds['f1_min'],
                'message': f"F1 {metrics['f1_score']:.4f} 低于阈值 {self.thresholds['f1_min']}",
            })

        if 'latency_p99' in metrics and metrics['latency_p99'] > self.thresholds['latency_p99_max']:
            alerts.append({
                'level': 'WARNING',
                'metric': 'latency_p99',
                'value': metrics['latency_p99'],
                'threshold': self.thresholds['latency_p99_max'],
                'message': f"P99延迟 {metrics['latency_p99']:.1f}ms 超过阈值 {self.thresholds['latency_p99_max']}ms",
            })

        self.alerts.extend(alerts)
        return alerts

    def detect_trend(self, metric_name, window=7):
        """检测指标趋势"""
        if len(self.metrics_history) < window:
            return 'insufficient_data', 0

        recent = [m.get(metric_name) for m in self.metrics_history[-window:]
                  if metric_name in m]
        if len(recent) < window:
            return 'insufficient_data', 0

        # 线性回归斜率
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]

        if slope < -0.005:
            return 'declining', slope
        elif slope > 0.005:
            return 'improving', slope
        else:
            return 'stable', slope

    def health_check(self):
        """健康检查"""
        if not self.metrics_history:
            return {'status': 'NO_DATA', 'details': '无监控数据'}

        latest = self.metrics_history[-1]
        alerts = self.check_thresholds(latest)
        critical = [a for a in alerts if a['level'] == 'CRITICAL']

        if critical:
            status = 'UNHEALTHY'
        elif alerts:
            status = 'DEGRADED'
        else:
            status = 'HEALTHY'

        return {
            'status': status,
            'latest_metrics': latest,
            'alert_count': len(alerts),
            'critical_count': len(critical),
        }

    def print_dashboard(self):
        """打印监控仪表盘"""
        health = self.health_check()
        status_colors = {
            'HEALTHY': '绿色', 'DEGRADED': '黄色',
            'UNHEALTHY': '红色', 'NO_DATA': '灰色',
        }

        print(f"\n  ┌─────────────────────────────────────────────────┐")
        print(f"  │ 模型监控仪表盘: {self.model_name:<30}│")
        print(f"  │ 状态: {health['status']:<44}│")
        print(f"  ├─────────────────────────────────────────────────┤")

        if self.metrics_history:
            latest = self.metrics_history[-1]
            for key, value in latest.items():
                if isinstance(value, float):
                    print(f"  │   {key:<20} {value:>20.4f}          │")
                else:
                    print(f"  │   {key:<20} {str(value):>20}          │")

        print(f"  ├─────────────────────────────────────────────────┤")
        print(f"  │ 告警: {health['alert_count']} 个 ({health.get('critical_count', 0)} 严重){' ' * 25}│")

        # 趋势
        for metric in ['accuracy', 'f1_score']:
            trend, slope = self.detect_trend(metric)
            arrow = '↑' if trend == 'improving' else '↓' if trend == 'declining' else '→'
            print(f"  │   {metric} 趋势: {arrow} {trend} (slope={slope:.4f}){' ' * (15 - len(trend))}│")

        print(f"  └─────────────────────────────────────────────────┘")


# ============================================================
# 3. 模拟模型监控
# ============================================================
print("\n--- 3. 模拟模型监控 ---")

# 训练模型
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 创建监控器
monitor = ModelMonitor('breast_cancer_classifier_v1')

# 模拟30天的监控数据
np.random.seed(42)
for day in range(30):
    # 模拟性能逐渐下降 (概念漂移)
    noise = np.random.randn(len(X_test)) * (1 + day * 0.02)

    # 随着时间推移, 引入更多预测错误
    y_pred = model.predict(X_test)
    flip_prob = min(0.01 * day, 0.15)
    flip_mask = np.random.random(len(y_pred)) < flip_prob
    y_pred_monitored = y_pred.copy()
    y_pred_monitored[flip_mask] = 1 - y_pred_monitored[flip_mask]

    # 模拟延迟
    latencies = np.random.exponential(5 + day * 0.3, len(X_test))

    metrics = monitor.compute_metrics(
        y_test, y_pred_monitored,
        latency_ms=latencies
    )
    alerts = monitor.check_thresholds(metrics)

    if day % 5 == 0 or alerts:
        acc = metrics['accuracy']
        f1 = metrics['f1_score']
        print(f"  Day {day:2d}: acc={acc:.4f}, f1={f1:.4f}, "
              f"P99={metrics.get('latency_p99', 0):.1f}ms"
              f"{' [ALERT]' if alerts else ''}")

# 仪表盘
monitor.print_dashboard()

# ============================================================
# 4. 告警策略
# ============================================================
print("\n--- 4. 告警策略 ---")
print("""
  告警级别:
    - INFO:     信息通知, 无需行动
    - WARNING:  需要关注, 可能需要行动
    - CRITICAL: 必须立即行动

  告警策略设计:
    1. 阈值告警:   指标超过预设阈值
    2. 趋势告警:   指标持续下降N天
    3. 异常告警:   指标突然跳变
    4. 组合告警:   多个指标同时异常

  防止告警疲劳:
    - 合理设置阈值 (不要太敏感)
    - 告警聚合 (短时间内同类告警合并)
    - 告警升级 (持续异常逐步提升级别)
    - 静默期 (处理后暂时静默)

  通知渠道:
    - Email, Slack, 钉钉, 企业微信
    - PagerDuty (值班系统)
    - Grafana/自定义仪表盘
""")

# 告警汇总
if monitor.alerts:
    print(f"\n  告警汇总 ({len(monitor.alerts)} 条):")
    alert_counts = {}
    for alert in monitor.alerts:
        key = f"{alert['level']}:{alert['metric']}"
        alert_counts[key] = alert_counts.get(key, 0) + 1

    for key, count in sorted(alert_counts.items(), key=lambda x: -x[1]):
        print(f"    {key}: {count} 次")

# ============================================================
# 5. 可视化 - 监控仪表盘
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 左上: 准确率趋势
ax = axes[0, 0]
accs = [m['accuracy'] for m in monitor.metrics_history]
days = range(len(accs))
ax.plot(days, accs, 'b-', linewidth=2)
ax.axhline(monitor.thresholds['accuracy_min'], color='red', linestyle='--',
           label=f'阈值 ({monitor.thresholds["accuracy_min"]})')
ax.fill_between(days, monitor.thresholds['accuracy_min'], min(accs) - 0.05,
                alpha=0.1, color='red')
ax.set_xlabel('天数')
ax.set_ylabel('准确率')
ax.set_title('模型准确率趋势')
ax.legend()
ax.grid(True, alpha=0.3)

# 右上: F1趋势
ax = axes[0, 1]
f1s = [m['f1_score'] for m in monitor.metrics_history]
ax.plot(days, f1s, 'g-', linewidth=2)
ax.axhline(monitor.thresholds['f1_min'], color='red', linestyle='--',
           label=f'阈值 ({monitor.thresholds["f1_min"]})')
ax.set_xlabel('天数')
ax.set_ylabel('F1 Score')
ax.set_title('F1 Score 趋势')
ax.legend()
ax.grid(True, alpha=0.3)

# 左下: 延迟趋势
ax = axes[1, 0]
p50s = [m.get('latency_p50', 0) for m in monitor.metrics_history]
p95s = [m.get('latency_p95', 0) for m in monitor.metrics_history]
p99s = [m.get('latency_p99', 0) for m in monitor.metrics_history]
ax.plot(days, p50s, label='P50', color='#4CAF50')
ax.plot(days, p95s, label='P95', color='#FF9800')
ax.plot(days, p99s, label='P99', color='#F44336')
ax.axhline(monitor.thresholds['latency_p99_max'], color='red', linestyle='--',
           alpha=0.5, label='P99阈值')
ax.set_xlabel('天数')
ax.set_ylabel('延迟 (ms)')
ax.set_title('推理延迟趋势')
ax.legend()
ax.grid(True, alpha=0.3)

# 右下: 正例率变化 (数据分布监控)
ax = axes[1, 1]
pos_rates = [m['positive_rate'] for m in monitor.metrics_history]
ax.plot(days, pos_rates, 'purple', linewidth=2)
baseline_rate = pos_rates[0]
ax.axhline(baseline_rate, color='green', linestyle='--', alpha=0.5, label=f'基线 ({baseline_rate:.3f})')
ax.axhline(baseline_rate * 1.2, color='orange', linestyle='--', alpha=0.5, label='+20%边界')
ax.axhline(baseline_rate * 0.8, color='orange', linestyle='--', alpha=0.5, label='-20%边界')
ax.set_xlabel('天数')
ax.set_ylabel('正例率')
ax.set_title('预测正例率变化')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.suptitle('模型监控仪表盘', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W27/d1_model_monitoring.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存: d1_model_monitoring.png")

print("\n完成! 模型监控要点:")
print("  1. 多维度监控: 性能、数据、漂移、延迟")
print("  2. 阈值告警: 设置合理阈值, 及时发现问题")
print("  3. 趋势检测: 关注长期趋势, 而非单次波动")
print("  4. 告警策略: 防止告警疲劳, 合理升级")
print("  5. 仪表盘: 可视化关键指标, 便于快速判断")
