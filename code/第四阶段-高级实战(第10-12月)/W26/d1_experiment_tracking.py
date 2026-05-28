"""
W26-D1 实验追踪
================
实验追踪概念, MLflow基础(Tracking/Projects/Models), 参数和指标记录, 实验对比
"""

import os
import json
import time
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W26-D1 实验追踪")
print("=" * 60)

# ============================================================
# 1. MLflow 概念
# ============================================================
print("\n--- 1. MLflow 概念 ---")
print("""
MLflow 是一个开源的ML实验管理平台:

  核心组件:
    - Tracking:   实验追踪(参数/指标/Artifact)
    - Projects:   可复现的打包格式
    - Models:     模型打包和部署
    - Registry:   模型版本管理

  Tracking 核心概念:
    - Experiment:  实验(一组相关的Run)
    - Run:         一次运行(记录参数/指标/输出)
    - Parameter:   输入参数(如 learning_rate=0.01)
    - Metric:      评估指标(如 accuracy=0.95)
    - Artifact:    输出文件(模型/图表/日志)

  安装: pip install mlflow
  启动UI: mlflow ui
""")

# ============================================================
# 2. 简化版实验追踪系统
# ============================================================
print("\n--- 2. 简化版实验追踪系统 ---")


class ExperimentTracker:
    """简化版实验追踪器"""

    def __init__(self, experiment_name='default'):
        self.experiment_name = experiment_name
        self.runs = []
        self.current_run = None

    def start_run(self, run_name=''):
        """开始一次运行"""
        self.current_run = {
            'run_id': hashlib.md5(f"{time.time()}".encode()).hexdigest()[:8],
            'run_name': run_name,
            'experiment': self.experiment_name,
            'start_time': datetime.now().isoformat(),
            'params': {},
            'metrics': {},
            'tags': {},
            'artifacts': [],
            'status': 'RUNNING',
        }
        print(f"  开始运行: {run_name} (id={self.current_run['run_id']})")
        return self

    def log_param(self, key, value):
        """记录参数"""
        if self.current_run:
            self.current_run['params'][key] = value

    def log_params(self, params_dict):
        """批量记录参数"""
        for k, v in params_dict.items():
            self.log_param(k, v)

    def log_metric(self, key, value):
        """记录指标"""
        if self.current_run:
            self.current_run['metrics'][key] = value

    def log_metrics(self, metrics_dict):
        """批量记录指标"""
        for k, v in metrics_dict.items():
            self.log_metric(k, v)

    def log_artifact(self, path):
        """记录Artifact"""
        if self.current_run:
            self.current_run['artifacts'].append(path)

    def set_tag(self, key, value):
        """设置标签"""
        if self.current_run:
            self.current_run['tags'][key] = value

    def end_run(self, status='FINISHED'):
        """结束运行"""
        if self.current_run:
            self.current_run['end_time'] = datetime.now().isoformat()
            self.current_run['status'] = status
            self.runs.append(self.current_run)
            print(f"  结束运行: {self.current_run['run_name']} ({status})")
            self.current_run = None

    def compare_runs(self):
        """对比所有运行"""
        if not self.runs:
            print("  无运行记录")
            return

        print(f"\n  实验对比 ({len(self.runs)} 次运行):")
        print(f"  {'运行名':<25} {'状态':<10} {'参数':<30} {'指标'}")
        print("  " + "-" * 80)
        for run in self.runs:
            params_str = ', '.join(f'{k}={v}' for k, v in list(run['params'].items())[:3])
            metrics_str = ', '.join(f'{k}={v:.4f}' for k, v in run['metrics'].items())
            print(f"  {run['run_name']:<25} {run['status']:<10} {params_str:<30} {metrics_str}")

    def get_best_run(self, metric='accuracy', mode='max'):
        """获取最佳运行"""
        if not self.runs:
            return None
        if mode == 'max':
            return max(self.runs, key=lambda r: r['metrics'].get(metric, 0))
        return min(self.runs, key=lambda r: r['metrics'].get(metric, float('inf')))

    def to_dataframe(self):
        """转为DataFrame"""
        rows = []
        for run in self.runs:
            row = {
                'run_id': run['run_id'],
                'run_name': run['run_name'],
                'status': run['status'],
            }
            row.update({f'param_{k}': v for k, v in run['params'].items()})
            row.update({f'metric_{k}': v for k, v in run['metrics'].items()})
            rows.append(row)
        return pd.DataFrame(rows)


# ============================================================
# 3. 使用实验追踪器进行多模型对比
# ============================================================
print("\n--- 3. 多模型实验对比 ---")

# 加载数据
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

tracker = ExperimentTracker('breast_cancer_classification')

# --- 实验1: Logistic Regression ---
tracker.start_run('LogisticRegression')
tracker.log_params({
    'model_type': 'LogisticRegression',
    'C': 1.0,
    'max_iter': 1000,
    'scaler': 'StandardScaler',
})
lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
tracker.log_metrics({
    'accuracy': accuracy_score(y_test, y_pred),
    'f1': f1_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
})
tracker.set_tag('model_family', 'linear')
tracker.end_run()

# --- 实验2: Random Forest ---
tracker.start_run('RandomForest')
tracker.log_params({
    'model_type': 'RandomForest',
    'n_estimators': 100,
    'max_depth': 10,
    'scaler': 'StandardScaler',
})
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
tracker.log_metrics({
    'accuracy': accuracy_score(y_test, y_pred),
    'f1': f1_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
})
tracker.set_tag('model_family', 'ensemble')
tracker.end_run()

# --- 实验3: Gradient Boosting ---
tracker.start_run('GradientBoosting')
tracker.log_params({
    'model_type': 'GradientBoosting',
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 5,
    'scaler': 'StandardScaler',
})
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)
tracker.log_metrics({
    'accuracy': accuracy_score(y_test, y_pred),
    'f1': f1_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
})
tracker.set_tag('model_family', 'ensemble')
tracker.end_run()

# --- 实验4: SVM ---
tracker.start_run('SVM_RBF')
tracker.log_params({
    'model_type': 'SVC',
    'C': 1.0,
    'kernel': 'rbf',
    'scaler': 'StandardScaler',
})
svc = SVC(C=1.0, kernel='rbf', random_state=42)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
tracker.log_metrics({
    'accuracy': accuracy_score(y_test, y_pred),
    'f1': f1_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
})
tracker.set_tag('model_family', 'kernel')
tracker.end_run()

# --- 实验5: Random Forest (更多树) ---
tracker.start_run('RandomForest_200')
tracker.log_params({
    'model_type': 'RandomForest',
    'n_estimators': 200,
    'max_depth': 15,
    'scaler': 'StandardScaler',
})
rf2 = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
rf2.fit(X_train, y_train)
y_pred = rf2.predict(X_test)
tracker.log_metrics({
    'accuracy': accuracy_score(y_test, y_pred),
    'f1': f1_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
})
tracker.set_tag('model_family', 'ensemble')
tracker.end_run()

# 对比
tracker.compare_runs()

# 最佳模型
best = tracker.get_best_run('accuracy')
print(f"\n  最佳模型: {best['run_name']} (accuracy={best['metrics']['accuracy']:.4f})")

# ============================================================
# 4. 参数敏���性分析
# ============================================================
print("\n--- 4. 参数敏感性分析 ---")

param_tracker = ExperimentTracker('rf_param_sensitivity')

for n_est in [10, 50, 100, 150, 200]:
    for max_depth in [3, 5, 10, 15, None]:
        tracker.start_run(f'RF_n{n_est}_d{max_depth}')
        tracker.log_params({
            'n_estimators': n_est,
            'max_depth': max_depth if max_depth else -1,
        })
        rf = RandomForestClassifier(n_estimators=n_est, max_depth=max_depth, random_state=42)
        scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')
        tracker.log_metrics({
            'cv_accuracy_mean': scores.mean(),
            'cv_accuracy_std': scores.std(),
        })
        tracker.end_run()
        param_tracker.runs.append(tracker.runs[-1])

# 敏感性分析
sens_df = param_tracker.to_dataframe()
print(f"\n  完成了 {len(sens_df)} 次参数组合实验")

# ============================================================
# 5. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 左上: 模型对比
ax = axes[0, 0]
runs_df = tracker.to_dataframe()
models = runs_df['run_name'].tolist()
accuracies = runs_df['metric_accuracy'].tolist()
f1_scores = runs_df['metric_f1'].tolist()

x = np.arange(len(models))
width = 0.35
ax.bar(x - width/2, accuracies, width, label='Accuracy', color='#2196F3')
ax.bar(x + width/2, f1_scores, width, label='F1 Score', color='#4CAF50')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=30, ha='right', fontsize=8)
ax.set_ylabel('分数')
ax.set_title('模型性能对比')
ax.legend()
ax.set_ylim(0.9, 1.0)

# 右上: 参数热力图 (n_estimators vs max_depth)
ax = axes[0, 1]
n_ests = sorted(sens_df['param_n_estimators'].unique())
depths = sorted(sens_df['param_max_depth'].unique())
heatmap_data = np.zeros((len(n_ests), len(depths)))
for _, row in sens_df.iterrows():
    i = n_ests.index(row['param_n_estimators'])
    j = depths.index(row['param_max_depth'])
    heatmap_data[i, j] = row['metric_cv_accuracy_mean']

im = ax.imshow(heatmap_data, cmap='YlGn', aspect='auto')
ax.set_xticks(range(len(depths)))
ax.set_xticklabels([str(d) for d in depths])
ax.set_yticks(range(len(n_ests)))
ax.set_yticklabels([str(n) for n in n_ests])
ax.set_xlabel('max_depth')
ax.set_ylabel('n_estimators')
ax.set_title('参数组合准确率热力图')
for i in range(len(n_ests)):
    for j in range(len(depths)):
        ax.text(j, i, f'{heatmap_data[i,j]:.3f}', ha='center', va='center', fontsize=8)
plt.colorbar(im, ax=ax)

# 左下: 学习曲线 (n_estimators对性能的影响)
ax = axes[1, 0]
for depth in [3, 5, 10]:
    mask = sens_df['param_max_depth'] == depth
    subset = sens_df[mask].sort_values('param_n_estimators')
    ax.plot(subset['param_n_estimators'], subset['metric_cv_accuracy_mean'],
            'o-', label=f'max_depth={depth}')
ax.set_xlabel('n_estimators')
ax.set_ylabel('CV准确率')
ax.set_title('n_estimators对性能的影响')
ax.legend()
ax.grid(True, alpha=0.3)

# 右下: 运行时间线
ax = axes[1, 1]
metrics_data = {}
for run in tracker.runs:
    for metric_name, value in run['metrics'].items():
        if metric_name not in metrics_data:
            metrics_data[metric_name] = []
        metrics_data[metric_name].append(value)

for metric_name, values in metrics_data.items():
    ax.plot(range(1, len(values) + 1), values, 'o-', label=metric_name, markersize=5)

ax.set_xticks(range(1, len(tracker.runs) + 1))
ax.set_xticklabels([r['run_name'] for r in tracker.runs], rotation=30, ha='right', fontsize=7)
ax.set_ylabel('指标值')
ax.set_title('实验指标趋势')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W26/d1_experiment_tracking.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存: d1_experiment_tracking.png")

print("\n完成! 实验追踪要点:")
print("  1. MLflow Tracking: 记录参数、指标、Artifact")
print("  2. Experiment: 一组相关Run的集合")
print("  3. Run: 一次完整的训练运行")
print("  4. 实验对比: 多模型/多参数的横向比较")
print("  5. 参数敏感性: 理解超参数对性能的影响")
