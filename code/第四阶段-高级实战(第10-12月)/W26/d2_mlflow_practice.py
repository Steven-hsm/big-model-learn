"""
W26-D2 MLflow实践
==================
MLflow实践: 自动日志记录, 自定义指标, Artifact管理, 实验搜索和过滤
"""

import os
import json
import time
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W26-D2 MLflow实践")
print("=" * 60)

# ============================================================
# 1. MLflow自动日志记录 (概念)
# ============================================================
print("\n--- 1. MLflow自动日志记录 (概念) ---")
print("""
MLflow自动日志记录 (autolog):

  # scikit-learn 自动记录
  import mlflow
  import mlflow.sklearn

  mlflow.sklearn.autolog()

  with mlflow.start_run():
      model = RandomForestClassifier(n_estimators=100)
      model.fit(X_train, y_train)
      # MLflow自动记录: 参数, 指标, 模型, 混淆矩阵...

  支持的框架:
    - mlflow.sklearn.autolog()
    - mlflow.tensorflow.autolog()
    - mlflow.pytorch.autolog()
    - mlflow.xgboost.autolog()
    - mlflow.lightgbm.autolog()
""")

# ============================================================
# 2. 简化版MLflow实践系统
# ============================================================
print("\n--- 2. 简化版MLflow实践系统 ---")


class MLflowSimulator:
    """MLflow功能模拟器"""

    def __init__(self, tracking_uri='./mlruns'):
        self.tracking_uri = tracking_uri
        self.experiments = {}
        self.current_experiment = None
        self.current_run = None

    def create_experiment(self, name):
        """创建实验"""
        exp_id = hashlib.md5(name.encode()).hexdigest()[:8]
        self.experiments[name] = {
            'experiment_id': exp_id,
            'name': name,
            'runs': [],
            'created_at': datetime.now().isoformat(),
        }
        self.current_experiment = name
        print(f"  创建实验: {name} (id={exp_id})")
        return exp_id

    def start_run(self, run_name=None, tags=None):
        """开始运行"""
        run_id = hashlib.md5(f"{time.time()}{run_name}".encode()).hexdigest()[:12]
        self.current_run = {
            'run_id': run_id,
            'run_name': run_name or f'run_{len(self.experiments.get(self.current_experiment, {}).get("runs", []))}',
            'experiment': self.current_experiment,
            'params': {},
            'metrics': {},
            'metric_history': {},  # metric -> [(step, value)]
            'tags': tags or {},
            'artifacts': [],
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'status': 'RUNNING',
        }
        return self

    def log_param(self, key, value):
        self.current_run['params'][key] = value

    def log_params(self, params):
        self.current_run['params'].update(params)

    def log_metric(self, key, value, step=None):
        self.current_run['metrics'][key] = value
        if key not in self.current_run['metric_history']:
            self.current_run['metric_history'][key] = []
        self.current_run['metric_history'][key].append({
            'step': step or len(self.current_run['metric_history'][key]),
            'value': value,
            'timestamp': datetime.now().isoformat(),
        })

    def log_metrics(self, metrics):
        for k, v in metrics.items():
            self.log_metric(k, v)

    def log_artifact(self, name, artifact_type='file', metadata=None):
        self.current_run['artifacts'].append({
            'name': name,
            'type': artifact_type,
            'metadata': metadata or {},
        })

    def log_model(self, model, name='model'):
        """记录模型信息"""
        model_info = {
            'name': name,
            'type': type(model).__name__,
            'module': type(model).__module__,
            'params': model.get_params(),
        }
        self.log_artifact(name, artifact_type='model', metadata=model_info)

    def end_run(self, status='FINISHED'):
        self.current_run['end_time'] = datetime.now().isoformat()
        self.current_run['status'] = status
        if self.current_experiment and self.current_experiment in self.experiments:
            self.experiments[self.current_experiment]['runs'].append(self.current_run)
        print(f"  运行结束: {self.current_run['run_name']} -> {status}")
        self.current_run = None

    def search_runs(self, experiment_name=None, filter_string=''):
        """搜索运行"""
        exp_name = experiment_name or self.current_experiment
        if exp_name not in self.experiments:
            return []

        runs = self.experiments[exp_name]['runs']
        if not filter_string:
            return runs

        # 简单过滤
        filtered = []
        for run in runs:
            # 支持 "metric.accuracy > 0.9" 风格的过滤
            if '>' in filter_string:
                parts = filter_string.split('>')
                key = parts[0].strip().replace('metric.', '')
                threshold = float(parts[1].strip())
                if key in run['metrics'] and run['metrics'][key] > threshold:
                    filtered.append(run)
            elif '=' in filter_string:
                parts = filter_string.split('=')
                key = parts[0].strip().replace('param.', '')
                value = parts[1].strip()
                if key in run['params'] and str(run['params'][key]) == value:
                    filtered.append(run)
            else:
                filtered.append(run)

        return filtered

    def print_experiment_summary(self, experiment_name=None):
        """打印实验摘要"""
        exp_name = experiment_name or self.current_experiment
        if exp_name not in self.experiments:
            print("  实验不存在")
            return

        exp = self.experiments[exp_name]
        print(f"\n  实验: {exp['name']} ({len(exp['runs'])} 次运行)")
        print(f"  {'运行名':<20} {'状态':<10} {'accuracy':<10} {'f1':<10} {'参数摘要'}")
        print("  " + "-" * 80)

        for run in exp['runs']:
            acc = run['metrics'].get('accuracy', 'N/A')
            f1 = run['metrics'].get('f1', 'N/A')
            params_summary = ', '.join(f'{k}={v}' for k, v in list(run['params'].items())[:3])
            if isinstance(acc, float):
                acc = f'{acc:.4f}'
            if isinstance(f1, float):
                f1 = f'{f1:.4f}'
            print(f"  {run['run_name']:<20} {run['status']:<10} {acc:<10} {f1:<10} {params_summary}")


# ============================================================
# 3. 实践: 多模型对比实验
# ============================================================
print("\n--- 3. 多模型对比实验 ---")

# 加载��据
wine = load_wine()
X, y = wine.data, wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mlflow = MLflowSimulator()
mlflow.create_experiment('wine_classification')

# --- Run 1: Logistic Regression ---
mlflow.start_run('LR_baseline')
mlflow.log_params({'model': 'LogisticRegression', 'C': 1.0, 'max_iter': 1000})
lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42, multi_class='ovr')
lr.fit(X_train_scaled, y_train)
y_pred = lr.predict(X_test_scaled)
mlflow.log_metrics({
    'accuracy': accuracy_score(y_test, y_pred),
    'f1_macro': f1_score(y_test, y_pred, average='macro'),
    'precision_macro': precision_score(y_test, y_pred, average='macro'),
    'recall_macro': recall_score(y_test, y_pred, average='macro'),
})
mlflow.log_model(lr, 'lr_model')
mlflow.log_artifact('confusion_matrix_lr.png', 'image')
mlflow.end_run()

# --- Run 2-6: Random Forest 不同参数 ---
for n_est, max_d in [(50, 5), (100, 10), (100, None), (200, 10), (200, None)]:
    mlflow.start_run(f'RF_n{n_est}_d{max_d}')
    mlflow.log_params({
        'model': 'RandomForest',
        'n_estimators': n_est,
        'max_depth': max_d if max_d else -1,
    })

    rf = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, random_state=42)
    rf.fit(X_train_scaled, y_train)
    y_pred = rf.predict(X_test_scaled)

    # 记录训练集和测试集指标 (检测过拟合)
    train_acc = accuracy_score(y_train, rf.predict(X_train_scaled))
    test_acc = accuracy_score(y_test, y_pred)

    mlflow.log_metrics({
        'accuracy': test_acc,
        'train_accuracy': train_acc,
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'overfit_gap': train_acc - test_acc,
    })

    # CV指标
    cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=5, scoring='accuracy')
    mlflow.log_metric('cv_accuracy', cv_scores.mean())

    mlflow.log_model(rf, 'rf_model')
    mlflow.end_run()

# ============================================================
# 4. 自定义指标: 逐epoch记录
# ============================================================
print("\n--- 4. 自定义指标: 训练曲线记录 ---")

mlflow.start_run('RF_training_curve')
mlflow.log_params({'model': 'RandomForest', 'n_estimators': 200, 'max_depth': 10})

rf_curve = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, warm_start=True)

# 模拟逐步增加训练数据
train_sizes = np.linspace(0.1, 1.0, 10)
for step, frac in enumerate(train_sizes):
    n = int(len(X_train_scaled) * frac)
    rf_step = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf_step.fit(X_train_scaled[:n], y_train[:n])
    train_acc = accuracy_score(y_train[:n], rf_step.predict(X_train_scaled[:n]))
    test_acc = accuracy_score(y_test, rf_step.predict(X_test_scaled))
    mlflow.log_metric('train_accuracy', train_acc, step=step)
    mlflow.log_metric('test_accuracy', test_acc, step=step)

mlflow.log_model(rf_curve, 'rf_curve_model')
mlflow.end_run()

# ============================================================
# 5. 实验搜索和过滤
# ============================================================
print("\n--- 5. 实验搜索和过滤 ---")

# 搜索高准确率运行
print("\n  搜索 accuracy > 0.95 的运行:")
good_runs = mlflow.search_runs(filter_string='metric.accuracy > 0.95')
for run in good_runs:
    print(f"    {run['run_name']}: accuracy={run['metrics']['accuracy']:.4f}")

# 按模型类型过滤
print("\n  搜索 RandomForest 模型:")
rf_runs = mlflow.search_runs(filter_string='param.model=RandomForest')
for run in rf_runs:
    print(f"    {run['run_name']}: accuracy={run['metrics']['accuracy']:.4f}, "
          f"overfit_gap={run['metrics'].get('overfit_gap', 'N/A')}")

# 实验摘要
mlflow.print_experiment_summary()

# ============================================================
# 6. Artifact管理
# ============================================================
print("\n--- 6. Artifact管理 ---")

print("""
  Artifact类型:
    - 模型文件:   model.pkl, model.onnx
    - 可视化:     confusion_matrix.png, feature_importance.png
    - 数据:       predictions.csv, train_stats.json
    - 配置:       config.yaml, preprocessing.pkl
    - 日志:       training.log

  Artifact存储:
    - 本地文件系统: file:///
    - S3:          s3://bucket/path
    - Azure Blob:  wasbs://container@account/path
    - GCS:         gs://bucket/path
""")

# 列出所有Artifact
print("  本实验中的Artifact:")
for run in mlflow.experiments['wine_classification']['runs']:
    if run['artifacts']:
        print(f"    {run['run_name']}:")
        for art in run['artifacts']:
            print(f"      - {art['name']} ({art['type']})")

# ============================================================
# 7. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 左上: 模型性能对比
ax = axes[0, 0]
exp = mlflow.experiments['wine_classification']
runs = exp['runs'][:7]  # 前7个运行
names = [r['run_name'] for r in runs]
accs = [r['metrics'].get('accuracy', 0) for r in runs]
f1s = [r['metrics'].get('f1_macro', 0) for r in runs]

x = np.arange(len(names))
width = 0.35
ax.bar(x - width/2, accs, width, label='Accuracy', color='#2196F3')
ax.bar(x + width/2, f1s, width, label='F1 Macro', color='#4CAF50')
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
ax.set_ylabel('分数')
ax.set_title('各运行性能对比')
ax.legend()
ax.set_ylim(0.8, 1.05)

# 右上: 过拟合分析
ax = axes[0, 1]
rf_only = [r for r in exp['runs'] if r['params'].get('model') == 'RandomForest' and 'train_accuracy' in r['metrics']]
if rf_only:
    rf_names = [r['run_name'] for r in rf_only]
    train_accs = [r['metrics']['train_accuracy'] for r in rf_only]
    test_accs = [r['metrics']['accuracy'] for r in rf_only]
    gaps = [r['metrics']['overfit_gap'] for r in rf_only]

    x = np.arange(len(rf_names))
    ax.bar(x - 0.2, train_accs, 0.2, label='Train Acc', color='#2196F3')
    ax.bar(x, test_accs, 0.2, label='Test Acc', color='#4CAF50')
    ax.bar(x + 0.2, gaps, 0.2, label='Overfit Gap', color='#F44336')
    ax.set_xticks(x)
    ax.set_xticklabels(rf_names, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('分数')
    ax.set_title('过拟合分析')
    ax.legend(fontsize=8)

# 左下: 学习曲线
ax = axes[1, 0]
curve_run = [r for r in exp['runs'] if r['run_name'] == 'RF_training_curve'][0]
train_hist = curve_run['metric_history']['train_accuracy']
test_hist = curve_run['metric_history']['test_accuracy']
train_steps = [h['step'] for h in train_hist]
train_vals = [h['value'] for h in train_hist]
test_steps = [h['step'] for h in test_hist]
test_vals = [h['value'] for h in test_hist]

ax.plot(train_steps, train_vals, 'b-o', label='训练集准确率')
ax.plot(test_steps, test_vals, 'r-s', label='测试集准确率')
ax.set_xlabel('训练数据比例')
ax.set_ylabel('准确率')
ax.set_title('学习曲线')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(range(len(train_sizes)))
ax.set_xticklabels([f'{t:.0%}' for t in train_sizes], fontsize=8)

# 右下: CV准确率箱线图
ax = axes[1, 1]
cv_data = {}
for r in exp['runs']:
    if 'cv_accuracy' in r['metrics']:
        cv_data[r['run_name']] = r['metrics']['cv_accuracy']

if cv_data:
    ax.bar(cv_data.keys(), cv_data.values(), color='#FF9800')
    ax.set_ylabel('CV准确率')
    ax.set_title('5折交叉验证准确率')
    ax.set_xticklabels(cv_data.keys(), rotation=45, ha='right', fontsize=7)
    ax.set_ylim(0.9, 1.0)

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W26/d2_mlflow_practice.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存: d2_mlflow_practice.png")

print("\n完成! MLflow实践要点:")
print("  1. 自动日志: mlflow.sklearn.autolog() 自动记录参数和指标")
print("  2. 自定义指标: log_metric + step 支持训练曲线")
print("  3. Artifact: 模型、图表、数据文件的版本管理")
print("  4. 搜索过滤: 按指标或参数筛选运行")
print("  5. 过拟合检测: 记录训练/测试指标差距")
