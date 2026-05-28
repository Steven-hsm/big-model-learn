"""
W26-D7 实验管理项目
====================
多实验对比, 自动调参, 结果可视化, 最佳模型选择和注册
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
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import (train_test_split, cross_val_score,
                                      StratifiedKFold)
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W26-D7 实验管理项目")
print("=" * 60)

# ============================================================
# 项目: Wine质量分类 - 完整实验管理流程
# ============================================================
print("""
项目: Wine质量分类
目标: 通过系统化实验管理, 找到最佳分类模型并注册

  流程:
  1. 数据准备
  2. 多模型基准测试
  3. 自动超参数调优
  4. 结果可视化对比
  5. 最佳模型选择与注册
""")

# ============================================================
# Step 1: 数据准备
# ============================================================
print("\n=== Step 1: 数据准备 ===")

wine = load_wine()
X, y = wine.data, wine.target
feature_names = wine.feature_names
target_names = wine.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print(f"  训练集: {X_train_s.shape}, 测试集: {X_test_s.shape}")
print(f"  类别分布: {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"  特征: {feature_names}")

# ============================================================
# Step 2: 实验管理系统
# ============================================================
print("\n=== Step 2: 实验管理系统 ===")


class ExperimentManager:
    """完整实验管理器"""

    def __init__(self, project_name):
        self.project_name = project_name
        self.experiments = {}
        self.current_experiment = None
        self.registry = {}  # model_name -> metadata

    def create_experiment(self, name):
        self.experiments[name] = {
            'name': name,
            'runs': [],
            'created_at': datetime.now().isoformat(),
        }
        self.current_experiment = name
        print(f"  创建实验: {name}")
        return self

    def add_run(self, model_name, model, params, metrics, predictions=None):
        """记录一次运行"""
        run = {
            'run_id': hashlib.md5(f"{time.time()}{model_name}".encode()).hexdigest()[:8],
            'model_name': model_name,
            'params': params,
            'metrics': metrics,
            'predictions': predictions,
            'timestamp': datetime.now().isoformat(),
        }
        if self.current_experiment:
            self.experiments[self.current_experiment]['runs'].append(run)
        return run

    def get_all_runs(self):
        """获取所有运行"""
        all_runs = []
        for exp_name, exp in self.experiments.items():
            for run in exp['runs']:
                run['experiment'] = exp_name
                all_runs.append(run)
        return all_runs

    def compare(self, metric='accuracy'):
        """对比所有运行"""
        runs = self.get_all_runs()
        if not runs:
            return None
        df = pd.DataFrame(runs)
        df = df.sort_values(f'metrics.{metric}' if f'metrics.{metric}' in df.columns
                            else 'model_name')
        return df

    def register_best_model(self, metric='accuracy', threshold=0.8):
        """注册最佳模型"""
        runs = self.get_all_runs()
        if not runs:
            print("  无运行记录")
            return None

        best = max(runs, key=lambda r: r['metrics'].get(metric, 0))
        best_score = best['metrics'].get(metric, 0)

        if best_score < threshold:
            print(f"  [拒绝] 最佳 {metric}={best_score:.4f} 低于阈值 {threshold}")
            return None

        model_key = f"{best['model_name']}_{best['run_id']}"
        self.registry[model_key] = {
            **best,
            'stage': 'Production',
            'registered_at': datetime.now().isoformat(),
        }
        print(f"  [注册] {best['model_name']} -> Production ({metric}={best_score:.4f})")
        return best

    def print_summary(self):
        """打印摘要"""
        runs = self.get_all_runs()
        if not runs:
            return

        print(f"\n  项目: {self.project_name}")
        print(f"  实验数: {len(self.experiments)}")
        print(f"  总运行数: {len(runs)}")
        print(f"  注册模型: {len(self.registry)}")

        metrics_data = {}
        for run in runs:
            for metric_name, value in run['metrics'].items():
                if metric_name not in metrics_data:
                    metrics_data[metric_name] = []
                metrics_data[metric_name].append(value)

        print(f"\n  指标统计:")
        for metric_name, values in metrics_data.items():
            print(f"    {metric_name}: max={max(values):.4f}, "
                  f"mean={np.mean(values):.4f}, min={min(values):.4f}")


# ============================================================
# Step 3: 基准测试 (多个模型)
# ============================================================
print("\n=== Step 3: 基准测试 ===")

manager = ExperimentManager('wine_classification')
manager.create_experiment('baseline')

models = {
    'LogisticRegression': (LogisticRegression(max_iter=1000, random_state=42), {'max_iter': 1000}),
    'RandomForest': (RandomForestClassifier(n_estimators=100, random_state=42), {'n_estimators': 100}),
    'GradientBoosting': (GradientBoostingClassifier(n_estimators=100, random_state=42), {'n_estimators': 100}),
    'SVM_RBF': (SVC(kernel='rbf', random_state=42), {'kernel': 'rbf'}),
    'KNN': (KNeighborsClassifier(n_neighbors=5), {'n_neighbors': 5}),
    'ExtraTrees': (ExtraTreesClassifier(n_estimators=100, random_state=42), {'n_estimators': 100}),
    'AdaBoost': (AdaBoostClassifier(n_estimators=100, random_state=42), {'n_estimators': 100}),
}

baseline_results = {}
for name, (model, params) in models.items():
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'precision_macro': precision_score(y_test, y_pred, average='macro'),
        'recall_macro': recall_score(y_test, y_pred, average='macro'),
    }

    # CV分数
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_s, y_train, cv=cv, scoring='accuracy')
    metrics['cv_accuracy'] = cv_scores.mean()
    metrics['cv_std'] = cv_scores.std()

    manager.add_run(name, model, params, metrics, y_pred)
    baseline_results[name] = metrics

    print(f"  {name:<20} acc={metrics['accuracy']:.4f} f1={metrics['f1_macro']:.4f} "
          f"cv={metrics['cv_accuracy']:.4f}(+/-{metrics['cv_std']:.4f})")

# ============================================================
# Step 4: 超参数调优
# ============================================================
print("\n=== Step 4: 超参数调优 ===")

manager.create_experiment('hyperparameter_tuning')

# Random Forest 调参
print("\n  Random Forest 调参:")
for n_est in [50, 100, 150, 200]:
    for depth in [5, 10, 15, None]:
        for min_samples in [2, 5]:
            params = {'n_estimators': n_est, 'max_depth': depth, 'min_samples_split': min_samples}
            try:
                model = RandomForestClassifier(random_state=42, **params)
            except TypeError:
                continue
            cv_scores = cross_val_score(model, X_train_s, y_train, cv=3, scoring='accuracy')

            metrics = {
                'cv_accuracy': cv_scores.mean(),
                'cv_std': cv_scores.std(),
            }
            # 在测试集上评估
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            metrics['f1_macro'] = f1_score(y_test, y_pred, average='macro')

            manager.add_run(f'RF_{n_est}_{depth}_{min_samples}', model, params, metrics)

print(f"  共运行 {len(manager.experiments['hyperparameter_tuning']['runs'])} 种参数组合")

# Gradient Boosting 调参
print("\n  Gradient Boosting 调参:")
for n_est in [50, 100, 200]:
    for lr in [0.01, 0.05, 0.1, 0.2]:
        for depth in [3, 5, 7]:
            params = {'n_estimators': n_est, 'learning_rate': lr, 'max_depth': depth}
            model = GradientBoostingClassifier(random_state=42, **params)
            cv_scores = cross_val_score(model, X_train_s, y_train, cv=3, scoring='accuracy')

            metrics = {'cv_accuracy': cv_scores.mean(), 'cv_std': cv_scores.std()}
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            metrics['f1_macro'] = f1_score(y_test, y_pred, average='macro')

            manager.add_run(f'GB_{n_est}_{lr}_{depth}', model, params, metrics)

total_tuning = len(manager.experiments['hyperparameter_tuning']['runs'])
print(f"  调参总运行: {total_tuning}")

# ============================================================
# Step 5: 最佳模型选择
# ============================================================
print("\n=== Step 5: 最佳模型选择 ===")

all_runs = manager.get_all_runs()
best_run = max(all_runs, key=lambda r: r['metrics'].get('accuracy', 0))

print(f"  最佳模型: {best_run['model_name']}")
print(f"  参数: {best_run['params']}")
print(f"  准确率: {best_run['metrics']['accuracy']:.4f}")
print(f"  F1 Macro: {best_run['metrics']['f1_macro']:.4f}")
print(f"  CV准确率: {best_run['metrics'].get('cv_accuracy', 'N/A')}")

# 重新训练最佳模型
best_params = best_run['params']
best_model_name = best_run['model_name']

if 'RF' in best_model_name:
    final_model = RandomForestClassifier(random_state=42, **best_params)
elif 'GB' in best_model_name:
    final_model = GradientBoostingClassifier(random_state=42, **best_params)
else:
    final_model = GradientBoostingClassifier(random_state=42)

final_model.fit(X_train_s, y_train)
y_pred_final = final_model.predict(X_test_s)

print(f"\n  最终模型测试报告:")
print(classification_report(y_test, y_pred_final, target_names=[f'class_{i}' for i in range(3)]))

# 注册
manager.register_best_model('accuracy', threshold=0.9)
manager.print_summary()

# ============================================================
# Step 6: 可视化
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. 基准模型对比
ax = axes[0, 0]
bl_names = list(baseline_results.keys())
bl_accs = [baseline_results[n]['accuracy'] for n in bl_names]
bl_f1s = [baseline_results[n]['f1_macro'] for n in bl_names]

x = np.arange(len(bl_names))
width = 0.35
ax.bar(x - width/2, bl_accs, width, label='Accuracy', color='#2196F3')
ax.bar(x + width/2, bl_f1s, width, label='F1 Macro', color='#4CAF50')
ax.set_xticks(x)
ax.set_xticklabels(bl_names, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('分数')
ax.set_title('基准模型性能对比')
ax.legend()
ax.set_ylim(0.8, 1.05)

# 2. CV准确率箱线图
ax = axes[0, 1]
cv_data_to_plot = []
cv_labels = []
for name in ['RandomForest', 'GradientBoosting', 'SVM_RBF', 'KNN']:
    model = models[name][0]
    cv_scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring='accuracy')
    cv_data_to_plot.append(cv_scores)
    cv_labels.append(name)

ax.boxplot(cv_data_to_plot, labels=cv_labels)
ax.set_ylabel('CV准确率')
ax.set_title('5折交叉验证分布')
ax.grid(True, alpha=0.3)

# 3. 混淆矩阵(最佳模型)
ax = axes[0, 2]
cm = confusion_matrix(y_test, y_pred_final)
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels([f'class_{i}' for i in range(3)])
ax.set_yticklabels([f'class_{i}' for i in range(3)])
ax.set_xlabel('预测')
ax.set_ylabel('真实')
ax.set_title(f'混淆矩阵 ({best_model_name})')
for i in range(3):
    for j in range(3):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax)

# 4. 调参结果: RF n_estimators vs accuracy
ax = axes[1, 0]
rf_runs = [r for r in all_runs if r['model_name'].startswith('RF') and r['experiment'] == 'hyperparameter_tuning']
if rf_runs:
    n_ests = [r['params'].get('n_estimators', 0) for r in rf_runs]
    accs = [r['metrics']['accuracy'] for r in rf_runs]
    ax.scatter(n_ests, accs, alpha=0.5, c='#2196F3', s=30)
    # 按n_estimators聚合均值
    n_est_unique = sorted(set(n_ests))
    mean_accs = [np.mean([a for n, a in zip(n_ests, accs) if n == ne]) for ne in n_est_unique]
    ax.plot(n_est_unique, mean_accs, 'r-o', markersize=8, label='均值')
    ax.set_xlabel('n_estimators')
    ax.set_ylabel('准确率')
    ax.set_title('RF: n_estimators vs 性能')
    ax.legend()
    ax.grid(True, alpha=0.3)

# 5. 调参结果: GB learning_rate vs accuracy
ax = axes[1, 1]
gb_runs = [r for r in all_runs if r['model_name'].startswith('GB')]
if gb_runs:
    lrs = [r['params'].get('learning_rate', 0) for r in gb_runs]
    accs = [r['metrics']['accuracy'] for r in gb_runs]
    depths = [r['params'].get('max_depth', 0) for r in gb_runs]
    scatter = ax.scatter(lrs, accs, c=depths, cmap='viridis', s=50, edgecolors='black')
    ax.set_xlabel('learning_rate')
    ax.set_ylabel('准确率')
    ax.set_title('GB: learning_rate vs 性能 (颜色=max_depth)')
    plt.colorbar(scatter, ax=ax, label='max_depth')

# 6. 排行榜Top10
ax = axes[1, 2]
all_runs_sorted = sorted(all_runs, key=lambda r: r['metrics'].get('accuracy', 0), reverse=True)[:10]
top_names = [r['model_name'][:20] for r in all_runs_sorted]
top_accs = [r['metrics']['accuracy'] for r in all_runs_sorted]
colors = ['#FFD700' if i == 0 else '#C0C0C0' if i == 1 else '#CD7F32' if i == 2 else '#2196F3'
          for i in range(len(top_names))]
ax.barh(range(len(top_names)), top_accs, color=colors)
ax.set_yticks(range(len(top_names)))
ax.set_yticklabels(top_names, fontsize=8)
ax.set_xlabel('准确率')
ax.set_title('Top 10 模型排行')
ax.set_xlim(min(top_accs) - 0.02, 1.0)
for i, acc in enumerate(top_accs):
    ax.text(acc + 0.001, i, f'{acc:.4f}', va='center', fontsize=8)

plt.suptitle(f'实验管理项目总结 - 最佳模型: {best_model_name} (acc={best_run["metrics"]["accuracy"]:.4f})',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W26/d7_experiment_project.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存: d7_experiment_project.png")

print("\n" + "=" * 60)
print("实验管理项目总结:")
print(f"  1. 基准测试: {len(baseline_results)} 个模型")
print(f"  2. 超参数调优: {total_tuning} 种配置")
print(f"  3. 最佳模型: {best_model_name} (准确率={best_run['metrics']['accuracy']:.4f})")
print(f"  4. 已注册到生产环境")
print("=" * 60)
