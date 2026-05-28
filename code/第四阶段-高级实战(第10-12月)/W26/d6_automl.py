"""
W26-D6 AutoML
==============
AutoML概念, 自动模型选择, 自动特征工程, 神经网络搜索(NAS概念), 简化版AutoML实现
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W26-D6 AutoML")
print("=" * 60)

# ============================================================
# 1. AutoML概念
# ============================================================
print("\n--- 1. AutoML概念 ---")
print("""
AutoML (自动化机器学习) 目标: 让非专家也能使用ML

  AutoML Pipeline:
  ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐
  │ 数据   │──>│ 特征   │──>│ 模型   │──>│ 超参数 │──>│ 集成   │
  │ 预处理 │   │ 工程   │   │ 选择   │   │ 优化   │   │ 学习   │
  └────────┘   └────────┘   └────────┘   └────────┘   └────────┘

  核心组件:
    1. 自动数据预处理: 类型推断、缺失值、编码
    2. 自动特征工程: 特征合成、选择
    3. 自动模型选择: 从候选模型中选最佳
    4. 超参数优化: 贝叶斯/进化/随机搜索
    5. 神经网络搜索 (NAS): 自动设计网络结构
    6. 自动集成: 多模型融合

  工具:
    - auto-sklearn:  基于scikit-learn的AutoML
    - TPOT:          基于遗传编程的AutoML
    - H2O AutoML:    企业级AutoML
    - Google AutoML: 云端AutoML服务
    - FLAML:         微软开源, 高效AutoML
""")

# ============================================================
# 2. 简化版AutoML实现
# ============================================================
print("\n--- 2. 简化版AutoML实现 ---")


class SimpleAutoML:
    """简化版AutoML框架"""

    def __init__(self, time_limit=60, metric='accuracy', cv=5):
        self.time_limit = time_limit
        self.metric = metric
        self.cv = cv
        self.results = []
        self.best_model = None
        self.best_score = 0

        # 候选模型池
        self.model_pool = [
            ('LogisticRegression', LogisticRegression, {
                'max_iter': [500, 1000],
                'C': [0.01, 0.1, 1.0, 10.0],
            }),
            ('RandomForest', RandomForestClassifier, {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
            }),
            ('GradientBoosting', GradientBoostingClassifier, {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
            }),
            ('SVM_RBF', SVC, {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['rbf'],
            }),
            ('KNN', KNeighborsClassifier, {
                'n_neighbors': [3, 5, 7, 11],
                'weights': ['uniform', 'distance'],
            }),
            ('DecisionTree', DecisionTreeClassifier, {
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
            }),
            ('AdaBoost', AdaBoostClassifier, {
                'n_estimators': [50, 100],
                'learning_rate': [0.5, 1.0],
            }),
            ('ExtraTrees', ExtraTreesClassifier, {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
            }),
        ]

    def _generate_configs(self, param_grid):
        """生成参数组合"""
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        from itertools import product
        configs = []
        for combo in product(*values):
            configs.append(dict(zip(keys, combo)))
        return configs

    def fit(self, X, y):
        """运行AutoML"""
        start_time = time.time()
        total_configs = sum(len(self._generate_configs(m[2])) for m in self.model_pool)
        evaluated = 0

        print(f"  AutoML启动: {len(self.model_pool)} 种模型, ~{total_configs} 种配置")
        print(f"  时间限制: {self.time_limit}s\n")

        for model_name, model_class, param_grid in self.model_pool:
            configs = self._generate_configs(param_grid)

            for config in configs:
                # 检查时间
                elapsed = time.time() - start_time
                if elapsed > self.time_limit:
                    print(f"\n  [时间到] 已运行 {elapsed:.1f}s, 评估了 {evaluated} 种配置")
                    return self

                try:
                    model = model_class(random_state=42, **config)
                except TypeError:
                    try:
                        model = model_class(**config)
                    except Exception:
                        continue

                scores = cross_val_score(model, X, y, cv=self.cv, scoring=self.metric)
                mean_score = scores.mean()
                std_score = scores.std()

                self.results.append({
                    'model': model_name,
                    'params': config,
                    'mean_score': mean_score,
                    'std_score': std_score,
                    'cv_scores': scores,
                    'training_time': time.time() - start_time,
                })

                if mean_score > self.best_score:
                    self.best_score = mean_score
                    self.best_model = (model_name, config, mean_score)
                    print(f"  New Best: {model_name} {config} -> {mean_score:.4f} (+/- {std_score:.4f})")

                evaluated += 1

        print(f"\n  AutoML完成: 评估了 {evaluated}/{total_configs} 种配置")
        return self

    def leaderboard(self, top_n=10):
        """排行榜"""
        df = pd.DataFrame(self.results)
        df = df.sort_values('mean_score', ascending=False).head(top_n)
        print(f"\n  AutoML 排行榜 Top {top_n}:")
        print(f"  {'排名':<4} {'模型':<20} {'参数':<40} {'得分':<10} {'标准差'}")
        print("  " + "-" * 85)
        for i, (_, row) in enumerate(df.iterrows(), 1):
            params_str = str(row['params'])[:38]
            print(f"  {i:<4} {row['model']:<20} {params_str:<40} "
                  f"{row['mean_score']:.4f}    {row['std_score']:.4f}")
        return df

    def get_best_model_instance(self, X, y):
        """获取最佳模型的训练实例"""
        if self.best_model is None:
            return None
        name, config, _ = self.best_model
        for model_name, model_class, _ in self.model_pool:
            if model_name == name:
                try:
                    model = model_class(random_state=42, **config)
                except TypeError:
                    model = model_class(**config)
                model.fit(X, y)
                return model
        return None


# 加载数据
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 运行AutoML
automl = SimpleAutoML(time_limit=30, metric='accuracy', cv=3)
automl.fit(X_train_s, y_train)
automl.leaderboard(top_n=10)

# 最佳模型在测试集上评估
best_model = automl.get_best_model_instance(X_train_s, y_train)
if best_model:
    test_acc = accuracy_score(y_test, best_model.predict(X_test_s))
    print(f"\n  最佳模型测试集准确率: {test_acc:.4f}")

# ============================================================
# 3. 自动特征选择
# ============================================================
print("\n--- 3. 自动特征选择 ---")


class AutoFeatureSelector:
    """自动特征选择"""

    def __init__(self, max_features=None):
        self.max_features = max_features
        self.selected_features = None
        self.scores_history = []

    def fit(self, X, y, model=None):
        """贪心前向特征选择"""
        if model is None:
            model = RandomForestClassifier(n_estimators=50, random_state=42)

        n_features = X.shape[1]
        max_feat = self.max_features or n_features
        selected = []
        remaining = list(range(n_features))
        best_overall_score = 0

        print(f"  前向特征选择 (最多 {max_feat} 个特征):")

        while remaining and len(selected) < max_feat:
            best_score = 0
            best_feat = None

            for feat in remaining:
                trial_features = selected + [feat]
                scores = cross_val_score(model, X[:, trial_features], y, cv=3, scoring='accuracy')

                if scores.mean() > best_score:
                    best_score = scores.mean()
                    best_feat = feat

            if best_feat is not None:
                selected.append(best_feat)
                remaining.remove(best_feat)
                self.scores_history.append(best_score)

                if best_score > best_overall_score:
                    best_overall_score = best_score

                print(f"    特征 {best_feat:>2d} 加入 -> "
                      f"累计 {len(selected)} 个, 得分 {best_score:.4f}")

                # 如果性能不再提升, 提前停止
                if len(self.scores_history) > 3:
                    recent = self.scores_history[-3:]
                    if max(recent) - min(recent) < 0.001:
                        print(f"    性能提升<0.001, 提前停止")
                        break

        self.selected_features = selected
        print(f"\n  最终选择 {len(selected)} 个特征: {selected}")
        print(f"  最佳得分: {best_overall_score:.4f}")
        return self


# 特征选择 (使用部分数据加速)
selector = AutoFeatureSelector(max_features=15)
selector.fit(X_train_s, y_train)

# 对比: 全特征 vs 选择特征
rf_full = RandomForestClassifier(n_estimators=100, random_state=42)
rf_selected = RandomForestClassifier(n_estimators=100, random_state=42)

full_scores = cross_val_score(rf_full, X_train_s, y_train, cv=5)
sel_scores = cross_val_score(rf_selected, X_train_s[:, selector.selected_features], y_train, cv=5)

print(f"\n  全特征 ({X_train_s.shape[1]}): 准确率={full_scores.mean():.4f}")
print(f"  选择特征 ({len(selector.selected_features)}): 准确率={sel_scores.mean():.4f}")

# ============================================================
# 4. NAS (Neural Architecture Search) 概念
# ============================================================
print("\n--- 4. NAS (神经架构搜索) 概念 ---")
print("""
  NAS 目标: 自动搜索最优神经网络结构

  搜索空间:
    - 层数: 1-10
    - 每层类型: Conv, Pool, FC, Skip
    - 每层参数: filters, kernel_size, units
    - 连接方式: Sequential, Residual, Dense

  搜索策略:
    1. 随机搜索:   随机采样网络结构
    2. 进化搜索:   遗传算法优化结构
    3. 强化学习:   Controller生成结构
    4. 可微搜索:   DARTS, 将搜索空间连续化

  里程碑:
    - NASNet (2017): RL搜索, ImageNet SOTA
    - EfficientNet (2019): 复合缩放
    - DARTS (2019):   可微架构搜索

  工具:
    - AutoKeras:   开源NAS工具
    - NNI:         微软神经网络搜索
    - Ray Tune:    分布式超参/NAS
""")

# 简化NAS模拟: 搜索最佳"网络宽度"配置
class SimpleNAS:
    """简化版NAS: 搜索MLP网络结构"""

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.search_history = []

    def evaluate_architecture(self, layers, X, y):
        """评估一个网络结构 (用RF模拟)"""
        # 用特征变换模拟不同网络宽度的影响
        np.random.seed(42)
        transformed = X.copy()
        for i, width in enumerate(layers):
            # 随机投影模拟全连接层
            proj = np.random.randn(transformed.shape[1], width) / np.sqrt(width)
            transformed = np.maximum(0, transformed @ proj)  # ReLU

        # 用简单模型评估
        from sklearn.linear_model import LogisticRegression
        try:
            scores = cross_val_score(
                LogisticRegression(max_iter=500, random_state=42),
                transformed, y, cv=3, scoring='accuracy'
            )
            return scores.mean()
        except Exception:
            return 0.0

    def search(self, X, y, max_layers=4, max_width=128, n_trials=20):
        """搜索最佳架构"""
        print(f"  NAS搜索: 最多{max_layers}层, 最大宽度{max_width}")

        for trial in range(n_trials):
            n_layers = np.random.randint(1, max_layers + 1)
            architecture = [np.random.choice([16, 32, 64, 128, 256]) for _ in range(n_layers)]

            score = self.evaluate_architecture(architecture, X, y)
            params = sum(architecture)  # 参数量近似

            self.search_history.append({
                'trial': trial + 1,
                'architecture': architecture,
                'n_layers': n_layers,
                'params_approx': params,
                'score': score,
            })

            if (trial + 1) % 5 == 0:
                best = max(self.search_history, key=lambda x: x['score'])
                print(f"    Trial {trial+1}: arch={architecture}, "
                      f"score={score:.4f}, best={best['score']:.4f}")

        best_arch = max(self.search_history, key=lambda x: x['score'])
        print(f"\n  NAS最佳架构: {best_arch['architecture']}")
        print(f"  得分: {best_arch['score']:.4f}")
        return best_arch


nas = SimpleNAS(X_train_s.shape[1], 2)
best_nas = nas.search(X_train_s, y_train, n_trials=20)

# ============================================================
# 5. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 左上: AutoML排行榜
ax = axes[0, 0]
results_df = pd.DataFrame(automl.results)
model_scores = results_df.groupby('model')['mean_score'].agg(['mean', 'max', 'std'])
model_scores = model_scores.sort_values('max', ascending=True)
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(model_scores)))
ax.barh(model_scores.index, model_scores['max'], xerr=model_scores['std'],
        color=colors, capsize=3)
ax.set_xlabel('准确率')
ax.set_title('AutoML模型性能排行榜')
ax.axvline(automl.best_score, color='red', linestyle='--', alpha=0.5, label=f'最佳={automl.best_score:.4f}')
ax.legend(fontsize=8)

# 右上: 特征选择过程
ax = axes[0, 1]
ax.plot(range(1, len(selector.scores_history) + 1), selector.scores_history,
        'b-o', markersize=6)
ax.set_xlabel('特征数量')
ax.set_ylabel('交叉验证准确率')
ax.set_title('前向特征选择过程')
ax.grid(True, alpha=0.3)

# 左下: NAS搜索历史
ax = axes[1, 0]
nas_df = pd.DataFrame(nas.search_history)
scatter = ax.scatter(nas_df['params_approx'], nas_df['score'],
                     c=nas_df['n_layers'], cmap='viridis', s=80, edgecolors='black')
ax.set_xlabel('参数量(近似)')
ax.set_ylabel('准确率')
ax.set_title('NAS搜索: 参数量 vs 性能')
plt.colorbar(scatter, ax=ax, label='层数')
best_nas_row = nas_df.loc[nas_df['score'].idxmax()]
ax.scatter([best_nas_row['params_approx']], [best_nas_row['score']],
           c='red', s=200, marker='*', zorder=5, label='最佳架构')
ax.legend()

# 右下: AutoML运行时间 vs 性能
ax = axes[1, 1]
ax.scatter(results_df['training_time'], results_df['mean_score'],
           c='#2196F3', alpha=0.5, s=30)
# 标记帕累托前沿
sorted_by_time = results_df.sort_values('training_time')
best_so_far = 0
pareto_times = []
pareto_scores = []
for _, row in sorted_by_time.iterrows():
    if row['mean_score'] > best_so_far:
        best_so_far = row['mean_score']
        pareto_times.append(row['training_time'])
        pareto_scores.append(row['mean_score'])

ax.plot(pareto_times, pareto_scores, 'r-', linewidth=2, label='最佳性能轨迹')
ax.set_xlabel('运行时间 (s)')
ax.set_ylabel('准确率')
ax.set_title('AutoML: 时间 vs 性能')
ax.legend()

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W26/d6_automl.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存: d6_automl.png")

print("\n完成! AutoML要点:")
print("  1. 自动模型选择: 多模型对比, 自动选最佳")
print("  2. 自动特征选择: 前向/后向/基于重要性")
print("  3. 超参数优化: 网格/随机/贝叶斯")
print("  4. NAS: 自动搜索最优网络结构")
print("  5. 自动集成: 多模型融合提升性能")
