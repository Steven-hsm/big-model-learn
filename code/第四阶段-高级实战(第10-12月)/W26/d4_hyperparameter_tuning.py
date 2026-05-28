"""
W26-D4 超参数优化
==================
超参数优化策略(Grid/Random/Bayesian), Optuna完整实践, 多目标优化, 早停策略
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import time

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W26-D4 超参数优化")
print("=" * 60)

# ============================================================
# 1. 超参数优化策略概述
# ============================================================
print("\n--- 1. 超参数优化策略概述 ---")
print("""
超参数优化方法对比:

  ┌──────────────┬────────────┬────────────┬──────────────┐
  │ 方法         │ 搜索效率   │ 适用场景   │ 优缺点       │
  ├──────────────┼────────────┼────────────┼──────────────┤
  │ Grid Search  │ 低(穷举)   │ 参数少     │ 简单但慢     │
  │ Random       │ 中(随机)   │ 参数多     │ 比Grid高效   │
  │ Bayesian     │ 高(智能)   │ 昂贵模型   │ 利用历史信息 │
  │ Optuna       │ 高(自动)   │ 通用       │ 剪枝+分布式  │
  │ Hyperband    │ 高(自适应) │ 迭代模型   │ 早停省资源   │
  └──────────────┴────────────┴────────────┴──────────────┘
""")

# 准备数据
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ============================================================
# 2. Grid Search
# ============================================================
print("\n--- 2. Grid Search (网格搜索) ---")


def grid_search(param_grid, X, y, cv=5):
    """网格搜索"""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(product(*values))

    results = []
    total = len(combinations)
    print(f"  搜索空间: {total} 种组合")

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        model = RandomForestClassifier(random_state=42, **params)
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        result = {**params, 'mean_accuracy': scores.mean(), 'std_accuracy': scores.std()}
        results.append(result)
        if (i + 1) % 5 == 0 or i == total - 1:
            print(f"    [{i+1}/{total}] params={params}, acc={scores.mean():.4f}")

    return pd.DataFrame(results).sort_values('mean_accuracy', ascending=False)


param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5],
}
grid_results = grid_search(param_grid, X_train_s, y_train)
print(f"\n  Grid Search 最佳参数:")
best_grid = grid_results.iloc[0]
print(f"    {best_grid.to_dict()}")

# ============================================================
# 3. Random Search
# ============================================================
print("\n--- 3. Random Search (随机搜索) ---")


def random_search(param_distributions, X, y, n_iter=20, cv=5):
    """随机搜索"""
    results = []

    for i in range(n_iter):
        params = {}
        for key, dist in param_distributions.items():
            if isinstance(dist, list):
                params[key] = np.random.choice(dist)
            elif isinstance(dist, tuple) and dist[0] == 'int':
                params[key] = np.random.randint(dist[1], dist[2] + 1)
            elif isinstance(dist, tuple) and dist[0] == 'float':
                params[key] = np.random.uniform(dist[1], dist[2])
            elif isinstance(dist, tuple) and dist[0] == 'log':
                params[key] = 10 ** np.random.uniform(dist[1], dist[2])

        model = RandomForestClassifier(random_state=42, **params)
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        result = {**params, 'mean_accuracy': scores.mean(), 'std_accuracy': scores.std()}
        results.append(result)

        if (i + 1) % 5 == 0:
            print(f"    [{i+1}/{n_iter}] acc={scores.mean():.4f}")

    return pd.DataFrame(results).sort_values('mean_accuracy', ascending=False)


param_dist = {
    'n_estimators': ('int', 50, 300),
    'max_depth': ('int', 3, 20),
    'min_samples_split': ('int', 2, 20),
    'min_samples_leaf': ('int', 1, 10),
    'max_features': ['sqrt', 'log2', None],
}
random_results = random_search(param_dist, X_train_s, y_train, n_iter=20)
print(f"\n  Random Search 最佳参数:")
best_random = random_results.iloc[0]
print(f"    {best_random.to_dict()}")

# ============================================================
# 4. Bayesian Optimization (简化版)
# ============================================================
print("\n--- 4. Bayesian Optimization (简化版) ---")


class SimpleBayesianOptimizer:
    """简化版贝叶斯优化"""

    def __init__(self, param_space, objective_fn):
        self.param_space = param_space
        self.objective_fn = objective_fn
        self.history = []

    def _sample_random(self):
        """随机采样"""
        params = {}
        for key, space in self.param_space.items():
            if space['type'] == 'int':
                params[key] = np.random.randint(space['low'], space['high'] + 1)
            elif space['type'] == 'float':
                params[key] = np.random.uniform(space['low'], space['high'])
            elif space['type'] == 'choice':
                params[key] = np.random.choice(space['values'])
        return params

    def _acquisition(self, explored_params, explored_scores, n_candidates=100):
        """采集函数: Expected Improvement (简化版)"""
        best_score = max(explored_scores) if explored_scores else 0

        # 生成候选点
        candidates = [self._sample_random() for _ in range(n_candidates)]

        best_candidate = None
        best_ei = -np.inf

        for candidate in candidates:
            # 计算与已探索点的距离
            if explored_params:
                distances = []
                for ep in explored_params:
                    d = sum(
                        abs(candidate.get(k, 0) - ep.get(k, 0))
                        for k in candidate if isinstance(candidate.get(k, 0), (int, float))
                    )
                    distances.append(d)

                # 用距离加权预测分数
                weights = np.array([1.0 / (d + 1) for d in distances])
                weights /= weights.sum()
                predicted_score = np.dot(weights, explored_scores)
                uncertainty = np.std(explored_scores) / (np.mean(distances) + 1)

                # EI = (predicted - best) * uncertainty
                ei = (predicted_score - best_score + uncertainty)
            else:
                ei = 1.0

            if ei > best_ei:
                best_ei = ei
                best_candidate = candidate

        return best_candidate

    def optimize(self, n_trials=20, n_initial=5):
        """运行优化"""
        print(f"  贝叶斯优化: {n_trials} 次试验 (初始随机: {n_initial})")

        explored_params = []
        explored_scores = []

        # 初始随机探索
        for i in range(n_initial):
            params = self._sample_random()
            score = self.objective_fn(params)
            explored_params.append(params)
            explored_scores.append(score)
            self.history.append({'trial': i + 1, **params, 'score': score, 'type': 'random'})

        # 贝叶斯优化
        for i in range(n_initial, n_trials):
            params = self._acquisition(explored_params, explored_scores)
            score = self.objective_fn(params)
            explored_params.append(params)
            explored_scores.append(score)
            self.history.append({'trial': i + 1, **params, 'score': score, 'type': 'bayesian'})

            if (i + 1) % 5 == 0:
                print(f"    Trial {i+1}: score={score:.4f}, best={max(explored_scores):.4f}")

        best_idx = np.argmax(explored_scores)
        print(f"\n  Bayesian Optimization 最佳:")
        print(f"    params={explored_params[best_idx]}")
        print(f"    score={explored_scores[best_idx]:.4f}")

        return explored_params[best_idx], explored_scores[best_idx]


def objective(params):
    """目标函数"""
    model = RandomForestClassifier(random_state=42, **params)
    scores = cross_val_score(model, X_train_s, y_train, cv=3, scoring='accuracy')
    return scores.mean()


param_space = {
    'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
    'max_depth': {'type': 'int', 'low': 3, 'high': 20},
    'min_samples_split': {'type': 'int', 'low': 2, 'high': 15},
}

optimizer = SimpleBayesianOptimizer(param_space, objective)
best_bayes_params, best_bayes_score = optimizer.optimize(n_trials=20, n_initial=5)

# ============================================================
# 5. Optuna概念与实践 (简化版)
# ============================================================
print("\n--- 5. Optuna概念与实践 ---")
print("""
  Optuna 核心概念:
    - Study:   一次优化任务
    - Trial:   一次参数尝试
    - Suggest: 参数建议 API
    - Prune:   早停剪枝

  代码示例 (需要安装 optuna):
    import optuna

    def objective(trial):
        lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        depth = trial.suggest_int('depth', 3, 10)

        model = XGBClassifier(max_depth=depth, learning_rate=lr)
        score = cross_val_score(model, X, y, cv=5).mean()
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
""")


class SimpleOptunaSimulator:
    """简化版Optuna模拟器"""

    def __init__(self, direction='maximize'):
        self.direction = direction
        self.trials = []
        self.best_trial = None

    def suggest_int(self, trial, name, low, high):
        return np.random.randint(low, high + 1)

    def suggest_float(self, trial, name, low, high, log=False):
        if log:
            return 10 ** np.random.uniform(np.log10(low), np.log10(high))
        return np.random.uniform(low, high)

    def suggest_categorical(self, trial, name, choices):
        return np.random.choice(choices)

    def optimize(self, objective_fn, n_trials=30):
        """运行优化"""
        for i in range(n_trials):
            trial = {'number': i}
            try:
                value = objective_fn(trial)
                state = 'COMPLETE'
            except Exception as e:
                value = None
                state = 'FAIL'

            trial_record = {
                'number': i,
                'value': value,
                'state': state,
            }
            self.trials.append(trial_record)

            if value is not None:
                if self.best_trial is None:
                    self.best_trial = trial_record
                elif self.direction == 'maximize' and value > self.best_trial['value']:
                    self.best_trial = trial_record
                elif self.direction == 'minimize' and value < self.best_trial['value']:
                    self.best_trial = trial_record

            if (i + 1) % 10 == 0:
                best_val = self.best_trial['value'] if self.best_trial else 0
                print(f"    Trial {i+1}: current={value:.4f}, best={best_val:.4f}")

        return self.best_trial


sim_optuna = SimpleOptunaSimulator(direction='maximize')


def optuna_objective(trial):
    n_est = sim_optuna.suggest_int(trial, 'n_estimators', 50, 300)
    depth = sim_optuna.suggest_int(trial, 'max_depth', 3, 20)
    min_samples = sim_optuna.suggest_int(trial, 'min_samples_split', 2, 15)

    model = RandomForestClassifier(
        n_estimators=n_est, max_depth=depth,
        min_samples_split=min_samples, random_state=42
    )
    return cross_val_score(model, X_train_s, y_train, cv=3, scoring='accuracy').mean()


best_optuna = sim_optuna.optimize(optuna_objective, n_trials=30)
print(f"\n  Optuna模拟 最佳: score={best_optuna['value']:.4f}")

# ============================================================
# 6. 多目标优化概念
# ============================================================
print("\n--- 6. 多目标优化 ---")
print("""
  多目标优化: 同时优化多个冲突指标

  常见场景:
    - 准确率 vs 推理速度
    - 模型大小 vs 性能
    - 公平性 vs 准确率

  Pareto前沿: 没有一个解能在所有目标上同时更好

  Optuna多目标:
    study = optuna.create_study(directions=['maximize', 'minimize'])
    # 最大化准确率, 最小化延迟
""")

# 多目标优化实验
multi_obj_results = []
for _ in range(30):
    n_est = np.random.randint(10, 300)
    depth = np.random.randint(2, 20)
    model = RandomForestClassifier(n_estimators=n_est, max_depth=depth, random_state=42)
    model.fit(X_train_s, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_s))

    # 推理时间作为第二个目标
    t0 = time.time()
    for _ in range(100):
        model.predict(X_test_s[:10])
    latency = (time.time() - t0) * 1000  # ms

    multi_obj_results.append({
        'n_estimators': n_est, 'max_depth': depth,
        'accuracy': acc, 'latency_ms': latency,
    })

multi_df = pd.DataFrame(multi_obj_results)

# 找Pareto前沿
def find_pareto(df, obj1, obj2, direction1='max', direction2='min'):
    pareto = []
    for i, row in df.iterrows():
        dominated = False
        for j, other in df.iterrows():
            if i == j:
                continue
            better1 = other[obj1] > row[obj1] if direction1 == 'max' else other[obj1] < row[obj1]
            better2 = other[obj2] < row[obj2] if direction2 == 'min' else other[obj2] > row[obj2]
            if better1 and better2:
                dominated = True
                break
        if not dominated:
            pareto.append(i)
    return pareto

pareto_idx = find_pareto(multi_df, 'accuracy', 'latency_ms', 'max', 'min')
print(f"  Pareto前沿: {len(pareto_idx)} 个非支配解")
for idx in pareto_idx:
    r = multi_df.iloc[idx]
    print(f"    n_est={r['n_estimators']}, depth={r['max_depth']}: "
          f"acc={r['accuracy']:.4f}, latency={r['latency_ms']:.2f}ms")

# ============================================================
# 7. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 左上: Grid vs Random 搜索最佳值收敛
ax = axes[0, 0]
grid_accs = grid_results['mean_accuracy'].values
random_accs = random_results['mean_accuracy'].values

ax.plot(range(1, len(grid_accs) + 1), np.maximum.accumulate(grid_accs),
        'b-o', label='Grid Search', markersize=4)
ax.plot(range(1, len(random_accs) + 1), np.maximum.accumulate(random_accs),
        'r-s', label='Random Search', markersize=4)
ax.set_xlabel('试验次数')
ax.set_ylabel('最佳准确率')
ax.set_title('搜索策略收敛速度')
ax.legend()
ax.grid(True, alpha=0.3)

# 右上: Bayesian优化历史
ax = axes[0, 1]
bayes_df = pd.DataFrame(optimizer.history)
random_trials = bayes_df[bayes_df['type'] == 'random']
bayes_trials = bayes_df[bayes_df['type'] == 'bayesian']

ax.scatter(random_trials['trial'], random_trials['score'],
           c='#2196F3', label='随机探索', s=50)
ax.scatter(bayes_trials['trial'], bayes_trials['score'],
           c='#F44336', label='贝叶斯优化', s=50)
all_scores = bayes_df['score'].values
ax.plot(bayes_df['trial'], np.maximum.accumulate(all_scores),
        'g--', linewidth=2, label='最佳值')
ax.set_xlabel('Trial')
ax.set_ylabel('准确率')
ax.set_title('贝叶斯优化过程')
ax.legend()
ax.grid(True, alpha=0.3)

# 左下: 多目标优化Pareto前沿
ax = axes[1, 0]
pareto_df = multi_df.iloc[pareto_idx].sort_values('accuracy')
ax.scatter(multi_df['latency_ms'], multi_df['accuracy'],
           c='#BBDEFB', s=30, label='所有解', alpha=0.7)
ax.scatter(pareto_df['latency_ms'], pareto_df['accuracy'],
           c='#F44336', s=80, label='Pareto前沿', zorder=5, edgecolors='black')
ax.plot(pareto_df['latency_ms'], pareto_df['accuracy'],
        'r--', alpha=0.5, linewidth=2)
ax.set_xlabel('推理延迟 (ms)')
ax.set_ylabel('准确率')
ax.set_title('多目标优化: 准确率 vs 延迟')
ax.legend()

# 右下: 各方法最佳结果对比
ax = axes[1, 1]
methods = ['Grid\nSearch', 'Random\nSearch', 'Bayesian\nOptim', 'Optuna\n(模拟)']
best_scores = [
    grid_results['mean_accuracy'].max(),
    random_results['mean_accuracy'].max(),
    best_bayes_score,
    best_optuna['value'],
]
colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
bars = ax.bar(methods, best_scores, color=colors)
ax.set_ylabel('最佳准确率')
ax.set_title('优化方法对比')
ax.set_ylim(0.9, 1.0)
for bar, score in zip(bars, best_scores):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
            f'{score:.4f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W26/d4_hyperparameter_tuning.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存: d4_hyperparameter_tuning.png")

print("\n完成! 超参数优化要点:")
print("  1. Grid Search: 穷举搜索, 简单但昂贵")
print("  2. Random Search: 随机采样, 比Grid更高效")
print("  3. Bayesian: 利用历史信息智能选择参数")
print("  4. Optuna: 自动化调参框架, 支持剪枝和分布式")
print("  5. 多目标优化: Pareto前沿, 平衡冲突指标")
