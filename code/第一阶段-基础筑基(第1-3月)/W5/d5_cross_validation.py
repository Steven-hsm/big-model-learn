### Day 5：交叉验证 + 超参数调优
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (KFold, StratifiedKFold,
                                     cross_val_score, GridSearchCV,
                                     RandomizedSearchCV, train_test_split)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 加载数据
# ============================================================
data = load_breast_cancer()
X, y = data.data, data.target

print("=" * 60)
print("交叉验证 + 超参数调优")
print("=" * 60)
print(f"数据集: breast_cancer")
print(f"样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
print(f"类别分布: 0(恶性)={sum(y==0)}, 1(良性)={sum(y==1)}")

# ============================================================
# 2. KFold vs StratifiedKFold
# ============================================================
print("\n" + "-" * 60)
print("2. KFold vs StratifiedKFold")
print("-" * 60)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nKFold 各折类别分布:")
for fold, (_, val_idx) in enumerate(kf.split(X, y)):
    print(f"  Fold {fold+1}: 类别0={sum(y[val_idx]==0)}, 类别1={sum(y[val_idx]==1)}")

print("\nStratifiedKFold 各折类别分布:")
for fold, (_, val_idx) in enumerate(skf.split(X, y)):
    print(f"  Fold {fold+1}: 类别0={sum(y[val_idx]==0)}, 类别1={sum(y[val_idx]==1)}")

# ============================================================
# 3. cross_val_score — 多种评估指标
# ============================================================
print("\n" + "-" * 60)
print("3. cross_val_score 多种评估指标")
print("-" * 60)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=5000, random_state=42))
])

scorings = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
for scoring in scorings:
    scores = cross_val_score(pipe, X, y, cv=5, scoring=scoring)
    print(f"  {scoring:>10s}: {scores.mean():.4f} (+/- {scores.std():.4f})  {scores}")

# ============================================================
# 4. GridSearchCV 超参数调优
# ============================================================
print("\n" + "-" * 60)
print("4. GridSearchCV 网格搜索")
print("-" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

param_grid = {
    'lr__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'lr__penalty': ['l1', 'l2'],
    'lr__solver': ['liblinear']
}

grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy',
                           return_train_score=True, n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳CV分数: {grid_search.best_score_:.4f}")
print(f"测试集准确率: {grid_search.score(X_test, y_test):.4f}")

print("\n各参数组合结果:")
results = grid_search.cv_results_
for i in range(len(results['params'])):
    params = results['params'][i]
    mean_score = results['mean_test_score'][i]
    std_score = results['std_test_score'][i]
    print(f"  C={params['lr__C']:>6}, penalty={params['lr__penalty']:>2} "
          f"=> {mean_score:.4f} (+/- {std_score:.4f})")

# ============================================================
# 5. RandomizedSearchCV
# ============================================================
print("\n" + "-" * 60)
print("5. RandomizedSearchCV 随机搜索")
print("-" * 60)

from scipy.stats import loguniform

param_dist = {
    'lr__C': loguniform(1e-3, 1e3),
    'lr__penalty': ['l1', 'l2'],
    'lr__solver': ['liblinear']
}

random_search = RandomizedSearchCV(pipe, param_dist, n_iter=20, cv=5,
                                   scoring='accuracy', random_state=42,
                                   n_jobs=-1)
random_search.fit(X_train, y_train)

print(f"最佳参数: {random_search.best_params_}")
print(f"最佳CV分数: {random_search.best_score_:.4f}")
print(f"测试集准确率: {random_search.score(X_test, y_test):.4f}")

# ============================================================
# 6. Pipeline + GridSearchCV 完整示例
# ============================================================
print("\n" + "-" * 60)
print("6. Pipeline + GridSearchCV 完整流程")
print("-" * 60)

from sklearn.decomposition import PCA

pipe_full = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('lr', LogisticRegression(max_iter=5000, random_state=42))
])

param_grid_full = {
    'pca__n_components': [5, 10, 15, 20, 30],
    'lr__C': [0.01, 0.1, 1, 10],
    'lr__penalty': ['l1', 'l2'],
    'lr__solver': ['liblinear']
}

grid_full = GridSearchCV(pipe_full, param_grid_full, cv=5,
                         scoring='accuracy', n_jobs=-1)
grid_full.fit(X_train, y_train)

print(f"最佳参数: {grid_full.best_params_}")
print(f"最佳CV分数: {grid_full.best_score_:.4f}")
print(f"测试集准确率: {grid_full.score(X_test, y_test):.4f}")

# ============================================================
# 7. 可视化: C值对模型的影响
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# GridSearchCV结果热力图 (l2 penalty)
ax1 = axes[0]
l2_results = [results['mean_test_score'][i]
              for i in range(len(results['params']))
              if results['params'][i]['lr__penalty'] == 'l2']
l1_results = [results['mean_test_score'][i]
              for i in range(len(results['params']))
              if results['params'][i]['lr__penalty'] == 'l1']
C_values = [0.001, 0.01, 0.1, 1, 10, 100]

ax1.plot(C_values, l2_results, 'bo-', label='L2 penalty', linewidth=2)
ax1.plot(C_values, l1_results, 'rs-', label='L1 penalty', linewidth=2)
ax1.set_xscale('log')
ax1.set_xlabel('C (正则化强度的倒数)')
ax1.set_ylabel('交叉验证准确率')
ax1.set_title('GridSearchCV: C值与正则化类型的影响')
ax1.legend()
ax1.grid(True, alpha=0.3)

# PCA维度对模型的影响
ax2 = axes[1]
pca_results = {}
for i in range(len(grid_full.cv_results_['params'])):
    n_comp = grid_full.cv_results_['params'][i]['pca__n_components']
    score = grid_full.cv_results_['mean_test_score'][i]
    if n_comp not in pca_results:
        pca_results[n_comp] = []
    pca_results[n_comp].append(score)

pca_means = {k: np.mean(v) for k, v in sorted(pca_results.items())}
ax2.bar(range(len(pca_means)), list(pca_means.values()), color='steelblue')
ax2.set_xticks(range(len(pca_means)))
ax2.set_xticklabels([str(k) for k in pca_means.keys()])
ax2.set_xlabel('PCA组件数')
ax2.set_ylabel('平均交叉验证准确率')
ax2.set_title('PCA维度对模型性能的影响')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
