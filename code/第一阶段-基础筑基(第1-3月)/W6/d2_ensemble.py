"""
W06 Day2: 随机森林 + XGBoost + LightGBM
========================================
内容:
1. 加载 breast_cancer 数据集
2. 随机森林训练与 OOB 评分
3. 随机森林特征重要性
4. 对比 RF、GradientBoosting（try/except 兼容 xgboost/lightgbm）
5. 交叉验证对比
6. 模型对比柱状图
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier
)
from sklearn.metrics import accuracy_score, classification_report

# 尝试导入 xgboost 和 lightgbm
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[提示] xgboost 未安装，跳过 XGBoost 模型")

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("[提示] lightgbm 未安装，跳过 LightGBM 模型")

# ========== matplotlib 中文显示设置 ==========
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ================================================================
# 1. 加载数据
# ================================================================
print("=" * 60)
print("1. 加载 Breast Cancer 数据集")
print("=" * 60)

data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names
target_names = data.target_names

print(f"数据形状: {X.shape}")
print(f"类别: {target_names} (0=恶性, 1=良性)")
print(f"类别分布: 恶性={np.sum(y == 0)}, 良性={np.sum(y == 1)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")

# ================================================================
# 2. 随机森林 + OOB 评分
# ================================================================
print("\n" + "=" * 60)
print("2. 随机森林训练与 OOB 评分")
print("=" * 60)

# 不同 n_estimators 的 OOB 评分
n_estimators_range = [10, 50, 100, 200, 300, 500]
oob_scores = []
test_scores = []

for n in n_estimators_range:
    rf = RandomForestClassifier(
        n_estimators=n, oob_score=True, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    oob_scores.append(rf.oob_score_)
    test_scores.append(accuracy_score(y_test, rf.predict(X_test)))
    print(f"  n_estimators={n:>3d}  OOB={rf.oob_score_:.4f}  Test={test_scores[-1]:.4f}")

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(n_estimators_range, oob_scores, 'o-', label='OOB 评分', color='#FF6B6B', linewidth=2)
ax.plot(n_estimators_range, test_scores, 's-', label='测试集准确率', color='#4ECDC4', linewidth=2)
ax.set_xlabel('n_estimators（树的数量）', fontsize=12)
ax.set_ylabel('准确率', fontsize=12)
ax.set_title('随机森林: OOB 评分 vs 树的数量', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("D:/code/big-model-learn/code/q_01/W6/d2_oob_scores.png", dpi=150, bbox_inches='tight')
plt.close()
print("OOB评分图已保存")

# ================================================================
# 3. 随机森林特征重要性
# ================================================================
print("\n" + "=" * 60)
print("3. 随机森林特征重要性 (Top 10)")
print("=" * 60)

rf_best = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_best.fit(X_train, y_train)

importances = rf_best.feature_importances_
indices = np.argsort(importances)[::-1][:10]  # Top 10

print("Top 10 重要特征:")
for rank, idx in enumerate(indices, 1):
    print(f"  {rank:>2d}. {feature_names[idx]}: {importances[idx]:.4f}")

fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(indices)))[::-1]
ax.barh(range(len(indices)), importances[indices], color=colors)
ax.set_yticks(range(len(indices)))
ax.set_yticklabels([feature_names[i] for i in indices], fontsize=10)
ax.set_xlabel('特征重要性', fontsize=12)
ax.set_title('随机森林 Top 10 特征重要性', fontsize=14, fontweight='bold')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig("D:/code/big-model-learn/code/q_01/W6/d2_rf_importance.png", dpi=150, bbox_inches='tight')
plt.close()
print("特征重要性图已保存")

# ================================================================
# 4. 集成模型对比
# ================================================================
print("\n" + "=" * 60)
print("4. 集成模型对比")
print("=" * 60)

models = {
    'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
}

if HAS_XGBOOST:
    models['XGBoost'] = XGBClassifier(
        n_estimators=200, random_state=42, use_label_encoder=False,
        eval_metric='logloss', verbosity=0
    )

if HAS_LIGHTGBM:
    models['LightGBM'] = LGBMClassifier(
        n_estimators=200, random_state=42, verbose=-1
    )

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = {'accuracy': acc, 'model': model}
    print(f"\n--- {name} ---")
    print(f"  测试集准确率: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['恶性', '良性'], digits=4))

# ================================================================
# 5. 交叉验证对比
# ================================================================
print("\n" + "=" * 60)
print("5. 交叉验证对比 (5-Fold)")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    cv_results[name] = scores
    print(f"  {name:>20s}: {scores.mean():.4f} +/- {scores.std():.4f}")

# ================================================================
# 6. 模型对比柱状图
# ================================================================
print("\n" + "=" * 60)
print("6. 模型对比可视化")
print("=" * 60)

model_names = list(cv_results.keys())
means = [cv_results[n].mean() for n in model_names]
stds = [cv_results[n].std() for n in model_names]
test_accs = [results[n]['accuracy'] for n in model_names]

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(model_names))
width = 0.35

bars1 = ax.bar(x - width / 2, means, width, yerr=stds,
               label='交叉验证准确率', color='#4ECDC4', edgecolor='white',
               capsize=5, error_kw={'linewidth': 1.5})
bars2 = ax.bar(x + width / 2, test_accs, width,
               label='测试集准确率', color='#FF6B6B', edgecolor='white')

ax.set_ylabel('准确率', fontsize=12)
ax.set_title('集成模型对比', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=11)
ax.legend(fontsize=11)
ax.set_ylim(0.9, 1.0)
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height + 0.002,
            f'{height:.3f}', ha='center', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height + 0.002,
            f'{height:.3f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig("D:/code/big-model-learn/code/q_01/W6/d2_model_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("模型对比图已保存")

# ================================================================
# 总结
# ================================================================
print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print("1. 随机森林通过 Bagging + 随机特征选择降低方差")
print("2. OOB 评分是无偏的泛化性能估计，不需要额外验证集")
print("3. GradientBoosting 通过串行训练残差降低偏差")
print("4. XGBoost/LightGBM 是 GradientBoosting 的高效实现")
print("5. 集成方法通常优于单一决策树")
