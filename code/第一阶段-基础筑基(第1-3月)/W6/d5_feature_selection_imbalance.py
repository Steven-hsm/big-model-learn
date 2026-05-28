"""
W06 Day5: 特征选择 + 不平衡数据处理
====================================
内容:
1. 特征选择方法:
   - VarianceThreshold
   - SelectKBest + mutual_info_classif
   - RandomForest 特征重要性
   - RFE (Recursive Feature Elimination)
2. 不平衡数据处理:
   - 生成不平衡数据集
   - 类别分布可视化
   - class_weight='balanced'
   - SMOTE 过采样 (try/except)
   - 对比 accuracy / F1 / recall
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, mutual_info_classif, RFE
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler

# 尝试导入 imblearn
try:
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    print("[提示] imblearn 未安装，跳过 SMOTE 演示")
    print("       安装命令: pip install imbalanced-learn")

# ========== matplotlib 中文显示设置 ==========
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ================================================================
# Part 1: 特征选择
# ================================================================
print("=" * 60)
print("Part 1: 特征选择方法")
print("=" * 60)

# 使用 breast_cancer 数据集
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"原始特征数: {X.shape[1]}")

# --------------------------------------------------
# 1.1 VarianceThreshold
# --------------------------------------------------
print("\n--- 1.1 VarianceThreshold ---")

variances = np.var(X_train, axis=0)
print(f"各特征方差范围: [{variances.min():.4f}, {variances.max():.4f}]")

thresholds = [0.0, 0.1, 1.0, 10.0]
for threshold in thresholds:
    selector = VarianceThreshold(threshold=threshold)
    X_selected = selector.fit_transform(X_train)
    n_kept = X_selected.shape[1]
    n_removed = X_train.shape[1] - n_kept
    print(f"  threshold={threshold:>6.1f} -> 保留 {n_kept} 个特征, 移除 {n_removed} 个")

# --------------------------------------------------
# 1.2 SelectKBest + mutual_info_classif
# --------------------------------------------------
print("\n--- 1.2 SelectKBest (mutual_info_classif) ---")

mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
mi_indices = np.argsort(mi_scores)[::-1]

print("互信息 Top 10 特征:")
for i, idx in enumerate(mi_indices[:10], 1):
    print(f"  {i:>2d}. {feature_names[idx]}: {mi_scores[idx]:.4f}")

# 不同 K 值对准确率的影响
k_values = [5, 10, 15, 20, 25, 30]
accs_kbest = []

for k in k_values:
    selector = SelectKBest(mutual_info_classif, k=k)
    X_tr_sel = selector.fit_transform(X_train, y_train)
    X_te_sel = selector.transform(X_test)

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_tr_sel, y_train)
    acc = accuracy_score(y_test, clf.predict(X_te_sel))
    accs_kbest.append(acc)
    print(f"  K={k:>2d} -> 准确率={acc:.4f}")

# --------------------------------------------------
# 1.3 RandomForest 特征重要性
# --------------------------------------------------
print("\n--- 1.3 RandomForest 特征重要性 ---")

rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

rf_importances = rf.feature_importances_
rf_indices = np.argsort(rf_importances)[::-1]

print("RF 特征重要性 Top 10:")
for i, idx in enumerate(rf_indices[:10], 1):
    print(f"  {i:>2d}. {feature_names[idx]}: {rf_importances[idx]:.4f}")

# 累积重要性
cumulative = np.cumsum(rf_importances[rf_indices])
n_90 = np.searchsorted(cumulative, 0.9) + 1
print(f"达到 90% 累积重要性需要 {n_90} 个特征")

# --------------------------------------------------
# 1.4 RFE (Recursive Feature Elimination)
# --------------------------------------------------
print("\n--- 1.4 RFE (递归特征消除) ---")

estimator = LogisticRegression(max_iter=5000, random_state=42)
n_features_to_select_list = [5, 10, 15, 20]
accs_rfe = []

for n in n_features_to_select_list:
    rfe = RFE(estimator, n_features_to_select=n, step=2)
    X_tr_rfe = rfe.fit_transform(X_train, y_train)
    X_te_rfe = rfe.transform(X_test)

    clf = LogisticRegression(max_iter=5000, random_state=42)
    clf.fit(X_tr_rfe, y_train)
    acc = accuracy_score(y_test, clf.predict(X_te_rfe))
    accs_rfe.append(acc)

    selected = [feature_names[i] for i in range(len(rfe.support_)) if rfe.support_[i]]
    print(f"  选择 {n} 个特征 -> 准确率={acc:.4f}")
    if n == 5:
        print(f"    选中: {selected}")

# 特征选择方法对比图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# SelectKBest
axes[0].plot(k_values, accs_kbest, 'o-', color='#4ECDC4', linewidth=2, markersize=8)
axes[0].axhline(y=accuracy_score(y_test, rf.predict(X_test)), color='red',
                linestyle='--', alpha=0.7, label='全部特征(RF)')
axes[0].set_xlabel('选择的特征数 K', fontsize=12)
axes[0].set_ylabel('准确率', fontsize=12)
axes[0].set_title('SelectKBest: 特征数 vs 准确率', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# RFE
axes[1].plot(n_features_to_select_list, accs_rfe, 's-', color='#FF6B6B', linewidth=2, markersize=8)
axes[1].axhline(y=accuracy_score(y_test, LogisticRegression(max_iter=5000, random_state=42).fit(X_train, y_train).predict(X_test)),
                color='red', linestyle='--', alpha=0.7, label='全部特征(LR)')
axes[1].set_xlabel('选择的特征数', fontsize=12)
axes[1].set_ylabel('准确率', fontsize=12)
axes[1].set_title('RFE: 特征数 vs 准确率', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.suptitle('特征选择方法对比', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("D:/code/big-model-learn/code/q_01/W6/d5_feature_selection.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n特征选择对比图已保存")

# ================================================================
# Part 2: 不平衡数据处理
# ================================================================
print("\n" + "=" * 60)
print("Part 2: 不平衡数据处理")
print("=" * 60)

# --------------------------------------------------
# 2.1 生成不平衡数据集
# --------------------------------------------------
print("\n--- 2.1 生成不平衡数据集 ---")

X_imb, y_imb = make_classification(
    n_samples=2000,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    n_classes=2,
    weights=[0.9, 0.1],  # 90% vs 10%
    random_state=42
)

print(f"数据形状: {X_imb.shape}")
print(f"类别分布: 类0={np.sum(y_imb == 0)} ({np.mean(y_imb == 0) * 100:.1f}%), "
      f"类1={np.sum(y_imb == 1)} ({np.mean(y_imb == 1) * 100:.1f}%)")

X_tr_imb, X_te_imb, y_tr_imb, y_te_imb = train_test_split(
    X_imb, y_imb, test_size=0.3, random_state=42
)

# 类别分布可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 原始分布
classes, counts = np.unique(y_imb, return_counts=True)
colors = ['#4ECDC4', '#FF6B6B']
axes[0].bar(['类 0 (多数类)', '类 1 (少数类)'], counts, color=colors, edgecolor='white')
for i, (c, n) in enumerate(zip(classes, counts)):
    axes[0].text(i, n + 20, f'{n} ({n / len(y_imb) * 100:.1f}%)',
                 ha='center', fontsize=11, fontweight='bold')
axes[0].set_title('不平衡数据集类别分布', fontsize=13, fontweight='bold')
axes[0].set_ylabel('样本数', fontsize=12)

# SMOTE 后的分布 (如果可用)
if HAS_IMBLEARN:
    smote = SMOTE(random_state=42)
    X_tr_smote, y_tr_smote = smote.fit_resample(X_tr_imb, y_tr_imb)
    classes_smote, counts_smote = np.unique(y_tr_smote, return_counts=True)
    axes[1].bar(['类 0 (多数类)', '类 1 (少数类)'], counts_smote, color=colors, edgecolor='white')
    for i, (c, n) in enumerate(zip(classes_smote, counts_smote)):
        axes[1].text(i, n + 20, f'{n} ({n / len(y_tr_smote) * 100:.1f}%)',
                     ha='center', fontsize=11, fontweight='bold')
    axes[1].set_title('SMOTE 过采样后类别分布', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('样本数', fontsize=12)
else:
    axes[1].text(0.5, 0.5, 'imblearn 未安装\n请运行: pip install imbalanced-learn',
                 ha='center', va='center', fontsize=14, transform=axes[1].transAxes)
    axes[1].set_title('SMOTE 过采样 (需要 imblearn)', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig("D:/code/big-model-learn/code/q_01/W6/d5_class_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print("类别分布图已保存")

# --------------------------------------------------
# 2.2 不同策略对比
# --------------------------------------------------
print("\n--- 2.2 不平衡数据不同处理策略对比 ---")

# 标准化
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr_imb)
X_te_scaled = scaler.transform(X_te_imb)

strategies = {}

# (1) 基线: 不处理不平衡
rf_baseline = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_baseline.fit(X_tr_scaled, y_tr_imb)
y_pred_base = rf_baseline.predict(X_te_scaled)
strategies['基线(不处理)'] = {
    'accuracy': accuracy_score(y_te_imb, y_pred_base),
    'f1': f1_score(y_te_imb, y_pred_base),
    'recall': recall_score(y_te_imb, y_pred_base),
    'precision': precision_score(y_te_imb, y_pred_base),
}

# (2) class_weight='balanced'
rf_balanced = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                     random_state=42, n_jobs=-1)
rf_balanced.fit(X_tr_scaled, y_tr_imb)
y_pred_bal = rf_balanced.predict(X_te_scaled)
strategies['class_weight=balanced'] = {
    'accuracy': accuracy_score(y_te_imb, y_pred_bal),
    'f1': f1_score(y_te_imb, y_pred_bal),
    'recall': recall_score(y_te_imb, y_pred_bal),
    'precision': precision_score(y_te_imb, y_pred_bal),
}

# (3) SMOTE
if HAS_IMBLEARN:
    rf_smote = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf_smote.fit(X_tr_smote, y_tr_smote)
    y_pred_smote = rf_smote.predict(X_te_scaled)
    strategies['SMOTE过采样'] = {
        'accuracy': accuracy_score(y_te_imb, y_pred_smote),
        'f1': f1_score(y_te_imb, y_pred_smote),
        'recall': recall_score(y_te_imb, y_pred_smote),
        'precision': precision_score(y_te_imb, y_pred_smote),
    }

# 打印结果
print(f"\n{'策略':>25s} | {'Accuracy':>8s} | {'F1':>8s} | {'Recall':>8s} | {'Precision':>9s}")
print("-" * 70)
for name, metrics in strategies.items():
    print(f"{name:>25s} | {metrics['accuracy']:>8.4f} | {metrics['f1']:>8.4f} | "
          f"{metrics['recall']:>8.4f} | {metrics['precision']:>9.4f}")

# 可视化
fig, ax = plt.subplots(figsize=(12, 6))

metric_names = ['accuracy', 'f1', 'recall', 'precision']
x = np.arange(len(metric_names))
width = 0.2
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

for i, (name, metrics) in enumerate(strategies.items()):
    values = [metrics[m] for m in metric_names]
    bars = ax.bar(x + i * width, values, width, label=name, color=colors[i], edgecolor='white')

ax.set_ylabel('分数', fontsize=12)
ax.set_title('不平衡数据: 不同处理策略的指标对比', fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(['Accuracy', 'F1 Score', 'Recall', 'Precision'], fontsize=11)
ax.legend(fontsize=10)
ax.set_ylim(0, 1.1)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("D:/code/big-model-learn/code/q_01/W6/d5_imbalance_strategies.png", dpi=150, bbox_inches='tight')
plt.close()
print("不平衡策略对比图已保存")

# --------------------------------------------------
# 2.3 为什么 Accuracy 不够用
# --------------------------------------------------
print("\n--- 2.3 为什么不平衡数据不能用 Accuracy ---")

# 全部预测为多数类
y_pred_all_majority = np.zeros_like(y_te_imb)
acc_majority = accuracy_score(y_te_imb, y_pred_all_majority)
f1_majority = f1_score(y_te_imb, y_pred_all_majority)
recall_majority = recall_score(y_te_imb, y_pred_all_majority)

print(f"全部预测为多数类:")
print(f"  Accuracy = {acc_majority:.4f} (看起来很高! 但毫无意义)")
print(f"  F1 Score = {f1_majority:.4f} (正确反映模型无效)")
print(f"  Recall   = {recall_majority:.4f} (少数类完全被忽略)")
print()
print("结论: 不平衡数据应使用 F1/Recall/AUC 等指标，而非 Accuracy")

# ================================================================
# 总结
# ================================================================
print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print("特征选择:")
print("  1. VarianceThreshold: 移除低方差特征（最简单）")
print("  2. SelectKBest: 基于统计检验选择 Top-K 特征")
print("  3. RF 特征重要性: 基于模型的特征选择")
print("  4. RFE: 递归消除最不重要特征，逐步精简")
print("不平衡数据:")
print("  5. class_weight='balanced': 自动调整类别权重，最简单")
print("  6. SMOTE: 合成少数类样本，增加少数类数据量")
print("  7. 评估不平衡数据应用 F1/Recall 而非 Accuracy")
