"""
W06 Day7: 项目2 - 客户流失预测 (模型对比 + 调参 + 评估)
======================================================
内容:
1. 加载 d6 预处理后的数据（或重新生成）
2. 构建 Pipeline: 预处理 + 模型
3. 对比 LogisticRegression, RandomForest, GradientBoosting
4. 处理不平衡 (class_weight)
5. GridSearchCV 调参
6. 最终评估: 混淆矩阵、分类报告、ROC-AUC
7. 特征重要性分析
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    classification_report, confusion_matrix, roc_curve, auc,
    roc_auc_score
)

# ========== matplotlib 中文显示设置 ==========
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ================================================================
# 1. 数据准备 (与 d6 相同的数据生成逻辑，确保一致性)
# ================================================================
print("=" * 60)
print("1. 数据准备")
print("=" * 60)

# 尝试加载 d6 保存的数据
data_path = "D:/code/big-model-learn/code/q_01/W6/d6_churn_data.npz"
try:
    data = np.load(data_path, allow_pickle=True)
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    feature_names = data['feature_names']
    print(f"从 d6_churn_data.npz 加载成功")
except FileNotFoundError:
    print("d6 数据文件未找到，重新生成数据...")
    # 重新生成与 d6 相同的数据
    np.random.seed(42)
    n_samples = 2000

    tenure = np.clip(np.random.exponential(scale=30, size=n_samples).astype(int), 1, 72)
    monthly_charges = np.round(np.random.uniform(20, 120, n_samples), 2)
    total_charges = np.round(tenure * monthly_charges * np.random.uniform(0.8, 1.2, n_samples), 2)
    age = np.random.randint(18, 75, n_samples)
    num_dependents = np.random.choice([0, 0, 0, 1, 1, 2, 3], size=n_samples)
    contract_type = np.random.choice(['月付', '一年', '两年'], size=n_samples, p=[0.5, 0.3, 0.2])
    internet_service = np.random.choice(['DSL', '光纤', '无'], size=n_samples, p=[0.35, 0.45, 0.2])
    payment_method = np.random.choice(['电子支票', '邮寄支票', '银行转账', '信用卡'],
                                       size=n_samples, p=[0.35, 0.2, 0.25, 0.2])
    online_security = np.random.choice(['是', '否'], size=n_samples, p=[0.4, 0.6])
    tech_support = np.random.choice(['是', '否'], size=n_samples, p=[0.35, 0.65])
    gender = np.random.choice(['男', '女'], size=n_samples, p=[0.5, 0.5])

    churn_prob = (
        0.1 - 0.008 * tenure + 0.005 * monthly_charges
        + (contract_type == '月付') * 0.25 - (contract_type == '两年') * 0.2
        + (internet_service == '光纤') * 0.1 + (online_security == '否') * 0.1
        + (tech_support == '否') * 0.08 + (payment_method == '电子支票') * 0.12
        + 0.003 * (age - 40) ** 2 / 100
    )
    churn_prob = np.clip(churn_prob, 0.02, 0.9)
    churn = (np.random.random(n_samples) < churn_prob).astype(int)

    df = pd.DataFrame({
        '性别': gender, '年龄': age, '在网时长(月)': tenure,
        '月费(元)': monthly_charges, '总费用(元)': total_charges,
        '家属数': num_dependents, '合同类型': contract_type,
        '网络服务': internet_service, '支付方式': payment_method,
        '在线安全': online_security, '技术支持': tech_support, '是否流失': churn,
    })

    X = df.drop('是否流失', axis=1)
    y = df['是否流失']

    numeric_features = ['年龄', '在网时长(月)', '月费(元)', '总费用(元)', '家属数']
    categorical_features = ['性别', '合同类型', '网络服务', '支付方式', '在线安全', '技术支持']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'),
             categorical_features),
        ]
    )

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    num_names = numeric_features
    cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    feature_names = list(num_names) + list(cat_names)

print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
print(f"特征数: {len(feature_names)}")
print(f"训练集流失率: {y_train.mean() * 100:.1f}%")
print(f"测试集流失率: {y_test.mean() * 100:.1f}%")

# ================================================================
# 2. 基线模型对比
# ================================================================
print("\n" + "=" * 60)
print("2. 基线模型对比")
print("=" * 60)

models = {
    'LogisticRegression': LogisticRegression(
        max_iter=5000, random_state=42, class_weight='balanced'
    ),
    'RandomForest': RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight='balanced', n_jobs=-1
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=200, random_state=42
    ),
}

baseline_results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    # 尝试获取预测概率
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        roc_auc = 0.0

    baseline_results[name] = {
        'accuracy': acc, 'f1': f1, 'recall': recall,
        'precision': precision, 'roc_auc': roc_auc,
        'y_pred': y_pred, 'y_proba': y_proba if hasattr(model, 'predict_proba') else None,
    }

    print(f"\n--- {name} ---")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")

# ================================================================
# 3. 交叉验证对比
# ================================================================
print("\n" + "=" * 60)
print("3. 交叉验证对比 (5-Fold)")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
    cv_results[name] = scores
    print(f"  {name:>22s}: F1={scores.mean():.4f} +/- {scores.std():.4f}")

# ================================================================
# 4. 模型对比可视化
# ================================================================
print("\n" + "=" * 60)
print("4. 模型对比可视化")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 柱状图对比各指标
metrics_to_plot = ['accuracy', 'f1', 'recall', 'precision', 'roc_auc']
model_names = list(baseline_results.keys())
x = np.arange(len(metrics_to_plot))
width = 0.25
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

for i, name in enumerate(model_names):
    values = [baseline_results[name][m] for m in metrics_to_plot]
    bars = axes[0].bar(x + i * width, values, width, label=name,
                       color=colors[i], edgecolor='white')
    for bar, val in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', fontsize=7, rotation=45)

axes[0].set_ylabel('分数', fontsize=12)
axes[0].set_title('模型性能指标对比', fontsize=13, fontweight='bold')
axes[0].set_xticks(x + width)
axes[0].set_xticklabels(['Accuracy', 'F1', 'Recall', 'Precision', 'ROC-AUC'], fontsize=10)
axes[0].legend(fontsize=9)
axes[0].set_ylim(0, 1.15)
axes[0].grid(True, alpha=0.3, axis='y')

# 交叉验证箱线图
bp = axes[1].boxplot([cv_results[n] for n in model_names],
                     labels=model_names, patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[1].set_ylabel('F1 Score', fontsize=12)
axes[1].set_title('5折交叉验证 F1 分数', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

plt.suptitle('客户流失预测: 模型对比', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("D:/code/big-model-learn/code/q_01/W6/d7_model_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("模型对比图已保存")

# ================================================================
# 5. GridSearchCV 调参 (以 RandomForest 为例)
# ================================================================
print("\n" + "=" * 60)
print("5. GridSearchCV 调参 (RandomForest)")
print("=" * 60)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'class_weight': ['balanced', None],
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    refit=True,
    verbose=0
)
grid_search.fit(X_train, y_train)

print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证 F1: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
y_proba_best = best_model.predict_proba(X_test)[:, 1]

print(f"测试集 F1: {f1_score(y_test, y_pred_best):.4f}")
print(f"测试集 ROC-AUC: {roc_auc_score(y_test, y_proba_best):.4f}")

# ================================================================
# 6. 最终评估
# ================================================================
print("\n" + "=" * 60)
print("6. 最终模型评估")
print("=" * 60)

print("\n分类报告:")
print(classification_report(y_test, y_pred_best,
                            target_names=['未流失', '已流失'], digits=4))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['未流失', '已流失'],
            yticklabels=['未流失', '已流失'])
axes[0].set_xlabel('预测标签', fontsize=12)
axes[0].set_ylabel('真实标签', fontsize=12)
axes[0].set_title('混淆矩阵 (最优模型)', fontsize=13, fontweight='bold')

# 添加百分比
total = cm.sum()
for i in range(2):
    for j in range(2):
        pct = cm[i, j] / total * 100
        axes[0].text(j + 0.5, i + 0.7, f'({pct:.1f}%)',
                     ha='center', fontsize=10, color='gray')

# ROC 曲线
fpr, tpr, thresholds = roc_curve(y_test, y_proba_best)
roc_auc_val = auc(fpr, tpr)

axes[1].plot(fpr, tpr, color='#FF6B6B', linewidth=2,
             label=f'ROC 曲线 (AUC = {roc_auc_val:.4f})')
axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='随机猜测')
axes[1].fill_between(fpr, tpr, alpha=0.15, color='#FF6B6B')
axes[1].set_xlabel('假正率 (FPR)', fontsize=12)
axes[1].set_ylabel('真正率 (TPR)', fontsize=12)
axes[1].set_title('ROC 曲线', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11, loc='lower right')
axes[1].grid(True, alpha=0.3)

plt.suptitle('客户流失预测: 最终模型评估', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("D:/code/big-model-learn/code/q_01/W6/d7_evaluation.png", dpi=150, bbox_inches='tight')
plt.close()
print("评估图已保存")

# ================================================================
# 7. 多模型 ROC 曲线对比
# ================================================================
print("\n" + "=" * 60)
print("7. 多模型 ROC 曲线对比")
print("=" * 60)

fig, ax = plt.subplots(figsize=(8, 7))

for name, color in zip(model_names, colors):
    y_proba = baseline_results[name]['y_proba']
    if y_proba is not None:
        fpr_i, tpr_i, _ = roc_curve(y_test, y_proba)
        auc_i = auc(fpr_i, tpr_i)
        ax.plot(fpr_i, tpr_i, color=color, linewidth=2,
                label=f'{name} (AUC={auc_i:.4f})')

# 最优模型
fpr_best, tpr_best, _ = roc_curve(y_test, y_proba_best)
auc_best = auc(fpr_best, tpr_best)
ax.plot(fpr_best, tpr_best, color='green', linewidth=2.5, linestyle='--',
        label=f'最优RF (AUC={auc_best:.4f})')

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
ax.set_xlabel('假正率 (FPR)', fontsize=12)
ax.set_ylabel('真正率 (TPR)', fontsize=12)
ax.set_title('多模型 ROC 曲线对比', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("D:/code/big-model-learn/code/q_01/W6/d7_roc_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("ROC对比图已保存")

# ================================================================
# 8. 特征重要性分析
# ================================================================
print("\n" + "=" * 60)
print("8. 特征重要性分析")
print("=" * 60)

# RandomForest 特征重要性
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]

top_n = 15
print(f"Top {top_n} 重要特征:")
for i, idx in enumerate(indices[:top_n], 1):
    fname = feature_names[idx] if idx < len(feature_names) else f'特征{idx}'
    print(f"  {i:>2d}. {fname}: {importances[idx]:.4f}")

fig, ax = plt.subplots(figsize=(10, 7))

top_indices = indices[:top_n]
top_importances = importances[top_indices]
top_names = [feature_names[i] if i < len(feature_names) else f'特征{i}' for i in top_indices]

colors_bar = plt.cm.RdYlGn(np.linspace(0.2, 0.8, top_n))[::-1]
bars = ax.barh(range(top_n), top_importances, color=colors_bar, edgecolor='white')
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_names, fontsize=10)
ax.set_xlabel('特征重要性', fontsize=12)
ax.set_title(f'客户流失预测: Top {top_n} 重要特征', fontsize=14, fontweight='bold')
ax.invert_yaxis()

for bar, val in zip(bars, top_importances):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f'{val:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig("D:/code/big-model-learn/code/q_01/W6/d7_feature_importance.png", dpi=150, bbox_inches='tight')
plt.close()
print("特征重要性图已保存")

# LogisticRegression 系数分析
print("\n--- LogisticRegression 系数分析 ---")
lr_model = models['LogisticRegression']
lr_coefs = lr_model.coef_[0]
lr_indices = np.argsort(np.abs(lr_coefs))[::-1]

print("Top 10 特征系数 (绝对值排序):")
for i, idx in enumerate(lr_indices[:10], 1):
    fname = feature_names[idx] if idx < len(feature_names) else f'特征{idx}'
    direction = "正向(促进流失)" if lr_coefs[idx] > 0 else "负向(减少流失)"
    print(f"  {i:>2d}. {fname}: {lr_coefs[idx]:+.4f} ({direction})")

# ================================================================
# 总结
# ================================================================
print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print("1. 对比了 LogisticRegression、RandomForest、GradientBoosting 三个模型")
print("2. 使用 class_weight='balanced' 处理不平衡数据")
print("3. GridSearchCV 系统搜索最优超参数")
print("4. 最终评估使用混淆矩阵、分类报告和 ROC-AUC")
print("5. 特征重要性分析揭示影响客户流失的关键因素")
print("6. 在网时长、合同类型、月费是影响流失的核心特征")
print("7. 完整流程: EDA -> 特征工程 -> 模型对比 -> 调参 -> 评估")
