"""
W06 Day1: 决策树原理 + sklearn实战
===============================
内容:
1. 加载Iris数据集，划分训练/测试集
2. 不同max_depth的决策树训练与可视化
3. plot_tree可视化决策树结构
4. 特征重要性柱状图
5. accuracy vs max_depth 曲线（展示过拟合）
6. 成本复杂度剪枝 (ccp_alpha)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# ========== matplotlib 中文显示设置 ==========
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ================================================================
# 1. 加载数据与划分
# ================================================================
print("=" * 60)
print("1. 加载 Iris 数据��")
print("=" * 60)

iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print(f"特征名称: {feature_names}")
print(f"类别名称: {target_names}")
print(f"数据形状: {X.shape}")
print(f"类别分布: {np.bincount(y)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

# ================================================================
# 2. 训练决策树（默认参数）
# ================================================================
print("\n" + "=" * 60)
print("2. 训练默认决策树")
print("=" * 60)

dt_default = DecisionTreeClassifier(random_state=42)
dt_default.fit(X_train, y_train)

y_pred_default = dt_default.predict(X_test)
acc_default = accuracy_score(y_test, y_pred_default)
print(f"默认参数决策树深度: {dt_default.get_depth()}")
print(f"默认参数叶子节点数: {dt_default.get_n_leaves()}")
print(f"默认参数测试集准确率: {acc_default:.4f}")

# ================================================================
# 3. 可视化决策树结构（不同 max_depth）
# ================================================================
print("\n" + "=" * 60)
print("3. 可视化不同 max_depth 的决策树")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
depths = [2, 3, None]  # None 表示不限制

for ax, depth in zip(axes, depths):
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    acc = accuracy_score(y_test, dt.predict(X_test))

    plot_tree(
        dt,
        feature_names=feature_names,
        class_names=target_names,
        filled=True,
        rounded=True,
        ax=ax,
        fontsize=8
    )
    depth_label = depth if depth is not None else "不限"
    ax.set_title(f"max_depth={depth_label}\n准确率={acc:.4f}", fontsize=12)

plt.suptitle("不同 max_depth 的决策树结构对比", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("D:/code/big-model-learn/code/q_01/W6/d1_tree_structures.png", dpi=150, bbox_inches='tight')
plt.close()
print("决策树结构图已保存")

# ================================================================
# 4. 特征重要性柱状图
# ================================================================
print("\n" + "=" * 60)
print("4. 特征重要性分析")
print("=" * 60)

dt_full = DecisionTreeClassifier(random_state=42)
dt_full.fit(X_train, y_train)

importances = dt_full.feature_importances_
indices = np.argsort(importances)[::-1]

print("特征重要性排名:")
for i, idx in enumerate(indices):
    print(f"  {i + 1}. {feature_names[idx]}: {importances[idx]:.4f}")

fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
bars = ax.bar(range(len(importances)), importances[indices], color=colors)
ax.set_xticks(range(len(importances)))
ax.set_xticklabels([feature_names[i] for i in indices], rotation=15, ha='right')
ax.set_ylabel('重要性', fontsize=12)
ax.set_title('决策树特征重要性', fontsize=14, fontweight='bold')

for bar, val in zip(bars, importances[indices]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig("D:/code/big-model-learn/code/q_01/W6/d1_feature_importance.png", dpi=150, bbox_inches='tight')
plt.close()
print("特征重要性图已保存")

# ================================================================
# 5. accuracy vs max_depth（展示过拟合）
# ================================================================
print("\n" + "=" * 60)
print("5. 准确率 vs max_depth（过拟合分析）")
print("=" * 60)

max_depths = range(1, 11)
train_accs = []
test_accs = []

for d in max_depths:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt.fit(X_train, y_train)
    train_accs.append(accuracy_score(y_train, dt.predict(X_train)))
    test_accs.append(accuracy_score(y_test, dt.predict(X_test)))

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(max_depths, train_accs, 'o-', label='训练集准确率', color='#FF6B6B', linewidth=2)
ax.plot(max_depths, test_accs, 's-', label='测试集准确率', color='#4ECDC4', linewidth=2)
ax.fill_between(max_depths, train_accs, test_accs, alpha=0.15, color='red', label='过拟合区域')
ax.set_xlabel('max_depth', fontsize=12)
ax.set_ylabel('准确率', fontsize=12)
ax.set_title('决策树: 准确率 vs max_depth（过拟合分析）', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.set_xticks(list(max_depths))
ax.grid(True, alpha=0.3)

best_depth = list(max_depths)[np.argmax(test_accs)]
ax.axvline(x=best_depth, color='green', linestyle='--', alpha=0.7, label=f'最佳深度={best_depth}')
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig("D:/code/big-model-learn/code/q_01/W6/d1_overfitting.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"最佳 max_depth = {best_depth}, 测试集准确率 = {max(test_accs):.4f}")
print("过拟合分析图已保存")

# ================================================================
# 6. 成本复杂度剪枝 (Cost Complexity Pruning)
# ================================================================
print("\n" + "=" * 60)
print("6. 成本复杂度剪枝 (ccp_alpha)")
print("=" * 60)

# 获取剪枝路径
dt_path = DecisionTreeClassifier(random_state=42)
path = dt_path.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
impurities = path.impurities

print(f"ccp_alpha 数量: {len(ccp_alphas)}")
print(f"ccp_alpha 范围: [{ccp_alphas[0]:.6f}, {ccp_alphas[-1]:.6f}]")

# 训练不同 ccp_alpha 的决策树
ccp_alphas_filtered = ccp_alphas[:-1]  # 去掉最后一个（只剩根节点）
train_accs_prune = []
test_accs_prune = []
node_counts = []

for alpha in ccp_alphas_filtered:
    dt = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
    dt.fit(X_train, y_train)
    train_accs_prune.append(accuracy_score(y_train, dt.predict(X_train)))
    test_accs_prune.append(accuracy_score(y_test, dt.predict(X_test)))
    node_counts.append(dt.tree_.node_count)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 准确率 vs ccp_alpha
axes[0].plot(ccp_alphas_filtered, train_accs_prune, 'o-', label='训练集', color='#FF6B6B', markersize=4)
axes[0].plot(ccp_alphas_filtered, test_accs_prune, 's-', label='测试集', color='#4ECDC4', markersize=4)
axes[0].set_xlabel('ccp_alpha', fontsize=12)
axes[0].set_ylabel('准确率', fontsize=12)
axes[0].set_title('准确率 vs ccp_alpha', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# 标记最佳 ccp_alpha
best_idx = np.argmax(test_accs_prune)
best_alpha = ccp_alphas_filtered[best_idx]
axes[0].axvline(x=best_alpha, color='green', linestyle='--', alpha=0.7)
axes[0].annotate(f'最佳 alpha={best_alpha:.4f}',
                 xy=(best_alpha, test_accs_prune[best_idx]),
                 xytext=(best_alpha + 0.005, test_accs_prune[best_idx] - 0.05),
                 arrowprops=dict(arrowstyle='->', color='green'),
                 fontsize=10, color='green')

# 节点数 vs ccp_alpha
axes[1].plot(ccp_alphas_filtered, node_counts, 'D-', color='#45B7D1', markersize=4)
axes[1].set_xlabel('ccp_alpha', fontsize=12)
axes[1].set_ylabel('节点数', fontsize=12)
axes[1].set_title('树节点数 vs ccp_alpha', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.suptitle('成本复杂度剪枝分析', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("D:/code/big-model-learn/code/q_01/W6/d1_pruning.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"最佳 ccp_alpha = {best_alpha:.6f}")
print(f"剪枝后测试集准确率 = {test_accs_prune[best_idx]:.4f}")
print(f"剪枝后节点数 = {node_counts[best_idx]}")
print("剪枝分析图已保存")

# ================================================================
# 总结
# ================================================================
print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print("1. 决策树通过递归分裂构建，max_depth 控制树深度")
print("2. 特征重要性可从训练好的树中直接获取")
print("3. max_depth 过大导致过拟合（训练集高、测试集低）")
print("4. 成本复杂度剪枝(ccp_alpha)是有效的正则化手段")
print("5. Iris 数据集简单，树深度 3 已能达到很好效果")
