"""
W06 Day3: SVM原理 + 核函数
==========================
内容:
1. 生成非线性可分数据 (make_moons)
2. 不同核函数的 SVM 对比: linear, rbf, poly
3. 每种核函数的决策边界可视化 (meshgrid)
4. GridSearchCV 搜索 C 和 gamma 参数
5. C 和 gamma 对决策边界的影响
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# ========== matplotlib 中文显示设置 ==========
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ================================================================
# 1. 生成非线性可分数据
# ================================================================
print("=" * 60)
print("1. 生成非线性可分数据")
print("=" * 60)

X_moons, y_moons = make_moons(n_samples=500, noise=0.25, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X_moons, y_moons, test_size=0.3, random_state=42
)

print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")
print(f"类别分布: 类0={np.sum(y_moons == 0)}, 类1={np.sum(y_moons == 1)}")


# 辅助函数: 绘制决策边界
def plot_decision_boundary(ax, model, X, y, title):
    """在二维空间绘制 SVM 决策边界"""
    h = 0.02  # 网格步长
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    ax.contour(xx, yy, Z, colors='black', linewidths=0.5, linestyles='--')

    # 绘制数据点
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu,
                         edgecolors='black', s=30, alpha=0.8)

    # 如果是 SVM，绘制支持向量
    if hasattr(model, 'support_vectors_'):
        ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                   s=100, linewidth=1.5, facecolors='none', edgecolors='green',
                   label=f'支持向量 ({len(model.support_vectors_)}个)')

    acc = accuracy_score(y, model.predict(X))
    ax.set_title(f"{title}\n准确率={acc:.4f}", fontsize=11)
    ax.set_xlabel('特征 1', fontsize=10)
    ax.set_ylabel('特征 2', fontsize=10)
    if hasattr(model, 'support_vectors_'):
        ax.legend(fontsize=8, loc='upper right')


# ================================================================
# 2. 不同核函数的 SVM 对比
# ================================================================
print("\n" + "=" * 60)
print("2. 不同核函数对比")
print("=" * 60)

kernels = {
    'linear': SVC(kernel='linear', C=1.0, random_state=42),
    'rbf': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
    'poly': SVC(kernel='poly', C=1.0, degree=3, random_state=42),
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (kernel_name, model) in zip(axes, kernels.items()):
    model.fit(X_train, y_train)
    plot_decision_boundary(ax, model, X_test, y_test,
                           f"SVM 核函数: {kernel_name}")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    n_sv = len(model.support_vectors_)
    print(f"  核函数={kernel_name:>8s}  准确率={acc:.4f}  支持向量数={n_sv}")

plt.suptitle('SVM 不同核函数的决策边界对比', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("D:/code/big-model-learn/code/q_01/W6/d3_kernels.png", dpi=150, bbox_inches='tight')
plt.close()
print("核函数对比图已保存")

# ================================================================
# 3. 另一个数据集: make_circles
# ================================================================
print("\n" + "=" * 60)
print("3. 同心圆数据上的核函数对比")
print("=" * 60)

X_circles, y_circles = make_circles(n_samples=500, noise=0.08, factor=0.5, random_state=42)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (kernel_name, _) in zip(axes, kernels.items()):
    model = SVC(kernel=kernel_name, C=1.0, random_state=42)
    if kernel_name == 'rbf':
        model.gamma = 'scale'
    model.fit(X_circles, y_circles)
    plot_decision_boundary(ax, model, X_circles, y_circles,
                           f"同心圆 - 核: {kernel_name}")
    print(f"  核={kernel_name:>8s}  准确率={accuracy_score(y_circles, model.predict(X_circles)):.4f}")

plt.suptitle('同心圆数据: 不同核函数的表现', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("D:/code/big-model-learn/code/q_01/W6/d3_circles.png", dpi=150, bbox_inches='tight')
plt.close()
print("同心圆对比图已保存")

# ================================================================
# 4. GridSearchCV 搜索最优参数
# ================================================================
print("\n" + "=" * 60)
print("4. GridSearchCV 搜索最优 C 和 gamma")
print("=" * 60)

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'gamma': ['scale', 0.001, 0.01, 0.1, 1, 10],
}

grid_search = GridSearchCV(
    SVC(kernel='rbf', random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    return_train_score=True
)
grid_search.fit(X_train, y_train)

print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证准确率: {grid_search.best_score_:.4f}")
print(f"测试集准确率: {accuracy_score(y_test, grid_search.predict(X_test)):.4f}")

# 可视化 GridSearch 结果
cv_results = grid_search.cv_results_
scores = cv_results['mean_test_score'].reshape(len(param_grid['C']), len(param_grid['gamma']))

fig, ax = plt.subplots(figsize=(8, 6))
gamma_labels = [str(g) for g in param_grid['gamma']]
im = ax.imshow(scores, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(gamma_labels)))
ax.set_xticklabels(gamma_labels, fontsize=10)
ax.set_yticks(range(len(param_grid['C'])))
ax.set_yticklabels(param_grid['C'], fontsize=10)
ax.set_xlabel('gamma', fontsize=12)
ax.set_ylabel('C', fontsize=12)
ax.set_title('GridSearchCV: C vs gamma 热力图', fontsize=14, fontweight='bold')

# 添加数值标注
for i in range(len(param_grid['C'])):
    for j in range(len(gamma_labels)):
        text_color = 'white' if scores[i, j] < 0.85 else 'black'
        ax.text(j, i, f'{scores[i, j]:.3f}', ha='center', va='center',
                fontsize=9, color=text_color)

plt.colorbar(im, ax=ax, label='交叉验证准确率')
plt.tight_layout()
plt.savefig("D:/code/big-model-learn/code/q_01/W6/d3_gridsearch.png", dpi=150, bbox_inches='tight')
plt.close()
print("GridSearch 热力图已保存")

# ================================================================
# 5. C 和 gamma 对决策边界的影响
# ================================================================
print("\n" + "=" * 60)
print("5. C 和 gamma 对决策边界的影响")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 第一行: 固定 gamma, 改变 C
C_values = [0.1, 1.0, 100.0]
gamma_fixed = 1.0

for i, C_val in enumerate(C_values):
    model = SVC(kernel='rbf', C=C_val, gamma=gamma_fixed, random_state=42)
    model.fit(X_train, y_train)
    plot_decision_boundary(axes[0, i], model, X_test, y_test,
                           f"C={C_val}, gamma={gamma_fixed}")
    print(f"  C={C_val:>6.1f}, gamma={gamma_fixed} -> 准确率={accuracy_score(y_test, model.predict(X_test)):.4f}")

# 第二行: 固定 C, 改变 gamma
gamma_values = [0.1, 1.0, 10.0]
C_fixed = 1.0

for i, gamma_val in enumerate(gamma_values):
    model = SVC(kernel='rbf', C=C_fixed, gamma=gamma_val, random_state=42)
    model.fit(X_train, y_train)
    plot_decision_boundary(axes[1, i], model, X_test, y_test,
                           f"C={C_fixed}, gamma={gamma_val}")
    print(f"  C={C_fixed:>6.1f}, gamma={gamma_val} -> 准确率={accuracy_score(y_test, model.predict(X_test)):.4f}")

axes[0, 0].set_ylabel('改变 C (gamma=1.0)', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('改变 gamma (C=1.0)', fontsize=12, fontweight='bold')

plt.suptitle('SVM 参数 C 和 gamma 对决策边界的影响', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("D:/code/big-model-learn/code/q_01/W6/d3_c_gamma_effect.png", dpi=150, bbox_inches='tight')
plt.close()
print("参数影响图已保存")

# ================================================================
# 总结
# ================================================================
print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print("1. 线性核只能处理线性可分数据，对非线性数据效果差")
print("2. RBF 核通过将数据映射到高维空间处理非线性问题")
print("3. 多项式核介于线性和 RBF 之间，degree 控制多项式阶数")
print("4. C 控制正则化强度: C大->复杂边界(过拟合), C小->简单边界(欠拟合)")
print("5. gamma 控制 RBF 核的影响范围: gamma大->过拟合, gamma小->欠拟合")
print("6. GridSearchCV 可系统地搜索最优参数组合")
