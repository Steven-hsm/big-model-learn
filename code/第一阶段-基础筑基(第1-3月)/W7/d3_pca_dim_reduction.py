### Day 3（周三）：PCA降维原理与实战
# 手动PCA实现、方差解释率、降维可视化

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 手动PCA实现
# ============================================================
print("=== 手动PCA实现 ===\n")

# ���载Iris数据
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print(f"数据形状: {X.shape}")
print(f"特征: {feature_names}")
print(f"类别: {target_names}")


def pca_scratch(X, n_components=2):
    """从零实现PCA降维"""
    # 步骤1: 标准化（均值中心化）
    X_mean = X.mean(axis=0)
    X_centered = X - X_mean

    # 步骤2: 计算协方差矩阵
    n = X_centered.shape[0]
    cov_matrix = np.dot(X_centered.T, X_centered) / (n - 1)
    print(f"协方差矩阵形状: {cov_matrix.shape}")

    # 步骤3: 特征值分解
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # 步骤4: 按特征值降序排序
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 步骤5: 选择前n_components个主成分
    components = eigenvectors[:, :n_components]
    explained_var = eigenvalues / eigenvalues.sum()

    # 步骤6: 投影到主成分空间
    X_projected = np.dot(X_centered, components)

    return X_projected, eigenvalues, explained_var, components


# 标准化数据后做PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_pca_scratch, eigenvalues, explained_var, components = pca_scratch(X_scaled, n_components=2)

print(f"\n特征值: {eigenvalues}")
print(f"方差解释比例: {explained_var}")
print(f"前2个主成分累计解释方差: {explained_var[:2].sum()*100:.2f}%")


# ============================================================
# 2. 与 sklearn PCA 对比
# ============================================================
print("\n=== 与 sklearn PCA 对比 ===\n")

pca_sklearn = PCA(n_components=2)
X_pca_sklearn = pca_sklearn.fit_transform(X_scaled)

print(f"sklearn PCA方差解释比例: {pca_sklearn.explained_variance_ratio_}")
print(f"sklearn PCA特征值: {pca_sklearn.explained_variance_}")

# 对比可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

colors = ['#e74c3c', '#2ecc71', '#3498db']
for i, target in enumerate(target_names):
    mask = y == i
    axes[0].scatter(X_pca_scratch[mask, 0], X_pca_scratch[mask, 1],
                    c=colors[i], label=target, s=30, alpha=0.7)
axes[0].set_title('PCA (手动实现)')
axes[0].set_xlabel(f'PC1 ({explained_var[0]*100:.1f}%)')
axes[0].set_ylabel(f'PC2 ({explained_var[1]*100:.1f}%)')
axes[0].legend()

for i, target in enumerate(target_names):
    mask = y == i
    axes[1].scatter(X_pca_sklearn[mask, 0], X_pca_sklearn[mask, 1],
                    c=colors[i], label=target, s=30, alpha=0.7)
axes[1].set_title('PCA (sklearn)')
axes[1].set_xlabel(f'PC1 ({pca_sklearn.explained_variance_ratio_[0]*100:.1f}%)')
axes[1].set_ylabel(f'PC2 ({pca_sklearn.explained_variance_ratio_[1]*100:.1f}%)')
axes[1].legend()

plt.suptitle('手动PCA vs sklearn PCA', fontsize=14)
plt.tight_layout()
plt.show()


# ============================================================
# 3. 方差解释比例图 (bar + cumulative step)
# ============================================================
print("\n=== 方差解释比例 ===\n")

pca_full = PCA()
pca_full.fit(X_scaled)

var_ratio = pca_full.explained_variance_ratio_
cum_var = np.cumsum(var_ratio)

print("各主成分方差解释比例:")
for i, (vr, cv) in enumerate(zip(var_ratio, cum_var)):
    print(f"  PC{i+1}: {vr*100:.2f}% (累计: {cv*100:.2f}%)")

plt.figure(figsize=(10, 5))
plt.bar(range(1, len(var_ratio)+1), var_ratio, alpha=0.6, color='steelblue', label='单独解释比例')
plt.step(range(1, len(cum_var)+1), cum_var, where='mid', color='red', linewidth=2, label='累计解释比例')
plt.axhline(y=0.95, color='gray', linestyle='--', alpha=0.7, label='95%阈值')
plt.xlabel('主成分')
plt.ylabel('方差解释比例')
plt.title('PCA — 各主成分方差解释比例')
plt.xticks(range(1, len(var_ratio)+1))
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 4. 按95%方差阈值选择n_components
# ============================================================
print("\n=== 按95%方差阈值选择维度 ===\n")

pca_95 = PCA(n_components=0.95)
X_pca_95 = pca_95.fit_transform(X_scaled)

print(f"原始维度: {X_scaled.shape[1]}")
print(f"95%方差所需维度: {X_pca_95.shape[1]}")
print(f"实际解释方差: {pca_95.explained_variance_ratio_.sum()*100:.2f}%")
print(f"保留的成分数: {pca_95.n_components_}")


# ============================================================
# 5. PCA前后可视化对比
# ============================================================
print("\n=== PCA前后可视化对比 ===\n")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# PCA前: 使用前两个原始特征
for i, target in enumerate(target_names):
    mask = y == i
    axes[0].scatter(X_scaled[mask, 0], X_scaled[mask, 1],
                    c=colors[i], label=target, s=30, alpha=0.7)
axes[0].set_title('PCA前 (原始前两个特征)')
axes[0].set_xlabel(feature_names[0])
axes[0].set_ylabel(feature_names[1])
axes[0].legend()

# PCA后: 使用前两个主成分
for i, target in enumerate(target_names):
    mask = y == i
    axes[1].scatter(X_pca_95[mask, 0], X_pca_95[mask, 1] if X_pca_95.shape[1] > 1
                    else np.zeros(mask.sum()),
                    c=colors[i], label=target, s=30, alpha=0.7)
axes[1].set_title(f'PCA后 ({X_pca_95.shape[1]}个成分, 保留95%方差)')
axes[1].set_xlabel(f'PC1 ({pca_95.explained_variance_ratio_[0]*100:.1f}%)')
axes[1].set_ylabel(f'PC2 ({pca_95.explained_variance_ratio_[1]*100:.1f}%)')
axes[1].legend()

plt.suptitle('PCA降维前后对比', fontsize=14)
plt.tight_layout()
plt.show()

print("\n=== Day 3 完成: PCA降维原理与实战 ===")
