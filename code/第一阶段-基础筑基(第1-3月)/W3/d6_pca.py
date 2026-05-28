# W03 Day 6 - PCA主成分分析
# 从零实现PCA + sklearn PCA + Iris降维可视化 + PCA vs LDA对比

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# ========================
# 1. 手动实现PCA
# ========================

def pca_manual(X, n_components=2):
    """从零实现PCA"""
    n_samples, n_features = X.shape

    # 1. 中心化
    mean = X.mean(axis=0)
    X_centered = X - mean

    # 2. 协方差矩阵
    cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)

    # 3. 特征分解
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # 4. 按特征值从大到小排序
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 5. 选择前k个主成分
    components = eigenvectors[:, :n_components]

    # 6. 投影
    X_pca = X_centered @ components

    # 7. 方差解释比
    explained_variance_ratio = eigenvalues / eigenvalues.sum()

    return X_pca, components, eigenvalues, explained_variance_ratio, mean


# 生成测试数据
np.random.seed(42)
mean = [0, 0]
cov = [[3, 2], [2, 2]]
X = np.random.multivariate_normal(mean, cov, 200)

X_pca, components, eigenvalues, explained_ratio, mean = pca_manual(X, n_components=2)

print("特征值:", eigenvalues)
print("方差解释比:", explained_ratio)
print("累积方差:", np.cumsum(explained_ratio))

# 可视化：原始数据 + 主成分方向
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(X[:, 0], X[:, 1], alpha=0.5)
for i, (ev, vec) in enumerate(zip(eigenvalues, components.T)):
    axes[0].arrow(0, 0, vec[0] * ev**0.5 * 2, vec[1] * ev**0.5 * 2,
                  head_width=0.1, color=['red', 'blue'][i], linewidth=2)
axes[0].set_title('Original Data + Principal Components')
axes[0].set_aspect('equal')
axes[0].grid(True)

axes[1].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
axes[1].set_title('PCA Transformed Data')
axes[1].set_xlabel(f'PC1 ({explained_ratio[0]:.1%} variance)')
axes[1].set_ylabel(f'PC2 ({explained_ratio[1]:.1%} variance)')
axes[1].grid(True)

plt.tight_layout()
plt.savefig('pca_manual.png', dpi=150, bbox_inches='tight')
plt.show()

# ========================
# 2. 选择主成分数量
# ========================

# 碎石图（Scree Plot）
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_ratio) + 1), explained_ratio, 'bo-')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.axhline(y=0.05, color='r', linestyle='--', label='5% threshold')
plt.legend()
plt.grid(True)
plt.show()

# 累积方差图
cumsum = np.cumsum(explained_ratio)
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumsum) + 1), cumsum, 'ro-')
plt.axhline(y=0.95, color='g', linestyle='--', label='95% threshold')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance')
plt.legend()
plt.grid(True)
plt.show()

# ========================
# 3. 用sklearn PCA对Iris降维可视化
# ========================

iris = load_iris()
X_iris = iris.data
y_iris = iris.target
target_names = iris.target_names

# PCA前必须标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_iris)

# sklearn PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"\nIris: 原始维度={X_iris.shape}, 降维后={X_pca.shape}")
print(f"方差解释比: {pca.explained_variance_ratio_}")
print(f"累积方差: {pca.explained_variance_ratio_.sum():.2%}")

# 可视化
plt.figure(figsize=(10, 8))
colors = ['red', 'blue', 'green']
for i, (color, target_name) in enumerate(zip(colors, target_names)):
    plt.scatter(X_pca[y_iris == i, 0], X_pca[y_iris == i, 1],
                c=color, label=target_name, alpha=0.7, edgecolors='k')

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.title('PCA: Iris Dataset (4D -> 2D)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('pca_iris.png', dpi=150, bbox_inches='tight')
plt.show()

# ========================
# 4. PCA vs LDA 对比
# ========================

lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_scaled, y_iris)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# PCA
for i, (color, name) in enumerate(zip(colors, target_names)):
    axes[0].scatter(X_pca[y_iris == i, 0], X_pca[y_iris == i, 1],
                    c=color, label=name, alpha=0.7)
axes[0].set_title('PCA (Unsupervised)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# LDA
for i, (color, name) in enumerate(zip(colors, target_names)):
    axes[1].scatter(X_lda[y_iris == i, 0], X_lda[y_iris == i, 1],
                    c=color, label=name, alpha=0.7)
axes[1].set_title('LDA (Supervised)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pca_vs_lda.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nPCA是无监督降维（不用标签），LDA是有监督降维（利用标签信息）")
