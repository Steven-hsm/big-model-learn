### Day 1（周一）：K-Means聚类原理与实战
# K-Means聚类：从零实现、肘部法则、轮廓系数、可视化

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 生成合成数据
# ============================================================
np.random.seed(42)
X, y_true = make_blobs(
    n_samples=500,
    centers=4,
    cluster_std=1.2,
    random_state=42
)

print("数据形状:", X.shape)
print("真实簇数:", len(np.unique(y_true)))


# ============================================================
# 2. 从零实现 K-Means
# ============================================================
class KMeansScratch:
    """从零实现的K-Means聚类算法"""

    def __init__(self, n_clusters=4, max_iters=100, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.centers_ = None
        self.labels_ = None
        self.n_iter_ = 0

    def fit(self, X):
        np.random.seed(self.random_state)
        # 随机选择初始中心点
        idx = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centers_ = X[idx].copy()

        for i in range(self.max_iters):
            self.n_iter_ = i + 1
            old_centers = self.centers_.copy()

            # 步骤1: 分配每个点到最近的中心
            self.labels_ = self._assign(X)

            # 步骤2: 更新中心点
            self._update_centers(X)

            # 检查收敛
            shift = np.linalg.norm(self.centers_ - old_centers)
            if shift < self.tol:
                print(f"  KMeans从零实现: 第{i+1}轮收敛 (中心偏移={shift:.6f})")
                break

        return self

    def _assign(self, X):
        """将每个样本分配到最近的中心"""
        distances = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            distances[:, k] = np.linalg.norm(X - self.centers_[k], axis=1)
        return np.argmin(distances, axis=1)

    def _update_centers(self, X):
        """更新每个簇的中心点"""
        for k in range(self.n_clusters):
            members = X[self.labels_ == k]
            if len(members) > 0:
                self.centers_[k] = members.mean(axis=0)

    def inertia(self, X):
        """计算惯性（簇内平方和）"""
        total = 0.0
        for k in range(self.n_clusters):
            members = X[self.labels_ == k]
            if len(members) > 0:
                total += np.sum((members - self.centers_[k]) ** 2)
        return total

    def predict(self, X):
        return self._assign(X)


print("\n--- 从零实现 K-Means ---")
km_scratch = KMeansScratch(n_clusters=4, random_state=42)
km_scratch.fit(X)
print(f"  惯性(inertia): {km_scratch.inertia(X):.2f}")


# ============================================================
# 3. 与 sklearn KMeans 对比
# ============================================================
print("\n--- sklearn KMeans ---")
km_sklearn = KMeans(n_clusters=4, random_state=42, n_init=10)
km_sklearn.fit(X)
print(f"  惯性(inertia): {km_sklearn.inertia_:.2f}")
print(f"  迭代次数: {km_sklearn.n_iter_}")

# 对比可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(X[:, 0], X[:, 1], c=km_scratch.labels_, cmap='viridis', s=15, alpha=0.6)
axes[0].scatter(km_scratch.centers_[:, 0], km_scratch.centers_[:, 1],
                c='red', marker='X', s=200, edgecolors='black', label='中心点')
axes[0].set_title('K-Means (从零实现)')
axes[0].legend()

axes[1].scatter(X[:, 0], X[:, 1], c=km_sklearn.labels_, cmap='viridis', s=15, alpha=0.6)
axes[1].scatter(km_sklearn.cluster_centers_[:, 0], km_sklearn.cluster_centers_[:, 1],
                c='red', marker='X', s=200, edgecolors='black', label='中心点')
axes[1].set_title('K-Means (sklearn)')

plt.suptitle('从零实现 vs sklearn KMeans', fontsize=14)
plt.tight_layout()
plt.show()


# ============================================================
# 4. 肘部法则 (Elbow Method)
# ============================================================
print("\n--- 肘部法则 (Elbow Method) ---")
K_range = range(2, 10)
inertias = []
silhouette_scores = []

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)
    sil = silhouette_score(X, km.labels_)
    silhouette_scores.append(sil)
    print(f"  K={k}: inertia={km.inertia_:.2f}, silhouette={sil:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 肘部法则图
axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('簇数 K')
axes[0].set_ylabel('惯性 (Inertia)')
axes[0].set_title('肘部法则 — 惯性 vs K')
axes[0].grid(True, alpha=0.3)

# 轮廓系数图
axes[1].plot(K_range, silhouette_scores, 'rs-', linewidth=2, markersize=8)
best_k = list(K_range)[np.argmax(silhouette_scores)]
axes[1].axvline(x=best_k, color='gray', linestyle='--', label=f'最优K={best_k}')
axes[1].set_xlabel('簇数 K')
axes[1].set_ylabel('轮廓系数 (Silhouette Score)')
axes[1].set_title('轮廓系数 — 越大越好')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('K值选择', fontsize=14)
plt.tight_layout()
plt.show()
print(f"轮廓系数最优K值: {best_k}")


# ============================================================
# 5. PCA 2D投影可视化聚类结果
# ============================================================
# 使用更高维数据展示PCA降维可视化的价值
X_high, y_high = make_blobs(n_samples=500, centers=4, n_features=10,
                             cluster_std=2.0, random_state=42)

km_vis = KMeans(n_clusters=4, random_state=42, n_init=10)
km_vis.fit(X_high)

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_high)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 真实标签
scatter1 = axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=y_high, cmap='viridis', s=15, alpha=0.6)
axes[0].set_title('真实标签 (PCA 2D投影)')
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')

# K-Means聚类结果
scatter2 = axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=km_vis.labels_, cmap='viridis', s=15, alpha=0.6)
centers_2d = pca.transform(km_vis.cluster_centers_)
axes[1].scatter(centers_2d[:, 0], centers_2d[:, 1],
                c='red', marker='X', s=200, edgecolors='black', label='中心点')
axes[1].set_title('K-Means聚类结果 (PCA 2D投影)')
axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
axes[1].legend()

plt.suptitle('高维数据PCA降维可视化', fontsize=14)
plt.tight_layout()
plt.show()

print("\n=== Day 1 完成: K-Means聚类原理与实战 ===")
