### Day 2（周二）：DBSCAN / 层次聚类 / GMM
# 对比不同聚类算法在不同数据分布上的表现

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. make_moons数据：KMeans vs DBSCAN 对比
# ============================================================
print("=== make_moons 数据集: KMeans vs DBSCAN ===\n")

X_moons, y_moons = make_moons(n_samples=500, noise=0.1, random_state=42)
X_moons = StandardScaler().fit_transform(X_moons)

# KMeans
km_moons = KMeans(n_clusters=2, random_state=42, n_init=10)
km_labels = km_moons.fit_predict(X_moons)

# DBSCAN
db_moons = DBSCAN(eps=0.3, min_samples=5)
db_labels = db_moons.fit_predict(X_moons)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 真实标签
axes[0].scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, cmap='viridis', s=15, alpha=0.6)
axes[0].set_title('真实标签')

# KMeans结果
axes[1].scatter(X_moons[:, 0], X_moons[:, 1], c=km_labels, cmap='viridis', s=15, alpha=0.6)
axes[1].set_title('KMeans (不适合非球形数据)')

# DBSCAN结果
n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise = list(db_labels).count(-1)
axes[2].scatter(X_moons[:, 0], X_moons[:, 1], c=db_labels, cmap='viridis', s=15, alpha=0.6)
axes[2].set_title(f'DBSCAN (簇数={n_clusters_db}, 噪声点={n_noise})')

plt.suptitle('make_moons: KMeans vs DBSCAN', fontsize=14)
plt.tight_layout()
plt.show()


# ============================================================
# 2. DBSCAN 参数调优
# ============================================================
print("\n=== DBSCAN 参数调优 ===\n")

eps_values = [0.2, 0.3, 0.5, 0.8]
min_samples_values = [3, 5, 10]

fig, axes = plt.subplots(len(eps_values), len(min_samples_values), figsize=(15, 16))

for i, eps in enumerate(eps_values):
    for j, min_s in enumerate(min_samples_values):
        db = DBSCAN(eps=eps, min_samples=min_s)
        labels = db.fit_predict(X_moons)
        n_clust = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        ax = axes[i, j]
        scatter = ax.scatter(X_moons[:, 0], X_moons[:, 1], c=labels, cmap='viridis', s=10, alpha=0.6)
        ax.set_title(f'eps={eps}, min_samples={min_s}\n簇={n_clust}, 噪声={n_noise}', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

plt.suptitle('DBSCAN参数调优 (eps vs min_samples)', fontsize=14)
plt.tight_layout()
plt.show()


# ============================================================
# 3. 层次聚类 + 树状图
# ============================================================
print("\n=== 层次聚类 ===\n")

# 生成适合层次聚类的数据
X_hier, y_hier = make_blobs(n_samples=80, centers=3, cluster_std=1.0, random_state=42)

# 计算层次聚类的连接矩阵
Z = linkage(X_hier, method='ward')

# 绘制树状图
plt.figure(figsize=(14, 6))
dendrogram(Z, truncate_mode='level', p=5, leaf_font_size=9)
plt.title('层次聚类树状图 (Ward方法)')
plt.xlabel('样本索引')
plt.ylabel('距离')
plt.axhline(y=15, color='r', linestyle='--', label='切割线 (K=3)')
plt.legend()
plt.tight_layout()
plt.show()

# AgglomerativeClustering
agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
agg_labels = agg.fit_predict(X_hier)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(X_hier[:, 0], X_hier[:, 1], c=y_hier, cmap='viridis', s=30, alpha=0.7)
axes[0].set_title('真实标签')

axes[1].scatter(X_hier[:, 0], X_hier[:, 1], c=agg_labels, cmap='viridis', s=30, alpha=0.7)
axes[1].set_title('Agglomerative Clustering (Ward)')

plt.suptitle('层次聚类结果', fontsize=14)
plt.tight_layout()
plt.show()

print(f"层次聚类结果: {len(np.unique(agg_labels))} 个簇")


# ============================================================
# 4. GMM (高斯混合模型)
# ============================================================
print("\n=== GMM 高斯混合模型 ===\n")

# 生成椭圆形簇数据（适合GMM）
np.random.seed(42)
X_gmm = np.vstack([
    np.random.multivariate_normal([0, 0], [[3, 1], [1, 1]], 200),
    np.random.multivariate_normal([5, 5], [[1, -0.5], [-0.5, 2]], 150),
    np.random.multivariate_normal([-4, 6], [[2, 0], [0, 0.5]], 150),
])

# GMM聚类
gmm = GaussianMixture(n_components=3, random_state=42)
gmm_labels = gmm.fit_predict(X_gmm)

# 软聚类概率
proba = gmm.predict_proba(X_gmm)
print(f"前5个样本的软聚类概率:")
for i in range(5):
    probs_str = ', '.join([f'{p:.3f}' for p in proba[i]])
    print(f"  样本{i}: [{probs_str}] → 簇 {gmm_labels[i]}")

# 可视化GMM结果 + 概率
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 聚类结果
axes[0].scatter(X_gmm[:, 0], X_gmm[:, 1], c=gmm_labels, cmap='viridis', s=15, alpha=0.6)
axes[0].set_title('GMM聚类结果 (硬标签)')

# 最大概率（表示置信度）
max_proba = proba.max(axis=1)
scatter = axes[1].scatter(X_gmm[:, 0], X_gmm[:, 1], c=max_proba, cmap='coolwarm', s=15, alpha=0.6)
plt.colorbar(scatter, ax=axes[1], label='最大概率')
axes[1].set_title('GMM软聚类 — 最大概率 (置信度)')

plt.suptitle('GMM高斯混合模型', fontsize=14)
plt.tight_layout()
plt.show()


# ============================================================
# 5. BIC曲线 — GMM模型选择
# ============================================================
print("\n=== BIC曲线 — GMM模型选择 ===\n")

n_components_range = range(1, 8)
bics = []
aics = []

for n in n_components_range:
    g = GaussianMixture(n_components=n, random_state=42)
    g.fit(X_gmm)
    bics.append(g.bic(X_gmm))
    aics.append(g.aic(X_gmm))
    print(f"  n_components={n}: BIC={g.bic(X_gmm):.1f}, AIC={g.aic(X_gmm):.1f}")

best_bic_n = list(n_components_range)[np.argmin(bics)]
best_aic_n = list(n_components_range)[np.argmin(aics)]

plt.figure(figsize=(10, 5))
plt.plot(n_components_range, bics, 'bo-', label='BIC', linewidth=2, markersize=8)
plt.plot(n_components_range, aics, 'rs-', label='AIC', linewidth=2, markersize=8)
plt.axvline(x=best_bic_n, color='blue', linestyle='--', alpha=0.5, label=f'BIC最优={best_bic_n}')
plt.axvline(x=best_aic_n, color='red', linestyle='--', alpha=0.5, label=f'AIC最优={best_aic_n}')
plt.xlabel('高斯分量数')
plt.ylabel('信息准则值')
plt.title('GMM模型选择 — BIC / AIC 曲线')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nBIC选择的最优分量数: {best_bic_n}")
print(f"AIC选择的最优分量数: {best_aic_n}")

print("\n=== Day 2 完成: DBSCAN / 层次聚类 / GMM ===")
