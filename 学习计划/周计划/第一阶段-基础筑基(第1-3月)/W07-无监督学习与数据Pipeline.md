# W07 - 无监督学习与数据 Pipeline

> 学习时间：第 7 周（第 2 月第 3 周）
> 本周学习聚类、降维和 sklearn Pipeline，完成用户分群项目

---

## 本周目标

1. 掌握 K-Means/DBSCAN/GMM 聚类算法及 K 值选择方法
2. 掌握 PCA/t-SNE/UMAP 降维和可视化
3. 能构建 sklearn Pipeline 实现标准化预处理流程
4. 了解异常检测的基本方法
5. 完成项目 3：电商用户分群

---

## 时间安排

| 时间 | 内容 |
|------|------|
| 周一（2h） | K-Means 聚类原理与实战 |
| 周二（2h） | DBSCAN / 层次聚类 / GMM |
| 周三（2h） | PCA 降维原理与实战 |
| 周四（2h） | t-SNE / UMAP 可视化 + sklearn Pipeline |
| 周五（2h） | 异常检测方法 |
| 周六（4h） | 项目 3：用户分群（RFM + 聚类） |
| 周日（4h） | 项目 3：分析报告 + 可视化 |

---

## 详细学习内容

### Day 1（周一）：K-Means 聚类

#### 1.1 算法流程

```
1. 随机初始化 K 个聚类中心 μ₁, μ₂, ..., μₖ
2. 分配：将每个样本分配到最近的中心 → cᵢ = argminⱼ ||xᵢ - μⱼ||²
3. 更新：重新计算每个簇的中心 → μⱼ = (1/|Cⱼ|) × Σxᵢ (xᵢ∈Cⱼ)
4. 重复 2-3 直到中心不再变化或达到最大迭代次数

K-Means++ 初始化（sklearn 默认）：
  第一个中心随机选，后续中心选离已有中心最远的点
  → 避免初始中心太近导致收敛到局部最优
```

#### 1.2 K 值选择

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 方法1：肘部法 Elbow Method
inertias = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)  # 簇内距离平方和

plt.plot(K_range, inertias, 'bo-')
plt.xlabel('K'); plt.ylabel('Inertia'); plt.title('Elbow Method')
plt.show()
# 选择拐点处的 K

# 方法2：轮廓系数 Silhouette Score（推荐）
# 范围 [-1, 1]，越接近 1 越好
sil_scores = []
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil_scores.append(silhouette_score(X_scaled, labels))

plt.plot(K_range, sil_scores, 'ro-')
plt.xlabel('K'); plt.ylabel('Silhouette Score')
plt.title('Silhouette Method')
plt.show()
# 选择轮廓系数最大的 K
```

#### 1.3 K-Means 实战

```python
# 聚类 + 可视化
km = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = km.fit_predict(X_scaled)

# 聚类中心
centers = scaler.inverse_transform(km.cluster_centers_)

# 可视化（用 PCA 降到 2D）
from sklearn.decomposition import PCA
X_pca = PCA(n_components=2).fit_transform(X_scaled)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.5)
plt.scatter(PCA(n_components=2).fit(X_scaled).transform(km.cluster_centers_)[:, 0],
            PCA(n_components=2).fit(X_scaled).transform(km.cluster_centers_)[:, 1],
            c='red', marker='X', s=200)
plt.title('K-Means Clustering (PCA 2D)')
plt.show()
```

---

### Day 2（周二）：其他聚类方法

#### 2.1 DBSCAN

```
核心概念：
- ε (eps)：邻域半径
- min_samples：核心点所需的最小邻居数
- 核心点：ε 邻域内至少有 min_samples 个点
- 边界点：在核心点邻域内但自身不是核心点
- 噪声点：既不是核心点也不是边界点

优点：不需要预设 K、能发现任意形状簇、自动识别噪声点
缺点：密度不均匀时效果差、高维数据距离度量失效
```

```python
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

# 月牙形数据（K-Means 做不好，DBSCAN 做得好）
X, _ = make_moons(n_samples=500, noise=0.1)
dbscan = DBSCAN(eps=0.15, min_samples=5)
labels = dbscan.fit_predict(X)
# labels=-1 的是噪声点
```

#### 2.2 层次聚类

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# 树状图（帮助选择聚类数）
Z = linkage(X_scaled, method='ward')
plt.figure(figsize=(12, 6))
dendrogram(Z)
plt.title('Dendrogram')
plt.axhline(y=10, color='r', linestyle='--')  # 在这画横线确定聚类数
plt.show()

# 层次聚类
hc = AgglomerativeClustering(n_clusters=4, linkage='ward')
labels = hc.fit_predict(X_scaled)
```

#### 2.3 高斯混合模型 GMM

```python
from sklearn.mixture import GaussianaussianMixture

gmm = GaussianMixture(n_components=4, random_state=42)
labels = gmm.fit_predict(X_scaled)

# 软聚类：每个样本属于每个簇的概率
probs = gmm.predict_proba(X_scaled)
print(probs[:5])  # 每行是一个样本的概率分布

# BIC/AIC 选择最优 K（越小越好）
bics = []
for k in range(2, 11):
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(X_scaled)
    bics.append(gmm.bic(X_scaled))
```

---

### Day 3（周三）：PCA 降维

#### 3.1 PCA 原理

```
PCA 目标：找到数据方差最大的方向（主成分）

数学步骤：
1. 数据中心化 X_centered = X - X.mean(axis=0)
2. 计算协方差矩阵 C = (1/n) × X^T × X
3. 对 C 做特征分解 C = V × Λ × V^T
4. 选最大的 k 个特征值对应的特征向量
5. 投影 X_pca = X_centered × V_k

注意：PCA 前必须标准化！（否则方差大的特征会主导结果）
```

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)  # 保留 95% 方差
X_pca = pca.fit_transform(X_scaled)

print(f"原始维度: {X_scaled.shape[1]}")
print(f"降维后维度: {X_pca.shape[1]}")
print(f"各主成分方差解释比: {pca.explained_variance_ratio_}")
print(f"累积方差解释比: {pca.explained_variance_ratio_.cumsum()}")

# 可视化方差解释比
plt.bar(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_)
plt.step(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_.cumsum(), where='mid')
plt.axhline(y=0.95, color='r', linestyle='--')
plt.xlabel('Principal Component'); plt.ylabel('Variance Explained')
plt.show()
```

---

### Day 4（周四）：t-SNE / UMAP + Pipeline

#### 4.1 t-SNE 可视化

```python
from sklearn.manifold import TSNE

# t-SNE 适合可视化（2D/3D），不适合特征提取
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis', alpha=0.5)
plt.title('t-SNE Visualization')
plt.show()
# perplexity 通常在 5-50 之间，需要尝试
```

#### 4.2 UMAP（更快更好的可视化）

```python
# pip install umap-learn
import umap

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

plt.scatter(X_umap[:, 0], X_umap[:, 1], c=labels, cmap='viridis', alpha=0.5)
plt.title('UMAP Visualization')
plt.show()
```

#### 4.3 sklearn Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

# 自定义 Transformer
class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.feature_names]

# 完整 Pipeline
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), num_features),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', sparse_output=False))
    ]), cat_features)
])

full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(n_estimators=100))
])

# 一行训练
full_pipeline.fit(X_train, y_train)
y_pred = full_pipeline.predict(X_test)

# Pipeline + GridSearchCV
param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [5, 10, None]
}
grid = GridSearchCV(full_pipeline, param_grid, cv=5)
```

---

### Day 5（周五）：异常检测

```python
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# 方法1：Z-Score（简单快速）
z_scores = np.abs((X - X.mean()) / X.std())
outliers = (z_scores > 3).any(axis=1)

# 方法2：IQR 方法
Q1 = X.quantile(0.25); Q3 = X.quantile(0.75)
IQR = Q3 - Q1
outliers = ((X < (Q1 - 1.5*IQR)) | (X > (Q3 + 1.5*IQR))).any(axis=1)

# 方法3：Isolation Forest（推荐）
iso = IsolationForest(contamination=0.05, random_state=42)
labels = iso.fit_predict(X)  # -1=异常, 1=正常

# 方法4：LOF 局部异常因子
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
labels = lof.fit_predict(X)
```

---

### Day 6-7（周末）：项目 3 - 电商用户分群

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ===== 1. 加载用户行为数据 =====
df = pd.read_csv('ecommerce_data.csv')
# 假设有列: user_id, order_date, order_amount

# ===== 2. 计算 RFM 特征 =====
reference_date = df['order_date'].max() + pd.Timedelta(days=1)

rfm = df.groupby('user_id').agg({
    'order_date': lambda x: (reference_date - x.max()).days,   # Recency
    'user_id': 'count',                                         # Frequency
    'order_amount': 'sum'                                       # Monetary
}).rename(columns={'order_date': 'recency', 'user_id': 'frequency', 'order_amount': 'monetary'})

# ===== 3. 数据预处理 =====
# RFM 通常右偏，取对数
rfm_log = np.log1p(rfm)
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_log)

# ===== 4. 选择 K =====
sil_scores = []
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(rfm_scaled)
    sil_scores.append(silhouette_score(rfm_scaled, labels))

best_k = range(2, 8)[np.argmax(sil_scores)]
print(f"最优 K={best_k}, Silhouette={max(sil_scores):.4f}")

# ===== 5. 最终聚类 =====
km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
rfm['cluster'] = km.fit_predict(rfm_scaled)

# ===== 6. 分析各簇特征 =====
cluster_analysis = rfm.groupby('cluster').agg({
    'recency': 'mean',
    'frequency': 'mean',
    'monetary': 'mean',
    'cluster': 'count'
}).rename(columns={'cluster': 'count'})

print(cluster_analysis)

# 给各簇命名（根据特征分析）
# 例如：高价值客户/普通客户/流失风险客户/沉睡客户

# ===== 7. 可视化 =====
# 3D 散点图
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(rfm['recency'], rfm['frequency'], rfm['monetary'],
                     c=rfm['cluster'], cmap='Set1', alpha=0.6)
ax.set_xlabel('Recency'); ax.set_ylabel('Frequency'); ax.set_zlabel('Monetary')
plt.title('RFM Clustering 3D')
plt.colorbar(scatter)
plt.show()

# 雷达图
from sklearn.preprocessing import MinMaxScaler
cluster_means = rfm.groupby('cluster')[['recency', 'frequency', 'monetary']].mean()
cluster_means_scaled = MinMaxScaler().fit_transform(cluster_means)

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
angles = np.linspace(0, 2*np.pi, 3, endpoint=False).tolist()
angles += angles[:1]
for i, row in enumerate(cluster_means_scaled):
    values = row.tolist() + row.tolist()[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {i}')
    ax.fill(angles, values, alpha=0.1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(['Recency', 'Frequency', 'Monetary'])
ax.legend()
plt.title('Cluster Profiles')
plt.show()
```

---

## 本周产出

- [ ] K-Means/DBSCAN/GMM 对比实验（含不同形状数据集的可视化）
- [ ] PCA 方差解释比可视化图
- [ ] sklearn Pipeline 模板代码
- [ ] **项目 3：用户分群**（RFM 特征 + K-Means 聚类 + 3D/雷达图分析报告）

---

## 自测题

1. **K-Means 的假设前提？什么数据不适合？**
   <details><summary>参考答案</summary>
   假设：簇是球形（各方向方差相同）、大小相近、密度均匀。不适合：非球形簇（月牙形/环形，应用 DBSCAN）、大小差异大的簇、密度不均匀的簇、高维数据（维度灾难，距离失效）。K-Means++ 解决初始化问题但不解决形状假设。
   </details>

2. **DBSCAN 相比 K-Means 的优势和劣势？**
   <details><summary>参考答案</summary>
   优势：不需要预设 K、能发现任意形状的簇、自动识别噪声点（label=-1）。劣势：密度不均匀时效果差（统一 eps 不适用所有簇）、高维数据效果差、eps 和 min_samples 需要调、不能很好地处理不同密度的簇。
   </details>

3. **PCA 为什么要先标准化？**
   <details><summary>参考答案</summary>
   PCA 找方差最大的方向。如果不标准化，量纲大的特征（如收入万元级）方差远大于量纲小的特征（如年龄），PCA 会偏向量纲大的特征，这不是我们想要的。StandardScaler 让所有特征方差相同，PCA 能公平地选择方向。
   </details>

4. **sklearn Pipeline 的好处？**
   <details><summary>参考答案</summary>
   (1) 代码简洁：一行 fit/predict 完成所有步骤。
   (2) 防止数据泄露：交叉验证时每折都会重新 fit transform，不会用验证集的信息训练。
   (3) 便于调参：可以和 GridSearchCV 配合，对 Pipeline 中任何步骤的参数搜索。
   (4) 可复现：保存 Pipeline 就保存了完整的预处理+模型流程。
   </details>

5. **RFM 模型三个维度的含义？**
   <details><summary>参考答案</summary>
   Recency（最近一次购买距今多少天）：越小越好，最近购买的用户更活跃。
   Frequency（购买频次）：越大越好，高频购买说明忠诚度高。
   Monetary（累计消费金额）：越大越好，反映用户价值。
   通过这三个维度将用户分成高价值/普通/流失风险等群体，针对性运营。
   </details>

---

## Java 开发者提示

| Python/ML 概念 | Java 类比 |
|----------------|-----------|
| K-Means 聚类 | 类似负载均衡中的一致性哈希分桶 |
| sklearn Pipeline | 类似 Spring Integration 的消息管道 |
| PCA 降维 | 类似数据库索引（用少量维度表达大部分信息） |
| ColumnTransformer | 类似策略模式（对不同字段用不同处理策略） |
| RFM 模型 | 类似用户画像标签系统 |
