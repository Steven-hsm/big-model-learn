### Day 6（周六）：项目3 — 用户分群（RFM + 聚类）
# RFM特征计算、K-Means聚类、3D可视化、雷达图

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 生成合成电商交易数据
# ============================================================
print("=== 生成合成电商交易数据 ===\n")

np.random.seed(42)

n_customers = 800
n_transactions = 5000

# 客户ID
customer_ids = [f"C{str(i).zfill(5)}" for i in range(1, n_customers + 1)]

# 交易数据
transaction_customer = np.random.choice(customer_ids, size=n_transactions)
# 交易日期: 过去365天内的随机日期
transaction_days_ago = np.random.randint(0, 365, size=n_transactions)
# 交易金额: 右偏分布（模拟真实消费模式）
transaction_amount = np.random.exponential(scale=200, size=n_transactions) + 10

print(f"客户数: {n_customers}")
print(f"交易数: {n_transactions}")
print(f"交易金额范围: {transaction_amount.min():.2f} ~ {transaction_amount.max():.2f}")
print(f"交易金额均值: {transaction_amount.mean():.2f}")


# ============================================================
# 2. 计算RFM特征
# ============================================================
print("\n=== 计算RFM特征 ===\n")

from collections import defaultdict

# 按客户聚合
customer_data = defaultdict(lambda: {'days': [], 'amounts': []})
for cid, day, amt in zip(transaction_customer, transaction_days_ago, transaction_amount):
    customer_data[cid]['days'].append(day)
    customer_data[cid]['amounts'].append(amt)

# 计算RFM
rfm_data = []
for cid in customer_ids:
    data = customer_data[cid]
    recency = min(data['days'])            # 最近一次购买距今天数
    frequency = len(data['days'])          # 购买次数
    monetary = sum(data['amounts'])        # 总消费金额
    rfm_data.append({
        'customer_id': cid,
        'recency': recency,
        'frequency': frequency,
        'monetary': monetary
    })

rfm = np.array([[d['recency'], d['frequency'], d['monetary']] for d in rfm_data])

print(f"RFM数据形状: {rfm.shape}")
print(f"\nRFM统计:")
print(f"  Recency:   均值={rfm[:, 0].mean():.1f}, 中位数={np.median(rfm[:, 0]):.1f}")
print(f"  Frequency:  均值={rfm[:, 1].mean():.1f}, 中位数={np.median(rfm[:, 1]):.1f}")
print(f"  Monetary:   均值={rfm[:, 2].mean():.1f}, 中位数={np.median(rfm[:, 2]):.1f}")


# ============================================================
# 3. Log变换 + StandardScaler
# ============================================================
print("\n=== Log变换 + 标准化 ===\n")

# RFM特征通常右偏，先做log变换
rfm_log = np.log1p(rfm)

print("Log变换前偏度:")
for i, name in enumerate(['Recency', 'Frequency', 'Monetary']):
    from scipy.stats import skew
    print(f"  {name}: {skew(rfm[:, i]):.3f}")

print("\nLog变换后偏度:")
for i, name in enumerate(['Recency', 'Frequency', 'Monetary']):
    print(f"  {name}: {skew(rfm_log[:, i]):.3f}")

# 标准化
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_log)

print(f"\n标准化后: 均值={rfm_scaled.mean(axis=0).round(4)}, 标准差={rfm_scaled.std(axis=0).round(4)}")


# ============================================================
# 4. 轮廓系数确定最优K
# ============================================================
print("\n=== 轮廓系数确定最优K ===\n")

K_range = range(2, 8)
silhouette_scores = []
inertias = []

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(rfm_scaled)
    sil = silhouette_score(rfm_scaled, labels)
    silhouette_scores.append(sil)
    inertias.append(km.inertia_)
    print(f"  K={k}: 轮廓系数={sil:.4f}, 惯性={km.inertia_:.1f}")

best_k = list(K_range)[np.argmax(silhouette_scores)]
print(f"\n最优K值: {best_k} (轮廓系数={max(silhouette_scores):.4f})")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(K_range, inertias, 'bo-', linewidth=2)
axes[0].set_xlabel('K')
axes[0].set_ylabel('Inertia')
axes[0].set_title('肘部法则')
axes[0].grid(True, alpha=0.3)

axes[1].plot(K_range, silhouette_scores, 'rs-', linewidth=2)
axes[1].axvline(x=best_k, color='gray', linestyle='--', label=f'最优K={best_k}')
axes[1].set_xlabel('K')
axes[1].set_ylabel('轮廓系数')
axes[1].set_title('轮廓系数')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('RFM聚类: 确定最优K', fontsize=14)
plt.tight_layout()
plt.show()


# ============================================================
# 5. K-Means聚类
# ============================================================
print(f"\n=== K-Means聚类 (K={best_k}) ===\n")

km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = km_final.fit_predict(rfm_scaled)

# 各簇统计
for c in range(best_k):
    mask = cluster_labels == c
    count = mask.sum()
    pct = count / len(cluster_labels) * 100
    r_mean = rfm[mask, 0].mean()
    f_mean = rfm[mask, 1].mean()
    m_mean = rfm[mask, 2].mean()
    print(f"  簇{c}: {count}人 ({pct:.1f}%) | R={r_mean:.1f}天, F={f_mean:.1f}次, M={m_mean:.0f}元")


# ============================================================
# 6. 3D散点图
# ============================================================
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

colors = plt.cm.Set1(np.linspace(0, 1, best_k))
for c in range(best_k):
    mask = cluster_labels == c
    ax.scatter(rfm_log[mask, 0], rfm_log[mask, 1], rfm_log[mask, 2],
               c=[colors[c]], s=20, alpha=0.5, label=f'簇{c}')

ax.set_xlabel('Log(Recency)')
ax.set_ylabel('Log(Frequency)')
ax.set_zlabel('Log(Monetary)')
ax.set_title('RFM用户分群 3D散点图')
ax.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 7. 雷达图 — 簇画像
# ============================================================
print("\n=== 雷达图 — 各簇RFM画像 ===\n")

# 计算每个簇的平均RFM (标准化后的值，便于雷达图比较)
cluster_profiles = np.zeros((best_k, 3))
for c in range(best_k):
    mask = cluster_labels == c
    cluster_profiles[c] = rfm_scaled[mask].mean(axis=0)

# 归一化到0-1范围用于雷达图
profile_min = cluster_profiles.min(axis=0)
profile_max = cluster_profiles.max(axis=0)
profile_norm = (cluster_profiles - profile_min) / (profile_max - profile_min + 1e-8)

categories = ['Recency\n(越低越好)', 'Frequency\n(越高越好)', 'Monetary\n(越高越好)']
n_cats = len(categories)
angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
angles += angles[:1]  # 闭合

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for c in range(best_k):
    values = profile_norm[c].tolist()
    values += values[:1]  # 闭合
    ax.plot(angles, values, 'o-', linewidth=2, label=f'簇{c}', color=colors[c])
    ax.fill(angles, values, alpha=0.15, color=colors[c])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0, 1.1)
ax.set_title('各簇RFM雷达图', fontsize=14, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

plt.tight_layout()
plt.show()

print("\n=== Day 6 完成: 用户分群（RFM + 聚类） ===")
