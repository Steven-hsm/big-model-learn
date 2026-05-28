### Day 7（周日）：项目3 — 分析报告 + 可视化
# 完整RFM分析流水线、多面板可视化、业��解读、策略建议

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from collections import defaultdict

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 生成合成电商交易数据（与d6相同）
# ============================================================
print("=" * 60)
print("   用户分群分析报告 — RFM模型 + K-Means聚类")
print("=" * 60)

np.random.seed(42)

n_customers = 800
n_transactions = 5000

customer_ids = [f"C{str(i).zfill(5)}" for i in range(1, n_customers + 1)]
transaction_customer = np.random.choice(customer_ids, size=n_transactions)
transaction_days_ago = np.random.randint(0, 365, size=n_transactions)
transaction_amount = np.random.exponential(scale=200, size=n_transactions) + 10

print(f"\n数据概览: {n_customers}位客户, {n_transactions}笔交易")
print(f"平均交易金额: {transaction_amount.mean():.0f}元")
print(f"交易金额范围: {transaction_amount.min():.0f} ~ {transaction_amount.max():.0f}元")


# ============================================================
# 2. 完整RFM分析流水线
# ============================================================
print("\n--- RFM特征计算 ---")

customer_data = defaultdict(lambda: {'days': [], 'amounts': []})
for cid, day, amt in zip(transaction_customer, transaction_days_ago, transaction_amount):
    customer_data[cid]['days'].append(day)
    customer_data[cid]['amounts'].append(amt)

rfm_data = []
for cid in customer_ids:
    data = customer_data[cid]
    recency = min(data['days'])
    frequency = len(data['days'])
    monetary = sum(data['amounts'])
    rfm_data.append({'customer_id': cid, 'recency': recency,
                     'frequency': frequency, 'monetary': monetary})

rfm = np.array([[d['recency'], d['frequency'], d['monetary']] for d in rfm_data])

# Log变换 + 标准化
rfm_log = np.log1p(rfm)
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_log)

print(f"RFM特征: {rfm.shape[0]}个客户 x 3个特征")


# ============================================================
# 3. 聚类
# ============================================================
print("\n--- K-Means聚类 ---")

# 寻找最优K
K_range = range(2, 8)
sil_scores = []
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(rfm_scaled)
    sil_scores.append(silhouette_score(rfm_scaled, labels))

best_k = list(K_range)[np.argmax(sil_scores)]
print(f"最优K��: {best_k} (轮廓系数={max(sil_scores):.4f})")

# 最终聚类
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = km_final.fit_predict(rfm_scaled)


# ============================================================
# 4. 簇画像汇总表
# ============================================================
print("\n" + "=" * 80)
print("  簇画像汇总表")
print("=" * 80)
print(f"{'簇':<6} {'人数':>6} {'占比':>8} {'R均值':>8} {'F均值':>8} {'M均值':>10} {'客户类型':<14}")
print("-" * 80)

cluster_stats = []
for c in range(best_k):
    mask = cluster_labels == c
    count = mask.sum()
    pct = count / len(cluster_labels) * 100
    r_mean = rfm[mask, 0].mean()
    f_mean = rfm[mask, 1].mean()
    m_mean = rfm[mask, 2].mean()
    cluster_stats.append({
        'cluster': c, 'count': count, 'pct': pct,
        'r_mean': r_mean, 'f_mean': f_mean, 'm_mean': m_mean
    })

# 按Monetary降序排列，赋予业务标签
cluster_stats.sort(key=lambda x: x['m_mean'], reverse=True)

labels_map = ['高价值客户', '中高价值客户', '中等价值客户', '低价值客户', '潜在流失客户', '沉睡客户', '待定']

for i, cs in enumerate(cluster_stats):
    label = labels_map[i] if i < len(labels_map) else f'群组{i}'
    print(f"  簇{cs['cluster']:<4} {cs['count']:>5}人 {cs['pct']:>7.1f}% "
          f"{cs['r_mean']:>7.1f}天 {cs['f_mean']:>7.1f}次 {cs['m_mean']:>9.0f}元  {label}")
    cs['label'] = label

print("-" * 80)


# ============================================================
# 5. 多面板可视化
# ============================================================

# --- 5a. RFM分布直方图 ---
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

rfm_names = ['Recency (天)', 'Frequency (次)', 'Monetary (元)']
rfm_colors = ['#e74c3c', '#2ecc71', '#3498db']

for i in range(3):
    axes[i].hist(rfm[:, i], bins=40, color=rfm_colors[i], alpha=0.7, edgecolor='white')
    axes[i].axvline(rfm[:, i].mean(), color='black', linestyle='--', label=f"均值={rfm[:, i].mean():.0f}")
    axes[i].set_title(rfm_names[i])
    axes[i].set_xlabel(rfm_names[i])
    axes[i].set_ylabel('客户数')
    axes[i].legend(fontsize=8)

plt.suptitle('RFM特征分布', fontsize=13)
plt.tight_layout()
plt.show()


# --- 5b. 簇规模饼图 ---
fig, ax = plt.subplots(figsize=(8, 6))

sizes = [cs['count'] for cs in cluster_stats]
labels_pie = [f"{cs['label']}\n({cs['count']}人)" for cs in cluster_stats]
colors_pie = plt.cm.Set2(np.linspace(0, 1, best_k))

wedges, texts, autotexts = ax.pie(
    sizes, labels=labels_pie, colors=colors_pie,
    autopct='%1.1f%%', startangle=90, pctdistance=0.75,
    textprops={'fontsize': 10}
)
ax.set_title('各簇客户占比', fontsize=13)
plt.tight_layout()
plt.show()


# --- 5c. 各簇RFM对比柱状图 ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

cluster_order = [cs['cluster'] for cs in cluster_stats]
cluster_labels_sorted = [cs['label'] for cs in cluster_stats]
bar_colors = plt.cm.Set2(np.linspace(0, 1, best_k))

for i, (name, key) in enumerate([('Recency', 'r_mean'), ('Frequency', 'f_mean'), ('Monetary', 'm_mean')]):
    values = [cs[key] for cs in cluster_stats]
    bars = axes[i].bar(range(best_k), values, color=bar_colors, edgecolor='white')
    axes[i].set_xticks(range(best_k))
    axes[i].set_xticklabels([f'簇{cs["cluster"]}' for cs in cluster_stats], fontsize=9)
    axes[i].set_title(name, fontsize=12)
    axes[i].set_ylabel(name)

    # 标注数值
    for bar, val in zip(bars, values):
        axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=9)

plt.suptitle('各簇RFM均值对比', fontsize=13)
plt.tight_layout()
plt.show()


# --- 5d. 雷达图 ---
cluster_profiles = np.zeros((best_k, 3))
for c in range(best_k):
    mask = cluster_labels == c
    cluster_profiles[c] = rfm_scaled[mask].mean(axis=0)

# 归一化
prof_min = cluster_profiles.min(axis=0)
prof_max = cluster_profiles.max(axis=0)
prof_norm = (cluster_profiles - prof_min) / (prof_max - prof_min + 1e-8)

# 按业务排序
prof_norm_sorted = np.array([prof_norm[cs['cluster']] for cs in cluster_stats])

categories = ['Recency\n(新近度)', 'Frequency\n(频次)', 'Monetary\n(金额)']
angles = np.linspace(0, 2 * np.pi, 3, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
radar_colors = plt.cm.Set2(np.linspace(0, 1, best_k))

for i, cs in enumerate(cluster_stats):
    values = prof_norm_sorted[i].tolist()
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=cs['label'], color=radar_colors[i])
    ax.fill(angles, values, alpha=0.1, color=radar_colors[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0, 1.2)
ax.set_title('各簇RFM雷达图', fontsize=13, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1))
plt.tight_layout()
plt.show()


# --- 5e. PCA 2D投影 + 簇颜色 ---
pca = PCA(n_components=2)
X_2d = pca.fit_transform(rfm_scaled)

fig, ax = plt.subplots(figsize=(10, 7))

for i, cs in enumerate(cluster_stats):
    c = cs['cluster']
    mask = cluster_labels == c
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1], s=15, alpha=0.5,
               color=radar_colors[i], label=cs['label'])

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('RFM用户分群 — PCA 2D投影', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()


# ============================================================
# 6. 业务解读 + 策略建议
# ============================================================
print("\n" + "=" * 70)
print("   业务解读与营销策略建议")
print("=" * 70)

strategies = {
    '高价值客户': (
        "最近有购买、购买频次高、消费金额大的核心客户。",
        "→ 策略: VIP专属权益、新品优先体验、个性化推荐、专属客服。"
    ),
    '中高价值客户': (
        "消费能力较强，但活跃度或频次略有不足。",
        "→ 策略: 交叉销售/向上销售、限时优惠刺激消费、会员升级邀请。"
    ),
    '中等价值客户': (
        "消费表现居中，有提升潜力。",
        "→ 策略: 定期推送个性化优惠券、节日营销活动、积分加倍激励。"
    ),
    '低价值客户': (
        "消费金额和频次较低，需要培育。",
        "→ 策略: 入门级优惠、降低购买门槛、内容营销引导需求。"
    ),
    '潜在流失客户': (
        "曾经活跃但近期未消费，有流失风险。",
        "→ 策略: 召回活动、专属折扣、问卷调查流失原因、短信/邮件唤醒。"
    ),
    '沉睡客户': (
        "长期未消费，可能已流失。",
        "→ 策略: 高力度召回优惠、重新激活礼包，若仍无响应则降低营销投入。"
    ),
}

for cs in cluster_stats:
    label = cs['label']
    print(f"\n【{label}】 — {cs['count']}人 ({cs['pct']:.1f}%)")
    if label in strategies:
        desc, strategy = strategies[label]
        print(f"  特征: {desc}")
        print(f"  {strategy}")
    else:
        r, f, m = cs['r_mean'], cs['f_mean'], cs['m_mean']
        print(f"  特征: R={r:.0f}天, F={f:.0f}次, M={m:.0f}元")
        print(f"  → 策略: 根据RFM特征制定针对性营销方案。")

print("\n" + "=" * 70)
print("   分析总结")
print("=" * 70)
print(f"""
  1. 通过RFM模型将 {n_customers} 位客户分为 {best_k} 个群体
  2. 使用Log变换 + 标准化预处理，K-Means聚类效果良好
  3. 轮廓系数: {max(sil_scores):.4f}
  4. 各群体特征差异明显，可支撑差异化营销决策
  5. 建议定期（如月度）更新RFM分群，追踪客户迁移
""")

print("=== Day 7 完成: 分析报告 + 可视化 ===")
