"""
W08 Day 1: Kaggle实战 - EDA + 数据探索
使用 California Housing 数据集模拟 Kaggle 竞赛的完整 EDA 流程
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# ============================================================
# Matplotlib 中文显示设置
# ============================================================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 数据加载
# ============================================================
print("=" * 60)
print("California Housing 数据集 - Kaggle EDA 实战")
print("=" * 60)

housing_raw = fetch_california_housing(as_frame=True)
df = housing_raw.frame.copy()

# 重命名列为中文（方便展示）
df.columns = ['收入中位数', '房龄中位数', '平均房间数', '平均卧室数',
              '人口', '家庭数', '纬度', '经度', '房价中位数']

print("\n--- 1.1 数据形状 ---")
print(f"数据集形状: {df.shape}")
print(f"  样本数: {df.shape[0]}")
print(f"  特征数: {df.shape[1] - 1}")

print("\n--- 1.2 数据信息 ---")
print(df.info())

print("\n--- 1.3 描述性统计 ---")
print(df.describe().round(2))

# ============================================================
# 2. 缺失值检查
# ============================================================
print("\n--- 2. 缺失值检查 ---")
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({'缺失数量': missing, '缺失比例(%)': missing_pct})
print(missing_df)
print(f"\n总缺失值数量: {df.isnull().sum().sum()}")

# ============================================================
# 3. 目标变量分布分析
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 原始分布
axes[0].hist(df['房价中位数'], bins=50, color='steelblue', edgecolor='white', alpha=0.8)
axes[0].set_xlabel('房价中位数 (万美元)', fontsize=12)
axes[0].set_ylabel('频数', fontsize=12)
axes[0].set_title('目标变量原始分布', fontsize=14)
axes[0].axvline(df['房价中位数'].mean(), color='red', linestyle='--',
                label=f'均值={df["房价中位数"].mean():.2f}')
axes[0].legend()

# 对数变换后的分布
log_target = np.log1p(df['房价中位数'])
axes[1].hist(log_target, bins=50, color='darkorange', edgecolor='white', alpha=0.8)
axes[1].set_xlabel('log(房价中位数 + 1)', fontsize=12)
axes[1].set_ylabel('频数', fontsize=12)
axes[1].set_title('目标变量对数变换后分布', fontsize=14)
axes[1].axvline(log_target.mean(), color='red', linestyle='--',
                label=f'均值={log_target.mean():.2f}')
axes[1].legend()

plt.tight_layout()
plt.savefig('d1_target_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[图表已保存] d1_target_distribution.png")

# 偏度与峰度
from scipy import stats
skew = stats.skew(df['房价中位数'])
kurt = stats.kurtosis(df['房价中位数'])
skew_log = stats.skew(log_target)
print(f"\n原始目标变量 - 偏度: {skew:.4f}, 峰度: {kurt:.4f}")
print(f"对数变换后   - 偏度: {skew_log:.4f}")

# ============================================================
# 4. 相关性分析
# ============================================================
print("\n--- 4. 相关性分析 ---")
corr_matrix = df.corr()

# 与目标变量的相关系数排序
target_corr = corr_matrix['房价中位数'].drop('房价中位数').sort_values(ascending=False)
print("\n各特征与房价中位数的相关系数:")
print(target_corr.round(4).to_string())

# 相关性热力图（全部特征）
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# 完整热力图
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5, ax=axes[0],
            vmin=-1, vmax=1)
axes[0].set_title('特征间完整相关性热力图', fontsize=14)

# 与目标变量相关的 Top 特征热力图
top_features = target_corr.abs().nlargest(6).index.tolist() + ['房价中位数']
top_corr = df[top_features].corr()
sns.heatmap(top_corr, annot=True, fmt='.2f', cmap='YlOrRd',
            square=True, linewidths=0.5, ax=axes[1])
axes[1].set_title('Top 相关特征热力图', fontsize=14)

plt.tight_layout()
plt.savefig('d1_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("[图表已保存] d1_correlation_heatmap.png")

# ============================================================
# 5. Top 相关特征的散点图
# ============================================================
top3_features = target_corr.abs().nlargest(3).index.tolist()
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, feature in enumerate(top3_features):
    axes[idx].scatter(df[feature], df['房价中位数'], alpha=0.3, s=5, c='steelblue')
    axes[idx].set_xlabel(feature, fontsize=11)
    axes[idx].set_ylabel('房价中位数', fontsize=11)
    axes[idx].set_title(f'{feature} vs 房价中位数\n(r={target_corr[feature]:.3f})', fontsize=12)

    # 添加趋势线
    z = np.polyfit(df[feature], df['房价中位数'], 1)
    p = np.poly1d(z)
    x_sorted = np.sort(df[feature])
    axes[idx].plot(x_sorted, p(x_sorted), 'r-', linewidth=2, alpha=0.7)

plt.tight_layout()
plt.savefig('d1_scatter_top_features.png', dpi=150, bbox_inches='tight')
plt.close()
print("[图表已保存] d1_scatter_top_features.png")

# ============================================================
# 6. 地理可视化
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# 按房价着色的地理散点图
sc1 = axes[0].scatter(df['经度'], df['纬度'], c=df['房价中位数'],
                      cmap='jet', alpha=0.4, s=3)
axes[0].set_xlabel('经度', fontsize=12)
axes[0].set_ylabel('纬度', fontsize=12)
axes[0].set_title('加州房价地理分布', fontsize=14)
plt.colorbar(sc1, ax=axes[0], label='房价中位数 (万美元)')

# 按人口密度着色
sc2 = axes[1].scatter(df['经度'], df['纬度'], c=df['人口'],
                      cmap='YlOrRd', alpha=0.4, s=3,
                      norm=plt.Normalize(vmin=0, vmax=df['人口'].quantile(0.95)))
axes[1].set_xlabel('经度', fontsize=12)
axes[1].set_ylabel('纬度', fontsize=12)
axes[1].set_title('加州人口密度地理分布', fontsize=14)
plt.colorbar(sc2, ax=axes[1], label='人口')

plt.tight_layout()
plt.savefig('d1_geographic_visualization.png', dpi=150, bbox_inches='tight')
plt.close()
print("[图表已保存] d1_geographic_visualization.png")

# ============================================================
# 7. 特征分布总览
# ============================================================
fig, axes = plt.subplots(3, 3, figsize=(16, 14))
axes = axes.ravel()

features = df.columns.tolist()
colors = plt.cm.Set2(np.linspace(0, 1, len(features)))

for idx, (col, color) in enumerate(zip(features, colors)):
    axes[idx].hist(df[col], bins=50, color=color, edgecolor='white', alpha=0.8)
    axes[idx].set_title(col, fontsize=12)
    axes[idx].set_ylabel('频数')

    # 添加均值和中位数线
    axes[idx].axvline(df[col].mean(), color='red', linestyle='--', alpha=0.7)
    axes[idx].axvline(df[col].median(), color='green', linestyle='-', alpha=0.7)

plt.suptitle('所有特征分布总览', fontsize=16, y=1.01)
plt.tight_layout()
plt.savefig('d1_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("[图表已保存] d1_feature_distributions.png")

# ============================================================
# 8. EDA 关键发现总结
# ============================================================
print("\n" + "=" * 60)
print("EDA 关键发现总结")
print("=" * 60)
print(f"""
1. 数据规模: {df.shape[0]:,} 条样本, {df.shape[1]-1} 个特征
2. 缺失值: {df.isnull().sum().sum()} 个 (无需额外处理)
3. 目标变量: 偏度={skew:.2f}, 存在右偏, 对数变换可改善
4. 最相关特征: {top3_features[0]} (r={target_corr.iloc[0]:.3f})
5. 地理因素: 靠近海岸(SF/LA)房价显著更高
6. 房价存在上限截断(5.0万美元), 可能需要特殊处理
7. 多特征之间存在高相关性, 后续可考虑降维或特征选择
""")
