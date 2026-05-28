### Day 6：房价预测（EDA + 特征工程）
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 加载California Housing数据集
# ============================================================
print("=" * 60)
print("California Housing 数据集 — EDA + 特征工程")
print("=" * 60)

housing = fetch_california_housing(as_frame=True)
df = housing.frame.copy()

print(f"\n数据集形状: {df.shape}")
print(f"特征: {list(housing.feature_names)}")
print(f"目标变量: MedHouseVal (中位数房价, 单位: 10万美元)")

# ============================================================
# 2. 基本信息查看
# ============================================================
print("\n--- 前5行数据 ---")
print(df.head())

print("\n--- 数据类型 ---")
print(df.dtypes)

print("\n--- 基本统计 ---")
print(df.describe().round(2))

print("\n--- 缺失值 ---")
print(df.isnull().sum())

# ============================================================
# 3. 目标变量分布 + log变换
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
ax1.hist(df['MedHouseVal'], bins=50, color='steelblue', edgecolor='white')
ax1.set_xlabel('MedHouseVal (10万美元)')
ax1.set_ylabel('频数')
ax1.set_title(f'目标变量分布 (偏度={df["MedHouseVal"].skew():.2f})')
ax1.axvline(df['MedHouseVal'].mean(), color='red', linestyle='--', label='均值')
ax1.axvline(df['MedHouseVal'].median(), color='green', linestyle='--', label='中位数')
ax1.legend()

ax2 = axes[1]
df['MedHouseVal_log'] = np.log1p(df['MedHouseVal'])
ax2.hist(df['MedHouseVal_log'], bins=50, color='coral', edgecolor='white')
ax2.set_xlabel('log(MedHouseVal + 1)')
ax2.set_ylabel('频数')
ax2.set_title(f'Log变换后分布 (偏度={df["MedHouseVal_log"].skew():.2f})')
ax2.axvline(df['MedHouseVal_log'].mean(), color='red', linestyle='--', label='均值')
ax2.axvline(df['MedHouseVal_log'].median(), color='green', linestyle='--', label='中位数')
ax2.legend()

plt.tight_layout()
plt.show()

# ============================================================
# 4. 特征分布
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
axes = axes.flatten()

for ax, col in zip(axes, housing.feature_names):
    ax.hist(df[col], bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax.set_title(col)
    ax.set_ylabel('频数')

plt.suptitle('各特征分布', fontsize=14)
plt.tight_layout()
plt.show()

# ============================================================
# 5. 相关性热力图
# ============================================================
fig, ax = plt.subplots(figsize=(12, 10))
corr = df.drop(columns=['MedHouseVal_log']).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, ax=ax, square=True, linewidths=0.5)
ax.set_title('特征相关性热力图')
plt.tight_layout()
plt.show()

# 与目标变量的相关性排序
print("\n--- 与MedHouseVal的相关性排序 ---")
target_corr = corr['MedHouseVal'].drop('MedHouseVal').sort_values(ascending=False)
print(target_corr.round(3))

fig, ax = plt.subplots(figsize=(8, 5))
target_corr.plot(kind='barh', color=['steelblue' if v >= 0 else 'coral' for v in target_corr])
ax.set_xlabel('相关系数')
ax.set_title('各特征与目标变量的相关性')
ax.axvline(0, color='k', linewidth=0.5)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

# ============================================================
# 6. 关键特征散点图
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# MedInc vs MedHouseVal
ax1 = axes[0]
sample = df.sample(n=2000, random_state=42)
ax1.scatter(sample['MedInc'], sample['MedHouseVal'], alpha=0.3, s=10)
ax1.set_xlabel('MedInc (收入中位数)')
ax1.set_ylabel('MedHouseVal')
ax1.set_title('收入 vs 房价')
ax1.grid(True, alpha=0.3)

# AveRooms vs MedHouseVal
ax2 = axes[1]
ax2.scatter(sample['AveRooms'], sample['MedHouseVal'], alpha=0.3, s=10, color='green')
ax2.set_xlabel('AveRooms (平均房间数)')
ax2.set_ylabel('MedHouseVal')
ax2.set_title('房间数 vs 房价')
ax2.set_xlim(0, 15)
ax2.grid(True, alpha=0.3)

# HouseAge vs MedHouseVal
ax3 = axes[2]
ax3.scatter(sample['HouseAge'], sample['MedHouseVal'], alpha=0.3, s=10, color='coral')
ax3.set_xlabel('HouseAge (房屋年龄)')
ax3.set_ylabel('MedHouseVal')
ax3.set_title('房屋年龄 vs 房价')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# 7. 特征工程 — 创建新特征
# ============================================================
print("\n" + "-" * 60)
print("特征工程")
print("-" * 60)

# 创建派生特征
df['RoomsPerHousehold'] = df['AveRooms'] / df['AveOccup']  # 每户房间数
df['BedroomsPerRoom'] = df['AveBedrms'] / df['AveRooms']   # 卧室占比
df['PopulationPerHousehold'] = df['Population'] / df['AveOccup']  # 每户人口

# 处理异常值 (裁剪到99%分位数)
clip_cols = ['RoomsPerHousehold', 'BedroomsPerRoom', 'PopulationPerHousehold',
             'AveRooms', 'AveBedrms', 'AveOccup', 'Population']
for col in clip_cols:
    upper = df[col].quantile(0.99)
    lower = df[col].quantile(0.01)
    df[col] = df[col].clip(lower, upper)

print("新增特征:")
print("  RoomsPerHousehold = AveRooms / AveOccup")
print("  BedroomsPerRoom = AveBedrms / AveRooms")
print("  PopulationPerHousehold = Population / AveOccup")

# 新特征统计
print("\n新特征统计:")
print(df[['RoomsPerHousehold', 'BedroomsPerRoom', 'PopulationPerHousehold']].describe().round(3))

# ============================================================
# 8. 训练/测试划分 + 标准化 + 保存
# ============================================================
print("\n" + "-" * 60)
print("数据划分 + 标准化")
print("-" * 60)

# 使用原始目标变量 (不用log变换)
feature_cols = list(housing.feature_names) + ['RoomsPerHousehold', 'BedroomsPerRoom', 'PopulationPerHousehold']
X = df[feature_cols].values
y = df['MedHouseVal'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"训练集: {X_train.shape}")
print(f"测试集: {X_test.shape}")

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 保存处理后的数据
save_dir = os.path.dirname(os.path.abspath(__file__))
np.savez(os.path.join(save_dir, 'california_housing_processed.npz'),
         X_train=X_train_scaled,
         X_test=X_test_scaled,
         y_train=y_train,
         y_test=y_test,
         feature_names=feature_cols)

print(f"\n处理后的数据已保存到: california_housing_processed.npz")
print(f"特征列表: {feature_cols}")
