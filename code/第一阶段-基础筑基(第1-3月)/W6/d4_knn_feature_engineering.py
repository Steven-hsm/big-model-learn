"""
W06 Day4: KNN + 特征工程方法论
===============================
内容:
1. KNN 不同 K 值对比，准确率 vs K 曲线
2. 特征缩放对 KNN 的重要性
3. 特征工程示例:
   - SimpleImputer 处理缺失值
   - StandardScaler vs MinMaxScaler 对比
   - LabelEncoder 处理有序分类变量
   - OneHotEncoder 处理无序分类变量
4. 综合演示
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_wine, make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ========== matplotlib 中文显示设置 ==========
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ================================================================
# 1. KNN: 不同 K 值对比
# ================================================================
print("=" * 60)
print("1. KNN: 不同 K 值的准确率对比")
print("=" * 60)

wine = load_wine()
X, y = wine.data, wine.target
print(f"Wine 数据集形状: {X.shape}, 类别数: {len(np.unique(y))}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 先标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

k_range = range(1, 31)
train_accs = []
test_accs = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    train_accs.append(accuracy_score(y_train, knn.predict(X_train_scaled)))
    test_accs.append(accuracy_score(y_test, knn.predict(X_test_scaled)))

best_k = list(k_range)[np.argmax(test_accs)]
print(f"最佳 K 值: {best_k}, 测试集准确率: {max(test_accs):.4f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(k_range, train_accs, 'o-', label='训练集准确率', color='#FF6B6B', markersize=5)
ax.plot(k_range, test_accs, 's-', label='测试集准确率', color='#4ECDC4', markersize=5)
ax.axvline(x=best_k, color='green', linestyle='--', alpha=0.7,
           label=f'最佳 K={best_k}')
ax.set_xlabel('K (近邻数)', fontsize=12)
ax.set_ylabel('准确率', fontsize=12)
ax.set_title('KNN: 准确率 vs K 值', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.set_xticks(list(k_range))
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("D:/code/big-model-learn/code/q_01/W6/d4_knn_k_values.png", dpi=150, bbox_inches='tight')
plt.close()
print("K值对比图已保存")

# ================================================================
# 2. 特征缩放对 KNN 的重要性
# ================================================================
print("\n" + "=" * 60)
print("2. 特征缩放对 KNN 的重要性")
print("=" * 60)

# 使用 make_classification 生成数据
X_mc, y_mc = make_classification(
    n_samples=500, n_features=5, n_informative=3,
    n_redundant=1, random_state=42
)

# 人为制造不同量纲: 某些特征放大 100 倍
X_mc_scaled_version = X_mc.copy()
X_mc_scaled_version[:, 0] *= 100
X_mc_scaled_version[:, 1] *= 50

X_tr, X_te, y_tr, y_te = train_test_split(
    X_mc_scaled_version, y_mc, test_size=0.3, random_state=42
)

scenarios = {
    '无缩放': (X_tr, X_te),
    'StandardScaler': (StandardScaler().fit_transform(X_tr),
                       StandardScaler().fit(X_tr).transform(X_te)),
    'MinMaxScaler': (MinMaxScaler().fit_transform(X_tr),
                     MinMaxScaler().fit(X_tr).transform(X_te)),
}

knn = KNeighborsClassifier(n_neighbors=7)

print(f"{'缩放方式':>18s} | {'准确率':>8s}")
print("-" * 35)
scaling_results = {}
for name, (Xtr, Xte) in scenarios.items():
    knn.fit(Xtr, y_tr)
    acc = accuracy_score(y_te, knn.predict(Xte))
    scaling_results[name] = acc
    print(f"{name:>18s} | {acc:>8.4f}")

fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
bars = ax.bar(scaling_results.keys(), scaling_results.values(), color=colors, edgecolor='white', width=0.5)
for bar, val in zip(bars, scaling_results.values()):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{val:.4f}', ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('准确率', fontsize=12)
ax.set_title('特征缩放对 KNN 准确率的影响', fontsize=14, fontweight='bold')
ax.set_ylim(0, 1.1)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("D:/code/big-model-learn/code/q_01/W6/d4_scaling_importance.png", dpi=150, bbox_inches='tight')
plt.close()
print("特征缩放对比图已保存")

# ================================================================
# 3. SimpleImputer 处理缺失值
# ================================================================
print("\n" + "=" * 60)
print("3. SimpleImputer 处理缺失值")
print("=" * 60)

np.random.seed(42)
X_missing = wine.data[:20].copy()
# 人为制造缺失值
mask = np.random.random(X_missing.shape) < 0.1
X_missing[mask] = np.nan

print(f"缺失值数量: {np.sum(mask)}")

strategies = ['mean', 'median', 'most_frequent']
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, strategy in zip(axes, strategies):
    imputer = SimpleImputer(strategy=strategy)
    X_imputed = imputer.fit_transform(X_missing)

    n_missing = np.sum(mask)
    original_vals = wine.data[:20][mask]
    imputed_vals = X_imputed[mask]

    ax.scatter(range(n_missing), original_vals, color='#4ECDC4', label='原始值', s=40, zorder=3)
    ax.scatter(range(n_missing), imputed_vals, color='#FF6B6B', marker='x',
               label='填充值', s=60, zorder=4)
    ax.set_title(f"填充策略: {strategy}", fontsize=12)
    ax.set_xlabel('缺失值索引', fontsize=10)
    ax.set_ylabel('特征值', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('SimpleImputer 不同填充策略对比', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("D:/code/big-model-learn/code/q_01/W6/d4_imputation.png", dpi=150, bbox_inches='tight')
plt.close()
print("缺失值填充对比图已保存")

# ================================================================
# 4. StandardScaler vs MinMaxScaler 对比
# ================================================================
print("\n" + "=" * 60)
print("4. StandardScaler vs MinMaxScaler 对比")
print("=" * 60)

# 使用 wine 数据集前两个特征做可视化
X_2feat = wine.data[:, :2]  # alcohol, malic_acid
feat_names = [wine.feature_names[0], wine.feature_names[1]]

std_scaler = StandardScaler()
mm_scaler = MinMaxScaler()
X_std = std_scaler.fit_transform(X_2feat)
X_mm = mm_scaler.fit_transform(X_2feat)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 原始数据
axes[0].scatter(X_2feat[:, 0], X_2feat[:, 1], c=wine.target,
                cmap=plt.cm.Set1, edgecolors='black', alpha=0.7)
axes[0].set_title('原始数据', fontsize=13, fontweight='bold')
axes[0].set_xlabel(feat_names[0], fontsize=10)
axes[0].set_ylabel(feat_names[1], fontsize=10)

# StandardScaler
axes[1].scatter(X_std[:, 0], X_std[:, 1], c=wine.target,
                cmap=plt.cm.Set1, edgecolors='black', alpha=0.7)
axes[1].set_title('StandardScaler (均值=0, 方差=1)', fontsize=13, fontweight='bold')
axes[1].set_xlabel(f'{feat_names[0]} (标准化)', fontsize=10)
axes[1].set_ylabel(f'{feat_names[1]} (标准化)', fontsize=10)

# MinMaxScaler
axes[2].scatter(X_mm[:, 0], X_mm[:, 1], c=wine.target,
                cmap=plt.cm.Set1, edgecolors='black', alpha=0.7)
axes[2].set_title('MinMaxScaler (范围 [0,1])', fontsize=13, fontweight='bold')
axes[2].set_xlabel(f'{feat_names[0]} (归一化)', fontsize=10)
axes[2].set_ylabel(f'{feat_names[1]} (归一化)', fontsize=10)

for ax in axes:
    ax.grid(True, alpha=0.3)

plt.suptitle('特征缩放方法对比', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("D:/code/big-model-learn/code/q_01/W6/d4_scaler_comparison.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"原始数据范围: {feat_names[0]}=[{X_2feat[:, 0].min():.2f}, {X_2feat[:, 0].max():.2f}], "
      f"{feat_names[1]}=[{X_2feat[:, 1].min():.2f}, {X_2feat[:, 1].max():.2f}]")
print(f"StandardScaler: 均值=[{X_std[:, 0].mean():.4f}, {X_std[:, 1].mean():.4f}], "
      f"标准差=[{X_std[:, 0].std():.4f}, {X_std[:, 1].std():.4f}]")
print(f"MinMaxScaler:   范围=[{X_mm[:, 0].min():.4f}, {X_mm[:, 0].max():.4f}]")
print("缩放对比图已保存")

# ================================================================
# 5. LabelEncoder 与 OneHotEncoder
# ================================================================
print("\n" + "=" * 60)
print("5. LabelEncoder 与 OneHotEncoder")
print("=" * 60)

# 创建一个含有分类变量的示例 DataFrame
df = pd.DataFrame({
    '学历': ['高中', '本科', '硕士', '博士', '本科', '高中', '硕士', '博士',
             '本科', '硕士', '高中', '本科', '博士', '硕士', '本科'],
    '城市': ['北京', '上海', '广州', '深圳', '北京', '上海', '广州', '深圳',
             '北京', '上海', '广州', '深圳', '北京', '上海', '广州'],
    '薪资水平': ['低', '中', '高', '高', '中', '低', '高', '高',
                '中', '中', '低', '中', '高', '高', '中']
})

print("原始数据:")
print(df.head(10))

# LabelEncoder: 适用于有序分类变量 (如学历)
le = LabelEncoder()
df['学历_编码'] = le.fit_transform(df['学历'])
print(f"\nLabelEncoder (学历 - 有序变量):")
for cls, code in zip(le.classes_, le.transform(le.classes_)):
    print(f"  {cls} -> {code}")

# OneHotEncoder: 适用于无序分类变量 (如城市)
ohe = OneHotEncoder(sparse_output=False, drop='first')  # drop='first' 避免多重共线性
city_encoded = ohe.fit_transform(df[['城市']])
city_cols = [f'城市_{c}' for c in ohe.categories_[0][1:]]
df_ohe = pd.DataFrame(city_encoded, columns=city_cols)

print(f"\nOneHotEncoder (城市 - 无序变量, drop='first'):")
print(df_ohe.head())

# 为什么要区分?
print("\n--- LabelEncoder vs OneHotEncoder 选择原则 ---")
print("LabelEncoder: 用于有序分类变量 (如: 低<中<高, 小学<中学<大学)")
print("OneHotEncoder: 用于无序分类变量 (如: 北京/上海/广州, 红色/蓝色/绿色)")
print("原因: LabelEncoder 引入了数值大小关系，对无序变量会造成误导")

# ================================================================
# 6. 综合演示: Pipeline 特征工程 + KNN
# ================================================================
print("\n" + "=" * 60)
print("6. 综合演示: Pipeline 特征工程 + KNN")
print("=" * 60)

# 使用 wine 数据集，人为加入缺失值和不同量纲
np.random.seed(42)
X_wine = wine.data.copy()
y_wine = wine.target

# 人为添加缺失值
missing_mask = np.random.random(X_wine.shape) < 0.05
X_wine[missing_mask] = np.nan

X_tr, X_te, y_tr, y_te = train_test_split(X_wine, y_wine, test_size=0.3, random_state=42)

# 构建 Pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=7)),
])

pipeline.fit(X_tr, y_tr)
y_pred = pipeline.predict(X_te)
acc = accuracy_score(y_te, y_pred)

print(f"Pipeline (Imputer + Scaler + KNN) 准确率: {acc:.4f}")
print(f"Pipeline 步骤: {[name for name, _ in pipeline.steps]}")

# 对比不使用 Pipeline 的结果
knn_raw = KNeighborsClassifier(n_neighbors=7)
# 直接用训练集（有缺失值会报错，用填充后的）
X_tr_filled = SimpleImputer(strategy='median').fit_transform(X_tr)
X_te_filled = SimpleImputer(strategy='median').fit_transform(X_te)
knn_raw.fit(X_tr_filled, y_tr)
acc_raw = accuracy_score(y_te, knn_raw.predict(X_te_filled))
print(f"仅填充无缩放 KNN 准确率: {acc_raw:.4f}")
print(f"Pipeline 提升: {(acc - acc_raw) * 100:.2f}%")

# ================================================================
# 总结
# ================================================================
print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print("1. K 值选择: K太小过拟合, K太大欠拟合, 需要交叉验证选择")
print("2. 特征缩放对 KNN 至关重要 (KNN 基于距离计算)")
print("3. StandardScaler: 均值0方差1, 适合近似正态分布的数据")
print("4. MinMaxScaler: 缩放到[0,1], 适合有界数据")
print("5. SimpleImputer: mean/median/most_frequent 三种填充策略")
print("6. LabelEncoder 用于有序分类, OneHotEncoder 用于无序分类")
print("7. Pipeline 将预处理和模型训练整合，避免数据泄露")
