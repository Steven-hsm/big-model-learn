"""
W06 Day6: 项目2 - 客户流失预测 (EDA + 特征工程)
================================================
内容:
1. 生成合成客户流失数据
2. EDA: 目标分布、特征分布、分组统计
3. 相关性分析
4. 特征工程: 编码分类变量、缩放数值特征
5. 使用 ColumnTransformer 构建预处理管道
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ========== matplotlib 中文显示设置 ==========
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ================================================================
# 1. 生成合成客户流失数据
# ================================================================
print("=" * 60)
print("1. 生成合成客户流失数据")
print("=" * 60)

np.random.seed(42)
n_samples = 2000

# 数值特征
tenure = np.random.exponential(scale=30, size=n_samples).astype(int)
tenure = np.clip(tenure, 1, 72)
monthly_charges = np.round(np.random.uniform(20, 120, n_samples), 2)
total_charges = np.round(tenure * monthly_charges * np.random.uniform(0.8, 1.2, n_samples), 2)
age = np.random.randint(18, 75, n_samples)
num_dependents = np.random.choice([0, 0, 0, 1, 1, 2, 3], size=n_samples)

# 分类特征
contract_type = np.random.choice(
    ['月付', '一年', '两年'], size=n_samples, p=[0.5, 0.3, 0.2]
)
internet_service = np.random.choice(
    ['DSL', '光纤', '无'], size=n_samples, p=[0.35, 0.45, 0.2]
)
payment_method = np.random.choice(
    ['电子支票', '邮寄支票', '银行转账', '信用卡'],
    size=n_samples, p=[0.35, 0.2, 0.25, 0.2]
)
online_security = np.random.choice(['是', '否'], size=n_samples, p=[0.4, 0.6])
tech_support = np.random.choice(['是', '否'], size=n_samples, p=[0.35, 0.65])
gender = np.random.choice(['男', '女'], size=n_samples, p=[0.5, 0.5])

# 生成流失标签 (与某些特征相关)
churn_prob = (
    0.1
    - 0.008 * tenure
    + 0.005 * monthly_charges
    + (contract_type == '月付') * 0.25
    - (contract_type == '两年') * 0.2
    + (internet_service == '光纤') * 0.1
    + (online_security == '否') * 0.1
    + (tech_support == '否') * 0.08
    + (payment_method == '电子支票') * 0.12
    + 0.003 * (age - 40) ** 2 / 100
)
churn_prob = np.clip(churn_prob, 0.02, 0.9)
churn = (np.random.random(n_samples) < churn_prob).astype(int)

# 构建 DataFrame
df = pd.DataFrame({
    '性别': gender,
    '年龄': age,
    '在网时长(月)': tenure,
    '月费(元)': monthly_charges,
    '总费用(元)': total_charges,
    '家属数': num_dependents,
    '合同类型': contract_type,
    '网络服务': internet_service,
    '支付方式': payment_method,
    '在线安全': online_security,
    '技术支持': tech_support,
    '是否流失': churn,
})

print(f"数据形状: {df.shape}")
print(f"\n前5行:")
print(df.head())
print(f"\n数据类型:")
print(df.dtypes)
print(f"\n基本统计:")
print(df.describe())

# ================================================================
# 2. EDA: 目标分布
# ================================================================
print("\n" + "=" * 60)
print("2. EDA: 目标分布")
print("=" * 60)

churn_counts = df['是否流失'].value_counts()
churn_rate = churn_counts[1] / len(df) * 100

print(f"未流失: {churn_counts[0]} ({100 - churn_rate:.1f}%)")
print(f"已流失: {churn_counts[1]} ({churn_rate:.1f}%)")
print(f"流失率: {churn_rate:.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 饼图
axes[0].pie(churn_counts.values, labels=['未流失', '已流失'],
            autopct='%1.1f%%', colors=['#4ECDC4', '#FF6B6B'],
            startangle=90, explode=[0, 0.05])
axes[0].set_title('客户流失分布', fontsize=14, fontweight='bold')

# 柱状图
axes[1].bar(['未流失', '已流失'], churn_counts.values,
            color=['#4ECDC4', '#FF6B6B'], edgecolor='white')
for i, v in enumerate(churn_counts.values):
    axes[1].text(i, v + 20, f'{v}', ha='center', fontsize=12, fontweight='bold')
axes[1].set_ylabel('客户数', fontsize=12)
axes[1].set_title('客户流失数量', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig("D:/code/big-model-learn/code/q_01/W6/d6_churn_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print("流失分布图已保存")

# ================================================================
# 3. EDA: 数值特征分布 (按流失分组)
# ================================================================
print("\n" + "=" * 60)
print("3. EDA: 数值特征分布")
print("=" * 60)

numeric_cols = ['在网时长(月)', '月费(元)', '总费用(元)', '年龄']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, col in zip(axes.flat, numeric_cols):
    for label, color, name in [(0, '#4ECDC4', '未流失'), (1, '#FF6B6B', '已流失')]:
        subset = df[df['是否流失'] == label][col]
        ax.hist(subset, bins=30, alpha=0.6, color=color, label=name, edgecolor='white')

    ax.set_xlabel(col, fontsize=11)
    ax.set_ylabel('频数', fontsize=11)
    ax.set_title(f'{col} 分布 (按流失分组)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.suptitle('数值特征分布 (按是否流失分组)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("D:/code/big-model-learn/code/q_01/W6/d6_numeric_distributions.png", dpi=150, bbox_inches='tight')
plt.close()
print("数值特征分布图已保存")

# 打印分组统计
print("\n数值特征分组均值:")
group_stats = df.groupby('是否流失')[numeric_cols].mean()
print(group_stats.round(2))

# ================================================================
# 4. EDA: 分类特征与流失率
# ================================================================
print("\n" + "=" * 60)
print("4. EDA: 分类特征与流失率")
print("=" * 60)

categorical_cols = ['合同类型', '网络服务', '支付方式', '在线安全', '技术支持']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes_flat = axes.flat

for ax, col in zip(axes_flat, categorical_cols):
    churn_by_cat = df.groupby(col)['是否流失'].mean() * 100
    bars = ax.bar(churn_by_cat.index, churn_by_cat.values,
                  color=plt.cm.Set2(np.linspace(0, 1, len(churn_by_cat))),
                  edgecolor='white')
    ax.set_ylabel('流失率 (%)', fontsize=11)
    ax.set_title(f'{col} vs 流失率', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, churn_by_cat.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')

    ax.tick_params(axis='x', rotation=15)

# 隐藏多余的子图
axes_flat[-1].set_visible(False)

plt.suptitle('分类特征与流失率关系', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("D:/code/big-model-learn/code/q_01/W6/d6_categorical_churn.png", dpi=150, bbox_inches='tight')
plt.close()
print("分类特征流失率图已保存")

# 打印各分类特征的流失率
for col in categorical_cols:
    print(f"\n{col} 各类别流失率:")
    churn_rate_by_cat = df.groupby(col)['是否流失'].agg(['mean', 'count'])
    churn_rate_by_cat.columns = ['流失率', '样本数']
    churn_rate_by_cat['流失率'] = (churn_rate_by_cat['流失率'] * 100).round(2)
    print(churn_rate_by_cat)

# ================================================================
# 5. 相关性分析
# ================================================================
print("\n" + "=" * 60)
print("5. 相关性分析")
print("=" * 60)

numeric_df = df[numeric_cols + ['是否流失']]
corr_matrix = numeric_df.corr()

print("与流失的相关系数:")
churn_corr = corr_matrix['是否流失'].drop('是否流失').sort_values(ascending=False)
for feat, corr_val in churn_corr.items():
    print(f"  {feat}: {corr_val:.4f}")

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r',
            center=0, ax=ax, square=True, linewidths=0.5,
            vmin=-1, vmax=1)
ax.set_title('数值特征相关性热力图', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("D:/code/big-model-learn/code/q_01/W6/d6_correlation.png", dpi=150, bbox_inches='tight')
plt.close()
print("相关性热力图已保存")

# ================================================================
# 6. 特征工程
# ================================================================
print("\n" + "=" * 60)
print("6. 特征工程")
print("=" * 60)

# 分离特征和标签
X = df.drop('是否流失', axis=1)
y = df['是否流失']

print(f"特征形状: {X.shape}")
print(f"特征列: {list(X.columns)}")

# 定义数值列和分类列
numeric_features = ['年龄', '在网时长(月)', '月费(元)', '总费用(元)', '家属数']
categorical_features = ['性别', '合同类型', '网络服务', '支付方式', '在线安全', '技术支持']

print(f"\n数值特征: {numeric_features}")
print(f"分类特征: {categorical_features}")

# 创建 ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'),
         categorical_features),
    ],
    remainder='drop'
)

# 演示预处理效果
X_sample = X.head(5)
X_transformed = preprocessor.fit_transform(X)

# 获取编码后的特征名
num_feature_names = numeric_features
cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names = list(num_feature_names) + list(cat_feature_names)

print(f"\n编码后特征数: {len(all_feature_names)}")
print(f"编码后形状: {X_transformed.shape}")
print(f"\n新增的 OneHot 特征 (前10个): {list(cat_feature_names)[:10]}")

# 转为 DataFrame 查看
X_transformed_df = pd.DataFrame(X_transformed, columns=all_feature_names)
print(f"\n转换后前3行:")
print(X_transformed_df.head(3).round(3))

# ================================================================
# 7. 划分数据集并保存预处理信息
# ================================================================
print("\n" + "=" * 60)
print("7. 划分数据集")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"训练集: {X_train.shape[0]} 样本")
print(f"测试集: {X_test.shape[0]} 样本")
print(f"训练集流失率: {y_train.mean() * 100:.1f}%")
print(f"测试集流失率: {y_test.mean() * 100:.1f}%")

# 演示完整 Pipeline 预处理
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
])

X_train_processed = full_pipeline.fit_transform(X_train)
X_test_processed = full_pipeline.transform(X_test)

print(f"\nPipeline 处理后:")
print(f"  训练集形状: {X_train_processed.shape}")
print(f"  测试集形状: {X_test_processed.shape}")

# 保存数据供 d7 使用
np.savez(
    "D:/code/big-model-learn/code/q_01/W6/d6_churn_data.npz",
    X_train=X_train_processed,
    X_test=X_test_processed,
    y_train=y_train.values,
    y_test=y_test.values,
    feature_names=np.array(all_feature_names),
    X_train_raw=X_train.values,
    X_test_raw=X_test.values,
    numeric_features=np.array(numeric_features),
    categorical_features=np.array(categorical_features),
)
print("\n处理后的数据已保存到 d6_churn_data.npz")

# ================================================================
# 总结
# ================================================================
print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print("1. 合成数据模拟了真实客户流失场景，流失率约 30%")
print("2. EDA 发现: 月付合同、电子支票、无在线安全 -> 流失率更高")
print("3. 在网时长与流失负相关，月费与流失正相关")
print("4. ColumnTransformer 可同时处理数值和分类特征")
print("5. StandardScaler 标准化数值特征，OneHotEncoder 编码分类特征")
print("6. 处理后的数据已保存，供 d7 模型训练使用")
