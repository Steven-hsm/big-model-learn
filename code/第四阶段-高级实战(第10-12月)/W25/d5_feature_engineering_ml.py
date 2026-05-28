"""
W25-D5 自动特征工程
====================
自动特征工程(Featuretools概念), 特征交叉, 时序特征, 文本特征提取
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    HAS_TFIDF = True
except ImportError:
    HAS_TFIDF = False

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W25-D5 自动特征工程")
print("=" * 60)

# ============================================================
# 1. 自动特征工程概念
# ============================================================
print("\n--- 1. 自动特征工程概念 ---")
print("""
Featuretools 核心概念:

  - EntitySet:    数据集集合(多表关联)
  - Entity:       单个数据表
  - Relationship: 表间关系(类似外键)
  - Feature Primitives: 特征原语(聚合/转换)
    - Aggregation:  MEAN, SUM, COUNT, MAX, MIN ...
    - Transform:   HOUR, DAY, ABS, DIFF ...

  Deep Feature Synthesis (DFS):
    自动组合原语生成新特征, 支持多表关联

  其他工具:
  - AutoFeat: 自动特征合成与选择
  - TSFresh:  时序特征自动提取(1000+特征)
  - Feature-engine: 特征工程Pipeline工具
""")

# ============================================================
# 2. 简化版自动特征合成
# ============================================================
print("\n--- 2. 简化版自动特征合成 ---")


class AutoFeatureEngineer:
    """简化版自动特征工程"""

    def __init__(self):
        self.generated_features = {}
        self.feature_importances = {}

    def numeric_transforms(self, df, columns):
        """数值特征变换"""
        new_features = pd.DataFrame(index=df.index)
        generated = []

        for col in columns:
            # 对数变换
            if (df[col] > 0).all():
                new_features[f'{col}_log'] = np.log1p(df[col])
                generated.append(f'{col}_log')

            # 平方根变换
            if (df[col] >= 0).all():
                new_features[f'{col}_sqrt'] = np.sqrt(df[col])
                generated.append(f'{col}_sqrt')

            # 平方
            new_features[f'{col}_sq'] = df[col] ** 2
            generated.append(f'{col}_sq')

            # 标准化
            mean, std = df[col].mean(), df[col].std()
            if std > 0:
                new_features[f'{col}_zscore'] = (df[col] - mean) / std
                generated.append(f'{col}_zscore')

            # 分箱
            try:
                new_features[f'{col}_binned'] = pd.qcut(df[col], q=5, labels=False, duplicates='drop')
                generated.append(f'{col}_binned')
            except ValueError:
                pass

        self.generated_features['numeric_transforms'] = generated
        print(f"  数值变换: 生成 {len(generated)} 个特征")
        return new_features

    def feature_crosses(self, df, columns, max_pairs=10):
        """特征交叉"""
        new_features = pd.DataFrame(index=df.index)
        generated = []
        pairs = []

        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                pairs.append((columns[i], columns[j]))

        # 限制数量
        pairs = pairs[:max_pairs]

        for col_a, col_b in pairs:
            # 乘法交叉
            new_features[f'{col_a}_x_{col_b}'] = df[col_a] * df[col_b]
            generated.append(f'{col_a}_x_{col_b}')

            # 加法交叉
            new_features[f'{col_a}_plus_{col_b}'] = df[col_a] + df[col_b]
            generated.append(f'{col_a}_plus_{col_b}')

            # 比率
            if (df[col_b] != 0).all():
                new_features[f'{col_a}_div_{col_b}'] = df[col_a] / df[col_b]
                generated.append(f'{col_a}_div_{col_b}')

            # 差值绝对值
            new_features[f'{col_a}_diff_{col_b}'] = (df[col_a] - df[col_b]).abs()
            generated.append(f'{col_a}_diff_{col_b}')

        self.generated_features['feature_crosses'] = generated
        print(f"  特征交叉: 生成 {len(generated)} 个特征")
        return new_features

    def aggregation_features(self, df, group_col, value_cols):
        """聚合特征"""
        new_features = pd.DataFrame(index=df.index)
        generated = []

        for val_col in value_cols:
            group_stats = df.groupby(group_col)[val_col].agg(['mean', 'std', 'min', 'max', 'count'])
            group_stats.columns = [f'{val_col}_grp_{stat}' for stat in group_stats.columns]
            group_stats = group_stats.rename_axis(group_col).reset_index()

            # 合并回原表
            merged = df[[group_col]].merge(group_stats, on=group_col, how='left')
            for stat_col in group_stats.columns:
                if stat_col != group_col:
                    new_features[stat_col] = merged[stat_col]
                    generated.append(stat_col)

        self.generated_features['aggregations'] = generated
        print(f"  聚合特征: 生成 {len(generated)} 个特征")
        return new_features

    def select_features(self, X, y, top_k=20):
        """基于随机森林的特征选择"""
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        importances = pd.Series(rf.feature_importances_, index=X.columns)
        importances = importances.sort_values(ascending=False)

        self.feature_importances = importances
        selected = importances.head(top_k).index.tolist()
        print(f"  特征选择: 从 {len(importances)} 个中选择 Top {top_k}")
        return selected, importances


# --- 生成模拟数据 ---
np.random.seed(42)
n = 1000
df = pd.DataFrame({
    'user_id': np.random.choice([f'U{i:03d}' for i in range(200)], n),
    'age': np.random.randint(18, 65, n),
    'income': np.random.normal(50000, 15000, n),
    'purchase_amount': np.random.exponential(300, n),
    'browse_time': np.random.exponential(15, n),
    'cart_items': np.random.poisson(3, n),
    'city_tier': np.random.choice([1, 2, 3], n),
    'is_vip': np.random.choice([0, 1], n, p=[0.8, 0.2]),
})

# 目标变量
y = ((df['purchase_amount'] > 400) & (df['cart_items'] > 2)).astype(int)

print(f"\n原始数据: {df.shape}, 正例比例: {y.mean():.2%}")

# --- 运行自动特征工程 ---
afe = AutoFeatureEngineer()

numeric_cols = ['age', 'income', 'purchase_amount', 'browse_time', 'cart_items']

# 数值变换
numeric_feats = afe.numeric_transforms(df, numeric_cols)

# 特征交叉
cross_feats = afe.feature_crosses(df, ['age', 'income', 'browse_time', 'cart_items'], max_pairs=5)

# 聚合特征
agg_feats = afe.aggregation_features(df, 'user_id', ['purchase_amount', 'browse_time'])

# 合并所有特征
all_features = pd.concat([
    df[['age', 'income', 'purchase_amount', 'browse_time', 'cart_items', 'city_tier', 'is_vip']],
    numeric_feats,
    cross_feats,
    agg_feats,
], axis=1)

# 处理无穷大和NaN
all_features = all_features.replace([np.inf, -np.inf], np.nan)
all_features = all_features.fillna(0)

print(f"\n特征工程后: {all_features.shape[1]} 个特征")

# 特征选择
selected, importances = afe.select_features(all_features, y, top_k=15)
print(f"\nTop 15 重要特征:")
for feat, imp in importances.head(15).items():
    bar = '*' * int(imp * 200)
    print(f"  {feat:<35} {imp:.4f} {bar}")

# 评估原始 vs 自动特征
print("\n--- 特征工程效果对比 ---")
# 原始特征
orig_cols = ['age', 'income', 'purchase_amount', 'browse_time', 'cart_items', 'city_tier', 'is_vip']
rf_orig = RandomForestClassifier(n_estimators=100, random_state=42)
scores_orig = cross_val_score(rf_orig, all_features[orig_cols], y, cv=5, scoring='accuracy')

# 自动特征(选择后)
rf_auto = RandomForestClassifier(n_estimators=100, random_state=42)
scores_auto = cross_val_score(rf_auto, all_features[selected], y, cv=5, scoring='accuracy')

print(f"  原始特征 (7个):   准确率={scores_orig.mean():.4f} (+/- {scores_orig.std():.4f})")
print(f"  自动特征 ({len(selected)}个): 准确率={scores_auto.mean():.4f} (+/- {scores_auto.std():.4f})")

# ============================================================
# 3. 时序特征提取
# ============================================================
print("\n\n--- 3. 时序特征提取 ---")


class TimeSeriesFeatureExtractor:
    """时序特征提取器"""

    @staticmethod
    def extract(df, time_col, value_col, group_col=None):
        """提取时序特征"""
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col)

        features = pd.DataFrame(index=df.index)

        # 时间成分
        features['hour'] = df[time_col].dt.hour
        features['day_of_week'] = df[time_col].dt.dayofweek
        features['day_of_month'] = df[time_col].dt.day
        features['month'] = df[time_col].dt.month
        features['is_weekend'] = (df[time_col].dt.dayofweek >= 5).astype(int)

        # 周期编码
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['dow_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['dow_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)

        # 滞后特征
        for lag in [1, 7, 14, 30]:
            if group_col:
                features[f'lag_{lag}'] = df.groupby(group_col)[value_col].shift(lag)
            else:
                features[f'lag_{lag}'] = df[value_col].shift(lag)

        # 滚动统计
        for window in [7, 14, 30]:
            if group_col:
                roll = df.groupby(group_col)[value_col].rolling(window, min_periods=1)
            else:
                roll = df[value_col].rolling(window, min_periods=1)

            features[f'rolling_mean_{window}'] = roll.mean().values
            features[f'rolling_std_{window}'] = roll.std().values

        # 差分
        if group_col:
            features['diff_1'] = df.groupby(group_col)[value_col].diff()
            features['pct_change'] = df.groupby(group_col)[value_col].pct_change()
        else:
            features['diff_1'] = df[value_col].diff()
            features['pct_change'] = df[value_col].pct_change()

        # 趋势特征 (最近N天的线性回归斜率)
        if group_col:
            for n_days in [7, 30]:
                features[f'trend_{n_days}'] = df.groupby(group_col)[value_col].transform(
                    lambda x: x.rolling(n_days, min_periods=3).apply(
                        lambda v: np.polyfit(np.arange(len(v)), v, 1)[0] if len(v) >= 3 else 0
                    )
                )

        n_features = features.shape[1]
        print(f"  提取时序特征: {n_features} 个")
        return features


# 生成时序数据
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=180, freq='D')
ts_df = pd.DataFrame({
    'date': np.tile(dates, 3),
    'store_id': np.repeat(['S001', 'S002', 'S003'], 180),
    'sales': np.concatenate([
        100 + np.cumsum(np.random.randn(180) * 5) + np.sin(np.arange(180) * 2 * np.pi / 7) * 20,
        80 + np.cumsum(np.random.randn(180) * 3) + np.sin(np.arange(180) * 2 * np.pi / 7) * 15,
        120 + np.cumsum(np.random.randn(180) * 4) + np.sin(np.arange(180) * 2 * np.pi / 7) * 25,
    ]),
})

ts_extractor = TimeSeriesFeatureExtractor()
ts_features = ts_extractor.extract(ts_df, 'date', 'sales', 'store_id')
ts_features = ts_features.fillna(0)
print(f"  时序特征矩阵: {ts_features.shape}")
print(ts_features.head(3))

# ============================================================
# 4. 文本特征提取
# ============================================================
print("\n--- 4. 文本特征提取 ---")

# 示例文本数据
texts = pd.Series([
    "这个产品质量很好，价格实惠，物流也很快",
    "太差了，完全不值这个价格，退货了",
    "一般般吧，中规中矩，没有特别惊喜",
    "超级喜欢！已经是第三次购买了，非常推荐",
    "客服态度很差，包装破损，不会再来",
    "性价比很高，推荐给朋友们了",
    "失望，和图片差距很大，色差严重",
    "质量可以，就是物流有点慢",
    "五星好评！完美的购物体验",
    "不推荐，质量太差了，浪费钱",
])

# 基础统计特征
text_features = pd.DataFrame()
text_features['char_count'] = texts.str.len()
text_features['word_count'] = texts.str.split().str.len()
text_features['exclamation_count'] = texts.str.count('！') + texts.str.count('!')
text_features['has_positive'] = texts.str.contains('好|喜欢|推荐|完美|满意').astype(int)
text_features['has_negative'] = texts.str.contains('差|失望|退货|不推荐|浪费').astype(int)

print("\n  文本统计特征:")
print(text_features.to_string())

# TF-IDF
if HAS_TFIDF:
    tfidf = TfidfVectorizer(max_features=20)
    tfidf_matrix = tfidf.fit_transform(texts)
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(),
                             columns=[f'tfidf_{w}' for w in tfidf.get_feature_names_out()])
    print(f"\n  TF-IDF特征: {tfidf_df.shape[1]} 个")
    print(f"  Top关键词: {list(tfidf.get_feature_names_out()[:10])}")

# ============================================================
# 5. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 左上: 特征重要性Top15
ax = axes[0, 0]
top15 = importances.head(15).sort_values()
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top15)))
ax.barh(range(len(top15)), top15.values, color=colors)
ax.set_yticks(range(len(top15)))
ax.set_yticklabels(top15.index, fontsize=8)
ax.set_xlabel('重要性')
ax.set_title('自动生成特征重要性 Top15')

# 右上: 原始 vs 自动特征性能对比
ax = axes[0, 1]
categories = ['原始特征', '自动特征']
means = [scores_orig.mean(), scores_auto.mean()]
stds = [scores_orig.std(), scores_auto.std()]
bars = ax.bar(categories, means, yerr=stds, capsize=5, color=['#2196F3', '#4CAF50'])
ax.set_ylabel('5折交叉验证准确率')
ax.set_title('特征工程效果对比')
ax.set_ylim(0.8, 1.0)
for bar, mean in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
            f'{mean:.4f}', ha='center', fontweight='bold')

# 左下: 时序数据可视化
ax = axes[1, 0]
for store in ['S001', 'S002', 'S003']:
    mask = ts_df['store_id'] == store
    ax.plot(ts_df.loc[mask, 'date'], ts_df.loc[mask, 'sales'], label=store, alpha=0.8)
ax.set_xlabel('日期')
ax.set_ylabel('销售额')
ax.set_title('多店铺销售时序数据')
ax.legend()
ax.grid(True, alpha=0.3)

# 右下: 特征生成统计
ax = axes[1, 1]
gen_types = list(afe.generated_features.keys())
gen_counts = [len(v) for v in afe.generated_features.values()]
colors_gen = ['#2196F3', '#4CAF50', '#FF9800']
bars = ax.bar(gen_types, gen_counts, color=colors_gen[:len(gen_types)])
ax.set_ylabel('生成特征数')
ax.set_title('各类型特征生成数量')
for bar, count in zip(bars, gen_counts):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
            str(count), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W25/d5_feature_engineering.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存: d5_feature_engineering.png")

print("\n完成! 自动特征工程要点:")
print("  1. 数值变换: log, sqrt, 平方, 标准化, 分箱")
print("  2. 特征交叉: 乘法, 加法, 比率, 差值")
print("  3. 时序特征: 时间成分, 滞后, 滚动统计, 趋势")
print("  4. 文本特征: 统计特征, TF-IDF, 情感特征")
print("  5. 特征选择: 基于重要性筛选最有价值的特征")
