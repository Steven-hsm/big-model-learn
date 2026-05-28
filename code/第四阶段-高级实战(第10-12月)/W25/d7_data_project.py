"""
W25-D7 数据工程项目
====================
构建完整的ML数据Pipeline, 从数据采集到特征服务的端到端流程
"""

import os
import json
import hashlib
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W25-D7 数据工程项目: 端到端ML数据Pipeline")
print("=" * 60)

# ============================================================
# 项目概述: 电商用户购买预测
# ============================================================
print("""
项目: 电商用户购买预测系统

  Pipeline流程:
  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
  │ 数据采集 │──>│ 数据清洗 │──>│ 特征工程 │──>│ 质量验证 │──>│ 特征服务 │
  └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
       │              │              │              │              │
    原始数据       清洁数据       特征表        验证报告       在线/离线
""")

# ============================================================
# Step 1: 数据采集 (模拟)
# ============================================================
print("\n=== Step 1: 数据采集 ===")


class DataCollector:
    """数据采集模块"""

    def collect_user_data(self, n=2000):
        """采集用户基础数据"""
        np.random.seed(42)
        users = pd.DataFrame({
            'user_id': [f'U{i:05d}' for i in range(n)],
            'age': np.random.randint(18, 65, n),
            'gender': np.random.choice(['M', 'F'], n),
            'city': np.random.choice(['北京', '上海', '广州', '深圳', '杭州', '成都', '武汉'], n),
            'register_date': [datetime(2023, 1, 1) + timedelta(days=random.randint(0, 365)) for _ in range(n)],
            'vip_level': np.random.choice([0, 1, 2], n, p=[0.6, 0.3, 0.1]),
        })
        print(f"  采集用户数据: {users.shape}")
        return users

    def collect_behavior_data(self, user_ids, days=30):
        """采集用户行为数据"""
        np.random.seed(43)
        records = []
        for uid in user_ids:
            n_events = np.random.poisson(15)
            for _ in range(n_events):
                records.append({
                    'user_id': uid,
                    'event_type': np.random.choice(['view', 'click', 'cart', 'purchase'], p=[0.5, 0.25, 0.15, 0.1]),
                    'item_id': f'I{np.random.randint(1, 500):04d}',
                    'timestamp': datetime.now() - timedelta(days=np.random.randint(0, days)),
                    'page_duration': np.random.exponential(30),
                })
        behavior = pd.DataFrame(records)
        print(f"  采集行为数据: {behavior.shape}")
        return behavior

    def collect_order_data(self, user_ids):
        """采集订单数据"""
        np.random.seed(44)
        records = []
        for uid in user_ids:
            n_orders = np.random.poisson(3)
            for _ in range(n_orders):
                records.append({
                    'order_id': f'O{len(records):06d}',
                    'user_id': uid,
                    'amount': np.random.exponential(300),
                    'items_count': np.random.poisson(2),
                    'category': np.random.choice(['电子产品', '服装', '食品', '家居', '美妆']),
                    'order_date': datetime.now() - timedelta(days=np.random.randint(1, 60)),
                    'status': np.random.choice(['completed', 'cancelled', 'returned'], p=[0.7, 0.2, 0.1]),
                })
        orders = pd.DataFrame(records)
        print(f"  采集订单数据: {orders.shape}")
        return orders


collector = DataCollector()
users_df = collector.collect_user_data(2000)
behavior_df = collector.collect_behavior_data(users_df['user_id'].tolist(), days=30)
orders_df = collector.collect_order_data(users_df['user_id'].tolist())

# ============================================================
# Step 2: 数据清洗
# ============================================================
print("\n=== Step 2: 数据清洗 ===")


class DataCleaner:
    """数据清洗模块"""

    def clean_users(self, df):
        """清洗用户数据"""
        original = len(df)
        # 去重
        df = df.drop_duplicates(subset='user_id')
        # 修复异常值
        df = df[(df['age'] >= 18) & (df['age'] <= 65)]
        print(f"  用户清洗: {original} -> {len(df)} (去除 {original - len(df)} 条)")
        return df.reset_index(drop=True)

    def clean_behavior(self, df):
        """清洗行为数据"""
        original = len(df)
        # 去除异常时长
        df = df[df['page_duration'] > 0]
        df = df[df['page_duration'] < 3600]  # 超过1小时的可能是异常
        # 去重
        df = df.drop_duplicates()
        print(f"  行为清洗: {original} -> {len(df)} (去除 {original - len(df)} 条)")
        return df.reset_index(drop=True)

    def clean_orders(self, df):
        """清洗订单数据"""
        original = len(df)
        # 只保留已完成订单
        df = df[df['status'] == 'completed']
        # 去除异常金额
        df = df[df['amount'] > 0]
        df = df[df['amount'] < 10000]
        print(f"  订单清洗: {original} -> {len(df)} (去除 {original - len(df)} 条)")
        return df.reset_index(drop=True)


cleaner = DataCleaner()
users_clean = cleaner.clean_users(users_df)
behavior_clean = cleaner.clean_behavior(behavior_df)
orders_clean = cleaner.clean_orders(orders_df)

# ============================================================
# Step 3: 特征工程
# ============================================================
print("\n=== Step 3: 特征工程 ===")


class FeatureEngine:
    """特征计算引擎"""

    def compute_user_features(self, users, behavior, orders):
        """计算用户特征"""
        features = users[['user_id', 'age', 'gender', 'vip_level']].copy()

        # 行为特征
        behavior_agg = behavior.groupby('user_id').agg(
            total_views=('event_type', lambda x: (x == 'view').sum()),
            total_clicks=('event_type', lambda x: (x == 'click').sum()),
            total_cart=('event_type', lambda x: (x == 'cart').sum()),
            total_purchases=('event_type', lambda x: (x == 'purchase').sum()),
            avg_page_duration=('page_duration', 'mean'),
            total_page_duration=('page_duration', 'sum'),
            active_days=('timestamp', lambda x: x.dt.date.nunique()),
        ).reset_index()

        # 点击率
        behavior_agg['click_rate'] = behavior_agg['total_clicks'] / (behavior_agg['total_views'] + 1)
        # 加购率
        behavior_agg['cart_rate'] = behavior_agg['total_cart'] / (behavior_agg['total_views'] + 1)
        # 购买率
        behavior_agg['purchase_rate'] = behavior_agg['total_purchases'] / (behavior_agg['total_views'] + 1)

        features = features.merge(behavior_agg, on='user_id', how='left')

        # 订单特征
        order_agg = orders.groupby('user_id').agg(
            total_orders=('order_id', 'count'),
            total_spent=('amount', 'sum'),
            avg_order_amount=('amount', 'mean'),
            max_order_amount=('amount', 'max'),
            unique_categories=('category', 'nunique'),
            recent_order_days=('order_date', lambda x: (datetime.now() - x.max()).days),
        ).reset_index()

        # 平均订单间隔
        order_agg['avg_order_interval'] = orders.sort_values(['user_id', 'order_date']).groupby('user_id')[
            'order_date'].diff().dt.days.groupby(orders.sort_values(['user_id', 'order_date'])['user_id']).mean()
        order_agg['avg_order_interval'] = order_agg['avg_order_interval'].fillna(30)

        features = features.merge(order_agg, on='user_id', how='left')

        # 填充缺失(无行为/订单的用户)
        fill_cols = [c for c in features.columns if c not in ['user_id', 'gender']]
        for col in fill_cols:
            features[col] = features[col].fillna(0)

        # 衍生特征
        features['age_group'] = pd.cut(features['age'], bins=[0, 25, 35, 45, 55, 100],
                                        labels=[1, 2, 3, 4, 5]).astype(float)
        features['is_vip'] = (features['vip_level'] > 0).astype(int)
        features['high_spender'] = (features['total_spent'] > features['total_spent'].quantile(0.75)).astype(int)
        features['engagement_score'] = (
            features['active_days'] * 0.3
            + features['total_views'] * 0.1
            + features['total_clicks'] * 0.2
            + features['total_cart'] * 0.4
        )

        print(f"  特征计算完成: {features.shape[1]} 个特征, {features.shape[0]} 条记录")
        return features


feature_engine = FeatureEngine()
features_df = feature_engine.compute_user_features(users_clean, behavior_clean, orders_clean)
print(f"\n  特征概览:")
print(features_df.describe().round(2).to_string())

# ============================================================
# Step 4: 质量验证
# ============================================================
print("\n=== Step 4: 质量验证 ===")


class QualityGate:
    """质量门控"""

    def __init__(self):
        self.checks = []

    def check_completeness(self, df, threshold=0.95):
        """完整性检查"""
        completeness = df.notna().mean()
        overall = completeness.mean()
        passed = overall >= threshold
        self.checks.append(('完整性', passed, f'整体完整率={overall:.2%}'))
        return passed

    def check_freshness(self, df, date_col, max_days=7):
        """新鲜度检查"""
        if date_col in df.columns:
            latest = pd.to_datetime(df[date_col]).max()
            days_old = (datetime.now() - latest).days
            passed = days_old <= max_days
            self.checks.append(('新鲜度', passed, f'最新数据={days_old}天前'))
            return passed
        self.checks.append(('新鲜度', True, '无日期列, 跳过'))
        return True

    def check_distribution(self, df, col, reference_mean, reference_std, tolerance=0.3):
        """分布检查"""
        current_mean = df[col].mean()
        current_std = df[col].std()
        mean_shift = abs(current_mean - reference_mean) / (reference_std + 1e-10)
        passed = mean_shift < tolerance
        self.checks.append((
            f'分布({col})', passed,
            f'均值偏移={mean_shift:.3f} (阈值={tolerance})'
        ))
        return passed

    def check_cardinality(self, df, col, min_unique=2):
        """基数检查"""
        n_unique = df[col].nunique()
        passed = n_unique >= min_unique
        self.checks.append((f'基数({col})', passed, f'唯一值={n_unique}'))
        return passed

    def report(self):
        """输出���告"""
        all_passed = all(c[1] for c in self.checks)
        print(f"\n  质量门控报告 ({'全部通过' if all_passed else '存在失败'}):")
        print(f"  {'检查项':<25} {'状态':<6} {'详情'}")
        print("  " + "-" * 60)
        for name, passed, detail in self.checks:
            status = 'PASS' if passed else 'FAIL'
            print(f"  {name:<25} {status:<6} {detail}")
        return all_passed


gate = QualityGate()
gate.check_completeness(features_df, 0.90)
gate.check_cardinality(features_df, 'user_id', min_unique=100)
gate.check_distribution(features_df, 'age', reference_mean=41, reference_std=14)
gate.check_distribution(features_df, 'total_views', reference_mean=7, reference_std=4, tolerance=0.5)
all_passed = gate.report()

# ============================================================
# Step 5: 构建训练数据
# ============================================================
print("\n=== Step 5: 构建训练数据 ===")

# 目标变量: 用户是否有高购买意向
features_df['label'] = (
    (features_df['total_purchases'] >= 1) &
    (features_df['total_spent'] > features_df['total_spent'].median())
).astype(int)

print(f"  正例比例: {features_df['label'].mean():.2%}")

# 选择特征列
feature_cols = [c for c in features_df.columns
                if c not in ['user_id', 'gender', 'label'] and features_df[c].dtype in ['int64', 'float64']]

X = features_df[feature_cols].fillna(0)
y = features_df['label']

print(f"  特征数: {len(feature_cols)}, 样本数: {len(X)}")
print(f"  特征列: {feature_cols[:10]}...")

# 训练测试分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ============================================================
# Step 6: 模型训练与评估
# ============================================================
print("\n=== Step 6: 模型训练与评估 ===")

model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"  模型准确率: {acc:.4f}")
print(classification_report(y_test, y_pred, target_names=['低意向', '高意向']))

# 特征重要性
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\n  Top 10 重要特征:")
for feat, imp in importances.head(10).items():
    print(f"    {feat:<25} {imp:.4f}")

# ============================================================
# Step 7: 特征服务 (模拟)
# ============================================================
print("\n=== Step 7: 特征服务 ===")


class FeatureService:
    """特征服务 (模拟在线推理)"""

    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.feature_store = {}  # user_id -> features
        self.request_log = []

    def load_features(self, user_id, features):
        """加载用户特征到在线存储"""
        self.feature_store[user_id] = features

    def predict(self, user_id):
        """在线预测"""
        t0 = time.time()
        features = self.feature_store.get(user_id)
        if features is None:
            return None, 0

        X = np.array([features[f] for f in self.feature_names]).reshape(1, -1)
        prob = self.model.predict_proba(X)[0][1]
        elapsed = (time.time() - t0) * 1000

        self.request_log.append({
            'user_id': user_id,
            'probability': prob,
            'latency_ms': elapsed,
        })
        return prob, elapsed

    def batch_load(self, feature_df):
        """批量加载"""
        for _, row in feature_df.iterrows():
            self.feature_store[row['user_id']] = row.to_dict()
        print(f"  加载 {len(self.feature_store)} 个用户特征到在线存储")

    def latency_stats(self):
        """延迟统计"""
        if not self.request_log:
            return {}
        latencies = [r['latency_ms'] for r in self.request_log]
        return {
            'count': len(latencies),
            'mean_ms': np.mean(latencies),
            'p50_ms': np.percentile(latencies, 50),
            'p95_ms': np.percentile(latencies, 95),
            'p99_ms': np.percentile(latencies, 99),
        }


# 启动特征服务
service = FeatureService(model, feature_cols)
service.batch_load(features_df)

# 模拟在线请求
print("\n  模拟在线预测:")
sample_users = features_df['user_id'].sample(5).tolist()
for uid in sample_users:
    prob, latency = service.predict(uid)
    label = '高意向' if prob > 0.5 else '低意向'
    print(f"    {uid}: 购买概率={prob:.3f} ({label}), 延迟={latency:.2f}ms")

# 压测
print("\n  模拟1000次请求...")
for _ in range(1000):
    uid = np.random.choice(features_df['user_id'].tolist())
    service.predict(uid)

stats = service.latency_stats()
print(f"  延迟统计: avg={stats['mean_ms']:.2f}ms, p50={stats['p50_ms']:.2f}ms, "
      f"p95={stats['p95_ms']:.2f}ms, p99={stats['p99_ms']:.2f}ms")

# ============================================================
# 可视化: 端到端Pipeline总结
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. 数据量变化
ax = axes[0, 0]
stages = ['采集', '清洗', '特征', '训练']
user_counts = [len(users_df), len(users_clean), len(features_df), len(X_train)]
bars = ax.bar(stages, user_counts, color=['#2196F3', '#4CAF50', '#FF9800', '#9C27B0'])
ax.set_ylabel('记录数')
ax.set_title('Pipeline各阶段数据量')
for bar, count in zip(bars, user_counts):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 20,
            str(count), ha='center', fontweight='bold')

# 2. 特征分布 - engagement_score
ax = axes[0, 1]
features_df['engagement_score'].hist(bins=30, ax=ax, color='#2196F3', alpha=0.7)
ax.axvline(features_df['engagement_score'].mean(), color='red', linestyle='--', label='均值')
ax.set_title('参与度得分分布')
ax.set_xlabel('得分')
ax.legend()

# 3. 目标变量分布
ax = axes[0, 2]
label_counts = features_df['label'].value_counts()
ax.pie(label_counts, labels=['低意向', '高意向'], autopct='%1.1f%%',
       colors=['#90CAF9', '#F44336'], startangle=90)
ax.set_title('目标变量分布')

# 4. 特征重要性Top10
ax = axes[1, 0]
top10 = importances.head(10).sort_values()
ax.barh(range(len(top10)), top10.values, color='#4CAF50')
ax.set_yticks(range(len(top10)))
ax.set_yticklabels(top10.index, fontsize=8)
ax.set_xlabel('重要性')
ax.set_title('特征重要性 Top10')

# 5. 预测概率分布
ax = axes[1, 1]
all_probs = model.predict_proba(X_test)[:, 1]
ax.hist(all_probs[y_test == 0], bins=30, alpha=0.5, label='低意向', color='#2196F3', density=True)
ax.hist(all_probs[y_test == 1], bins=30, alpha=0.5, label='高意向', color='#F44336', density=True)
ax.set_title('预测概率分布')
ax.set_xlabel('购买概率')
ax.legend()

# 6. 在线延迟分布
ax = axes[1, 2]
if service.request_log:
    latencies = [r['latency_ms'] for r in service.request_log[-1000:]]
    ax.hist(latencies, bins=50, color='#9C27B0', alpha=0.7)
    ax.axvline(np.percentile(latencies, 95), color='red', linestyle='--', label='P95')
    ax.set_title('在线推理延迟分布')
    ax.set_xlabel('延迟 (ms)')
    ax.legend()

plt.suptitle('端到端ML数据Pipeline总结', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W25/d7_data_project.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存: d7_data_project.png")

print("\n" + "=" * 60)
print("项目总结:")
print("  1. 数据采集: 多源数据整合(用户/行为/订单)")
print("  2. 数据清洗: 去重、异常值处理、格式统一")
print("  3. 特征工程: 行为统计、订单聚合、衍生特征")
print("  4. 质量验证: 完整性、分布、基数检查")
print("  5. 模型训练: 随机森林分类, 准确率={:.2%}".format(acc))
print("  6. 特征服务: 在线推理, P99延迟={:.2f}ms".format(stats.get('p99_ms', 0)))
print("=" * 60)
