"""
W25-D2 特征存储 (Feature Store)
================================
特征存储概念(在线/离线), 特征注册和发现, 特征血缘追踪, 简化版Feature Store实现
"""

import os
import json
import hashlib
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W25-D2 特征存储 (Feature Store)")
print("=" * 60)

# ============================================================
# 1. Feature Store 概念
# ============================================================
print("\n--- 1. Feature Store 概念 ---")
print("""
Feature Store 是ML系统中的特征管理平台:

  ┌─────────────────────────────────────────────┐
  │              Feature Store                   │
  │                                              │
  │  ┌──────────────┐  ┌──────────────────┐     │
  │  │ 离线存储     │  │ 在线存储         │     │
  │  │ (Offline)    │  │ (Online)         │     │
  │  │              │  │                   │     │
  │  │ - 训练数据   │  │ - 低延迟服务     │     │
  │  │ - 批量计算   │  │ - 实时特征       │     │
  │  │ - 历史快照   │  │ - KV存储         │     │
  │  │ - Parquet    │  │ - Redis/DynamoDB  │     │
  │  └──────────────┘  └──────────────────┘     │
  │                                              │
  │  特征注册表 | 血缘追踪 | 版本管理          │
  └─────────────────────────────────────────────┘

核心价值:
  - 特征共享: 避免重复开发, 团队复用
  - 一致性:   训练和推理使用同一套特征逻辑
  - 可追溯:   特征从哪来, 谁在用, 版本是什么
  - 实时性:   在线服务低延迟获取特征

常见工具: Feast, Tecton, Hopsworks, AWS SageMaker Feature Store
""")

# ============================================================
# 2. 简化版 Feature Store 实现
# ============================================================
print("\n--- 2. 简化版 Feature Store 实现 ---")


class FeatureDefinition:
    """特征定义"""

    def __init__(self, name, dtype, description, source='', tags=None):
        self.name = name
        self.dtype = dtype
        self.description = description
        self.source = source
        self.tags = tags or []
        self.created_at = datetime.now().isoformat()
        self.version = 1

    def to_dict(self):
        return {
            'name': self.name,
            'dtype': self.dtype,
            'description': self.description,
            'source': self.source,
            'tags': self.tags,
            'created_at': self.created_at,
            'version': self.version,
        }


class FeatureGroup:
    """特征组 - 一组相关特征的集合"""

    def __init__(self, name, description, entity_keys, features=None):
        self.name = name
        self.description = description
        self.entity_keys = entity_keys  # 实体键, 如 ['user_id']
        self.features = features or []
        self.created_at = datetime.now().isoformat()

    def add_feature(self, feature_def):
        self.features.append(feature_def)

    def to_dict(self):
        return {
            'name': self.name,
            'description': self.description,
            'entity_keys': self.entity_keys,
            'features': [f.to_dict() for f in self.features],
            'created_at': self.created_at,
        }


class FeatureStore:
    """简化版特征存储"""

    def __init__(self, name='default'):
        self.name = name
        self.feature_groups = {}  # name -> FeatureGroup
        self.offline_data = {}    # group_name -> DataFrame (离线存储)
        self.online_data = {}     # group_name -> dict of dict (在线存储, KV)
        self.lineage = {}         # feature -> [upstream_features]
        self.serving_stats = {'requests': 0, 'cache_hits': 0}

    def register_feature_group(self, feature_group):
        """注册特征组"""
        self.feature_groups[feature_group.name] = feature_group
        print(f"  注册特征组: {feature_group.name} ({len(feature_group.features)} 个特征)")

    def ingest_offline(self, group_name, df):
        """写入离线存储 (用于训练)"""
        self.offline_data[group_name] = df.copy()
        print(f"  离线存储 [{group_name}]: 写入 {len(df)} 条记录")

    def ingest_online(self, group_name, df, key_column):
        """写入在线存储 (用于实时推理)"""
        online = {}
        for _, row in df.iterrows():
            key = str(row[key_column])
            online[key] = row.to_dict()
        self.online_data[group_name] = online
        print(f"  在线存储 [{group_name}]: 写入 {len(online)} 条记录")

    def get_offline_features(self, group_name, feature_names=None):
        """从离线存储获取特征 (训练用)"""
        df = self.offline_data.get(group_name)
        if df is None:
            raise ValueError(f"特征组 {group_name} 不存在")
        if feature_names:
            entity_keys = self.feature_groups[group_name].entity_keys
            cols = entity_keys + [f for f in feature_names if f in df.columns]
            return df[cols]
        return df

    def get_online_features(self, group_name, entity_id, feature_names=None):
        """从在线存储获取特征 (实时推理用)"""
        self.serving_stats['requests'] += 1
        group_data = self.online_data.get(group_name, {})
        entity_data = group_data.get(str(entity_id))
        if entity_data is None:
            return None
        if feature_names:
            return {f: entity_data.get(f) for f in feature_names}
        return entity_data

    def add_lineage(self, target, sources):
        """记录特征血缘"""
        self.lineage[target] = sources

    def search_features(self, keyword=''):
        """搜索特征"""
        results = []
        for gname, group in self.feature_groups.items():
            for f in group.features:
                if (keyword.lower() in f.name.lower()
                        or keyword.lower() in f.description.lower()):
                    results.append({
                        'group': gname,
                        'feature': f.name,
                        'dtype': f.dtype,
                        'description': f.description,
                    })
        return results

    def print_registry(self):
        """打印特征注册表"""
        print(f"\n  特征注册表 (共 {len(self.feature_groups)} 个特征组):")
        print(f"  {'特征组':<20} {'实体键':<15} {'特征数':<8} {'描述'}")
        print("  " + "-" * 70)
        for gname, group in self.feature_groups.items():
            print(f"  {gname:<20} {str(group.entity_keys):<15} {len(group.features):<8} {group.description}")


# ============================================================
# 3. 使用Feature Store
# ============================================================
print("\n--- 3. 使用Feature Store ---")

# 创建Feature Store
fs = FeatureStore('user_recommendation')

# --- 定义特征组1: 用户画像特征 ---
user_profile = FeatureGroup('user_profile', '用户基础画像特征', ['user_id'])
user_profile.add_feature(FeatureDefinition('age', 'int', '用户年龄', 'raw_user_table', ['demographic']))
user_profile.add_feature(FeatureDefinition('gender', 'str', '性别', 'raw_user_table', ['demographic']))
user_profile.add_feature(FeatureDefinition('city_tier', 'int', '城市等级', 'city_mapping', ['geographic']))
user_profile.add_feature(FeatureDefinition('income_level', 'int', '收入等级(1-5)', 'computed', ['financial']))

fs.register_feature_group(user_profile)

# --- 定义特征组2: 用户行为特征 ---
user_behavior = FeatureGroup('user_behavior', '用户行为统计特征', ['user_id'])
user_behavior.add_feature(FeatureDefinition('login_count_7d', 'int', '近7天登录次数', 'event_log', ['engagement']))
user_behavior.add_feature(FeatureDefinition('purchase_count_30d', 'int', '近30天购买次数', 'order_table', ['purchase']))
user_behavior.add_feature(FeatureDefinition('avg_order_value', 'float', '平均订单金额', 'computed', ['purchase', 'financial']))
user_behavior.add_feature(FeatureDefinition('click_rate', 'float', '点击率', 'computed', ['engagement']))
user_behavior.add_feature(FeatureDefinition('favorite_count', 'int', '收藏商品数', 'interaction_log', ['engagement']))

fs.register_feature_group(user_behavior)

# --- 定义特征组3: 商品特征 ---
item_features = FeatureGroup('item_features', '商品画像特征', ['item_id'])
item_features.add_feature(FeatureDefinition('price', 'float', '商品价格', 'item_table', ['price']))
item_features.add_feature(FeatureDefinition('category', 'str', '商品类目', 'item_table', ['category']))
item_features.add_feature(FeatureDefinition('sales_30d', 'int', '30天销量', 'computed', ['sales']))
item_features.add_feature(FeatureDefinition('rating', 'float', '平均评分', 'review_table', ['quality']))

fs.register_feature_group(item_features)

# 打印注册表
fs.print_registry()

# --- 记录特征血缘 ---
fs.add_lineage('avg_order_value', ['purchase_count_30d', 'total_spend_30d'])
fs.add_lineage('income_level', ['income', 'city_tier'])
fs.add_lineage('click_rate', ['click_count', 'impression_count'])

# ============================================================
# 4. 模拟数据写入与读取
# ============================================================
print("\n--- 4. 模拟数据写入与读取 ---")

np.random.seed(42)
n_users = 500

# 用户画像数据
profile_df = pd.DataFrame({
    'user_id': [f'U{i:04d}' for i in range(n_users)],
    'age': np.random.randint(18, 65, n_users),
    'gender': np.random.choice(['M', 'F'], n_users),
    'city_tier': np.random.randint(1, 4, n_users),
    'income_level': np.random.randint(1, 6, n_users),
})

# 行为数据
behavior_df = pd.DataFrame({
    'user_id': [f'U{i:04d}' for i in range(n_users)],
    'login_count_7d': np.random.poisson(5, n_users),
    'purchase_count_30d': np.random.poisson(3, n_users),
    'avg_order_value': np.random.normal(200, 80, n_users).clip(10, 1000),
    'click_rate': np.random.beta(2, 5, n_users),
    'favorite_count': np.random.poisson(8, n_users),
})

# 写入离线存储
fs.ingest_offline('user_profile', profile_df)
fs.ingest_offline('user_behavior', behavior_df)

# 写入在线存储 (取前100条模拟)
fs.ingest_online('user_profile', profile_df.head(100), 'user_id')
fs.ingest_online('user_behavior', behavior_df.head(100), 'user_id')

# --- 离线读取 (训练场景) ---
print("\n  离线特征获取 (训练):")
train_features = fs.get_offline_features('user_behavior',
                                          ['purchase_count_30d', 'avg_order_value', 'click_rate'])
print(f"  获取到 {len(train_features)} 条训练数据")
print(train_features.head(3))

# --- 在线读取 (实时推理场景) ---
print("\n  在线特征获取 (实时推理):")
t0 = time.time()
for _ in range(100):
    uid = f'U{np.random.randint(0, 100):04d}'
    features = fs.get_online_features('user_profile', uid, ['age', 'city_tier', 'income_level'])
elapsed = time.time() - t0
print(f"  100次在线特征查询耗时: {elapsed*1000:.1f}ms (avg: {elapsed*10:.2f}ms/次)")
print(f"  示例 - U0042 特征: {fs.get_online_features('user_profile', 'U0042')}")

# ============================================================
# 5. 特征搜索与发现
# ============================================================
print("\n--- 5. 特征搜索与发现 ---")

print("\n  搜索 'purchase' 相关特征:")
results = fs.search_features('purchase')
for r in results:
    print(f"    [{r['group']}] {r['feature']} ({r['dtype']}): {r['description']}")

print("\n  搜索 'engagement' 标签:")
for gname, group in fs.feature_groups.items():
    for f in group.features:
        if 'engagement' in f.tags:
            print(f"    [{gname}] {f.name}: {f.description}")

# ============================================================
# 6. 特征血缘追踪
# ============================================================
print("\n--- 6. 特征血缘追踪 ---")

print("\n  特征血缘关系:")
for target, sources in fs.lineage.items():
    print(f"    {target} <- {', '.join(sources)}")

print("""
  血缘追踪的用途:
    - 影响分析: 上游数据变化影响哪些模型?
    - 根因分析: 模型异常是哪个特征导致的?
    - 合规审计: 特征是否符合隐私法规?
    - 重算优化: 上游变化后哪些特征需要重算?
""")

# ============================================================
# 7. 可视化
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图: 特征组分布
groups = list(fs.feature_groups.keys())
counts = [len(g.features) for g in fs.feature_groups.values()]
colors = ['#2196F3', '#4CAF50', '#FF9800']
bars = axes[0].bar(groups, counts, color=colors[:len(groups)])
axes[0].set_ylabel('特征数量')
axes[0].set_title('各特征组特征数量')
for bar, count in zip(bars, counts):
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                 str(count), ha='center', fontweight='bold')

# 右图: 特征血缘图 (简化版)
axes[1].set_xlim(0, 10)
axes[1].set_ylim(0, 10)
axes[1].set_title('特征血缘关系')
axes[1].axis('off')

# 绘制节点
raw_nodes = {'income': (1, 8), 'city_tier': (1, 6), 'click_count': (1, 4),
             'impression_count': (1, 2), 'purchase_count_30d': (1, 0.5),
             'total_spend_30d': (1, -0.5)}
derived_nodes = {'income_level': (5, 7), 'click_rate': (5, 3), 'avg_order_value': (5, 0.5)}

for name, (x, y) in {**raw_nodes, **derived_nodes}.items():
    color = '#BBDEFB' if name in raw_nodes else '#C8E6C9'
    axes[1].add_patch(plt.Rectangle((x - 0.9, y - 0.3), 1.8, 0.6,
                                     facecolor=color, edgecolor='#333', linewidth=1.5))
    axes[1].text(x, y, name, ha='center', va='center', fontsize=8)

# 绘制连线
lineage_pairs = [
    (('income', 1, 8), ('income_level', 5, 7)),
    (('city_tier', 1, 6), ('income_level', 5, 7)),
    (('click_count', 1, 4), ('click_rate', 5, 3)),
    (('impression_count', 1, 2), ('click_rate', 5, 3)),
    (('purchase_count_30d', 1, 0.5), ('avg_order_value', 5, 0.5)),
    (('total_spend_30d', 1, -0.5), ('avg_order_value', 5, 0.5)),
]
for src, dst in lineage_pairs:
    axes[1].annotate('', xy=(dst[1] - 0.9, dst[2]),
                     xytext=(src[1] + 0.9, src[2]),
                     arrowprops=dict(arrowstyle='->', color='#666', lw=1.5))

axes[1].text(1, 9, '原始特征', ha='center', fontsize=10, fontweight='bold', color='#1565C0')
axes[1].text(5, 9, '派生特征', ha='center', fontsize=10, fontweight='bold', color='#2E7D32')

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W25/d2_feature_store.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存: d2_feature_store.png")

print("\n完成! Feature Store要点:")
print("  1. 离线存储: 批量计算, 历史数据, 训练场景")
print("  2. 在线存储: 低延迟KV查询, 实时推理场景")
print("  3. 特征注册: 统一管理, 搜索发现, 版本控制")
print("  4. 血缘追踪: 理解特征依赖, 影响分析, 根因分析")
