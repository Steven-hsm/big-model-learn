"""
W25-D6 数据目录管理
====================
数据目录管理, 元数据管理, 数据字典生成, 数据血缘可视化
"""

import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W25-D6 数据目录管理")
print("=" * 60)

# ============================================================
# 1. 数据目录概念
# ============================================================
print("\n--- 1. 数据目录概念 ---")
print("""
数据目录 (Data Catalog) 是组织数据资产的核心工具:

  核心功能:
    - 数据发现:   搜索和浏览可用数据集
    - 元数据管理: 描述数据的"数据"(schema, owner, freshness ...)
    - 数据字典:   每个字段的定义、类型、约束
    - 数据血缘:   数据从哪来, 到哪去, 经过了什么变换
    - 数据质量:   质量评分, 质量规则, SLA
    - 访问控制:   数据权限, 敏感等级, 合规标记

  常见工具:
    - Apache Atlas:   Hadoop生态数据治理
    - DataHub:        LinkedIn开源, 现代数据目录
    - Amundsen:       Lyft开源, 数据发现
    - OpenMetadata:   统一元数据平台
    - CKAN:           开放数据门户
""")

# ============================================================
# 2. 数据目录实现
# ============================================================
print("\n--- 2. 数据目录实现 ---")


class DataCatalog:
    """数据目录管理器"""

    def __init__(self):
        self.datasets = {}  # name -> metadata
        self.lineage = {}   # target -> [sources]

    def register_dataset(self, name, description='', owner='', tier='',
                         sensitivity='internal', tags=None, schema=None):
        """注册数据集"""
        self.datasets[name] = {
            'name': name,
            'description': description,
            'owner': owner,
            'tier': tier,
            'sensitivity': sensitivity,
            'tags': tags or [],
            'schema': schema or [],
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'quality_score': None,
            'row_count': None,
            'size_mb': None,
        }
        print(f"  注册数据集: {name}")

    def update_stats(self, name, row_count=None, size_mb=None, quality_score=None):
        """更新数据集统计信息"""
        if name in self.datasets:
            if row_count is not None:
                self.datasets[name]['row_count'] = row_count
            if size_mb is not None:
                self.datasets[name]['size_mb'] = size_mb
            if quality_score is not None:
                self.datasets[name]['quality_score'] = quality_score
            self.datasets[name]['updated_at'] = datetime.now().isoformat()

    def add_schema(self, dataset_name, columns):
        """添加schema信息"""
        if dataset_name in self.datasets:
            self.datasets[dataset_name]['schema'] = columns

    def add_lineage(self, target, sources, transformation=''):
        """添加血缘关系"""
        self.lineage[target] = {
            'sources': sources,
            'transformation': transformation,
        }

    def search(self, keyword=''):
        """搜索数据集"""
        results = []
        keyword = keyword.lower()
        for name, meta in self.datasets.items():
            if (keyword in name.lower()
                    or keyword in meta['description'].lower()
                    or keyword in ' '.join(meta['tags'])):
                results.append(meta)
        return results

    def get_dataset(self, name):
        return self.datasets.get(name)

    def list_datasets(self):
        """列出所有数据集"""
        print(f"\n  {'名称':<25} {'所有者':<10} {'敏感度':<10} {'标签':<20} {'行数'}")
        print("  " + "-" * 80)
        for name, meta in self.datasets.items():
            tags = ', '.join(meta['tags'][:3])
            rows = meta.get('row_count', 'N/A')
            print(f"  {name:<25} {meta['owner']:<10} {meta['sensitivity']:<10} {tags:<20} {rows}")


class DataDictionaryGenerator:
    """数据字典生成器"""

    @staticmethod
    def generate(df, dataset_name, descriptions=None):
        """从DataFrame生成数据字典"""
        dictionary = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            null_count = df[col].isna().sum()
            null_pct = null_count / len(df) * 100

            entry = {
                'column': col,
                'dtype': dtype,
                'description': descriptions.get(col, '') if descriptions else '',
                'null_count': null_count,
                'null_pct': round(null_pct, 2),
                'unique_count': df[col].nunique(),
                'sample_values': df[col].dropna().head(3).tolist(),
            }

            if df[col].dtype in ['int64', 'float64']:
                entry.update({
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'mean': round(df[col].mean(), 2),
                    'std': round(df[col].std(), 2),
                })

            dictionary.append(entry)

        return dictionary

    @staticmethod
    def print_dictionary(dictionary, dataset_name):
        """打印数据字典"""
        print(f"\n  数据字典: {dataset_name}")
        print(f"  {'列名':<20} {'类型':<10} {'空值率':<8} {'唯一值':<8} {'描述'}")
        print("  " + "-" * 70)
        for entry in dictionary:
            desc = entry.get('description', '')[:25]
            print(f"  {entry['column']:<20} {entry['dtype']:<10} "
                  f"{entry['null_pct']:<8.1f} {entry['unique_count']:<8} {desc}")

            if 'min' in entry:
                print(f"  {'':20} 范围: [{entry['min']:.2f}, {entry['max']:.2f}] "
                      f"均值: {entry['mean']} 标准差: {entry['std']}")

    @staticmethod
    def to_markdown(dictionary, dataset_name):
        """生成Markdown格式数据字典"""
        lines = [f"# 数据字典: {dataset_name}\n"]
        lines.append("| 列名 | 类型 | 空值率 | 唯一值 | 最小值 | 最大值 | 均值 | 描述 |")
        lines.append("|------|------|--------|--------|--------|--------|------|------|")

        for entry in dictionary:
            min_v = f"{entry['min']:.2f}" if 'min' in entry else '-'
            max_v = f"{entry['max']:.2f}" if 'max' in entry else '-'
            mean_v = f"{entry['mean']}" if 'mean' in entry else '-'
            desc = entry.get('description', '')
            lines.append(
                f"| {entry['column']} | {entry['dtype']} | {entry['null_pct']:.1f}% "
                f"| {entry['unique_count']} | {min_v} | {max_v} | {mean_v} | {desc} |"
            )

        return '\n'.join(lines)


# ============================================================
# 3. 构建数据目录
# ============================================================
print("\n--- 3. 构建数据目录 ---")

catalog = DataCatalog()

# 注册数据集
catalog.register_dataset('raw_users', '原始用户数据', '数据团队', 'tier-1',
                          'confidential', ['用户', '画像', 'PII'])
catalog.register_dataset('raw_orders', '原始订单数据', '业务团队', 'tier-1',
                          'confidential', ['订单', '交易'])
catalog.register_dataset('raw_products', '商品基础信息', '商品团队', 'tier-1',
                          'internal', ['商品', '类目'])
catalog.register_dataset('clean_users', '清洗后用户数据', '数据工程', 'tier-2',
                          'confidential', ['用户', '清洗', '特征'])
catalog.register_dataset('user_features', '用户特征表', 'ML团队', 'tier-2',
                          'internal', ['特征', 'ML', '用户'])
catalog.register_dataset('order_features', '订单特征表', 'ML团队', 'tier-2',
                          'internal', ['特征', 'ML', '订单'])
catalog.register_dataset('training_dataset', '训练数据集', 'ML团队', 'tier-3',
                          'internal', ['训练', 'ML'])
catalog.register_dataset('model_predictions', '模型预测结果', 'ML团队', 'tier-3',
                          'internal', ['预测', '推理'])

# 更新统计信息
catalog.update_stats('raw_users', row_count=50000, size_mb=25, quality_score=0.92)
catalog.update_stats('raw_orders', row_count=200000, size_mb=120, quality_score=0.88)
catalog.update_stats('raw_products', row_count=10000, size_mb=5, quality_score=0.95)
catalog.update_stats('user_features', row_count=50000, size_mb=45, quality_score=0.90)
catalog.update_stats('training_dataset', row_count=40000, size_mb=60, quality_score=0.85)

# 添加血缘关系
catalog.add_lineage('clean_users', ['raw_users'], '去重+缺失值填充+类型转换')
catalog.add_lineage('user_features', ['clean_users', 'raw_orders'], '聚合+特征计算')
catalog.add_lineage('order_features', ['raw_orders', 'raw_products'], '关联+聚合统计')
catalog.add_lineage('training_dataset', ['user_features', 'order_features'], 'Join+采样+标签')
catalog.add_lineage('model_predictions', ['training_dataset'], '模型推理')

# 列出所有数据集
catalog.list_datasets()

# 搜索
print("\n  搜索 'ML' 相关数据集:")
for r in catalog.search('ML'):
    print(f"    - {r['name']}: {r['description']}")

# ============================================================
# 4. 生成数据字典
# ============================================================
print("\n--- 4. 生成数据字典 ---")

# 创建模拟数据
np.random.seed(42)
user_df = pd.DataFrame({
    'user_id': [f'U{i:05d}' for i in range(1000)],
    'age': np.random.randint(18, 65, 1000),
    'gender': np.random.choice(['M', 'F'], 1000),
    'city': np.random.choice(['北京', '上海', '广州', '深圳', '杭州'], 1000),
    'income': np.random.normal(50000, 15000, 1000),
    'register_date': pd.date_range('2023-01-01', periods=1000, freq='6H'),
    'vip_level': np.random.choice([0, 1, 2, 3], 1000, p=[0.5, 0.3, 0.15, 0.05]),
})

# 注入一些空值
user_df.loc[10:20, 'income'] = np.nan
user_df.loc[30:35, 'city'] = None

# 列描述
descriptions = {
    'user_id': '用户唯一标识',
    'age': '用户年龄(18-65)',
    'gender': '性别(M=男, F=女)',
    'city': '所在城市',
    'income': '年收入(元)',
    'register_date': '注册时间',
    'vip_level': 'VIP等级(0=普通, 1-3=VIP)',
}

dd_gen = DataDictionaryGenerator()
dictionary = dd_gen.generate(user_df, 'raw_users', descriptions)
dd_gen.print_dictionary(dictionary, 'raw_users')

# 生成Markdown
md_output = dd_gen.to_markdown(dictionary, 'raw_users')
print(f"\n  Markdown数据字典 (前5行):")
for line in md_output.split('\n')[:8]:
    print(f"  {line}")

# ============================================================
# 5. 数据血缘可视化
# ============================================================
print("\n--- 5. 数据血缘可视化 ---")


def draw_lineage(catalog, figsize=(14, 7)):
    """绘制数据血缘图"""
    fig, ax = plt.subplots(figsize=figsize)

    # 层级布局
    tiers = {
        'raw': ['raw_users', 'raw_orders', 'raw_products'],
        'clean': ['clean_users'],
        'features': ['user_features', 'order_features'],
        'ml': ['training_dataset'],
        'serving': ['model_predictions'],
    }

    tier_x = {'raw': 0, 'clean': 3, 'features': 6, 'ml': 9, 'serving': 12}
    tier_colors = {
        'raw': '#BBDEFB', 'clean': '#C8E6C9', 'features': '#FFE0B2',
        'ml': '#E1BEE7', 'serving': '#FFCDD2'
    }

    positions = {}
    for tier_name, datasets in tiers.items():
        x = tier_x[tier_name]
        for i, ds in enumerate(datasets):
            if ds in catalog.datasets:
                y = (len(datasets) - 1) / 2 - i
                positions[ds] = (x, y)

    # 绘制节点
    for tier_name, datasets in tiers.items():
        color = tier_colors[tier_name]
        for ds in datasets:
            if ds in positions:
                x, y = positions[ds]
                meta = catalog.datasets[ds]
                rect = mpatches.FancyBboxPatch(
                    (x - 1.2, y - 0.35), 2.4, 0.7,
                    boxstyle="round,pad=0.1",
                    facecolor=color, edgecolor='#333', linewidth=1.5
                )
                ax.add_patch(rect)
                ax.text(x, y + 0.08, ds, ha='center', va='center',
                        fontsize=8, fontweight='bold')
                quality = meta.get('quality_score')
                q_text = f"Q:{quality:.0%}" if quality else ''
                rows = meta.get('row_count')
                r_text = f"({rows//1000}K行)" if rows else ''
                ax.text(x, y - 0.15, f"{q_text} {r_text}", ha='center', va='center', fontsize=6, color='#666')

    # 绘制连线
    for target, info in catalog.lineage.items():
        if target in positions:
            tx, ty = positions[target]
            for source in info['sources']:
                if source in positions:
                    sx, sy = positions[source]
                    ax.annotate(
                        '', xy=(tx - 1.2, ty), xytext=(sx + 1.2, sy),
                        arrowprops=dict(arrowstyle='->', color='#666', lw=1.5,
                                        connectionstyle='arc3,rad=0.1')
                    )

    # 标签
    for tier_name, x in tier_x.items():
        ax.text(x, 2.5, tier_name.upper(), ha='center', fontsize=11,
                fontweight='bold', color=tier_colors.get(tier_name, '#333'),
                bbox=dict(boxstyle='round,pad=0.3', facecolor=tier_colors.get(tier_name, '#eee'),
                          edgecolor='#999'))

    ax.set_xlim(-2, 15)
    ax.set_ylim(-3, 3.5)
    ax.set_title('数据血缘关系图', fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('D:/code/big-model-learn/code/q_01/W25/d6_data_catalog_lineage.png', dpi=150, bbox_inches='tight')
    plt.show()


draw_lineage(catalog)

# ============================================================
# 6. 数据资产统计可视化
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 按层级统计
tier_counts = {'tier-1': 0, 'tier-2': 0, 'tier-3': 0}
for meta in catalog.datasets.values():
    tier = meta.get('tier', 'tier-3')
    if tier in tier_counts:
        tier_counts[tier] += 1

axes[0].bar(tier_counts.keys(), tier_counts.values(), color=['#2196F3', '#4CAF50', '#FF9800'])
axes[0].set_title('数据集按层级分布')
axes[0].set_ylabel('数量')
for i, (k, v) in enumerate(tier_counts.items()):
    axes[0].text(i, v + 0.1, str(v), ha='center', fontweight='bold')

# 敏感度分布
sensitivity_counts = {}
for meta in catalog.datasets.values():
    s = meta.get('sensitivity', 'unknown')
    sensitivity_counts[s] = sensitivity_counts.get(s, 0) + 1

colors_sens = ['#4CAF50', '#FF9800', '#F44336']
axes[1].pie(sensitivity_counts.values(), labels=sensitivity_counts.keys(),
            colors=colors_sens[:len(sensitivity_counts)], autopct='%1.0f%%', startangle=90)
axes[1].set_title('数据敏感度分布')

# 质量评分
quality_data = {name: meta['quality_score'] for name, meta in catalog.datasets.items()
                if meta.get('quality_score') is not None}
if quality_data:
    names = list(quality_data.keys())
    scores = list(quality_data.values())
    colors_q = ['#4CAF50' if s >= 0.9 else '#FF9800' if s >= 0.8 else '#F44336' for s in scores]
    axes[2].barh(names, scores, color=colors_q)
    axes[2].set_xlim(0.7, 1.0)
    axes[2].set_xlabel('质量评分')
    axes[2].set_title('数据集质量评分')
    axes[2].axvline(0.9, color='green', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W25/d6_data_catalog_stats.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存: d6_data_catalog_lineage.png, d6_data_catalog_stats.png")

print("\n完成! 数据目录管理要点:")
print("  1. 数据目录: 注册、发现、搜索组织数据资产")
print("  2. 元数据: Schema、所有者、标签、统计信息")
print("  3. 数据字典: 自动生成字段定义、类型、约束")
print("  4. 数据血缘: 追踪数据流向, 理解依赖关系")
