"""
W28-D1 MLOps概述
=================
MLOps成熟度模型, Google MLOps三级, 完整MLOps架构图, 工具链选择
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W28-D1 MLOps概述")
print("=" * 60)

# ============================================================
# 1. MLOps定义与成熟度模型
# ============================================================
print("\n--- 1. MLOps定义与成熟度模型 ---")
print("""
  MLOps = Machine Learning + Operations
  目标: 可靠、高效地将ML模型投入生产并持续维护

  MLOps成熟度模型:

  Level 0 - 无MLOps (手动流程)
    - 手动训练, 脚本执行
    - 手动部署
    - 无监控, 无自动化

  Level 1 - ML Pipeline自动化
    - 自动化训练Pipeline
    - 持续训练 (CT)
    - 实验追踪
    - 基础监控

  Level 2 - CI/CD自动化
    - ML Pipeline的CI/CD
    - 自动化测试
    - 模型注册与版本管理
    - 特征存储
    - 完整监控

  Level 3 - 全自动化
    - 自动特征工程
    - 自动模型选择
    - A/B测试自动化
    - 自适应重训练
    - 全链路可观测性
""")

# ============================================================
# 2. Google MLOps三级模型
# ============================================================
print("\n--- 2. Google MLOps三级模型 ---")
print("""
  Google MLOps成熟度三级:

  ┌─────────────────────────────────────────────────────────────┐
  │ Level 1: ML Pipeline (开发者驱动)                           │
  │                                                              │
  │  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
  │  │实验     │->│Pipeline  │->│模型注册  │->│手动部署  │   │
  │  │Notebook │  │自动化    │  │版本管理  │  │          │   │
  │  └─────────┘  └──────────┘  └──────────┘  └──────────┘   │
  │  关键能力: Pipeline编排, 实验追踪, 模型版本化              │
  ├─────────────────────────────────────────────────────────────┤
  │ Level 2: ML Pipeline + CI/CD                                │
  │                                                              │
  │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐       │
  │  │代码CI│->│Pipeline│->│模型测试│->│CD部署│->│监控  │       │
  │  │      │  │训练   │  │验证   │  │      │  │告警  │       │
  │  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘       │
  │  关键能力: 自动化测试, CI/CD, 特征存储, 监控              │
  ├─────────────────────────────────────────────────────────────┤
  │ Level 3: 自动化 + 持续训练                                  │
  │                                                              │
  │  ┌─────────────────────────────────────────────┐            │
  │  │  自动数据验证 → 自动训练 → 自动测试 → 自动部署│           │
  │  │  ↑                                            │           │
  │  │  └─ 监控反馈 ← 漂移检测 ← 性能监控 ←────────┘           │
  │  └─────────────────────────────────────────────┘            │
  │  关键能力: 自动重训练, 漂移检测, A/B测试, 自适应           │
  └─────────────────────────────────────────────────────────────┘
""")

# ============================================================
# 3. 完整MLOps架构
# ============================================================
print("\n--- 3. 完整MLOps架构 ---")
print("""
  ┌─────────────────────────────────────────────────────────────────┐
  │                       MLOps 完整架构                            │
  │                                                                  │
  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
  │  │  数据层      │  │  训练层      │  │  服务层      │          │
  │  │              │  │              │  │              │          │
  │  │ 数据湖/仓库  │  │ 实验追踪    │  │ 模型服务    │          │
  │  │ 特征存储    │  │ Pipeline编排 │  │ API网关     │          │
  │  │ 数据目录    │  │ 超参优化    │  │ 批量推理    │          │
  │  │ 数据质量    │  │ 模型注册    │  │ 负载均衡    │          │
  │  └──────────────┘  └──────────────┘  └──────────────┘          │
  │                                                                  │
  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
  │  │  监控层      │  │  治理层      │  │  基础设施层  │          │
  │  │              │  │              │  │              │          │
  │  │ 性能监控    │  │ 访问控制    │  │ 容器编排(K8s)│          │
  │  │ 漂移检测    │  │ 合规审计    │  │ GPU调度     │          │
  │  │ 日志追踪    │  │ 成本管理    │  │ 存储        │          │
  │  │ 告警通知    │  │ 模型文档    │  │ 网络        │          │
  │  └──────────────┘  └──────────────┘  └──────────────┘          │
  └─────────────────────────────────────────────────────────────────┘
""")

# ============================================================
# 4. 工具链选择
# ============================================================
print("\n--- 4. 工具链选择 ---")

tools_data = {
    '类别': [
        '数据Pipeline', '数据Pipeline', '数据Pipeline',
        '特征存储', '特征存储',
        '实验追踪', '实验追踪', '实验追踪',
        'Pipeline编排', 'Pipeline编排', 'Pipeline编排',
        '模型服务', '模型服务', '模型服务',
        '监控', '监控', '监控',
        'CI/CD', 'CI/CD',
        '基础设施', '基础设施',
    ],
    '工具': [
        'Apache Spark', 'Apache Airflow', 'Prefect',
        'Feast', 'Tecton',
        'MLflow', 'Weights & Biases', 'Neptune',
        'Kubeflow', 'Airflow', 'Prefect',
        'TF Serving', 'Triton', 'Seldon',
        'Prometheus', 'Grafana', 'Evidently',
        'GitHub Actions', 'GitLab CI',
        'Kubernetes', 'Docker',
    ],
    '语言/生态': [
        'Python/Scala', 'Python', 'Python',
        'Python', 'Python/Java',
        'Python', 'Python', 'Python',
        'Python/K8s', 'Python', 'Python',
        'C++/Python', 'C++/Python', 'Python/K8s',
        'Go', 'JS/TS', 'Python',
        'YAML', 'YAML',
        'Go', 'Go',
    ],
    '适用场景': [
        '大规模数据处理', '通用工作流', '轻量级工作流',
        '开源特征库', '企业级特征平台',
        '通用ML实验管理', '深度学习实验', '轻量实验追踪',
        'K8s原生ML', '通用编排', '现代数据流',
        'TensorFlow模型', '多框架GPU推理', 'K8s模型服务',
        '指标收集', '可视化仪表盘', 'ML专用监控',
        'GitHub生态', 'GitLab生态',
        '容器编排', '容器化',
    ],
}

tools_df = pd.DataFrame(tools_data)
print(f"\n  MLOps工具链 ({len(tools_df)} 个工具, {tools_df['类别'].nunique()} 个类别):\n")

for category in tools_df['类别'].unique():
    cat_tools = tools_df[tools_df['类别'] == category]
    print(f"  {category}:")
    for _, row in cat_tools.iterrows():
        print(f"    - {row['工具']:<20} ({row['语言/生态']:<15}) {row['适用场景']}")
    print()

# ============================================================
# 5. MLOps成熟度自评
# ============================================================
print("\n--- 5. MLOps成熟度自评 ---")


class MLOpsMaturityAssessment:
    """MLOps成熟度评估"""

    CATEGORIES = {
        '数据管理': [
            '数据版本控制',
            '自动化数据Pipeline',
            '数据质量监控',
            '特征存储',
        ],
        '实验管理': [
            '实验追踪系统',
            '超参数自动化',
            '模型版本管理',
            '实验可复现性',
        ],
        '模型部署': [
            '自动化部署Pipeline',
            'A/B测试',
            '灰度发布',
            '模型回滚',
        ],
        '监控运维': [
            '模型性能监控',
            '数据漂移检测',
            '自动重训练',
            '告警与通知',
        ],
        '治理合规': [
            '访问控制',
            '模型文档',
            '公平性审计',
            '成本管理',
        ],
    }

    def assess(self, scores=None):
        """评估 (每项0-3分)"""
        if scores is None:
            # 默认示例分数
            scores = {
                '数据管理': [2, 1, 1, 0],
                '实验管理': [2, 1, 2, 1],
                '模型部署': [1, 0, 0, 1],
                '监控运维': [1, 0, 0, 1],
                '治理合规': [0, 0, 0, 0],
            }

        print(f"\n  {'类别':<12} {'当前':<6} {'满分':<6} {'百分比':<8} {'等级'}")
        print("  " + "-" * 50)

        total_score = 0
        total_max = 0
        category_results = {}

        for category, items in self.CATEGORIES.items():
            cat_scores = scores.get(category, [0] * len(items))
            cat_max = len(items) * 3
            cat_total = sum(cat_scores)
            total_score += cat_total
            total_max += cat_max

            pct = cat_total / cat_max
            if pct >= 0.75:
                level = '优秀'
            elif pct >= 0.5:
                level = '良好'
            elif pct >= 0.25:
                level = '基础'
            else:
                level = '初始'

            category_results[category] = {
                'score': cat_total, 'max': cat_max, 'pct': pct, 'level': level,
            }
            print(f"  {category:<12} {cat_total:<6} {cat_max:<6} {pct:<8.0%} {level}")

        overall_pct = total_score / total_max
        print(f"\n  总分: {total_score}/{total_max} ({overall_pct:.0%})")

        if overall_pct >= 0.75:
            mlops_level = 3
        elif overall_pct >= 0.5:
            mlops_level = 2
        elif overall_pct >= 0.25:
            mlops_level = 1
        else:
            mlops_level = 0

        print(f"  MLOps成熟度等级: Level {mlops_level}")
        return category_results, mlops_level


assessment = MLOpsMaturityAssessment()
results, level = assessment.assess()

# ============================================================
# 6. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 左上: MLOps成熟度雷达图
ax = axes[0, 0]
categories = list(results.keys())
scores_pct = [results[c]['pct'] * 100 for c in categories]

angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
scores_pct_closed = scores_pct + [scores_pct[0]]
angles_closed = angles + [angles[0]]

ax = fig.add_subplot(2, 2, 1, polar=True)
ax.fill(angles_closed, scores_pct_closed, alpha=0.25, color='#2196F3')
ax.plot(angles_closed, scores_pct_closed, 'o-', color='#2196F3', linewidth=2)
ax.set_xticks(angles)
ax.set_xticklabels(categories, fontsize=9)
ax.set_ylim(0, 100)
ax.set_title('MLOps成熟度评估', pad=20)

# 右上: 工具类别分布
ax = axes[0, 1]
tool_counts = tools_df['类别'].value_counts()
ax.barh(tool_counts.index, tool_counts.values, color='#4CAF50')
ax.set_xlabel('工具数量')
ax.set_title('MLOps工具类别分布')
for i, (cat, count) in enumerate(zip(tool_counts.index, tool_counts.values)):
    ax.text(count + 0.1, i, str(count), va='center')

# 左下: MLOps层级对比
ax = axes[1, 0]
levels = ['Level 0\n手动', 'Level 1\nPipeline', 'Level 2\nCI/CD', 'Level 3\n全自动化']
automation_pct = [10, 40, 70, 95]
effort_pct = [90, 60, 30, 10]

x = np.arange(len(levels))
width = 0.35
ax.bar(x - width/2, automation_pct, width, label='自动化比例', color='#4CAF50')
ax.bar(x + width/2, effort_pct, width, label='手动工作量', color='#F44336')
ax.set_xticks(x)
ax.set_xticklabels(levels)
ax.set_ylabel('百分比 (%)')
ax.set_title('MLOps成熟度 vs 自动化/工作量')
ax.legend()

# 右下: 推荐工具链
ax = axes[1, 1]
ax.axis('off')
ax.set_title('推荐起步工具链')

recommendations = [
    ('实验追踪', 'MLflow'),
    ('Pipeline', 'Prefect / Airflow'),
    ('特征存储', 'Feast'),
    ('模型服务', 'FastAPI + Docker'),
    ('监控', 'Evidently + Grafana'),
    ('CI/CD', 'GitHub Actions'),
    ('基础设施', 'Docker + Kubernetes'),
    ('版本管理', 'Git + DVC'),
]

for i, (cat, tool) in enumerate(recommendations):
    y = 0.9 - i * 0.1
    ax.text(0.1, y, f'{cat}:', fontsize=11, fontweight='bold', transform=ax.transAxes)
    ax.text(0.45, y, tool, fontsize=11, transform=ax.transAxes, color='#1565C0')

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W28/d1_mlops_overview.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存: d1_mlops_overview.png")

print("\n完成! MLOps概述要点:")
print("  1. 成熟度模型: Level 0(手动) -> Level 3(全自动化)")
print("  2. Google三级: Pipeline自动化 -> CI/CD -> 持续训练")
print("  3. 架构: 数据层/训练层/服务层/监控层/治理层/基础设施")
print("  4. 工具选择: 根据团队规模和技术栈选择合适工具")
print("  5. 渐进式: 从Level 0逐步提升, 不要一步到位")
