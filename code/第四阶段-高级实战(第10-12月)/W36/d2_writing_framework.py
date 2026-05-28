"""
W36 Day 2 - 写作框架
====================
主题: 技术博客结构, 代码展示, 图表制作指南
"""

import os
from datetime import datetime

try:
    import matplotlib
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def print_section(title):
    """打印分隔线"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# =============================================
# 1. 技术博客结构
# =============================================
def demonstrate_blog_structure():
    """展示技术博客的标准结构"""
    print_section("技术博客标准结构")

    structures = {
        "教程型博客": {
            "模板": """
# [技术名称]实战教程: 从零开始[做什么]

## 前言
- 为什么写这篇文章
- 目标读者是谁
- 你将学到什么

## 环境准备
- 系统要求
- 安装依赖
- 代码仓库链接

## 概念介绍
- 核心概念用类比解释
- 配图说明工作原理
- 与已有技术的对比

## 实战步骤

### Step 1: [第一步]
- 具体代码
- 运行结果
- 代码解释

### Step 2: [第二步]
...

## 完整代码
- 可直接运行的完整代码

## 运行效果
- 截图或输出结果

## 常见问题
- Q&A格式

## 总结与下一步
- 回顾要点
- 进阶方向
- 参考资料
""",
            "适用": "教程、入门指南、实战文章"
        },
        "深度分析型博客": {
            "模板": """
# 深度解析: [技术/论文/架构]

## 引言
- 背景与动机
- 解决什么问题
- 为什么重要

## 基础概念
- 预备知识回顾
- 关键术语解释

## 核心原理
- 详细的原理解释
- 数学推导(简化版)
- 架构图/流程图

## 代码实现
- 关键部分的代码
- 逐行解释

## 实验与分析
- 实验设置
- 结果对比
- 数据可视化

## 优缺点分析
- 优势
- 局限性
- 与同类方案的对比

## 实际应用
- 使用场景
- 最佳实践
- 注意事项

## 总结
- 关键收获
- 个人见解
- 未来展望
""",
            "适用": "论文解读、技术分析、架构分析"
        },
        "经验分享型博客": {
            "模板": """
# 我的[经历/经验]: [主题]

## 背景
- 我的情况
- 面临的挑战

## 过程
### 阶段一: [开始]
- 做了什么
- 遇到什么问题

### 阶段二: [进展]
- 关键转折点
- 重要发现

### 阶段三: [成果]
- 最终结果
- 关键数据

## 经验总结
### 成功经验
- 做对了什么

### 踩过的坑
- 做错了什么
- 如果重来会怎么做

## 建议
- 给后来者的建议
- 推荐资源

## 下一步计划
""",
            "适用": "学习心得、项目总结、转型经历"
        }
    }

    for struct_name, details in structures.items():
        print(f"\n【{struct_name}】适用: {details['适用']}")
        print(details["模板"][:200] + "...")
        print(f"  (完整模板共 {len(details['模板'])} 字符)")


# =============================================
# 2. 代码展示技巧
# =============================================
def demonstrate_code_presentation():
    """展示代码展示技巧"""
    print_section("代码展示技巧")

    tips = {
        "代码块规范": [
            "使用语法高亮: ```python ... ```",
            "添加文件名标识: ```python:title=main.py",
            "代码行数控制在30行以内(单段)",
            "关键行添加行号引用"
        ],
        "好代码展示 vs 差代码展示": {
            "差的做法": [
                "直接粘贴几百行代码不加说明",
                "缺少import语句",
                "变量名无意义(a, b, c)",
                "没有输出结果"
            ],
            "好的做法": [
                "分段展示, 每段有说明",
                "包含完整的import",
                "有意义的变量名和函数名",
                "展示运行结果或输出",
                "关键部分加注释"
            ]
        },
        "代码展示模板": '''```python
# 文件: example.py
# 功能: 展示代码展示模板

from typing import List

def calculate_similarity(text1: str, text2: str) -> float:
    """计算两个文本的相似度

    Args:
        text1: 第一个文本
        text2: 第二个文本

    Returns:
        相似度分数, 范围[0, 1]

    Example:
        >>> calculate_similarity("hello", "hello world")
        0.833
    """
    # 核心逻辑
    words1 = set(text1.split())
    words2 = set(text2.split())

    # Jaccard相似度
    intersection = words1 & words2
    union = words1 | words2

    return len(intersection) / len(union) if union else 0.0

# 运行示例
result = calculate_similarity("大语言模型", "大模型语言")
print(f"相似度: {result:.3f}")  # 输出: 相似度: 0.667
```''',
        "图表展示建议": [
            "架构图: 使用draw.io或Excalidraw",
            "流程图: 使用Mermaid语法",
            "数据图表: 使用Matplotlib/Seaborn",
            "时序图: 使用Mermaid或PlantUML",
            "思维导图: 使用XMind或Markmap"
        ]
    }

    for category, items in tips.items():
        print(f"\n【{category}】")
        if isinstance(items, list):
            for item in items:
                print(f"  {item}")
        elif isinstance(items, dict):
            for key, values in items.items():
                print(f"  {key}:")
                for v in values:
                    print(f"    - {v}")
        elif isinstance(items, str):
            print(items)


# =============================================
# 3. 图表制作指南
# =============================================
def create_blog_charts():
    """创建博客常用的图表"""
    print_section("博客图表制作")

    if not HAS_MATPLOTLIB or not HAS_NUMPY:
        print("  需要matplotlib和numpy库来生成图表")
        print("  pip install matplotlib numpy")
        return

    output_dir = os.path.dirname(__file__)

    # 图表1: 学习进度曲线
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 学习曲线
    weeks = np.arange(1, 40)
    theory = 100 * (1 - np.exp(-0.08 * weeks))
    practice = 100 * (1 - np.exp(-0.05 * weeks))
    project = 100 * (1 - np.exp(-0.03 * weeks))

    axes[0, 0].plot(weeks, theory, 'b-', linewidth=2, label='理论知识')
    axes[0, 0].plot(weeks, practice, 'r--', linewidth=2, label='实践能力')
    axes[0, 0].plot(weeks, project, 'g-.', linewidth=2, label='项目经验')
    axes[0, 0].set_xlabel('学习周数')
    axes[0, 0].set_ylabel('掌握程度(%)')
    axes[0, 0].set_title('AI学习进度曲线')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 技术栈雷达图
    categories = ['Python', 'ML/DL', 'NLP', 'LLM', 'MLOps', '数据工程']
    values = [85, 70, 75, 80, 65, 60]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values_plot = values + values[:1]
    angles += angles[:1]

    axes[0, 1].remove()
    ax_radar = fig.add_subplot(2, 2, 2, polar=True)
    ax_radar.fill(angles, values_plot, alpha=0.25)
    ax_radar.plot(angles, values_plot, 'o-', linewidth=2)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(categories)
    ax_radar.set_title('AI工程师技能雷达图', pad=20)

    # 博客阅读量趋势
    months = ['1月', '2月', '3月', '4月', '5月', '6月']
    views = [200, 500, 1200, 2500, 3800, 5200]
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0', '#00BCD4']

    axes[1, 0].bar(months, views, color=colors)
    axes[1, 0].set_xlabel('月份')
    axes[1, 0].set_ylabel('阅读量')
    axes[1, 0].set_title('博客月度阅读量趋势')
    for i, v in enumerate(views):
        axes[1, 0].text(i, v + 100, str(v), ha='center')

    # 技术关注度
    techs = ['LLM', 'Agent', 'RAG', '多模态', 'MLOps', '边缘AI']
    attention = [95, 90, 85, 80, 75, 60]
    colors2 = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(techs)))

    axes[1, 1].barh(techs, attention, color=colors2)
    axes[1, 1].set_xlabel('关注度指数')
    axes[1, 1].set_title('2025年AI技术关注度排行')
    axes[1, 1].set_xlim(0, 100)
    for i, v in enumerate(attention):
        axes[1, 1].text(v + 1, i, f'{v}', va='center')

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "blog_charts.png")
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  博客图表已保存: {chart_path}")

    # 图表2: Mermaid语法示例
    print("\n--- 博客中的架构图(Mermaid语法) ---")
    mermaid_examples = [
        "```mermaid",
        "graph TD",
        "    A[用户输入] --> B[文本预处理]",
        "    B --> C[Embedding模型]",
        "    C --> D[向量数据库]",
        "    D --> E[检索相关文档]",
        "    E --> F[构建Prompt]",
        "    F --> G[LLM生成回答]",
        "    G --> H[返回结果]",
        "```"
    ]
    for line in mermaid_examples:
        print(f"  {line}")


# =============================================
# 4. 写作质量检查
# =============================================
def writing_quality_checklist():
    """写作质量检查清单"""
    print_section("写作质量检查清单")

    checklist = {
        "内容质量": [
            "[ ] 标题是否吸引人且准确",
            "[ ] 开头是否能在3秒内抓住读者",
            "[ ] 内容是否有独特的价值(不是简单搬运)",
            "[ ] 代码示例是否可以运行",
            "[ ] 是否有清晰的逻辑结构",
            "[ ] 结论是否呼应开头"
        ],
        "技术准确性": [
            "[ ] 技术概念描述是否准确",
            "[ ] 代码是否经过测试",
            "[ ] 版本号和API是否是最新的",
            "[ ] 数据和结论是否有依据"
        ],
        "可读性": [
            "[ ] 段落是否简短(3-5句/段)",
            "[ ] 是否使用了小标题分段",
            "[ ] 是否有列表和表格",
            "[ ] 是否有代码高亮",
            "[ ] 图片是否清晰有标注",
            "[ ] 专业术语是否有解释"
        ],
        "SEO优化": [
            "[ ] 关键词是否在标题中出现",
            "[ ] 是否有meta描述",
            "[ ] 图片是否有alt属性",
            "[ ] URL是否简洁有意义",
            "[ ] 是否有内部链接和外部链接"
        ],
        "发布前": [
            "[ ] 通读全文检查错别字",
            "[ ] 检查所有链接有效",
            "[ ] 检查代码格式正确",
            "[ ] 在手机端预览效果",
            "[ ] 设置发布时间",
            "[ ] 准备社交媒体推广文案"
        ]
    }

    for category, items in checklist.items():
        print(f"\n【{category}】")
        for item in items:
            print(f"  {item}")


# =============================================
# 主程序
# =============================================
if __name__ == "__main__":
    print("=" * 60)
    print("  W36 Day 2 - 写作框架")
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    demonstrate_blog_structure()
    demonstrate_code_presentation()
    create_blog_charts()
    writing_quality_checklist()

    print("\n" + "=" * 60)
    print("  好的写作框架让博客写作事半功倍!")
    print("  建议: 先搭建骨架, 再填充内容, 最后润色")
    print("=" * 60)
