"""
W39 Day 2 - 技能矩阵
====================
主题: AI工程师技能树生成, 当前水平评估, 提升方向
"""

import os
import json
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
# 1. AI工程师技能树
# =============================================
def generate_skill_tree():
    """生成AI工程师技能树"""
    print_section("AI工程师技能树")

    skill_tree = {
        "AI应用工程师": {
            "核心编程": {
                "Python (必须)": {
                    "等级": "L4-熟练",
                    "子技能": ["语法基础", "面向对象", "函数式编程", "异步编程", "类型注解"],
                    "下一步": "深入学习Python高级特性"
                },
                "数据科学工具": {
                    "等级": "L3-良好",
                    "子技能": ["NumPy", "Pandas", "Matplotlib", "Seaborn"],
                    "下一步": "学习Polars和高级可视化"
                },
                "SQL与数据库": {
                    "等级": "L4-熟练",
                    "子技能": ["SQL查询", "MySQL", "Redis", "向量数据库"],
                    "下一步": "深入学习向量数据库优化"
                }
            },
            "AI/ML核心": {
                "机器学习": {
                    "等级": "L3-良好",
                    "子技能": ["监督学习", "无监督学习", "模型评估", "特征工程"],
                    "下一步": "更多实战项目, 深入理解算法原理"
                },
                "深度学习": {
                    "等级": "L3-良好",
                    "子技能": ["神经网络", "CNN", "RNN/Transformer", "PyTorch"],
                    "下一步": "深入Transformer变体和模型架构"
                },
                "NLP": {
                    "等级": "L4-熟练",
                    "子技能": ["文本处理", "词向量", "文本分类", "大语言模型"],
                    "下一步": "深入研究LLM训练和对齐技术"
                }
            },
            "LLM应用": {
                "RAG系统": {
                    "等级": "L4-熟练",
                    "子技能": ["文档处理", "向量化", "检索优化", "RAG评估"],
                    "下一步": "高级RAG技术(Self-RAG, CRAG)"
                },
                "Prompt Engineering": {
                    "等级": "L4-熟练",
                    "子技能": ["Few-shot", "CoT", "结构化输出", "角色设定"],
                    "下一步": "自动Prompt优化技术"
                },
                "AI Agent": {
                    "等级": "L3-良好",
                    "子技能": ["ReAct", "工具调用", "记忆系统", "LangGraph"],
                    "下一步": "Multi-Agent系统和复杂工作流"
                },
                "模型微调": {
                    "等级": "L3-良好",
                    "子技能": ["LoRA/QLoRA", "数据准备", "训练配置", "评估"],
                    "下一步": "RLHF和DPO对齐技术"
                }
            },
            "工程能力": {
                "系统设计": {
                    "等级": "L4-熟练",
                    "子技能": ["架构设计", "API设计", "数据库设计", "缓存策略"],
                    "下一步": "大规模AI系统设计"
                },
                "模型部署": {
                    "等级": "L3-良好",
                    "子技能": ["FastAPI", "Docker", "量化推理", "vLLM"],
                    "下一步": "Kubernetes和自动扩缩容"
                },
                "MLOps": {
                    "等级": "L2-基础",
                    "子技能": ["MLflow", "实验管理", "模型注册"],
                    "下一步": "完整的ML Pipeline自动化"
                }
            },
            "软技能": {
                "技术写作": {
                    "等级": "L3-良好",
                    "子技能": ["技术博客", "文档编写", "教程制作"],
                    "下一步": "持续输出高质量内容"
                },
                "开源贡献": {
                    "等级": "L2-基础",
                    "子技能": ["Git协作", "PR流程", "代码审查"],
                    "下一步": "成为活跃Contributor"
                }
            }
        }
    }

    for role, categories in skill_tree.items():
        print(f"\n  角色: {role}")
        for category, skills in categories.items():
            print(f"\n  【{category}】")
            for skill, details in skills.items():
                print(f"    {skill}: {details['等级']}")
                print(f"      子技能: {', '.join(details['子技能'])}")
                print(f"      下一步: {details['下一步']}")


# =============================================
# 2. 当前水平评估
# =============================================
def assess_current_level():
    """评估当前水平"""
    print_section("当前水平评估")

    if HAS_MATPLOTLIB and HAS_NUMPY:
        # 生成技能雷达图
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        categories = ['Python\n编程', '数学\n基础', '机器\n学习', '深度\n学习',
                      'NLP/\nLLM', 'RAG\n系统', 'AI\nAgent', '模型\n微调',
                      '系统\n设计', '模型\n部署', 'MLOps', '技术\n写作']

        current = [4.0, 3.0, 3.5, 3.5, 4.0, 4.0, 3.0, 3.0, 4.0, 3.0, 2.0, 3.0]
        target = [4.5, 3.5, 4.0, 4.0, 4.5, 4.5, 4.0, 4.0, 4.5, 4.0, 3.5, 3.5]

        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

        current_plot = current + current[:1]
        target_plot = target + target[:1]
        angles_plot = angles + angles[:1]

        ax.fill(angles_plot, target_plot, alpha=0.1, color='blue')
        ax.plot(angles_plot, target_plot, 'b--', linewidth=1, label='目标水平')

        ax.fill(angles_plot, current_plot, alpha=0.25, color='red')
        ax.plot(angles_plot, current_plot, 'r-', linewidth=2, label='当前水平')

        ax.set_xticks(angles)
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_yticklabels(['1', '2', '3', '4', '5'])
        ax.set_title('AI工程师技能评估雷达图', size=15, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

        plt.tight_layout()
        chart_path = os.path.join(os.path.dirname(__file__), "skill_radar.png")
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  技能雷达图已保存: {chart_path}")

    # 技能差距分析
    print("\n--- 技能差距分析 ---")
    gaps = [
        {"技能": "MLOps", "当前": 2.0, "目标": 3.5, "差距": 1.5, "优先级": "高"},
        {"技能": "AI Agent", "当前": 3.0, "目标": 4.0, "差距": 1.0, "优先级": "高"},
        {"技能": "模型微调", "当前": 3.0, "目标": 4.0, "差距": 1.0, "优先级": "高"},
        {"技能": "模型部署", "当前": 3.0, "目标": 4.0, "差距": 1.0, "优先级": "中"},
        {"技能": "数学基础", "当前": 3.0, "目标": 3.5, "差距": 0.5, "优先级": "中"},
        {"技能": "技术写作", "当前": 3.0, "目标": 3.5, "差距": 0.5, "优先级": "低"},
    ]

    print(f"  {'技能':12s} {'当前':>4s} {'目标':>4s} {'差距':>4s} {'优先级':>6s}")
    print("  " + "-" * 40)
    for gap in gaps:
        print(f"  {gap['技能']:12s} {gap['当前']:>4.1f} {gap['目标']:>4.1f} "
              f"{gap['差距']:>4.1f} {gap['优先级']:>6s}")


# =============================================
# 3. 提升方向
# =============================================
def suggest_improvement_directions():
    """建议提升方向"""
    print_section("提升方向与行动计划")

    directions = {
        "短期(1个月内)": {
            "重点": "补齐最大短板, 准备面试",
            "行动": [
                "深入学习MLOps: 完成一个MLflow实战项目",
                "完善AI Agent项目: 使用LangGraph构建多Agent系统",
                "打磨3个核心项目的GitHub展示",
                "准备面试: 复习50道面试题"
            ]
        },
        "中期(3个月内)": {
            "重点": "深入核心方向, 建立专业形象",
            "行动": [
                "深入RAG系统: 实现高级RAG技术(Self-RAG, CRAG)",
                "模型微调深入: 完成一个RLHF/DPO微调项目",
                "技术博客: 发布8-12篇高质量文章",
                "开源贡献: 在1-2个项目中成为活跃Contributor"
            ]
        },
        "长期(6-12个月)": {
            "重点": "成为某一方向的专家",
            "行动": [
                "选择专精方向: RAG系统 或 AI Agent 或 LLM微调",
                "在企业级场景中积累深度经验",
                "建立行业影响力: 技术演讲、开源项目",
                "持续学习前沿技术, 保持技术敏感度"
            ]
        },
        "差异化定位": {
            "方向": "企业级LLM应用工程师",
            "核心优势": [
                "后端架构设计经验(Java)",
                "LLM应用开发能力(RAG/Agent)",
                "系统工程化能力(部署/监控)",
                "业务理解和产品思维"
            ],
            "目标": "成为能独立设计和实现生产级AI系统的工程师"
        }
    }

    for timeframe, details in directions.items():
        print(f"\n【{timeframe}】")
        if isinstance(details, dict):
            for key, value in details.items():
                if isinstance(value, list):
                    print(f"  {key}:")
                    for item in value:
                        print(f"    - {item}")
                else:
                    print(f"  {key}: {value}")


# =============================================
# 主程序
# =============================================
if __name__ == "__main__":
    print("=" * 60)
    print("  W39 Day 2 - 技能矩阵")
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    generate_skill_tree()
    assess_current_level()
    suggest_improvement_directions()

    print("\n" + "=" * 60)
    print("  了解自己的技能全景, 才能有针对性地提升!")
    print("  建议: 每季度做一次技能评估, 追踪成长进度")
    print("=" * 60)
