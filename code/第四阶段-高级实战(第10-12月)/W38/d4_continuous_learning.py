"""
W38 Day 4 - 持续学习
====================
主题: 学习资源整理, AI技术雷达, 成长路径图
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
# 1. 学习资源整理
# =============================================
def compile_learning_resources():
    """整理AI学习资源"""
    print_section("AI学习资源整理")

    resources = {
        "在线课程": [
            {"名称": "Andrew Ng - Machine Learning", "平台": "Coursera",
             "特点": "ML入门经典, 理论扎实"},
            {"名称": "Deep Learning Specialization", "平台": "Coursera",
             "特点": "DL深度学习5门课程系列"},
            {"名称": "Stanford CS224N (NLP)", "平台": "YouTube",
             "特点": "NLP方向顶级课程"},
            {"名称": "Stanford CS231N (CV)", "平台": "YouTube",
             "特点": "计算机视觉经典课程"},
            {"名称": "fast.ai", "平台": "fast.ai",
             "特点": "实战导向, 自顶向下教学法"},
            {"名称": "Hugging Face Course", "平台": "huggingface.co",
             "特点": "Transformers库官方教程"},
            {"名称": "LangChain Academy", "平台": "LangChain",
             "特点": "LLM应用开发实战"}
        ],
        "经典书籍": [
            {"名称": "Hands-On ML with Scikit-Learn, Keras, TF", "难度": "初级-中级"},
            {"名称": "Deep Learning (Goodfellow)", "难度": "高级(理论)"},
            {"名称": "Python机器学习( Sebastian Raschka)", "难度": "中级"},
            {"名称": "Designing ML Systems (Chip Huyen)", "难度": "中级(工程)"},
            {"名称": "LLM Book (大模型相关)", "难度": "中级"},
            {"名称": "百面机器学习", "难度": "面试导向"},
        ],
        "论文资源": [
            "arXiv (arxiv.org) - 最新论文预印本",
            "Papers With Code (paperswithcode.com) - 论文+代码+基准",
            "Semantic Scholar (semanticscholar.org) - AI论文搜索引擎",
            "Google Scholar - 学术搜索",
            "Connected Papers - 论文关系图谱"
        ],
        "实战平台": [
            "Kaggle - 数据科学竞赛",
            "Hugging Face Spaces - 模型Demo展示",
            "Google Colab - 免费GPU环境",
            "GitHub - 开源项目和代码",
            "LeetCode - 编程练习"
        ],
        "技术博客/社区": [
            "Hugging Face Blog - AI最新动态",
            "OpenAI Blog - 前沿研究",
            "Google AI Blog - Google AI研究",
            "The Batch (Andrew Ng) - AI新闻周报",
            "机器之心 / 量子位 - 国内AI媒体",
            "Towards Data Science - Medium AI专栏"
        ],
        "工具与框架文档": [
            "PyTorch官方文档 (pytorch.org)",
            "Transformers文档 (huggingface.co/docs)",
            "LangChain文档 (python.langchain.com)",
            "FastAPI文档 (fastapi.tiangolo.com)",
            "scikit-learn文档 (scikit-learn.org)"
        ]
    }

    for category, items in resources.items():
        print(f"\n【{category}】")
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    details = " | ".join(f"{k}: {v}" for k, v in item.items())
                    print(f"  {details}")
                else:
                    print(f"  {item}")


# =============================================
# 2. AI技术雷达
# =============================================
def generate_tech_radar():
    """生成AI技术雷达"""
    print_section("AI技术雷达(2025-2026)")

    radar = {
        "采用(Adopt) - 建议立即使用": [
            "RAG (检索增强生成) - LLM应用的标准模式",
            "LangChain/LlamaIndex - LLM应用开发框架",
            "LoRA/QLoRA - 高效模型微调",
            "Hugging Face Transformers - 模型和工具",
            "FastAPI - AI应用API服务",
            "Docker - 应用容器化"
        ],
        "试验(Trial) - 值得尝试": [
            "AI Agent (ReAct, LangGraph) - 自主决策系统",
            "多模态大模型 (GPT-4V, LLaVA) - 图文理解",
            "向量数据库 (ChromaDB, Milvus) - 语义检索",
            "vLLM - 高吞吐LLM推理",
            "模型量化 (GPTQ, AWQ) - 推理优化",
            "MLflow - ML实验管理"
        ],
        "评估(Assess) - 关注发展": [
            "Mixture of Experts (MoE) - 混合专家模型",
            "Embodied AI - 具身智能",
            "Small Language Models - 小模型蒸馏",
            "On-device AI - 端侧推理",
            "AI Safety & Alignment - AI安全与对齐",
            "Neural Architecture Search - 自动机器学习"
        ],
        "暂缓(Hold) - 暂不建议大量投入": [
            "从零训练大模型 - 成本太高, 不适合大多数公司",
            "纯符号AI - 深度学习已占主导",
            "区块链+AI - 应用场景不明确"
        ]
    }

    for category, items in radar.items():
        print(f"\n【{category}】")
        for item in items:
            print(f"  {item}")

    # 技术成熟度图表
    if HAS_MATPLOTLIB and HAS_NUMPY:
        fig, ax = plt.subplots(figsize=(12, 6))

        techs = ['RAG', 'LoRA', 'LangChain', 'AI Agent', '多模态',
                 '向量数据库', 'vLLM', 'MoE', 'Embodied AI',
                 '端侧AI', 'AGI']
        maturity = [90, 85, 80, 60, 55, 75, 70, 40, 25, 35, 10]
        importance = [95, 85, 85, 90, 80, 80, 75, 65, 50, 60, 70]

        scatter = ax.scatter(maturity, importance,
                           c=range(len(techs)), cmap='viridis',
                           s=200, alpha=0.7, edgecolors='black')

        for i, tech in enumerate(techs):
            ax.annotate(tech, (maturity[i], importance[i]),
                       textcoords="offset points", xytext=(10, 5),
                       fontsize=9)

        ax.set_xlabel('技术成熟度')
        ax.set_ylabel('重要程度')
        ax.set_title('AI技术雷达: 成熟度 vs 重要程度')
        ax.grid(True, alpha=0.3)

        # 区域划分
        ax.axhline(y=70, color='r', linestyle='--', alpha=0.3)
        ax.axvline(x=70, color='r', linestyle='--', alpha=0.3)
        ax.text(85, 90, '优先采用', fontsize=12, alpha=0.3, ha='center')
        ax.text(40, 90, '重点跟踪', fontsize=12, alpha=0.3, ha='center')

        plt.tight_layout()
        chart_path = os.path.join(os.path.dirname(__file__), "tech_radar.png")
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  技术雷达图已保存: {chart_path}")


# =============================================
# 3. 成长路径图
# =============================================
def generate_growth_path():
    """生成AI工程师成长路径图"""
    print_section("AI工程师成长路径")

    growth_path = {
        "初级AI工程师 (0-1年)": {
            "技能要求": [
                "Python编程扎实",
                "了解ML/DL基础理论",
                "能使用现有框架(Transformers, LangChain)构建应用",
                "基本的数据处理和可视化能力"
            ],
            "典型工作": [
                "LLM应用开发(Prompt Engineering, RAG)",
                "数据处理和特征工程",
                "简单的模型微调",
                "编写测试和文档"
            ],
            "薪资范围": "20-35K",
            "学习重点": "夯实基础, 多做项目, 建立代码作品集"
        },
        "中级AI工程师 (1-3年)": {
            "技能要求": [
                "深入理解Transformer架构",
                "能独立设计和实现AI系统",
                "模型训练、微调、部署全流程",
                "系统设计和性能优化能力"
            ],
            "典型工作": [
                "设计和实现RAG/Agent系统",
                "模型微调和评估",
                "AI系统架构设计",
                "指导和Review初级工程师"
            ],
            "薪资范围": "35-55K",
            "学习重点": "深入特定方向, 积累系统设计经验"
        },
        "高级AI工程师 (3-5年)": {
            "技能要求": [
                "某一方向的深度专长(NLP/CV/推荐)",
                "大规模系统设计经验",
                "技术选型和架构决策能力",
                "团队指导和技术传播"
            ],
            "典型工作": [
                "AI平台架构设计",
                "技术方向决策",
                "复杂系统性能优化",
                "跨团队技术协调"
            ],
            "薪资范围": "50-80K",
            "学习重点": "技术领导力, 行业深度, 创新能力"
        },
        "资深/专家 (5年+)": {
            "技能要求": [
                "行业技术影响力",
                "战略级技术决策",
                "跨领域技术洞察",
                "团队建设和人才培养"
            ],
            "薪资范围": "80K+ / 股权激励",
            "学习重点": "技术战略, 行业视野, 领导力"
        }
    }

    for level, details in growth_path.items():
        print(f"\n【{level}】")
        for key, value in details.items():
            if isinstance(value, list):
                print(f"  {key}:")
                for item in value:
                    print(f"    - {item}")
            else:
                print(f"  {key}: {value}")

    # 个性化成长建议
    print("\n--- 个性化成长建议 ---")
    advice = {
        "当前定位": "Java后端 + AI应用开发 = LLM应用工程师",
        "当前阶段": "初级-中级(转型后约6个月)",
        "3个月目标": "能独立设计和实现生产级RAG/Agent系统",
        "6个月目标": "成为中级AI工程师, 有2-3个高质量项目",
        "1年目标": "在LLM应用方向有深度, 可指导他人",
        "核心差异化": "后端架构经验 + AI应用能力 = 企业级AI应用专家"
    }
    for key, value in advice.items():
        print(f"  {key}: {value}")


# =============================================
# 主程序
# =============================================
if __name__ == "__main__":
    print("=" * 60)
    print("  W38 Day 4 - 持续学习")
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    compile_learning_resources()
    generate_tech_radar()
    generate_growth_path()

    print("\n" + "=" * 60)
    print("  技术更新很快, 但核心原理是不变的!")
    print("  建议: 建立自己的学习资源库, 定期更新")
    print("=" * 60)
