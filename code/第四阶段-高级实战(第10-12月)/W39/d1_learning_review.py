"""
W39 Day 1 - 学习回顾
====================
主题: 39周知识点总表, 掌握程度自评, 知识图谱
"""

import os
import json
from datetime import datetime


def print_section(title):
    """打印分隔线"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# =============================================
# 1. 39周知识点总表
# =============================================
def generate_knowledge_summary():
    """生成39周知识点总表"""
    print_section("39周知识点总表")

    weeks = {
        "第一阶段: 基础建设 (W1-W8)": {
            "W1": "Python基础语法、数据类型、控制流",
            "W2": "Python函数、模块、Pandas进阶",
            "W3": "NumPy进阶、线性代数基础",
            "W4": "数据可视化(Matplotlib/Seaborn)",
            "W5": "概率论基础、统计概念",
            "W6": "微积分基础、梯度概念",
            "W7": "线性代数进阶(矩阵分解、SVD)",
            "W8": "最优化方法(梯度下降、凸优化)"
        },
        "第二阶段: 机器学习 (W9-W16)": {
            "W9": "机器学习概述、监督学习基础",
            "W10": "线性回归、逻辑回归、正则化",
            "W11": "决策树、随机森林、集成学习",
            "W12": "SVM、聚类、降维",
            "W13": "神经网络基础、感知机、��活函数",
            "W14": "前向传播、反向传播、优化器",
            "W15": "CNN卷积神经网络",
            "W16": "RNN/LSTM循环神经网络"
        },
        "第三阶段: 深度学习与NLP (W17-W24)": {
            "W17": "NLP基础、文本预处理、分词",
            "W18": "词向量(Word2Vec/GloVe/FastText)",
            "W19": "文本分类、情感分析",
            "W20": "序列标注、NER、文本生成",
            "W21": "Transformer架构详解",
            "W22": "Self-Attention、多头注意力、位置编码",
            "W23": "BERT原理与微调",
            "W24": "GPT系列、大模型原理"
        },
        "第四阶段: LLM应用 (W25-W32)": {
            "W25": "Transformers库实战",
            "W26": "LangChain框架入门",
            "W27": "RAG系统构建(基础)",
            "W28": "RAG系统构建(进阶)",
            "W29": "Prompt Engineering技巧",
            "W30": "模型微调(LoRA/QLoRA)",
            "W31": "AI Agent开发",
            "W32": "模型部署与优化"
        },
        "第五阶段: 综合与求职 (W33-W39)": {
            "W33": "综合项目(一): 端到端AI应用",
            "W34": "综合项目(二): 项目展示与文档",
            "W35": "开源贡献: 文化、源码阅读、PR流程",
            "W36": "技术博客: 选题、写作、个人品牌",
            "W37": "求职准备: 简历、面试题、系统设计",
            "W38": "项目打磨: GitHub优化、求职策略",
            "W39": "最终总结: 回顾、技能矩阵、职业规划"
        }
    }

    for phase, week_details in weeks.items():
        print(f"\n【{phase}】")
        for week, content in week_details.items():
            print(f"  {week}: {content}")

    # 知识点统计
    total_topics = {
        "Python编程": "30+ 知识点",
        "数学基础": "20+ 知识点",
        "机器学习": "25+ 知识点",
        "深度学习": "20+ 知识点",
        "NLP": "20+ 知识点",
        "LLM应用": "25+ 知识点",
        "工程实践": "15+ 知识点",
        "求职技能": "10+ 知识点"
    }
    print("\n--- 知识点统计 ---")
    for category, count in total_topics.items():
        print(f"  {category}: {count}")
    print(f"  总计: 165+ 核心知识点")


# =============================================
# 2. 掌握程度自评
# =============================================
def self_assessment():
    """掌握程度自评"""
    print_section("掌握程度自评")

    assessment = {
        "编程能力": {
            "Python基础": {"自评": 4, "说明": "能熟练使用Python开发"},
            "NumPy/Pandas": {"自评": 4, "说明": "能进行数据处理和分析"},
            "数据可视化": {"自评": 3, "说明": "能制作常用图表"},
            "代码质量": {"自评": 4, "说明": "有良好的编码习惯"}
        },
        "数学基础": {
            "线性代数": {"自评": 3, "说明": "理解矩阵运算和分解"},
            "概率统计": {"自评": 3, "说明": "理解常用分布和假设检验"},
            "微积分": {"自评": 3, "说明": "理解导数和梯度概念"},
            "最优化": {"自评": 3, "说明": "理解梯度下降和凸优化"}
        },
        "机器学习": {
            "监督学习": {"自评": 4, "说明": "能使用sklearn进行建模"},
            "无监督学习": {"自评": 3, "说明": "了解聚类和降维"},
            "模型评估": {"自评": 4, "说明": "熟悉各种评估指标"},
            "特征工程": {"自评": 3, "说明": "能进行基本特征处理"}
        },
        "深度学习": {
            "神经网络": {"自评": 4, "说明": "理解前向/反向传播"},
            "CNN": {"自评": 3, "说明": "了解卷积网络原理"},
            "RNN/LSTM": {"自评": 3, "说明": "理解序列模型"},
            "Transformer": {"自评": 4, "说明": "深入理解Attention机制"},
            "PyTorch": {"自评": 4, "说明": "能独立编写训练代码"}
        },
        "NLP/LLM": {
            "NLP基础": {"自评": 4, "说明": "掌握分词、词向量等"},
            "大语言模型": {"自评": 4, "说明": "理解GPT/BERT原理"},
            "RAG系统": {"自评": 4, "说明": "能独立构建RAG系统"},
            "Prompt Engineering": {"自评": 4, "说明": "掌握多种Prompt技巧"},
            "模型微调": {"自评": 3, "说明": "能进行LoRA微调"},
            "AI Agent": {"自评": 3, "说明": "了解Agent架构和开发"}
        },
        "工程能力": {
            "系统设计": {"自评": 4, "说明": "有后端架构设计经验"},
            "模型部署": {"自评": 3, "说明": "能使用FastAPI/Docker部署"},
            "MLOps": {"自评": 2, "说明": "了解但实践经验有限"},
            "性能优化": {"自评": 3, "说明": "了解常用优化方法"}
        }
    }

    total_score = 0
    max_score = 0
    category_scores = {}

    for category, skills in assessment.items():
        cat_score = 0
        cat_max = len(skills) * 5
        print(f"\n【{category}】")
        for skill, details in skills.items():
            score = details["自评"]
            cat_score += score
            bar = "★" * score + "☆" * (5 - score)
            print(f"  {skill:20s} {bar} ({score}/5) - {details['说明']}")

        avg = cat_score / len(skills)
        category_scores[category] = round(avg, 2)
        total_score += cat_score
        max_score += cat_max
        print(f"  平均: {avg:.1f}/5")

    # 总体评估
    overall_avg = total_score / max_score * 5
    print(f"\n{'='*40}")
    print(f"  总体平均: {overall_avg:.2f}/5")

    # 评估等级
    if overall_avg >= 4.0:
        level = "优秀 - 已做好求职准备"
    elif overall_avg >= 3.5:
        level = "良好 - 大部分技能已掌握, 需要针对性提升"
    elif overall_avg >= 3.0:
        level = "合格 - 基础扎实, 需要更多实践"
    else:
        level = "需加强 - 部分领域需要补强"

    print(f"  评估等级: {level}")

    # 薄弱点
    weak_areas = [k for k, v in sorted(category_scores.items(), key=lambda x: x[1]) if v < 3.5]
    if weak_areas:
        print(f"\n  重点关注: {', '.join(weak_areas)}")


# =============================================
# 3. 知识图谱
# =============================================
def generate_knowledge_graph():
    """生成知识图谱"""
    print_section("AI工程师知识图谱")

    graph = {
        "核心": "AI应用工程师",
        "分支": {
            "编程基础": {
                "子节点": ["Python", "数据结构", "算法", "Git"],
                "掌握度": "强"
            },
            "数学基础": {
                "子节点": ["线性代数", "概率统计", "微积分", "最优化"],
                "掌握度": "中"
            },
            "机器学习": {
                "子节点": ["监督学习", "无监督学习", "模型评估", "特征工程"],
                "掌握度": "中强"
            },
            "深度学习": {
                "子节点": ["神经网络", "CNN", "RNN", "Transformer", "PyTorch"],
                "掌握度": "强"
            },
            "NLP": {
                "子节点": ["文本处理", "词向量", "文本分类", "序列标注", "文本生成"],
                "掌握度": "中强"
            },
            "LLM应用": {
                "子节点": ["Transformers", "RAG", "Agent", "Prompt Engineering", "微调"],
                "掌握度": "强"
            },
            "工程实践": {
                "子节点": ["系统设计", "模型部署", "Docker", "API设计", "监控"],
                "掌握度": "中强"
            },
            "软技能": {
                "子节点": ["技术写作", "开源贡献", "沟通协作", "项目管理"],
                "掌握度": "中"
            }
        }
    }

    print(f"\n  中心: {graph['核心']}")
    print(f"\n  分支:")
    for branch, details in graph["分支"].items():
        nodes = ", ".join(details["子节点"])
        print(f"    {branch} ({details['掌握度']})")
        print(f"      -> {nodes}")

    # 知识关联
    print("\n--- 关键知识关联 ---")
    connections = [
        "Python + NumPy -> 数据处理 -> ML特征工程",
        "线性代数 + 概率统计 -> 机器学习 -> 深度学习",
        "Transformer + Self-Attention -> BERT/GPT -> LLM应用",
        "LLM + 检索 -> RAG系统 -> 企业级AI应用",
        "LLM + 工具调用 -> AI Agent -> 自主决策系统",
        "系统设计 + AI应用 -> 生产级AI系统 -> MLOps"
    ]
    for conn in connections:
        print(f"  {conn}")

    # 保存知识图谱
    output_file = os.path.join(os.path.dirname(__file__), "knowledge_graph.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
    print(f"\n  知识图谱已保存: {output_file}")


# =============================================
# 主程序
# =============================================
if __name__ == "__main__":
    print("=" * 60)
    print("  W39 Day 1 - 学习回顾")
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    generate_knowledge_summary()
    self_assessment()
    generate_knowledge_graph()

    print("\n" + "=" * 60)
    print("  39周的学习是一个了不起的成就!")
    print("  回顾不是为了结束, 而是为了更好地出发!")
    print("=" * 60)
