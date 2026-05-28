"""
W38 Day 3 - 技术栈回顾
=====================
主题: Python/AI技术知识体系梳理, 薄弱点识别
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
# 1. Python技术知识体系
# =============================================
def review_python_knowledge():
    """回顾Python技术知识体系"""
    print_section("Python技术知识体系")

    knowledge = {
        "Python基础": {
            "数据类型": {
                "掌握程度": "★★★★★",
                "知识点": ["list, dict, set, tuple", "字符串操作", "f-string格式化",
                           "可变/不可变类型", "深拷贝/浅拷贝"]
            },
            "函数与装饰器": {
                "掌握程度": "★★★★☆",
                "知识点": ["*args, **kwargs", "闭包", "装饰器", "生成器(yield)",
                           "lambda表达式", "函数式编程(map/filter/reduce)"]
            },
            "面向对象": {
                "掌握程度": "★★★★☆",
                "知识点": ["类和继承", "多态", "魔术方法(__init__, __repr__等)",
                           "property装饰器", "抽象类", "数据类(dataclass)"]
            },
            "异步编程": {
                "掌握程度": "★★★☆☆",
                "知识点": ["async/await", "asyncio", "aiohttp", "异步上下文管理器",
                           "异步生成器"]
            }
        },
        "Python进阶": {
            "类型系统": {
                "掌握程度": "★★★☆☆",
                "知识点": ["type hints", "typing模块", "Generic类型", "Protocol",
                           "mypy静态检查"]
            },
            "并发编程": {
                "掌握程度": "★★★☆☆",
                "知识点": ["threading", "multiprocessing", "concurrent.futures",
                           "GIL理解", "进程池/线程池"]
            },
            "包管理": {
                "掌握程度": "★★★★☆",
                "知识点": ["pip", "venv", "poetry", "pyproject.toml", "setuptools"]
            },
            "性能优化": {
                "掌握程度": "★★★☆☆",
                "知识点": ["cProfile", "列表推导式", "生成器", "functools.lru_cache",
                           "Cython/numba"]
            }
        },
        "Python数据处理": {
            "NumPy": {
                "掌握程度": "★★★★☆",
                "知识点": ["ndarray操作", "广播机制", "向量化计算", "线性代数",
                           "随机数"]
            },
            "Pandas": {
                "掌握程度": "★★★★☆",
                "知识点": ["DataFrame操作", "数据清洗", "分组聚合", "合并连接",
                           "时间序列"]
            },
            "可视化": {
                "掌握程度": "★★★☆☆",
                "知识点": ["Matplotlib", "Seaborn", "Plotly(交互式)",
                           "图表美化技巧"]
            }
        }
    }

    for category, topics in knowledge.items():
        print(f"\n【{category}】")
        for topic, details in topics.items():
            print(f"  {topic} {details['掌握程度']}")
            print(f"    关键知识点: {', '.join(details['知识点'][:4])}...")


# =============================================
# 2. AI技术知识体系
# =============================================
def review_ai_knowledge():
    """回顾AI技术知识体系"""
    print_section("AI技术知识体系")

    knowledge = {
        "机器学习基础": {
            "监督学习": {
                "掌握程度": "★★★★☆",
                "知识点": ["线性回归/逻辑回归", "决策树/随机森林", "SVM",
                           "集成学习(XGBoost/LightGBM)", "交叉验证"]
            },
            "无监督学习": {
                "掌握程度": "★★★☆☆",
                "知识点": ["K-Means", "DBSCAN", "PCA降维", "异常检测",
                           "聚类评估"]
            },
            "评估方法": {
                "掌握程度": "★★★★☆",
                "知识点": ["混淆矩阵", "Precision/Recall/F1", "ROC/AUC",
                           "过拟合/欠拟合", "偏差-方差权衡"]
            }
        },
        "深度学习": {
            "神经网络基础": {
                "掌握程度": "★★★★☆",
                "知识点": ["前向传播/反向传播", "激活函数", "损失函数",
                           "优化器(SGD/Adam)", "正则化"]
            },
            "CNN": {
                "掌握程度": "★★★☆☆",
                "知识点": ["卷积操作", "池化", "ResNet", "图像分类",
                           "目标检测"]
            },
            "RNN/Transformer": {
                "掌握程度": "★★★★☆",
                "知识点": ["LSTM/GRU", "Self-Attention", "多头注意力",
                           "位置编码", "编码器-解码器"]
            },
            "PyTorch": {
                "掌握程度": "★★★★☆",
                "知识点": ["张量操作", "nn.Module", "训练循环", "数据加载",
                           "GPU训练"]
            }
        },
        "NLP": {
            "基础NLP": {
                "掌握程度": "★★★★☆",
                "知识点": ["分词", "词向量(Word2Vec/GloVe)", "文本分类",
                           "命名实体识别", "文本相似度"]
            },
            "大语言模型": {
                "掌握程度": "★★★★☆",
                "知识点": ["GPT系列", "BERT系列", "LLaMA", "微调(LoRA)",
                           "Prompt Engineering"]
            },
            "RAG": {
                "掌握程度": "★★★★☆",
                "知识点": ["向量检索", "文档分块", "Embedding模型",
                           "混合检索", "RAG评估"]
            }
        },
        "AI工程": {
            "模型部署": {
                "掌握程度": "★★★☆☆",
                "知识点": ["FastAPI服务化", "Docker容器化", "模型量化",
                           "vLLM/TGI", "Kubernetes"]
            },
            "MLOps": {
                "掌握程度": "★★★☆☆",
                "知识点": ["MLflow", "实验管理", "模型版本", "CI/CD",
                           "监控告警"]
            },
            "AI Agent": {
                "掌握程度": "★★★☆☆",
                "知识点": ["ReAct模式", "工具调用", "记忆系统",
                           "LangGraph", "Multi-Agent"]
            }
        }
    }

    for category, topics in knowledge.items():
        print(f"\n【{category}】")
        for topic, details in topics.items():
            print(f"  {topic} {details['掌握程度']}")
            for point in details['知识点']:
                print(f"    - {point}")


# =============================================
# 3. 薄弱点识别与提升计划
# =============================================
def identify_weaknesses():
    """识别薄弱点并制定提升计划"""
    print_section("薄弱点识别与提升计划")

    # 自评表
    print("\n--- 技能自评表(1-5分) ---")
    skills = [
        {"技能": "Python编程", "自评": 4, "目标": 5},
        {"技能": "NumPy/Pandas", "自评": 4, "目标": 4},
        {"技能": "PyTorch", "自评": 3, "目标": 4},
        {"技能": "Transformer原理", "自评": 4, "目标": 5},
        {"技能": "NLP基础", "自评": 4, "目标": 4},
        {"技能": "LLM应用开发", "自评": 4, "目标": 5},
        {"技能": "RAG系统", "自评": 4, "目标": 5},
        {"技能": "模型微调", "自评": 3, "目标": 4},
        {"技能": "模型部署", "自评": 3, "目标": 4},
        {"技能": "MLOps", "自评": 2, "目标": 3},
        {"技能": "AI Agent", "自评": 3, "目标": 4},
        {"技能": "系统设计", "自评": 4, "目标": 4},
        {"技能": "数据处理", "自评": 3, "目标": 4},
        {"技能": "Docker/K8s", "自评": 3, "目标": 4},
    ]

    print(f"  {'技能':20s} {'自评':>4s} {'目标':>4s} {'差距':>4s} {'优先级':>6s}")
    print("  " + "-" * 45)

    weak_areas = []
    for skill in skills:
        gap = skill["目标"] - skill["自评"]
        priority = "高" if gap >= 2 else ("中" if gap == 1 else "-")
        print(f"  {skill['技能']:20s} {skill['自评']:>4d} {skill['目标']:>4d} "
              f"{gap:>4d} {priority:>6s}")
        if gap >= 1:
            weak_areas.append({**skill, "差距": gap, "优先级": priority})

    # 提升计划
    print("\n--- 薄弱点提升计划 ---")
    improvement_plans = {
        "MLOps(差距2)": {
            "学习内容": "MLflow实验管理、模型注册、自动化Pipeline",
            "学习方式": "官方教程 + 实践项目",
            "时间": "1周",
            "资源": "MLflow官方文档, Kubeflow教程"
        },
        "模型微调(差距1)": {
            "学习内容": "LoRA/QLoRA微调, 数据准备, 超参调优",
            "学习方式": "Hugging Face教程 + 实际微调项目",
            "时间": "1周",
            "资源": "PEFT文档, HF微调教程"
        },
        "模型部署(差距1)": {
            "学习内容": "vLLM部署, Docker容器化, 量化推理",
            "学习方式": "vLLM文档 + Docker实战",
            "时间": "1周",
            "资源": "vLLM文档, FastAPI部署教程"
        },
        "AI Agent(差距1)": {
            "学习内容": "LangGraph, ReAct模式, Multi-Agent",
            "学习方式": "LangChain教程 + Agent项目",
            "时间": "1周",
            "资源": "LangChain文档, LangGraph教程"
        }
    }

    for area, plan in improvement_plans.items():
        print(f"\n  {area}")
        for key, value in plan.items():
            print(f"    {key}: {value}")


# =============================================
# 主程序
# =============================================
if __name__ == "__main__":
    print("=" * 60)
    print("  W38 Day 3 - 技术栈回顾")
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    review_python_knowledge()
    review_ai_knowledge()
    identify_weaknesses()

    print("\n" + "=" * 60)
    print("  知道自己不知道什么, 比知道自己知道什么更重要!")
    print("  建议: 根据薄弱点提升计划, 有针对性地补强")
    print("=" * 60)
