"""
W39 Day 5 - AI开发工具箱
========================
主题: 常用工具清单, 效率工具, 学习资源列表
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
# 1. 常用工具清单
# =============================================
def list_ai_tools():
    """列出AI开发常用工具"""
    print_section("AI开发常用工具清单")

    tools = {
        "开发环境": {
            "IDE/编辑器": [
                {"名称": "VS Code", "用途": "主力IDE, 插件丰富", "推荐插件": ["Python", "Pylance", "Jupyter", "GitLens"]},
                {"名称": "PyCharm Professional", "用途": "Python专业IDE, 调试强", "特点": "智能补全, 重构"},
                {"名称": "Jupyter Notebook/Lab", "用途": "交互式开发, 数据探索", "特点": "可视化友好"},
                {"名称": "Google Colab", "用途": "免费GPU, 快速实验", "特点": "无需本地环境"}
            ],
            "版本控制": [
                {"名称": "Git", "用途": "版本控制"},
                {"名称": "GitHub", "用途": "代码托管, 协作"},
                {"名称": "GitHub CLI (gh)", "用途": "命令行操作GitHub"}
            ]
        },
        "AI/ML框架": {
            "深度学习框架": [
                {"名称": "PyTorch", "用途": "深度学习训练和推理", "特点": "动态图, 灵活"},
                {"名称": "TensorFlow", "用途": "深度学习, 部署", "特点": "生态完善, 生产友好"},
                {"名称": "Hugging Face Transformers", "用途": "预训练模型库", "特点": "10万+模型, NLP必备"},
                {"名称": "PEFT", "用途": "参数高效微调", "特点": "LoRA, QLoRA, Prefix Tuning"}
            ],
            "LLM应用框架": [
                {"名称": "LangChain", "用途": "LLM应用开发框架", "特点": "Chain, Agent, Tool"},
                {"名称": "LlamaIndex", "用途": "RAG框架", "特点": "数据连接, 索引"},
                {"名称": "LangGraph", "用途": "Agent工作流", "特点": "状态机, 多Agent"},
                {"名称": "CrewAI / AutoGen", "用途": "Multi-Agent框架", "特点": "角色协作"}
            ],
            "数据处理": [
                {"名称": "NumPy", "用途": "数值计算", "特点": "矩阵运算, 高效"},
                {"名称": "Pandas", "用途": "数据分析", "特点": "DataFrame, 数据处理"},
                {"名称": "Polars", "用途": "高性能DataFrame", "特点": "比Pandas快5-10倍"},
                {"名称": "scikit-learn", "用途": "传统ML", "特点": "算法丰富, 易用"}
            ]
        },
        "向量数据库": [
            {"名称": "ChromaDB", "用途": "轻量向量数据库", "特点": "本地开发, 简单易用"},
            {"名称": "Milvus", "用途": "生产级向量数据库", "特点": "可扩展, 高性能"},
            {"名称": "Weaviate", "用途": "向量搜索引擎", "特点": "内置模块化"},
            {"名称": "Pinecone", "用途": "云向量数据库", "特点": "全托管, 零运维"},
            {"名称": "FAISS", "用途": "向量相似度搜索库", "特点": "Facebook开源, 高效"}
        ],
        "模型推理": [
            {"名称": "vLLM", "用途": "高吞吐LLM推理", "特点": "PagedAttention, 快"},
            {"名称": "TGI (Text Generation Inference)", "用途": "HuggingFace推理服务", "特点": "生产级, 易部署"},
            {"名称": "Ollama", "用途": "本地LLM运行", "特点": "一行命令运行模型"},
            {"名称": "llama.cpp", "用途": "C++ LLM推理", "特点": "CPU友好, GGUF格式"},
            {"名称": "ONNX Runtime", "用途": "跨平台推理", "特点": "通用, 可优化"}
        ],
        "实验管理": [
            {"名称": "MLflow", "用途": "ML实验管理", "特点": "开源, 全流程"},
            {"名称": "Weights & Biases", "用途": "实验追踪, 可视化", "特点": "功能强大, 有免费版"},
            {"名称": "TensorBoard", "用途": "训练可视化", "特点": "PyTorch/TF集成"},
            {"名称": "ClearML", "用途": "MLOps平台", "特点": "开源, 全功能"}
        ]
    }

    for category, sub_categories in tools.items():
        print(f"\n【{category}】")
        if isinstance(sub_categories, list):
            for tool in sub_categories:
                if isinstance(tool, dict):
                    name = tool.get("名称", "")
                    purpose = tool.get("用途", "")
                    feature = tool.get("特点", "")
                    print(f"  {name}: {purpose} ({feature})")
        elif isinstance(sub_categories, dict):
            for sub_cat, items in sub_categories.items():
                print(f"\n  {sub_cat}:")
                for item in items:
                    if isinstance(item, dict):
                        name = item.get("名称", "")
                        purpose = item.get("用途", "")
                        feature = item.get("特点", "")
                        print(f"    {name}: {purpose} - {feature}")


# =============================================
# 2. 效率工具
# =============================================
def list_efficiency_tools():
    """列出效率工具"""
    print_section("效率工具")

    efficiency_tools = {
        "AI辅助编程": [
            {"名称": "GitHub Copilot", "用途": "AI代码补全", "推荐": "★★★★★"},
            {"名称": "Claude Code", "用途": "AI编程助手", "推荐": "★★★★★"},
            {"名称": "Cursor", "用途": "AI IDE", "推荐": "★★★★☆"},
            {"名称": "Codeium", "用途": "免费AI补全", "推荐": "★★★★☆"}
        ],
        "文档与写作": [
            {"名称": "Notion", "用途": "知识管理, 文档", "推荐": "★★★★☆"},
            {"名称": "Obsidian", "用途": "Markdown笔记, 双链", "推���": "★★★★☆"},
            {"名称": "Typora", "用途": "Markdown编辑器", "推荐": "★★★★★"},
            {"名称": "Excalidraw", "用途": "手绘风格图表", "推荐": "★★★★☆"}
        ],
        "命令行工具": [
            {"名称": "tmux", "用途": "终端复用", "推荐": "★★★★★"},
            {"名称": "fzf", "用途": "模糊搜索", "推荐": "★★★★☆"},
            {"名称": "ripgrep (rg)", "用途": "快速文本搜索", "推荐": "★★★★★"},
            {"名称": "bat", "用途": "增强版cat", "推荐": "★★★★☆"},
            {"名称": "htop/btop", "用途": "系统监控", "推荐": "★★★★☆"}
        ],
        "浏览器插件": [
            {"名称": "Monica", "用途": "AI助手", "推荐": "★★★★☆"},
            {"名称": "JSON Viewer", "用途": "JSON格式化", "推荐": "★★★★☆"},
            {"名称": "React Developer Tools", "用途": "前端调试", "推荐": "★★★★☆"}
        ]
    }

    for category, items in efficiency_tools.items():
        print(f"\n【{category}】")
        for item in items:
            print(f"  {item['名称']}: {item['用途']} {item['推荐']}")


# =============================================
# 3. 学习资源列表
# =============================================
def list_learning_resources():
    """列出学习资源"""
    print_section("持续学习资源列表")

    resources = {
        "每日必读": [
            "Hugging Face Blog - AI最新技术和模型",
            "Twitter/X AI列表 - 研究者和工程师的分享",
            "arXiv Daily - 最新论文摘要",
            "Papers With Code - 论文+代码+基准"
        ],
        "每周学习": [
            "一篇深度技术博客(阅读+笔记)",
            "一个开源项目源码阅读",
            "一个新技术/工具尝试",
            "一篇论文精读"
        ],
        "进阶资源": [
            "Stanford CS224N (NLP) - 课程视频和笔记",
            "Stanford CS336 (Language Modeling from Scratch) - 最新",
            "Andrej Karpathy YouTube - 神经网络教学",
            "fast.ai - 实战深度学习",
            "Lex Fridman Podcast - AI深度访谈"
        ],
        "中文资源": [
            "机器之心 - AI新闻和深度文章",
            "量子位 - AI行业动态",
            "AI科技评论 - 技术分析",
            "李沐动手学深度学习 - 中文DL教材",
            "Hugging Face中文社区"
        ]
    }

    for category, items in resources.items():
        print(f"\n【{category}】")
        for item in items:
            print(f"  - {item}")

    # 保存工具清单
    output_file = os.path.join(os.path.dirname(__file__), "ai_toolkit.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"tools": tools if 'tools' in dir() else {},
                    "efficiency_tools": efficiency_tools}, f, ensure_ascii=False, indent=2)
    print(f"\n  工具清单已保存: {output_file}")


# =============================================
# 主程序
# =============================================
if __name__ == "__main__":
    print("=" * 60)
    print("  W39 Day 5 - AI开发工具箱")
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    list_ai_tools()
    list_efficiency_tools()
    list_learning_resources()

    print("\n" + "=" * 60)
    print("  工欲善其事, 必先利其器!")
    print("  建议: 尝试每个分类中的至少一个新工具")
    print("=" * 60)
