"""
W37 Day 7 - 模拟面试
====================
主题: 自我介绍模板, 项目介绍模板, 反问准备
"""

import os
from datetime import datetime


def print_section(title):
    """打印分隔线"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# =============================================
# 1. 自我介绍模板
# =============================================
def self_introduction_templates():
    """自我介绍模板"""
    print_section("自我介绍模板(1-3分钟)")

    templates = {
        "1分钟版本(简短版)": """
面试官您好, 我叫张三, 有5年Java后端和1年AI应用开发经验。

最近一年专注于LLM应用开发, 主导了公司RAG知识库问答系统和AI Agent工作流的设计与实现。
检索准确率从72%提升至91%, 客服工作量减少50%。

技术栈方面, 我熟练使用Python、LangChain、Transformers等AI工具,
同时具备Java后端架构和DevOps经验。

我希望能加入贵公司, 在AI应用开发领域继续深耕, 为业务创造价值。
谢谢!
""",
        "3分钟版本(标准版)": """
面试官您好, 感谢给我这次面试机会。我叫张三, 简单介绍一下我的背景和经验。

【工作背景】
我有5年Java后端开发经验, 在XX公司负责核心业务系统的开发和架构设计。
近一年开始转型AI应用开发, 目前主要负责公司AI相关项目的落地。

【核心项目】
我最引以为豪的是主导构建了公司的RAG知识库问答系统。
这个系统解决了客服团队每天处理大量重复问题的人力瓶颈。

技术方案上, 我选型了LangChain + ChromaDB构建检索系统,
设计了混合检索策略(向量检索+BM25), 检索准确率从72%提升至91%。
使用GPT-4作为生成模型, 设计了带引用的Prompt模板, 减少幻觉。

上线后, 系统每月处理10万+查询, 自动解决85%的常见问题,
客服工作量减少50%, 用户满意度从65%提升至89%。

此外, 我还搭建了公司的模型微调平台, 支持LoRA微调和量化部署,
将微调时间从2周缩短至2天, 推理成本降低60%。

【技术能力】
AI方面, 我熟悉LangChain、Transformers、PEFT等主流框架,
有RAG系统、Agent开发、模型微调和部署的实战经验。
后端方面, 我有扎实的系统设计和工程化能力。

【个人特点】
我是一个持续学习的人, 维护着个人技术博客(50+篇文章),
有开源项目贡献经历, 也是AI社区的活跃参与者。

我对贵公司的XX产品/方向非常感兴趣, 相信我的后端架构经验
和AI应用开发能力能为团队带来价值。

谢谢, 以上是我的自我介绍。
""",
        "英文版本": """
Good morning/afternoon. Thank you for the opportunity to interview today.
My name is Zhang San.

I have 5 years of Java backend experience and 1 year of AI application development.
I'm currently focused on building LLM-powered applications, particularly RAG systems
and AI Agents.

My most significant project was building an enterprise RAG knowledge base system.
I designed a hybrid retrieval strategy combining vector search and BM25,
achieving 91% retrieval accuracy, up from 72%.
The system now handles 100K+ queries per month, reducing customer service workload by 50%.

My tech stack includes Python, LangChain, Transformers, PyTorch,
along with solid backend engineering skills in Java, Docker, and Kubernetes.

I also maintain a technical blog with 50+ articles and contribute to open-source projects.

I'm excited about this opportunity because [company-specific reason].

Thank you. I'm happy to answer any questions.
"""
    }

    for name, template in templates.items():
        print(f"\n【{name}】")
        print(template.strip())

    # 自我介绍要点
    print("\n--- 自我介绍注意事项 ---")
    tips = [
        "控制时间: 1分钟版用于快速介绍, 3分钟版用于正式面试",
        "突出亮点: 用数据和成果说话, 不要流水账",
        "匹配JD: 根据职位描述调整自我介绍的重点",
        "练习: 对着镜子或录音练习至少5遍",
        "自然: 不要背稿, 像讲故事一样自然"
    ]
    for tip in tips:
        print(f"  {tip}")


# =============================================
# 2. 项目介绍模板
# =============================================
def project_introduction_templates():
    """项目介绍模板"""
    print_section("项目介绍模板(STAR法则)")

    projects = {
        "项目1: RAG知识库系统": {
            "一句话概括": "我主导构建了企业级RAG知识库问答系统, 实现了85%常见问题自动解答",
            "背景(S)": "公司客服团队每天处理5000+重复问题, 响应慢、人力成本高",
            "任务(T)": "构建智能问答系统, 自动回答80%以上的常见问题",
            "行动(A)": [
                "技术选型: 调研后选择LangChain + ChromaDB + GPT-4方案",
                "数据处理: 设计多格式文档解析和语义分块策略",
                "检索优化: 实现混合检索(向量+BM25)+Cross-Encoder重排",
                "生成优化: 设计带引用的Prompt模板, 减少幻觉",
                "工程化: 使用FastAPI + Docker + Redis构建高可用服务"
            ],
            "结果(R)": [
                "检索准确率: 72% -> 91%",
                "自动解决率: 85%的常见问题",
                "工作量减少: 50%",
                "用户满意度: 65% -> 89%",
                "月处理: 10万+查询"
            ],
            "难点与挑战": "最大的挑战是检索质量优化。初始方案用纯向量检索只有72%准确率, "
                          "通过引入BM25混合检索和重排序解决了这个问题",
            "如果重来": "会更早引入评估框架, 用数据驱动优化而不是凭感觉"
        },
        "项目2: 模型微调平台": {
            "一句话概括": "搭建了统一的LoRA微调平台, 将微调时间从2周缩短至2天",
            "背景(S)": "5个业务团队都需要微调大模型, 缺乏统一工具和流程",
            "任务(T)": "搭建统一平台, 降低微调门槛, 提高效率",
            "行动(A)": [
                "设计LoRA微调流水线, 封装复杂流程",
                "实现INT8/INT4量化推理",
                "开发自动评估和对比功能",
                "搭建vLLM推理服务"
            ],
            "结果(R)": [
                "微调时间: 2周 -> 2天",
                "推理成本: 降低60%",
                "支持: 10+种模型, 5个团队"
            ]
        }
    }

    for name, details in projects.items():
        print(f"\n【{name}】")
        print(f"  一句话: {details['一句话概括']}")
        print(f"  S(背景): {details['背景(S)']}")
        print(f"  T(任务): {details['任务(T)']}")
        print(f"  A(行动):")
        for action in details['行动(A)']:
            print(f"    - {action}")
        print(f"  R(结果):")
        for result in details['结果(R)']:
            print(f"    - {result}")
        if '难点与挑战' in details:
            print(f"  难点: {details['难点与挑战']}")
        if '如果重来' in details:
            print(f"  改进: {details['如果重来']}")

    # 面试话术技巧
    print("\n--- 面试话术技巧 ---")
    techniques = [
        "先说结论: '这个项目的核心成果是...'",
        "用数据说话: '准确率从X%提升到Y%'",
        "展示思考: '我选择这个方案是因为...'",
        "承认不足: '如果重来, 我会...' (展示学习能力)",
        "关联JD: '这个经验与贵公司的XX需求很匹配'"
    ]
    for t in techniques:
        print(f"  {t}")


# =============================================
# 3. 反问准备
# =============================================
def prepare_counter_questions():
    """准备反问问题"""
    print_section("反问面试官的问题")

    questions = {
        "关于团队": [
            "团队的规模和分工是怎样的?",
            "团队目前使用的AI技术栈是什么?",
            "团队在做什么样的AI项目? 最大的挑战是什么?",
            "团队有多少AI工程师? 技术氛围如何?"
        ],
        "关于技术": [
            "公司目前在使用哪些LLM? 是自己训练还是调用API?",
            "AI应用的架构是怎样的? 使用什么基础设施?",
            "团队如何评估AI应用的效果?",
            "技术栈的选择是固定的还是可以提议新技术?"
        ],
        "关于工作": [
            "这个岗位日常的工作内容是什么?",
            "AI应用的落地过程中, 最大的挑战是什么?",
            "一个典型的AI项目周期是多长?",
            "团队如何平衡技术债务和新功能开发?"
        ],
        "关于成长": [
            "公司对AI工程师的成长路径是怎样的?",
            "有没有技术分享、培训或参加技术会议的机会?",
            "团队如何保持技术更新?",
            "新人入职后的onboarding流程是怎样的?"
        ],
        "关于业务": [
            "AI在公司业务中扮演什么角色?",
            "AI应用的用户是谁? 用户反馈如何?",
            "未来6-12个月AI方向有什么规划?"
        ],
        "关于面试流程": [
            "后续的面试流程是怎样的?",
            "大概什么时候会有结果反馈?",
            "还有什么需要我补充的信息吗?"
        ]
    }

    for category, qs in questions.items():
        print(f"\n【{category}】")
        for q in qs:
            print(f"  - {q}")

    # 反问技巧
    print("\n--- 反问技巧 ---")
    tips = [
        "准备5-8个问题, 根据面试氛围选择3-4个提问",
        "展示你的技术深度: 问一些有深度的问题",
        "展示你的兴趣: 问关于团队和项目的问题",
        "不要问: 薪资(等HR谈)、太基础的问题(网上能查到的)",
        "认真倾听回答, 可以追问展示你在思考"
    ]
    for tip in tips:
        print(f"  {tip}")


# =============================================
# 4. 面试策略总结
# =============================================
def interview_strategy_summary():
    """面试策略总结"""
    print_section("面试策略总结与W37回顾")

    strategy = {
        "面试前准备": [
            "研究公司: 产品、技术栈、最近动态",
            "准备3个项目(STAR法则, 每个准备3个深度问题)",
            "复习技术基础: ML概念、手撕代码、系统设计",
            "准备反问问题(5-8个)",
            "准备好自我介绍(1分钟和3分钟版本)"
        ],
        "面试中注意": [
            "先思考再回答: 可以说'让我想一下'",
            "不懂就问: 需求不明确时主动澄清",
            "展示思考过程: 边想边说, 让面试官看到你的思路",
            "坦诚: 不会的说不会, 但可以说相关经验",
            "积极: 保持热情和好奇心"
        ],
        "面试后跟进": [
            "记录面试问题, 用于后续复盘",
            "24小时内发感谢邮件",
            "复盘表现, 改进不足",
            "不要因一次面试结果影响心态"
        ],
        "W37学习总结": [
            "Day 1: 简历生成器 - STAR法则、量化经验",
            "Day 2: 个人网站 - HTML模板、项目展示",
            "Day 3: 面试题库 - 50题覆盖AI/ML/系统设计",
            "Day 4: ML面试题 - 概念、手撕代码、数学推导",
            "Day 5: 系统设计 - 推荐/RAG/ML系统设计",
            "Day 6: 编程练习 - 10道LeetCode经典题",
            "Day 7: 模拟面试 - 自我介绍、项目介绍、反问"
        ]
    }

    for category, items in strategy.items():
        print(f"\n【{category}】")
        for item in items:
            print(f"  {item}")


# =============================================
# 主程序
# =============================================
if __name__ == "__main__":
    print("=" * 60)
    print("  W37 Day 7 - 模拟面试")
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    self_introduction_templates()
    project_introduction_templates()
    prepare_counter_questions()
    interview_strategy_summary()

    print("\n" + "=" * 60)
    print("  W37周学习完成! 求职准备是一个系统工程!")
    print("  建议: 找朋友做模拟面试, 录音回听改进")
    print("=" * 60)
