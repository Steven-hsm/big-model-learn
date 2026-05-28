"""
W37 Day 1 - 简历生成器
======================
主题: AI工程师简历模板, 项目经历量化, STAR法则示例
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
# 1. AI工程师简历模板
# =============================================
def generate_resume_template():
    """生成AI工程师简历模板"""
    print_section("AI工程师简历模板")

    resume = {
        "��人信息": {
            "姓名": "张三",
            "电话": "138-xxxx-xxxx",
            "邮箱": "zhangsan@email.com",
            "GitHub": "github.com/zhangsan",
            "博客": "zhangsan.dev",
            "LinkedIn": "linkedin.com/in/zhangsan",
            "所在地": "北京/上海/深圳"
        },
        "求职意向": {
            "职位": "AI应用工程师 / LLM应用开发工程师",
            "期望薪资": "面议",
            "到岗时间": "一个月内"
        },
        "个人简介": (
            "5年Java后端开发经验 + 1年AI应用开发经验的工程师, "
            "专注于LLM应用开发和RAG系统构建。"
            "熟悉大模型微调、Agent开发和MLOps流程, "
            "具备将AI能力集成到企业级系统的丰富经验。"
        ),
        "专业技能": {
            "编程语言": ["Python(熟练)", "Java(精通)", "SQL(熟练)", "Shell(熟悉)"],
            "AI/ML": [
                "大模型应用: LangChain, LlamaIndex, Transformers",
                "模型微调: LoRA, QLoRA, PEFT",
                "向量数据库: ChromaDB, Milvus, Weaviate",
                "ML框架: PyTorch, scikit-learn",
                "评估: RAGAS, BLEU, ROUGE"
            ],
            "后端开发": [
                "框架: Spring Boot, FastAPI, Flask",
                "数据库: MySQL, PostgreSQL, Redis, MongoDB",
                "消息队列: Kafka, RabbitMQ",
                "微服务: Docker, Kubernetes, gRPC"
            ],
            "DevOps": [
                "CI/CD: GitHub Actions, Jenkins",
                "云平台: AWS, 阿里云",
                "监控: Prometheus, Grafana"
            ]
        },
        "工作经历": [
            {
                "公司": "XX科技有限公司",
                "职位": "高级Java开发工程师 / AI应用开发",
                "时间": "2021.06 - 至今",
                "描述": [
                    "负责公司核心业务系统的开发和维护, 日活用户100万+",
                    "主导AI能力接入, 构建基于RAG的智能客服系统",
                    "设计并实现LLM应用的评估和监控框架"
                ]
            }
        ],
        "项目经历": [
            {
                "名称": "企业级RAG知识库问答系统",
                "时间": "2025.09 - 2025.12",
                "技术栈": "Python, LangChain, ChromaDB, FastAPI, Docker",
                "描述": [
                    "设计并实现支持多格式文档(PDF/Word/Markdown)的知识库系统",
                    "采用混合检索(向量+BM25)策略, 检索准确率从72%提升至91%",
                    "实现流式响应和会话记忆, 平均响应时间1.2秒",
                    "部署后每月处理10万+查询, 用户满意度从65%提升至89%"
                ]
            },
            {
                "名称": "大模型微调与部署平台",
                "时间": "2025.06 - 2025.08",
                "技术栈": "PyTorch, Transformers, LoRA, vLLM, Kubernetes",
                "描述": [
                    "搭建LoRA微调流水线, 支持10+种开源模型的一键微调",
                    "实现INT8/INT4量化, 推理速度提升3倍, 显存减少50%",
                    "构建模型评估框架, 自动化评测多个维度指标",
                    "平台支持5个业务团队的模型微调需求"
                ]
            }
        ],
        "教育背景": {
            "学校": "XX大学",
            "专业": "计算机科学与技术",
            "学历": "本科",
            "时间": "2013.09 - 2017.06"
        },
        "开源贡献": [
            "Hugging Face Transformers: 贡献3个PR(文档翻译+Bug修复)",
            "LangChain: 贡献2个PR(新增集成组件)",
            "个人开源项目 rag-toolkit: GitHub 200+ stars"
        ],
        "技术博客": [
            "个人技术博客累计50+篇文章, 月阅读量10000+",
            "知乎AI话题优秀回答者, 专栏关注者5000+"
        ]
    }

    print(json.dumps(resume, ensure_ascii=False, indent=2))

    # 保存简历模板
    output_file = os.path.join(os.path.dirname(__file__), "resume_template.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(resume, f, ensure_ascii=False, indent=2)
    print(f"\n  简历模板已保存: {output_file}")


# =============================================
# 2. 项目经历量化
# =============================================
def demonstrate_project_quantification():
    """展示项目经历量化方法"""
    print_section("项目经历量化方法")

    print("\n--- 量化公式 ---")
    print("  成果 = 动作 + 数据 + 影响")
    print("  例: 使用[技术]优化[模块], [指标]从X提升到Y, 提升Z%")

    quantification_examples = {
        "性能优化": {
            "弱描述": "优化了系统性能",
            "强描述": "通过引入Redis缓存和SQL优化, "
                      "API响应时间从800ms降至120ms, QPS从200提升至1500",
            "量化指标": ["响应时间", "QPS/TPS", "CPU/内存使用率", "延迟P99"]
        },
        "AI模型": {
            "弱描述": "训练了一个模型, 效果不错",
            "强描述": "使用LoRA微调LLaMA-7B模型, "
                      "在垂直领域问答任务上准确率从72%提升至91%, "
                      "推理延迟控制在1.2秒以内",
            "量化指标": ["准确率/召回率", "F1 Score", "推理延迟", "BLEU/ROUGE"]
        },
        "系统设计": {
            "弱描述": "设计了一个推荐系统",
            "强描述": "设计并实现基于协同过滤+深度学习的推荐系统, "
                      "日处理1000万+请求, CTR提升15%, GMV增长8%",
            "量化指标": ["日活/月活", "转化率", "GMV/收入", "用户留存"]
        },
        "项目管理": {
            "弱描述": "带领团队完成了项目",
            "强描述": "带领5人团队, 3个月内从0到1完成AI客服系统, "
                      "上线后减少50%人工客服工作量, 客户满意度提升20%",
            "量化指标": ["团队规模", "项目周期", "成本节省", "效率提升"]
        }
    }

    for category, examples in quantification_examples.items():
        print(f"\n【{category}】")
        print(f"  弱: {examples['弱描述']}")
        print(f"  强: {examples['强描述']}")
        print(f"  可量化指标: {', '.join(examples['量化指标'])}")


# =============================================
# 3. STAR法则示例
# =============================================
def demonstrate_star_method():
    """展示STAR法则应用"""
    print_section("STAR法则: 项目描述技巧")

    print("\n  STAR = Situation(情境) + Task(任务) + Action(行动) + Result(结果)")

    star_examples = [
        {
            "项目": "RAG知识库系统",
            "S(情境)": "公司客服团队每天处理5000+重复问题, 人力成本高, 响应慢",
            "T(任务)": "构建智能问答系统, 自动回答80%以上的常见问题",
            "A(行动)": [
                "调研RAG技术方案, 选型LangChain + ChromaDB",
                "设计文档解析流水线, 支持PDF/Word/Markdown",
                "实现混合检索(向量+BM25), 优化检索质量",
                "使用GPT-4作为生成模型, 设计Prompt模板",
                "构建评估框架, 持续优化系统效果"
            ],
            "R(结果)": [
                "检索准确率从72%提升至91%",
                "自动解决85%的常见问题",
                "客服团队工作量减少50%",
                "用户满意度从65%提升至89%"
            ]
        },
        {
            "项目": "大模型微调平台",
            "S(情境)": "5个业务团队都需要微调大模型, 但缺乏统一工具和流程",
            "T(任务)": "搭建统一的模型微调平台, 降低微调门槛, 提高效率",
            "A(行动)": [
                "设计LoRA微调流水线, 封装复杂流程",
                "实现配置化的训练参数管理",
                "开发自动评估和对比功能",
                "搭建vLLM推理服务, 支持快速部署"
            ],
            "R(结果)": [
                "微调时间从2周缩短至2天",
                "支持10+种开源模型的一键微调",
                "5个业务团队全部接入平台",
                "推理成本降低60%(量化+批处理)"
            ]
        },
        {
            "项目": "AI Agent工作流",
            "S(情境)": "数据分析团队需要手动处理大量报告, 耗时且容易出错",
            "T(任务)": "构建AI Agent自动完成数据分析和报告生成",
            "A(行动)": [
                "设计ReAct模式的Agent架构",
                "实现工具调用(Python代码执行、数据库查询、图表生成)",
                "添加记忆系统支持多轮交互",
                "构建人机协作的工作流"
            ],
            "R(结果)": [
                "报告生成时间从4小时缩短至30分钟",
                "数据准确性提升至99%",
                "每月节省200+人工小时",
                "团队可专注于高级分析任务"
            ]
        }
    ]

    for i, example in enumerate(star_examples, 1):
        print(f"\n--- 示例{i}: {example['项目']} ---")
        print(f"  S(情境): {example['S(情境)']}")
        print(f"  T(任务): {example['T(任务)']}")
        print(f"  A(行动):")
        for action in example["A(行动)"]:
            print(f"    - {action}")
        print(f"  R(结果):")
        for result in example["R(结果)"]:
            print(f"    - {result}")


# =============================================
# 4. 简历优化建议
# =============================================
def resume_optimization_tips():
    """简历优化建议"""
    print_section("简历优化建议")

    tips = {
        "AI岗位简历特别注意事项": [
            "突出AI相关的项目经历(放在最前面)",
            "列出使用的具体AI工具和框架",
            "量化模型效果和业务影响",
            "展示开源贡献和技术博客",
            "附上GitHub和Demo链接"
        ],
        "常见错误": [
            "只列技术栈, 不描述项目细节",
            "用'参与'、'了解'等弱动词",
            "缺少量化数据",
            "简历超过2页",
            "格式混乱, 排版不专业"
        ],
        "加分项": [
            "开源项目贡献(附链接)",
            "技术博客(附链接)",
            "Kaggle竞赛成绩",
            "论文发表(如有)",
            "技术演讲/分享经历",
            "GitHub star数高的项目"
        ],
        "投递策略": {
            "针对性": "根据JD调整简历关键词",
            "关键词": "确保包含JD中的技能关键词(ATS系统友好)",
            "格式": "PDF格式(防止格式错乱)",
            "命名": "姓名_职位_工作年限_简历.pdf"
        }
    }

    for category, items in tips.items():
        print(f"\n【{category}】")
        if isinstance(items, list):
            for item in items:
                print(f"  {item}")
        elif isinstance(items, dict):
            for key, value in items.items():
                print(f"  {key}: {value}")


# =============================================
# 主程序
# =============================================
if __name__ == "__main__":
    print("=" * 60)
    print("  W37 Day 1 - 简历生成器")
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    generate_resume_template()
    demonstrate_project_quantification()
    demonstrate_star_method()
    resume_optimization_tips()

    print("\n" + "=" * 60)
    print("  简历是求职的第一关, 用数据和结果说话!")
    print("  建议: 用STAR法则重写每条项目经历")
    print("=" * 60)
