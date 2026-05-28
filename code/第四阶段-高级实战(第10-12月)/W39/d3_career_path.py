"""
W39 Day 3 - 职业路径
====================
主题: AI工程师成长路径(初级→中级→高级), 关键节点
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
# 1. AI工程师成长路径
# =============================================
def describe_career_path():
    """描述AI工程师成长路径"""
    print_section("AI工程师成长路径")

    career_path = {
        "初级AI应用工程师 (L1-L2, 0-2年AI经验)": {
            "能力要求": [
                "能使用Python和AI框架构建应用",
                "理解ML/DL基础概念",
                "能使用LangChain/Transformers开发LLM应用",
                "能独立完成数据处理和模型调用",
                "编写清晰、可维护的代码"
            ],
            "典型工作": [
                "开发和维护LLM应用(RAG/Chatbot)",
                "数据处理和特征工程",
                "简单的模型微调(LoRA)",
                "编写单元测试和文档"
            ],
            "关键里程碑": [
                "独立完成一个完整的AI应用项目",
                "理解Transformer架构的核心思想",
                "掌握至少一种LLM应用框架(LangChain/Transformers)"
            ],
            "薪资参考": "20-35K/月",
            "晋升标准": "能独立设计和实现中等复杂度的AI应用"
        },
        "中级AI应用工程师 (L3, 2-4年AI经验)": {
            "能力要求": [
                "深入理解LLM和NLP技术",
                "能设计和实现复杂的AI系统(RAG/Agent)",
                "有模型微调和评估经验",
                "系统设计和性能优化能力",
                "能指导初级工程师"
            ],
            "典型工作": [
                "设计和实现生产级AI系统",
                "优化RAG/Agent系统效果和性能",
                "模型选型、微调和部署",
                "技术方案设计和评审",
                "代码Review和技术指导"
            ],
            "关键里程碑": [
                "设计和部署一个生产级AI系统",
                "在某一方向有深度(RAG/Agent/微调)",
                "有技术分享和开源贡献"
            ],
            "薪资参考": "35-55K/月",
            "晋升标准": "能独立负责一个AI方向, 带领小团队"
        },
        "高级AI工程师 (L4-L5, 4-7年AI经验)": {
            "能力要求": [
                "在某一方向有深度专长",
                "大规模AI系统设计经验",
                "技术选型和架构决策能力",
                "跨团队协调和技术影响力",
                "能解决复杂的技术难题"
            ],
            "典型工作": [
                "AI平台架构设计",
                "技术方向决策和路线图",
                "复杂系统性能优化",
                "跨团队技术协调",
                "技术分享和社区建设"
            ],
            "关键里程碑": [
                "设计并落地大规模AI平台",
                "在行业技术会议分享",
                "有影响力的开源项目或技术文章",
                "培养出优秀的初级/中级工程师"
            ],
            "薪资参考": "55-90K/月",
            "晋升标准": "在技术方向上有行业影响力"
        },
        "资深/专家 (L6+, 7年+)": {
            "能力要求": [
                "行业技术影响力和视野",
                "战略级技术决策",
                "团队建设和组织能力",
                "商业和技术双重理解"
            ],
            "薪资参考": "90K+/月 + 股权"
        }
    }

    for level, details in career_path.items():
        print(f"\n【{level}】")
        for key, value in details.items():
            if isinstance(value, list):
                print(f"  {key}:")
                for item in value:
                    print(f"    - {item}")
            else:
                print(f"  {key}: {value}")


# =============================================
# 2. 关键节点与决策
# =============================================
def describe_key_decisions():
    """描述职业关键节点"""
    print_section("职业关键节点与决策")

    decisions = {
        "节点1: 选择方向(转型初期)": {
            "问题": "AI领域很广, 选择哪个方向?",
            "选项": {
                "LLM应用开发": {
                    "适合": "喜欢做产品、解决实际问题",
                    "前景": "需求大, 应用场景广",
                    "建议": "推荐! Java后端经验能很好地迁移"
                },
                "ML算法研究": {
                    "适合": "喜欢数学推导、研究新算法",
                    "前景": "门槛高, 竞争激烈",
                    "建议": "需要博士学位或强研究背景"
                },
                "数据工程": {
                    "适合": "喜欢数据处理、Pipeline",
                    "前景": "需求稳定, 不那么热门",
                    "建议": "如果擅长数据可以尝试"
                },
                "MLOps": {
                    "适合": "喜欢基础设施、自动化",
                    "前景": "需求增长快",
                    "建议": "有DevOps经验的人很适合"
                }
            }
        },
        "节点2: 深度 vs 广度(转型中期)": {
            "建议": "先广后深",
            "理由": [
                "初期需要了解全貌(广度), 知道各个方向",
                "中期选择1-2个方向深入(深度), 建立专长",
                "深度方向选择: 基于兴趣+市场需求+自身优势",
                "推荐深入方向: RAG系统(实用) 或 AI Agent(前沿)"
            ]
        },
        "节点3: 技术路线 vs 管理路线(转型后期)": {
            "技术路线": {
                "特点": "继续深入技术, 成为专家",
                "适合": "热爱技术, 不喜欢管人",
                "发展": "Staff Engineer -> Principal Engineer -> Distinguished Engineer"
            },
            "管理路线": {
                "特点": "带领团队, 管理项目和人员",
                "适合": "喜欢管理, 有领导力",
                "发展": "Tech Lead -> EM -> Director -> VP"
            },
            "建议": "不要过早选择, 先积累技术深度, 两条路线可以切换"
        }
    }

    for node, details in decisions.items():
        print(f"\n【{node}】")
        if isinstance(details, dict):
            for key, value in details.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        if isinstance(v, dict):
                            print(f"    {k}:")
                            for kk, vv in v.items():
                                print(f"      {kk}: {vv}")
                        else:
                            print(f"    {k}: {v}")
                elif isinstance(value, list):
                    print(f"  {key}:")
                    for item in value:
                        print(f"    - {item}")
                else:
                    print(f"  {key}: {value}")


# =============================================
# 3. 个人职业规划
# =============================================
def create_personal_plan():
    """创建个人职业规划"""
    print_section("个人职业规划")

    plan = {
        "当前位置": "Java后端工程师 → AI应用工程师(转型中)",
        "目标位置": "中级AI应用工程师(L3)",
        "时间目标": "转型后1年内达到L3水平",

        "第1阶段: 基础与转型(已完成, 39周)": [
            "Python编程基础",
            "ML/DL理论学习",
            "NLP和LLM基础",
            "RAG/Agent实战项目"
        ],
        "第2阶段: 深入与专精(未来3个月)": [
            "深入RAG系统(高级技术)",
            "AI Agent开发(LangGraph)",
            "MLOps和模型部署",
            "技术博客和开源贡献"
        ],
        "第3阶段: 独当一面(未来6个月)": [
            "能独立设计和实现生产级AI系统",
            "在某一方向有深度(RAG/Agent)",
            "建立技术影响力",
            "达到中级AI工程师水平"
        ],
        "第4阶段: 专家方向(未来1-2年)": [
            "成为某一方向的专家",
            "带领AI项目或小团队",
            "在行业中有一定影响力",
            "冲击高级AI工程师"
        ],

        "核心差异化": "后端架构经验 + AI应用能力",
        "一句话定位": "能设计和实现企业级AI应用的工程师"
    }

    for key, value in plan.items():
        if isinstance(value, list):
            print(f"\n【{key}】")
            for item in value:
                print(f"  - {item}")
        else:
            print(f"\n  {key}: {value}")

    # 保存职业规划
    output_file = os.path.join(os.path.dirname(__file__), "career_plan.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)
    print(f"\n  职业规划已保存: {output_file}")


# =============================================
# 主程序
# =============================================
if __name__ == "__main__":
    print("=" * 60)
    print("  W39 Day 3 - 职业路径")
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    describe_career_path()
    describe_key_decisions()
    create_personal_plan()

    print("\n" + "=" * 60)
    print("  职业发展是一场马拉松, 不是短跑!")
    print("  建议: 明确你的长期目标, 制定清晰的行动计划")
    print("=" * 60)
