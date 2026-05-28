"""
W35 Day 7 - 贡献总结
====================
主题: 记录贡献, 建立信誉, 持续参与策略
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
# 1. 贡献记录管理
# =============================================
def demonstrate_contribution_tracking():
    """展示贡献记录管理方法"""
    print_section("贡献记录管理")

    # 贡献记录结构
    contribution_log = {
        "元信息": {
            "维护者": "你的名字",
            "GitHub": "https://github.com/yourusername",
            "开始日期": "2025-01-01",
            "最后更新": datetime.now().strftime("%Y-%m-%d")
        },
        "贡献统计": {
            "总PR数": 0,
            "已合并PR": 0,
            "待审核PR": 0,
            "Issue报告数": 0,
            "Code Review数": 0,
            "文档贡献数": 0
        },
        "贡献记录": [
            {
                "日期": "2025-01-15",
                "项目": "Hugging Face Transformers",
                "类型": "Bug修复",
                "描述": "修复tokenizer处理空字符串的Bug",
                "PR链接": "https://github.com/huggingface/transformers/pull/xxxxx",
                "状态": "已合并",
                "学到的": "深入理解了tokenizer的处理流程"
            },
            {
                "日期": "2025-02-01",
                "项目": "LangChain",
                "类型": "文档",
                "描述": "翻译Getting Started文档为中文",
                "PR链接": "https://github.com/langchain-ai/langchain/pull/xxxxx",
                "状态": "已合并",
                "学到的": "文档编写和翻译技巧"
            }
        ],
        "参与项目": [
            {
                "名称": "Hugging Face Transformers",
                "角色": "Contributor",
                "贡献数": 5,
                "最近活跃": "2025-03-01"
            }
        ]
    }

    print("\n贡献记录结构:")
    print(json.dumps(contribution_log, ensure_ascii=False, indent=2))

    # 贡献统计脚本
    print("\n--- 贡献统计脚本示例 ---")
    stats_script = '''
import json
from datetime import datetime, timedelta
from collections import Counter

def analyze_contributions(log_file):
    """分析贡献记录"""
    with open(log_file, 'r', encoding='utf-8') as f:
        log = json.load(f)

    records = log.get("贡献记录", [])

    # 按类型统计
    type_counts = Counter(r["类型"] for r in records)

    # 按项目统计
    project_counts = Counter(r["项目"] for r in records)

    # 按月份统计
    monthly = Counter()
    for r in records:
        month = r["日期"][:7]  # YYYY-MM
        monthly[month] += 1

    # 按状态统计
    status_counts = Counter(r["状态"] for r in records)

    print("=== 贡献统计报告 ===")
    print(f"总贡献数: {len(records)}")
    print(f"\\n按类型: {dict(type_counts)}")
    print(f"按项目: {dict(project_counts)}")
    print(f"按月份: {dict(sorted(monthly.items()))}")
    print(f"按状态: {dict(status_counts)}")

    # 计算合并率
    merged = status_counts.get("已合并", 0)
    total = len(records)
    merge_rate = (merged / total * 100) if total > 0 else 0
    print(f"\\n合并率: {merge_rate:.1f}%")

analyze_contributions("contribution_log.json")
'''
    print(stats_script)

    # 保存贡献日志模板
    output_file = os.path.join(os.path.dirname(__file__), "contribution_log.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(contribution_log, f, ensure_ascii=False, indent=2)
    print(f"\n贡献记录模板已保存: {output_file}")


# =============================================
# 2. 建立开源信誉
# =============================================
def demonstrate_building_reputation():
    """展示建立开源信誉的方法"""
    print_section("建立开源信誉")

    reputation = {
        "信誉建立路径": {
            "第1阶段(1-3个月)": {
                "目标": "成为项目的活跃Contributor",
                "行动": [
                    "提交5-10个PR(文档、Bug修复)",
                    "积极参与Issue讨论",
                    "帮助回答新用户问题",
                    "熟悉项目代码和流程"
                ]
            },
            "第2阶段(3-6个月)": {
                "目标": "成为可信的Contributor",
                "行动": [
                    "提交功能级PR",
                    "帮助Review他人代码",
                    "参与设计讨论",
                    "写技术博客分享经验"
                ]
            },
            "第3阶段(6-12个月)": {
                "目标": "成为项目的核心Contributor",
                "行动": [
                    "承担重要功能开发",
                    "成为Reviewer",
                    "帮助新Contributor上手",
                    "参与项目Roadmap讨论"
                ]
            },
            "第4阶段(12个月+)": {
                "目标": "成为Maintainer",
                "行动": [
                    "持续高质量贡献",
                    "帮助管理项目",
                    "组织社区活动",
                    "指导新Contributor"
                ]
            }
        },
        "信誉指标": [
            "GitHub贡献图: 保持活跃(Green Dot)",
            "PR合并率: 目标>80%",
            "Review响应时间: 及时回应反馈",
            "代码质量: 测试覆盖、文档完整",
            "社区影响力: 回答问题、写文章"
        ],
        "最佳实践": [
            "质量优先: 一个高质量PR胜过十个低质量的",
            "及时响应: Review反馈在24-48小时内处理",
            "保持耐心: PR可能需要数周才能合并",
            "友善沟通: 始终保持礼貌和专业",
            "持续学习: 从每次贡献中学习新知识"
        ]
    }

    for category, items in reputation.items():
        print(f"\n【{category}】")
        if isinstance(items, dict):
            for key, value in items.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    print(f"    ���标: {value['目标']}")
                    print(f"    行动:")
                    for action in value["行动"]:
                        print(f"      - {action}")
                else:
                    print(f"  {key}: {value}")
        elif isinstance(items, list):
            for item in items:
                print(f"  {item}")


# =============================================
# 3. 持续参与策略
# =============================================
def demonstrate_sustained_contribution():
    """展示持续参与策略"""
    print_section("持续参与策略")

    strategies = {
        "时间管理": {
            "每日(15-30分钟)": [
                "浏览感兴趣的项目动态",
                "回复Issue和Discussion",
                "Review待处理的PR"
            ],
            "每周(2-4小时)": [
                "推进进行中的PR",
                "参与社区讨论",
                "学习项目新功能"
            ],
            "每月(1天)": [
                "完成一个中等规模的贡献",
                "整理贡献记录",
                "规划下月计划"
            ]
        },
        "避免倦怠": [
            "不要同时参与太多项目(1-3个为宜)",
            "选择真正感兴趣的项目",
            "不要给自己太大压力",
            "享受贡献的过程",
            "适当休息, 充电后继续"
        ],
        "持续成长": [
            "从简单到复杂: 逐步挑战更难的任务",
            "深度参与: 选择1-2个项目深入参与",
            "跨界贡献: 尝试不同类型的贡献",
            "帮助他人: 教学相长, 指导新Contributor",
            "写技术博客: 分享学习经验和贡献经历"
        ],
        "利用贡献助力职业发展": {
            "简历亮点": "开源贡献是工程师能力的直接证明",
            "面试话题": "开源经历是面试中的加分项",
            "行业人脉": "通过开源认识行业专家",
            "技术视野": "接触最前沿的技术实践",
            "软技能": "协作、沟通、项目管理能力"
        }
    }

    for category, items in strategies.items():
        print(f"\n【{category}】")
        if isinstance(items, dict):
            for key, value in items.items():
                if isinstance(value, list):
                    print(f"  {key}:")
                    for item in value:
                        print(f"    - {item}")
                else:
                    print(f"  {key}: {value}")
        elif isinstance(items, list):
            for item in items:
                print(f"  {item}")


# =============================================
# 4. W35周总结
# =============================================
def weekly_summary():
    """W35周总结"""
    print_section("W35周总结: 开源贡献")

    summary = {
        "本周学习内容": [
            "Day 1: 开源文化与贡献方式",
            "Day 2: 阅读源码方法与调试技巧",
            "Day 3: 第一个PR完整流程",
            "Day 4: 文档贡献与翻译",
            "Day 5: Bug修复流程与测试验证",
            "Day 6: 功能贡献与Code Review",
            "Day 7: 贡献总结与持续参与"
        ],
        "关键收获": [
            "1. 开源贡献不仅是代码, 文档和社区参与同样重要",
            "2. 从小处着手: 文档修复→Bug修复→功能开发",
            "3. 良好的沟通和代码规范是成功PR的关键",
            "4. 持续参与比一次性大量贡献更有价值",
            "5. 开源贡献是提升技术能力和行业影响力的最佳途径"
        ],
        "行动计划": {
            "本周": "选择1个项目, 提交第一个PR(文档修复)",
            "本月": "提交3-5个PR, 包括1个Bug修复",
            "本季度": "成为1个项目的活跃Contributor",
            "半年": "在1-2个项目中建立信誉, 参与功能开发"
        },
        "推荐资源": [
            "https://opensource.guide/ - GitHub开源指南",
            "https://www.firsttimersonly.com/ - 第一次贡献指南",
            "https://goodfirstissues.com/ - 寻找good-first-issue",
            "https://up-for-grabs.net/ - 适合新手的开源任务",
            "https://24pullrequests.com/ - 12月开源贡献活动"
        ]
    }

    for category, items in summary.items():
        print(f"\n【{category}】")
        if isinstance(items, list):
            for item in items:
                print(f"  {item}")
        elif isinstance(items, dict):
            for key, value in items.items():
                print(f"  {key}: {value}")

    # 生成个人开源行动计划
    action_plan = {
        "个人信息": {
            "GitHub": "",
            "主要技术栈": "Python, AI/ML",
            "感兴趣的领域": "LLM, NLP, MLOps"
        },
        "目标项目(优先级排序)": [
            {"名称": "", "理由": "", "首次贡献计划": ""}
        ],
        "里程碑": {
            "第1周": "Fork项目, 熟悉代码, 提交第一个文档PR",
            "第2-4周": "提交3-5个PR, 包含Bug修复",
            "第2月": "开始参与Issue讨论和Code Review",
            "第3月": "提交第一个功能PR",
            "第6月": "成为活跃Contributor, 建立信誉"
        }
    }

    output_file = os.path.join(os.path.dirname(__file__), "oss_action_plan.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(action_plan, f, ensure_ascii=False, indent=2)
    print(f"\n  个人开源行动计划模板已保存: {output_file}")


# =============================================
# 主程序
# =============================================
if __name__ == "__main__":
    print("=" * 60)
    print("  W35 Day 7 - 贡献总结")
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    demonstrate_contribution_tracking()
    demonstrate_building_reputation()
    demonstrate_sustained_contribution()
    weekly_summary()

    print("\n" + "=" * 60)
    print("  W35周学习完成!")
    print("  开源贡献是一次马拉松, 不是短跑!")
    print("  从今天开始你的第一个开源贡献吧!")
    print("=" * 60)
