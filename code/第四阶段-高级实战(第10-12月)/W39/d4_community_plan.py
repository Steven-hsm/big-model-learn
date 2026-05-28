"""
W39 Day 4 - 社区计划
====================
主题: 持续贡献策略, 技术分享计划, 导师网络
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
# 1. 持续贡献策略
# =============================================
def plan_sustained_contribution():
    """规划持续贡献策略"""
    print_section("持续贡献策略")

    strategy = {
        "开源贡献计划": {
            "目标项目(优先级排序)": [
                {"项目": "Hugging Face Transformers", "目标": "每月1-2个PR", "类型": "文档+Bug修复"},
                {"项目": "LangChain", "目标": "每月1个PR", "类型": "集成组件"},
                {"项目": "个人开源项目", "目标": "持续维护和更新", "类型": "RAG工具包"}
            ],
            "贡献时间规划": {
                "工作日": "每天30分钟(Review Issue, 回复讨论)",
                "周末": "每周2-3小时(编写代码, 提交PR)",
                "总计": "每周约5小时"
            },
            "成长路径": [
                "第1-3月: Contributor (文档和Bug修复)",
                "第4-6月: Active Contributor (功能开发)",
                "第7-12月: Reviewer (帮助Review他人PR)",
                "1年+: Committer (项目核心贡献者)"
            ]
        },
        "内容创作计划": {
            "博客": {
                "频率": "每周1篇",
                "类型": ["教程(40%)", "经验分享(25%)", "深度分析(20%)", "资源推荐(15%)"],
                "平台": ["个人博客(主)", "掘金", "知乎", "CSDN"]
            },
            "开源项目": {
                "频率": "每月1个小项目/工具",
                "方向": "RAG工具、LLM评估、Agent框架",
                "目标": "3个GitHub 100+ star的项目"
            },
            "技术分享": {
                "频率": "每月1次(线上或线下)",
                "形式": ["技术Meetup演讲", "公司内部分享", "线上直播", "播客参与"]
            }
        }
    }

    for category, details in strategy.items():
        print(f"\n【{category}】")
        for key, value in details.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    if isinstance(v, list):
                        print(f"    {k}: {', '.join(str(i) for i in v)}")
                    else:
                        print(f"    {k}: {v}")
            elif isinstance(value, list):
                print(f"  {key}:")
                for item in value:
                    if isinstance(item, dict):
                        details_str = " | ".join(f"{k}: {v}" for k, v in item.items())
                        print(f"    - {details_str}")
                    else:
                        print(f"    - {item}")


# =============================================
# 2. 技术分享计划
# =============================================
def plan_tech_sharing():
    """规划技术分享"""
    print_section("技术分享计划")

    sharing_plan = {
        "分享主题库": [
            {"主题": "从零构建RAG系统", "类型": "实战教程", "时长": "45分钟",
             "适合": "Meetup, 公司分享"},
            {"主题": "大模型微调实战: LoRA从原理到代码", "类型": "深度分析", "时长": "60分钟",
             "适合": "技术会议, 公司分享"},
            {"主题": "Java工程师转型AI的实战经验", "类型": "经验分享", "时长": "30分钟",
             "适合": "Meetup, 直播"},
            {"主题": "生产级AI系统设计", "类型": "系统设计", "时长": "60分钟",
             "适合": "技术会议"},
            {"主题": "AI Agent开发实战", "类型": "实战教程", "时长": "45分钟",
             "适合": "Meetup, 公司分享"},
            {"主题": "Prompt Engineering高级技巧", "类型": "技巧分享", "时长": "30分钟",
             "适合": "闪电演讲, 公司分享"}
        ],
        "分享时间线": {
            "第1个月": "公司内部技术分享(试水)",
            "第2个月": "线上技术直播或YouTube/B站视频",
            "第3个月": "参加线下Meetup做演讲",
            "第6个月": "参加技术会议做分享"
        },
        "分享模板": {
            "结构": [
                "1. 开场(2分钟): 自我介绍, 今天讲什么",
                "2. 背景(5分钟): 为什么这个话题重要",
                "3. 核心内容(20-30分钟): 原理+代码+示例",
                "4. 实战演示(10分钟): 现场演示",
                "5. 总结(5分钟): 关键要点回顾",
                "6. Q&A(5-10分钟): 互动问答"
            ],
            "准备清单": [
                "[ ] PPT/Slides准备完成",
                "[ ] Demo代码测试通过",
                "[ ] 至少排练3次",
                "[ ] 准备备用方案(Demo失败时)",
                "[ ] 准备Q&A可能的回答"
            ]
        }
    }

    for category, details in sharing_plan.items():
        print(f"\n【{category}】")
        if isinstance(details, list):
            for item in details:
                if isinstance(item, dict):
                    details_str = " | ".join(f"{k}: {v}" for k, v in item.items())
                    print(f"  {details_str}")
                else:
                    print(f"  {item}")
        elif isinstance(details, dict):
            for key, value in details.items():
                if isinstance(value, list):
                    print(f"  {key}:")
                    for item in value:
                        print(f"    {item}")
                elif isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    {k}: {v}")

    # 社区参与日历
    print("\n--- 社区参与周计划 ---")
    weekly_plan = {
        "周一": "发布博客文章",
        "周二": "浏览GitHub Issue, 回复讨论(30min)",
        "周三": "参与Discord/Slack技术讨论(30min)",
        "周四": "阅读一篇AI论文或技术文章",
        "周五": "分享本周学习收获(推文/短文)",
        "周六": "开源项目贡献(2-3小时)",
        "周日": "规划下周内容和社区参与"
    }
    for day, activity in weekly_plan.items():
        print(f"  {day}: {activity}")


# =============================================
# 3. 导师网络
# =============================================
def plan_mentor_network():
    """规划导师网络"""
    print_section("导师网络建设")

    network_plan = {
        "为什么需要导师网络": [
            "获得职业发展建议和技术指导",
            "拓宽视野, 了解行业趋势",
            "获得内推和职业机会",
            "建立长期的技术人脉"
        ],
        "导师类型": {
            "技术导师": {
                "作用": "技术方向指导, 代码Review",
                "寻找方式": "开源项目Maintainer, 技术社区活跃者",
                "互动频率": "每月1-2次"
            },
            "职业导师": {
                "作用": "职业规划, 软技能发展",
                "寻找方式": "公司前辈, 行业资深人士",
                "互动频率": "每季度1次"
            },
            "同行伙伴": {
                "作用": "互相学习, 共同进步",
                "寻找方式": "学习小组, 技术社区",
                "互动频率": "每周交流"
            }
        },
        "建立导师网络的方法": [
            "1. 在开源项目中认识资深开发者",
            "2. 参加技术Meetup和会议",
            "3. 在Twitter/LinkedIn关注并互动",
            "4. 主动发邮件表达学习意愿(但要简短有礼貌)",
            "5. 先给对方提供价值(帮忙解决Issue, 分享文章)",
            "6. 建立学习小组, 互为伙伴"
        ],
        "维护导师关系": [
            "定期更新进展(每月一封简短邮件)",
            "尊重对方时间(准备好具体问题)",
            "表达感谢(真诚的感谢信)",
            "分享你的成长(让对方感到指导有价值)",
            "回馈社区(帮助新人, 传递帮助)"
        ]
    }

    for category, details in network_plan.items():
        print(f"\n【{category}】")
        if isinstance(details, list):
            for item in details:
                print(f"  {item}")
        elif isinstance(details, dict):
            for key, value in details.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {value}")


# =============================================
# 主程序
# =============================================
if __name__ == "__main__":
    print("=" * 60)
    print("  W39 Day 4 - 社区计划")
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    plan_sustained_contribution()
    plan_tech_sharing()
    plan_mentor_network()

    print("\n" + "=" * 60)
    print("  社区参与让你的成长加速!")
    print("  建议: 今天就开始, 在一个社区中回答第一个问题")
    print("=" * 60)
