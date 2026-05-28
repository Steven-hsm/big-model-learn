"""
W36 Day 5 - 社区参与
====================
主题: 技术论坛列表, 开源社区, 技术会议, 网络建设
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
# 1. 技术论坛与平台
# =============================================
def list_tech_forums():
    """列出AI领域技术论坛和平台"""
    print_section("AI领域技术论坛与平台")

    forums = {
        "国际平台": {
            "GitHub": {
                "网址": "https://github.com",
                "用途": "代码托管、开源协作、代码审查",
                "建议": "关注star高的AI项目, 参与Discussion"
            },
            "Stack Overflow": {
                "网址": "https://stackoverflow.com",
                "用途": "技术问答, 解决编程问题",
                "建议": "回答Python/AI相关问题建立声誉"
            },
            "Hugging Face": {
                "网址": "https://huggingface.co",
                "用途": "模型分享、数据集、Spaces演示",
                "建议": "上传模型、创建Spaces展示项目"
            },
            "Reddit": {
                "网址": "https://reddit.com",
                "相关子版": ["r/MachineLearning", "r/LocalLLaMA", "r/artificial"],
                "建议": "关注最新讨论和论文分享"
            },
            "Dev.to": {
                "网址": "https://dev.to",
                "用途": "技术博客平台",
                "建议": "发布英文技术文章, 扩大国际影响力"
            },
            "Medium / Towards Data Science": {
                "网址": "https://medium.com",
                "用途": "高质量技术文章",
                "建议": "发布深度技术文章"
            },
            "Papers With Code": {
                "网址": "https://paperswithcode.com",
                "用途": "论文+代码+基准测试",
                "建议": "追踪前沿研究, 学习最新实现"
            }
        },
        "国内平台": {
            "知乎": {
                "网址": "https://zhihu.com",
                "用途": "技术问答、专栏文章",
                "建议": "写AI技术专栏, 回答热门问题"
            },
            "CSDN": {
                "网址": "https://csdn.net",
                "用途": "技术博客、教程",
                "建议": "发布教程和实践文章"
            },
            "掘金": {
                "网址": "https://juejin.cn",
                "用途": "技术社区, 前端+AI",
                "建议": "发布高质量技术文章"
            },
            "博客园": {
                "网址": "https://cnblogs.com",
                "用途": "技术博客",
                "建议": "发布深度技术文章"
            },
            "AI研习社": {
                "网址": "https://ai.yanxishe.com",
                "用途": "AI学习社区",
                "建议": "参与竞赛和讨论"
            },
            "Kaggle (中文社区)": {
                "网址": "https://kaggle.com",
                "用途": "数据科学竞赛",
                "建议": "参加AI相关竞赛积累经验"
            }
        }
    }

    for region, platforms in forums.items():
        print(f"\n【{region}】")
        for name, info in platforms.items():
            print(f"  {name} ({info['网址']})")
            print(f"    用途: {info['用途']}")
            print(f"    建议: {info['建议']}")


# =============================================
# 2. 开源社区
# =============================================
def list_oss_communities():
    """列出AI开源社区"""
    print_section("AI开源社区")

    communities = {
        "大模型社区": [
            {
                "名称": "Hugging Face Community",
                "链接": "https://discuss.huggingface.co",
                "活跃度": "高",
                "贡献机会": "模型开发、文档、评估"
            },
            {
                "名称": "LangChain Community",
                "链接": "https://github.com/langchain-ai/langchain/discussions",
                "活跃度": "高",
                "贡献机会": "集成开发、模板、工具"
            },
            {
                "名称": "LlamaIndex Community",
                "链接": "https://discord.gg/llamaindex",
                "活跃度": "中高",
                "贡献机会": "数据连接器、索引"
            }
        ],
        "ML/DL框架社区": [
            {
                "名称": "PyTorch Community",
                "链接": "https://discuss.pytorch.org",
                "活跃度": "高",
                "贡献机会": "算子开发、文档、教程"
            },
            {
                "名称": "scikit-learn",
                "链接": "https://github.com/scikit-learn/scikit-learn",
                "活跃度": "高",
                "贡献机会": "算法实现、文档"
            }
        ],
        "AI应用社区": [
            {
                "名称": "AutoGPT Community",
                "链接": "https://discord.gg/autogpt",
                "活跃度": "中",
                "贡献机会": "Agent开发、插件"
            },
            {
                "名称": "Ollama Community",
                "链接": "https://discord.gg/ollama",
                "活跃度": "高",
                "贡献机会": "模型适配、工具集成"
            }
        ]
    }

    for category, items in communities.items():
        print(f"\n【{category}】")
        for item in items:
            print(f"  {item['名称']}")
            print(f"    链接: {item['链接']}")
            print(f"    活跃度: {item['活跃度']}")
            print(f"    贡献机会: {item['贡献机会']}")

    # 社区参与建议
    print("\n--- 社区参与建议 ---")
    suggestions = [
        "1. 选择2-3个社区深度参与, 不要贪多",
        "2. 先潜水学习社区文化和规范",
        "3. 从回答简单问题开始建立信任",
        "4. 定期分享有价值的内容",
        "5. 参加社区线上/线下活动",
        "6. 尊重他人, 建设性讨论",
        "7. 帮助新人, 教学相长"
    ]
    for s in suggestions:
        print(f"  {s}")


# =============================================
# 3. 技术会议
# =============================================
def list_tech_conferences():
    """列出AI相关技术会议"""
    print_section("AI技术会议")

    conferences = {
        "顶级学术会议": [
            {"名称": "NeurIPS", "全称": "Neural Information Processing Systems",
             "时间": "每年12月", "方向": "通用ML/DL"},
            {"名称": "ICML", "全称": "International Conference on ML",
             "时间": "每年7月", "方向": "机器学习"},
            {"名称": "ICLR", "全称": "International Conference on Learning Representations",
             "时间": "每年4-5月", "方向": "表示学习"},
            {"名称": "ACL", "全称": "Association for Computational Linguistics",
             "时间": "每年8月", "方向": "NLP"},
            {"名称": "CVPR", "全称": "Computer Vision and Pattern Recognition",
             "时间": "每年6月", "方向": "计算机视觉"},
            {"名称": "AAAI", "全称": "AAAI Conference on AI",
             "时间": "每年2月", "方向": "通用AI"}
        ],
        "行业会议": [
            {"名称": "AI Summit", "描述": "全球AI行业峰会"},
            {"名称": "Google I/O", "描述": "Google技术大会, 含AI内容"},
            {"名称": "Microsoft Build", "描述": "微软开发者大会, 含Azure AI"},
            {"名称": "AWS re:Invent", "描述": "AWS大会, 含AI/ML服务"},
            {"名称": "NVIDIA GTC", "描述": "GPU技术大会, AI计算"},
            {"名称": "Hugging Face AI Community Day", "描述": "HF社区日"}
        ],
        "国内会议": [
            {"名称": "WAIC", "描述": "世界人工智能大会(上海)"},
            {"名称": "QCon", "描述": "全球软件开发大会, 含AI专题"},
            {"名称": "ArchSummit", "描述": "架构师峰会, 含AI架构"},
            {"名称": "AI开发者大会", "描述": "百度/阿里/腾讯等举办"},
            {"名称": "DataFunSummit", "描述": "数据智能峰会"}
        ],
        "Meetup/线上活动": [
            "Hugging Face Community Events",
            "PyTorch Developer Day",
            "LangChain Meetup",
            "本地AI/ML Meetup (通过meetup.com查找)",
            "各种AI Discord服务器的线上分享"
        ]
    }

    for category, items in conferences.items():
        print(f"\n【{category}】")
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    details = " | ".join(f"{k}: {v}" for k, v in item.items())
                    print(f"  {details}")
                else:
                    print(f"  {item}")


# =============================================
# 4. 网络建设
# =============================================
def demonstrate_networking():
    """展示网络建设策略"""
    print_section("网络建设策略")

    networking = {
        "线上网络建设": {
            "GitHub": [
                "关注并star优秀的AI项目",
                "参与Issue和Discussion讨论",
                "给项目提交PR建立联系",
                "关注项目Maintainer"
            ],
            "Twitter/X": [
                "关注AI领域的研究者和工程师",
                "转发并评论有价值的内容",
                "分享自己的学习笔记和项目",
                "参与AI话题讨论"
            ],
            "LinkedIn": [
                "完善个人资料, 突出AI技能",
                "连接AI领域的专业人士",
                "发布技术文章和观点",
                "加入AI相关的群组"
            ],
            "Discord/Slack": [
                "加入技术社区频道",
                "积极参与技术讨论",
                "帮助回答问题",
                "分享有价值的资源"
            ]
        },
        "线下网络建设": {
            "技术Meetup": [
                "参加本地AI/Python Meetup",
                "主动介绍自己和技术方向",
                "交换联系方式"
            ],
            "技术会议": [
                "参加行业会议和学术会议",
                "准备简短的自我介绍",
                "积极参加Workshop和Tutorial"
            ],
            "公司内部": [
                "组织内部技术分享",
                "参与跨部门AI项目",
                "建立内部AI学习小组"
            ]
        },
        "导师网络": {
            "为什么需要导师": [
                "获得职业发展指导",
                "技术方向建议",
                "行业人脉介绍",
                "避免走弯路"
            ],
            "如何找到导师": [
                "在社区中认识资深开发者",
                "参加导师匹配项目",
                "通过开源项目建立联系",
                "主动发邮件表达学习意愿"
            ],
            "如何维护关系": [
                "定期更新进展",
                "尊重对方时间",
                "准备好具体问题",
                "表达感谢"
            ]
        }
    }

    for category, items in networking.items():
        print(f"\n【{category}】")
        if isinstance(items, dict):
            for platform, actions in items.items():
                print(f"  {platform}:")
                for action in actions:
                    print(f"    - {action}")

    # 网络建设行动计划
    action_plan = {
        "第1周": "完善LinkedIn和GitHub Profile",
        "第2周": "加入3-5个AI社区Discord/Slack",
        "第3周": "在社区中回答第一个问题",
        "第4周": "联系3位AI领域的专业人士",
        "第2月": "参加一次线下技术Meetup",
        "第3月": "在社区中建立初步声誉"
    }
    print("\n--- 网络建设行动计划 ---")
    for week, action in action_plan.items():
        print(f"  {week}: {action}")

    # 保存社区资源列表
    resources = {
        "forums": list(forums.keys()),
        "communities": list(communities.keys()),
        "conferences": list(conferences.keys())
    }
    output_file = os.path.join(os.path.dirname(__file__), "community_resources.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(resources, f, ensure_ascii=False, indent=2)
    print(f"\n  社区资源列表已保存: {output_file}")


# =============================================
# 主程序
# =============================================
if __name__ == "__main__":
    print("=" * 60)
    print("  W36 Day 5 - 社区参与")
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    list_tech_forums()
    list_oss_communities()
    list_tech_conferences()
    demonstrate_networking()

    print("\n" + "=" * 60)
    print("  社区参与是技术人成长的重要途径!")
    print("  建议: 选择2-3个社区深度参与, 质量比数量重要")
    print("=" * 60)
