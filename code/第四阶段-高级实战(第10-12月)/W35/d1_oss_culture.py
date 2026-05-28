"""
W35 Day 1 - 开源文化与贡献指南
==============================
���题: 了解开源文化, 贡献方式, 寻找合适项目, GitHub使用技巧
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
# 1. 开源文化概述
# =============================================
def demonstrate_oss_culture():
    """展示开源文化核心概念"""
    print_section("开源文化核心概念")

    oss_principles = {
        "自由与开放": {
            "描述": "源代码公开, 任何人可以查看、修改和分发",
            "核心理念": "知识共享, 协作创新",
            "关键许可": ["MIT", "Apache 2.0", "GPL v3", "BSD"]
        },
        "协作精神": {
            "描述": "全球开发者共同参与, 代码审查, 知识共享",
            "核心工具": ["Git", "GitHub", "GitLab", "Gitee"],
            "沟通方式": ["Issue", "PR", "Discussion", "Discord/Slack"]
        },
        "Meritocracy": {
            "描述": "基于贡献和能力获得认可和权限",
            "角色晋升": "Contributor -> Reviewer -> Committer -> Maintainer",
            "关键": "代码质量和社区贡献决定影响力"
        },
        "尊重与包容": {
            "描述": "遵守行为准则, 尊重不同观点和背景",
            "准则": "CODE_OF_CONDUCT.md",
            "原则": "建设性讨论, 友善交流"
        }
    }

    for principle, details in oss_principles.items():
        print(f"\n【{principle}】")
        for key, value in details.items():
            if isinstance(value, list):
                print(f"  {key}: {', '.join(value)}")
            else:
                print(f"  {key}: {value}")

    # 开源许可证对比
    print("\n--- 常见开源许可证对比 ---")
    licenses = [
        {"名称": "MIT", "自由度": "★★★★★", "要求": "保留版权声明", "适用": "最宽松, 适合工具库"},
        {"名称": "Apache 2.0", "自由度": "★★★★☆", "要求": "保留版权+声明修改", "适用": "企业友好, 专利保护"},
        {"名称": "GPL v3", "自由度": "★★★☆☆", "要求": "衍生作品也须开源", "适用": "确保开源传播"},
        {"名称": "BSD", "自由度": "★★★★★", "要求": "保留版权声明", "适用": "类似MIT, 学术友好"},
    ]

    for lic in licenses:
        print(f"  {lic['名称']:12s} | 自由度: {lic['自由度']} | {lic['适用']}")


# =============================================
# 2. 贡献方式
# =============================================
def demonstrate_contribution_types():
    """展示开源贡献的多种方式"""
    print_section("开源贡献的多种方式")

    contributions = {
        "代码贡献": {
            "难度": "★★★★☆",
            "价值": "★★★★★",
            "具体方式": [
                "修复Bug - 从Issue列表寻找good-first-issue",
                "实现新功能 - 参与Roadmap讨论",
                "代码重构 - 提升代码质量和可读性",
                "性能优化 - 分析瓶颈, 提出改进方案",
                "编写测试 - 提高测试覆盖率"
            ],
            "入门建议": "从小改动开始, 如修复文档错误、添加类型注解"
        },
        "文档贡献": {
            "难度": "★★☆☆☆",
            "价值": "★★★★☆",
            "具体方式": [
                "修复文档错误 - 拼写、格式、过时内容",
                "翻译文档 - 中文化, 扩大影响力",
                "补充示例 - 编写使用示例和教程",
                "改进API文档 - 参数说明、返回值描述",
                "编写FAQ - 收集常见问题并解答"
            ],
            "入门建议": "文档贡献是最友好的入门方式, 不需要深入理解代码"
        },
        "社区贡献": {
            "难度": "★★☆☆☆",
            "价值": "★★★★☆",
            "具体方式": [
                "回答问题 - 在Issue和论坛中帮助他人",
                "代码审查 - Review他人的PR",
                "Bug报告 - 提交清晰完整的Issue",
                "功能建议 - 提出改进建议",
                "组织活动 - Meetup、线上分享"
            ],
            "入门建议": "积极参与讨论, 建立社区关系"
        },
        "其他贡献": {
            "难度": "★★☆☆☆",
            "价值": "★★★☆☆",
            "具体方式": [
                "UI/UX设计 - 改进用户界面",
                "项目管理 - 整理Issue, 维护Roadmap",
                "资金支持 - Open Collective, GitHub Sponsors",
                "宣传推广 - 写文章, 做分享",
                "数据标注 - 为AI项目贡献训练数据"
            ],
            "入门建议": "发挥自己的特长, 任何贡献都有价值"
        }
    }

    for ctype, details in contributions.items():
        print(f"\n【{ctype}】难度: {details['难度']}  价值: {details['价值']}")
        print("  具体方式:")
        for method in details["具体方式"]:
            print(f"    - {method}")
        print(f"  入门建议: {details['入门建议']}")


# =============================================
# 3. 寻找合适项目
# =============================================
def find_suitable_projects():
    """寻找合适的开源项目"""
    print_section("如何寻找合适的开源项目")

    # AI领域热门开源项目
    ai_projects = {
        "大模型框架": [
            {"名称": "Hugging Face Transformers", "语言": "Python", "Stars": "130k+",
             "贡献机会": "模型支持、文档翻译、Bug修复", "难度": "中"},
            {"名称": "LangChain", "语言": "Python/JS", "Stars": "90k+",
             "贡献机会": "集成开发、文档完善、示例编写", "难度": "中"},
            {"名称": "vLLM", "语言": "Python/C++", "Stars": "30k+",
             "贡献机会": "推理优化、模型支持", "难度": "高"},
        ],
        "数据处理": [
            {"名称": "Pandas", "语言": "Python", "Stars": "43k+",
             "贡献机会": "性能优化、文档改进", "难度": "中高"},
            {"名称": "Polars", "语言": "Rust/Python", "Stars": "30k+",
             "贡献机会": "Python绑定、文档", "难度": "中"},
        ],
        "可视化": [
            {"名称": "Matplotlib", "语言": "Python", "Stars": "19k+",
             "贡献机会": "Bug修复、文档改进", "难度": "中"},
            {"名称": "Streamlit", "语言": "Python", "Stars": "35k+",
             "贡献机会": "组件开发、示例", "难度": "中"},
        ]
    }

    for category, projects in ai_projects.items():
        print(f"\n【{category}】")
        for proj in projects:
            print(f"  {proj['名称']} ({proj['语言']}) ★{proj['Stars']}")
            print(f"    贡献机会: {proj['贡献机会']}")
            print(f"    难度: {proj['难度']}")

    # 项目选择策略
    print("\n--- 项目选择策略 ---")
    strategies = [
        "1. 选择你日常使用的项目 - 熟悉度高, 动机强",
        "2. 查看标签: good-first-issue, help-wanted, documentation",
        "3. 关注社区活跃度 - Issue响应速度、PR合并频率",
        "4. 阅读CONTRIBUTING.md - 了解贡献流程",
        "5. 加入社区聊天频道 - Discord/Slack/Gitter",
        "6. 从小处着手 - 文档修复、测试补充、小Bug修复",
        "7. 选择有良好Code of Conduct的项目"
    ]
    for s in strategies:
        print(f"  {s}")


# =============================================
# 4. GitHub使用技巧
# =============================================
def demonstrate_github_tips():
    """GitHub使用技巧"""
    print_section("GitHub使用技巧")

    tips = {
        "搜索技巧": [
            "language:python stars:>1000 topic:llm",
            "good-first-issues:>5 label:help-wanted",
            "fork:true (搜索包含fork的结果)",
            "is:open is:issue label:bug",
            "sort:updated-desc (按更新时间排序)"
        ],
        "快捷键": [
            "T - 激活文件搜索",
            "W - 切换分支",
            "L - 跳转到指定行",
            "B - 查看Blame",
            ". - 在浏览器中打开VS Code编辑器"
        ],
        "实用功能": [
            "GitHub Actions - CI/CD自动化",
            "GitHub Codespaces - 在线开发环境",
            "GitHub Copilot - AI辅助编程",
            "GitHub Discussions - 社区讨论",
            "GitHub Projects - 项目管理看板"
        ],
        "Git命令技巧": [
            "git log --oneline --graph --all  # 查看分支图",
            "git stash  # 暂存未提交的修改",
            "git rebase -i HEAD~3  # 交互式变基最近3个提交",
            "git cherry-pick <commit>  # 选择性合并提交",
            "git bisect  # 二分查找引入Bug的提交"
        ]
    }

    for category, items in tips.items():
        print(f"\n【{category}】")
        for item in items:
            print(f"  {item}")

    # 生成GitHub Profile优化建议
    print("\n--- GitHub Profile优化 ---")
    profile_tips = [
        "设置美观的Profile README (.github/README.md)",
        "Pinned repositories展示最佳项目",
        "保持绿色贡献图(Green Dot)",
        "编写清晰的commit message",
        "使用GitHub Projects展示进行中的工作"
    ]
    for tip in profile_tips:
        print(f"  - {tip}")


# =============================================
# 5. 生成贡献指南
# =============================================
def generate_contribution_guide():
    """生成个人贡献指南"""
    print_section("个人开源贡献指南")

    guide = {
        "第一步: 准备工作": [
            "完善GitHub Profile",
            "学习Git基本操作(fork, clone, branch, commit, push, PR)",
            "阅读目标项目的CONTRIBUTING.md",
            "设置本地开发环境",
            "了解项目的代码风格和测试规范"
        ],
        "第二步: 选择任务": [
            "浏览good-first-issue标签",
            "选择自己熟悉的领域",
            "评估任务难度和时间",
            "在Issue中表达贡献意愿",
            "等待Maintainer确认"
        ],
        "第三步: 开始贡献": [
            "Fork项目到自己的账号",
            "创建功能分支(feature-branch)",
            "编写代码并编写测试",
            "确保通过所有CI检查",
            "编写清晰的commit message"
        ],
        "第四步: 提交PR": [
            "推送分支到Fork仓库",
            "创建Pull Request",
            "填写PR模板(改动描述、测试说明)",
            "关联相关Issue(Closes #xxx)",
            "响应Review反馈并及时修改"
        ],
        "第五步: 持续参与": [
            "关注项目动态",
            "帮助Review他人的PR",
            "参与Design Discussion",
            "分享使用经验",
            "建立长期合作关系"
        ]
    }

    for step, items in guide.items():
        print(f"\n【{step}】")
        for i, item in enumerate(items, 1):
            print(f"  {i}. {item}")

    # 生成贡献记录模板
    print("\n--- 贡献记录模板 ---")
    contribution_record = {
        "日期": datetime.now().strftime("%Y-%m-%d"),
        "项目": "项目名称",
        "类型": "Bug修复/功能开发/文档改进/其他",
        "描述": "简要描述贡献内容",
        "PR链接": "https://github.com/...",
        "状态": "已提交/审核中/已合并",
        "学到的": "记录从这次贡献中学到的知识",
        "下次改进": "下次可以改进的地方"
    }
    print(json.dumps(contribution_record, ensure_ascii=False, indent=2))

    # 保存贡献记录模板到文件
    output_file = os.path.join(os.path.dirname(__file__), "contribution_log_template.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(contribution_record, f, ensure_ascii=False, indent=2)
    print(f"\n贡献记录模板已保存到: {output_file}")


# =============================================
# 主程序
# =============================================
if __name__ == "__main__":
    print("=" * 60)
    print("  W35 Day 1 - 开源文化与贡献指南")
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    demonstrate_oss_culture()
    demonstrate_contribution_types()
    find_suitable_projects()
    demonstrate_github_tips()
    generate_contribution_guide()

    print("\n" + "=" * 60)
    print("  开源贡献是一次旅程, 从小处开始, 持续参与!")
    print("=" * 60)
