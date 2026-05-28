"""
W38 Day 2 - GitHub审查
=====================
主题: 检查所有项目一致性, 统一风格, 补充README检查
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
# 1. 项目一致性检查
# =============================================
def check_project_consistency():
    """检查所有项目的一致性"""
    print_section("项目一致性检查")

    consistency_checks = {
        "文件结构一致性": {
            "每个项目应包含": [
                "README.md - 项目说明",
                "requirements.txt - 依赖清单",
                ".gitignore - Git忽略规则",
                "LICENSE - 开源许可证",
                "src/ 或 项目名/ - 源代码目录",
                "tests/ - 测试目录(推荐)",
                "docs/ - 文档目录(可选)"
            ],
            "不应包含": [
                "__pycache__/ 目录",
                ".env 文件(含密钥)",
                "*.pyc 编译文件",
                "大型数据文件(用Git LFS或除外)",
                "IDE配置文件(.idea/, .vscode/)"
            ]
        },
        "代码风格一致性": {
            "命名规范": [
                "文件名: snake_case (如 data_processor.py)",
                "类名: PascalCase (如 DataProcessor)",
                "函数名: snake_case (如 process_data)",
                "常量: UPPER_SNAKE_CASE (如 MAX_BATCH_SIZE)",
                "变量: snake_case (如 batch_size)"
            ],
            "格式规范": [
                "缩进: 4个空格(不用Tab)",
                "行宽: 不超过120字符",
                "import顺序: 标准库 -> 第三方库 -> 本地模块",
                "字符串: 统一使用单引号或双引号",
                "文档字符串: 使用三引号"
            ]
        },
        "README模板一致性": {
            "必含章节": [
                "# 项目名称 + 一句话描述",
                "## 功能特点",
                "## 安装",
                "## 快速开始",
                "## 使用示例",
                "## 技术栈",
                "## 项目结构",
                "## 许可证"
            ],
            "可选章节": [
                "## 架构设计",
                "## 性能基准",
                "## 常见问题",
                "## 贡献指南",
                "## 更新日志"
            ]
        }
    }

    for category, checks in consistency_checks.items():
        print(f"\n【{category}】")
        for sub_category, items in checks.items():
            print(f"  {sub_category}:")
            for item in items:
                print(f"    - {item}")


# =============================================
# 2. README质量评分
# =============================================
def score_readme_quality():
    """README质量评分系统"""
    print_section("README质量评分系统")

    scoring = {
        "评分维度": {
            "标题和描述(15分)": {
                "15分": "标题清晰, 有项目Logo/Badge, 一句话描述精准",
                "10分": "有标题和描述, 但不够吸引人",
                "5分": "只有标题, 没有描述",
                "0分": "没有README或只有默认内容"
            },
            "安装说明(15分)": {
                "15分": "详细的安装步骤, 包含系统要求和常见问题",
                "10分": "有基本的安装命令",
                "5分": "安装说明不完整",
                "0分": "没有安装说明"
            },
            "使用示例(20分)": {
                "20分": "可运行的代码示例, 带输出结果",
                "15分": "有代码示例但没有输出",
                "10分": "有简单示例",
                "5分": "只有API说明没有示例"
            },
            "项目说明(20分)": {
                "20分": "架构图 + 技术栈 + 设计思路",
                "15分": "有技术栈和基本说明",
                "10分": "简单描述",
                "5分": "几乎没说明"
            },
            "文档完整性(15分)": {
                "15分": "API文档 + 教程 + FAQ",
                "10分": "有部分文档",
                "5分": "缺少文档",
                "0分": "完全没有文档"
            },
            "视觉呈现(15分)": {
                "15分": "截图 + GIF + 架构图 + Badge",
                "10分": "有一些图片",
                "5分": "纯文字",
                "0分": "格式混乱"
            }
        },
        "总分评估": {
            "90-100分": "优秀 - 可以直接展示给面试官",
            "70-89分": "良好 - 稍加改进即可",
            "50-69分": "一般 - 需要较多改进",
            "50分以下": "需要大幅改进"
        }
    }

    for category, items in scoring.items():
        print(f"\n【{category}】")
        for score, desc in items.items():
            if isinstance(desc, dict):
                print(f"  {score}:")
                for s, d in desc.items():
                    print(f"    {s}")
            else:
                print(f"  {score}: {desc}")


# =============================================
# 3. GitHub项目优化清单
# =============================================
def github_optimization_checklist():
    """GitHub项目优化清单"""
    print_section("GitHub项目优化清单")

    checklist = {
        "仓库设置": [
            "[ ] 项目描述和Website链接已填写",
            "[ ] Topics标签已添加(python, ai, llm, rag等)",
            "[ ] 主分支已设置为默认分支",
            "[ ] Issues和Discussions已启用",
            "[ ] GitHub Pages已配置(如有文档站)"
        ],
        "README优化": [
            "[ ] 添加项目Badge(Python版本, License, Stars)",
            "[ ] 添加项目截图或Demo GIF",
            "[ ] 安装命令可直接复制运行",
            "[ ] 快速开始代码可直接运行",
            "[ ] 添加架构图(Mermaid语法)",
            "[ ] 性能基准数据"
        ],
        "代码优化": [
            "[ ] 统一代码格式(black/isort)",
            "[ ] 添加类型注解",
            "[ ] 完善docstring",
            "[ ] 移除debug代码和硬编码",
            "[ ] 添加单元测试",
            "[ ] CI/CD流水线(GitHub Actions)"
        ],
        "文档优化": [
            "[ ] README完整(见评分标准)",
            "[ ] CHANGELOG.md记录版本变更",
            "[ ] CONTRIBUTING.md贡献指南",
            "[ ] API文档(如有公共API)",
            "[ ] 使用示例和教程"
        ],
        "安全检查": [
            "[ ] .gitignore包含敏感文件模式",
            "[ ] 没有提交密钥、密码或Token",
            "[ ] .env文件在.gitignore中",
            "[ ] 使用环境变量管理敏感配置",
            "[ ] 依赖版本固定(requirements.txt)"
        ]
    }

    for category, items in checklist.items():
        print(f"\n【{category}】")
        for item in items:
            print(f"  {item}")

    # 自动化工具推荐
    print("\n--- 自动化工具推荐 ---")
    tools = {
        "代码格式化": "black (格式化) + isort (import排序)",
        "代码检查": "flake8 / pylint / ruff",
        "类型检查": "mypy",
        "安全检查": "bandit / safety",
        "测试覆盖": "pytest + pytest-cov",
        "CI/CD": "GitHub Actions",
        "依赖管理": "pip-tools / poetry"
    }
    for tool, desc in tools.items():
        print(f"  {tool}: {desc}")


# =============================================
# 主程序
# =============================================
if __name__ == "__main__":
    print("=" * 60)
    print("  W38 Day 2 - GitHub审查")
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    check_project_consistency()
    score_readme_quality()
    github_optimization_checklist()

    print("\n" + "=" * 60)
    print("  统一的项目风格展示专业素养!")
    print("  建议: 按照评分标准逐个审查你的GitHub项目")
    print("=" * 60)
