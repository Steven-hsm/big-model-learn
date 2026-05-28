"""
W35 Day 3 - 第一个Pull Request
===============================
主题: Fork→Clone→Branch→Commit→PR完整流程, 生成PR模板
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
# 1. PR完整流程
# =============================================
def demonstrate_pr_workflow():
    """展示完整的PR提交流程"""
    print_section("Pull Request 完整流程")

    workflow = [
        {
            "步骤": "Step 0: 准备工作",
            "命令": [
                "git config --global user.name 'Your Name'",
                "git config --global user.email 'your.email@example.com'",
                "git config --global core.editor 'code --wait'",
                "# 设置SSH密钥 (如果尚未设置)",
                "ssh-keygen -t ed25519 -C 'your.email@example.com'",
                "cat ~/.ssh/id_ed25519.pub  # 复制到GitHub SSH设置"
            ],
            "说明": "确保Git环境配置正确, SSH密钥已添加到GitHub"
        },
        {
            "步骤": "Step 1: Fork项目",
            "命令": [
                "# 在GitHub网页上点击 Fork 按钮",
                "# 或使用GitHub CLI:",
                "gh repo fork owner/repo --clone=false"
            ],
            "说明": "将项目复制到自己的GitHub账号下, 形成独立副本"
        },
        {
            "步骤": "Step 2: Clone到本地",
            "命令": [
                "git clone https://github.com/YOUR_USERNAME/repo.git",
                "cd repo",
                "# 添加上游仓库(同步原始仓库的更新)",
                "git remote add upstream https://github.com/ORIGINAL_OWNER/repo.git",
                "git remote -v  # 验证远程仓库配置"
            ],
            "说明": "克隆Fork的仓库到本地, 并添加上游仓库地址"
        },
        {
            "步骤": "Step 3: 同步主分支",
            "命令": [
                "git checkout main",
                "git fetch upstream",
                "git merge upstream/main",
                "# 或者使用 rebase",
                "git rebase upstream/main"
            ],
            "说明": "确保本地代码与原始仓库同步, 避免冲突"
        },
        {
            "步骤": "Step 4: 创建功能分支",
            "命令": [
                "git checkout -b fix/issue-123-typo-in-docs",
                "# 分支命名规范:",
                "#   fix/xxx  - Bug修复",
                "#   feat/xxx - 新功能",
                "#   docs/xxx - 文档更新",
                "#   test/xxx - 测试相关",
                "#   refactor/xxx - 代码重构"
            ],
            "说明": "创建有意义的分支名, 关联Issue编号"
        },
        {
            "步骤": "Step 5: 编写代码",
            "命令": [
                "# 进行修改...",
                "# 编写/更新测试",
                "# 运行测试确保通过",
                "pytest tests/",
                "# 运行代码风格检查",
                "flake8/ --max-line-length=120",
                "black --check .",
                "# 本地验证功能正常"
            ],
            "说明": "编写代码并确保通过所有测试和风格检查"
        },
        {
            "步骤": "Step 6: 提交代码",
            "命令": [
                "git add specific_file.py  # 添加修改的文件",
                "# 不要使用 git add . (可能添加不需要的文件)",
                "git status  # 检查暂存区",
                "git commit -m 'fix: correct typo in documentation'",
                "# Commit message规范(Conventional Commits):",
                "#   feat: 新功能",
                "#   fix: Bug修复",
                "#   docs: 文档更新",
                "#   style: 代码格式(不影响功能)",
                "#   refactor: 代码重构",
                "#   test: 测试相关",
                "#   chore: 构建/工具变更"
            ],
            "说明": "编写清晰、规范的commit message"
        },
        {
            "步骤": "Step 7: 推送分支",
            "命令": [
                "git push origin fix/issue-123-typo-in-docs",
                "# 如果之前已经推送过, 可能需要:",
                "git push origin fix/issue-123-typo-in-docs --force-with-lease"
            ],
            "说明": "将分支推送到你的Fork仓库"
        },
        {
            "步骤": "Step 8: 创建Pull Request",
            "命令": [
                "# 在GitHub网页上点击 'New Pull Request'",
                "# 或使用GitHub CLI:",
                "gh pr create --title 'fix: correct typo in documentation' \\",
                "  --body 'Fixes #123\\n\\n## Changes\\n- Fixed typo in xyz.md'\\",
                "  --base main",
                "# 或使用浏览器"
            ],
            "说明": "创建PR, 填写完整的描述, 关联Issue"
        },
        {
            "步骤": "Step 9: 代码审查",
            "命令": [
                "# 等待Review反馈",
                "# 根据反馈修改代码:",
                "git add .",
                "git commit -m 'fix: address review comments'",
                "git push origin fix/issue-123-typo-in-docs",
                "# PR会自动更新"
            ],
            "说明": "响应审查意见, 及时修改代码"
        },
        {
            "步骤": "Step 10: 合并后清理",
            "命令": [
                "# PR合并后删除分支",
                "git checkout main",
                "git pull upstream main",
                "git branch -d fix/issue-123-typo-in-docs",
                "git push origin --delete fix/issue-123-typo-in-docs",
                "# 删除GitHub上的分支(也可在PR页面操作)"
            ],
            "说明": "清理已合并的分支, 保持仓库整洁"
        }
    ]

    for step in workflow:
        print(f"\n【{step['步骤']}】")
        print(f"  说明: {step['说明']}")
        print("  命令:")
        for cmd in step["命令"]:
            print(f"    {cmd}")


# =============================================
# 2. 常见问题与解决方案
# =============================================
def demonstrate_common_issues():
    """展示PR过程中的常见问题"""
    print_section("常见问题与解决方案")

    issues = {
        "合并冲突": {
            "问题": "你的分支与目标分支有冲��",
            "解决": [
                "git fetch upstream",
                "git rebase upstream/main",
                "# 手动解决冲突文件",
                "git add <resolved-files>",
                "git rebase --continue",
                "git push origin <branch> --force-with-lease"
            ]
        },
        "CI检查失败": {
            "问题": "自动化测试或代码风格检查未通过",
            "解决": [
                "查看CI日志, 定位失败原因",
                "在本地复现并修复",
                "运行完整测试套件确认",
                "提交修复并推送"
            ]
        },
        "Review要求修改": {
            "问题": "审查者提出了修改建议",
            "解决": [
                "仔细阅读每条建议",
                "在本地修改代码",
                "提交并推送(不要新开PR)",
                "在评论中回复每条建议的处理结果"
            ]
        },
        "上游有新变更": {
            "问题": "原始仓库在你工作期间有新提交",
            "解决": [
                "git fetch upstream",
                "git rebase upstream/main",
                "解决可能的冲突",
                "git push origin <branch> --force-with-lease"
            ]
        }
    }

    for issue, details in issues.items():
        print(f"\n【{issue}】")
        print(f"  问题: {details['问题']}")
        print("  解决步骤:")
        for step in details["解决"]:
            print(f"    {step}")


# =============================================
# 3. 生成PR模板
# =============================================
def generate_pr_templates():
    """生成各种PR模板"""
    print_section("PR模板生成器")

    # Bug修复PR模板
    bug_fix_template = """## Bug修复

### 问题描述
简要描述修复的Bug, 关联Issue编号。

Fixes #{issue_number}

### 问题原因
分析Bug产生的根本原因。

### 修复方案
描述修复方案和实现方式。

### 变更内容
- [ ] 修复了xxx
- [ ] 添加了xxx
- [ ] 更新了xxx

### 测试
- [ ] 添加了单元测试
- [ ] 所有现有测试通过
- [ ] 手动验证修复有效

### 截图(如适用)
修复前后的对比截图。
"""

    # 功能开发PR模板
    feature_template = """## 新功能

### 功能描述
简要描述新增功能。

Closes #{issue_number}

### 实现方案
描述技术方案和设计决策。

### 变更内容
- [ ] 新增xxx功能
- [ ] 添加了相关测试
- [ ] 更新了文档
- [ ] 更新了CHANGELOG

### 使用示例
```python
# 展示新功能的使用方式
```

### 性能影响
分析对性能的影响(如果有)。

### Checklist
- [ ] 代码遵循项目风格指南
- [ ] 进行了自我Review
- [ ] 添加了必要的注释
- [ ] 文档已更新
- [ ] 没有引入新的warning
"""

    # 文档更新PR模板
    docs_template = """## 文档更新

### 更新内容
简要描述文档更新内容。

### 变更类型
- [ ] 错误修正
- [ ] 内容补充
- [ ] 格式优化
- [ ] 翻译

### 变更详情
具体列出修改的文件和内容。

### 验证
- [ ] 本地构建文档通过
- [ ] 链接有效
- [ ] 代码示例可运行
"""

    templates = {
        "Bug修复模板": bug_fix_template,
        "功能开发模板": feature_template,
        "文档更新模板": docs_template
    }

    for name, template in templates.items():
        print(f"\n【{name}】")
        print(template)

    # 保存模板文件
    templates_dir = os.path.dirname(__file__)
    template_files = {
        "pr_template_bug_fix.md": bug_fix_template,
        "pr_template_feature.md": feature_template,
        "pr_template_docs.md": docs_template
    }

    for filename, content in template_files.items():
        filepath = os.path.join(templates_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  已保存: {filepath}")


# =============================================
# 4. Commit Message规范
# =============================================
def demonstrate_commit_conventions():
    """展示Commit Message规范"""
    print_section("Commit Message规范")

    print("\n--- Conventional Commits格式 ---")
    print("""
格式: <type>(<scope>): <subject>

<body>

<footer>

类型(type):
  feat:     新功能
  fix:      Bug修复
  docs:     文档变更
  style:    代码格式(不影响功能)
  refactor: 代码重构
  perf:     性能优化
  test:     测试相关
  build:    构建系统或外部依赖
  ci:       CI配置变更
  chore:    其他不修改src或test的变更

范围(scope): 可选, 表示影响的模块
主题(subject): 简短描述, 不超过50字符
正文(body): 详细描述, 每行不超过72字符
页脚(footer): 关联Issue(Breaking Changes等)
""")

    examples = [
        "fix(tokenizer): handle empty input correctly",
        "feat(model): add support for LLaMA 3",
        "docs(readme): update installation instructions",
        "perf(inference): optimize attention computation",
        "test(pipeline): add tests for text classification",
        "refactor(utils): extract common helper functions",
        "fix: resolve memory leak in batch processing\n\nCloses #456"
    ]
    print("--- 示例 ---")
    for ex in examples:
        print(f"  {ex}")


# =============================================
# 主程序
# =============================================
if __name__ == "__main__":
    print("=" * 60)
    print("  W35 Day 3 - 第一个Pull Request")
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    demonstrate_pr_workflow()
    demonstrate_common_issues()
    generate_pr_templates()
    demonstrate_commit_conventions()

    print("\n" + "=" * 60)
    print("  第一个PR是最重要的开始, 不要怕犯错!")
    print("  建议: 从文档修复或good-first-issue开始")
    print("=" * 60)
