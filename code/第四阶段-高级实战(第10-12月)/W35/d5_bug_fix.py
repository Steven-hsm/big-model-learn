"""
W35 Day 5 - Bug修复
===================
主题: Issue分析模板, 复现步骤, 修复流程, 测试验证
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
# 1. Bug报告模板
# =============================================
def demonstrate_bug_report():
    """展示Bug报告模板"""
    print_section("Bug报告模板")

    bug_report_template = """## Bug描述
简要描述遇到的Bug。

## 环境信息
- OS: Windows 10 / Ubuntu 22.04 / macOS 14
- Python版本: 3.10.x
- 库版本: package==x.y.z
- 其他相关环境信息

## 复现步骤
1. 步骤一
2. 步骤二
3. 步骤三

## 期望行为
描述你期望发生的行为。

## 实际行为
描述实际发生的行为。

## 错误信息
```
完整的错误traceback
```

## 最小复现代码
```python
# 提供能复现Bug的最小代码
from package import SomeClass

obj = SomeClass()
obj.problematic_method()  # 触发Bug
```

## 附加信息
- 截图(如有)
- 相关的Issue/PR链接
- 任何可能有助于诊断的信息
"""

    print(bug_report_template)

    # 好的Bug报告 vs 差的Bug报告
    print("--- 好的Bug报告 vs 差的Bug报告 ---")
    comparison = {
        "差的Bug报告": {
            "标题": "程序崩溃了",
            "描述": "我运行程序的时候崩溃了",
            "问题": "缺少环境信息、复现步骤、错误信息"
        },
        "好的Bug报告": {
            "标题": "使用batch_size>32时Trainer在多GPU环境下OOM",
            "描述": "详细的环境、复现步骤、最小复现代码",
            "优点": "标题清晰、可复现、有完整信息"
        }
    }
    for quality, details in comparison.items():
        print(f"\n  【{quality}】")
        for key, value in details.items():
            print(f"    {key}: {value}")


# =============================================
# 2. Issue分析方法
# =============================================
def demonstrate_issue_analysis():
    """展示Issue分析方法"""
    print_section("Issue分析方法")

    analysis_framework = {
        "Issue分类": {
            "Bug": "功能异常, 需要修复",
            "Feature": "新功能请求",
            "Enhancement": "现有功能改进",
            "Documentation": "文档相关问题",
            "Question": "使用问题",
            "Performance": "性能问题"
        },
        "Issue评估维度": [
            "严重程度: Critical / High / Medium / Low",
            "影响范围: 影响多少用户, 哪些功能",
            "复现难度: 容易复现 / 偶发 / 无法复现",
            "修复难度: 简单 / 中等 / 复杂",
            "优先级: P0(紧急) / P1(高) / P2(中) / P3(低)"
        ],
        "分析步骤": [
            "1. 阅读Issue描述, 理解问题",
            "2. 检查是否已有相同Issue(duplicate)",
            "3. 尝试复现问题",
            "4. 定位相关代码",
            "5. 分析根本原因",
            "6. 设计修复方案",
            "7. 评估修复影响范围",
            "8. 在Issue中回复分析结果"
        ],
        "分析工具": [
            "git log --all --source -- <file>  # 查看文件修改历史",
            "git blame <file>  # 查看每行的最后修改者",
            "git log -S 'buggy_code'  # 搜索引入特定代码的提交",
            "git bisect  # 二分查找引入Bug的提交",
            "grep -rn 'error_pattern' --include='*.py'  # 搜索错误模式"
        ]
    }

    for category, items in analysis_framework.items():
        print(f"\n【{category}】")
        if isinstance(items, dict):
            for key, desc in items.items():
                print(f"  {key:15s}: {desc}")
        elif isinstance(items, list):
            for item in items:
                print(f"  {item}")


# =============================================
# 3. Bug复现与定位
# =============================================
def demonstrate_bug_reproduction():
    """展示Bug复现与定位方法"""
    print_section("Bug复现与定位")

    reproduction_guide = {
        "复现原则": [
            "最小化: 精简到最少的代码和步骤",
            "确定性: 每次都能稳定复现",
            "隔离性: 排除外部因素的干扰",
            "记录性: 详细记录每一步操作"
        ],
        "复现技巧": [
            "使用与报告者相同的环境和版本",
            "设置随机种子确保可复现: torch.manual_seed(42)",
            "使用Docker确保环境一致",
            "添加日志定位问题发生的位置",
            "使用二分法缩小代码范围"
        ],
        "定位方法": {
            "日志分析法": "添加详细日志, 追踪数据流和执行路径",
            "断点调试法": "在IDE中设置断点, 逐步执行观察变量",
            "打印调试法": "在关键位置打印变量值和类型",
            "对比分析法": "与正常工作的版本对比, 找出差异",
            "单元测试法": "编写测试用例精确定位问题"
        }
    }

    for category, items in reproduction_guide.items():
        print(f"\n【{category}】")
        if isinstance(items, list):
            for item in items:
                print(f"  {item}")
        elif isinstance(items, dict):
            for method, desc in items.items():
                print(f"  {method}: {desc}")

    # Bug复现示例代码
    print("\n--- Bug复现最小代码示例 ---")
    reproduction_code = '''
import torch
import numpy as np

def reproduce_bug():
    """复现Bug的最小代码示例"""
    # 设置随机种子确保可复现
    torch.manual_seed(42)
    np.random.seed(42)

    # 构造最小输入
    input_data = torch.randn(2, 3)  # 最小的batch_size

    # 执行有问题的操作
    try:
        from package import BuggyClass
        obj = BuggyClass()
        result = obj.method(input_data)
        print(f"结果: {result}")
    except Exception as e:
        print(f"Bug复现成功! 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    reproduce_bug()
'''
    print(reproduction_code)


# =============================================
# 4. 修复流程
# =============================================
def demonstrate_fix_workflow():
    """展示Bug修复流程"""
    print_section("Bug修复流程")

    workflow = {
        "修复前准备": [
            "1. 在Issue中声明你将修复此Bug",
            "2. 确认修复方案与Maintainer达成一致",
            "3. 创建修复分���: git checkout -b fix/issue-xxx",
            "4. 确保能复现Bug(编写失败的测试)"
        ],
        "修复步骤": [
            "1. 编写失败的测试用例(证明Bug存在)",
            "2. 实现修复代码",
            "3. 运行测试确认Bug已修复",
            "4. 运行完整测试套件确认没有回归",
            "5. 检查代码风格",
            "6. 更新相关文档(如有必要)"
        ],
        "修复原则": [
            "最小修改原则: 只修改必要的代码",
            "不引入新Bug: 修复不能破坏现有功能",
            "有测试覆盖: 每个修复都有对应测试",
            "代码可读: 添加必要的注释说明修复逻辑"
        ],
        "常见修复类型": {
            "空值/None处理": "添加None检查和默认值",
            "类型错误": "添加类型检查和转换",
            "边界条件": "处理空列表、零除等边界情况",
            "竞态条件": "添加锁或使用线程安全的数据结构",
            "内存泄漏": "确保资源正确释放",
            "API变更": "更新调用方式适配新API"
        }
    }

    for category, items in workflow.items():
        print(f"\n【{category}】")
        if isinstance(items, list):
            for item in items:
                print(f"  {item}")
        elif isinstance(items, dict):
            for fix_type, desc in items.items():
                print(f"  {fix_type}: {desc}")


# =============================================
# 5. 测试验证
# =============================================
def demonstrate_testing():
    """展示测试验证方法"""
    print_section("测试验证")

    testing_guide = {
        "测试金字塔": {
            "单元测试(Unit Tests)": {
                "数量": "最多",
                "速度": "最快",
                "范围": "单个函数/方法",
                "工具": "pytest, unittest"
            },
            "集成测试(Integration Tests)": {
                "数量": "中等",
                "速度": "中等",
                "范围": "模块间交互",
                "工具": "pytest, requests"
            },
            "端到端测试(E2E Tests)": {
                "数量": "最少",
                "速度": "最慢",
                "范围": "完整功能流程",
                "工具": "pytest, selenium"
            }
        },
        "Bug修复测试模板": '''import pytest

class TestBugFix:
    """Bug修复的测试用例"""

    def test_bug_reproduction(self):
        """测试: 能复现原始Bug(修复前应该失败)"""
        # Arrange
        input_data = ...

        # Act & Assert
        with pytest.raises(ExpectedError):
            buggy_function(input_data)

    def test_fix_works(self):
        """测试: 修复后功能正常"""
        # Arrange
        input_data = ...

        # Act
        result = fixed_function(input_data)

        # Assert
        assert result == expected_result

    def test_edge_cases(self):
        """测试: 边界情况"""
        # 空输入
        assert fixed_function([]) == default_value
        # None输入
        assert fixed_function(None) == default_value
        # 大输入
        large_input = list(range(100000))
        result = fixed_function(large_input)
        assert result is not None

    @pytest.mark.parametrize("input,expected", [
        ("normal", "expected1"),
        ("edge1", "expected2"),
        ("edge2", "expected3"),
    ])
    def test_various_inputs(self, input, expected):
        """参数化测试: 覆盖多种输入"""
        assert fixed_function(input) == expected
''',
        "验证清单": [
            "[ ] 修复测试通过",
            "[ ] 所有现有测试通过(无回归)",
            "[ ] 边界情况测试通过",
            "[ ] 手动验证修复有效",
            "[ ] 代码风格检查通过",
            "[ ] CI/CD流水线通过"
        ]
    }

    for category, items in testing_guide.items():
        print(f"\n【{category}】")
        if isinstance(items, dict):
            for key, value in items.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {value}")
        elif isinstance(items, list):
            for item in items:
                print(f"  {item}")
        elif isinstance(items, str):
            print(items)

    # 保存Bug修复Issue模板
    issue_template = {
        "bug_fix_checklist": {
            "issue_number": "",
            "description": "",
            "root_cause": "",
            "fix_description": "",
            "test_added": True,
            "all_tests_pass": True,
            "no_regression": True,
            "docs_updated": False
        }
    }
    output_file = os.path.join(os.path.dirname(__file__), "bug_fix_template.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(issue_template, f, ensure_ascii=False, indent=2)
    print(f"\n  Bug修复模板已保存: {output_file}")


# =============================================
# 主程序
# =============================================
if __name__ == "__main__":
    print("=" * 60)
    print("  W35 Day 5 - Bug修复")
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    demonstrate_bug_report()
    demonstrate_issue_analysis()
    demonstrate_bug_reproduction()
    demonstrate_fix_workflow()
    demonstrate_testing()

    print("\n" + "=" * 60)
    print("  Bug修复是深入理解代码的最佳方式!")
    print("  建议: 从good-first-issue开始, 积累经验后处理更复杂的Bug")
    print("=" * 60)
