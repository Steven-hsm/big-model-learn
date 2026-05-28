"""
W35 Day 6 - 功能贡献
====================
主题: 需求分析, 设计讨论, 代码实现, Review流程
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
# 1. 需求分析
# =============================================
def demonstrate_requirement_analysis():
    """展示需求分析方法"""
    print_section("需求分析")

    analysis = {
        "需求来源": {
            "Issue请求": "用户提交Feature Request",
            "Roadmap": "项目路线图中的计划功能",
            "自己发现": "使用中发现的功能缺口",
            "社区讨论": "论坛/Discord中的讨论"
        },
        "需求评估框架": {
            "用户价值": "这个功能对多少用户有用?",
            "技术可行性": "现有架构是否支持?",
            "实现复杂度": "需要多少工作量?",
            "维护成本": "长期维护的难度?",
            "与项目目标的契合度": "是否符合项目定位?"
        },
        "需求分析清单": [
            "1. 理解需求: 这个功能要解决什么问题?",
            "2. 用户场景: 谁会使用? 在什么场景下使用?",
            "3. 输入输出: 需要什么输入? 产生什么输出?",
            "4. 边界条件: 有哪些特殊情况需要处理?",
            "5. 性能要求: 对性能有什么要求?",
            "6. 兼容性: 是否影响现有API?",
            "7. 替代方案: 是否有其他实现方式?",
        ],
        "RFC模板(Request for Comments)": """## 功能提案: [功能名称]

### 摘要
一句话描述要实现的功能。

### 动机
为什么需要这个功能? 解决什么问题?
引用相关的Issue和讨论。

### 详细设计
#### API设计
```python
# 新的API接口
class NewFeature:
    def __init__(self, param1: str, param2: int = 10):
        ...

    def process(self, input: str) -> dict:
        ...
```

#### 实现方案
- 模块划分
- 数据流
- 关键算法

### 替代方案
考虑过的其他实现方式及为什么没选。

### 向后兼容性
是否影响现有API? 如何处理兼容性?

### 测试计划
如何测试这个功能?

### 时间估算
预计需要的时间和工作量。
"""
    }

    for category, items in analysis.items():
        print(f"\n【{category}】")
        if isinstance(items, dict):
            for key, desc in items.items():
                print(f"  {key}: {desc}")
        elif isinstance(items, list):
            for item in items:
                print(f"  {item}")
        elif isinstance(items, str):
            print(items)


# =============================================
# 2. 设计讨论
# =============================================
def demonstrate_design_discussion():
    """展示设计讨论方法"""
    print_section("设计讨论")

    discussion = {
        "讨论渠道": {
            "GitHub Issue": "正式的功能讨论, 有记录",
            "GitHub Discussion": "开放式讨论, 适合头脑风暴",
            "Discord/Slack": "实时讨论, 快速反馈",
            "开发者会议": "定期会议, 深入讨论",
            "邮件列表": "正式提案和公告"
        },
        "讨论流程": [
            "1. 在Issue/Discussion中发起讨论",
            "2. 描述功能和初步设计方案",
            "3. 邀请相关Maintainer参与",
            "4. 收集反馈, 迭代方案",
            "5. 达成共识, 获得批准",
            "6. 开始实现"
        ],
        "设计文档模板": {
            "标题": "功能名称 + 版本号",
            "状态": "Draft / In Review / Approved / Implemented",
            "作者": "你的名字",
            "审阅者": "Maintainer列表",
            "目标": "要实现什么",
            "非目标": "明确不在范围内的内容",
            "背景": "为什么需要这个功能",
            "设计": "详细的实现方案",
            "API变更": "新增/修改的API",
            "测试计划": "如何验证功能正确",
            "迁移计划": "用户如何迁移到新API"
        }
    }

    for category, items in discussion.items():
        print(f"\n【{category}】")
        if isinstance(items, dict):
            for key, desc in items.items():
                print(f"  {key}: {desc}")
        elif isinstance(items, list):
            for item in items:
                print(f"  {item}")


# =============================================
# 3. 代码实现
# =============================================
def demonstrate_implementation():
    """展示代码实现最佳实践"""
    print_section("代码实现最佳实践")

    practices = {
        "实现原则": [
            "YAGNI: 不要实现当前不需要的功能",
            "DRY: 不要重复自己, 提取公共逻辑",
            "KISS: 保持简单直接",
            "单一职责: 每个函数/类只做一件事",
            "渐进式实现: 先实现核心功能, 再逐步完善"
        ],
        "代码组织": {
            "模块划分": "按功能划分, 高内聚低耦合",
            "接口设计": "面向接口编程, 便于扩展",
            "错误处理": "优雅处理异常, 提供有意义的错误信息",
            "日志记录": "关键操作添加日志, 方便调试",
            "配置管理": "可配置的参数, 合理的默认值"
        },
        "实现步骤": [
            "1. 创建功能分支: git checkout -b feat/new-feature",
            "2. 先写测试(TDD): 定义期望行为",
            "3. 实现核心逻辑: 最小可工作实现",
            "4. 运行测试: 确保通过",
            "5. 重构优化: 改善代码质量",
            "6. 处理边界情况: 添加异常处理",
            "7. 添加文档: 文档字符串和注释",
            "8. 代码自审: 自我Review一遍"
        ],
        "Python代码示例": '''from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)


class TextProcessor:
    """文本处理器, 支持多种文本处理操作。

    Args:
        language: 处理语言, 默认为中文
        max_length: 最大处理长度, 默认512
        device: 运行设备, auto为自动选择

    Example:
        >>> processor = TextProcessor(language="zh", max_length=1024)
        >>> result = processor.process("要处理的文本")
        >>> print(result)
    """

    def __init__(
        self,
        language: str = "zh",
        max_length: int = 512,
        device: str = "auto"
    ):
        self.language = language
        self.max_length = max_length
        self._device = self._resolve_device(device)
        logger.info(f"TextProcessor初始化: language={language}, "
                     f"max_length={max_length}, device={self._device}")

    def process(
        self,
        text: str,
        options: Optional[Dict] = None
    ) -> Dict:
        """处理文本, 返回处理结果。

        Args:
            text: 要处理的文本
            options: 可选的处理选项

        Returns:
            包含处理结果的字典

        Raises:
            ValueError: 当文本为空或超过最大长度时
        """
        # 输入验证
        if not text or not text.strip():
            raise ValueError("文本不能为空")

        if len(text) > self.max_length:
            raise ValueError(
                f"文本长度({len(text)})超过最大限制({self.max_length})"
            )

        options = options or {}

        try:
            # 核心处理逻辑
            result = self._process_internal(text, options)
            logger.debug(f"处理完成: {len(result)}个结果")
            return result
        except Exception as e:
            logger.error(f"处理失败: {e}")
            raise

    def _process_internal(
        self,
        text: str,
        options: Dict
    ) -> Dict:
        """内部处理逻辑(私有方法)"""
        # 实现具体处理逻辑
        return {"text": text, "status": "processed"}

    @staticmethod
    def _resolve_device(device: str) -> str:
        """解析设备字符串"""
        try:
            import torch
            if device == "auto":
                return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pass
        return device
'''
    }

    for category, items in practices.items():
        print(f"\n【{category}】")
        if isinstance(items, list):
            for item in items:
                print(f"  {item}")
        elif isinstance(items, dict):
            for key, desc in items.items():
                print(f"  {key}: {desc}")
        elif isinstance(items, str):
            print(items)


# =============================================
# 4. Code Review流程
# =============================================
def demonstrate_code_review():
    """展示Code Review流程"""
    print_section("Code Review流程")

    review = {
        "Review前的准备": [
            "确保CI通过",
            "自我Review一遍代码",
            "编写清晰的PR描述",
            "添加必要的截图/视频",
            "关联相关Issue"
        ],
        "Review关注点": {
            "正确性": "逻辑是否正确? 是否处理了边界情况?",
            "可读性": "代码是否容易理解? 命名是否清晰?",
            "性能": "是否有性能问题? 是否有不必要的计算?",
            "安全性": "是否有安全漏洞? 输入是否验证?",
            "测试": "是否有足够的测试覆盖?",
            "文档": "是否有必要的文档和注释?",
            "风格": "是否遵循项目代码风格?",
            "设计": "整体设计是否合理? 是否过度设计?"
        },
        "应对Review反馈": {
            "原则": [
                "保持开放心态, 不要把批评当作个人攻击",
                "理解每条反馈的原因",
                "不确定的及时提问",
                "合理表达自己的观点"
            ],
            "常见反馈类型": {
                "必须修改(Blocker)": "必须修改才能合并",
                "建议修改(Suggestion)": "建议但不强制",
                "讨论(Discussion)": "开放讨论, 共同决策",
                "称赞(Praise)": "写得好的部分"
            }
        },
        "Review常用词汇": {
            "LGTM": "Looks Good To Me - 可以合并",
            "Nit": "小问题(命名、格式等)",
            "NACK": "不赞同这个方案",
            "ACK": "赞同这个方案",
            "WIP": "Work In Progress - 还在进行中",
            "PTAL": "Please Take A Look - 请查看"
        }
    }

    for category, items in review.items():
        print(f"\n【{category}】")
        if isinstance(items, list):
            for item in items:
                print(f"  {item}")
        elif isinstance(items, dict):
            for key, desc in items.items():
                if isinstance(desc, list):
                    print(f"  {key}:")
                    for item in desc:
                        print(f"    - {item}")
                elif isinstance(desc, dict):
                    print(f"  {key}:")
                    for k, v in desc.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {desc}")


# =============================================
# 5. 功能贡献完整Checklist
# =============================================
def generate_feature_checklist():
    """生成功能贡献Checklist"""
    print_section("功能贡献Checklist")

    checklist = {
        "需求分析阶段": [
            "[ ] 阅读相关Issue, 理解需求",
            "[ ] 在Issue中表达贡献意愿",
            "[ ] 与Maintainer讨论方案",
            "[ ] 获得实现许可"
        ],
        "设计阶段": [
            "[ ] 编写设计文档/RFC",
            "[ ] 定义API接口",
            "[ ] 讨论并确定实现方案",
            "[ ] 评估向后兼容性"
        ],
        "实现阶段": [
            "[ ] 创建功能分支",
            "[ ] 先编写测试用例(TDD)",
            "[ ] 实现核心功能",
            "[ ] 处理边界情况和异常",
            "[ ] 添加日志和文档字符串",
            "[ ] 代码自审"
        ],
        "测试阶段": [
            "[ ] 单元测试通过",
            "[ ] 集成测试通过",
            "[ ] 性能测试(如需要)",
            "[ ] 手动验证功能正常"
        ],
        "提交阶段": [
            "[ ] 代码风格检查通过",
            "[ ] CI流水线通过",
            "[ ] 编写清晰的PR描述",
            "[ ] 关联Issue",
            "[ ] 请求Review"
        ],
        "Review阶段": [
            "[ ] 响应Review反馈",
            "[ ] 修改代码并推送",
            "[ ] 获得LGTM",
            "[ ] 合并PR",
            "[ ] 清理分支"
        ]
    }

    for phase, items in checklist.items():
        print(f"\n【{phase}】")
        for item in items:
            print(f"  {item}")

    # 保存Checklist
    output_file = os.path.join(os.path.dirname(__file__), "feature_checklist.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(checklist, f, ensure_ascii=False, indent=2)
    print(f"\n  Checklist已保存: {output_file}")


# =============================================
# 主程序
# =============================================
if __name__ == "__main__":
    print("=" * 60)
    print("  W35 Day 6 - 功能贡献")
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    demonstrate_requirement_analysis()
    demonstrate_design_discussion()
    demonstrate_implementation()
    demonstrate_code_review()
    generate_feature_checklist()

    print("\n" + "=" * 60)
    print("  功能贡献是开源参与的高级形式!")
    print("  建议: 从小功能开始, 逐步承担更大的开发任务")
    print("=" * 60)
