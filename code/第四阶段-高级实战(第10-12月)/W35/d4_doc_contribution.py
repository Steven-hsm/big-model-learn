"""
W35 Day 4 - 文档贡献
====================
主题: 改进文档, 翻译文档, 示例代码编写, 生成文档模板
"""

import os
from datetime import datetime


def print_section(title):
    """打印分隔线"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# =============================================
# 1. 文档贡献的重要性与类型
# =============================================
def demonstrate_doc_importance():
    """展示文档贡献的重要性"""
    print_section("文档贡献的重要性与类型")

    importance = {
        "为什么文档贡献重要": [
            "文档是项目的门面, 直接影响用户体验",
            "好的文档减少重复的问题, 降低维护成本",
            "文档贡献门槛低, 是参与开源的最佳入口",
            "帮助新用户快速上手, 扩大项目影响力",
            "Contributor中有30%以上从文档贡献开始"
        ],
        "文档类型": {
            "README.md": "项目首页, 快速了解项目",
            "CONTRIBUTING.md": "贡献指南, 如何参与项目",
            "API文档": "函数/类的详细说明",
            "Tutorials": "教程, 引导用户逐步学习",
            "How-to指南": "解决具体问题的步骤",
            "CHANGELOG.md": "版本变更记录",
            "CODE_OF_CONDUCT.md": "社区行为准则",
            "FAQ": "常见问题解答"
        }
    }

    print("\n为什么文档贡献重要:")
    for item in importance["为什么文档贡献重要"]:
        print(f"  - {item}")

    print("\n文档类型:")
    for doc_type, desc in importance["文档类型"].items():
        print(f"  {doc_type:25s} - {desc}")


# =============================================
# 2. 文档改进技巧
# =============================================
def demonstrate_doc_improvement():
    """展示文档改进技巧"""
    print_section("文档改进技巧")

    improvements = {
        "常见文档问题": [
            "过时的代码示例(API已变更但示例未更新)",
            "缺失的参数说明(函数参数没有解释)",
            "不清晰的描述(术语未解释, 逻辑跳跃)",
            "缺少错误处理说明(只展示成功路径)",
            "格式不一致(混用Markdown风格)",
            "链接失效(外部链接或内部链接404)"
        ],
        "改进方法": {
            "代码示例改进": {
                "改进前": "使用model.predict()进行预测",
                "改进后": """# 使用model.predict()进行预测
import numpy as np
from mymodel import MyModel

model = MyModel.load("model.pkl")
data = np.array([[1.0, 2.0, 3.0]])
result = model.predict(data)
print(f"预测结果: {result}")  # 输出: 预测结果: [0.95]"""
            },
            "参数说明改进": {
                "改进前": "learning_rate: 学习率",
                "改进后": """learning_rate (float, 默认=0.001):
    学习率, 控制参数更新的步长。
    - 建议范围: 1e-5 到 1.0
    - 较小的值训练更稳定但更慢
    - 较大的值训练更快但可能不稳定
    示例: learning_rate=0.0001"""
            }
        },
        "文档审查清单": [
            "[ ] 所有代码示例可运行",
            "[ ] 参数类型和默认值正确",
            "[ ] 返回值有说明",
            "[ ] 异常情况有文档",
            "[ ] 链接有效",
            "[ ] 格式一致",
            "[ ] 语言简洁清晰",
            "[ ] 中文文档用词准确统一"
        ]
    }

    for category, items in improvements.items():
        print(f"\n【{category}】")
        if isinstance(items, list):
            for item in items:
                print(f"  {item}")
        elif isinstance(items, dict):
            for key, value in items.items():
                print(f"  {key}:")
                if isinstance(value, dict):
                    for k, v in value.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"    {value}")


# =============================================
# 3. 翻译文档指南
# =============================================
def demonstrate_translation_guide():
    """展示翻译文档指南"""
    print_section("翻译文档指南")

    guide = {
        "翻译原则": [
            "准确性第一: 技术术语翻译准确",
            "可读性: 符合中文表达习惯",
            "一致性: 同一术语全文统一翻译",
            "完整性: 不遗漏原文内容",
            "及时性: 基于最新版本翻译"
        ],
        "技术术语对照表": {
            "Model": "模型",
            "Training": "训练",
            "Inference": "推理",
            "Fine-tuning": "微调",
            "Pre-training": "预训练",
            "Token": "Token (保留英文或译为'标记')",
            "Embedding": "嵌入 / Embedding",
            "Attention": "注意力机制",
            "Batch": "批次 / Batch",
            "Epoch": "轮次 / Epoch",
            "Loss": "损失",
            "Gradient": "梯度",
            "Overfitting": "过拟合",
            "Underfitting": "欠拟合",
            "Hyperparameter": "超参数",
            "Dataset": "数据集",
            "Pipeline": "管道 / Pipeline",
            "Framework": "框架",
            "Repository": "仓库",
            "Pull Request": "Pull Request / PR (保留英文)",
            "Commit": "提交 / Commit",
            "Branch": "分支",
            "Merge": "合并",
            "Deploy": "部署",
            "Container": "容器",
            "Orchestration": "编排"
        },
        "翻译工作流": [
            "1. 确认翻译范围(整篇/章节/段落)",
            "2. 建立术语表(统一关键术语翻译)",
            "3. 初译(逐段翻译, 保持代码不变)",
            "4. 校对(检查准确性, 修正错误)",
            "5. 润色(优化表达, 提升可读性)",
            "6. 审查(请母语者或领域专家审查)",
            "7. 提交PR(附上翻译说明)"
        ]
    }

    for category, items in guide.items():
        print(f"\n【{category}】")
        if isinstance(items, list):
            for item in items:
                print(f"  {item}")
        elif isinstance(items, dict):
            for en, cn in items.items():
                print(f"  {en:25s} -> {cn}")


# =============================================
# 4. 示例代码编写指南
# =============================================
def demonstrate_example_writing():
    """展示示例代码编写指南"""
    print_section("示例代码编写指南")

    guidelines = {
        "好示例的标准": [
            "可以直接运行(不缺少依赖和上下文)",
            "有清晰的注释说明每一步",
            "包含输入输出示例",
            "处理常见错误情况",
            "代码简洁, 聚焦核心功能",
            "使用有意义的变量名"
        ],
        "示例代码模板": """'''
XXX功能使用示例
=================
本示例展示如何使用xxx实现yyy功能。

前置条件:
  pip install transformers torch
'''

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def example_text_classification():
    \"\"\"文本分类示例\"\"\"

    # 1. 加载模型和分词器
    model_name = "bert-base-chinese"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # 2. 准备输入数据
    text = "这是一条测试文本"
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # 3. 模型推理
    with torch.no_grad():
        outputs = model(**inputs)

    # 4. 获取结果
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = predictions.argmax().item()

    print(f"输入文本: {text}")
    print(f"预测类别: {predicted_class}")
    print(f"置信度: {predictions[0][predicted_class]:.4f}")

    return predicted_class

if __name__ == "__main__":
    example_text_classification()
""",
        "不同类型示例": {
            "快速开始(Quickstart)": "最简代码, 3-5行实现核心功能",
            "教程示例(Tutorial)": "逐步引导, 每步有详细说明",
            "高级示例(Advanced)": "展示高级用法和自定义功能",
            "性能示例(Performance)": "展示优化技巧和最佳实践",
            "端到端示例(E2E)": "完整的应用场景, 从数据到结果"
        }
    }

    for category, items in guidelines.items():
        print(f"\n【{category}】")
        if isinstance(items, list):
            for item in items:
                print(f"  {item}")
        elif isinstance(items, str):
            print(items)
        elif isinstance(items, dict):
            for key, desc in items.items():
                print(f"  {key}: {desc}")


# =============================================
# 5. 生成文档模板
# =============================================
def generate_doc_templates():
    """生成各种文档模板"""
    print_section("文档模板生成")

    templates = {}

    # README模板
    templates["README_template.md"] = """# 项目名称

> 一句话描述项目功能

## 功能特点

- 功能1: 简要描述
- 功能2: 简要描述
- 功能3: 简要描述

## 快速开始

### 安装

```bash
pip install package-name
```

### 基本使用

```python
from package import main_function

result = main_function("input")
print(result)
```

## 文档

- [安装指南](docs/installation.md)
- [使用教程](docs/tutorial.md)
- [API参考](docs/api.md)
- [常见问题](docs/faq.md)

## 开发

```bash
# 克隆仓库
git clone https://github.com/username/repo.git
cd repo

# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest tests/
```

## 贡献

欢迎贡献! 请阅读 [贡献指南](CONTRIBUTING.md)。

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件。

## 致谢

感谢所有贡献者!
"""

    # API文档模板
    templates["api_doc_template.md"] = """# API文档

## 类名: `ClassName`

类的简要描述。

### 参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| param1 | str | 必填 | 第一个参数的描述 |
| param2 | int | 10 | 第二个参数的描述 |
| param3 | bool | False | 第三个参数的描述 |

### 方法

#### `method_name(param1, param2)`

方法的简要描述。

**参数:**
- `param1` (str): 描述
- `param2` (int): 描述

**返回值:**
- `dict`: 返回值的描述

**异常:**
- `ValueError`: 当xxx时抛出

**示例:**

```python
obj = ClassName(param1="hello")
result = obj.method_name(param1="world", param2=42)
print(result)
# 输出: {'key': 'value'}
```
"""

    # 教程模板
    templates["tutorial_template.md"] = """# 教程: XXX功能使用指南

## 前置条件

在开始之前, 请确保:
- Python 3.8+ 已安装
- 已安装所需依赖: `pip install xxx`

## 目标

通过本教程, 你将学会:
1. 如何xxx
2. 如何yyy
3. 如何zzz

## Step 1: 准备工作

```python
import necessary_libraries
```

说明...

## Step 2: 核心操作

```python
# 具体代码
```

说明...

## Step 3: 进阶用法

```python
# 进阶代码
```

说明...

## 常见问题

### Q: 遇到xxx错误怎么办?
A: 解决方案...

### Q: 如何yyy?
A: 方法...

## 总结

本教程覆盖了xxx的基本用法。更多内容请参考[API文档](api.md)。

## 下一步

- [ ] 高级教程
- [ ] 性能优化指南
- [ ] 自定义扩展
"""

    # 保存模板文件
    output_dir = os.path.dirname(__file__)
    for filename, content in templates.items():
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  已保存模板: {filepath}")


# =============================================
# 主程序
# =============================================
if __name__ == "__main__":
    print("=" * 60)
    print("  W35 Day 4 - 文档贡献")
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    demonstrate_doc_importance()
    demonstrate_doc_improvement()
    demonstrate_translation_guide()
    demonstrate_example_writing()
    generate_doc_templates()

    print("\n" + "=" * 60)
    print("  好的文档是项目成功的一半!")
    print("  建议: 从修复文档中的小错误开始, 逐步参与翻译和编写")
    print("=" * 60)
