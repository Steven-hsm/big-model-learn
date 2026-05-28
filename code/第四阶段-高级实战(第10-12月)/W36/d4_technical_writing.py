"""
W36 Day 4 - 技术写作
====================
主题: 案例研究模板, 教程写作模板, 深度分析模板
"""

import os
from datetime import datetime


def print_section(title):
    """打印分隔线"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# =============================================
# 1. 案例研究模板
# =============================================
def generate_case_study_template():
    """生成案例研究模板"""
    print_section("案例研究(Case Study)模板")

    template = """# 案例研究: [项目名称] - [一句话描述成果]

## 摘要
> 用2-3句话总结项目的背景、方案和成果

**关键词**: 关键词1, 关键词2, 关键词3

---

## 1. 项目背景

### 1.1 业务场景
描述业务背景和应用场景:
- 行业/领域: [行业名称]
- 用户群体: [目标用户]
- 业务痛点: [要解决的核心问题]

### 1.2 问题分析
详细描述遇到的问题:
- 问题1: [具体问题描述]
- 问题2: [具体问题描述]
- 影响: [对业务的影响, 最好有量化数据]

### 1.3 项目目标
明确的目标:
- [ ] 目标1: [量化目标, 如"准确率达到90%"]
- [ ] 目标2: [量化目标]
- [ ] 目标3: [量化目标]

---

## 2. 技术方案

### 2.1 架构设计
```mermaid
graph TB
    A[数据层] --> B[处理层]
    B --> C[模型层]
    C --> D[应用层]
```

### 2.2 技术选型
| 组件 | 选择 | 理由 |
|------|------|------|
| 语言 | Python | AI生态最丰富 |
| 框架 | FastAPI | 高性能异步 |
| 模型 | GPT-4 + 微调 | 效果与成本平衡 |
| 向量库 | ChromaDB | 轻量易用 |
| 部署 | Docker + K8s | 可扩展 |

### 2.3 核心实现

#### 数据处理
```python
def process_documents(documents):
    '''文档处理流程'''
    # 1. 解析
    parsed = parse_documents(documents)
    # 2. 分块
    chunks = split_into_chunks(parsed, chunk_size=512)
    # 3. 向量化
    embeddings = embed_chunks(chunks)
    # 4. 存储
    store_in_vector_db(embeddings)
```

#### 模型推理
```python
def inference(query):
    '''推理流程'''
    # 1. 检索
    context = retrieve(query, top_k=5)
    # 2. 构建Prompt
    prompt = build_prompt(query, context)
    # 3. 生成
    response = llm.generate(prompt)
    return response
```

---

## 3. 实验结果

### 3.1 评估指标
| 指标 | 基线 | 优化后 | 提升 |
|------|------|--------|------|
| 准确率 | 72% | 91% | +19% |
| 响应时间 | 3.5s | 1.2s | -66% |
| 用户满意度 | 65% | 89% | +24% |

### 3.2 A/B测试结果
展示A/B测试的数据和结论。

### 3.3 性能优化
| 优化措施 | 效果 |
|----------|------|
| KV Cache | 延迟降低40% |
| 批处理 | 吞吐提升3x |
| 量化(INT8) | 显存减少50% |

---

## 4. 经验与教训

### 4.1 成功经验
1. [经验1: 具体描述]
2. [经验2: 具体描述]

### 4.2 踩过的坑
1. [坑1: 问题描述 + 解决方案]
2. [坑2: 问题描述 + 解决方案]

### 4.3 如果重来
描述你会做出的不同选择。

---

## 5. 总结

### 5.1 关键成果
- [成果1]
- [成果2]

### 5.2 后续规划
- [ ] 规划1
- [ ] 规划2

---

## 参考资料
- [参考1]
- [参考2]
"""

    print(template)

    output_file = os.path.join(os.path.dirname(__file__), "case_study_template.md")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(template)
    print(f"\n  案例研究模板已保存: {output_file}")


# =============================================
# 2. 教程写作模板
# =============================================
def generate_tutorial_template():
    """生成教程写作模板"""
    print_section("教程(Tutorial)写作模板")

    template = """# 教程: [技术名称]从入门到实战

> **难度**: ★★☆☆☆ 初级 | **预计时间**: 60分钟 | **标签**: [标签1] [标签2]

## 你将学到什么

完成本教程后, 你将能够:
- ✅ 理解[核心概念]的原理
- ✅ 搭建[技术名称]开发环境
- ✅ 实现[具体功能]
- ✅ 部署[应用/模型]

## 前置知识

在开始之前, 你需要:
- Python基础(变量、函数、类)
- 基本的命令行操作
- (可选) [其他前置知识]

## 环境准备

### 系统要求
- Python 3.10+
- 8GB+ RAM (推荐16GB)
- (可选) NVIDIA GPU + CUDA

### 安装步骤

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate

# 安装依赖
pip install -r requirements.txt

# 验证安装
python -c "import torch; print(torch.__version__)"
```

<details>
<summary>依赖列表(requirements.txt)</summary>

```
torch>=2.0.0
transformers>=4.30.0
langchain>=0.1.0
```
</details>

---

## Part 1: 理解核心概念 (10分钟)

### 什么是[技术名称]?

用简单的类比解释:
> [技术名称]就像是[日常生活中的类比]...

### 工作原理

```mermaid
graph LR
    输入 --> 处理 --> 输出
```

**关键术语**:
| 术语 | 解释 |
|------|------|
| Term1 | 解释 |
| Term2 | 解释 |

---

## Part 2: 第一个程序 (15分钟)

### Step 1: 导入库

```python
import torch
from transformers import AutoTokenizer, AutoModel
```

> 💡 **提示**: 如果你没有GPU, 代码会自动使用CPU运行

### Step 2: 初始化

```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModel.from_pretrained("bert-base-chinese")
print("模型加载完成!")
```

**预期输出**:
```
模型加载完成!
```

### Step 3: 运行推理

```python
text = "这是一段测试文本"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
print(f"输出形状: {outputs.last_hidden_state.shape}")
```

**预期输出**:
```
输出形状: torch.Size([1, 12, 768])
```

> ⚠️ **注意**: 输出形状可能因输入文本长度不同而变化

---

## Part 3: 进阶功能 (20分钟)

### Step 4: [进阶功能1]

```python
def advanced_function(input_data):
    '''进阶功能实现'''
    # 实现代码
    pass
```

### Step 5: [进阶功能2]

```python
# 继续实现...
```

---

## Part 4: 完整项目 (15分钟)

### 完整代码

```python
"""
[项目名称] - 完整实现
"""
# 完整可运行代码
```

---

## 常见问题

<details>
<summary>Q: 遇到CUDA out of memory怎么办?</summary>

A: 尝试以下方法:
1. 减小batch_size
2. 使用梯度检查点
3. 使用更小的模型
4. 使用8-bit量化

</details>

<details>
<summary>Q: 模型下载速度太慢怎么办?</summary>

A: 使用镜像源:
```bash
export HF_ENDPOINT=https://hf-mirror.com
```
</details>

## 下一步

恭喜完成本教程! 接下来你可以:
- 📖 阅读[进阶教程链接]
- 🏗️ 尝试[挑战项目]
- 💬 在[讨论区]分享你的成果

## 参考资料
- [官方文档](链接)
- [相关论文](链接)
"""

    print(template)

    output_file = os.path.join(os.path.dirname(__file__), "tutorial_template.md")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(template)
    print(f"\n  教程模板已保存: {output_file}")


# =============================================
# 3. 深度分析模板
# =============================================
def generate_deep_analysis_template():
    """生成深度分析模板"""
    print_section("深度分析(Deep Dive)模板")

    template = """# 深度分析: [技术/论文/架构名称]

> **字数**: 5000-8000字 | **阅读时间**: 15-20分钟 | **难度**: 高级

## 引言: 为什么值得关注

[技术名称]在[领域]引起了广泛关注, 因为它解决了...

**本文价值**: 本文将从原理、实现、应用三个维度全面解析[技术名称]。

---

## 一、历史与背景

### 1.1 技术演进

```
技术A (2020) → 技术B (2022) → [本技术] (2024)
```

### 1.2 核心问题
[技术名称]试图解决的核心问题是什么?

---

## 二、原理详解

### 2.1 核心思想

用一句话概括: ...

### 2.2 数学基础

$$
\\text{核心公式} = f(x, \\theta)
$$

**公式解释**:
- $x$: 输入
- $\\theta$: 参数
- $f$: 变换函数

### 2.3 算法流程

```
输入: data
1. 预处理
2. 特征提取
3. 模型推理
4. 后处理
输出: result
```

### 2.4 关键创新

与之前的方法相比, [技术名称]的创新在于:
1. **创新1**: [描述]
2. **创新2**: [描述]
3. **创新3**: [描述]

---

## 三、代码实现

### 3.1 从零实现核心组件

```python
import torch
import torch.nn as nn

class CoreComponent(nn.Module):
    '''核心组件的简化实现'''

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        # ... 完整实现

    def forward(self, x):
        # Step 1: 线性变换
        # Step 2: 注意力计算
        # Step 3: 输出
        return output
```

### 3.2 与官方实现对比

| 方面 | 从零实现 | 官方实现 |
|------|----------|----------|
| 代码量 | 100行 | 2000行 |
| 性能 | 基准 | 5-10x |
| 功能 | 核心功能 | 完整功能 |

---

## 四、实验与分析

### 4.1 实验设置
- 数据集: [数据集名称]
- 评估指标: [指标列表]
- 基线方法: [方法列表]

### 4.2 结果对比

| 方法 | 指标1 | 指标2 | 指标3 |
|------|-------|-------|-------|
| 基线A | xx | xx | xx |
| 基线B | xx | xx | xx |
| **本方法** | **xx** | **xx** | **xx** |

### 4.3 消融实验

分析各组件的贡献:
- 去掉组件A: 性能下降xx%
- 去掉组件B: 性能下降xx%

---

## 五、应用场景

### 5.1 最佳适用场景
1. 场景1
2. 场景2

### 5.2 不适用场景
1. 场景3
2. 场景4

---

## 六、优缺点分析

### 优势
1. [优势1]
2. [优势2]

### 局限
1. [局限1]
2. [局限2]

---

## 七、未来展望

### 发展趋势
1. [趋势1]
2. [趋势2]

### 研究方向
1. [方向1]
2. [方向2]

---

## 总结

[技术名称]是[领域]的重要突破, 它通过[核心方法]解决了[问题]。
虽然存在[局限], 但在[场景]中展现了出色的表现。

**关键收获**:
1. [收获1]
2. [收获2]
3. [收获3]

## 参考资料
- [1] 论文原文: [链接]
- [2] 官方代码: [链接]
- [3] 相关博客: [链接]
"""

    print(template)

    output_file = os.path.join(os.path.dirname(__file__), "deep_analysis_template.md")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(template)
    print(f"\n  深度分析模板已保存: {output_file}")


# =============================================
# 主程序
# =============================================
if __name__ == "__main__":
    print("=" * 60)
    print("  W36 Day 4 - 技术写作")
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    generate_case_study_template()
    generate_tutorial_template()
    generate_deep_analysis_template()

    print("\n" + "=" * 60)
    print("  好的模板让写作效率翻倍!")
    print("  建议: 收集3-5篇你认为写得好的技术博客, 分析其结构")
    print("=" * 60)
