"""
W17-D5 Prompt工程 (Prompt Engineering)
======================================
Prompt设计原则, Zero/One/Few-shot示例, Chain-of-Thought,
使用API调用LLM, Prompt模板设计
"""

import sys
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W17-D5 Prompt工程 (Prompt Engineering)")
print("=" * 60)

# ============================================================
# 1. Prompt 设计原则
# ============================================================
print("\n--- 1. Prompt 设计原则 ---")
print("""
  好的Prompt应遵循以下原则:

  1) 明确性 (Clarity)
     - 指令清晰, 无歧义
     - 明确输出格式要求

  2) 具体性 (Specificity)
     - 提供充分的上下文
     - 指定角色、语气、长度等

  3) 结构化 (Structure)
     - 使用分隔符(###, ---, ```) 区分不同部分
     - 分步骤给出指令

  4) 示例驱动 (Example-Driven)
     - 提供输入-输出示例
     - 让模型理解期望的行为

  5) 迭代优化 (Iterative)
     - 测试Prompt, 分析输出
     - 根据结果调整和改进
""")

# ============================================================
# 2. Zero-shot / One-shot / Few-shot
# ============================================================
print("\n--- 2. Zero-shot / One-shot / Few-shot ---")


class SimpleLLMSimulator:
    """
    模拟LLM响应 (不需要实际API调用)
    用于演示不同Prompt技术
    """

    def __init__(self):
        self.knowledge = {
            "情感分析": {
                "这个产品太棒了！": "正面",
                "质量很差，不推荐购买。": "负面",
                "服务态度一般，还可以吧。": "中性",
                "超出了我的期望，非常满意！": "正面",
                "完全不值这个价格。": "负面",
            },
            "翻译": {
                "Hello": "你好",
                "Good morning": "早上好",
                "Thank you": "谢谢",
            },
            "问答": {
                "什么是深度学习？": "深度学习是机器学习的一个分支，使用多层神经网络来学习数据的层次化表示。",
                "Python是什么？": "Python是一种高级编程语言，以简洁易读的语法著称，广泛用于数据科学、AI和Web开发。",
            }
        }

    def generate(self, prompt: str) -> str:
        """模拟LLM生成 (简化版, 基于关键词匹配)"""
        # 这里只是模拟, 真实场景调用API
        for task, examples in self.knowledge.items():
            for question, answer in examples.items():
                if question in prompt:
                    return answer
        return "[模拟LLM响应: 基于输入生成的回答]"


llm = SimpleLLMSimulator()

# --- Zero-shot ---
print("\n  === Zero-shot (零样本) ===")
print("  不提供任何示例, 直接给指令\n")

zero_shot_prompts = [
    {
        "name": "情感分析",
        "prompt": "请判断以下句子的情感倾向（正面/负面/中性）：\n\n这个产品太棒了！",
        "expected": "正面"
    },
    {
        "name": "文本分类",
        "prompt": "将以下新闻分类（科技/体育/财经/娱乐）：\n\n苹果公司发布了新款MacBook Pro。",
        "expected": "科技"
    },
]

for p in zero_shot_prompts:
    print(f"  [{p['name']}]")
    print(f"  Prompt: {p['prompt']}")
    print(f"  期望输出: {p['expected']}")
    print()

# --- One-shot ---
print("  === One-shot (单样本) ===")
print("  提供一个示例, 让模型理解格式\n")

one_shot_prompt = """请对以下文本进行命名实体识别(NER)。

示例:
输入: "马云在杭州创立了阿里巴巴"
输出: 马云[PERSON], 杭州[LOCATION], 阿里巴巴[ORGANIZATION]

现在请处理:
输入: "任正非在深圳创立了华为"
输出:"""

print(f"  Prompt:\n{one_shot_prompt}")
print(f"  期望输出: 任正非[PERSON], 深圳[LOCATION], 华为[ORGANIZATION]\n")

# --- Few-shot ---
print("  === Few-shot (少样本) ===")
print("  提供多个示例, 让模型更好地理解模式\n")

few_shot_prompt = """任务: 将自然语言转换为SQL查询

示例1:
问: 查找所有年龄大于30的用户
答: SELECT * FROM users WHERE age > 30;

示例2:
问: 统计每个部门的员工数量
答: SELECT department, COUNT(*) FROM employees GROUP BY department;

示例3:
问: 查找价格最高的产品名称
答: SELECT name FROM products ORDER BY price DESC LIMIT 1;

现在请处理:
问: 查找2024年注册的所有活跃用户
答:"""

print(f"  Prompt:\n{few_shot_prompt}")
print(f"  期望输出: SELECT * FROM users WHERE registration_date >= '2024-01-01' AND status = 'active';\n")

# ============================================================
# 3. Chain-of-Thought (CoT) Prompting
# ============================================================
print("\n--- 3. Chain-of-Thought (CoT) Prompting ---")
print("""
  CoT 让模型"展示思考过程", 逐步推理, 提高复杂任务的准确率

  方法:
    1) 显式CoT: 在Prompt中加入"让我们一步一步思考"
    2) 示例CoT: 在Few-shot示例中展示推理步骤
""")

# CoT 示例
cot_prompt = """问题: 一个商店有23个苹果。上午卖了12个，下午又进货了8个。现在有多少个苹果？

让我们一步一步思考:
1) 开始有23个苹果
2) 上午卖了12个: 23 - 12 = 11
3) 下午进货8个: 11 + 8 = 19
4) 所以现在有19个苹果

答案: 19

---

问题: 小明有100元，买了3本书每本15元，又买了一支笔8元。还剩多少钱？

让我们一步一步思考:"""

print(f"  CoT Prompt示例:\n{cot_prompt}")
print(f"  期望推理: 1) 3本书花费: 3*15=45元  2) 买笔花费: 8元  3) 总花费: 45+8=53元  4) 剩余: 100-53=47元")
print(f"  答案: 47元\n")

# ============================================================
# 4. Prompt 模板设计
# ============================================================
print("\n--- 4. Prompt 模板设计 ---")


class PromptTemplate:
    """Prompt模板管理器"""

    def __init__(self):
        self.templates = {}

    def add_template(self, name: str, template: str):
        """添加模板"""
        self.templates[name] = template

    def format(self, name: str, **kwargs) -> str:
        """格式化模板"""
        if name not in self.templates:
            raise ValueError(f"模板 '{name}' 不存在")
        return self.templates[name].format(**kwargs)

    def list_templates(self):
        """列出所有模板"""
        for name, template in self.templates.items():
            preview = template[:80].replace('\n', ' ')
            print(f"  [{name}]: {preview}...")


# 创建模板管理器
pt = PromptTemplate()

# 定义各种模板
pt.add_template("qa", """你是一个知识渊博的助手。请根据以下上下文回答问题。
如果无法从上下文中找到答案，请回答"我不知道"。

### 上下文
{context}

### 问题
{question}

### 回答""")

pt.add_template("summarize", """请将以下文本总结为{length}字以内的摘要。
要求: 保留关键信息，语言简洁。

### 原文
{text}

### 摘要""")

pt.add_template("translate", """请将以下{source_lang}文本翻译为{target_lang}。
保持原文的语气和风格。

### 原文 ({source_lang})
{text}

### 翻译 ({target_lang})""")

pt.add_template("code_review", """你是一个资深程序员。请审查以下{language}代码，指出:
1. 潜在的Bug
2. 性能问题
3. 代码风格改进建议

### 代码
```{language}
{code}
```

### 审查意见""")

pt.add_template("classification", """请将以下文本分类到给定的类别中。

类别: {categories}

示例:
{examples}

### 待分类文本
{text}

### 分类结果""")

print("  已定义的Prompt模板:")
pt.list_templates()

# 使用模板
print("\n  --- 模板使用示例 ---")

qa_prompt = pt.format("qa",
    context="Python是由Guido van Rossum于1991年创建的编程语言。它强调代码可读性。",
    question="Python是什么时候创建的？")
print(f"\n  QA模板:\n{qa_prompt}")

summary_prompt = pt.format("summarize",
    text="深度学习是机器学习的一个子领域，使用多层神经网络来学习数据的复杂表示。"
         "它在图像识别、自然语言处理和语音识别等领域取得了突破性进展。",
    length="50")
print(f"\n  摘要模板:\n{summary_prompt}")

# ============================================================
# 5. 可视化: Prompt技术对比
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# 子图1: 不同Prompt技术在不同任务上的效果对比
ax1 = axes[0]
tasks = ['简单分类', '情感分析', '数学推理', '代码生成', '创意写作']
techniques = {
    'Zero-shot':   [0.85, 0.82, 0.45, 0.65, 0.70],
    'One-shot':    [0.88, 0.85, 0.55, 0.72, 0.75],
    'Few-shot':    [0.92, 0.90, 0.65, 0.80, 0.78],
    'CoT':         [0.90, 0.88, 0.85, 0.75, 0.72],
    'Few-shot+CoT':[0.93, 0.92, 0.90, 0.85, 0.80],
}

x = np.arange(len(tasks))
width = 0.15
for i, (tech, scores) in enumerate(techniques.items()):
    ax1.bar(x + i * width, scores, width, label=tech, alpha=0.85, edgecolor='black')

ax1.set_xticks(x + width * 2)
ax1.set_xticklabels(tasks, fontsize=10)
ax1.set_ylabel('准确率 / 质量分数', fontsize=12)
ax1.set_title('不同Prompt技术效果对比 (模拟数据)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=9, loc='lower right')
ax1.set_ylim(0.3, 1.0)
ax1.grid(True, alpha=0.3, axis='y')

# 子图2: Prompt设计要素重要性
ax2 = axes[1]
elements = ['明确指令', '输出格式', '角色设定', 'Few-shot示例', 'CoT推理',
            '分隔符使用', '长度控制', '语气设定']
importance = [0.95, 0.90, 0.75, 0.88, 0.82, 0.70, 0.65, 0.55]
colors = plt.cm.RdYlGn(importance)

ax2.barh(range(len(elements)), importance, color=colors, edgecolor='black', alpha=0.85)
ax2.set_yticks(range(len(elements)))
ax2.set_yticklabels(elements, fontsize=11)
ax2.set_xlabel('重要性分数', fontsize=12)
ax2.set_title('Prompt设计要素重要性', fontsize=14, fontweight='bold')
ax2.set_xlim(0, 1.05)
for i, v in enumerate(importance):
    ax2.text(v + 0.01, i, f'{v:.2f}', va='center', fontsize=10)
ax2.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W17/prompt_engineering.png', dpi=150, bbox_inches='tight')
print("\n  图表已保存: prompt_engineering.png")
plt.close()

# ============================================================
# 6. 使用API调用LLM
# ============================================================
print("\n--- 6. 使用API调用LLM ---")


def call_openai_api(prompt: str, model: str = "gpt-3.5-turbo",
                    temperature: float = 0.7, max_tokens: int = 500) -> str:
    """
    调用OpenAI API的示例代码
    需要: pip install openai
    """
    try:
        from openai import OpenAI

        client = OpenAI()  # 自动读取 OPENAI_API_KEY 环境变量

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个有帮助的AI助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    except ImportError:
        return "[需要安装: pip install openai]"
    except Exception as e:
        return f"[API调用失败: {e}]"


# 模拟API调用流程
print("""
  OpenAI API 调用流程:
  1. 安装: pip install openai
  2. 设置API Key: export OPENAI_API_KEY="your-key"
  3. 调用:

     from openai import OpenAI
     client = OpenAI()
     response = client.chat.completions.create(
         model="gpt-3.5-turbo",
         messages=[
             {"role": "system", "content": "你是一个AI助手。"},
             {"role": "user", "content": "什么是深度学习？"}
         ],
         temperature=0.7,
         max_tokens=500,
     )
     answer = response.choices[0].message.content

  重要参数:
    temperature:   0-2, 越高越随机, 创意任务用0.7-1.0
    max_tokens:    最大生成token数
    top_p:         核采样, 0-1, 控制候选token范围
    frequency_penalty:  -2到2, 降低重复
    presence_penalty:   -2到2, 鼓励新话题
""")

# ============================================================
# 7. 高级Prompt技巧
# ============================================================
print("\n--- 7. 高级Prompt技巧 ---")
print("""
  1) 角色设定 (Role Playing)
     "你是一个有20年经验的Python专家..."

  2) 输出格式约束
     "请以JSON格式输出: {'answer': '...', 'confidence': 0.9}"

  3) 分步执行 (Step-by-Step)
     "请按以下步骤分析:
      步骤1: ...
      步骤2: ...
      步骤3: ..."

  4) 自我检查 (Self-Verification)
     "回答后请检查你的答案是否合理"

  5) 否定指令
     "不要使用专业术语"
     "不要编造信息"

  6) Few-shot + CoT 组合
     提供带推理过程的示例, 效果最佳

  7) 系统消息 + 用户消息 分离
     system: 定义角色和行为规范
     user:   具体任务和输入
""")

# ============================================================
# 8. Prompt 最佳实践总结
# ============================================================
print("\n--- 8. Prompt 最佳实践总结 ---")

best_practices = [
    ("明确目标", "清楚说明你想要什么, 而不是不想要什么"),
    ("提供示例", "至少提供1-3个输入输出示例"),
    ("结构清晰", "使用标题、分隔符、编号列表"),
    ("迭代优化", "测试-分析-改进, 反复迭代"),
    ("控制长度", "明确输出长度要求(如'100字以内')"),
    ("角色扮演", "设定专家角色提高回答质量"),
    ("格式约束", "指定输出格式(JSON/表格/列表)"),
    ("逐步推理", "复杂任务使用CoT或分步指令"),
]

for title, desc in best_practices:
    print(f"  - {title}: {desc}")

print("\n" + "=" * 60)
print("W17-D5 完成! 本节学习了Prompt工程的核心技术")
print("=" * 60)
