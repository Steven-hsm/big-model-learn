"""
W20-D6 数据准备 (Data Preparation)
====================================
指令数据集格式(Alpaca/ShareGPT), 数据质量控制,
数据增强, 格式转换工具, 小数据集构建
"""

import sys
import re
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W20-D6 数据准备 (Data Preparation)")
print("=" * 60)

# ============================================================
# 1. 指令数据集格式
# ============================================================
print("\n--- 1. 指令数据集格式 ---")
print("""
  两种主流格式:

  1) Alpaca格式:
     {
       "instruction": "任务描述",
       "input": "输入内容(可选)",
       "output": "期望输出",
       "system": "系统提示(可选)"
     }

  2) ShareGPT格式 (多轮对话):
     {
       "conversations": [
         {"from": "system", "value": "你是一个AI助手"},
         {"from": "human", "value": "用户消息"},
         {"from": "gpt", "value": "助手回复"},
         {"from": "human", "value": "追问"},
         {"from": "gpt", "value": "继续回复"}
       ]
     }

  3) ChatML格式:
     {"messages": [
       {"role": "system", "content": "..."},
       {"role": "user", "content": "..."},
       {"role": "assistant", "content": "..."}
     ]}
""")


# ============================================================
# 2. 示例数据集
# ============================================================
print("\n--- 2. 创建示例数据集 ---")


class AlpacaDataset:
    """Alpaca格式数据集"""

    def __init__(self):
        self.data = [
            {"instruction": "解释以下技术概念", "input": "什么是RAG检索增强生成",
             "output": "RAG(检索增强生成)是一种结合信息检索和文本生成的技术。它的工作流程是: 1)从外部知识库检索相关文档 2)将检索到的文档作为上下文 3)由大语言模型基于上下文生成答案。RAG的主要优势包括减少幻觉、支持实时知识更新和可追溯来源。"},
            {"instruction": "对比两种技术", "input": "LoRA微调 vs 全量微调",
             "output": "LoRA微调只训练0.1-2%的参数，通过低秩矩阵分解实现。全量微调训练所有参数。LoRA的优势:内存需求低、训练速度快、可切换任务。全量微调的优势:效果最好、适合数据充足的场景。推荐优先使用LoRA。"},
            {"instruction": "编写Python代码", "input": "实现余弦相似度计算",
             "output": "import numpy as np\n\ndef cosine_similarity(a, b):\n    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)\n\n# 使用示例\nvec_a = np.array([1, 2, 3])\nvec_b = np.array([4, 5, 6])\nprint(f'余弦相似度: {cosine_similarity(vec_a, vec_b):.4f}')"},
            {"instruction": "回答问题", "input": "Transformer架构的核心是什么",
             "output": "Transformer的核心是自注意力机制(Self-Attention)。它通过Query、Key、Value三个矩阵计算序列中每个位置对所有其他位置的关注度，捕获长距离依赖关系。多头注意力(Multi-Head Attention)允许模型同时关注不同的表示子空间。"},
            {"instruction": "列出要点", "input": "向量数据库的选择标准",
             "output": "选择向量数据库的标准: 1)数据规模:百万级以下用ChromaDB，更大用Milvus 2)延迟要求:实时用HNSW索引 3)部署方式:自建用FAISS/Milvus，托管用Pinecone 4)是否需要混合检索:Weaviate支持 5)语言支持:多语言用Qdrant 6)成本:开源方案更经济"},
            {"instruction": "解释技术原理", "input": "为什么QLoRA能用4-bit训练",
             "output": "QLoRA的关键创新是NF4量化:将模型权重从fp16压缩到4-bit，节省75%内存。训练时动态反量化为fp16进行计算，只更新LoRA参数(保持fp16)。通过双重量化和分页优化器进一步节省内存。实验证明NF4量化的精度损失极小(<0.5%)。"},
            {"instruction": "给出建议", "input": "如何构建高质量的RAG系统",
             "output": "构建高质量RAG系统的建议: 1)数据质量:清洗文档，去除噪声 2)切分策略:使用语义切分保持完整性 3)嵌入模型:选择bge-large-zh等中文优秀模型 4)混合检索:BM25+向量+RRF融合 5)重排序:Cross-Encoder精确排序 6)评估:建立评估数据集持续优化"},
            {"instruction": "编写函数", "input": "Python实现BM25算法",
             "output": "import math\nfrom collections import Counter\n\ndef bm25_score(query_tokens, doc_tokens, df, n_docs, avgdl, k1=1.5, b=0.75):\n    tf = Counter(doc_tokens)\n    score = 0.0\n    for token in query_tokens:\n        if token in tf:\n            idf = math.log((n_docs - df.get(token, 0) + 0.5) / (df.get(token, 0) + 0.5) + 1)\n            score += idf * tf[token] * (k1 + 1) / (tf[token] + k1 * (1 - b + b * len(doc_tokens) / avgdl))\n    return score"},
        ]

    def __len__(self):
        return len(self.data)

    def format_prompt(self, item):
        """格式化为训练prompt"""
        if item.get('input'):
            return f"### 指令:\n{item['instruction']}\n\n### 输入:\n{item['input']}\n\n### 回答:\n{item['output']}"
        return f"### 指令:\n{item['instruction']}\n\n### 回答:\n{item['output']}"

    def to_chatml(self, item):
        """转换为ChatML格式"""
        return {
            "messages": [
                {"role": "system", "content": "你是一个专业的AI助手。"},
                {"role": "user", "content": f"{item['instruction']}\n{item.get('input', '')}".strip()},
                {"role": "assistant", "content": item['output']}
            ]
        }

    def to_sharegpt(self, item):
        """转换为ShareGPT格式"""
        return {
            "conversations": [
                {"from": "human", "value": f"{item['instruction']}\n{item.get('input', '')}".strip()},
                {"from": "gpt", "value": item['output']}
            ]
        }


dataset = AlpacaDataset()
print(f"  数据集大小: {len(dataset)}")
print(f"\n  Alpaca格式示例:")
print(f"  {json.dumps(dataset.data[0], ensure_ascii=False, indent=2)[:200]}...")

print(f"\n  ChatML格式示例:")
print(f"  {json.dumps(dataset.to_chatml(dataset.data[0]), ensure_ascii=False, indent=2)[:200]}...")


# ============================================================
# 3. 数据质量控制
# ============================================================
print("\n--- 3. 数据质量控制 ---")
print("""
  数据质量检查���单:
    1) 长度检查: 输入/输出不能太短或太长
    2) 重复检查: 去除重复或高度相似的数据
    3) 格式检查: 确保JSON格式正确
    4) 内容检查: 输出是否相关、准确
    5) 语言检查: 检测语言一致性
""")


class DataQualityChecker:
    """数据质量检查器"""

    def __init__(self, min_output_len=10, max_output_len=2000, similarity_threshold=0.9):
        self.min_output_len = min_output_len
        self.max_output_len = max_output_len
        self.similarity_threshold = similarity_threshold

    def check_length(self, data):
        """长度检查"""
        issues = []
        for i, item in enumerate(data):
            output_len = len(item['output'])
            if output_len < self.min_output_len:
                issues.append(f"  [{i}] 输出太短: {output_len}字")
            elif output_len > self.max_output_len:
                issues.append(f"  [{i}] 输出太长: {output_len}字")
        return issues

    def check_duplicates(self, data):
        """重复检查"""
        seen = set()
        duplicates = []
        for i, item in enumerate(data):
            key = item['instruction'] + item.get('input', '')
            if key in seen:
                duplicates.append(f"  [{i}] 与前面的数据重复")
            seen.add(key)
        return duplicates

    def check_empty_fields(self, data):
        """空字段检查"""
        issues = []
        for i, item in enumerate(data):
            if not item.get('instruction', '').strip():
                issues.append(f"  [{i}] instruction为空")
            if not item.get('output', '').strip():
                issues.append(f"  [{i}] output为空")
        return issues

    def run_all_checks(self, data):
        """运行所有检查"""
        print(f"\n  数据质量检查报告:")
        print(f"  总数据量: {len(data)}")

        # 长度检查
        length_issues = self.check_length(data)
        print(f"\n  长度问题: {len(length_issues)}")
        for issue in length_issues[:3]:
            print(issue)

        # 重复检查
        dup_issues = self.check_duplicates(data)
        print(f"\n  重复问题: {len(dup_issues)}")
        for issue in dup_issues[:3]:
            print(issue)

        # 空字段检查
        empty_issues = self.check_empty_fields(data)
        print(f"\n  空字段问题: {len(empty_issues)}")
        for issue in empty_issues[:3]:
            print(issue)

        # 统计
        output_lens = [len(item['output']) for item in data]
        print(f"\n  输出长度统计:")
        print(f"    平均: {np.mean(output_lens):.0f}字")
        print(f"    中位数: {np.median(output_lens):.0f}字")
        print(f"    最短: {min(output_lens)}字")
        print(f"    最长: {max(output_lens)}字")

        total_issues = len(length_issues) + len(dup_issues) + len(empty_issues)
        print(f"\n  总问题数: {total_issues}")
        print(f"  合格率: {(len(data) - total_issues) / len(data) * 100:.1f}%")

        return total_issues == 0


checker = DataQualityChecker()
checker.run_all_checks(dataset.data)

# ============================================================
# 4. 数据增强
# ============================================================
print("\n--- 4. 数据增强 ---")
print("""
  数据增强方法:
    1) 指令改写: 同一个任务用不同的表述
    2) 输入变换: 同义词替换、句子重组
    3) 输出改写: 保持语义但改变表达方式
    4) 反向生成: 从输出反推指令
""")


class DataAugmentor:
    """数据增强器"""

    def __init__(self):
        self.instruction_templates = {
            "解释": ["解释以下概念", "请说明", "什么是", "介绍一下", "帮我理解", "详细描述"],
            "对比": ["对比两种技术", "比较以下概念", "分析区别", "哪个更好", "有什么不同"],
            "代码": ["编写Python代码", "写一个函数实现", "用代码实现", "给出代码示例", "如何用Python实现"],
            "列举": ["列出要点", "总结关键点", "有哪些", "要点是什么", "请列举"],
            "建议": ["给出建议", "有什么建议", "如何做", "推荐方案", "最佳实践是什么"],
        }

    def augment_instruction(self, item):
        """改写指令"""
        original_inst = item['instruction']
        new_items = []

        for keyword, templates in self.instruction_templates.items():
            if keyword in original_inst:
                for template in templates[:2]:  # 每种类型取2个
                    if template != original_inst:
                        new_item = item.copy()
                        new_item['instruction'] = template
                        new_items.append(new_item)
                break

        return new_items

    def augment_dataset(self, data, factor=2):
        """数据集增强"""
        augmented = list(data)
        for item in data:
            new_items = self.augment_instruction(item)
            augmented.extend(new_items[:factor - 1])

        random.shuffle(augmented)
        return augmented


augmentor = DataAugmentor()
augmented_data = augmentor.augment_dataset(dataset.data, factor=3)
print(f"  原始数据量: {len(dataset.data)}")
print(f"  增强后数据量: {len(augmented_data)}")

# 展示增强效果
print(f"\n  增强示例:")
for item in augmented_data[:4]:
    print(f"    指令: {item['instruction']}, 输入: {item['input'][:20]}...")

# ============================================================
# 5. 格式转换工具
# ============================================================
print("\n--- 5. 格式转换工具 ---")


class FormatConverter:
    """数据格式转换器"""

    @staticmethod
    def alpaca_to_chatml(data):
        """Alpaca -> ChatML"""
        converted = []
        for item in data:
            user_msg = item['instruction']
            if item.get('input'):
                user_msg += f"\n{item['input']}"

            converted.append({
                "messages": [
                    {"role": "system", "content": "你是一个有帮助的AI助手。"},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": item['output']}
                ]
            })
        return converted

    @staticmethod
    def alpaca_to_sharegpt(data):
        """Alpaca -> ShareGPT"""
        converted = []
        for item in data:
            user_msg = item['instruction']
            if item.get('input'):
                user_msg += f"\n{item['input']}"

            converted.append({
                "conversations": [
                    {"from": "human", "value": user_msg},
                    {"from": "gpt", "value": item['output']}
                ]
            })
        return converted

    @staticmethod
    def chatml_to_training_text(data):
        """ChatML -> 训练文本"""
        texts = []
        for item in data:
            parts = []
            for msg in item['messages']:
                parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
            texts.append("\n".join(parts))
        return texts


converter = FormatConverter()
chatml_data = converter.alpaca_to_chatml(dataset.data)
sharegpt_data = converter.alpaca_to_sharegpt(dataset.data)
training_texts = converter.chatml_to_training_text(chatml_data)

print(f"  ChatML格式数量: {len(chatml_data)}")
print(f"  ShareGPT格式数量: {len(sharegpt_data)}")
print(f"  训练文本示例:\n  {training_texts[0][:100]}...")

# ============================================================
# 6. 可视化
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 子图1: 输出长度分布
ax1 = axes[0]
output_lens = [len(item['output']) for item in dataset.data]
instruction_lens = [len(item['instruction']) for item in dataset.data]

ax1.hist(output_lens, bins=15, alpha=0.7, label='输出长度', color='#3498db', edgecolor='black')
ax1.hist(instruction_lens, bins=15, alpha=0.7, label='指令长度', color='#e74c3c', edgecolor='black')
ax1.set_xlabel('字符数', fontsize=12)
ax1.set_ylabel('频次', fontsize=12)
ax1.set_title('数据长度分布', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 子图2: 任务类型分布
ax2 = axes[1]
task_types = Counter()
for item in dataset.data:
    inst = item['instruction']
    if "解释" in inst:
        task_types["解释概念"] += 1
    elif "对比" in inst:
        task_types["对比分析"] += 1
    elif "代码" in inst or "函数" in inst:
        task_types["代码生成"] += 1
    elif "要点" in inst or "列出" in inst:
        task_types["列举要点"] += 1
    elif "建议" in inst:
        task_types["提供建议"] += 1
    else:
        task_types["其他"] += 1

labels = list(task_types.keys())
sizes = list(task_types.values())
colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%', startangle=90)
ax2.set_title('任务类型分布', fontsize=14, fontweight='bold')

# 子图3: 数据增强效果
ax3 = axes[2]
augmentation_methods = ['原始数据', '指令改写', '输入变换', '反向生成', '组合增强']
data_counts = [len(dataset.data), len(dataset.data) * 2, len(dataset.data) * 2,
               len(dataset.data) * 3, len(dataset.data) * 5]
quality_scores = [0.90, 0.85, 0.82, 0.80, 0.78]

ax3_twin = ax3.twinx()
bars = ax3.bar(range(len(augmentation_methods)), data_counts, color='#3498db', alpha=0.7, edgecolor='black')
ax3_twin.plot(range(len(augmentation_methods)), quality_scores, 'r^-', linewidth=2, markersize=10, label='质量分数')
ax3.set_xticks(range(len(augmentation_methods)))
ax3.set_xticklabels(augmentation_methods, fontsize=9, rotation=20)
ax3.set_ylabel('数据量', fontsize=12)
ax3_twin.set_ylabel('质量分数', fontsize=12, color='red')
ax3.set_title('数据增强: 量 vs 质', fontsize=14, fontweight='bold')
ax3_twin.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W20/data_preparation.png', dpi=150, bbox_inches='tight')
print("\n  图表已保存: data_preparation.png")
plt.close()

# ============================================================
# 7. 小数据集构建建议
# ============================================================
print("\n--- 7. 小数据集构建建议 ---")
print("""
  最小可用数据集:
    - 50-100条高质量数据即可开始
    - 每种任务类型至少10条
    - 质量远比数量重要

  数据收集方法:
    1) 手动编写:  最可控, 质量最高
    2) GPT辅助:  用GPT-4生成, 人工审核
    3) 开源数据:  Alpaca, Belle, MOSS等
    4) 自我指令:  用种⼦任务自动生成更多
    5) 用户反馈:  从实际使用中收集

  质量优于数量:
    100条精心标注 > 10000条自动生成
""")

print("\n" + "=" * 60)
print("W20-D6 完成! 本节学习了微调数据的准备和质量控制")
print("=" * 60)
