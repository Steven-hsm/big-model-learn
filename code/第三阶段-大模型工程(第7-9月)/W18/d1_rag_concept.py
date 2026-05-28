"""
W18-D1 RAG概念与动机 (RAG Concept & Motivation)
================================================
RAG���念和动机(解决幻觉/知识更新), RAG vs 微调对比,
RAG系统架构图, 简单关键词检索demo
"""

import sys
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W18-D1 RAG概念与动机 (RAG Concept & Motivation)")
print("=" * 60)

# ============================================================
# 1. RAG 概念和动机
# ============================================================
print("\n--- 1. RAG 概念和动机 ---")
print("""
  RAG = Retrieval-Augmented Generation (检索增强生成)

  论文: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
       (Lewis et al., 2020, Meta AI)

  ��心思想:
    在生成回答前, 先从外部知识库检索相关文档,
    然后将检索到的文档作为上下文提供给LLM生成答案

  为什么需要RAG?  LLM的三大问题:
    1) 幻觉 (Hallucination): 模型可能编造不存在的事实
    2) 知识过时:  训练数据有截止日期, 无法回答新信息
    3) 缺乏领域知识: 通用模型对专业领域了解有限

  RAG的优势:
    + 事实性更强 (基于检索到的真实文档)
    + 知识可实时更新 (更新知识库即可)
    + 可追溯来源 (知道答案出自哪个文档)
    + 无需重新训练模型
""")

# ============================================================
# 2. RAG vs 微调对比
# ============================================================
print("\n--- 2. RAG vs 微调对比 ---")
print("""
  +------------------+------------------------+------------------------+
  |                  |        RAG             |       微调(Fine-tune)  |
  +------------------+------------------------+------------------------+
  | 知识更新         | 更新知识库即可, 即时   | 需要重新训练           |
  | 成本             | 较低 (无需训练)        | 较高 (需要GPU训练)     |
  | 幻觉问题         | 大幅减少               | 仍然存在               |
  | 来源追踪         | 可以 (知道用了哪些文档) | 不可以                  |
  | 领域适应         | 适合知识密集型任务     | 适合风格/格式适应      |
  | 实时性           | 支持实时数据           | 受训练数据截止日期限制  |
  | 风格调整         | 较弱                   | 强 (可调整输出风格)    |
  | 推理速度         | 稍慢 (需要检索步骤)    | 较快 (直接推理)        |
  | 数据需求         | 文档/知识库            | 标注的指令数据         |
  | 适用场景         | QA/客服/知识管理       | 特定任务/风格迁移      |
  +------------------+------------------------+------------------------+

  最佳实践: RAG + 微调 组合使用!
    1) 微调: 让模型学习回答的风格和格式
    2) RAG:  提供最新、准确的知识
""")

# ============================================================
# 3. RAG 系统架构
# ============================================================
print("\n--- 3. RAG 系统架构 ---")
print("""
  完整RAG系统流程:

  知识库构建阶段 (离线):
  ================================
  文档集合
    |-- 文档加载 (PDF/HTML/TXT/DB)
    |-- 文本切分 (Chunking)
    |-- 向量化 (Embedding)
    |-- 存入向量数据库 (Vector DB)

  查询阶段 (在线):
  ================================
  用户问题 (Query)
    |-- Query预处理/改写
    |-- 向量化 (Embedding)
    |-- 向量检索 (Top-K相似文档)
    |-- (可选) 重排序 (Reranking)
    |-- 拼接上下文 (Context)
    |-- LLM生成回答
    |-- 返回答案 (+ 来源)

  关键组件:
    1) 文档处理:    加载、清洗、切分
    2) Embedding:   将文本转为向量
    3) 向量数据库:  存储和高效检索
    4) LLM:         基于上下文生成回答
    5) Reranker:    对检索结果重排序
""")

# ============================================================
# 4. 简单关键词检索Demo
# ============================================================
print("\n--- 4. 简单关键词检索 Demo ---")


class SimpleKeywordSearch:
    """简单的基于关键词的检索系统"""

    def __init__(self):
        self.documents = []
        self.inverted_index = defaultdict(list)  # 词 -> 文档ID列表
        self.doc_freq = defaultdict(int)          # 词 -> 出现文档数

    def add_document(self, doc_id: int, text: str):
        """添加文档到索引"""
        self.documents.append({"id": doc_id, "text": text})
        # 简单分词: 转小写, 去除标点, 按空格和中文切分
        words = set(self._tokenize(text))
        for word in words:
            self.inverted_index[word].append(doc_id)
            self.doc_freq[word] += 1

    def _tokenize(self, text):
        """简单分词"""
        text = text.lower()
        # 中文字符逐字 + 英文单词
        tokens = re.findall(r'[一-鿿]|[a-z]+|[0-9]+', text)
        return tokens

    def search(self, query: str, top_k: int = 3):
        """关键词检索"""
        query_tokens = self._tokenize(query)
        doc_scores = defaultdict(float)
        total_docs = len(self.documents)

        for token in query_tokens:
            if token in self.inverted_index:
                # TF-IDF-like评分
                idf = np.log(total_docs / (1 + self.doc_freq.get(token, 0)))
                for doc_id in self.inverted_index[token]:
                    doc_text = self.documents[doc_id]["text"]
                    tf = doc_text.lower().count(token)
                    doc_scores[doc_id] += tf * idf

        # 排序
        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for doc_id, score in ranked:
            results.append({
                "doc_id": doc_id,
                "text": self.documents[doc_id]["text"],
                "score": score
            })
        return results


# 构建知识库
print("  构建知识库...")
search_engine = SimpleKeywordSearch()

docs = [
    (0, "Python是一种高级编程语言，由Guido van Rossum于1991年创建。Python强调代码可读性和简洁性。"),
    (1, "深度学习是机器学习的一个分支，使用多层神经网络来学习数据的层次化表示。深度学习在图像识别、自然语言处理等领域取得突破。"),
    (2, "Transformer架构由Google在2017年提出，完全基于注意力机制，是现代大语言模型的基础。"),
    (3, "RAG检索增强生成技术通过检索外部知识库来增强大语言模型的回答质量和事实性。"),
    (4, "LoRA是一种参数高效微调方法，通过低秩分解来减少可训练参数数量，使得在消费级GPU上微调大模型成为可能。"),
    (5, "向量数据库如FAISS、Milvus、ChromaDB专门用于存储和检索高维向量，是RAG系统的核心组件。"),
    (6, "Python的NumPy库提供了高效的多维数组操作，是科学计算和机器学习的基础库。"),
    (7, "自然语言处理NLP是人工智能的重要方向，包括文本分类、命名实体识别、机器翻译等任务。"),
]

for doc_id, text in docs:
    search_engine.add_document(doc_id, text)

print(f"  已索引 {len(docs)} 篇文档")

# 测试检索
queries = ["Python编程", "深度学习神经网络", "RAG检索", "向量数据库"]
for query in queries:
    print(f"\n  查询: '{query}'")
    results = search_engine.search(query, top_k=3)
    for r in results:
        print(f"    [文档{r['doc_id']}] 分数={r['score']:.2f}: {r['text'][:60]}...")

# ============================================================
# 5. 可视化
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 子图1: RAG vs 微调 对比雷达图
ax1 = axes[0]
categories = ['知识更新', '成本效率', '事实准确', '来源追踪', '实时性', '风格控制', '推理速度']
rag_scores = [9, 8, 9, 9, 9, 4, 5]
finetune_scores = [3, 4, 6, 2, 2, 9, 8]

x = np.arange(len(categories))
width = 0.35
bars1 = ax1.bar(x - width/2, rag_scores, width, label='RAG', color='#3498db', edgecolor='black', alpha=0.85)
bars2 = ax1.bar(x + width/2, finetune_scores, width, label='微调', color='#e74c3c', edgecolor='black', alpha=0.85)

ax1.set_xticks(x)
ax1.set_xticklabels(categories, fontsize=10, rotation=30, ha='right')
ax1.set_ylabel('评分 (1-10)', fontsize=12)
ax1.set_title('RAG vs 微调 能力对比', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.set_ylim(0, 11)
ax1.grid(True, alpha=0.3, axis='y')

# 子图2: RAG处理流程
ax2 = axes[1]
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_title('RAG系统处理流程', fontsize=14, fontweight='bold')

# 绘制流程框
boxes = [
    (5, 9.2, "用户问题", '#3498db'),
    (5, 8.0, "Query向量化", '#2ecc71'),
    (5, 6.8, "向量检索 (Top-K)", '#e67e22'),
    (5, 5.6, "重排序 (可选)", '#9b59b6'),
    (5, 4.4, "拼接上下文", '#1abc9c'),
    (5, 3.2, "LLM生成回答", '#e74c3c'),
    (5, 2.0, "返回答案+来源", '#3498db'),
]

# 侧边: 知识库
ax2.add_patch(plt.Rectangle((8, 4.5), 1.8, 4.5, fill=True, facecolor='#f39c12',
                              edgecolor='black', alpha=0.3, linewidth=2))
ax2.text(8.9, 6.75, '知识库\n向量DB', ha='center', va='center', fontsize=11, fontweight='bold')

for x_pos, y_pos, text, color in boxes:
    ax2.add_patch(plt.Rectangle((x_pos - 1.8, y_pos - 0.4), 3.6, 0.7,
                                  fill=True, facecolor=color, edgecolor='black',
                                  alpha=0.3, linewidth=2, zorder=2))
    ax2.text(x_pos, y_pos, text, ha='center', va='center', fontsize=11,
             fontweight='bold', zorder=3)

# 箭头
for i in range(len(boxes) - 1):
    ax2.annotate('', xy=(5, boxes[i + 1][1] + 0.4), xytext=(5, boxes[i][1] - 0.4),
                 arrowprops=dict(arrowstyle='->', color='black', lw=2))

# 检索箭头
ax2.annotate('', xy=(6.8, 7.0), xytext=(8, 7.0),
             arrowprops=dict(arrowstyle='->', color='#e67e22', lw=2, linestyle='dashed'))

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W18/rag_concept.png', dpi=150, bbox_inches='tight')
print("\n  图表已保存: rag_concept.png")
plt.close()

# ============================================================
# 6. RAG系统的类型
# ============================================================
print("\n--- 6. RAG系统的类型 ---")
print("""
  1) Naive RAG (基础RAG)
     简单的: 检索 -> 拼接 -> 生成
     优点: 实现简单
     缺点: 检索质量不稳定, 可能引入噪声

  2) Advanced RAG (进阶RAG)
     增加了: 查询改写, 混合检索, 重排序, 多路召回
     优点: 检索质量更高, 回答更准确
     缺点: 实现更复杂

  3) Modular RAG (模块化RAG)
     将RAG拆分为独立模块, 灵活组合
     模块: 索引, 检索, 重排序, 生成, 缓存等
     优点: 可定制, 可扩展
     缺点: 系统复杂度高

  4) Graph RAG (图RAG)
     利用知识图谱进行结构化检索
     优点: 能处理复杂的关系推理
     缺点: 知识图谱构建成本高

  5) Agentic RAG (智能体RAG)
     Agent自主决定是否检索, 检索什么, 如何使用
     优点: 灵活, 自适应
     缺点: 成本高, 延迟大
""")

print("\n" + "=" * 60)
print("W18-D1 完成! 本节介绍了RAG的概念、动机和基本架构")
print("=" * 60)
