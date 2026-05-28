"""
W19-D1 高级切分策略 (Advanced Chunking)
=======================================
语义切分(基于嵌入相似度), 父子文档检索,
滑动窗口切分, 多粒度索引, 切分策略对比实验
"""

import sys
import re
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W19-D1 高级切分策略 (Advanced Chunking)")
print("=" * 60)

# ============================================================
# 1. 语义切分
# ============================================================
print("\n--- 1. 语义切分 (Semantic Chunking) ---")
print("""
  语义切分: 基于语义相似度动态决定切分点

  原理:
    1) 将文本按句子拆分
    2) 计算相邻句子的嵌入相似度
    3) 当相似度低于阈值时, 进行切分
    4) 相比固定长度切分, 能保持语义完整性
""")


def mock_embedding(text: str, dim: int = 32) -> np.ndarray:
    """模拟文本嵌入"""
    rng = np.random.RandomState(abs(hash(text)) % (2**31))
    base = rng.randn(dim) * 0.3
    # 语义增强
    keywords_vecs = {
        "机器学习": [1, 0.8, 0, 0], "深度学习": [0.9, 0.7, 0.3, 0],
        "神经网络": [0.8, 0.6, 0.4, 0], "自然语言": [0, 0, 1, 0.8],
        "Transformer": [0.5, 0.4, 0.7, 0.6], "RAG": [0.3, 0.4, 0.5, 0.7],
    }
    semantic = np.zeros(dim)
    for kw, vec in keywords_vecs.items():
        if kw in text:
            semantic[:4] += np.array(vec)
    combined = base + semantic
    return combined / (np.linalg.norm(combined) + 1e-10)


class SemanticChunker:
    """语义切分器"""

    def __init__(self, similarity_threshold=0.6, min_chunk_size=50):
        self.threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size

    def _split_sentences(self, text: str) -> List[str]:
        """按句子分割"""
        sentences = re.split(r'(?<=[。！？.!?\n])\s*', text)
        return [s.strip() for s in sentences if s.strip()]

    def split(self, text: str) -> List[Dict]:
        """语义切分"""
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return [{'text': text, 'sentences': sentences, 'strategy': 'semantic'}]

        # 计算相邻句子相似度
        embeddings = [mock_embedding(s) for s in sentences]
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        # 找切分点 (相似度低于阈值的位置)
        chunks = []
        current_sentences = [sentences[0]]

        for i in range(len(similarities)):
            if similarities[i] < self.threshold and len(''.join(current_sentences)) >= self.min_chunk_size:
                chunks.append({
                    'text': ''.join(current_sentences),
                    'sentences': current_sentences[:],
                    'strategy': 'semantic',
                    'break_similarity': similarities[i],
                })
                current_sentences = []
            current_sentences.append(sentences[i + 1])

        if current_sentences:
            chunks.append({
                'text': ''.join(current_sentences),
                'sentences': current_sentences[:],
                'strategy': 'semantic',
            })

        return chunks


# 测试语义切分
sample_text = """机器学习是人工智能的核心领域。它使用数据驱动的方法来训练模型。常见的算法包括决策树、支持向量机和随机森林。

深度学习是机器学习的子领域。它使用多层神经网络来学习数据表示。卷积神经网络在图像处理中表现出色。

自然语言处理研究计算机理解人类语言。Transformer架构是现代NLP的基石。BERT和GPT都基于Transformer。

向量数据库用于存储和检索高维向量。FAISS是最流行的向量检索库之一。它支持多种索引类型如IVF和HNSW。"""

chunker = SemanticChunker(similarity_threshold=0.5, min_chunk_size=30)
semantic_chunks = chunker.split(sample_text)

print(f"  语义切分结果 (阈值={chunker.threshold}):")
for i, chunk in enumerate(semantic_chunks):
    print(f"\n  Chunk {i} ({len(chunk['text'])}字, {len(chunk['sentences'])}句):")
    print(f"    {chunk['text'][:80]}...")

# ============================================================
# 2. 父子文档检索
# ============================================================
print("\n\n--- 2. 父子文档检索 (Parent-Child Document Retrieval) ---")
print("""
  父子文档检索策略:
    1) 将文档切分为大chunk(父)和小chunk(子)
    2) 索引子chunk (细粒度, 精确匹配)
    3) 检索时: 先检索子chunk, 然后返回对应的父chunk
    4) 好处: 精确检索的同时保持完整上下文

  示例:
    父chunk (500字): "机器学习概述...监督学习...无监督学习..."
      |-- 子chunk1 (100字): "机器学习概述..."
      |-- 子chunk2 (100字): "监督学习..."
      |-- 子chunk3 (100字): "无监督学习..."

    检索 "监督学习" -> 命中子chunk2 -> 返回父chunk (完整上下文)
""")


class ParentChildChunker:
    """父子文档切分器"""

    def __init__(self, parent_size=300, child_size=100, overlap=20):
        self.parent_size = parent_size
        self.child_size = child_size
        self.overlap = overlap

    def split(self, text: str) -> Dict:
        """生成父子文档对"""
        # 生成父chunk
        parents = []
        start = 0
        while start < len(text):
            end = start + self.parent_size
            parent_text = text[start:end]
            parents.append({
                'text': parent_text,
                'start': start,
                'end': min(end, len(text)),
            })
            start = end - self.overlap

        # 对每个父chunk生成子chunk
        parent_child_pairs = []
        for parent in parents:
            children = []
            child_start = 0
            while child_start < len(parent['text']):
                child_end = child_start + self.child_size
                child_text = parent['text'][child_start:child_end]
                if child_text.strip():
                    children.append({
                        'text': child_text,
                        'parent_text': parent['text'],
                        'parent_start': parent['start'],
                    })
                child_start = child_end - 10

            parent_child_pairs.append({
                'parent': parent,
                'children': children,
            })

        return parent_child_pairs


# 测试
pc_chunker = ParentChildChunker(parent_size=200, child_size=60)
pc_pairs = pc_chunker.split(sample_text)

print(f"  父子文档切分结果:")
print(f"  父chunk数: {len(pc_pairs)}")
for i, pair in enumerate(pc_pairs[:3]):
    print(f"\n  父chunk {i} ({len(pair['parent']['text'])}字):")
    print(f"    {pair['parent']['text'][:60]}...")
    print(f"    子chunk数: {len(pair['children'])}")
    for j, child in enumerate(pair['children'][:2]):
        print(f"      子{j}: {child['text'][:40]}...")

# ============================================================
# 3. 滑动窗口切分
# ============================================================
print("\n--- 3. 滑动窗口切分 ---")
print("""
  滑动窗口: 固定大小的窗口以固定步长滑动

  参数:
    window_size: 窗口大小
    step_size:   步长 (步长 < 窗口大小 = 有重叠)

  特点:
    - 保证上下文连续性
    - 重叠部分确保不丢失跨边界的信息
    - 适合时序数据和长文档
""")


class SlidingWindowChunker:
    """滑动窗口切分器"""

    def __init__(self, window_size=200, step_size=100):
        self.window_size = window_size
        self.step_size = step_size

    def split(self, text: str) -> List[Dict]:
        chunks = []
        start = 0
        idx = 0
        while start < len(text):
            end = min(start + self.window_size, len(text))
            chunk_text = text[start:end]
            chunks.append({
                'text': chunk_text,
                'index': idx,
                'start': start,
                'end': end,
                'strategy': 'sliding_window',
            })
            start += self.step_size
            idx += 1
            if end == len(text):
                break
        return chunks


sw_chunker = SlidingWindowChunker(window_size=150, step_size=80)
sw_chunks = sw_chunker.split(sample_text)
print(f"  滑动窗口切分 (window=150, step=80):")
print(f"  生成 {len(sw_chunks)} 个chunks")
for chunk in sw_chunks[:3]:
    print(f"    [{chunk['index']}] ({chunk['start']}-{chunk['end']}): {chunk['text'][:50]}...")

# ============================================================
# 4. 多粒度索引
# ============================================================
print("\n--- 4. 多粒度索引 ---")
print("""
  多粒度索引: 同时维护不同粒度的索引

  粒度级别:
    Level 1 (粗): 段落/章节级 (~500字) -> 概览检索
    Level 2 (中): 句子组级 (~200字)    -> 主题检索
    Level 3 (细): 句子级 (~50字)       -> 精确检索

  检索策略:
    1) 先在粗粒度索引中找到大致范围
    2) 再在细粒度索引中精确定位
    3) 返回细粒度结果 + 粗粒度上下文
""")


class MultiGranularityIndex:
    """多粒度索引"""

    def __init__(self):
        self.levels = {
            'coarse': [],   # 粗粒度
            'medium': [],   # 中粒度
            'fine': [],     # 细粒度
        }

    def build(self, text: str):
        """构建多粒度索引"""
        # 粗粒度: 按段落
        paragraphs = [p.strip() for p in re.split(r'\n\n+', text) if p.strip()]
        for i, para in enumerate(paragraphs):
            self.levels['coarse'].append({
                'text': para, 'level': 'coarse', 'index': i,
                'embedding': mock_embedding(para),
            })

        # 中粒度: 按句子组
        sentences = re.split(r'(?<=[。！？.!?])\s*', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        group_size = 3
        for i in range(0, len(sentences), group_size):
            group = ''.join(sentences[i:i + group_size])
            self.levels['medium'].append({
                'text': group, 'level': 'medium', 'index': i // group_size,
                'embedding': mock_embedding(group),
            })

        # 细粒度: 按句子
        for i, sent in enumerate(sentences):
            self.levels['fine'].append({
                'text': sent, 'level': 'fine', 'index': i,
                'embedding': mock_embedding(sent),
            })

        print(f"  多粒度索引构建完成:")
        for level, items in self.levels.items():
            print(f"    {level}: {len(items)} 个chunk")

    def search(self, query: str, top_k: int = 3, level: str = 'medium') -> List[Dict]:
        """在指定粒度检索"""
        q_emb = mock_embedding(query)
        items = self.levels.get(level, [])
        scores = []
        for item in items:
            sim = np.dot(q_emb, item['embedding'])
            scores.append((item, float(sim)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# 测试
mgi = MultiGranularityIndex()
mgi.build(sample_text)

print("\n  多粒度检索测试:")
query = "深度学习"
for level in ['coarse', 'medium', 'fine']:
    results = mgi.search(query, top_k=2, level=level)
    print(f"\n  [{level}] 检索 '{query}':")
    for item, score in results:
        print(f"    ({score:.3f}) {item['text'][:60]}...")

# ============================================================
# 5. 切分策略对比实验
# ============================================================
print("\n--- 5. 切分策略对比实验 ---")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 子图1: 不同策略的chunk数量和大小
ax1 = axes[0]
strategies_names = ['固定长度', '语义切分', '滑动窗口', '父子(子)', '多粒度(中)']
# 模拟数据
chunk_counts = [8, 5, 10, 18, 7]
avg_chunk_sizes = [150, 250, 150, 60, 180]
quality_scores = [0.70, 0.88, 0.75, 0.82, 0.85]

x = np.arange(len(strategies_names))
bars = ax1.bar(x, quality_scores, color=['#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c'],
               edgecolor='black', alpha=0.85)

# 添加chunk数量标注
for i, (bar, cnt, size) in enumerate(zip(bars, chunk_counts, avg_chunk_sizes)):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{quality_scores[i]:.2f}\n({cnt}块, 均{size}字)',
             ha='center', va='bottom', fontsize=9)

ax1.set_xticks(x)
ax1.set_xticklabels(strategies_names, fontsize=10)
ax1.set_ylabel('语义完整性分数', fontsize=12)
ax1.set_title('切分策略质量对比', fontsize=14, fontweight='bold')
ax1.set_ylim(0.5, 1.1)
ax1.grid(True, alpha=0.3, axis='y')

# 子图2: 语义切分相似度曲线
ax2 = axes[1]
sentences = re.split(r'(?<=[。！？.!?])\s*', sample_text)
sentences = [s.strip() for s in sentences if s.strip()]

embeddings = [mock_embedding(s) for s in sentences]
similarities = [np.dot(embeddings[i], embeddings[i + 1]) for i in range(len(embeddings) - 1)]

ax2.plot(range(len(similarities)), similarities, 'o-', linewidth=2, markersize=8, color='#3498db')
ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='切分阈值=0.5')

# 标注切分点
for i, sim in enumerate(similarities):
    if sim < 0.5:
        ax2.axvline(x=i + 0.5, color='green', linestyle=':', alpha=0.7)
        ax2.annotate('切分', (i + 0.5, sim), fontsize=8, color='green',
                     xytext=(5, -15), textcoords='offset points')

ax2.set_xlabel('句子对索引', fontsize=12)
ax2.set_ylabel('相邻句子相似度', fontsize=12)
ax2.set_title('语义切分: 句子间相似度曲线', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# 添加句子标注
for i, sent in enumerate(sentences):
    short = sent[:10] + '...' if len(sent) > 10 else sent
    ax2.annotate(short, (i, min(similarities) - 0.05), fontsize=7, rotation=45, ha='right')

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W19/advanced_chunking.png', dpi=150, bbox_inches='tight')
print("  图表已保存: advanced_chunking.png")
plt.close()

# ============================================================
# 6. 切分策略选择建议
# ============================================================
print("\n--- 6. 切分策略选择建议 ---")
print("""
  +------------------+------------------+------------------+
  |      策略        |    适用场景      |    推荐参数      |
  +------------------+------------------+------------------+
  | 固定长度         | 快速原型        | size=256, ov=50  |
  | 语义切分         | 高质量需求      | threshold=0.5    |
  | 滑动窗口         | 需要全覆盖      | window=200,step=100|
  | 父子文档         | 需要上下文      | parent=500,child=100|
  | 多粒度           | 灵活检索        | 3级: 段/组/句     |
  +------------------+------------------+------------------+

  实践建议:
    1) 先用固定长度快速验证
    2) 效果不够好时尝试语义切分
    3) 需要上下文时用父子文档
    4) 生产环境推荐多粒度 + 重排序
""")

print("\n" + "=" * 60)
print("W19-D1 完成! 本节实现了多种高级切分策略")
print("=" * 60)
