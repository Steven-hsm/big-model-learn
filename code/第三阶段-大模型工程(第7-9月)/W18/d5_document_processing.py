"""
W18-D5 文档处理 (Document Processing)
======================================
文档加载(PDF/HTML/TXT), 文本切分策略(固定长度/递归/语义切分),
元数据管理, LangChain文档处理工具
"""

import sys
import re
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W18-D5 文档处理 (Document Processing)")
print("=" * 60)

# ============================================================
# 1. 文档加载
# ============================================================
print("\n--- 1. 文档加载 ---")
print("""
  RAG系统需要处理多种格式的文档:

  常用格式:
    - TXT:    纯文本, 直接读取
    - PDF:    使用 PyPDF2 / pdfplumber / PyMuPDF
    - HTML:   使用 BeautifulSoup
    - DOCX:   使用 python-docx
    - CSV:    使用 pandas
    - Markdown: 使用 mistune / markdown

  LangChain 提供统一接口:
    from langchain.document_loaders import (
        TextLoader, PyPDFLoader, UnstructuredHTMLLoader,
        Docx2txtLoader, CSVLoader
    )
""")


class SimpleDocument:
    """文档数据结构"""

    def __init__(self, content: str, metadata: Optional[Dict] = None):
        self.content = content
        self.metadata = metadata or {}

    def __repr__(self):
        preview = self.content[:50].replace('\n', ' ')
        return f"Document(content='{preview}...', metadata={self.metadata})"


class DocumentLoader:
    """文档加载器集合"""

    @staticmethod
    def load_txt(file_path: str, encoding: str = 'utf-8') -> SimpleDocument:
        """加载TXT文件"""
        with open(file_path, 'r', encoding=encoding) as f:
            content = f.read()
        return SimpleDocument(content, {
            'source': file_path,
            'type': 'txt',
            'size': len(content)
        })

    @staticmethod
    def load_text_string(text: str, source: str = "inline") -> SimpleDocument:
        """从字符串创建文档"""
        return SimpleDocument(text, {
            'source': source,
            'type': 'text',
            'size': len(text)
        })

    @staticmethod
    def load_mock_pdf(filename: str) -> SimpleDocument:
        """模拟PDF加载 (演示用)"""
        mock_content = f"""
        {filename} - 第1页
        深度学习是机器学习的一个重要分支。它使用多层神经网络来学习数据的层次化表示。
        深度学习在图像识别、自然语言处理和语音识别等领域取得了显著突破。

        {filename} - 第2页
        Transformer架构是现代深度学习的基石。它完全基于注意力机制，
        能够捕获序列中的长距离依赖关系。BERT和GPT都是基于Transformer的模型。

        {filename} - 第3页
        检索增强生成(RAG)是一种结合检索和生成的技术。它通过检索外部知识库中的相关文档，
        来增强大语言模型的回答质量和事实性。
        """
        return SimpleDocument(mock_content.strip(), {
            'source': filename,
            'type': 'pdf',
            'pages': 3,
            'size': len(mock_content)
        })


# 演示
loader = DocumentLoader()
doc = loader.load_mock_pdf("AI技术综述.pdf")
print(f"  加载文档: {doc}")
print(f"  文档长度: {len(doc.content)} 字符")

# ============================================================
# 2. 文本切分策略
# ============================================================
print("\n--- 2. 文本切分策略 ---")
print("""
  为什么需要切分?
    - LLM有上下文长度限制 (如4K, 8K, 128K tokens)
    - 太长的上下文会降低检索精度
    - 太短的切分会丢失上下文

  常见策略:
    1) 固定长度切分:     简单但可能切断语义
    2) 递归字符切分:     按分隔符递归切分, 尽量保持段落完整
    3) 语义切分:         基于语义相似度动态切分
    4) 句子级切分:       按句子边界切分
""")


# --- 策略1: 固定长度切分 ---
class FixedLengthChunker:
    """固定长度切分"""

    def __init__(self, chunk_size=200, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str) -> List[SimpleDocument]:
        chunks = []
        start = 0
        chunk_idx = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            chunks.append(SimpleDocument(chunk_text, {
                'chunk_index': chunk_idx,
                'start': start,
                'end': min(end, len(text)),
                'strategy': 'fixed_length',
                'chunk_size': self.chunk_size,
                'overlap': self.overlap,
            }))
            start = end - self.overlap
            chunk_idx += 1
        return chunks


# --- 策略2: 递归字符切分 ---
class RecursiveCharacterChunker:
    """递归字符切分 (模拟LangChain的RecursiveCharacterTextSplitter)"""

    def __init__(self, chunk_size=200, overlap=50, separators=None):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators or ['\n\n', '\n', '。', '！', '？', '.', ' ', '']

    def split(self, text: str) -> List[SimpleDocument]:
        chunks = self._recursive_split(text, self.separators)
        result = []
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) > 0:
                result.append(SimpleDocument(chunk.strip(), {
                    'chunk_index': i,
                    'strategy': 'recursive',
                    'char_count': len(chunk),
                }))
        return result

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        if not separators:
            return [text]

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == '':
            # 最后手段: 按字符切分
            return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size - self.overlap)]

        splits = text.split(separator)

        # 合并小片段
        merged = []
        current = ""
        for split in splits:
            candidate = current + separator + split if current else split
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    merged.append(current)
                # 如果单个片段就超长, 递归使用下一个分隔符
                if len(split) > self.chunk_size:
                    sub_splits = self._recursive_split(split, remaining_separators)
                    merged.extend(sub_splits)
                    current = ""
                else:
                    current = split
        if current:
            merged.append(current)

        return merged


# --- 策略3: 句子级切分 ---
class SentenceChunker:
    """按句子切分"""

    def __init__(self, chunk_size=300, overlap_sentences=1):
        self.chunk_size = chunk_size
        self.overlap_sentences = overlap_sentences

    def split(self, text: str) -> List[SimpleDocument]:
        # 中英文句子分割
        sentences = re.split(r'(?<=[。！？.!?])\s*', text)
        sentences = [s for s in sentences if s.strip()]

        chunks = []
        current_sentences = []
        current_len = 0

        for sent in sentences:
            if current_len + len(sent) > self.chunk_size and current_sentences:
                chunk_text = ''.join(current_sentences)
                chunks.append(SimpleDocument(chunk_text, {
                    'strategy': 'sentence',
                    'sentence_count': len(current_sentences),
                    'char_count': len(chunk_text),
                }))
                # 保留overlap个句子
                current_sentences = current_sentences[-self.overlap_sentences:]
                current_len = sum(len(s) for s in current_sentences)

            current_sentences.append(sent)
            current_len += len(sent)

        if current_sentences:
            chunk_text = ''.join(current_sentences)
            chunks.append(SimpleDocument(chunk_text, {
                'strategy': 'sentence',
                'sentence_count': len(current_sentences),
                'char_count': len(chunk_text),
            }))

        return chunks


# 测试所有切分策略
sample_text = """
深度学习是机器学习的一个重要分支。它使用多层神经网络来学习数据的层次化表示。

深度学习在图像识别领域取得了显著突破。卷积神经网络(CNN)是图像处理的核心架构。
ResNet通过残差连接解决了深层网络的梯度消失问题。

在自然语言处理领域，Transformer架构彻底改变了NLP的研究方向。BERT通过双向预训练，
在多项NLP任务上刷新了记录。GPT系列模型则展示了大规模语言模型的强大生成能力。

检索增强生成(RAG)是一种新兴的技术范式。它通过检索外部知识库中的相关文档，
来增强大语言模型的回答质量。RAG系统包括文档处理、向量检索和生成三个核心步骤。

向量数据库是RAG系统的重要基础设施。FAISS、Milvus、ChromaDB等工具
提供了高效的向量存储和检索能力。选择合适的向量数据库需要考虑数据规模、
检索延迟和部署成本等因素。
""".strip()

print(f"\n  原始文本长度: {len(sample_text)} 字符")

# 固定长度切分
fixed_chunker = FixedLengthChunker(chunk_size=150, overlap=30)
fixed_chunks = fixed_chunker.split(sample_text)
print(f"\n  固定长度切分 (size=150, overlap=30):")
print(f"  生成 {len(fixed_chunks)} 个chunks")
for chunk in fixed_chunks[:3]:
    print(f"    [{chunk.metadata['chunk_index']}] ({chunk.metadata['start']}-{chunk.metadata['end']}): {chunk.content[:60]}...")

# 递归切分
recursive_chunker = RecursiveCharacterChunker(chunk_size=150, overlap=30)
recursive_chunks = recursive_chunker.split(sample_text)
print(f"\n  递归字符切分 (size=150, overlap=30):")
print(f"  生成 {len(recursive_chunks)} 个chunks")
for chunk in recursive_chunks[:3]:
    print(f"    [{chunk.metadata['chunk_index']}] ({chunk.metadata['char_count']}字): {chunk.content[:60]}...")

# 句子切分
sentence_chunker = SentenceChunker(chunk_size=150)
sentence_chunks = sentence_chunker.split(sample_text)
print(f"\n  句子级切分 (size=150):")
print(f"  生成 {len(sentence_chunks)} 个chunks")
for chunk in sentence_chunks[:3]:
    print(f"    ({chunk.metadata['sentence_count']}句, {chunk.metadata['char_count']}字): {chunk.content[:60]}...")

# ============================================================
# 3. 元数据管理
# ============================================================
print("\n--- 3. 元数据管理 ---")
print("""
  元数据 (Metadata) 是RAG系统中重要的附加信息:

  常见元数据字段:
    - source:      文档来源 (文件名/URL)
    - chunk_index: 切分索引
    - page:        页码 (PDF)
    - section:     章节标题
    - author:      作者
    - date:        日期
    - char_count:  字符数
    - word_count:  词数

  元数据的用途:
    1) 过滤: 检索时按元数据过滤 (如只搜索特定章节)
    2) 排序: 结合元数据分数和相似度排序
    3) 展示: 在回答中引用来源
    4) 管理: 增量更新/删除文档
""")


class ChunkManager:
    """Chunk管理器"""

    def __init__(self):
        self.chunks = []

    def add_chunks(self, chunks: List[SimpleDocument], source: str = ""):
        """添加chunks"""
        for chunk in chunks:
            chunk.metadata['source'] = source
            self.chunks.append(chunk)

    def get_stats(self):
        """统计信息"""
        if not self.chunks:
            return {}
        lengths = [len(c.content) for c in self.chunks]
        return {
            'total_chunks': len(self.chunks),
            'avg_length': np.mean(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'total_chars': sum(lengths),
            'sources': list(set(c.metadata.get('source', '') for c in self.chunks)),
        }

    def filter_by_metadata(self, **kwargs) -> List[SimpleDocument]:
        """按元数据过滤"""
        result = []
        for chunk in self.chunks:
            match = all(chunk.metadata.get(k) == v for k, v in kwargs.items())
            if match:
                result.append(chunk)
        return result


# 使用管理器
manager = ChunkManager()
manager.add_chunks(fixed_chunks, source="AI技术综述")
manager.add_chunks(recursive_chunks, source="AI技术综述_递归")

stats = manager.get_stats()
print(f"\n  Chunk管理器统计:")
for k, v in stats.items():
    print(f"    {k}: {v}")

# ============================================================
# 4. 可视化
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 子图1: 不同切分策略的chunk大小分布
ax1 = axes[0]
fixed_lens = [len(c.content) for c in fixed_chunks]
recursive_lens = [len(c.content) for c in recursive_chunks]
sentence_lens = [len(c.content) for c in sentence_chunks]

data_to_plot = [fixed_lens, recursive_lens, sentence_lens]
labels = ['固定长度', '递归字符', '句子级']

bp = ax1.boxplot(data_to_plot, labels=labels, patch_artist=True)
colors = ['#e74c3c', '#3498db', '#2ecc71']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax1.set_ylabel('Chunk长度 (字符数)', fontsize=12)
ax1.set_title('不同切分策略的Chunk大小分布', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# 子图2: 切分策略对比
ax2 = axes[1]
strategies = ['固定长度', '递归字符', '句子级']
chunk_counts = [len(fixed_chunks), len(recursive_chunks), len(sentence_chunks)]
avg_lens = [np.mean(fixed_lens), np.mean(recursive_lens), np.mean(sentence_lens)]
quality_scores = [0.6, 0.85, 0.8]  # 语义完整性模拟

x = np.arange(len(strategies))
width = 0.25

ax2.bar(x - width, [c / max(chunk_counts) for c in chunk_counts], width,
        label='相对chunk数', color='#e74c3c', alpha=0.8, edgecolor='black')
ax2.bar(x, [l / max(avg_lens) for l in avg_lens], width,
        label='相对平均长度', color='#3498db', alpha=0.8, edgecolor='black')
ax2.bar(x + width, quality_scores, width,
        label='语义完整性', color='#2ecc71', alpha=0.8, edgecolor='black')

ax2.set_xticks(x)
ax2.set_xticklabels(strategies, fontsize=11)
ax2.set_ylabel('归一化分数', fontsize=12)
ax2.set_title('切分策略多维度对比', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W18/document_processing.png', dpi=150, bbox_inches='tight')
print("\n  图表已保存: document_processing.png")
plt.close()

# ============================================================
# 5. LangChain 文档处理工具 (可选)
# ============================================================
print("\n--- 5. LangChain 文档处理工具 (可选) ---")

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

    # 使用LangChain的切分器
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=30,
        separators=['\n\n', '\n', '。', '.', ' '],
    )

    lc_chunks = splitter.split_text(sample_text)
    print(f"  LangChain RecursiveCharacterTextSplitter:")
    print(f"  生成 {len(lc_chunks)} 个chunks")
    for i, chunk in enumerate(lc_chunks[:3]):
        print(f"    [{i}] ({len(chunk)}字): {chunk[:60]}...")

except ImportError:
    print("  [!] langchain 未安装, 跳过")
    print("  安装命令: pip install langchain")
except Exception as e:
    print(f"  [!] LangChain运行出错: {e}")

# ============================================================
# 6. 切分策略选择指南
# ============================================================
print("\n--- 6. 切分策略选择指南 ---")
print("""
  +------------------+------------------+------------------+------------------+
  |      策略        |    适用场景      |     优点         |     缺点         |
  +------------------+------------------+------------------+------------------+
  | 固定长度         | 简单场景        | 实现简单          | 可能切断语义      |
  | 递归字符         | 通用推荐        | 保持段落完整      | 需要调参          |
  | 句子级           | 短文档          | 语义完整          | chunk大小不均匀   |
  | 语义切分         | 高质量需求      | 最佳语义保持      | 计算成本高        |
  | 滑动窗口         | 需要重叠        | 上下文连贯        | 存储冗余          |
  +------------------+------------------+------------------+------------------+

  推荐参数:
    chunk_size:   256-1024 tokens (中文约 400-1500 字符)
    chunk_overlap: chunk_size的10-20%
    分隔符优先级: 段落 > 句子 > 词
""")

print("\n" + "=" * 60)
print("W18-D5 完成! 本节实现了文档加载和多种切分策略")
print("=" * 60)
