# W18 - RAG检索增强生成基础

> 第18周学习计划 | Java开发工程师转AI开发 | 工作日每晚2小时 + 周末6-8小时

---

## 一、本周目标

1. 理解RAG架构的核心思想和完整流程（离线索引 + 在线查询）
2. 掌握文本切分（Chunking）策略及参数调优方法
3. 理解Embedding模型原理，能对比不同模型的效果
4. 熟练使用向量数据库（FAISS/Chroma）进行存储和检索
5. 从零实现一个完整的RAG Pipeline，并掌握评估方法

---

## 二、时间安排

### 工作日（周一至周五，每晚2小时）

| 日期 | 主题 | 时间分配 |
|------|------|----------|
| Day 1 (周一) | RAG架构概览 | 理论学习90分钟 + 绘制架构图30分钟 |
| Day 2 (周二) | 文本切分Chunking | 理论40分钟 + 代码实验80分钟 |
| Day 3 (周三) | Embedding模型 | 理论60分钟 + 对比实验60分钟 |
| Day 4 (周四) | 向量数据库FAISS | 学习60分钟 + 代码练习60分钟 |
| Day 5 (周五) | 向量数据库进阶(Chroma) | 学习60分钟 + 代码练习60分钟 |

### 周末（周六至周日，每天6-8小时）

| 日期 | 主题 | 时间分配 |
|------|------|----------|
| Day 6 (周六) | 基础RAG实现 | 上午实现Pipeline(3h) + 下午调试优化(3h) + 文档(1h) |
| Day 7 (周日) | RAG评估 | 上午学习评估方法(3h) + 下午编写评估脚本(3h) + 总结(1h) |

---

## 三、详细学习内容

### Day 1: RAG架构概览

#### 3.1.1 为什么需要RAG

LLM的三大固有局限：

```
1. 知识截止：训练数据有截止日期，无法回答最新信息
   例：GPT-4知识截止到2024年，无法回答2025年的新闻

2. 幻觉问题：LLM可能生成看似合理但事实上错误的内容
   例：编造不存在的论文引用、虚构历史事件

3. 私有数据：无法访问企业内部文档、数据库等私有信息
   例：公司内部规章制度、客户资料、产品文档
```

#### 3.1.2 RAG vs Fine-tuning

```
| 维度       | RAG                          | Fine-tuning                  |
|-----------|------------------------------|------------------------------|
| 知识更新    | 实时更新文档即可              | 需要重新训练模型              |
| 成本       | 低（只需维护知识库）          | 高（需要GPU训练时间）         |
| 适用场景    | 知识注入、事实性问答          | 行为塑造、风格调整            |
| 可解释性    | 高（可追溯引用来源）          | 低（知识融入模型参数）        |
| 私有数据    | 天然支持                      | 数据混入训练集，有泄露风险    |
| 幻觉控制    | 较好（有上下文约束）          | 一般（仍可能产生幻觉）        |
```

**选择建议**：优先用RAG解决知识问题，用Fine-tuning解决行为/风格问题。两者可以结合使用。

#### 3.1.3 RAG基本流程

**离线索引阶段（Indexing）**：

```
文档(PDF/Word/HTML/...)
    ↓ 文档加载
原始文本
    ↓ 文本切分(Chunking)
文本片段(Chunks)
    ↓ Embedding向量化
向量(Embeddings)
    ↓ 存储
向量数据库(FAISS/Chroma/Milvus)
```

**在线查询阶段（Querying）**：

```
用户问题(Query)
    ↓ Embedding向量化
问题向量
    ↓ 向量检索(Similarity Search)
Top-K相关文档片段
    ↓ Prompt拼接
Context + Question → LLM
    ↓ 生成回答
最终回答(可附带引用来源)
```

#### 3.1.4 RAG演进路线

```
Naive RAG（基础RAG）
  → 简单的 索引→检索→生成 流程
  → 问题：检索质量差、生成不相关

Advanced RAG（高级RAG）
  → 增加预处理：Query改写、HyDE、混合检索
  → 增加后处理：Reranker重排序、上下文压缩
  → 提升检索和生成质量

Modular RAG（模块化RAG）
  → RAG流程模块化，每个模块可替换
  → 支持编排：迭代检索、自适应检索、多跳检索
  → 代表：LangChain/LlamaIndex的模块化设计
```

---

### Day 2: 文本切分Chunking

#### 3.2.1 固定长度切分

```python
def fixed_size_chunk(text, chunk_size=500, chunk_overlap=50):
    """固定长度切分，带重叠"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - chunk_overlap  # 重叠部分
    return chunks

text = "这是一段很长的文本..." * 100
chunks = fixed_size_chunk(text, chunk_size=500, chunk_overlap=50)
print(f"切分数量: {len(chunks)}")
print(f"第一段末尾: ...{chunks[0][-30:]}")
print(f"第二段开头: {chunks[1][:30]}...")  # 应与第一段末尾有重叠
```

**参数调优**：
- `chunk_size`太大：检索不精确，包含过多无关内容
- `chunk_size`太小：语义不完整，缺少上下文
- `chunk_overlap`：防止关键信息被切在边界处，通常为chunk_size的10%

#### 3.2.2 递归字符切分 RecursiveCharacterTextSplitter

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
)

text = """# 第一章 引言

自然语言处理是人工智能的重要分支。它研究计算机与人类语言之间的交互。

## 1.1 研究背景

近年来，大语言模型取得了突破性进展。从BERT到GPT-4，模型能力不断提升。

## 1.2 研究意义

NLP技术在各行各业都有广泛应用。包括搜索引擎、智能客服、机器翻译等。
"""

chunks = splitter.split_text(text)
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} (长度: {len(chunk)}) ---")
    print(chunk[:100] + "...")
```

**切分逻辑**：按分隔符优先级尝试切分。先用`\n\n`（段落级），如果段落在chunk_size内就保留；如果超出，再用`\n`（行级）切分；依此类推。

#### 3.2.3 语义切分

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def semantic_chunk(text, similarity_threshold=0.5):
    """基于语义相似度的切分"""
    # 先按句子切分
    sentences = text.replace('。', '。\n').replace('！', '！\n').replace('？', '？\n').split('\n')
    sentences = [s.strip() for s in sentences if s.strip()]

    # 计算相邻句子的语义相似度
    embeddings = model.encode(sentences)
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = np.dot(embeddings[i], embeddings[i+1]) / (
            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1])
        )
        similarities.append(sim)

    # 在相似度低的地方切分（语义转折点）
    chunks = []
    current_chunk = sentences[0]
    for i, sim in enumerate(similarities):
        if sim < similarity_threshold:
            chunks.append(current_chunk)
            current_chunk = sentences[i + 1]
        else:
            current_chunk += sentences[i + 1]
    chunks.append(current_chunk)

    return chunks
```

#### 3.2.4 Markdown结构切分

```python
from langchain.text_splitter import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_chunks = markdown_splitter.split_text(markdown_text)

# 每个chunk会保留其标题层级作为metadata
for chunk in md_chunks:
    print(f"Metadata: {chunk.metadata}")
    print(f"Content: {chunk.page_content[:100]}...")
```

#### 3.2.5 切分策略对比

```
| 策略         | 优点               | 缺点               | 适用场景           |
|-------------|-------------------|-------------------|-------------------|
| 固定长度      | 简单可控            | 可能切断语义        | 通用场景           |
| 递归字符      | 尊重自然边界        | 依赖分隔符设置      | 大多数场景（推荐）  |
| 语义切分      | 语义完整性好        | 计算成本高          | 高质量需求场景     |
| Markdown结构 | 保留文档结构        | 仅适用于Markdown   | 技术文档          |
```

---

### Day 3: Embedding模型

#### 3.3.1 词嵌入演进

```
Word2Vec (2013)
  → 静态词嵌入，每个词一个固定向量
  → 问题：无法处理多义词（"苹果"公司 vs "苹果"水果）

Sentence-BERT (2019)
  → 句子级别的嵌入
  → 通过对比学习训练，使相似句子的向量更接近

SimCSE (2021)
  → 简单对比学习框架
  → 无监督：Dropout作为数据增强，同一句子两次过模型产生正样本对
  → 有监督：用NLI数据集训练

OpenAI Embedding (2022-2024)
  → text-embedding-ada-002: 1536维
  → text-embedding-3-small: 1536维
  → text-embedding-3-large: 3072维

BGE (2023-2024)
  → bge-large-zh: 中文效果最好的开源模型之一
  → bge-m3: 多语言多功能

E5 (2023-2024)
  → multilingual-e5: 多语言支持
  → 训练时加入前缀指令："query:" / "passage:"
```

#### 3.3.2 句子嵌入原理

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('bge-large-zh-v1.5')

# 生成句子嵌入
sentences = [
    "大语言模型是AI的重要突破",
    "LLM是人工智能领域的关键进展",
    "今天天气真好",
]

embeddings = model.encode(sentences)
print(f"Embedding形状: {embeddings.shape}")  # (3, 1024)

# 计算余弦相似度
from sklearn.metrics.pairwise import cosine_similarity

sims = cosine_similarity(embeddings)
print("相似度矩阵:")
print(f"  句1 vs 句2: {sims[0][1]:.4f}")  # 高相似度
print(f"  句1 vs 句3: {sims[0][2]:.4f}")  # 低相似度
```

**原理说明**：
- CLS Token方法：取Transformer输出的第一个token（[CLS]）的向量作为整句表示
- Mean Pooling方法：对序列所有token的输出向量取平均
- BGE使用CLS token + 指令前缀

#### 3.3.3 余弦相似度

```
cos_sim(A, B) = (A · B) / (||A|| × ||B||)

取值范围: [-1, 1]
  1  → 完全相同方向（最相似）
  0  → 正交（无关联）
 -1  → 完全相反方向（最不相似）
```

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

vec1 = embeddings[0]
vec2 = embeddings[1]
print(f"余弦相似度: {cosine_similarity(vec1, vec2):.4f}")
```

#### 3.3.4 OpenAI Embedding API

```python
from openai import OpenAI
client = OpenAI()

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=["这是第一句话", "这是第二句话"],
    dimensions=1024,  # 可选：降低维度（Matryoshka嵌入）
)

for item in response.data:
    print(f"Embedding维度: {len(item.embedding)}")
```

#### 3.3.5 不同Embedding模型对比

```
| 模型                    | 维度  | 中文支持 | 大小    | 适用场景        |
|------------------------|-------|---------|---------|----------------|
| text-embedding-3-small | 1536  | 好      | API调用  | 通用、生产环境  |
| text-embedding-3-large | 3072  | 好      | API调用  | 高精度需求     |
| bge-large-zh-v1.5      | 1024  | 优秀    | 1.3GB   | 中文场景(推荐)  |
| bge-m3                 | 1024  | 优秀    | 2.2GB   | 多语言场景     |
| multilingual-e5-large  | 1024  | 好      | 2.2GB   | 多语言场景     |
| paraphrase-multilingual| 384   | 一般    | 470MB   | 快速原型       |
```

---

### Day 4: 向量数据库FAISS

#### 3.4.1 FAISS安装与基础

```bash
pip install faiss-cpu    # CPU版本
# pip install faiss-gpu  # GPU版本（需要CUDA）
```

#### 3.4.2 IndexFlatL2 - 暴力搜索

```python
import faiss
import numpy as np

# 生成模拟数据
np.random.seed(42)
dimension = 768  # Embedding维度
num_vectors = 10000

# 创建随机向量（模拟Embedding）
vectors = np.random.random((num_vectors, dimension)).astype('float32')

# 创建L2距离索引（暴力搜索，精确但慢）
index = faiss.IndexFlatL2(dimension)

# 添加向量
index.add(vectors)
print(f"索引中的向量数: {index.ntotal}")

# 搜索
query = np.random.random((1, dimension)).astype('float32')
k = 5  # 返回最相似的5个
distances, indices = index.search(query, k)

print(f"最近的{k}个向量索引: {indices}")
print(f"对应的L2距离: {distances}")
```

#### 3.4.3 IndexIVFFlat - 倒排索引

```python
import faiss
import numpy as np

dimension = 768
num_vectors = 100000
vectors = np.random.random((num_vectors, dimension)).astype('float32')

# 创建量化器（用于聚类）
nlist = 100  # 聚类中心数量
quantizer = faiss.IndexFlatL2(dimension)

# 创建IVF索引
index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

# 训练（IVF需要先训练建立聚类中心）
index.train(vectors[:10000])  # 用部分数据训练
index.add(vectors)

# 搜索
index.nprobe = 10  # 搜索时探测的聚类数，越大越精确但越慢
query = np.random.random((1, dimension)).astype('float32')
distances, indices = index.search(query, k=5)
```

#### 3.4.4 IndexHNSW - 层次导航小世界图

```python
import faiss
import numpy as np

dimension = 768
num_vectors = 100000
vectors = np.random.random((num_vectors, dimension)).astype('float32')

# 创建HNSW索引
M = 32  # 每个节点的邻居数，影响精度和内存
index = faiss.IndexHNSWFlat(dimension, M)

# HNSW不需要训练，直接添加
index.add(vectors)

# 搜索
query = np.random.random((1, dimension)).astype('float32')
distances, indices = index.search(query, k=5)
```

#### 3.4.5 余弦相似度 vs L2距离

```python
# FAISS的IndexFlatL2使用L2距离
# 要使用余弦相似度，需要先对向量做L2归一化

def normalize_vectors(vectors):
    """L2归一化，使L2距离等价于余弦相似度"""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

# 归一化后，L2距离越小 = 余弦相似度越高
normalized_vectors = normalize_vectors(vectors)
index = faiss.IndexFlatL2(dimension)
index.add(normalized_vectors)

# 查询时也需要归一化
query = np.random.random((1, dimension)).astype('float32')
query = normalize_vectors(query)
distances, indices = index.search(query, k=5)
```

#### 3.4.6 FAISS索引保存与加载

```python
# 保存索引
faiss.write_index(index, "my_index.faiss")

# 加载索引
loaded_index = faiss.read_index("my_index.faiss")

# 搜索
distances, indices = loaded_index.search(query, k=5)
```

#### 3.4.7 索引类型对比

```
| 索引类型         | 精确度 | 速度  | 内存  | 适用场景              |
|-----------------|--------|-------|-------|----------------------|
| IndexFlatL2     | 精确   | 慢    | 高    | 小数据集(<100K)       |
| IndexIVFFlat    | 近似   | 快    | 中    | 中等数据集(100K-10M)  |
| IndexHNSW       | 近似   | 很快  | 高    | 低延迟需求            |
| IndexIVFPQ      | 近似   | 很快  | 低    | 大数据集，内存受限     |
```

---

### Day 5: 向量数据库进阶

#### 3.5.1 Chroma入门

```bash
pip install chromadb
```

```python
import chromadb

# 创建客户端（默认使用内存存储）
client = chromadb.Client()
# 持久化存储：client = chromadb.PersistentClient(path="./chroma_db")

# 创建集合（类似数据库的表）
collection = client.create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
)

# 添加文档
collection.add(
    documents=["这是第一段文档内容", "这是第二段文档内容", "这是第三段文档内容"],
    metadatas=[{"source": "doc1"}, {"source": "doc2"}, {"source": "doc3"}],
    ids=["id1", "id2", "id3"]
)

# 查询
results = collection.query(
    query_texts=["搜索相关内容"],
    n_results=2,
    where={"source": "doc1"},  # 可选：元数据过滤
)

print(results)
# {'documents': [['这是第一段文档内容']], 'distances': [[0.23]], 'ids': [['id1']]}
```

#### 3.5.2 Chroma CRUD操作

```python
# 添加（带自定义Embedding）
collection.add(
    documents=["新文档"],
    embeddings=[[0.1, 0.2, 0.3, ...]],  # 自定义Embedding向量
    metadatas=[{"source": "new"}],
    ids=["id4"]
)

# 更新
collection.update(
    ids=["id1"],
    documents=["更新后的文档内容"],
    metadatas=[{"source": "updated"}]
)

# 删除
collection.delete(ids=["id3"])

# 获取
result = collection.get(ids=["id1"])
print(result)

# 查询（带过滤）
results = collection.query(
    query_texts=["搜索内容"],
    n_results=5,
    where={"source": "doc1"},       # 元数据过滤
    where_document={"$contains": "关键词"}  # 内容过滤
)
```

#### 3.5.3 Milvus概念

```
Milvus - 分布式向量数据库

特点：
- 分布式架构，支持水平扩展
- 支持十亿级向量检索
- 丰富的索引类型
- 云原生，支持Kubernetes部署

适用场景：
- 生产环境大规模部署
- 需要高可用和水平扩展
- 数据量超过单机承载能力

核心概念：
- Collection（集合）→ 类似数据库的表
- Field（字段）→ 包括向量字段和标量字段
- Index（索引）→ 向量索引类型
- Segment（段）→ 数据存储单元

选型建议：
- 开发/原型：Chroma或FAISS
- 生产/小规模：Chroma + 持久化
- 生产/大规模：Milvus或Pinecone
```

---

### Day 6: 基础RAG实现

#### 3.6.1 完整Pipeline

```python
"""
基础RAG Pipeline - 从零实现
依赖：pip install faiss-cpu sentence-transformers openai PyPDF2
"""
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ========== 1. 文档加载 ==========
def load_pdf(file_path):
    """加载PDF文档"""
    from PyPDF2 import PdfReader
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def load_text(file_path):
    """加载纯文本文档"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# ========== 2. 文本切分 ==========
def split_text(text, chunk_size=500, chunk_overlap=50):
    """递归字符切分"""
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)

# ========== 3. Embedding ==========
class Embedder:
    def __init__(self, model_name="BAAI/bge-large-zh-v1.5"):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts):
        """将文本列表转为向量"""
        # BGE模型推荐在查询前加 "为这个句子生成表示以用于检索相关文章："
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.astype('float32')

    def embed_query(self, query):
        """将查询文本转为向量"""
        # BGE模型查询需要加前缀
        prefixed_query = f"为这个句子生成表示以用于检索相关文章：{query}"
        embedding = self.model.encode([prefixed_query], normalize_embeddings=True)
        return embedding.astype('float32')

# ========== 4. 向量存储 ==========
class VectorStore:
    def __init__(self, dimension=1024):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # 内积索引（向量已归一化=余弦相似度）
        self.texts = []
        self.metadatas = []

    def add(self, texts, embeddings, metadatas=None):
        """添加文档"""
        self.index.add(embeddings)
        self.texts.extend(texts)
        if metadatas:
            self.metadatas.extend(metadatas)

    def search(self, query_embedding, top_k=5):
        """检索最相似的文档"""
        scores, indices = self.index.search(query_embedding, top_k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.texts):
                results.append({
                    "text": self.texts[idx],
                    "score": float(scores[0][i]),
                    "metadata": self.metadatas[idx] if self.metadatas else None,
                })
        return results

    def save(self, path):
        """保存索引"""
        faiss.write_index(self.index, path)
        import json
        with open(path + ".meta", 'w', encoding='utf-8') as f:
            json.dump({"texts": self.texts, "metadatas": self.metadatas}, f, ensure_ascii=False)

    def load(self, path):
        """加载索引"""
        self.index = faiss.read_index(path)
        import json
        with open(path + ".meta", 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.texts = data["texts"]
            self.metadatas = data["metadatas"]

# ========== 5. RAG Pipeline ==========
class SimpleRAG:
    def __init__(self, embedder_model="BAAI/bge-large-zh-v1.5", llm_model="gpt-4o"):
        self.embedder = Embedder(embedder_model)
        self.vector_store = VectorStore(dimension=1024)  # bge-large-zh维度
        self.llm_client = OpenAI()
        self.llm_model = llm_model

    def index_documents(self, file_paths):
        """索引文档"""
        all_chunks = []
        all_metadatas = []
        for file_path in file_paths:
            # 加载文档
            if file_path.endswith('.pdf'):
                text = load_pdf(file_path)
            else:
                text = load_text(file_path)

            # 切分
            chunks = split_text(text)
            all_chunks.extend(chunks)
            all_metadatas.extend([{"source": file_path}] * len(chunks))

        # Embedding
        embeddings = self.embedder.embed_texts(all_chunks)

        # 存储到向量数据库
        self.vector_store.add(all_chunks, embeddings, all_metadatas)
        print(f"索引完成: {len(all_chunks)}个文档片段")

    def query(self, question, top_k=5):
        """查询"""
        # 1. 查询向量化
        query_embedding = self.embedder.embed_query(question)

        # 2. 检索相关文档
        results = self.vector_store.search(query_embedding, top_k)

        # 3. 构建Prompt
        context = "\n\n".join([f"[来源{i+1}]: {r['text']}" for i, r in enumerate(results)])
        prompt = f"""基于以下上下文回答问题。如果上下文中没有相关信息，请说"根据现有资料无法回答"。

上下文：
{context}

问题：{question}

请基于上下文内容回答，并标注引用来源。"""

        # 4. LLM生成回答
        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": "你是一个准确的知识问答助手，只基于给定的上下文回答问题。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
        )

        answer = response.choices[0].message.content

        return {
            "answer": answer,
            "sources": results,
            "num_sources": len(results),
        }


# ========== 使用示例 ==========
if __name__ == "__main__":
    rag = SimpleRAG()

    # 索引文档
    rag.index_documents(["./docs/document1.txt", "./docs/document2.pdf"])

    # 查询
    result = rag.query("什么是机器学习？")
    print(f"回答: {result['answer']}")
    print(f"\n引用了 {result['num_sources']} 个文档片段")
    for i, source in enumerate(result['sources'][:3]):
        print(f"  来源{i+1} (相似度: {source['score']:.4f}): {source['text'][:50]}...")
```

---

### Day 7: RAG评估

#### 3.7.1 检索评估指标

**Recall@K**：前K个检索结果中包含正确答案的比例

```python
def recall_at_k(retrieved_docs, relevant_docs, k):
    """计算Recall@K"""
    retrieved_set = set(retrieved_docs[:k])
    relevant_set = set(relevant_docs)
    if len(relevant_set) == 0:
        return 0.0
    return len(retrieved_set & relevant_set) / len(relevant_set)

# 示例
retrieved = ["doc1", "doc3", "doc5", "doc7", "doc9"]
relevant = ["doc1", "doc5", "doc10"]
print(f"Recall@3: {recall_at_k(retrieved, relevant, 3):.2f}")  # 0.67
print(f"Recall@5: {recall_at_k(retrieved, relevant, 5):.2f}")  # 0.67
```

**MRR (Mean Reciprocal Rank)**：正确答案排名倒数的均值

```python
def mrr(queries_results):
    """计算MRR"""
    total_rr = 0
    for query, (retrieved, relevant) in queries_results.items():
        for i, doc in enumerate(retrieved):
            if doc in relevant:
                total_rr += 1 / (i + 1)
                break
    return total_rr / len(queries_results)
```

#### 3.7.2 生成质量评估

```python
def evaluate_faithfulness(answer, context, llm_client):
    """评估答案是否忠于上下文（Faithfulness）"""
    prompt = f"""评估以下回答是否完全基于给定的上下文。

上下文：
{context}

回答：
{answer}

请评估：
1. 回答中的每个声明是否都能在上下文中找到依据？
2. 是否有编造的信息？
3. Faithfulness评分(0-1)："""

    response = llm_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content


def evaluate_relevancy(answer, question, llm_client):
    """评估答案是否切题（Relevancy）"""
    prompt = f"""评估以下回答是否切题。

问题：{question}
回答：{answer}

请评估：
1. 回答是否直接回应了问题？
2. 是否有无关内容？
3. Relevancy评分(0-1)："""

    response = llm_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content
```

#### 3.7.3 RAGAS框架

```bash
pip install ragas
```

```python
from ragas import evaluate
from ragas.metrics import (
    context_precision,   # 检索精度
    context_recall,      # 检索召回
    answer_relevancy,    # 回答相关性
    faithfulness,        # 忠实度
)
from datasets import Dataset

# 准备评估数据
data = {
    "question": ["什么是机器学习？", "深度学习和机器学习的区别？"],
    "answer": ["机器学习是AI的子集...", "深度学习是机器学习的子集..."],
    "contexts": [["机器学习是人工智能的一个分支..."], ["深度学习使用神经网络..."]],
    "ground_truth": ["机器学习是让计算机从数据中学习的AI分支", "深度学习使用多层神经网络"],
}

dataset = Dataset.from_dict(data)

# 评估
results = evaluate(
    dataset,
    metrics=[context_precision, context_recall, answer_relevancy, faithfulness]
)
print(results)
```

#### 3.7.4 完整评估脚本

```python
class RAGEvaluator:
    """RAG系统评估器"""

    def __init__(self, rag_system):
        self.rag = rag_system
        self.llm_client = OpenAI()

    def evaluate(self, test_questions, ground_truths):
        """
        test_questions: [{"question": "...", "relevant_docs": ["doc1", "doc2"]}]
        ground_truths: ["正确答案1", "正确答案2"]
        """
        results = {
            "recall_at_k": [],
            "mrr": [],
            "faithfulness": [],
            "relevancy": [],
        }

        for q_data, truth in zip(test_questions, ground_truths):
            question = q_data["question"]
            relevant_docs = q_data.get("relevant_docs", [])

            # 查询
            rag_result = self.rag.query(question)
            retrieved_docs = [s["metadata"]["source"] for s in rag_result["sources"]]

            # 检索评估
            results["recall_at_k"].append(
                recall_at_k(retrieved_docs, relevant_docs, k=5)
            )

            # 生成质量评估
            context = "\n".join([s["text"] for s in rag_result["sources"]])
            results["faithfulness"].append(
                evaluate_faithfulness(rag_result["answer"], context, self.llm_client)
            )
            results["relevancy"].append(
                evaluate_relevancy(rag_result["answer"], question, self.llm_client)
            )

        # 汇总
        print("=" * 50)
        print("RAG系统评估报告")
        print("=" * 50)
        print(f"平均Recall@5: {np.mean(results['recall_at_k']):.4f}")
        print(f"Faithfulness评估: 见详细报告")
        print(f"Relevancy评估: 见详细报告")

        return results
```

---

## 四、代码练习

### Day 2 练习：对比3种切分策略

```
任务：
1. 准备一篇3000字以上的中文技术文章
2. 分别用固定长度、递归字��、语义切分3种方式处理
3. 对比：切分数量、平均chunk长度、语义完整性（人工评估）
4. 输出：chunking_comparison.py + 对比结果表
```

### Day 3 练习：对比3种Embedding模型

```
任务：
1. 准备10对中文句子（5对语义相似+5对语义不同）
2. 用3种模型计算相似度：bge-large-zh、multilingual-e5、OpenAI embedding
3. 对比哪个模型最能区分相似和不相似的句子
4. 输出：embedding_comparison.py + 对比结果
```

### Day 6 练习：从零实现RAG Pipeline

```
任务：
1. 准备3-5个文档（PDF/TXT/Markdown各至少一个）
2. 实现Day 6的完整SimpleRAG类
3. 准备10个测试问题，测试检索和生成效果
4. 输出：simple_rag.py
```

### Day 7 练习：编写RAG评估脚本

```
任务：
1. 为Day 6的RAG系统准备评估数据集（20个问题+标准答案）
2. 实现RAGEvaluator类
3. 跑完整评估，输出评估报告
4. 输出：rag_evaluator.py + 评估报告
```

---

## 五、本周产出

| 产出物 | 说明 | 完成标准 |
|--------|------|----------|
| chunking_comparison.py | 切分策略对比脚本 | 3种策略 + 对比结果 |
| embedding_comparison.py | Embedding模型对比脚本 | 3种模型 + 相似度对比 |
| simple_rag.py | 基础RAG Pipeline | 完整索引+检索+生成流程 |
| rag_evaluator.py | RAG评估脚本 | Recall/Faithfulness/Relevancy指标 |
| 学习笔记.md | 本周学习总结 | 包含核心概念和个人理解 |

---

## 六、自测题

### 题目

1. **RAG和Fine-tuning分别适合什么场景？**

<details>
<summary>参考答案</summary>

RAG适合：知识注入场景（如接入企业内部文档、实时信息查询）、需要可追溯引用来源的场景、知识频繁更新的场景。Fine-tuning适合：行为和风格塑造（如让模型模仿特定的写作风格、说话语气）、特定任务的能力增强（如代码生成、医学诊断）、输出格式控制。最佳实践是两者结合：先RAG注入知识，再Fine-tuning调整行为。
</details>

2. **chunk_size太大或太小分别有什么问题？**

<details>
<summary>参考答案</summary>

太大：(1) 检索不精确，一个chunk中可能只有一小部分与查询相关，但整个chunk都被检索回来；(2) 浪费LLM的上下文窗口，留给其他信息的空间变少；(3) 生成时可能被无关信息干扰。太小：(1) 语义不完整，一个完整的概念被拆成多个片段；(2) 缺乏上下文，模型难以理解片段的完整含义；(3) 需要检索更多片段才能覆盖完整信息。一般推荐chunk_size=300-500字，overlap=10%。
</details>

3. **向量检索的原理？FAISS的IVF和HNSW有什么区别？**

<details>
<summary>参考答案</summary>

向量检索的原理：将文本转为高维向量，通过计算向量间的距离（L2或余弦相似度）找到最相似的文档。IVF（倒排索引）：先对所有向量做K-Means聚类建立倒排索引，查询时只在最近的几个聚类中搜索，速度快但精度有损失。HNSW（层次导航小世界图）：构建一个多层图结构，类似跳表，从上层开始粗定位，逐层向下精确定位，查询速度极快且精度高，但内存占用较大。
</details>

4. **为什么检索到的文档要拼接到Prompt中？**

<details>
<summary>参考答案</summary>

因为LLM本身没有这些知识（知识截止/私有数据）。通过将检索到的文档作为上下文拼接到Prompt中，相当于让LLM"阅读"了相关资料后再回答问题。这样LLM的生成就有了事实依据，减少了幻觉。这就像考试时的"开卷考试"——题目不变，但可以参考资料来回答。
</details>

5. **RAG系统评估需要关注哪些维度？**

<details>
<summary>参考答案</summary>

两大维度：(1) 检索质量：Recall@K（前K个结果是否包含正确答案）、MRR（正确答案的排名）、Precision@K（前K个结果中有多少是相关的）。(2) 生成质量：Faithfulness/忠实度（回答是否忠于检索到的上下文）、Relevancy/相关性（回答是否切题）、完整性（是否完整回答了问题）。综合评估框架如RAGAS同时评估这两个维度。
</details>

---

## 七、Java开发者提示

### 向量数据库类比

```
向量数据库 ≈ Java中的搜索引擎（Elasticsearch）

Elasticsearch：
  索引文档 → 文本分词 → 倒排索引 → 关键词搜索
  doc.add(new TextField("content", text, Store.YES))

向量数据库：
  索引文档 → Embedding向量化 → 向量索引 → 语义搜索
  collection.add(documents=["文本"], embeddings=[[0.1, 0.2, ...]])

区别：ES基于关键词匹配，向量数据库基于语义相似度
例如搜索"计算机"，ES能找到含"计算机"的文档，
向量数据库还能找到含"电脑"、"PC"的语义相关文档
```

### Embedding类比

```
Embedding ≈ 将对象序列化为可比较的特征向量

Java中对对象排序：实现Comparable接口，用compareTo比较
AI中对文本比较：用Embedding转为向量，用余弦相似度比较

// Java
String s1 = "大语言模型"; String s2 = "LLM";
// 无法直接比较语义相似度

// Python + Embedding
vec1 = embed("大语言模型")  # [0.12, 0.34, ...]
vec2 = embed("LLM")         # [0.11, 0.35, ...]
cosine_sim(vec1, vec2)      # 0.92 → 语义高度相似
```

### RAG流程类比

```
RAG ≈ Java中的"查数据库 + 展示结果"

传统Java Web应用：
  用户搜索 → 查MySQL → 获取数据 → 渲染到页面

RAG应用：
  用户提问 → 向量检索 → 获取相关文档 → LLM生成回答

核心区别：最后一步从"模板渲染"变成了"AI生成"
```

### Chunking类比

```
Chunking ≈ Java中的分页查询

// Java分页：数据量大时，分页加载处理
Pageable pageable = PageRequest.of(0, 500);  // 每页500条
Page<Document> page = repo.findAll(pageable);

// RAG Chunking：文档太长时，分块处理
splitter = RecursiveCharacterTextSplitter(chunk_size=500)  # 每块500字

区别：分页是数据库层面的，Chunking是文档层面的语义切分
```
