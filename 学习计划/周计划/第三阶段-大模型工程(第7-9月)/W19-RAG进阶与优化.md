# W19 - RAG进阶与优化

> 第19周学习计划 | Java开发工程师转AI开发 | 工作日每晚2小时 + 周末6-8小时

---

## 一、本周目标

1. 掌握高级检索策略（混合检索、HyDE、Query改写）及其实际实现
2. 理解重排序（Reranker）的原理和两阶段检索流程
3. 掌握多格式文档处理技术（PDF/Word/Excel/OCR）
4. 理解并能实现至少一种高级RAG模式（Self-RAG/CRAG/Adaptive RAG）
5. 完成企业知识库RAG实战项目

---

## 二、时间安排

### 工作日（周一至周五，每晚2小时）

| 日期 | 主题 | 时间分配 |
|------|------|----------|
| Day 1 (周一) | 高级检索策略 | 理论60分钟 + 代码实验60分钟 |
| Day 2 (周二) | 重排序Reranker | 理论60分钟 + 代码实验60分钟 |
| Day 3 (周三) | 多格式文档处理 | 学习60分钟 + 代码练习60分钟 |
| Day 4 (周四) | 高级RAG模式 | 理论60分钟 + 实现一种模式60分钟 |
| Day 5 (周五) | RAG工程化 | 学习60分钟 + 代码实践60分钟 |

### 周末（周六至周日，每天6-8小时）

| 日期 | 主题 | 时间分配 |
|------|------|----------|
| Day 6 (周六) | LangChain/LlamaIndex RAG | 上午学习框架(3h) + 下午对比实现(3h) + 练习(1h) |
| Day 7 (周日) | 实战项目8 - 企业知识库RAG | 全天实现项目(7-8h) |

---

## 三、详细学习内容

### Day 1: 高级检索策略

#### 3.1.1 混合检索（向量 + BM25）

核心思想：向量检索捕获语义相似性，BM25关键词检索捕获精确匹配，两者互补。

```
混合检索分数 = α × 向量检索分数 + (1 - α) × BM25分数

α 通常取 0.5~0.7
```

**BM25算法原理**：

```
BM25(D, Q) = Σ IDF(qi) × (f(qi, D) × (k1 + 1)) / (f(qi, D) + k1 × (1 - b + b × |D|/avgdl))

其中：
- f(qi, D): 词qi在文档D中的词频
- |D|: 文档D的长度
- avgdl: 平均文档长度
- k1: 词频饱和参数（通常1.2-2.0）
- b: 长度归一化参数（通常0.75）
- IDF(qi): 逆文档频率 = log((N - n(qi) + 0.5) / (n(qi) + 0.5) + 1)
```

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class HybridRetriever:
    """混合检索器：向量检索 + BM25"""

    def __init__(self, embedder, vector_store, documents, alpha=0.7):
        self.embedder = embedder
        self.vector_store = vector_store
        self.documents = documents
        self.alpha = alpha

        # 构建BM25索引（使用TF-IDF近似）
        self.tfidf = TfidfVectorizer(max_features=5000)
        self.doc_texts = [doc["text"] for doc in documents]
        self.tfidf_matrix = self.tfidf.fit_transform(self.doc_texts)

    def search(self, query, top_k=10):
        # 1. 向量检索
        query_embedding = self.embedder.embed_query(query)
        vector_results = self.vector_store.search(query_embedding, top_k=top_k * 3)
        vector_scores = {r["id"]: r["score"] for r in vector_results}

        # 2. BM25检索
        query_tfidf = self.tfidf.transform([query])
        bm25_scores_arr = (self.tfidf_matrix @ query_tfidf.T).toarray().flatten()
        bm25_results = {}
        for i, score in enumerate(bm25_scores_arr):
            if score > 0:
                bm25_results[i] = score

        # 3. 分数归一化
        vector_norm = self._normalize_scores(vector_scores)
        bm25_norm = self._normalize_scores(bm25_results)

        # 4. 加权融合
        all_ids = set(vector_norm.keys()) | set(bm25_norm.keys())
        combined_scores = {}
        for doc_id in all_ids:
            v_score = vector_norm.get(doc_id, 0)
            b_score = bm25_norm.get(doc_id, 0)
            combined_scores[doc_id] = self.alpha * v_score + (1 - self.alpha) * b_score

        # 5. 排序返回top_k
        sorted_ids = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for doc_id, score in sorted_ids[:top_k]:
            results.append({
                "id": doc_id,
                "text": self.documents[doc_id]["text"],
                "score": score,
                "vector_score": vector_norm.get(doc_id, 0),
                "bm25_score": bm25_norm.get(doc_id, 0),
            })
        return results

    @staticmethod
    def _normalize_scores(scores):
        """Min-Max归一化到[0,1]"""
        if not scores:
            return {}
        values = list(scores.values())
        min_val, max_val = min(values), max(values)
        if max_val == min_val:
            return {k: 1.0 for k in scores}
        return {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}
```

#### 3.1.2 HyDE (Hypothetical Document Embedding)

```python
class HyDERetriever:
    """HyDE：先生成假设性答案，用答案的Embedding去检索"""

    def __init__(self, llm_client, embedder, vector_store):
        self.llm_client = llm_client
        self.embedder = embedder
        self.vector_store = vector_store

    def search(self, query, top_k=5):
        # Step 1: 让LLM生成假设性答案
        hypo_prompt = f"""请根据问题写一段可能包含答案的文档片段。
不要担心答案是否正确，只需写一段相关的文本。

问题：{query}

假设性文档："""

        response = self.llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": hypo_prompt}],
            temperature=0.3,
        )
        hypothetical_doc = response.choices[0].message.content

        # Step 2: 用假设性答案的Embedding去检索
        hypo_embedding = self.embedder.embed_texts([hypothetical_doc])
        results = self.vector_store.search(hypo_embedding, top_k=top_k)

        return {
            "hypothetical_document": hypothetical_doc,
            "results": results,
        }
```

**HyDE为什么有效**：用户的查询通常很短且不精确，而LLM生成的假设性答案在表达方式上更像真实文档，所以用假设性答案去检索能找到更相关的文档。

#### 3.1.3 Step-back Prompting

```python
def step_back_query(original_query, llm_client):
    """先问一个更抽象的问题，用抽象问题的答案帮助回答原始问题"""

    step_back_prompt = f"""你是一个知识渊博的助手。
给定一个具体问题，请生成一个更抽象、更广泛的问题，
该问题的答案可以帮助回答原始问题。

原始问题：{original_query}

更抽象的问题："""

    response = llm_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": step_back_prompt}],
    )
    return response.choices[0].message.content

# 示例
# 原始问题："Python 3.12有什么新特性？"
# 抽象问题："Python语言的版本演进历史和主要更新方向是什么？"
```

#### 3.1.4 多路召回

```python
class MultiRouteRetriever:
    """多路召回：多策略并行召回 → 合并去重"""

    def __init__(self, retrievers):
        self.retrievers = retrievers  # 多个检索器

    def search(self, query, top_k=10):
        all_results = []

        # 并行调用多个检索器
        for retriever in self.retrievers:
            results = retriever.search(query, top_k=top_k)
            all_results.extend(results)

        # 去重（按文档ID）
        seen = set()
        unique_results = []
        for r in all_results:
            if r["id"] not in seen:
                seen.add(r["id"])
                unique_results.append(r)

        # 按分数排序
        unique_results.sort(key=lambda x: x["score"], reverse=True)
        return unique_results[:top_k]
```

---

### Day 2: 重排序Reranker

#### 3.2.1 Cross-Encoder原理

```
Bi-Encoder（用于检索）：
  Query → Encoder → 向量Q
  Doc   → Encoder → 向量D
  相似度 = cosine(Q, D)
  优点：快（可预计算文档向量）
  缺点：Query和Doc各自编码，交互不充分

Cross-Encoder（用于重排序）：
  [Query, Doc] → Encoder → 相关性分数（单一标量）
  优点：准（Query和Doc联合编码，充分交互）
  缺点：慢（每个pair都要过一次模型，不能预计算）
```

```python
from sentence_transformers import CrossEncoder

# 加载Cross-Encoder模型
cross_encoder = CrossEncoder('BAAI/bge-reranker-large')

def rerank(query, documents, top_n=5):
    """使用Cross-Encoder重排序"""
    # 构造query-doc pair
    pairs = [[query, doc["text"]] for doc in documents]

    # 计算相关性分数
    scores = cross_encoder.predict(pairs)

    # 按分数排序
    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    return [
        {**doc, "rerank_score": float(score)}
        for doc, score in scored_docs[:top_n]
    ]
```

#### 3.2.2 两阶段检索流程

```
用户Query
    ↓
第一阶段：粗检索（Bi-Encoder向量检索，快）
    → 从10万文档中检索Top-50
    ↓
第二阶段：精排序（Cross-Encoder Reranker，准）
    → 对Top-50精排，返回Top-5
    ↓
送入LLM生成回答
```

```python
class TwoStageRetriever:
    """两阶段检索：粗检索 + 精排序"""

    def __init__(self, vector_store, embedder, reranker_model="BAAI/bge-reranker-large"):
        self.vector_store = vector_store
        self.embedder = embedder
        self.reranker = CrossEncoder(reranker_model)

    def search(self, query, top_k=50, top_n=5):
        # 第一阶段：向量粗检索
        query_embedding = self.embedder.embed_query(query)
        coarse_results = self.vector_store.search(query_embedding, top_k=top_k)

        # 第二阶段：Reranker精排
        pairs = [[query, r["text"]] for r in coarse_results]
        scores = self.reranker.predict(pairs)

        # 合并分数并排序
        for i, result in enumerate(coarse_results):
            result["rerank_score"] = float(scores[i])

        coarse_results.sort(key=lambda x: x["rerank_score"], reverse=True)
        return coarse_results[:top_n]
```

#### 3.2.3 Cohere Reranker API

```python
import cohere

co = cohere.Client("your-api-key")

def cohere_rerank(query, documents, top_n=5):
    """使用Cohere Reranker API"""
    response = co.rerank(
        model="rerank-multilingual-v3.0",
        query=query,
        documents=[doc["text"] for doc in documents],
        top_n=top_n,
    )

    results = []
    for result in response.results:
        results.append({
            "text": documents[result.index]["text"],
            "relevance_score": result.relevance_score,
            "index": result.index,
        })
    return results
```

#### 3.2.4 LLM-based Reranker

```python
def llm_rerank(query, documents, llm_client, top_n=5):
    """使用LLM作为Reranker"""
    doc_list = "\n".join([f"[{i}] {doc['text'][:200]}" for i, doc in enumerate(documents)])

    prompt = f"""给定一个查询和一组文档，请评估每个文档与查询的相关性。

查询：{query}

文档：
{doc_list}

请返回最相关的{top_n}个文档的编号和相关性评分(1-10)。
格式：文档编号:评分"""

    response = llm_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content
```

---

### Day 3: 多格式文档处理

#### 3.3.1 PDF解析

```python
# 方法1：PyPDF2（基础，纯文本提取）
from PyPDF2 import PdfReader

def extract_pdf_pypdf2(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# 方法2：pdfplumber（支持表格提取）
import pdfplumber

def extract_pdf_plumber(file_path):
    text = ""
    tables = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
            # 提取表格
            for table in page.extract_tables():
                tables.append(table)
    return text, tables

# 方法3：PyMuPDF / fitz（支持图片提取）
import fitz  # pip install PyMuPDF

def extract_pdf_pymupdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    images = []
    for page_num, page in enumerate(doc):
        text += page.get_text() + "\n"
        # 提取图片
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.n < 5:  # GRAY or RGB
                images.append(pix.tobytes("png"))
            else:  # CMYK → RGB
                pix = fitz.Pixmap(fitz.csRGB, pix)
                images.append(pix.tobytes("png"))
    return text, images

# 方法4：Unstructured（通用解析器）
from unstructured.partition.auto import partition

def extract_universal(file_path):
    elements = partition(filename=file_path)
    return "\n".join([str(el) for el in elements])
```

#### 3.3.2 Word/Excel/HTML/Markdown解析

```python
# Word解析
from docx import Document  # pip install python-docx

def extract_word(file_path):
    doc = Document(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    # 提取表格
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join([cell.text for cell in row.cells])
            text += row_text + "\n"
    return text

# Excel解析
import pandas as pd

def extract_excel(file_path):
    dfs = pd.read_excel(file_path, sheet_name=None)  # 读取所有sheet
    result = {}
    for sheet_name, df in dfs.items():
        result[sheet_name] = df.to_string()
        # 或者转为Markdown表格
        result[sheet_name + "_md"] = df.to_markdown()
    return result

# HTML解析
from bs4 import BeautifulSoup

def extract_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
    # 移除脚本和样式
    for script in soup(["script", "style"]):
        script.decompose()
    return soup.get_text(separator='\n', strip=True)

# Markdown解析
import mistune

def extract_markdown(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        md_text = f.read()
    # mistune可以将Markdown转为纯文本或HTML
    md = mistune.create_markdown()
    html = md(md_text)
    # 再用BeautifulSoup提取文本
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text(separator='\n', strip=True)
```

#### 3.3.3 OCR处理

```python
# 方法1：pytesseract（基于Tesseract OCR引擎）
import pytesseract
from PIL import Image

def ocr_image(image_path, lang='chi_sim+eng'):
    """使用Tesseract进行OCR"""
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang=lang)
    return text

# 方法2：PaddleOCR（中文效果更好）
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='ch')

def ocr_image_paddle(image_path):
    """使用PaddleOCR进行OCR"""
    result = ocr.ocr(image_path, cls=True)
    text_lines = []
    for line in result[0]:
        text_lines.append(line[1][0])  # 提取文本
    return "\n".join(text_lines)

# 处理扫描版PDF
def extract_scanned_pdf(file_path):
    """提取扫描版PDF的文字（OCR）"""
    doc = fitz.open(file_path)
    full_text = ""
    for page_num, page in enumerate(doc):
        # 将PDF页面转为图片
        pix = page.get_pixmap(dpi=300)
        img_path = f"temp_page_{page_num}.png"
        pix.save(img_path)
        # OCR
        text = ocr_image_paddle(img_path)
        full_text += text + "\n"
        os.remove(img_path)
    return full_text
```

#### 3.3.4 表格处理策略

```python
def table_to_markdown(table_data):
    """将表格数据转为Markdown格式"""
    if not table_data:
        return ""
    header = "| " + " | ".join(table_data[0]) + " |"
    separator = "| " + " | ".join(["---"] * len(table_data[0])) + " |"
    rows = []
    for row in table_data[1:]:
        rows.append("| " + " | ".join(row) + " |")
    return header + "\n" + separator + "\n" + "\n".join(rows)

def table_to_description(table_data):
    """将表格转为自然语言描述"""
    headers = table_data[0]
    descriptions = []
    for row in table_data[1:]:
        desc_parts = [f"{headers[i]}是{row[i]}" for i in range(len(headers))]
        descriptions.append("，".join(desc_parts) + "。")
    return "\n".join(descriptions)
```

---

### Day 4: 高级RAG模式

#### 3.4.1 Self-RAG

```
Self-RAG流程：
用户提问
    ↓
LLM判断：是否需要检索？(Retrieve: YES/NO)
    ↓ YES                    ↓ NO
检索相关文档              直接生成
    ↓                        ↓
LLM判断：文档相关吗？     LLM判断：有幻觉吗？(ISREL: RELEVANT/IRRELEVANT)
(ISREL相关标签)
    ↓ 相关                   ↓
基于文档生成回答           直接输出
    ↓
LLM自评：生成质量如何？
(ISSUP: 支持上下文/不支持)
    ↓ 质量好                 ↓ 质量差
输出回答                  重新检索或重新生成
```

```python
class SelfRAG:
    """Self-RAG实现：让LLM自我判断检索和生成质量"""

    def __init__(self, llm_client, retriever):
        self.llm = llm_client
        self.retriever = retriever

    def query(self, question, max_retries=3):
        # Step 1: 判断是否需要检索
        need_retrieve = self._judge_need_retrieve(question)

        if not need_retrieve:
            return self._direct_generate(question)

        # Step 2: 检索
        for attempt in range(max_retries):
            docs = self.retriever.search(question, top_k=5)

            # Step 3: 评估文档相关性
            relevant_docs = self._filter_relevant_docs(question, docs)
            if relevant_docs:
                break
            # 如果没有相关文档，尝试改写查询重试
            question = self._rewrite_query(question)
        else:
            return "抱歉，未能找到相关信息来回答您的问题。"

        # Step 4: 生成回答
        answer = self._generate_with_context(question, relevant_docs)

        # Step 5: 自评生成质量
        if self._judge_answer_quality(question, answer, relevant_docs):
            return answer
        else:
            # 质量不好，重新生成
            return self._generate_with_context(question, relevant_docs, retry=True)

    def _judge_need_retrieve(self, question):
        """判断是否需要检索"""
        prompt = f"""判断回答以下问题是否需要查找外部资料。
如果问题涉及具体事实、数据、事件，回答YES。
如果是常识、创意、观点类问题，回答NO。

问题：{question}
需要检索？(YES/NO)："""

        response = self.llm.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": prompt}], temperature=0
        )
        return "YES" in response.choices[0].message.content

    def _filter_relevant_docs(self, question, docs):
        """过滤不相关的文档"""
        relevant = []
        for doc in docs:
            prompt = f"""判断以下文档是否与问题相关。
问题：{question}
文档：{doc['text'][:300]}
是否相关？(RELEVANT/IRRELEVANT)："""
            response = self.llm.chat.completions.create(
                model="gpt-4o", messages=[{"role": "user", "content": prompt}], temperature=0
            )
            if "RELEVANT" in response.choices[0].message.content:
                relevant.append(doc)
        return relevant

    def _judge_answer_quality(self, question, answer, docs):
        """判断生成质量"""
        context = "\n".join([d["text"] for d in docs])
        prompt = f"""评估回答质量。
问题：{question}
上下文：{context}
回答：{answer}

回答是否：(1)忠实于上下文 (2)完整回答了问题？
(SUPPORTED/UNSUPPORTED)："""
        response = self.llm.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": prompt}], temperature=0
        )
        return "SUPPORTED" in response.choices[0].message.content
```

#### 3.4.2 Corrective RAG (CRAG)

```
CRAG流程：
用户提问 → 检索 → 评估检索质量
                          ↓
              质量好 → 直接用检索结果生成
              质量差 → 用Web搜索补充 → 合并结果 → 生成
```

```python
class CorrectiveRAG:
    """CRAG：检索结果不好时自动补充"""

    def __init__(self, llm_client, retriever, web_search_func):
        self.llm = llm_client
        self.retriever = retriever
        self.web_search = web_search_func

    def query(self, question):
        # 检索
        docs = self.retriever.search(question, top_k=5)

        # 评估检索质量
        quality_score = self._evaluate_retrieval_quality(question, docs)

        if quality_score >= 0.7:
            # 质量好，直接用
            context = "\n".join([d["text"] for d in docs])
        else:
            # 质量差，用Web搜索补充
            web_results = self.web_search(question)
            context = "\n".join([d["text"] for d in docs] + web_results)

        return self._generate(question, context)

    def _evaluate_retrieval_quality(self, question, docs):
        """评估检索质量分数"""
        doc_texts = "\n".join([d["text"][:200] for d in docs])
        prompt = f"""评估以下检索结果与问题的相关性。
问题：{question}
检索结果：{doc_texts}
相关性评分(0-1)："""
        response = self.llm.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": prompt}], temperature=0
        )
        try:
            return float(response.choices[0].message.content.strip())
        except ValueError:
            return 0.5
```

#### 3.4.3 Adaptive RAG

```python
class AdaptiveRAG:
    """根据问题复杂度选择检索策略"""

    def __init__(self, llm_client, simple_retriever, complex_retriever):
        self.llm = llm_client
        self.simple_retriever = simple_retriever
        self.complex_retriever = complex_retriever

    def query(self, question):
        complexity = self._assess_complexity(question)

        if complexity == "simple":
            # 简单问题：直接回答或简单检索
            return self._simple_query(question)
        elif complexity == "moderate":
            # 中等问题：单次检索
            docs = self.simple_retriever.search(question, top_k=5)
            return self._generate_with_docs(question, docs)
        else:
            # 复杂问题：多步检索+推理
            return self._complex_query(question)

    def _assess_complexity(self, question):
        prompt = f"""评估以下问题的复杂度：
- simple: 简单事实性问题，常识即可回答
- moderate: 需要查找特定信息
- complex: 需要多步推理或综合多个信息源

问题：{question}
复杂度(simple/moderate/complex)："""
        response = self.llm.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": prompt}], temperature=0
        )
        return response.choices[0].message.content.strip().lower()
```

---

### Day 5: RAG工程化

#### 3.5.1 Embedding缓存

```python
import hashlib
import json
import os

class EmbeddingCache:
    """Embedding缓存，避免重复计算"""

    def __init__(self, embedder, cache_dir="./embedding_cache"):
        self.embedder = embedder
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_key(self, text):
        """生成缓存key"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def embed_texts(self, texts, batch_size=32):
        """带缓存的Embedding计算"""
        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []

        # 检查缓存
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.npy")
            if os.path.exists(cache_path):
                results[i] = np.load(cache_path)
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # 计算未缓存的
        if uncached_texts:
            new_embeddings = self.embedder.embed_texts(uncached_texts)
            for i, idx in enumerate(uncached_indices):
                results[idx] = new_embeddings[i]
                # 保存到缓存
                cache_key = self._get_cache_key(uncached_texts[i])
                cache_path = os.path.join(self.cache_dir, f"{cache_key}.npy")
                np.save(cache_path, new_embeddings[i])

        return np.array(results)
```

#### 3.5.2 LLM响应缓存

```python
import redis
import json

class LLMResponseCache:
    """LLM响应缓存"""

    def __init__(self, redis_url="redis://localhost:6379", ttl=3600):
        self.redis = redis.from_url(redis_url)
        self.ttl = ttl

    def _cache_key(self, model, messages, **kwargs):
        """生成缓存key"""
        key_data = json.dumps({
            "model": model,
            "messages": messages,
            "kwargs": kwargs
        }, sort_keys=True)
        return f"llm_cache:{hashlib.md5(key_data.encode()).hexdigest()}"

    def get_or_call(self, llm_func, model, messages, **kwargs):
        """缓存优先的LLM调用"""
        cache_key = self._cache_key(model, messages, **kwargs)

        # 查缓存
        cached = self.redis.get(cache_key)
        if cached:
            return json.loads(cached)

        # 调用LLM
        result = llm_func(model, messages, **kwargs)

        # 存缓存
        self.redis.setex(cache_key, self.ttl, json.dumps(result))
        return result
```

#### 3.5.3 增量索引更新

```python
class IncrementalIndexer:
    """增量索引更新：只处理新增文档"""

    def __init__(self, vector_store, embedder, index_meta_path="./index_meta.json"):
        self.vector_store = vector_store
        self.embedder = embedder
        self.meta_path = index_meta_path
        self.indexed_files = self._load_meta()

    def _load_meta(self):
        """加载索引元数据"""
        if os.path.exists(self.meta_path):
            with open(self.meta_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_meta(self):
        """保存索引元数据"""
        with open(self.meta_path, 'w') as f:
            json.dump(self.indexed_files, f, indent=2)

    def _get_file_hash(self, file_path):
        """计算文件哈希"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            hasher.update(f.read())
        return hasher.hexdigest()

    def update(self, file_paths):
        """增量更新索引"""
        new_files = []
        changed_files = []

        for fp in file_paths:
            file_hash = self._get_file_hash(fp)
            if fp not in self.indexed_files:
                new_files.append(fp)
            elif self.indexed_files[fp]["hash"] != file_hash:
                changed_files.append(fp)

        if not new_files and not changed_files:
            print("没有新增或变更的文档")
            return

        # 处理新增文件
        if new_files:
            print(f"新增文档: {len(new_files)}个")
            self._index_files(new_files)

        # 处理变更文件（先删除旧索引）
        if changed_files:
            print(f"变更文档: {len(changed_files)}个")
            for fp in changed_files:
                old_ids = self.indexed_files[fp]["chunk_ids"]
                self.vector_store.delete(old_ids)
            self._index_files(changed_files)

        self._save_meta()
```

---

### Day 6: LangChain/LlamaIndex RAG

#### 3.6.1 LangChain版RAG

```python
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# 1. 加载文档
loaders = [
    PyPDFLoader("./docs/document.pdf"),
    TextLoader("./docs/document.txt"),
]
documents = []
for loader in loaders:
    documents.extend(loader.load())

# 2. 切分
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
chunks = text_splitter.split_documents(documents)

# 3. Embedding + 向量存储
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5")
vectorstore = FAISS.from_documents(chunks, embeddings)

# 4. 创建检索器
retriever = vectorstore.as_retriever(
    search_type="mmr",  # 最大边际相关性（去重+多样性）
    search_kwargs={"k": 5}
)

# 5. 创建QA链
llm = ChatOpenAI(model="gpt-4o", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # stuff/map_reduce/refine
    retriever=retriever,
    return_source_documents=True,
)

# 6. 查询
result = qa_chain({"query": "什么是机器学习？"})
print(result["result"])
print(f"来源文档: {[doc.metadata for doc in result['source_documents']]}")
```

#### 3.6.2 LlamaIndex版RAG

```python
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1. 配置
Settings.llm = OpenAI(model="gpt-4o", temperature=0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-zh-v1.5")

# 2. 加载文档
documents = SimpleDirectoryReader("./docs/").load_data()

# 3. 切分
node_parser = SentenceSplitter(chunk_size=500, chunk_overlap=50)
nodes = node_parser.get_nodes_from_documents(documents)

# 4. 创建索引
index = VectorStoreIndex(nodes)

# 5. 查询
query_engine = index.as_query_engine(similarity_top_k=5)
response = query_engine.query("什么是机器学习？")
print(response)

# 带引用的查询
query_engine = index.as_query_engine(
    similarity_top_k=5,
    response_mode="tree_summarize",  # refine/compact/tree_summarize
)
response = query_engine.query("详细解释深度学习的发展历程")
print(response)
print(f"引用来源: {[n.metadata for n in response.source_nodes]}")
```

#### 3.6.3 框架对比

```
| 特征        | LangChain                | LlamaIndex               |
|------------|--------------------------|--------------------------|
| 定位        | 通用LLM应用框架           | 专注数据索引和检索        |
| 灵活性      | 高，模块化程度高          | 中，索引/查询流程相对固定 |
| 学习曲线    | 较陡（概念多）            | 较平缓                   |
| RAG支持     | 完善                     | 更完善（核心功能）        |
| Agent支持   | 强（LangGraph/LangSmith）| 一般                     |
| 生产就绪    | 需要LangSmith监控        | 相对简单                 |
| 适用场景    | 复杂Agent/多步骤工作流    | 数据密集型RAG应用        |

建议：RAG场景优先LlamaIndex，Agent场景优先LangChain
```

---

### Day 7: 实战项目8 - 企业知识库RAG

#### 项目架构

```
enterprise_rag/
├── app.py                 # FastAPI主入口
├── config.py              # 配置文件
├── document_processor/    # 文档处理模块
│   ├── __init__.py
│   ├── pdf_parser.py
│   ├── word_parser.py
│   ├── excel_parser.py
│   └── chunker.py
├── retrieval/             # 检索模块
│   ├── __init__.py
│   ├── embedder.py
│   ├── vector_store.py
│   ├── bm25_retriever.py
│   ├── hybrid_retriever.py
│   └── reranker.py
├── generation/            # 生成模块
│   ├── __init__.py
│   └── generator.py
├── evaluation/            # 评估模块
│   ├── __init__.py
│   └── evaluator.py
├── data/                  # 数据目录
│   ├── documents/
│   └── index/
└── requirements.txt
```

#### 核心代码框架

```python
# app.py - FastAPI主入口
from fastapi import FastAPI, UploadFile, File, Query
from typing import List, Optional
import os

app = FastAPI(title="企业知识库RAG系统")

# 全局组件
doc_processor = DocumentProcessor()
retriever = HybridRetriever(alpha=0.7)
reranker = Reranker(model="BAAI/bge-reranker-large")
generator = AnswerGenerator()

@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """上传文档"""
    results = []
    for file in files:
        # 保存文件
        file_path = f"./data/documents/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # 处理文档
        process_result = doc_processor.process(file_path)
        results.append({
            "filename": file.filename,
            "chunks": process_result["num_chunks"],
            "status": "success"
        })

    return {"results": results}

@app.get("/query")
async def query(
    question: str = Query(...),
    top_k: int = Query(50),
    top_n: int = Query(5),
    with_sources: bool = Query(True)
):
    """查询"""
    # 1. 混合检索
    docs = retriever.search(question, top_k=top_k)

    # 2. Reranker重排序
    reranked = reranker.rerank(question, docs, top_n=top_n)

    # 3. 生成回答
    answer = generator.generate(question, reranked)

    response = {"answer": answer}
    if with_sources:
        response["sources"] = [
            {"text": d["text"][:200], "score": d.get("rerank_score", d["score"])}
            for d in reranked
        ]

    return response

@app.get("/evaluate")
async def evaluate():
    """运行评估"""
    evaluator = RAGEvaluator(retriever, reranker, generator)
    results = evaluator.run_evaluation()
    return results
```

---

## 四、代码练习

### Day 1 练习：实现混合检索

```
任务：
1. 准备100个文档片段的测试数据集
2. 实现HybridRetriever（向量+BM25融合）
3. 对比单一向量检索、单一BM25、混合检索的Recall@5
4. 测试不同alpha值(0.3/0.5/0.7)的效果
5. 输出：hybrid_retriever.py + 对比结果
```

### Day 2 练习：对比Bi-Encoder和Cross-Encoder

```
任务：
1. 准备20个query和100个文档的测试集
2. 用Bi-Encoder做向量检索，记录Top-10结果
3. 用Cross-Encoder对Bi-Encoder的Top-50结果重排序
4. 对比两者的检索质量（人工标注正确答案）
5. 输出：reranker_comparison.py + 对比报告
```

### Day 4 练习：实现Self-RAG或CRAG

```
任务：
1. 选择Self-RAG或CRAG实现
2. 准备测试数据（包含简单和复杂问题）
3. 对比Naive RAG和高级RAG的回答质量
4. 输出：advanced_rag.py + 对比结果
```

### Day 7 练习：完成企业知识库RAG

```
任务：
1. 完成Day 7的项目框架代码
2. 实现：多格式文档上传→解析→切分→混合检索→Reranker→生成→引用溯源
3. 准备评估数据集，跑评估
4. 输出：enterprise_rag/ 完整项目
```

---

## 五、本周产出

| 产出物 | 说明 | 完成标准 |
|--------|------|----------|
| hybrid_retriever.py | 混合检索实现 | 向量+BM25融合，alpha可调 |
| reranker_comparison.py | Reranker对比脚本 | Bi-Encoder vs Cross-Encoder |
| advanced_rag.py | 高级RAG模式实现 | Self-RAG或CRAG |
| enterprise_rag/ | 企业知识库项目 | 完整Pipeline+FastAPI接口 |
| 学习笔记.md | 本周学习总结 | 包含核心概念和个人理解 |

---

## 六、自测题

### 题目

1. **混合检索为什么比单一检索好？**

<details>
<summary>参考答案</summary>

向量检索擅长语义匹配（"电脑"和"计算机"被视为相似），但可能遗漏包含精确关键词的文档；BM25擅长精确匹配（搜索特定术语、产品编号），但无法理解语义。混合检索融合两者优势：向量检索捕获语义相似性，BM25捕获精确匹配，分数归一化后加权融合，取长补短。实验表明混合检索在大多数场景下优于单一检索方法。
</details>

2. **Reranker的作用？为什么需要两阶段检索？**

<details>
<summary>参考答案</summary>

Reranker使用Cross-Encoder对query和document做联合编码，计算更精确的相关性分数。两阶段检索的原因是效率：Cross-Encoder需要对每个query-doc pair单独计算，无法预计算，如果对全部10万文档都做Cross-Encoder评分，延迟不可接受。因此先快速粗检索出Top-50候选（Bi-Encoder向量检索），再对50个候选做精确重排序选出Top-5，兼顾速度和精度。
</details>

3. **HyDE的原理？为什么有效？**

<details>
<summary>参考答案</summary>

HyDE先用LLM为查询生成一个"假设性答案"，然后用这个假设性答案的Embedding去检索文档。有效的原因：用户的查询通常很短且表述不精确，而LLM生成的假设性答案在词汇和表达方式上更接近真实文档，所以用假设性答案去检索能找到更相关的文档。即使假设性答案本身不完全正确，它的表达方式（使用专业术语、完整句子）更适合做检索。
</details>

4. **Self-RAG的判断流程？**

<details>
<summary>参考答案</summary>

Self-RAG包含三个自省判断：(1) Retrieve判断：LLM判断当前问题是否需要检索外部知识，如果不需要（如常识问题），直接生成；(2) ISREL判断：对检索到的文档判断是否与问题相关，过滤不相关文档；(3) ISSUP判断：对生成的回答判断是否忠实于检索到的上下文。每个判断点如果结果不理想，都会触发重新检索或重新生成，形成自我纠正的闭环。
</details>

5. **RAG工程化中缓存策略有哪些？**

<details>
<summary>参考答案</summary>

三种主要缓存策略：(1) Embedding缓存：对已计算过的文本的Embedding做本地缓存（hash+文件映射），避免重复计算，增量索引时只需计算新文档；(2) LLM响应缓存：对相同的问答对缓存LLM的生成结果（可用Redis），相同问题直接返回，避免重复调用LLM；(3) 查询级缓存：对热门查询的完整检索+生成结果做缓存。此外还有增量索引更新（只处理新增/变更文档，用文件hash判断是否变更）。
</details>

---

## 七、Java开发者提示

### 混合检索类比

```
混合检索 ≈ Java中的多数据源联合查询

// Java中同时查MySQL和ES
List<Result> sqlResults = mysqlRepo.search(keyword);     // 精确匹配
List<Result> esResults = elasticsearchRepo.search(keyword); // 全文搜索
List<Result> combined = mergeAndRank(sqlResults, esResults); // 合并排序

// RAG混合检索
vector_results = vector_store.search(query)   // 语义检索
bm25_results = bm25_index.search(query)       // 关键词检索
combined = alpha * vector + (1-alpha) * bm25   # 加权融合
```

### 两阶段检索类比

```
两阶段检索 ≈ Java中的多级缓存/漏斗筛选

// Java电商搜索
Step1: ES粗筛 → 从100万商品中筛选出1000个候选
Step2: 精排模型 → 对1000个候选用复杂模型打分
Step3: 返回Top-20

// RAG两阶段
Step1: 向量粗检索 → 从10万文档中检索Top-50
Step2: Reranker精排 → 对Top-50精确评分
Step3: 返回Top-5给LLM

思路完全一致：粗筛降低计算量，精排保证质量
```

### 高级RAG模式类比

```
Self-RAG ≈ Java中的自检/补偿机制

// Java中的重试+校验
@Retry(maxAttempts = 3)
public Result process(Request req) {
    Result result = doProcess(req);
    if (!validate(result)) {
        throw new RetryableException("质量不达标"); // 触发重试
    }
    return result;
}

// Self-RAG
answer = generate(question, docs)
if not self_judge(answer):  # 质量自评
    answer = regenerate(question, docs)  # 重新生成

都是"执行→检验→不合格则重做"的模式
```

### RAG工程化类比

```
RAG工程化 ≈ Java Web应用的工程化最佳实践

缓存策略：
  Java: Redis缓存热点数据 + Caffeine本地缓存
  RAG:  Redis缓存LLM响应 + 本地文件缓存Embedding

增量更新：
  Java: 数据库监听binlog，增量同步到ES
  RAG:  文件hash比对，只处理新增/变更文档

并发处理：
  Java: CompletableFuture / 线程池
  RAG:  asyncio + Semaphore控制并发
```
