"""
W29-D7: 最小可用RAG (MVP)
=========================
功能:
  - 完整Pipeline: 加载文档 → 切分 → 嵌入 → 存储 → 检索 → 生成回答
  - 不依赖外部LLM API (模拟生成部分)
  - 可替换为真实LLM API
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import time
import re


# ============================================================
# 1. Pipeline组件 (复用前面几天的模块, 简化版)
# ============================================================

@dataclass
class Chunk:
    """文本块"""
    id: str
    content: str
    metadata: Dict = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


class SimpleTextSplitter:
    """简单文本切分器"""

    def __init__(self, chunk_size: int = 300, overlap: int = 30):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str) -> List[str]:
        """将文本切分为块"""
        chunks = []
        step = self.chunk_size - self.overlap
        i = 0
        while i < len(text):
            end = min(i + self.chunk_size, len(text))
            chunk = text[i:end].strip()
            if chunk:
                chunks.append(chunk)
            i += step
            if i >= len(text):
                break
        return chunks


class SimpleEmbedder:
    """简单嵌入器"""

    def __init__(self, dimension: int = 64):
        self.dimension = dimension

    def embed(self, text: str) -> np.ndarray:
        """将文本转为向量 (模拟)"""
        vector = np.zeros(self.dimension)
        # 基于字符的简单哈希嵌入
        words = text.lower().split()
        for word in words:
            h = hash(word) % self.dimension
            vector[h] += 1.0

        # 添加字符级特征
        for i, c in enumerate(text[:self.dimension]):
            vector[i % self.dimension] += ord(c) * 0.01

        # 归一化
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        return np.array([self.embed(t) for t in texts])


class SimpleVectorStore:
    """简单向量存储"""

    def __init__(self, dimension: int = 64):
        self.dimension = dimension
        self.chunks: List[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self._embedder = SimpleEmbedder(dimension)

    def add_chunks(self, texts: List[str], source: str = ""):
        """添加文本块"""
        for i, text in enumerate(texts):
            chunk = Chunk(
                id=f"{source}_chunk_{i}",
                content=text,
                metadata={"source": source, "index": i},
            )
            chunk.embedding = self._embedder.embed(text)
            self.chunks.append(chunk)

        # 重建索引
        self.embeddings = np.array(
            [c.embedding for c in self.chunks]
        )

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """检索最相关的文本块"""
        if not self.chunks:
            return []

        query_vec = self._embedder.embed(query)

        # 余弦相似度
        scores = self.embeddings @ query_vec
        norms = np.linalg.norm(self.embeddings, axis=1)
        norms = np.where(norms == 0, 1e-8, norms)
        scores = scores / norms

        # 排序取top_k
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "content": self.chunks[idx].content,
                "score": float(scores[idx]),
                "metadata": self.chunks[idx].metadata,
            })

        return results

    def count(self) -> int:
        return len(self.chunks)


class SimpleLLM:
    """
    简单LLM - 模拟生成回答
    生产环境替换为真实LLM API调用
    """

    def generate(self, prompt: str) -> str:
        """生成回答 (模拟)"""
        # 模拟: 基于上下文的简单回答
        # 在真实场景中, 这里调用GPT-4/Claude等API

        if "上下文" in prompt and "问题" in prompt:
            # 提取上下文和问题
            context_match = re.search(
                r'上下文[：:]\s*(.+?)(?=问题[：:]|$)',
                prompt, re.DOTALL
            )
            question_match = re.search(
                r'问题[：:]\s*(.+?)$',
                prompt, re.DOTALL
            )

            context = context_match.group(1).strip() if context_match else ""
            question = question_match.group(1).strip() if question_match else ""

            if context and question:
                # 简单规则: 找到上下文中与问题关键词最相关的句子
                sentences = re.split(r'[。！？\.\!\?]', context)
                sentences = [s.strip() for s in sentences if s.strip()]

                # 关键词匹配
                keywords = set(question.replace("？", "").replace("什么", "")
                              .replace("如何", "").replace("哪些", "").split())

                scored = []
                for sent in sentences:
                    overlap = sum(1 for k in keywords if k in sent)
                    scored.append((sent, overlap))

                scored.sort(key=lambda x: x[1], reverse=True)

                if scored and scored[0][1] > 0:
                    return (
                        f"根据文档内容, {scored[0][0]}。"
                        f"\n\n[来源: 检索到的相关文档片段]"
                    )

                return (
                    "根据检索到的上下文信息, "
                    "以下是相关内容:\n"
                    f"{context[:200]}...\n\n"
                    "[注: 此为模拟回答, 实际应用中请接入LLM API]"
                )

        return "请提供上下文和问题以生成回答。"

    def generate_with_real_llm(
        self, prompt: str, api_key: str = ""
    ) -> str:
        """使用真实LLM API生成回答"""
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "你是一个文档问答助手, 请根据提供的上下文回答问题。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=500,
            )
            return response.choices[0].message.content

        except ImportError:
            return "[错误: 请安装openai库]"
        except Exception as e:
            return f"[LLM调用失败: {e}]"


# ============================================================
# 2. RAG Pipeline
# ============================================================

class RAGPipeline:
    """RAG完整Pipeline"""

    def __init__(
        self,
        chunk_size: int = 300,
        chunk_overlap: int = 30,
        top_k: int = 3,
    ):
        self.splitter = SimpleTextSplitter(chunk_size, chunk_overlap)
        self.embedder = SimpleEmbedder(64)
        self.vector_store = SimpleVectorStore(64)
        self.llm = SimpleLLM()
        self.top_k = top_k

        # 统计信息
        self.stats = {
            "documents_loaded": 0,
            "chunks_created": 0,
            "queries_processed": 0,
        }

    def ingest(self, text: str, source: str = "unknown") -> int:
        """导入文档: 切分 + 嵌入 + 存储"""
        # 切分
        chunks = self.splitter.split(text)
        self.stats["documents_loaded"] += 1
        self.stats["chunks_created"] += len(chunks)

        # 嵌入并存储
        self.vector_store.add_chunks(chunks, source)

        return len(chunks)

    def query(self, question: str) -> Dict:
        """查询: 检索 + 生成"""
        start_time = time.time()

        # 1. 检索相关文档
        results = self.vector_store.search(question, self.top_k)

        # 2. 组装上下文
        context_parts = []
        sources = []
        for r in results:
            context_parts.append(r["content"])
            sources.append({
                "content": r["content"][:100],
                "score": round(r["score"], 4),
                "source": r["metadata"].get("source", ""),
            })

        context = "\n\n".join(context_parts)

        # 3. 构建Prompt
        prompt = self._build_prompt(question, context)

        # 4. 生成回答
        answer = self.llm.generate(prompt)

        elapsed = (time.time() - start_time) * 1000
        self.stats["queries_processed"] += 1

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "context_used": context[:300] + "..." if len(context) > 300 else context,
            "time_ms": round(elapsed, 2),
        }

    def _build_prompt(self, question: str, context: str) -> str:
        """构建Prompt"""
        return f"""请根据以下上下文信息回答问题。如果上下文中没有相关信息, 请说明。

上下文：
{context}

问题：{question}

请给出准确、简洁的回答, 并引用来源。"""

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            **self.stats,
            "vector_store_size": self.vector_store.count(),
        }


# ============================================================
# 3. 演示: 完整RAG Pipeline
# ============================================================

def demo():
    """演示完整RAG Pipeline"""

    print("=" * 60)
    print("W29-D7: 最小可用RAG (MVP)")
    print("=" * 60)

    # ---- 1. 准备知识库文档 ----
    print("\n[步骤1] 准备知识库文档")

    documents = [
        {
            "source": "ai_intro.txt",
            "content": (
                "人工智能（Artificial Intelligence，简称AI）是计算机科学的一个重要分支，"
                "致力于研究和开发能够模拟人类智能行为的系统和技术。人工智能的主要研究方向包括："
                "机器学习、深度学习、自然语言处理、计算机视觉、机器人技术等。"
                "人工智能技术已经广泛应用于医疗、金融、教育、交通等领域，"
                "正在深刻改变人类的生产和生活方式。"
            ),
        },
        {
            "source": "rag_guide.txt",
            "content": (
                "检索增强生成（RAG）是一种结合信息检索和文本生成的AI技术框架。"
                "RAG系统的工作流程包括：首先将文档切分为文本块，然后将文本块转换为向量表示，"
                "存储在向量数据库中。当用户提出问题时，系统会检索最相关的文档片段，"
                "然后将检索到的上下文和用户问题一起发送给大语言模型（LLM），"
                "由LLM生成最终答案。RAG的优势在于可以结合私有知识库，"
                "提供准确、可溯源的回答，同时减少LLM的幻觉问题。"
            ),
        },
        {
            "source": "vector_db.txt",
            "content": (
                "向量数据库是专门用于存储和检索高维向量的数据库系统。"
                "常见的向量数据库包括Chroma、FAISS、Milvus、Pinecone等。"
                "向量数据库的核心功能是相似度搜索，常用的相似度计算方法包括："
                "余弦相似度、欧几里得距离、内积等。"
                "在RAG系统中，向量数据库用于存储文档的嵌入向量，"
                "并支持快速检索与查询最相关的文档片段。"
                "选择向量数据库时需要考虑的因素包括：数据规模、查询延迟、"
                "部署方式（本地/云端）、是否支持过滤等。"
            ),
        },
        {
            "source": "embedding.txt",
            "content": (
                "文本嵌入（Text Embedding）是将文本转换为固定长度的数值向量的技术。"
                "常见的嵌入模型包括OpenAI的text-embedding-3-small、"
                "BGE（BAAI General Embedding）、Sentence-BERT等。"
                "嵌入模型的质量直接影响RAG系统的检索效果。"
                "选择嵌入模型时需要考虑：向量维度、多语言支持、"
                "推理速度、模型大小等因素。"
                "对于中文场景，BGE系列模型表现优秀。"
            ),
        },
    ]

    # ---- 2. 构建Pipeline并导入文档 ----
    print("\n[步骤2] 构建RAG Pipeline并导入文档")

    rag = RAGPipeline(chunk_size=200, chunk_overlap=20, top_k=3)

    for doc in documents:
        chunk_count = rag.ingest(doc["content"], doc["source"])
        print(f"  导入 {doc['source']}: {chunk_count} 个文本块")

    stats = rag.get_stats()
    print(f"\n  总计: {stats['documents_loaded']} 个文档, "
          f"{stats['chunks_created']} 个文本块")

    # ---- 3. 问答演示 ----
    print("\n[步骤3] 问答演示")
    print("=" * 60)

    questions = [
        "什么是人工智能?",
        "RAG系统的工作流程是什么?",
        "常见的向量数据库有哪些?",
        "如何选择嵌入模型?",
        "RAG有什么优势?",
    ]

    for q in questions:
        print(f"\n问题: {q}")
        result = rag.query(q)
        print(f"回答: {result['answer']}")
        print(f"来源: {len(result['sources'])} 个相关片段")
        print(f"耗时: {result['time_ms']:.1f}ms")

        # 显示来源片段
        for s in result["sources"]:
            print(f"  - [{s['score']:.4f}] {s['content'][:50]}...")

    # ---- 4. Pipeline统计 ----
    print("\n" + "=" * 60)
    print("Pipeline统计")
    print("=" * 60)
    final_stats = rag.get_stats()
    for k, v in final_stats.items():
        print(f"  {k}: {v}")

    # ---- 5. 架构总结 ----
    print("\n" + "=" * 60)
    print("RAG MVP架构总结")
    print("=" * 60)
    print("""
    用户提问 → 嵌入查询 → 向量检索 → 上下文组装 → LLM生成 → 返回答案

    组件:
    ├─ 文本切分器 (SimpleTextSplitter)
    │   └─ 固定长度切分, 支持重叠
    ├─ 嵌入器 (SimpleEmbedder)
    │   └─ 哈希模拟嵌入 (可替换为OpenAI/BGE)
    ├─ 向量存储 (SimpleVectorStore)
    │   └─ NumPy实现, 余弦相似度检索
    └─ LLM (SimpleLLM)
        └─ 规则匹配模拟 (可替换为GPT-4/Claude)

    升级路径:
    1. 替换嵌入器 → OpenAI / BGE
    2. 替换向量存储 → Chroma / FAISS
    3. 替换LLM → GPT-4 / Claude / 本地模型
    4. 添加重排序 → Cross-encoder
    5. 添加混合检索 → BM25 + 向量
    """)


# ============================================================
# 运行演示
# ============================================================

if __name__ == "__main__":
    demo()
