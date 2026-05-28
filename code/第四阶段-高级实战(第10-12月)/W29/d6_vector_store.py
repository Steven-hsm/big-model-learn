"""
W29-D6: 向量存储实现
====================
功能:
  - 使用NumPy实现向量存储
  - 文档嵌入 (简单哈希模拟 + 真实嵌入备选)
  - 相似度检索 (余弦相似度)
  - CRUD操作
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import hashlib
import json
import time


# ============================================================
# 1. 数据模型
# ============================================================

@dataclass
class VectorDocument:
    """向量文档"""
    id: str
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """序列化为字典"""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
        }


@dataclass
class SearchResult:
    """检索结果"""
    document: VectorDocument
    score: float
    rank: int


# ============================================================
# 2. 嵌入器 (Embedder)
# ============================================================

class SimpleEmbedder:
    """
    简单嵌入器 - 使用哈希模拟嵌入向量
    用于演示和学习, 生产环境请使用真实嵌入模型
    """

    def __init__(self, dimension: int = 128):
        self.dimension = dimension

    def embed(self, text: str) -> np.ndarray:
        """将文本转换为嵌入向量 (哈希模拟)"""
        # 使用多段哈希生成向量
        vector = np.zeros(self.dimension)

        # 将文本分成多段, 每段生成哈希值作为向量分量
        words = text.split()
        for i, word in enumerate(words):
            hash_val = int(hashlib.md5(word.encode()).hexdigest(), 16)
            idx = hash_val % self.dimension
            vector[idx] += 1.0

        # 加入位置信息
        for i, char in enumerate(text[:self.dimension]):
            vector[i % self.dimension] += ord(char) * 0.01

        # 归一化
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """批量嵌入"""
        return np.array([self.embed(t) for t in texts])


class OpenAIEmbedder:
    """
    OpenAI嵌入器 - 使用真实API
    需要安装: pip install openai
    """

    def __init__(
        self, model: str = "text-embedding-3-small",
        dimension: int = 1536
    ):
        self.model = model
        self.dimension = dimension

    def embed(self, text: str) -> np.ndarray:
        """使用OpenAI API获取嵌入"""
        try:
            from openai import OpenAI
            client = OpenAI()

            response = client.embeddings.create(
                input=text,
                model=self.model,
            )
            return np.array(response.data[0].embedding)

        except ImportError:
            raise ImportError("请安装openai: pip install openai")
        except Exception as e:
            raise RuntimeError(f"OpenAI嵌入失败: {e}")

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """批量嵌入"""
        try:
            from openai import OpenAI
            client = OpenAI()

            response = client.embeddings.create(
                input=texts,
                model=self.model,
            )
            return np.array([d.embedding for d in response.data])

        except Exception as e:
            raise RuntimeError(f"批量嵌入失败: {e}")


# ============================================================
# 3. 向量存储
# ============================================================

class VectorStore:
    """基于NumPy的向量存储"""

    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        self.documents: Dict[str, VectorDocument] = {}
        self.embeddings: Optional[np.ndarray] = None
        self.doc_ids: List[str] = []  # ID有序列表, 与embeddings对齐
        self._embedder = SimpleEmbedder(dimension)

    def add(
        self, doc_id: str, content: str,
        metadata: Dict = None,
        embedding: Optional[np.ndarray] = None
    ) -> str:
        """添加文档"""
        if embedding is None:
            embedding = self._embedder.embed(content)

        doc = VectorDocument(
            id=doc_id,
            content=content,
            embedding=embedding,
            metadata=metadata or {},
        )

        self.documents[doc_id] = doc
        self.doc_ids.append(doc_id)
        self._rebuild_index()

        return doc_id

    def add_batch(
        self,
        documents: List[Dict],
    ) -> List[str]:
        """批量添加文档"""
        ids = []
        embeddings = []

        for doc_data in documents:
            doc_id = doc_data.get("id", f"doc_{len(self.documents)}")
            content = doc_data["content"]
            metadata = doc_data.get("metadata", {})

            embedding = self._embedder.embed(content)
            embeddings.append(embedding)

            doc = VectorDocument(
                id=doc_id,
                content=content,
                embedding=embedding,
                metadata=metadata,
            )
            self.documents[doc_id] = doc
            self.doc_ids.append(doc_id)
            ids.append(doc_id)

        self._rebuild_index()
        return ids

    def get(self, doc_id: str) -> Optional[VectorDocument]:
        """获取文档"""
        return self.documents.get(doc_id)

    def update(
        self, doc_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """更新文档"""
        if doc_id not in self.documents:
            return False

        doc = self.documents[doc_id]

        if content is not None:
            doc.content = content
            doc.embedding = self._embedder.embed(content)
            self._rebuild_index()

        if metadata is not None:
            doc.metadata.update(metadata)

        return True

    def delete(self, doc_id: str) -> bool:
        """删除文档"""
        if doc_id not in self.documents:
            return False

        del self.documents[doc_id]
        self.doc_ids.remove(doc_id)
        self._rebuild_index()

        return True

    def search(
        self, query: str, top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[SearchResult]:
        """相似度检索"""
        if not self.doc_ids:
            return []

        query_embedding = self._embedder.embed(query)

        # 计算余弦相似度
        similarities = self._cosine_similarity_batch(
            query_embedding, self.embeddings
        )

        # 排序
        sorted_indices = np.argsort(similarities)[::-1]

        results = []
        for rank, idx in enumerate(sorted_indices[:top_k], 1):
            score = similarities[idx]
            if score < threshold:
                continue

            doc_id = self.doc_ids[idx]
            doc = self.documents[doc_id]

            results.append(SearchResult(
                document=doc,
                score=float(score),
                rank=rank,
            ))

        return results

    def count(self) -> int:
        """文档数量"""
        return len(self.documents)

    def _rebuild_index(self):
        """重建嵌入矩阵索引"""
        if not self.doc_ids:
            self.embeddings = np.array([])
            return

        embeddings = []
        for doc_id in self.doc_ids:
            doc = self.documents[doc_id]
            if doc.embedding is not None:
                embeddings.append(doc.embedding)
            else:
                embeddings.append(np.zeros(self.dimension))

        self.embeddings = np.array(embeddings)

    @staticmethod
    def _cosine_similarity_batch(
        query: np.ndarray, embeddings: np.ndarray
    ) -> np.ndarray:
        """批量计算余弦相似度"""
        if embeddings.size == 0:
            return np.array([])

        # (query dot each_embedding) / (norm_query * norm_each)
        dot_products = embeddings @ query
        query_norm = np.linalg.norm(query)
        embedding_norms = np.linalg.norm(embeddings, axis=1)

        # 避免除零
        denom = query_norm * embedding_norms
        denom = np.where(denom == 0, 1e-8, denom)

        return dot_products / denom

    def get_stats(self) -> Dict:
        """获取存储统计"""
        total_chars = sum(
            len(d.content) for d in self.documents.values()
        )

        return {
            "document_count": self.count(),
            "total_characters": total_chars,
            "dimension": self.dimension,
            "index_size_mb": (
                self.embeddings.nbytes / (1024 * 1024)
                if self.embeddings is not None and self.embeddings.size > 0
                else 0
            ),
        }


# ============================================================
# 4. 演示
# ============================================================

def demo():
    """演示向量存储"""

    print("=" * 60)
    print("W29-D6: 向量存储实现")
    print("=" * 60)

    # 创建向量存储
    store = VectorStore(dimension=128)
    print(f"向量存储初始化完成, 维度: {store.dimension}")

    # ---- 添加文档 ----
    print("\n--- 添加文档 ---")
    documents = [
        {
            "id": "doc_1",
            "content": "Python是一种广泛使用的高级编程语言, 以简洁易读著称",
            "metadata": {"source": "wiki", "topic": "python"},
        },
        {
            "id": "doc_2",
            "content": "机器学习是人工智能的一个重要分支, 通过数据训练模型",
            "metadata": {"source": "wiki", "topic": "ml"},
        },
        {
            "id": "doc_3",
            "content": "深度学习使用多层神经网络处理复杂任务, 如图像识别",
            "metadata": {"source": "wiki", "topic": "dl"},
        },
        {
            "id": "doc_4",
            "content": "自然语言处理让计算机能够理解和生成人类语言",
            "metadata": {"source": "wiki", "topic": "nlp"},
        },
        {
            "id": "doc_5",
            "content": "RAG系统结合检索和生成, 提供准确的问答能力",
            "metadata": {"source": "wiki", "topic": "rag"},
        },
        {
            "id": "doc_6",
            "content": "向量数据库用于存储和检索高维向量, 支持相似度搜索",
            "metadata": {"source": "wiki", "topic": "vector_db"},
        },
        {
            "id": "doc_7",
            "content": "FastAPI是现代高性能的Python Web框架, 支持异步",
            "metadata": {"source": "wiki", "topic": "fastapi"},
        },
    ]

    ids = store.add_batch(documents)
    print(f"批量添加 {len(ids)} 个文档")

    # ---- 检索 ----
    print("\n--- 语义检索 ---")
    queries = [
        "什么是深度学习?",
        "Python编程语言",
        "如何实现问答系统?",
    ]

    for query in queries:
        print(f"\n查询: \"{query}\"")
        results = store.search(query, top_k=3)

        for r in results:
            preview = r.document.content[:30]
            print(f"  #{r.rank} [{r.score:.4f}] {preview}..."
                  f" (主题: {r.document.metadata.get('topic', 'N/A')})")

    # ---- CRUD操作 ----
    print("\n--- CRUD操作 ---")

    # 读取
    doc = store.get("doc_3")
    print(f"读取doc_3: {doc.content[:40]}...")

    # 更新
    store.update("doc_3", metadata={"updated": "true"})
    print("更新doc_3元数据")

    # 删除
    store.delete("doc_7")
    print("删除doc_7")

    # 统计
    stats = store.get_stats()
    print(f"\n存储统计:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # ---- 性能测试 ----
    print("\n--- 性能测试 ---")

    # 添加更多文档
    for i in range(100):
        store.add(
            f"perf_{i}",
            f"这是第{i}条测试文档, 用于性能测试, 编号{i}",
        )

    print(f"添加后文档总数: {store.count()}")

    # 检索延迟
    start = time.time()
    for _ in range(100):
        store.search("测试查询", top_k=5)
    elapsed = (time.time() - start) * 1000

    print(f"100次检索平均延迟: {elapsed / 100:.2f}ms")


# ============================================================
# 运行演示
# ============================================================

if __name__ == "__main__":
    demo()
