"""
W30-D6 REST API设计
====================
实现RAG系统的REST API, 包括:
- 查询接口
- 文档管理接口
- 健康检查
- Pydantic数据验证

使用FastAPI风格设计, 可直接运行。
"""

import time
import json
import hashlib
import uuid
from typing import List, Optional, Dict
from dataclasses import dataclass, field, asdict

try:
    from pydantic import BaseModel, Field, validator
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    # 提供基础替代
    class BaseModel:
        """简化BaseModel替代"""
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
        def dict(self):
            return self.__dict__
        def json(self):
            return json.dumps(self.__dict__, ensure_ascii=False)

    def Field(default=None, **kwargs):
        return default

    def validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


# ============================================================
# 1. 数据模型 (Pydantic)
# ============================================================

if HAS_PYDANTIC:
    class QueryRequest(BaseModel):
        """查询请求"""
        query: str = Field(..., min_length=1, max_length=1000, description="用户查询")
        top_k: int = Field(default=5, ge=1, le=20, description="返回文档数")
        session_id: Optional[str] = Field(default=None, description="会话ID")
        temperature: float = Field(default=0.7, ge=0, le=2, description="生成温度")
        stream: bool = Field(default=False, description="是否流式输出")

        @validator('query')
        def query_not_blank(cls, v):
            if not v.strip():
                raise ValueError('查询不能为空')
            return v.strip()

    class DocumentUpload(BaseModel):
        """文档上传请求"""
        title: str = Field(..., min_length=1, max_length=200, description="文档标题")
        content: str = Field(..., min_length=1, description="文档内容")
        source: str = Field(default="manual", description="文档来源")
        tags: List[str] = Field(default_factory=list, description="标签列表")
        metadata: Dict = Field(default_factory=dict, description="额外元数据")

    class QueryResponse(BaseModel):
        """查询响应"""
        query: str
        answer: str
        sources: List[Dict] = Field(default_factory=list)
        session_id: str
        latency_ms: float
        metadata: Dict = Field(default_factory=dict)

    class DocumentResponse(BaseModel):
        """文档响应"""
        doc_id: str
        title: str
        source: str
        tags: List[str]
        chunk_count: int
        created_at: float

    class HealthResponse(BaseModel):
        """健康检查响应"""
        status: str
        version: str
        uptime_seconds: float
        components: Dict
else:
    # 无Pydantic时的简化模型
    class QueryRequest(BaseModel):
        def __init__(self, query="", top_k=5, session_id=None, temperature=0.7, stream=False):
            super().__init__(query=query, top_k=top_k, session_id=session_id,
                           temperature=temperature, stream=stream)

    class DocumentUpload(BaseModel):
        def __init__(self, title="", content="", source="manual", tags=None, metadata=None):
            super().__init__(title=title, content=content, source=source,
                           tags=tags or [], metadata=metadata or {})

    class QueryResponse(BaseModel):
        def __init__(self, query="", answer="", sources=None, session_id="",
                     latency_ms=0, metadata=None):
            super().__init__(query=query, answer=answer, sources=sources or [],
                           session_id=session_id, latency_ms=latency_ms,
                           metadata=metadata or {})

    class DocumentResponse(BaseModel):
        def __init__(self, doc_id="", title="", source="", tags=None, chunk_count=0, created_at=0):
            super().__init__(doc_id=doc_id, title=title, source=source,
                           tags=tags or [], chunk_count=chunk_count, created_at=created_at)

    class HealthResponse(BaseModel):
        def __init__(self, status="", version="", uptime_seconds=0, components=None):
            super().__init__(status=status, version=version,
                           uptime_seconds=uptime_seconds, components=components or {})


# ============================================================
# 2. 内存文档存储
# ============================================================

class DocumentStore:
    """内存文档存储(演示用)"""

    def __init__(self):
        self.documents: Dict[str, Dict] = {}
        self.start_time = time.time()

    def add_document(self, doc: DocumentUpload) -> str:
        """添加文档"""
        doc_id = hashlib.md5(
            f"{doc.title}_{time.time()}".encode()
        ).hexdigest()[:12]

        # 简单分块(按句子)
        chunks = self._split_chunks(doc.content, chunk_size=200)

        self.documents[doc_id] = {
            'doc_id': doc_id,
            'title': doc.title,
            'content': doc.content,
            'source': doc.source,
            'tags': doc.tags,
            'metadata': doc.metadata,
            'chunks': chunks,
            'created_at': time.time(),
        }
        return doc_id

    def _split_chunks(self, text: str, chunk_size: int = 200) -> List[str]:
        """将文本分块"""
        sentences = text.replace('。', '。\n').replace('！', '！\n').replace('？', '？\n').split('\n')
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = ""
        for sent in sentences:
            if len(current_chunk) + len(sent) > chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = sent
            else:
                current_chunk += sent
        if current_chunk:
            chunks.append(current_chunk)

        return chunks if chunks else [text]

    def get_document(self, doc_id: str) -> Optional[Dict]:
        return self.documents.get(doc_id)

    def list_documents(self, tag: str = None) -> List[Dict]:
        docs = list(self.documents.values())
        if tag:
            docs = [d for d in docs if tag in d['tags']]
        return docs

    def delete_document(self, doc_id: str) -> bool:
        if doc_id in self.documents:
            del self.documents[doc_id]
            return True
        return False

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """简单关键词搜索"""
        query_words = set(query.lower().split())
        results = []

        for doc_id, doc in self.documents.items():
            score = 0
            for chunk in doc['chunks']:
                chunk_words = set(chunk.lower().split())
                overlap = query_words & chunk_words
                score = max(score, len(overlap) / max(len(query_words), 1))

            if score > 0:
                results.append({
                    'doc_id': doc_id,
                    'title': doc['title'],
                    'score': score,
                    'relevant_chunk': doc['chunks'][0][:100] if doc['chunks'] else "",
                })

        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

    def get_stats(self) -> Dict:
        return {
            'total_documents': len(self.documents),
            'total_chunks': sum(len(d['chunks']) for d in self.documents.values()),
            'total_tags': len(set(tag for d in self.documents.values() for tag in d['tags'])),
        }


# ============================================================
# 3. API路由处理器
# ============================================================

class APIRouter:
    """API路由处理器(不依赖FastAPI的纯Python实现)"""

    def __init__(self):
        self.store = DocumentStore()
        self.start_time = time.time()
        self.request_count = 0
        self._init_sample_data()

    def _init_sample_data(self):
        """初始化示例数据"""
        samples = [
            DocumentUpload(title="RAG技术介绍", content="RAG(检索增强生成)是一种结合信息检索和文本生成的技术。"
                         "它通过检索相关文档来增强LLM的回答质量, 减少幻觉。", source="tech_doc", tags=["RAG", "LLM"]),
            DocumentUpload(title="Python入门", content="Python是一种广泛使用的高级编程语言。"
                         "它支持多种编程范式, 包括面向对象和函数式编程。", source="tutorial", tags=["Python", "编程"]),
        ]
        for doc in samples:
            self.store.add_document(doc)

    def handle_query(self, request: QueryRequest) -> Dict:
        """处理查询请求"""
        self.request_count += 1
        start = time.time()

        # 检索
        results = self.store.search(request.query, top_k=request.top_k)

        # 模拟生成回答
        answer = self._generate_answer(request.query, results)

        latency = (time.time() - start) * 1000

        response = QueryResponse(
            query=request.query,
            answer=answer,
            sources=results,
            session_id=request.session_id or str(uuid.uuid4())[:8],
            latency_ms=latency,
            metadata={'request_count': self.request_count}
        )
        return response.dict()

    def _generate_answer(self, query: str, results: List[Dict]) -> str:
        """模拟LLM生成回答"""
        if results:
            sources = ", ".join(r['title'] for r in results[:3])
            return f"根据{len(results)}篇相关文档, 关于「{query}」的回答: {results[0]['relevant_chunk']}..."
        return f"未找到与「{query}」相关的文档, 请尝试换一种表述方式。"

    def handle_upload(self, doc: DocumentUpload) -> Dict:
        """处理文档上传"""
        doc_id = self.store.add_document(doc)
        return {
            'status': 'success',
            'doc_id': doc_id,
            'message': f'文档「{doc.title}」已上传并索引'
        }

    def handle_list_docs(self, tag: str = None) -> Dict:
        """列出文档"""
        docs = self.store.list_documents(tag=tag)
        return {
            'total': len(docs),
            'documents': [
                DocumentResponse(
                    doc_id=d['doc_id'],
                    title=d['title'],
                    source=d['source'],
                    tags=d['tags'],
                    chunk_count=len(d['chunks']),
                    created_at=d['created_at']
                ).dict()
                for d in docs
            ]
        }

    def handle_delete(self, doc_id: str) -> Dict:
        """删除文档"""
        success = self.store.delete_document(doc_id)
        if success:
            return {'status': 'success', 'message': f'文档 {doc_id} 已删除'}
        return {'status': 'error', 'message': f'文档 {doc_id} 不存在'}

    def handle_health(self) -> Dict:
        """健康检查"""
        uptime = time.time() - self.start_time
        return HealthResponse(
            status='healthy',
            version='1.0.0',
            uptime_seconds=uptime,
            components={
                'document_store': 'ok',
                'retriever': 'ok',
                'generator': 'ok',
            }
        ).dict()

    def handle_stats(self) -> Dict:
        """获取统计信息"""
        return {
            **self.store.get_stats(),
            'uptime_seconds': time.time() - self.start_time,
            'request_count': self.request_count,
        }


# ============================================================
# 4. FastAPI应用(如果安装了FastAPI)
# ============================================================

def create_fastapi_app():
    """创建FastAPI应用(需要安装fastapi和uvicorn)"""
    if not HAS_FASTAPI:
        print("FastAPI未安装, 跳过。请运行: pip install fastapi uvicorn")
        return None

    app = FastAPI(title="RAG API", version="1.0.0", description="RAG系统REST API")
    router = APIRouter()

    @app.post("/api/query", response_model=dict, summary="查询接口")
    async def query(request: QueryRequest):
        return router.handle_query(request)

    @app.post("/api/documents", response_model=dict, summary="上传文档")
    async def upload(doc: DocumentUpload):
        return router.handle_upload(doc)

    @app.get("/api/documents", response_model=dict, summary="列出文档")
    async def list_docs(tag: Optional[str] = None):
        return router.handle_list_docs(tag)

    @app.delete("/api/documents/{doc_id}", response_model=dict, summary="删除文档")
    async def delete_doc(doc_id: str):
        result = router.handle_delete(doc_id)
        if result['status'] == 'error':
            raise HTTPException(status_code=404, detail=result['message'])
        return result

    @app.get("/api/health", response_model=dict, summary="健康检查")
    async def health():
        return router.handle_health()

    @app.get("/api/stats", response_model=dict, summary="统计信息")
    async def stats():
        return router.handle_stats()

    return app


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("W30-D6 REST API设计")
    print("=" * 60)

    # 直接使用路由处理器演示
    router = APIRouter()

    # --- 查询接口 ---
    print("\n--- 1. 查询接口 ---")
    query_req = QueryRequest(query="什么是RAG?", top_k=3)
    result = router.handle_query(query_req)
    print(f"查询: {result['query']}")
    print(f"回答: {result['answer']}")
    print(f"来源数: {len(result['sources'])}")
    print(f"延迟: {result['latency_ms']:.1f}ms")

    # --- 文档管理 ---
    print(f"\n{'='*60}")
    print("--- 2. 文档管理 ---")
    print(f"{'='*60}")

    # 上传
    new_doc = DocumentUpload(
        title="深度学习入门",
        content="深度学习使用多层神经网络来学习数据的复杂表示。"
                "它在图像识别、语音识别和自然语言处理等领域取得了突破性进展。",
        source="tutorial",
        tags=["深度学习", "AI", "神经网络"]
    )
    upload_result = router.handle_upload(new_doc)
    print(f"上传结果: {upload_result}")

    # 列表
    docs = router.handle_list_docs()
    print(f"\n文档列表 (共{docs['total']}篇):")
    for doc in docs['documents']:
        print(f"  - {doc['doc_id']}: {doc['title']} ({doc['chunk_count']}块, 标签: {doc['tags']})")

    # 按标签过滤
    ai_docs = router.handle_list_docs(tag="AI")
    print(f"\nAI标签文档: {ai_docs['total']}篇")

    # --- 健康检查 ---
    print(f"\n{'='*60}")
    print("--- 3. 健康检查 ---")
    print(f"{'='*60}")
    health = router.handle_health()
    print(f"状态: {health['status']}")
    print(f"版本: {health['version']}")
    print(f"运行时间: {health['uptime_seconds']:.1f}秒")
    print(f"组件状态: {health['components']}")

    # --- 统计 ---
    print(f"\n--- 4. 系统统计 ---")
    stats = router.handle_stats()
    print(f"文档总数: {stats['total_documents']}")
    print(f"分块总数: {stats['total_chunks']}")
    print(f"请求总数: {stats['request_count']}")

    # --- FastAPI启动提示 ---
    print(f"\n{'='*60}")
    print("--- 5. FastAPI集成 ---")
    print(f"{'='*60}")
    if HAS_FASTAPI:
        print("FastAPI已安装! 可以通过以下方式启动:")
        print("  app = create_fastapi_app()")
        print("  uvicorn.run(app, host='0.0.0.0', port=8000)")
    else:
        print("FastAPI未安装。安装方法:")
        print("  pip install fastapi uvicorn pydantic")
        print("\nAPI端点设计:")
        endpoints = [
            ("POST", "/api/query", "查询接口"),
            ("POST", "/api/documents", "上传文档"),
            ("GET",  "/api/documents", "列出文档"),
            ("DELETE", "/api/documents/{id}", "删除文档"),
            ("GET",  "/api/health", "健康检查"),
            ("GET",  "/api/stats", "统计信息"),
        ]
        for method, path, desc in endpoints:
            print(f"  {method:6s} {path:30s} {desc}")

    print("\n完成!")
