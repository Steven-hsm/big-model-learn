"""
W29-D3: 项目骨架代码生成器
==========================
功能:
  - 创建项目目录结构
  - FastAPI应用模板
  - 配置管理
  - requirements.txt生成
  - 生成项目文件到指定目录
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional


# ============================================================
# 1. 项目骨架模板
# ============================================================

# 项目目录结构模板
PROJECT_STRUCTURE = {
    "app": {
        "__init__.py": "",
        "main.py": "app_main_py",
        "config.py": "app_config_py",
        "api": {
            "__init__.py": "",
            "routes.py": "api_routes_py",
            "schemas.py": "api_schemas_py",
        },
        "core": {
            "__init__.py": "",
            "rag.py": "core_rag_py",
            "retrieval.py": "core_retrieval_py",
        },
        "models": {
            "__init__.py": "",
            "document.py": "models_document_py",
        },
        "services": {
            "__init__.py": "",
            "document_service.py": "services_document_py",
            "chat_service.py": "services_chat_py",
        },
    },
    "data": {
        "documents": {},
        "embeddings": {},
    },
    "tests": {
        "__init__.py": "",
        "test_api.py": "test_api_py",
        "test_rag.py": "test_rag_py",
    },
    "scripts": {
        "ingest.py": "scripts_ingest_py",
    },
    ".env.example": "env_example",
    "requirements.txt": "requirements_txt",
    "Dockerfile": "dockerfile",
    "README.md": "readme_md",
}


# ============================================================
# 2. 文件内容模板
# ============================================================

TEMPLATES: Dict[str, str] = {
    # ---- app/main.py ----
    "app_main_py": '''"""
RAG智能问答系统 - FastAPI主应用
"""
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import settings

# 创建FastAPI应用
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="基于RAG的智能文档问答系统",
    version="1.0.0",
)

# CORS中间件配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    """根路径 - 健康检查"""
    return {
        "status": "ok",
        "service": settings.PROJECT_NAME,
        "version": "1.0.0",
    }


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
''',

    # ---- app/config.py ----
    "app_config_py": '''"""
配置管理 - 使用pydantic-settings
"""
from typing import List
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


class Settings(BaseSettings):
    """应用配置"""
    # 项目基本信息
    PROJECT_NAME: str = "RAG智能问答系统"
    VERSION: str = "1.0.0"
    DEBUG: bool = True

    # 服务器配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # CORS
    ALLOWED_ORIGINS: List[str] = ["*"]

    # LLM配置
    LLM_PROVIDER: str = "openai"  # openai / local
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4"
    OPENAI_BASE_URL: str = ""

    # 嵌入模型配置
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSION: int = 1536

    # 向量存储配置
    VECTOR_STORE: str = "chroma"  # chroma / faiss
    VECTOR_STORE_PATH: str = "./data/embeddings"

    # 文档配置
    UPLOAD_DIR: str = "./data/documents"
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS: List[str] = [".pdf", ".docx", ".txt", ".md"]

    # 切分配置
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # 检索配置
    TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.7

    # Redis (可选)
    REDIS_URL: str = "redis://localhost:6379"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
''',

    # ---- app/api/routes.py ----
    "api_routes_py": '''"""
API路由定义
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List

from app.api.schemas import (
    QueryRequest, QueryResponse,
    DocumentInfo, HealthResponse,
)

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """问答接口"""
    # TODO: 实现RAG查询逻辑
    return QueryResponse(
        answer="这是示例回答",
        sources=[],
        conversation_id=request.conversation_id or "demo-001",
    )


@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """文档上传接口"""
    # TODO: 实现文档上传逻辑
    return {"filename": file.filename, "status": "uploaded"}


@router.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    """获取文档列表"""
    # TODO: 实现文档列表逻辑
    return []


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """删除文档"""
    # TODO: 实现删除逻辑
    return {"status": "deleted", "doc_id": doc_id}


@router.get("/health", response_model=HealthResponse)
async def health():
    """健康检查"""
    return HealthResponse(status="healthy")
''',

    # ---- app/api/schemas.py ----
    "api_schemas_py": '''"""
Pydantic数据模型
"""
from pydantic import BaseModel, Field
from typing import List, Optional


class QueryRequest(BaseModel):
    """查询请求"""
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    conversation_id: Optional[str] = Field(None, description="会话ID")
    top_k: int = Field(5, ge=1, le=20, description="检索文档数")


class SourceInfo(BaseModel):
    """来源信息"""
    document: str
    page: Optional[int] = None
    content: str
    score: float


class QueryResponse(BaseModel):
    """查询响应"""
    answer: str
    sources: List[SourceInfo] = []
    conversation_id: str


class DocumentInfo(BaseModel):
    """文档信息"""
    id: str
    filename: str
    size: int
    upload_time: str
    status: str
    chunk_count: int = 0


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
''',

    # ---- app/core/rag.py ----
    "core_rag_py": '''"""
RAG核心Pipeline
"""
from typing import List, Optional


class RAGPipeline:
    """RAG问答Pipeline"""

    def __init__(self):
        self.documents = []
        self.embeddings = []

    def ingest(self, file_path: str) -> int:
        """导入文档, 返回chunk数量"""
        # TODO: 实现文档导入
        print(f"导入文档: {file_path}")
        return 0

    def query(self, question: str, top_k: int = 5) -> dict:
        """查询"""
        # TODO: 实现查询逻辑
        return {
            "answer": "待实现",
            "sources": [],
        }
''',

    # ---- app/core/retrieval.py ----
    "core_retrieval_py": '''"""
检索模块
"""
from typing import List


class VectorRetriever:
    """向量检索器"""

    def __init__(self):
        pass

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        """语义检索"""
        # TODO: 实现检索逻辑
        return []

    def add_documents(self, documents: List[dict]):
        """添加文档到索引"""
        # TODO: 实现索引构建
        pass
''',

    # ---- app/models/document.py ----
    "models_document_py": '''"""
文档数据模型
"""
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime


@dataclass
class Document:
    """文档模型"""
    id: str
    filename: str
    content: str
    file_type: str
    size: int
    upload_time: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)
    chunks: List["Chunk"] = field(default_factory=list)


@dataclass
class Chunk:
    """文本块模型"""
    id: str
    document_id: str
    content: str
    index: int
    metadata: dict = field(default_factory=dict)
    embedding: Optional[List[float]] = None
''',

    # ---- app/services/document_service.py ----
    "services_document_py": '''"""
文档服务
"""
from typing import List


class DocumentService:
    """文档管理服务"""

    def upload(self, file_path: str) -> str:
        """上传文档"""
        # TODO: 实现
        return "doc_id"

    def delete(self, doc_id: str) -> bool:
        """删除文档"""
        # TODO: 实现
        return True

    def list_all(self) -> List[dict]:
        """列出所有文档"""
        # TODO: 实现
        return []
''',

    # ---- app/services/chat_service.py ----
    "services_chat_py": '''"""
对话服务
"""
from typing import List, Optional


class ChatService:
    """对话管理服务"""

    def __init__(self):
        self.sessions = {}

    def create_session(self) -> str:
        """创建新会话"""
        import uuid
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = []
        return session_id

    def add_message(self, session_id: str, role: str, content: str):
        """添加消息"""
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append({
            "role": role,
            "content": content,
        })

    def get_history(self, session_id: str) -> List[dict]:
        """获取对话历史"""
        return self.sessions.get(session_id, [])
''',

    # ---- tests/test_api.py ----
    "test_api_py": '''"""
API测试
"""
import pytest

# TODO: 使用 httpx 或 TestClient 实现API测试
# from fastapi.testclient import TestClient
# from app.main import app
# client = TestClient(app)


def test_health_check():
    """测试健康检查接口"""
    # response = client.get("/health")
    # assert response.status_code == 200
    # assert response.json()["status"] == "healthy"
    assert True  # 占位
''',

    # ---- tests/test_rag.py ----
    "test_rag_py": '''"""
RAG Pipeline测试
"""
import pytest


def test_placeholder():
    """占位测试"""
    # TODO: 实现RAG Pipeline测试
    assert True
''',

    # ---- scripts/ingest.py ----
    "scripts_ingest_py": '''"""
文档导入脚本
用法: python -m scripts.ingest --dir ./data/documents
"""
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="文档导入工具")
    parser.add_argument("--dir", required=True, help="文档目录")
    parser.add_argument("--reindex", action="store_true", help="重新索引")
    args = parser.parse_args()

    if not os.path.exists(args.dir):
        print(f"目录不存在: {args.dir}")
        return

    # TODO: 实现导入逻辑
    print(f"扫描目录: {args.dir}")
    print("导入功能待实现")


if __name__ == "__main__":
    main()
''',

    # ---- .env.example ----
    "env_example": '''# 环境变量配置
# 复制此文件为 .env 并填入实际值

# LLM配置
LLM_PROVIDER=openai
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-4
OPENAI_BASE_URL=

# 嵌入模型
EMBEDDING_MODEL=text-embedding-3-small

# 向量存储
VECTOR_STORE=chroma
VECTOR_STORE_PATH=./data/embeddings

# 服务器
HOST=0.0.0.0
PORT=8000
DEBUG=True
''',

    # ---- requirements.txt ----
    "requirements_txt": '''# Web框架
fastapi>=0.104.0
uvicorn>=0.24.0
python-multipart>=0.0.6

# 数据验证
pydantic>=2.0.0
pydantic-settings>=2.0.0

# 文档处理
pypdf2>=3.0.0
python-docx>=0.8.11
markdown>=3.5.0

# 向量与检索
numpy>=1.24.0
chromadb>=0.4.0
faiss-cpu>=1.7.4

# LLM
openai>=1.0.0
langchain>=0.1.0

# 工具
python-dotenv>=1.0.0
httpx>=0.25.0

# 可选: Redis
# redis>=5.0.0

# 开发
pytest>=7.4.0
pytest-asyncio>=0.21.0
''',

    # ---- Dockerfile ----
    "dockerfile": '''FROM python:3.11-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
''',

    # ---- README.md ----
    "readme_md": '''# RAG智能问答系统

基于RAG (检索增强生成) 的智能文档问答系统。

## 快速开始

1. 安装依赖:
```bash
pip install -r requirements.txt
```

2. 配置环境变量:
```bash
cp .env.example .env
# 编辑 .env 填入API Key
```

3. 启动服务:
```bash
python -m app.main
```

4. 访问API文档:
http://localhost:8000/docs

## 项目结构

app/           - 应用代码
  api/         - API路由
  core/        - 核心逻辑
  models/      - 数据模型
  services/    - 业务服务
data/          - 数据目录
tests/         - 测试
scripts/       - 工具脚本
''',
}


# ============================================================
# 3. 项目骨架生成器
# ============================================================

class ProjectSkeletonGenerator:
    """项目骨架生成器"""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.created_files: List[str] = []
        self.created_dirs: List[str] = []

    def generate(self, structure: dict = None, parent: Path = None):
        """递归生成项目结构"""
        if structure is None:
            structure = PROJECT_STRUCTURE
        if parent is None:
            parent = self.base_path

        for name, content in structure.items():
            path = parent / name

            if isinstance(content, dict):
                # 目录
                path.mkdir(parents=True, exist_ok=True)
                self.created_dirs.append(str(path))
                if content:  # 非空目录
                    self.generate(content, path)
            else:
                # 文件
                template_key = content if content else None
                file_content = ""

                if template_key and template_key in TEMPLATES:
                    file_content = TEMPLATES[template_key]

                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(file_content, encoding="utf-8")
                self.created_files.append(str(path))

    def print_summary(self):
        """打印生成摘要"""
        print("\n" + "=" * 60)
        print("项目骨架生成完成")
        print("=" * 60)
        print(f"\n创建目录 {len(self.created_dirs)} 个:")
        for d in self.created_dirs[:10]:
            print(f"  {d}")
        if len(self.created_dirs) > 10:
            print(f"  ... 共 {len(self.created_dirs)} 个")

        print(f"\n创建文件 {len(self.created_files)} 个:")
        for f in self.created_files:
            print(f"  {f}")


# ============================================================
# 4. 演示: 打印项目结构和关键文件
# ============================================================

def demo():
    """演示项目骨架"""

    print("=" * 60)
    print("W29-D3: 项目骨架代码生成器")
    print("=" * 60)

    # 打印目录结构树
    print("\n项目目录结构:")
    print("-" * 40)

    tree = """rag-qa-system/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI主应用
│   ├── config.py            # 配置管理
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py        # API路由
│   │   └── schemas.py       # 数据模型
│   ├── core/
│   │   ├── __init__.py
│   │   ├── rag.py           # RAG Pipeline
│   │   └── retrieval.py     # 检索模块
│   ├── models/
│   │   ├── __init__.py
│   │   └── document.py      # 文档模型
│   └── services/
│       ├── __init__.py
│       ├── document_service.py  # 文档服务
│       └── chat_service.py      # 对话服务
├── data/
│   ├── documents/           # 上传文档存储
│   └── embeddings/          # 向量数据存储
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   └── test_rag.py
├── scripts/
│   └── ingest.py            # 文档导入脚本
├── .env.example
├── requirements.txt
├── Dockerfile
└── README.md"""
    print(tree)

    # 打印关键配置说明
    print("\n关键配置说明:")
    print("-" * 40)

    configs = [
        ("PROJECT_NAME", "项目名称"),
        ("LLM_PROVIDER", "LLM提供商 (openai/local)"),
        ("OPENAI_API_KEY", "OpenAI API密钥"),
        ("VECTOR_STORE", "向量存储 (chroma/faiss)"),
        ("CHUNK_SIZE", "文本切分大小 (默认500)"),
        ("TOP_K", "检索返回文档数 (默认5)"),
    ]

    for key, desc in configs:
        print(f"  {key:<25} {desc}")

    # 生成骨架 (到临时目录展示)
    print("\n" + "-" * 40)
    print("骨架生成示例 (生成到当前目录的_demo_project):")

    demo_path = Path.cwd() / "_demo_rag_project"
    if demo_path.exists():
        import shutil
        shutil.rmtree(demo_path)

    generator = ProjectSkeletonGenerator(str(demo_path))
    generator.generate()
    generator.print_summary()

    # 清理
    if demo_path.exists():
        import shutil
        shutil.rmtree(demo_path)
        print("\n(演示目录已清理)")


# ============================================================
# 运行演示
# ============================================================

if __name__ == "__main__":
    demo()
