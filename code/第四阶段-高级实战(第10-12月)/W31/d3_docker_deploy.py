"""
W31-D3 Docker配置生成
=====================
生成RAG系统的Docker部署配置, 包括:
- Dockerfile生成
- docker-compose配置
- 环境变量管理
- 多阶段构建优化

Docker让应用部署变得标准化和可复现。
"""

import os
import json
from typing import Dict, List


# ============================================================
# 1. Dockerfile生成器
# ============================================================

class DockerfileGenerator:
    """Dockerfile生成器"""

    def __init__(self):
        self.base_image = "python:3.11-slim"
        self.workdir = "/app"
        self.requirements = []
        self.env_vars = {}
        self.expose_ports = [8000]
        self.copy_files = []
        self.commands = []
        self.entrypoint = "python"
        self.cmd_args = ["main.py"]

    def set_base(self, image: str):
        self.base_image = image
        return self

    def add_requirement(self, *packages: str):
        self.requirements.extend(packages)
        return self

    def set_env(self, key: str, value: str):
        self.env_vars[key] = value
        return self

    def expose(self, *ports: int):
        self.expose_ports.extend(ports)
        return self

    def generate(self) -> str:
        """生成Dockerfile内容"""
        lines = []

        # 基础镜像
        lines.append(f"FROM {self.base_image}")
        lines.append("")

        # 元数据
        lines.append("LABEL maintainer=\"your-email@example.com\"")
        lines.append("LABEL description=\"RAG知识库问答系统\"")
        lines.append("")

        # 工作目录
        lines.append(f"WORKDIR {self.workdir}")
        lines.append("")

        # 环境变量
        if self.env_vars:
            for key, value in self.env_vars.items():
                lines.append(f"ENV {key}={value}")
            lines.append("")

        # 依赖安装
        lines.append("# 安装系统依赖")
        lines.append("RUN apt-get update && apt-get install -y --no-install-recommends \\")
        lines.append("    build-essential \\")
        lines.append("    && rm -rf /var/lib/apt/lists/*")
        lines.append("")

        # 复制依赖文件
        lines.append("# 安装Python依赖")
        lines.append("COPY requirements.txt .")
        lines.append("RUN pip install --no-cache-dir -r requirements.txt")
        lines.append("")

        # 复制应用代码
        lines.append("# 复制应用代码")
        lines.append("COPY . .")
        lines.append("")

        # 暴露端口
        if self.expose_ports:
            ports_str = " ".join(str(p) for p in set(self.expose_ports))
            lines.append(f"EXPOSE {ports_str}")
            lines.append("")

        # 健康检查
        lines.append("# 健康检查")
        lines.append("HEALTHCHECK --interval=30s --timeout=10s --retries=3 \\")
        lines.append("    CMD python -c \"import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')\" || exit 1")
        lines.append("")

        # 启动命令
        lines.append("# 启动应用")
        lines.append(f'ENTRYPOINT ["{self.entrypoint}"]')
        lines.append(f'CMD {json.dumps(self.cmd_args)}')

        return "\n".join(lines)


class MultiStageDockerfileGenerator(DockerfileGenerator):
    """多阶段构建Dockerfile生成器

    多阶段构建的优势:
    1. 最终镜像更小(不包含编译工具)
    2. 构建缓存更高效
    3. 安全性更高(不暴露源码)
    """

    def generate(self) -> str:
        lines = []

        # === 阶段1: 构建阶段 ===
        lines.append("# === 阶段1: 构建 ===")
        lines.append(f"FROM {self.base_image} AS builder")
        lines.append("")
        lines.append(f"WORKDIR {self.workdir}")
        lines.append("")
        lines.append("COPY requirements.txt .")
        lines.append("RUN pip install --no-cache-dir --user -r requirements.txt")
        lines.append("")

        # === 阶段2: 运行阶段 ===
        lines.append("# === 阶段2: 运行 ===")
        lines.append(f"FROM {self.base_image} AS runtime")
        lines.append("")

        # 仅复制安装好的包
        lines.append("COPY --from=builder /root/.local /root/.local")
        lines.append("ENV PATH=/root/.local/bin:$PATH")
        lines.append("")

        # 工作目录和代码
        lines.append(f"WORKDIR {self.workdir}")
        lines.append("COPY . .")
        lines.append("")

        # 环境变量
        if self.env_vars:
            for key, value in self.env_vars.items():
                lines.append(f"ENV {key}={value}")
            lines.append("")

        # 非root用户
        lines.append("# 创建非root用户")
        lines.append("RUN useradd -m appuser")
        lines.append("USER appuser")
        lines.append("")

        if self.expose_ports:
            ports_str = " ".join(str(p) for p in set(self.expose_ports))
            lines.append(f"EXPOSE {ports_str}")
            lines.append("")

        lines.append("HEALTHCHECK --interval=30s --timeout=10s --retries=3 \\")
        lines.append("    CMD python -c \"import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')\" || exit 1")
        lines.append("")

        lines.append(f'ENTRYPOINT ["{self.entrypoint}"]')
        lines.append(f'CMD {json.dumps(self.cmd_args)}')

        return "\n".join(lines)


# ============================================================
# 2. docker-compose生成器
# ============================================================

class DockerComposeGenerator:
    """docker-compose.yml生成器"""

    def __init__(self):
        self.services = {}
        self.volumes = {}
        self.networks = {}

    def add_service(self, name: str, config: Dict):
        """添加服务"""
        self.services[name] = config
        return self

    def add_volume(self, name: str):
        """添加数据卷"""
        self.volumes[name] = {}
        return self

    def add_network(self, name: str):
        """添加网络"""
        self.networks[name] = {}
        return self

    def generate(self) -> str:
        """生成docker-compose.yml内容"""
        lines = ["version: '3.8'", ""]

        # 服务
        lines.append("services:")
        for name, config in self.services.items():
            lines.append(f"  {name}:")
            for key, value in config.items():
                if isinstance(value, list):
                    lines.append(f"    {key}:")
                    for item in value:
                        lines.append(f"      - {item}")
                elif isinstance(value, dict):
                    lines.append(f"    {key}:")
                    for k, v in value.items():
                        lines.append(f"      {k}: {v}")
                else:
                    lines.append(f"    {key}: {value}")
            lines.append("")

        # 数据卷
        if self.volumes:
            lines.append("volumes:")
            for name in self.volumes:
                lines.append(f"  {name}:")
            lines.append("")

        # 网络
        if self.networks:
            lines.append("networks:")
            for name in self.networks:
                lines.append(f"  {name}:")
                lines.append("    driver: bridge")

        return "\n".join(lines)


def create_rag_compose() -> str:
    """创建RAG系统的docker-compose配置"""
    gen = DockerComposeGenerator()

    # API服务
    gen.add_service("rag-api", {
        "build": ".",
        "container_name": "rag-api",
        "ports": ["8000:8000"],
        "environment": [
            "OPENAI_API_KEY=${OPENAI_API_KEY}",
            "DATABASE_URL=postgresql://postgres:password@db:5432/rag",
            "REDIS_URL=redis://redis:6379/0",
        ],
        "depends_on": ["db", "redis"],
        "volumes": ["./data:/app/data"],
        "restart": "unless-stopped",
    })

    # PostgreSQL
    gen.add_service("db", {
        "image": "postgres:15-alpine",
        "container_name": "rag-db",
        "environment": [
            "POSTGRES_DB=rag",
            "POSTGRES_PASSWORD=password",
        ],
        "volumes": ["pgdata:/var/lib/postgresql/data"],
        "ports": ["5432:5432"],
    })

    # Redis
    gen.add_service("redis", {
        "image": "redis:7-alpine",
        "container_name": "rag-redis",
        "ports": ["6379:6379"],
    })

    gen.add_volume("pgdata")
    gen.add_network("rag-network")

    return gen.generate()


# ============================================================
# 3. 环境变量管理
# ============================================================

class EnvManager:
    """环境变量管理器"""

    def __init__(self):
        self.vars = {}

    def add(self, key: str, value: str = "", description: str = "",
            required: bool = False, default: str = ""):
        self.vars[key] = {
            'value': value,
            'description': description,
            'required': required,
            'default': default,
        }
        return self

    def generate_env_file(self) -> str:
        """生成.env文件"""
        lines = ["# RAG系统环境变量配置", ""]
        for key, info in self.vars.items():
            lines.append(f"# {info['description']}")
            val = info['value'] or info['default'] or ""
            lines.append(f"{key}={val}")
            lines.append("")
        return "\n".join(lines)

    def generate_env_example(self) -> str:
        """生成.env.example文件"""
        lines = ["# 复制此文件为.env并填写实际值", ""]
        for key, info in self.vars.items():
            req = " [必填]" if info['required'] else ""
            lines.append(f"# {info['description']}{req}")
            lines.append(f"{key}=")
            lines.append("")
        return "\n".join(lines)

    def validate(self, env_dict: Dict) -> List[str]:
        """验证环境变量"""
        missing = []
        for key, info in self.vars.items():
            if info['required'] and not env_dict.get(key):
                missing.append(key)
        return missing


def create_rag_env_config() -> EnvManager:
    """创建RAG系统的环境变量配置"""
    env = EnvManager()
    env.add("APP_NAME", "RAG系统", "应用名称")
    env.add("APP_VERSION", "1.0.0", "应用版本")
    env.add("APP_HOST", "0.0.0.0", "监听地址", default="0.0.0.0")
    env.add("APP_PORT", "8000", "监听端口", default="8000")
    env.add("DEBUG", "false", "调试模式", default="false")
    env.add("LOG_LEVEL", "INFO", "日志级别", default="INFO")
    env.add("OPENAI_API_KEY", "", "OpenAI API密钥", required=True)
    env.add("OPENAI_MODEL", "gpt-4", "使用的模型", default="gpt-4")
    env.add("DATABASE_URL", "", "数据库连接", required=True)
    env.add("REDIS_URL", "", "Redis连接", default="redis://localhost:6379/0")
    env.add("EMBEDDING_MODEL", "text-embedding-3-small", "嵌入模型")
    env.add("CHUNK_SIZE", "500", "分块大小", default="500")
    env.add("CHUNK_OVERLAP", "50", "分块重叠", default="50")
    env.add("TOP_K", "5", "检索文档数", default="5")
    env.add("TEMPERATURE", "0.7", "生成温度", default="0.7")
    return env


# ============================================================
# 4. requirements.txt生成
# ============================================================

def generate_requirements() -> str:
    """生成requirements.txt"""
    packages = {
        # Web框架
        "fastapi": ">=0.100.0",
        "uvicorn": ">=0.23.0",
        "gradio": ">=4.0.0",
        # 数据验证
        "pydantic": ">=2.0.0",
        # 向量与ML
        "numpy": ">=1.24.0",
        "sentence-transformers": ">=2.2.0",
        # 数据库
        "sqlalchemy": ">=2.0.0",
        "psycopg2-binary": ">=2.9.0",
        "redis": ">=4.6.0",
        # 文档处理
        "pypdf": ">=3.0.0",
        "python-docx": ">=0.8.0",
        "markdown": ">=3.4.0",
        # 工具
        "python-dotenv": ">=1.0.0",
        "httpx": ">=0.24.0",
        "loguru": ">=0.7.0",
    }
    lines = []
    for pkg, version in packages.items():
        lines.append(f"{pkg}{version}")
    return "\n".join(lines)


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("W31-D3 Docker配置生成")
    print("=" * 60)

    # --- 1. Dockerfile ---
    print("\n--- 1. Dockerfile (普通构建) ---")
    gen = DockerfileGenerator()
    gen.add_requirement("fastapi", "uvicorn", "numpy")
    gen.set_env("PYTHONUNBUFFERED", "1")
    gen.expose(8000)
    dockerfile = gen.generate()
    print(dockerfile)

    # --- 2. 多阶段Dockerfile ---
    print(f"\n{'='*60}")
    print("--- 2. Dockerfile (多阶段构建) ---")
    print(f"{'='*60}")
    multi_gen = MultiStageDockerfileGenerator()
    multi_gen.set_env("PYTHONUNBUFFERED", "1")
    multi_dockerfile = multi_gen.generate()
    print(multi_dockerfile)

    # --- 3. docker-compose ---
    print(f"\n{'='*60}")
    print("--- 3. docker-compose.yml ---")
    print(f"{'='*60}")
    compose = create_rag_compose()
    print(compose)

    # --- 4. 环境变量 ---
    print(f"\n{'='*60}")
    print("--- 4. 环境变量配置 ---")
    print(f"{'='*60}")
    env_config = create_rag_env_config()
    print(env_config.generate_env_example())

    # 验证
    test_env = {"OPENAI_API_KEY": "sk-test", "DATABASE_URL": "postgresql://..."}
    missing = env_config.validate(test_env)
    if missing:
        print(f"缺少必填变量: {missing}")
    else:
        print("环境变量验证通过!")

    # --- 5. requirements.txt ---
    print(f"\n--- 5. requirements.txt ---")
    print(generate_requirements())

    print("\n完成!")
