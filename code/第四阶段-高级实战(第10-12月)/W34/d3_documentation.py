"""
W34-D3 文档生成
================
自动生成项目文档, 包括:
- API文档
- 架构文档
- 使用指南

好的文档让项目更易被理解和使用。
"""

import time
import json
from typing import List, Dict, Optional


# ============================================================
# 1. API文档生成器
# ============================================================

class APIDocGenerator:
    """API文档生成器"""

    def __init__(self):
        self.title = "RAG知识库问答系统 API"
        self.version = "1.0.0"
        self.endpoints = []

    def add_endpoint(self, method: str, path: str, description: str,
                      params: List[Dict] = None, response: Dict = None,
                      examples: List[Dict] = None):
        """添加API端点"""
        self.endpoints.append({
            'method': method,
            'path': path,
            'description': description,
            'params': params or [],
            'response': response or {},
            'examples': examples or [],
        })

    def generate_markdown(self) -> str:
        """生成Markdown格式的API文档"""
        lines = [
            f"# {self.title}",
            f"版本: {self.version}",
            f"基础URL: `http://localhost:8000`",
            "",
            "---",
            "",
            "## 接口概览",
            "",
            "| 方法 | 路径 | 说明 |",
            "|------|------|------|",
        ]

        for ep in self.endpoints:
            lines.append(f"| {ep['method']} | `{ep['path']}` | {ep['description']} |")

        lines.append("")
        lines.append("---")
        lines.append("")

        # 详细文档
        for ep in self.endpoints:
            lines.append(f"## {ep['method']} {ep['path']}")
            lines.append("")
            lines.append(ep['description'])
            lines.append("")

            # 参数
            if ep['params']:
                lines.append("### 参数")
                lines.append("")
                lines.append("| 参数名 | 类型 | 必填 | 说明 |")
                lines.append("|--------|------|------|------|")
                for param in ep['params']:
                    required = "是" if param.get('required') else "否"
                    lines.append(f"| {param['name']} | {param.get('type', 'string')} | "
                                 f"{required} | {param.get('description', '')} |")
                lines.append("")

            # 响应
            if ep['response']:
                lines.append("### 响应")
                lines.append("")
                lines.append("```json")
                lines.append(json.dumps(ep['response'], ensure_ascii=False, indent=2))
                lines.append("```")
                lines.append("")

            # 示例
            if ep['examples']:
                lines.append("### 示例")
                for ex in ep['examples']:
                    lines.append("")
                    lines.append(f"**{ex.get('title', '示例')}**")
                    lines.append("```bash")
                    lines.append(ex.get('curl', ''))
                    lines.append("```")
                lines.append("")

            lines.append("---")
            lines.append("")

        return "\n".join(lines)


# ============================================================
# 2. 架构文档生成器
# ============================================================

class ArchitectureDocGenerator:
    """架构文档生成器"""

    def __init__(self):
        self.project_name = "RAG知识库问答系统"
        self.components = []
        self.data_flows = []

    def add_component(self, name: str, description: str,
                       responsibilities: List[str],
                       dependencies: List[str] = None):
        self.components.append({
            'name': name,
            'description': description,
            'responsibilities': responsibilities,
            'dependencies': dependencies or [],
        })

    def add_data_flow(self, name: str, steps: List[str]):
        self.data_flows.append({'name': name, 'steps': steps})

    def generate(self) -> str:
        """生成架构文档"""
        lines = [
            f"# {self.project_name} 架构文档",
            "",
            "## 系统架构概览",
            "",
            "```",
            "用户层:    [Web浏览器] [移动端] [API客户端]",
            "                |           |          |",
            "接口层:    [FastAPI REST API]   [Gradio UI]",
            "                |",
            "业务层:    [查询引擎] [文档管理] [会话管理]",
            "                |",
            "检索层:    [BM25] [向量检索] [重排序] [HyDE]",
            "                |",
            "数据层:    [PostgreSQL] [Redis] [向量DB]",
            "```",
            "",
        ]

        # 组件详细说明
        lines.append("## 核心组件")
        lines.append("")

        for comp in self.components:
            lines.append(f"### {comp['name']}")
            lines.append(f"{comp['description']}")
            lines.append("")
            lines.append("**职责:**")
            for resp in comp['responsibilities']:
                lines.append(f"- {resp}")
            if comp['dependencies']:
                lines.append("")
                lines.append(f"**依赖:** {', '.join(comp['dependencies'])}")
            lines.append("")

        # 数据流
        if self.data_flows:
            lines.append("## 数据流")
            lines.append("")
            for flow in self.data_flows:
                lines.append(f"### {flow['name']}")
                for i, step in enumerate(flow['steps'], 1):
                    lines.append(f"{i}. {step}")
                lines.append("")

        # 技术选型
        lines.append("## 技术选型")
        lines.append("")
        tech_choices = [
            ("Python 3.11", "主要开发语言, AI生态完善"),
            ("FastAPI", "高性能异步Web框架, 自动生成API文档"),
            ("PostgreSQL", "关系型数据库, 存储文档元数据和用户信息"),
            ("Redis", "缓存和会话存储, 提升查询性能"),
            ("sentence-transformers", "文本嵌入, 将文本转为向量"),
            ("Gradio", "快速构建Web界面, 适合AI应用Demo"),
        ]
        lines.append("| 技术 | 选型理由 |")
        lines.append("|------|----------|")
        for tech, reason in tech_choices:
            lines.append(f"| {tech} | {reason} |")

        return "\n".join(lines)


# ============================================================
# 3. 使用指南生成器
# ============================================================

class UserGuideGenerator:
    """使用指南生成器"""

    def generate(self) -> str:
        return """# RAG知识库问答系统 使用指南

## 快速开始

### 1. 安装

```bash
# 克隆项目
git clone https://github.com/your-org/rag-system.git
cd rag-system

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置

```bash
# 复制配置模板
cp .env.example .env

# 编辑配置(必须填写OPENAI_API_KEY)
# 可选配置: DATABASE_URL, REDIS_URL等
```

### 3. 启动

```bash
# 方式一: 直接启动
python main.py

# 方式二: Docker启动
docker-compose up -d

# 方式三: 开发模式
uvicorn main:app --reload
```

## 使用方式

### API调用

```python
import requests

# 查询
response = requests.post(
    "http://localhost:8000/api/query",
    json={"query": "什么是RAG?", "top_k": 5}
)
print(response.json())

# 上传文档
with open("doc.txt", "rb") as f:
    requests.post(
        "http://localhost:8000/api/documents",
        files={"file": f}
    )
```

### Web界面

访问 http://localhost:7860 打开Gradio界面。

### 常用操作

1. **上传文档**: 支持txt、md格式
2. **查询**: 输入问题获取AI回答
3. **参数调节**: 调整Top-K和温度参数
4. **查看来源**: 点击查看回答的参考文档

## 高级配置

### 自定义检索参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| chunk_size | 500 | 文档分块大小 |
| chunk_overlap | 50 | 分块重叠字符数 |
| top_k | 5 | 检索返回文档数 |
| temperature | 0.7 | 生成温度 |

### 自定义Prompt模板

系统支持自定义Prompt模板, 修改 templates/ 目录下的文件即可。

## 常见问题

### Q: 如何提升检索质量?
A: 尝试以下方法:
- 使用混合检索(BM25+向量)
- 添加重排序器
- 调整chunk_size参数
- 补充更多高质量文档

### Q: 如何减少幻觉?
A: 在Prompt中强调"只基于上下文回答", 降低temperature参数。

### Q: 支持哪些文件格式?
A: 目前支持txt、md、pdf格式。
"""


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("W34-D3 文档生成")
    print("=" * 60)

    # --- 1. API文档 ---
    print("\n--- 1. API文档 ---")
    api_doc = APIDocGenerator()

    api_doc.add_endpoint(
        "POST", "/api/query", "查询接口",
        params=[
            {"name": "query", "type": "string", "required": True, "description": "查询文本"},
            {"name": "top_k", "type": "integer", "required": False, "description": "返回文档数(默认5)"},
            {"name": "temperature", "type": "float", "required": False, "description": "生成温度(默认0.7)"},
        ],
        response={"answer": "回答内容", "sources": [], "latency_ms": 150},
        examples=[{"title": "基本查询", "curl": "curl -X POST http://localhost:8000/api/query -d '{\"query\": \"什么是RAG?\"}'"}]
    )

    api_doc.add_endpoint(
        "POST", "/api/documents", "上传文档",
        params=[
            {"name": "file", "type": "file", "required": True, "description": "文档文件"},
            {"name": "title", "type": "string", "required": False, "description": "文档标题"},
        ],
        response={"doc_id": "abc123", "status": "success"}
    )

    api_doc.add_endpoint(
        "GET", "/api/health", "健康检查",
        response={"status": "healthy", "version": "1.0.0"}
    )

    md = api_doc.generate_markdown()
    print(md[:800])
    print(f"\n... (共{len(md)}字符)")

    # --- 2. 架构文档 ---
    print(f"\n{'='*60}")
    print("--- 2. 架构文档 ---")
    print(f"{'='*60}")

    arch_doc = ArchitectureDocGenerator()
    arch_doc.add_component("检索引擎", "负责从知识库中检索相关文档",
                           ["BM25关键词检索", "向量语义检索", "RRF融合", "重排序"],
                           ["向量数据库", "Redis缓存"])
    arch_doc.add_component("生成引擎", "基于检索结果生成回答",
                           ["Prompt模板管理", "LLM调用", "流式输出"],
                           ["LLM API"])
    arch_doc.add_component("API服务", "提供REST API接口",
                           ["请求路由", "数据验证", "错误处理"],
                           ["检索引擎", "生成引擎"])

    arch_doc.add_data_flow("查询流程", [
        "用户发送查询请求",
        "API层接收并验证请求",
        "检索引擎从知识库检索相关文档",
        "重排序器对结果精排序",
        "生成引擎基于上下文生成回答",
        "返回回答和引用来源",
    ])

    arch_md = arch_doc.generate()
    print(arch_md[:600])
    print(f"\n... (共{len(arch_md)}字符)")

    # --- 3. 使用指南 ---
    print(f"\n{'='*60}")
    print("--- 3. 使用指南 ---")
    print(f"{'='*60}")
    guide = UserGuideGenerator().generate()
    print(f"使用指南: {len(guide)}字符")

    print("\n完成!")
