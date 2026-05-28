"""
W31-D7 Demo准备
================
准备RAG系统的演示材料, 包括:
- 示例数据生成
- 演示脚本
- README内容生成

好的Demo能让项目展示事半功倍。
"""

import json
import time
from typing import List, Dict


# ============================================================
# 1. 示例数据生成
# ============================================================

class SampleDataGenerator:
    """示例数据生成器"""

    @staticmethod
    def generate_knowledge_base() -> List[Dict]:
        """生成知识库示例数据"""
        documents = [
            {
                "title": "RAG技术概述",
                "content": "RAG(检索增强生成, Retrieval-Augmented Generation)是一种将信息检索与文本生成相结合的AI技术框架。"
                          "其核心思想是: 在生成回答前, 先从知识库中检索相关文档, 然后将检索到的内容作为上下文传递给大语言模型。"
                          "RAG有效解决了LLM的幻觉问题和知识更新问题, 让AI的回答有据可依。",
                "category": "技术",
                "tags": ["RAG", "LLM", "检索"],
            },
            {
                "title": "向量嵌入原理",
                "content": "向量嵌入是将文本转换为高维数值向量的技术。通过嵌入模型, 语义相近的文本在向量空间中的距离也相近。"
                          "常用的嵌入模型包括OpenAI的text-embedding、BGE系列和E5系列。"
                          "向量嵌入是语义搜索和RAG系统的基础组件。",
                "category": "技术",
                "tags": ["嵌入", "向量", "NLP"],
            },
            {
                "title": "BM25检索算法",
                "content": "BM25是一种基于词频的经典信息检索算法。它考虑了词频(TF)、逆文档频率(IDF)和文档长度归一化三个因素。"
                          "BM25的优点是计算速度快, 不需要训练模型; 缺点是无法理解语义, 对同义词和近义词效果较差。"
                          "在实际RAG系统中, BM25常与向量检索结合使用。",
                "category": "技术",
                "tags": ["BM25", "检索", "算法"],
            },
            {
                "title": "Python数据科学生态",
                "content": "Python拥有丰富的数据科学生态: NumPy提供高效的数组计算, Pandas处理结构化数据, "
                          "Matplotlib和Seaborn用于可视化, Scikit-learn提供传统机器学习算法, "
                          "PyTorch和TensorFlow支持深度学习。这些库共同构成了Python在AI领域的强大工具链。",
                "category": "编程",
                "tags": ["Python", "数据科学", "工具"],
            },
            {
                "title": "Prompt工程指南",
                "content": "Prompt工程是设计有效提示词以引导LLM生成高质量输出的技术。"
                          "关键原则包括: 明确角色定义、提供充足的上下文、使用结构化格式、"
                          "给出具体的示例(Few-shot)、设置输出约束。"
                          "在RAG系统中, Prompt模板通常包含系统指令、检索上下文和用户问题三部分。",
                "category": "技术",
                "tags": ["Prompt", "LLM", "工程"],
            },
            {
                "title": "FastAPI框架入门",
                "content": "FastAPI是一个现代、高性能的Python Web框架。它基于类型提示自动生成API文档, "
                          "使用Pydantic进行数据验证, 支持异步处理, 性能接近Go和Node.js。"
                          "FastAPI非常适合构建RAG系统的REST API服务。",
                "category": "编程",
                "tags": ["FastAPI", "Web", "API"],
            },
            {
                "title": "Docker容器化部署",
                "content": "Docker通过容器化技术将应用及其依赖打包成标准化单元。"
                          "Dockerfile定义镜像构建步骤, docker-compose编排多个服务。"
                          "多阶段构建可以减小镜像体积, 非root用户运行提高安全性。"
                          "RAG系统通常需要API服务、数据库和向量数据库三个容器。",
                "category": "运维",
                "tags": ["Docker", "部署", "容器"],
            },
            {
                "title": "评估指标体系",
                "content": "RAG系统的评估分为检索评估和生成评估两部分。"
                          "检索评估指标: Recall@K(召回率)、Precision@K(精确率)、MRR、NDCG。"
                          "生成评估指标: Faithfulness(忠实度)、Relevance(相关性)、Fluency(流畅度)。"
                          "综合评估还需要考虑端到端延迟和用户满意度。",
                "category": "技术",
                "tags": ["评估", "指标", "质量"],
            },
        ]
        return documents

    @staticmethod
    def generate_demo_queries() -> List[Dict]:
        """生成演示查询"""
        return [
            {
                "query": "什么是RAG?",
                "expected_sources": ["RAG技术概述"],
                "category": "概念解释",
            },
            {
                "query": "如何评估RAG系统的质量?",
                "expected_sources": ["评估指标体系"],
                "category": "方法论",
            },
            {
                "query": "Python有哪些数据科学库?",
                "expected_sources": ["Python数据科学生态"],
                "category": "知识查询",
            },
            {
                "query": "BM25和向量检索有什么区别?",
                "expected_sources": ["BM25检索算法", "向量嵌入原理"],
                "category": "对比分析",
            },
            {
                "query": "如何设计好的Prompt?",
                "expected_sources": ["Prompt工程指南"],
                "category": "实践指导",
            },
            {
                "query": "RAG系统如何部署?",
                "expected_sources": ["Docker容器化部署", "FastAPI框架入门"],
                "category": "技术实践",
            },
        ]


# ============================================================
# 2. 演示脚本
# ============================================================

class DemoScript:
    """演示脚本生成器"""

    def __init__(self):
        self.steps = []
        self.data_gen = SampleDataGenerator()

    def add_step(self, title: str, description: str, action: str,
                  expected_result: str):
        self.steps.append({
            'title': title,
            'description': description,
            'action': action,
            'expected_result': expected_result,
        })

    def build_demo(self):
        """构建标准演示流程"""
        self.steps = [
            {
                'title': '开场介绍',
                'description': '介绍项目背景和目标',
                'action': '简述: 这是一个基于RAG技术的知识库问答系统, 支持文档上传、智能检索和AI回答。',
                'expected_result': '观众了解项目定位',
            },
            {
                'title': '文档上传',
                'description': '演示上传文档功能',
                'action': '上传3-5篇示例文档, 展示文档列表和索引状态',
                'expected_result': '文档成功索引, 显示分块数量',
            },
            {
                'title': '基础查询',
                'description': '演示基本问答功能',
                'action': '输入"什么是RAG?", 展示AI回答和参考来源',
                'expected_result': '回答准确, 来源可追溯',
            },
            {
                'title': '复杂查询',
                'description': '演示处理复杂问题的能力',
                'action': '输入"BM25和向量检索有什么区别?", 展示多文档综合回答',
                'expected_result': '回答对比了两种方法的优劣',
            },
            {
                'title': '参数调节',
                'description': '演示参数对结果的影响',
                'action': '调整Top-K从3到10, 观察结果变化',
                'expected_result': '展示更多文档带来更全面的信息',
            },
            {
                'title': '无结果处理',
                'description': '演示边界情况处理',
                'action': '输入一个知识库中不存在的问题',
                'expected_result': '系统友好地提示未找到相关信息',
            },
            {
                'title': '性能展示',
                'description': '展示系统响应速度',
                'action': '连续发送3个查询, 展示平均延迟',
                'expected_result': '延迟在可接受范围内(<500ms)',
            },
        ]

    def print_script(self):
        """打印演示脚本"""
        print(f"\n{'='*60}")
        print("RAG系统演示脚本")
        print(f"{'='*60}")
        for i, step in enumerate(self.steps, 1):
            print(f"\n步骤{i}: {step['title']}")
            print(f"  说明: {step['description']}")
            print(f"  操作: {step['action']}")
            print(f"  预期: {step['expected_result']}")


# ============================================================
# 3. README生成器
# ============================================================

class ReadmeGenerator:
    """README生成器"""

    def __init__(self):
        self.project_name = "RAG知识库问答系统"
        self.description = "基于检索增强生成技术的智能问答系统"
        self.features = []
        self.tech_stack = []
        self.quick_start = []
        self.api_docs = []

    def set_project(self, name: str, description: str):
        self.project_name = name
        self.description = description
        return self

    def add_feature(self, feature: str):
        self.features.append(feature)
        return self

    def add_tech(self, name: str, purpose: str):
        self.tech_stack.append((name, purpose))
        return self

    def generate(self) -> str:
        """生成README内容"""
        sections = []

        # 标题
        sections.append(f"# {self.project_name}")
        sections.append("")
        sections.append(self.description)
        sections.append("")

        # 功能特性
        if self.features:
            sections.append("## 功能特性")
            sections.append("")
            for f in self.features:
                sections.append(f"- {f}")
            sections.append("")

        # 技术栈
        if self.tech_stack:
            sections.append("## 技术栈")
            sections.append("")
            sections.append("| 技术 | 用途 |")
            sections.append("|------|------|")
            for name, purpose in self.tech_stack:
                sections.append(f"| {name} | {purpose} |")
            sections.append("")

        # 快速开始
        sections.append("## 快速开始")
        sections.append("")
        sections.append("```bash")
        sections.append("# 1. 克隆项目")
        sections.append("git clone https://github.com/yourname/rag-system.git")
        sections.append("cd rag-system")
        sections.append("")
        sections.append("# 2. 安装依赖")
        sections.append("pip install -r requirements.txt")
        sections.append("")
        sections.append("# 3. 配置环境变量")
        sections.append("cp .env.example .env")
        sections.append("# 编辑.env文件, 填写API密钥等配置")
        sections.append("")
        sections.append("# 4. 启动服务")
        sections.append("python main.py")
        sections.append("# 或使用Docker")
        sections.append("docker-compose up -d")
        sections.append("```")
        sections.append("")

        # API文档
        sections.append("## API接口")
        sections.append("")
        sections.append("| 方法 | 路径 | 说明 |")
        sections.append("|------|------|------|")
        sections.append("| POST | /api/query | 查询接口 |")
        sections.append("| POST | /api/documents | 上传文档 |")
        sections.append("| GET | /api/documents | 列出文档 |")
        sections.append("| DELETE | /api/documents/{id} | 删除文档 |")
        sections.append("| GET | /api/health | 健康检查 |")
        sections.append("")

        # 项目结构
        sections.append("## 项目结构")
        sections.append("```")
        sections.append("rag-system/")
        sections.append("  src/")
        sections.append("    retrieval/    # 检索模块(BM25, 向量检索)")
        sections.append("    generation/   # 生成模块(Prompt, LLM调用)")
        sections.append("    reranker/     # 重排序模块")
        sections.append("    api/          # API路由")
        sections.append("    core/         # 核心配置")
        sections.append("  tests/          # 测试")
        sections.append("  docs/           # 文档")
        sections.append("  Dockerfile")
        sections.append("  docker-compose.yml")
        sections.append("```")
        sections.append("")

        # 许可证
        sections.append("## 许可证")
        sections.append("MIT License")

        return "\n".join(sections)


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("W31-D7 Demo准备")
    print("=" * 60)

    # --- 1. 示例数据 ---
    print("\n--- 1. 示例数据 ---")
    data_gen = SampleDataGenerator()
    docs = data_gen.generate_knowledge_base()
    queries = data_gen.generate_demo_queries()

    print(f"知识库文档: {len(docs)}篇")
    for doc in docs:
        print(f"  - {doc['title']} ({doc['category']}, 标签: {', '.join(doc['tags'])})")

    print(f"\n演示查询: {len(queries)}个")
    for q in queries:
        print(f"  - [{q['category']}] {q['query']}")

    # --- 2. 演示脚本 ---
    print(f"\n{'='*60}")
    print("--- 2. 演示脚本 ---")
    print(f"{'='*60}")

    demo = DemoScript()
    demo.build_demo()
    demo.print_script()

    # --- 3. README ---
    print(f"\n{'='*60}")
    print("--- 3. README内容 ---")
    print(f"{'='*60}")

    readme = ReadmeGenerator()
    readme.set_project("RAG知识库问答系统", "基于检索增强生成(RAG)技术的智能知识库问答系统")
    readme.add_feature("智能问答: 基于知识库的准确回答")
    readme.add_feature("文档管理: 支持 txt/pdf/md 文档上传和索引")
    readme.add_feature("混合检索: BM25 + 向量语义检索 + RRF融合")
    readme.add_feature("重排序: Cross-encoder精准重排序")
    readme.add_feature("流式输出: SSE实时推送生成内容")
    readme.add_feature("Web界面: Gradio聊天界面")
    readme.add_feature("API服务: FastAPI RESTful API")
    readme.add_feature("容器化: Docker一键部署")
    readme.add_tech("Python 3.11", "主要开发语言")
    readme.add_tech("FastAPI", "Web框架和API服务")
    readme.add_tech("sentence-transformers", "文本嵌入模型")
    readme.add_tech("FAISS/Milvus", "向量存储和检索")
    readme.add_tech("Gradio", "前端界面")
    readme.add_tech("Docker", "容器化部署")
    readme.add_tech("PostgreSQL", "文档和元数据存储")
    readme.add_tech("Redis", "缓存和会话管理")

    readme_content = readme.generate()
    print(readme_content[:800])
    print(f"... (共{len(readme_content)}字符)")

    # 保存示例数据
    demo_data = {
        'documents': docs,
        'queries': queries,
    }
    print(f"\n示例数据已生成: {len(json.dumps(demo_data, ensure_ascii=False))} 字符")

    print("\n完成!")
