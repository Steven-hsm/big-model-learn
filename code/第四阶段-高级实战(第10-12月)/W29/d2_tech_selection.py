"""
W29-D2: 技术选型对比
====================
功能:
  - 技术选型对比 (FastAPI vs Flask, Chroma vs FAISS, 不同LLM)
  - 打印技术选型表
  - 架构图模板
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


# ============================================================
# 1. 数据模型
# ============================================================

class Category(Enum):
    """技术分类"""
    WEB_FRAMEWORK = "Web框架"
    VECTOR_STORE = "向量数据库"
    LLM_PROVIDER = "LLM提供商"
    EMBEDDING = "嵌入模型"
    ORCHESTRATION = "编排框架"
    FRONTEND = "前端框架"
    DEPLOYMENT = "部署方案"


class Score(Enum):
    """评分等级"""
    EXCELLENT = 5
    GOOD = 4
    AVERAGE = 3
    BELOW_AVERAGE = 2
    POOR = 1

    def __str__(self):
        return "★" * self.value + "☆" * (5 - self.value)


@dataclass
class TechOption:
    """技术选项"""
    name: str
    category: Category
    description: str
    pros: List[str] = field(default_factory=list)
    cons: List[str] = field(default_factory=list)
    scores: Dict[str, int] = field(default_factory=dict)  # 评分维度
    recommendation: str = ""


# ============================================================
# 2. 技术选型对比器
# ============================================================

class TechSelector:
    """技术选型对比器"""

    def __init__(self):
        self.options: List[TechOption] = []
        self.decisions: Dict[str, str] = {}  # category -> chosen option

    def add_option(self, option: TechOption):
        """添加技术选项"""
        self.options.append(option)

    def make_decision(self, category: Category, chosen: str, reason: str):
        """做出技术选型决策"""
        self.decisions[category.value] = f"{chosen} (原因: {reason})"

    def compare_by_category(self, category: Category) -> str:
        """按分类对比技术选项"""
        lines = []
        options = [o for o in self.options if o.category == category]

        if not options:
            return f"未找到 {category.value} 的技术选项"

        lines.append("\n" + "=" * 70)
        lines.append(f"  {category.value} 技术选型对比")
        lines.append("=" * 70)

        for opt in options:
            lines.append(f"\n▶ {opt.name}")
            lines.append(f"  描述: {opt.description}")
            lines.append("  优点:")
            for pro in opt.pros:
                lines.append(f"    ✓ {pro}")
            lines.append("  缺点:")
            for con in opt.cons:
                lines.append(f"    ✗ {con}")
            if opt.scores:
                lines.append("  评分:")
                for dim, score in opt.scores.items():
                    stars = "★" * score + "☆" * (5 - score)
                    lines.append(f"    {dim}: {stars} ({score}/5)")
            if opt.recommendation:
                lines.append(f"  推荐场景: {opt.recommendation}")
            lines.append("")

        return "\n".join(lines)

    def print_decision_summary(self) -> str:
        """打印选型决策摘要"""
        lines = []
        lines.append("\n" + "=" * 70)
        lines.append("  最终技术选型决策")
        lines.append("=" * 70)

        for category, decision in self.decisions.items():
            lines.append(f"\n  [{category}]")
            lines.append(f"    → {decision}")

        lines.append("")
        return "\n".join(lines)

    def generate_comparison_table(self, category: Category) -> str:
        """生成对比表格"""
        options = [o for o in self.options if o.category == category]
        if not options:
            return ""

        # 收集所有评分维度
        all_dims = set()
        for opt in options:
            all_dims.update(opt.scores.keys())
        all_dims = sorted(all_dims)

        lines = []
        lines.append(f"\n{category.value} 对比表:")
        lines.append("-" * 60)

        # 表头
        header = f"{'维度':<12}"
        for opt in options:
            header += f"{opt.name:<15}"
        lines.append(header)
        lines.append("-" * 60)

        # 每行评分
        for dim in all_dims:
            row = f"{dim:<12}"
            for opt in options:
                score = opt.scores.get(dim, 0)
                row += f"{score}/5{'':<11}"
            lines.append(row)

        # 平均分
        avg_row = f"{'平均分':<12}"
        for opt in options:
            if opt.scores:
                avg = sum(opt.scores.values()) / len(opt.scores)
                avg_row += f"{avg:.1f}{'':<13}"
            else:
                avg_row += f"{'N/A':<15}"
        lines.append(avg_row)
        lines.append("-" * 60)

        return "\n".join(lines)


# ============================================================
# 3. 架构图模板生成器
# ============================================================

class ArchitectureDiagram:
    """架构图文本模板"""

    @staticmethod
    def rag_architecture() -> str:
        """RAG系统架构图"""
        return """
╔══════════════════════════════════════════════════════════════╗
║                    RAG 系统架构图                           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║   [用户] ──→ [前端 (Gradio)] ──→ [API网关 (FastAPI)]        ║
║                                       │                      ║
║                    ┌──────────────────┼──────────────────┐   ║
║                    │                  │                  │   ║
║              [查询处理]        [文档管理]          [对话管理]  ║
║                    │                  │                  │   ║
║                    ▼                  ▼                  │   ║
║              [查询改写]        [文档解析器]              │   ║
║                    │           PDF/Word/TXT             │   ║
║                    ▼                  ▼                  │   ║
║              [混合检索]        [文本切分器]              │   ║
║             BM25+向量         固定/递归                  │   ║
║                    │                  │                  │   ║
║                    ▼                  ▼                  │   ║
║              [重排序器]        [嵌入模型]              │   ║
║            Cross-encoder     BGE/OpenAI                │   ║
║                    │                  │                  │   ║
║                    │                  ▼                  │   ║
║                    │           [向量数据库]              │   ║
║                    │           Chroma/FAISS             │   ║
║                    │                  │                  │   ║
║                    ▼                  │                  │   ║
║              [上下文组装] ◄────────────┘                  │   ║
║                    │                                      │   ║
║                    ▼                                      ▼   ║
║              [LLM 生成] ◄────────── [对话历史]              ║
║          GPT-4/Claude/本地模型        Redis/内存            ║
║                    │                                         ║
║                    ▼                                         ║
║              [流式输出] ──→ [前端展示] ──→ [用户]            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""

    @staticmethod
    def deployment_architecture() -> str:
        """部署架构图"""
        return """
╔══════════════════════════════════════════════════╗
║              部署架构图                          ║
╠══════════════════════════════════════════════════╣
║                                                  ║
║   [Nginx 反向代理]                               ║
║        │                                         ║
║   ┌────┴────┐                                    ║
║   │         │                                    ║
║  [前端]   [API服务]                               ║
║  静态文件  FastAPI                                ║
║              │                                    ║
║        ┌─────┼─────┐                              ║
║        │     │     │                              ║
║     [向量] [Redis] [LLM]                          ║
║     数据库  缓存   API/本地                         ║
║                                                  ║
║   Docker Compose 一键部署                         ║
╚══════════════════════════════════════════════════╝
"""


# ============================================================
# 4. 演示: RAG项目技术选型
# ============================================================

def demo():
    """演示技术选型过程"""

    print("=" * 60)
    print("W29-D2: 技术选型对比")
    print("=" * 60)

    selector = TechSelector()

    # ---- Web框架对比 ----
    selector.add_option(TechOption(
        name="FastAPI",
        category=Category.WEB_FRAMEWORK,
        description="现代高性能Python Web框架, 基于类型提示",
        pros=[
            "异步支持, 性能优异",
            "自动生成API文档 (Swagger)",
            "类型提示, 开发体验好",
            "Pydantic数据验证",
            "WebSocket/SSE支持",
        ],
        cons=[
            "相对较新, 生态不如Flask成熟",
            "学习曲线略陡",
        ],
        scores={
            "性能": 5,
            "开发效率": 5,
            "生态": 4,
            "学习曲线": 4,
            "异步支持": 5,
        },
        recommendation="RAG项目首选",
    ))

    selector.add_option(TechOption(
        name="Flask",
        category=Category.WEB_FRAMEWORK,
        description="轻量级Python Web框架, 灵活可扩展",
        pros=[
            "简单易学",
            "生态丰富",
            "社区庞大",
            "灵活性高",
        ],
        cons=[
            "异步支持需要额外配置",
            "缺少内置数据验证",
            "性能不如FastAPI",
        ],
        scores={
            "性能": 3,
            "开发效率": 3,
            "生态": 5,
            "学习曲线": 5,
            "异步支持": 2,
        },
        recommendation="小型项目或快速原型",
    ))

    # ---- 向量数据库对比 ----
    selector.add_option(TechOption(
        name="ChromaDB",
        category=Category.VECTOR_STORE,
        description="开源嵌入式向量数据库, 专为AI应用设计",
        pros=[
            "易于使用, 开箱即用",
            "内置嵌入支持",
            "轻量级, 适合开发",
            "支持过滤查询",
        ],
        cons=[
            "大规��性能有待验证",
            "生产环境支持较弱",
            "分布式支持有限",
        ],
        scores={
            "易用性": 5,
            "性能": 3,
            "可扩展性": 3,
            "功能": 4,
            "社区": 4,
        },
        recommendation="开发和MVP阶段",
    ))

    selector.add_option(TechOption(
        name="FAISS",
        category=Category.VECTOR_STORE,
        description="Facebook开源的高效向量相似度搜索库",
        pros=[
            "检索速度极快",
            "支持GPU加速",
            "大规模数据支持好",
            "多种索引类型",
        ],
        cons=[
            "仅支持向量检索, 需自建元数据管理",
            "不支持增量更新(部分索引)",
            "Python API不如Chroma友好",
        ],
        scores={
            "易用性": 3,
            "性能": 5,
            "可扩展性": 5,
            "功能": 3,
            "社区": 4,
        },
        recommendation="性能敏感的生产环境",
    ))

    selector.add_option(TechOption(
        name="Milvus",
        category=Category.VECTOR_STORE,
        description="开源分布式向量数据库",
        pros=[
            "分布式架构, 可扩展",
            "支持多种索引类型",
            "丰富的过滤功能",
            "生产级别可靠性",
        ],
        cons=[
            "部署复杂",
            "资源占用大",
            "学习成本较高",
        ],
        scores={
            "易用性": 2,
            "性能": 5,
            "可扩展性": 5,
            "功能": 5,
            "社区": 4,
        },
        recommendation="大规模生产环境",
    ))

    # ---- LLM对比 ----
    selector.add_option(TechOption(
        name="OpenAI GPT-4",
        category=Category.LLM_PROVIDER,
        description="OpenAI旗舰模型, 综合能力最强",
        pros=["综合能力最强", "指令遵循好", "中英文表现优秀"],
        cons=["价格较高", "API依赖外部服务", "数据隐私考量"],
        scores={"能力": 5, "价格": 2, "速度": 3, "隐私": 2},
        recommendation="追求质量的场景",
    ))

    selector.add_option(TechOption(
        name="Claude",
        category=Category.LLM_PROVIDER,
        description="Anthropic的AI助手, 长上下文支持",
        pros=["超长上下文", "安全性好", "中文能力强"],
        cons=["API依赖外部服务", "价格中等"],
        scores={"能力": 5, "价格": 3, "速度": 4, "隐私": 2},
        recommendation="需要长上下文的场景",
    ))

    selector.add_option(TechOption(
        name="本地模型 (Qwen/ChatGLM)",
        category=Category.LLM_PROVIDER,
        description="本地部署的开源模型",
        pros=["数据不出域", "无API费用", "可定制微调"],
        cons=["需要GPU资源", "能力不如GPT-4", "部署维护成本"],
        scores={"能力": 3, "价格": 5, "速度": 3, "隐私": 5},
        recommendation="数据敏感场景",
    ))

    # ---- 打印对比结果 ----
    for cat in Category:
        options = [o for o in selector.options if o.category == cat]
        if options:
            print(selector.compare_by_category(cat))
            print(selector.generate_comparison_table(cat))

    # ---- 做出选型决策 ----
    selector.make_decision(Category.WEB_FRAMEWORK, "FastAPI",
                           "异步支持好, 自动文档, 类型安全")
    selector.make_decision(Category.VECTOR_STORE, "ChromaDB(MVP) → FAISS/Milvus(生产)",
                           "开发用Chroma, 生产迁移FAISS/Milvus")
    selector.make_decision(Category.LLM_PROVIDER, "OpenAI GPT-4 + 本地模型备选",
                           "主要用GPT-4, 本地模型作降级方案")

    print(selector.print_decision_summary())

    # ---- 架构图 ----
    arch = ArchitectureDiagram()
    print(arch.rag_architecture())
    print(arch.deployment_architecture())


# ============================================================
# 运行演示
# ============================================================

if __name__ == "__main__":
    demo()
