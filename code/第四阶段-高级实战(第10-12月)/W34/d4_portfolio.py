"""
W34-D4 作品集
==============
提炼项目亮点, 包括:
- 项目亮点提炼
- 技术选型理由
- 量化成果展示

作品集是展示个人技术能力的重要载体。
"""

from typing import List, Dict
from dataclasses import dataclass, field

try:
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

import numpy as np


# ============================================================
# 1. 项目亮点提炼
# ============================================================

class PortfolioBuilder:
    """作品集构建器"""

    def __init__(self):
        self.project_name = "RAG知识库问答系统"
        self.highlights: List[Dict] = []
        self.tech_choices: List[Dict] = []
        self.metrics: List[Dict] = []
        self.skills: List[str] = []

    def add_highlight(self, title: str, description: str,
                       technologies: List[str] = None,
                       impact: str = ""):
        """添加项目亮点"""
        self.highlights.append({
            'title': title,
            'description': description,
            'technologies': technologies or [],
            'impact': impact,
        })

    def add_tech_choice(self, technology: str, reason: str,
                         alternatives: List[str] = None):
        """添加技术选型"""
        self.tech_choices.append({
            'technology': technology,
            'reason': reason,
            'alternatives': alternatives or [],
        })

    def add_metric(self, name: str, value: str, context: str = "",
                    before: str = ""):
        """添加量化成果"""
        self.metrics.append({
            'name': name,
            'value': value,
            'context': context,
            'before': before,
        })

    def generate_portfolio(self) -> str:
        """生成作品集内容"""
        lines = [
            f"# {self.project_name}",
            "",
            "## 项目简介",
            "基于检索增强生成(RAG)技术的企业级知识库问答系统。",
            "支持文档上传、智能检索、AI问答和流式输出。",
            "",
            "## 核心亮点",
            "",
        ]

        for i, hl in enumerate(self.highlights, 1):
            lines.append(f"### {i}. {hl['title']}")
            lines.append(f"{hl['description']}")
            if hl['impact']:
                lines.append(f"**影响:** {hl['impact']}")
            if hl['technologies']:
                lines.append(f"**技术:** {', '.join(hl['technologies'])}")
            lines.append("")

        lines.append("## 量化成果")
        lines.append("")
        lines.append("| 指标 | 数值 | 说明 |")
        lines.append("|------|------|------|")
        for m in self.metrics:
            before_str = f" (优化前: {m['before']})" if m['before'] else ""
            lines.append(f"| {m['name']} | {m['value']} | {m['context']}{before_str} |")

        lines.append("")
        lines.append("## 技术选型")
        lines.append("")
        for tc in self.tech_choices:
            lines.append(f"### {tc['technology']}")
            lines.append(f"**选型理由:** {tc['reason']}")
            if tc['alternatives']:
                lines.append(f"**备选方案:** {', '.join(tc['alternatives'])}")
            lines.append("")

        if self.skills:
            lines.append("## 技能标签")
            lines.append("")
            lines.append(", ".join(self.skills))

        return "\n".join(lines)

    def generate_elevator_pitch(self) -> str:
        """生成电梯演讲(30秒介绍)"""
        return f"""
项目名称: {self.project_name}

30秒介绍:
"我开发了一个基于RAG技术的知识库问答系统。系统采用混合检索策略，
结合BM25关键词检索和向量语义检索，通过RRF融合算法实现高精度文档召回。
检索准确率达到82%，端到端查询延迟控制在250ms以内。
系统支持多租户隔离、RBAC权限控制、审计日志等企业级功能，
并实现了完整的CI/CD流水线和Docker容器化部署。"

核心卖点:
1. 混合检索 + 重排序, 检索质量显著优于单一方法
2. 流式输出(SSE), 用户体验接近ChatGPT
3. 完整的企业级功能(多租户、权限、审计)
4. 全面的评估体系(检索+生成+用户反馈)
5. 生产级部署方案(Docker + CI/CD + 监控)
"""


# ============================================================
# 2. 技术深度分析
# ============================================================

class TechDepthAnalysis:
    """技术深度分析"""

    @staticmethod
    def analyze_retrieval_improvements() -> str:
        """检索优化分析"""
        return """
检索优化历程:
==========================================

阶段1: 基础向量检索
  - 使用TF-IDF作为向量表示
  - Recall@5: 45%
  - 问题: 无法理解语义, 同义词检索不到

阶段2: 引入BM25
  - 添加BM25关键词检索
  - Recall@5: 60%
  - 改进: 精确匹配效果好, 但语义理解仍不足

阶段3: 混合检索
  - BM25 + 向量检索 + RRF融合
  - Recall@5: 75%
  - 改进: 兼顾精确匹配和语义理解

阶段4: 添加重排序
  - Cross-encoder重排序
  - Recall@5: 82%
  - 改进: 排序质量大幅提升

阶段5: HyDE + 查询改写
  - 高级检索策略
  - Recall@5: 85%
  - 改进: 长尾查询效果提升

关键经验:
  1. 单一检索方法很难覆盖所有场景
  2. 混合检索是性价比最高的方案
  3. 重排序对最终效果影响最大
  4. 查询理解是容易被忽视但很关键的环节
"""

    @staticmethod
    def get_interview_questions() -> List[str]:
        """技术面试常见问题"""
        return [
            "为什么选择RAG而不是微调(Fine-tuning)?",
            "BM25和向量检���各自的优缺点是什么?",
            "如何解决LLM的幻觉问题?",
            "chunk_size如何选择? 太大太小有什么问题?",
            "如何评估RAG系统的质量?",
            "混合检索中RRF融合的原理是什么?",
            "如何处理用户查询与文档的语义鸿沟?",
            "多租户场景下如何保证数据隔离?",
            "流式输出(SSE)的实现原理?",
            "如何优化RAG系统的查询延迟?",
        ]


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("W34-D4 作品集")
    print("=" * 60)

    # --- 构建作品集 ---
    portfolio = PortfolioBuilder()

    # 亮点
    portfolio.add_highlight(
        "混合检索引擎",
        "实现了BM25+向量检索的混合方案, 使用RRF融合算法, "
        "检索Recall@5从45%提升至82%",
        technologies=["BM25", "向量检索", "RRF融合"],
        impact="检索质量提升82%, 显著改善回答准确性"
    )
    portfolio.add_highlight(
        "流式输出系统",
        "基于SSE协议实现流式输出, 支持FastAPI StreamingResponse, "
        "用户等待时间感知从2秒降至0.1秒",
        technologies=["SSE", "FastAPI", "异步IO"],
        impact="用户体验接近ChatGPT水平"
    )
    portfolio.add_highlight(
        "企业级安全架构",
        "实现了RBAC权限控制、审计日志、多租户数据隔离, "
        "通过了安全审查",
        technologies=["RBAC", "多租户", "审计日志"],
        impact="满足企业安全合规要求"
    )
    portfolio.add_highlight(
        "自动化评估体系",
        "构建了包含检索评估、生成评估和用户反馈的评估体系, "
        "实现了消融实验框架",
        technologies=["NDCG", "Faithfulness", "A/B测试"],
        impact="系统优化有据可依, 迭代效率提升50%"
    )

    # 技术选型
    portfolio.add_tech_choice(
        "FastAPI", "高性能异步框架, 自动生成API文档, 类型提示完善",
        ["Flask", "Django"]
    )
    portfolio.add_tech_choice(
        "混合检索", "兼顾关键词精确匹配和语义理解",
        ["纯BM25", "纯向量检索"]
    )

    # 量化成果
    portfolio.add_metric("检索Recall@5", "82%", "混合检索+重排序", "45%")
    portfolio.add_metric("查询延迟(P95)", "520ms", "含缓存优化", "1200ms")
    portfolio.add_metric("缓存命中率", "45%", "LRU+TTL缓存", "0%")
    portfolio.add_metric("用户满意度", "78%", "4-5分占比", "-")
    portfolio.add_metric("NPS", "42", "净推荐值", "-")

    # 技能
    portfolio.skills = [
        "Python", "RAG", "LLM", "FastAPI", "Docker",
        "向量检索", "BM25", "PostgreSQL", "Redis",
        "CI/CD", "性能优化", "系统设计",
    ]

    # --- 输出作品集 ---
    print("\n--- 项目亮点 ---")
    content = portfolio.generate_portfolio()
    print(content[:1000])
    print(f"\n... (共{len(content)}字符)")

    # --- 电梯演讲 ---
    print(f"\n{'='*60}")
    print("--- 电梯演讲 ---")
    print(f"{'='*60}")
    print(portfolio.generate_elevator_pitch())

    # --- 技术深度 ---
    print(f"{'='*60}")
    print("--- 技术深度分析 ---")
    print(f"{'='*60}")
    print(TechDepthAnalysis.analyze_retrieval_improvements())

    # --- 面试问题 ---
    print("--- 技术面试准备 ---")
    for i, q in enumerate(TechDepthAnalysis.get_interview_questions(), 1):
        print(f"  {i}. {q}")

    # --- 可视化 ---
    if HAS_PLT:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 检索优化历程
        stages = ['基础向量', '+BM25', '+混合', '+重排序', '+HyDE']
        recalls = [45, 60, 75, 82, 85]
        ax = axes[0]
        ax.plot(stages, recalls, 'o-', color='#3498db', linewidth=2, markersize=10)
        ax.set_ylabel('Recall@5 (%)')
        ax.set_title('检索优化历程', fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        for i, (s, r) in enumerate(zip(stages, recalls)):
            ax.annotate(f'{r}%', (s, r), textcoords="offset points",
                        xytext=(0, 10), ha='center', fontsize=10)

        # 量化成果
        metrics_names = ['Recall@5', '满意度', '缓存命中率']
        before_vals = [45, 0, 0]
        after_vals = [82, 78, 45]
        x = np.arange(len(metrics_names))
        width = 0.35
        ax = axes[1]
        ax.bar(x - width/2, before_vals, width, label='优化前', color='#e74c3c')
        ax.bar(x + width/2, after_vals, width, label='优化后', color='#2ecc71')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names)
        ax.set_ylabel('百分比(%)')
        ax.set_title('优化效果对比', fontweight='bold')
        ax.legend()

        plt.tight_layout()
        plt.savefig('D:/code/big-model-learn/code/q_01/W34/d4_portfolio.png', dpi=150)
        print("\n图表已保存为 d4_portfolio.png")
        plt.close()

    print("\n完成!")
