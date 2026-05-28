"""
W32-D7 项目报告生成
====================
自动生成RAG项目报告, 包括:
- 技术架构描述
- 性能指标汇总
- 改进方向建议

项目报告是项目成果的总结展示。
"""

import json
import time
from typing import Dict, List
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
# 1. 报告数据模型
# ============================================================

@dataclass
class SystemMetrics:
    """系统性能指标"""
    recall_at_5: float = 0.0
    precision_at_5: float = 0.0
    ndcg_at_5: float = 0.0
    mrr: float = 0.0
    faithfulness: float = 0.0
    relevance: float = 0.0
    fluency: float = 0.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    cache_hit_rate: float = 0.0
    user_satisfaction: float = 0.0
    nps_score: float = 0.0
    total_queries: int = 0
    total_documents: int = 0


# ============================================================
# 2. 报告生成器
# ============================================================

class ProjectReportGenerator:
    """项目报告生成器"""

    def __init__(self):
        self.project_name = "RAG知识库问答系统"
        self.version = "1.0.0"
        self.metrics = SystemMetrics()
        self.tech_stack = []
        self.features = []
        self.challenges = []
        self.improvements = []

    def set_project_info(self, name: str, version: str):
        self.project_name = name
        self.version = version

    def set_metrics(self, metrics: SystemMetrics):
        self.metrics = metrics

    def add_tech(self, name: str, purpose: str):
        self.tech_stack.append((name, purpose))

    def add_feature(self, feature: str):
        self.features.append(feature)

    def add_challenge(self, challenge: str, solution: str):
        self.challenges.append((challenge, solution))

    def add_improvement(self, area: str, description: str, priority: str):
        self.improvements.append((area, description, priority))

    def generate_technical_architecture(self) -> str:
        """生成技术架构描述"""
        return f"""
## 技术架构

### 系统架构图(文本版)
```
用户 --> [Gradio/Web界面] --> [FastAPI API层]
                                    |
                    +---------------+---------------+
                    |               |               |
            [检索模块]        [生成模块]        [管理模块]
            |   |   |         |    |          |    |
          BM25 向量 重排   Prompt LLM      文档  会话
            |   |              |              |
          [向量DB]         [LLM API]      [PostgreSQL]
            |
          [Redis缓存]
```

### 核心组件

1. 检索模块
   - BM25关键词检索: 基于词频的经典信息检索
   - 向量语义检索: 使用embedding模型进行语义匹配
   - 混合检索: RRF融合BM25和向量检索结果
   - 重排序: Cross-encoder对候选结果精排序

2. 生成模块
   - Prompt模板管理: 支持变量插值和多模板切换
   - 上下文窗口管理: Token预算分配和上下文裁剪
   - 流式输出: SSE协议支持实时内容推送
   - 引用溯源: 标注回答的信息来源

3. API服务
   - 查询接口: POST /api/query
   - 文档管理: CRUD操作
   - 健康检查: GET /api/health
   - Pydantic数据验证

4. 前端界面
   - Gradio聊天界面
   - 文档上传组件
   - 参数调节面板

5. 部署运维
   - Docker容器化
   - docker-compose编排
   - GitHub Actions CI/CD
   - 监控和日志系统
"""

    def generate_performance_section(self) -> str:
        """生成性能指标章节"""
        m = self.metrics
        return f"""
## 性能指标

### 检索质量
| 指标 | 值 | 说明 |
|------|------|------|
| Recall@5 | {m.recall_at_5:.4f} | 前5个结果中相关文档的召回比例 |
| Precision@5 | {m.precision_at_5:.4f} | 前5个结果中相关文档的精确比例 |
| NDCG@5 | {m.ndcg_at_5:.4f} | 排序质量(考虑位置权重) |
| MRR | {m.mrr:.4f} | 第一个相关文档的平均排名倒数 |

### 生成质量
| 指标 | 值 | 说明 |
|------|------|------|
| 忠实度 | {m.faithfulness:.4f} | 回答基于上下文的程度 |
| 相关性 | {m.relevance:.4f} | 回答与问题的相关程度 |
| 流畅度 | {m.fluency:.4f} | 回答的可读性和语言质量 |

### 系统性能
| 指标 | 值 | 说明 |
|------|------|------|
| 平均延迟 | {m.avg_latency_ms:.1f}ms | 端到端查询延迟 |
| P95延迟 | {m.p95_latency_ms:.1f}ms | 95%的请求在此时间内完成 |
| 缓存命中率 | {m.cache_hit_rate:.1%} | 缓存命中比例 |

### 用户满意度
| 指标 | 值 | 说明 |
|------|------|------|
| 满意度 | {m.user_satisfaction:.1%} | 4-5分评价占比 |
| NPS | {m.nps_score:.0f} | 净推荐值 |
| 总查询数 | {m.total_queries} | 累计查询次数 |
| 总文档数 | {m.total_documents} | 知识库文档数 |
"""

    def generate_improvement_section(self) -> str:
        """生成改进方向章节"""
        lines = [
            "## 改进方向",
            "",
        ]

        priority_labels = {
            'high': '高优先级',
            'medium': '中优先级',
            'low': '低优先级',
        }

        for area, desc, priority in self.improvements:
            lines.append(f"### {area} [{priority_labels.get(priority, priority)}]")
            lines.append(f"- {desc}")
            lines.append("")

        return "\n".join(lines)

    def generate_full_report(self) -> str:
        """生成完整报告"""
        sections = [
            f"# {self.project_name} 项目报告",
            f"版本: {self.version}",
            f"日期: {time.strftime('%Y-%m-%d')}",
            "",
            "---",
            "",
            "## 项目概述",
            f"{self.project_name}是一个基于检索增强生成(RAG)技术的智能知识库问答系统。",
            "系统支持文档上传、智能检索、AI问答和流式输出等功能。",
            "",
            "### 功能特性",
        ]

        for f in self.features:
            sections.append(f"- {f}")

        sections.append("")
        sections.append("### 技术栈")
        sections.append("")
        sections.append("| 技术 | 用途 |")
        sections.append("|------|------|")
        for name, purpose in self.tech_stack:
            sections.append(f"| {name} | {purpose} |")

        sections.append(self.generate_technical_architecture())
        sections.append(self.generate_performance_section())

        if self.challenges:
            sections.append("## 挑战与解决方案")
            for challenge, solution in self.challenges:
                sections.append(f"- **{challenge}**: {solution}")

        sections.append(self.generate_improvement_section())

        sections.append("")
        sections.append("---")
        sections.append("*本报告由项目报告生成器自动生成*")

        return "\n".join(sections)


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("W32-D7 项目报告生成")
    print("=" * 60)

    # --- 创建报告 ---
    generator = ProjectReportGenerator()
    generator.set_project_info("RAG知识库问答系统", "1.0.0")

    # 设置性能指标
    metrics = SystemMetrics(
        recall_at_5=0.8234,
        precision_at_5=0.7512,
        ndcg_at_5=0.7856,
        mrr=0.8567,
        faithfulness=0.8923,
        relevance=0.8645,
        fluency=0.9123,
        avg_latency_ms=245.5,
        p95_latency_ms=520.3,
        cache_hit_rate=0.45,
        user_satisfaction=0.78,
        nps_score=42.0,
        total_queries=1580,
        total_documents=256,
    )
    generator.set_metrics(metrics)

    # 技术栈
    techs = [
        ("Python 3.11", "开发语言"),
        ("FastAPI", "Web框架"),
        ("Gradio", "前端界面"),
        ("sentence-transformers", "文本嵌入"),
        ("FAISS", "向量检索"),
        ("PostgreSQL", "数据存储"),
        ("Redis", "缓存"),
        ("Docker", "容器化部署"),
    ]
    for name, purpose in techs:
        generator.add_tech(name, purpose)

    # 功能特性
    features = [
        "混合检索: BM25+向量+RRF融合",
        "重排序: Cross-encoder精排序",
        "流式输出: SSE实时推送",
        "多轮对话: 上下文压缩和摘要",
        "文档管理: 上传、分块、索引",
        "引用溯源: 标注信息来源",
        "监控告警: 性能指标和错误追踪",
    ]
    for f in features:
        generator.add_feature(f)

    # 挑战
    challenges = [
        ("检索精度不足", "引入混合检索和重排序, Recall@5提升15%"),
        ("长上下文丢失", "实现上下文压缩和摘要策略"),
        ("延迟过高", "添加缓存层, 平均延迟从450ms降至245ms"),
        ("幻觉问题", "优化Prompt和忠实度检查, 幻觉率降低60%"),
    ]
    for c, s in challenges:
        generator.add_challenge(c, s)

    # 改进方向
    improvements = [
        ("多模态支持", "支持图片、表格等非文本内容的检索和问答", "high"),
        ("多语言", "支持英文、日文等多语言文档和查询", "medium"),
        ("Agent能力", "让系统能执行工具调用、代码生成等复杂任务", "medium"),
        ("个性化", "基于用户历史调整检索策略和回答风格", "low"),
        ("知识图谱", "构建知识图谱增强结构化知识问答", "medium"),
    ]
    for area, desc, priority in improvements:
        generator.add_improvement(area, desc, priority)

    # --- 生成报告 ---
    report = generator.generate_full_report()
    print(report[:2000])
    print(f"\n... (完整报告共{len(report)}字符)")

    # --- 可视化仪表盘 ---
    if HAS_PLT:
        fig = plt.figure(figsize=(14, 8))

        # 性能指标仪表盘
        ax1 = fig.add_subplot(231)
        quality_metrics = ['Recall@5', 'Precision@5', 'NDCG@5', 'MRR']
        quality_values = [metrics.recall_at_5, metrics.precision_at_5,
                          metrics.ndcg_at_5, metrics.mrr]
        colors = ['#3498db' if v > 0.7 else '#e74c3c' for v in quality_values]
        ax1.bar(quality_metrics, quality_values, color=colors)
        ax1.set_ylim(0, 1)
        ax1.set_title('检索质量指标', fontweight='bold')
        ax1.tick_params(axis='x', rotation=20)

        # 生成质量
        ax2 = fig.add_subplot(232)
        gen_metrics = ['忠实度', '相关性', '流畅度']
        gen_values = [metrics.faithfulness, metrics.relevance, metrics.fluency]
        ax2.bar(gen_metrics, gen_values, color=['#2ecc71', '#3498db', '#9b59b6'])
        ax2.set_ylim(0, 1)
        ax2.set_title('生成质量指标', fontweight='bold')

        # 延迟分布
        ax3 = fig.add_subplot(233)
        ax3.bar(['平均延迟', 'P95延迟'],
                [metrics.avg_latency_ms, metrics.p95_latency_ms],
                color=['#2ecc71', '#f39c12'])
        ax3.set_ylabel('ms')
        ax3.set_title('查询延迟', fontweight='bold')

        # 用户满意度
        ax4 = fig.add_subplot(234)
        ax4.pie([metrics.user_satisfaction, 1-metrics.user_satisfaction],
                labels=['满意', '不满意'],
                autopct='%1.0f%%', colors=['#2ecc71', '#e74c3c'])
        ax4.set_title(f'用户满意度 ({metrics.user_satisfaction:.0%})', fontweight='bold')

        # 改进优先级
        ax5 = fig.add_subplot(235)
        priorities = {'high': 0, 'medium': 0, 'low': 0}
        for _, _, p in improvements:
            priorities[p] += 1
        labels = ['高优先级', '中优先级', '低优先级']
        values = [priorities['high'], priorities['medium'], priorities['low']]
        ax5.bar(labels, values, color=['#e74c3c', '#f39c12', '#2ecc71'])
        ax5.set_title('改进项优先级', fontweight='bold')

        # 综合评分
        ax6 = fig.add_subplot(236)
        overall_scores = {
            '检索': np.mean([metrics.recall_at_5, metrics.precision_at_5, metrics.ndcg_at_5]),
            '生成': np.mean([metrics.faithfulness, metrics.relevance, metrics.fluency]),
            '性能': max(0, 1 - metrics.avg_latency_ms / 1000),
            '满意度': metrics.user_satisfaction,
        }
        ax6.barh(list(overall_scores.keys()), list(overall_scores.values()),
                  color='#3498db')
        ax6.set_xlim(0, 1)
        ax6.set_title('综合评分', fontweight='bold')

        plt.suptitle('RAG系统项目报告 - 性能仪表盘', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('D:/code/big-model-learn/code/q_01/W32/d7_final_report.png', dpi=150)
        print("\n图表已保存为 d7_final_report.png")
        plt.close()

    print("\n完成!")
