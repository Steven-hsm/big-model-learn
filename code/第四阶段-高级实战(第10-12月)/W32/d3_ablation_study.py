"""
W32-D3 消融实验
================
通过消融实验对比不同配置对RAG系统性能的影响:
- chunk_size 对比
- top_k 对比
- embedding方法对比
- 实验设计与结果分析

消融实验帮助理解每个组件对整体性能的贡献。
"""

import time
import json
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass, field

try:
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    HAS_PLT = True
except ImportError:
    HAS_PLT = False


# ============================================================
# 1. 实验框架
# ============================================================

@dataclass
class ExperimentConfig:
    """实验配置"""
    name: str
    params: Dict
    description: str = ""


@dataclass
class ExperimentResult:
    """实验结果"""
    config_name: str
    metrics: Dict
    details: Dict = field(default_factory=dict)


class AblationStudy:
    """消融实验框架"""

    def __init__(self):
        self.configs: List[ExperimentConfig] = []
        self.results: List[ExperimentResult] = []

    def add_config(self, name: str, params: Dict, description: str = ""):
        self.configs.append(ExperimentConfig(name, params, description))

    def run_experiment(self, config: ExperimentConfig,
                        eval_func) -> ExperimentResult:
        """运行单个实验配置"""
        metrics = eval_func(config.params)
        result = ExperimentResult(
            config_name=config.name,
            metrics=metrics,
            details={'params': config.params}
        )
        self.results.append(result)
        return result

    def run_all(self, eval_func):
        """运行所有实验"""
        self.results = []
        for config in self.configs:
            print(f"  运行: {config.name}...")
            self.run_experiment(config, eval_func)
        print(f"  完成! 共{len(self.results)}个实验")

    def print_report(self):
        """打印实验报告"""
        print(f"\n{'='*70}")
        print(f"{'消融实验报告':^70}")
        print(f"{'='*70}")

        # 表头
        if not self.results:
            print("暂无结果")
            return

        metric_names = list(self.results[0].metrics.keys())
        header = f"{'配置':<20}" + "".join(f"{m:<12}" for m in metric_names)
        print(header)
        print("-" * len(header))

        for result in self.results:
            row = f"{result.config_name:<20}"
            for m in metric_names:
                value = result.metrics.get(m, 0)
                row += f"{value:<12.4f}"
            print(row)


# ============================================================
# 2. 模拟RAG系统(用于实验)
# ============================================================

class SimulatedRAG:
    """模拟RAG系统, 用于消融实验"""

    # 模拟知识库
    DOCUMENTS = [
        "RAG系统通过检索增强生成, 结合了信息检索和文本生成的优势。",
        "向量检索使用语义嵌入来匹配查询和文档, 比关键词检索更智能。",
        "BM25是一种经典的信息检索算法, 基于词频和文档频率计算相关性。",
        "大语言模型通过海量文本训练, 具有强大的理解和生成能力。",
        "分块策略影响检索质量: 过大的块包含过多噪声, 过小的块缺乏上下文。",
        "重排序器可以对初步检索结果进行精排序, 提高最终结果的相关性。",
        "Prompt工程是设计有效提示词的技术, 对LLM输出质量有重要影响。",
        "Python的NumPy库提供高效的多维数组运算, 是数据科学的基础工具。",
        "深度学习使用多层神经网络自动学习特征表示, 在多个领域取得突破。",
        "自然语言处理研究计算机理解和生成人类语言的技术和方法。",
    ]

    EVAL_QUERIES = [
        {"query": "什么是RAG?", "relevant": {0, 1}},
        {"query": "如何优化检索?", "relevant": {1, 2, 4, 5}},
        {"query": "Python有哪些AI库?", "relevant": {7}},
        {"query": "深度学习的原理?", "relevant": {8, 9}},
        {"query": "如何设计Prompt?", "relevant": {6, 3}},
    ]

    def evaluate_config(self, params: Dict) -> Dict:
        """评估给定配置的性能"""
        chunk_size = params.get('chunk_size', 200)
        top_k = params.get('top_k', 5)
        method = params.get('method', 'hybrid')

        np.random.seed(42)

        # 模拟不同配置下的检索质量
        recalls = []
        precisions = []
        ndcgs = []

        for q_data in self.EVAL_QUERIES:
            relevant = q_data['relevant']

            # 根据方法模拟不同的检索效果
            if method == 'bm25':
                base_perf = 0.6
            elif method == 'vector':
                base_perf = 0.7
            elif method == 'hybrid':
                base_perf = 0.8
            else:
                base_perf = 0.5

            # chunk_size影响
            chunk_factor = 1.0 - abs(chunk_size - 300) / 1000
            chunk_factor = max(0.3, min(1.0, chunk_factor))

            # top_k影响
            topk_factor = min(1.0, top_k / len(relevant))

            # 计算指标
            recall = min(1.0, base_perf * chunk_factor * topk_factor + np.random.normal(0, 0.05))
            precision = min(1.0, base_perf * chunk_factor * (5 / max(top_k, 1)) + np.random.normal(0, 0.05))
            ndcg = min(1.0, base_perf * chunk_factor + np.random.normal(0, 0.05))

            recalls.append(max(0, recall))
            precisions.append(max(0, precision))
            ndcgs.append(max(0, ndcg))

        return {
            'Recall@K': np.mean(recalls),
            'Precision@K': np.mean(precisions),
            'NDCG@K': np.mean(ndcgs),
            'Avg_Latency_ms': 50 + (top_k * 5) + (0 if method == 'bm25' else 20),
        }


# ============================================================
# 3. 预定义实验
# ============================================================

def create_chunk_size_experiment() -> AblationStudy:
    """创建chunk_size消融实验"""
    study = AblationStudy()
    for size in [100, 200, 300, 500, 800, 1000]:
        study.add_config(
            f"chunk={size}",
            {'chunk_size': size, 'top_k': 5, 'method': 'hybrid'},
            f"分块大小={size}字"
        )
    return study


def create_topk_experiment() -> AblationStudy:
    """创建top_k消融实验"""
    study = AblationStudy()
    for k in [1, 3, 5, 7, 10, 15, 20]:
        study.add_config(
            f"top_k={k}",
            {'chunk_size': 300, 'top_k': k, 'method': 'hybrid'},
            f"检索文档数={k}"
        )
    return study


def create_method_experiment() -> AblationStudy:
    """创建检索方法消融实验"""
    study = AblationStudy()
    for method in ['bm25', 'vector', 'hybrid']:
        study.add_config(
            method,
            {'chunk_size': 300, 'top_k': 5, 'method': method},
            f"检索方法={method}"
        )
    return study


def create_combined_experiment() -> AblationStudy:
    """创建组合消融实验"""
    study = AblationStudy()
    configs = [
        ("BM25_k3", {'method': 'bm25', 'top_k': 3, 'chunk_size': 200}),
        ("BM25_k5", {'method': 'bm25', 'top_k': 5, 'chunk_size': 200}),
        ("向量_k3", {'method': 'vector', 'top_k': 3, 'chunk_size': 300}),
        ("向量_k5", {'method': 'vector', 'top_k': 5, 'chunk_size': 300}),
        ("混合_k3", {'method': 'hybrid', 'top_k': 3, 'chunk_size': 300}),
        ("混合_k5", {'method': 'hybrid', 'top_k': 5, 'chunk_size': 300}),
        ("混合_k10", {'method': 'hybrid', 'top_k': 10, 'chunk_size': 500}),
    ]
    for name, params in configs:
        study.add_config(name, params)
    return study


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("W32-D3 消融实验")
    print("=" * 60)

    rag = SimulatedRAG()

    # --- 1. chunk_size实验 ---
    print("\n--- 1. chunk_size消融实验 ---")
    chunk_study = create_chunk_size_experiment()
    chunk_study.run_all(rag.evaluate_config)
    chunk_study.print_report()

    # --- 2. top_k实验 ---
    print(f"\n--- 2. top_k消融实验 ---")
    topk_study = create_topk_experiment()
    topk_study.run_all(rag.evaluate_config)
    topk_study.print_report()

    # --- 3. 检索方法实验 ---
    print(f"\n--- 3. 检索方法消融实验 ---")
    method_study = create_method_experiment()
    method_study.run_all(rag.evaluate_config)
    method_study.print_report()

    # --- 4. 组合实验 ---
    print(f"\n--- 4. 组合消融实验 ---")
    combined_study = create_combined_experiment()
    combined_study.run_all(rag.evaluate_config)
    combined_study.print_report()

    # --- 可视化 ---
    if HAS_PLT:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # chunk_size
        ax = axes[0, 0]
        sizes = [100, 200, 300, 500, 800, 1000]
        metrics = ['Recall@K', 'Precision@K', 'NDCG@K']
        for metric in metrics:
            values = [r.metrics[metric] for r in chunk_study.results]
            ax.plot(sizes, values, 'o-', label=metric, linewidth=2)
        ax.set_xlabel('chunk_size')
        ax.set_ylabel('分数')
        ax.set_title('chunk_size对检索质量的影响', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # top_k
        ax = axes[0, 1]
        ks = [1, 3, 5, 7, 10, 15, 20]
        for metric in metrics:
            values = [r.metrics[metric] for r in topk_study.results]
            ax.plot(ks, values, 'o-', label=metric, linewidth=2)
        ax.set_xlabel('top_k')
        ax.set_ylabel('分数')
        ax.set_title('top_k对检索质量的影响', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 检索方法
        ax = axes[1, 0]
        methods = ['bm25', 'vector', 'hybrid']
        x = range(len(methods))
        width = 0.25
        for i, metric in enumerate(metrics):
            values = [r.metrics[metric] for r in method_study.results]
            ax.bar([pos + i * width for pos in x], values, width, label=metric)
        ax.set_xticks([pos + width for pos in x])
        ax.set_xticklabels(methods)
        ax.set_title('检索方法对比', fontweight='bold')
        ax.legend()

        # 延迟对比
        ax = axes[1, 1]
        names = [r.config_name for r in combined_study.results]
        latencies = [r.metrics['Avg_Latency_ms'] for r in combined_study.results]
        ax.barh(names, latencies, color='#e74c3c')
        ax.set_xlabel('延迟(ms)')
        ax.set_title('各配置延迟对比', fontweight='bold')

        plt.tight_layout()
        plt.savefig('D:/code/big-model-learn/code/q_01/W32/d3_ablation_study.png', dpi=150)
        print("\n图表已保存为 d3_ablation_study.png")
        plt.close()

    print("\n完成!")
