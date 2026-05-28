"""
W33-D5 自动化评估套件
======================
实现RAG系统的自动化评估, 包括:
- 多维度评估
- 配置对比
- 报告生成

自动化评估让系统优化有据可依。
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
# 1. 评估数据集
# ============================================================

@dataclass
class EvalSample:
    """评估样本"""
    sample_id: str
    question: str
    expected_answer: str
    relevant_docs: List[str]
    context: str = ""
    category: str = "general"


def create_eval_dataset() -> List[EvalSample]:
    """创建评估数据集"""
    samples = [
        EvalSample("e1", "什么是RAG?",
                    "RAG是检索增强生成技术, 结合信息检索和文本生成",
                    ["doc_rag"], context="RAG是检索增强生成技术", category="概念"),
        EvalSample("e2", "BM25的原理是什么?",
                    "BM25基于词频TF和逆文档频率IDF计算相关性",
                    ["doc_bm25"], context="BM25是基于词频的检索算法", category="技术"),
        EvalSample("e3", "Python适合AI吗?",
                    "Python是AI首选语言, 拥有丰富的库生态",
                    ["doc_python"], context="Python在AI领域广泛应用", category="评估"),
        EvalSample("e4", "如何优化检索质量?",
                    "可通过混合检索、重排序、参数调优等方法",
                    ["doc_optimize"], context="检索优化方法包括混合检索和重排序", category="方法"),
        EvalSample("e5", "向量检索的原理?",
                    "向量检索通过计算查询和文档的语义相似度来匹配",
                    ["doc_vector"], context="向量检索使用语义嵌入匹配", category="技术"),
    ]
    return samples


# ============================================================
# 2. 评估指标计算
# ============================================================

class MetricCalculator:
    """指标计算器"""

    @staticmethod
    def keyword_overlap(prediction: str, reference: str) -> float:
        """关键词重叠度"""
        pred_words = set(prediction.lower().split())
        ref_words = set(reference.lower().split())
        if not ref_words:
            return 0.0
        return len(pred_words & ref_words) / len(ref_words)

    @staticmethod
    def context_support(answer: str, context: str) -> float:
        """上下文支撑度"""
        if not context:
            return 0.0
        ans_words = set(answer.lower().split())
        ctx_words = set(context.lower().split())
        if not ans_words:
            return 0.0
        return len(ans_words & ctx_words) / len(ans_words)

    @staticmethod
    def length_score(answer: str, min_len: int = 20, max_len: int = 500) -> float:
        """长度合理性"""
        length = len(answer)
        if length < min_len:
            return length / min_len
        elif length > max_len:
            return max(0.5, 1.0 - (length - max_len) / max_len)
        return 1.0

    @staticmethod
    def retrieval_accuracy(retrieved: List[str], relevant: List[str]) -> float:
        """检索准确率"""
        if not relevant:
            return 0.0
        hits = len(set(retrieved) & set(relevant))
        return hits / len(relevant)


# ============================================================
# 3. 自动评估器
# ============================================================

class AutoEvaluator:
    """自动化评估器"""

    def __init__(self):
        self.calculator = MetricCalculator()

    def evaluate_single(self, sample: EvalSample,
                         prediction: str,
                         retrieved_docs: List[str] = None) -> Dict:
        """评估单个样本"""
        metrics = {}

        # 检索评估
        if retrieved_docs is not None:
            metrics['retrieval_accuracy'] = self.calculator.retrieval_accuracy(
                retrieved_docs, sample.relevant_docs)
        else:
            metrics['retrieval_accuracy'] = 0.5  # 默认值

        # 生成评估
        metrics['keyword_overlap'] = self.calculator.keyword_overlap(
            prediction, sample.expected_answer)
        metrics['context_support'] = self.calculator.context_support(
            prediction, sample.context)
        metrics['length_score'] = self.calculator.length_score(prediction)

        # 综合分数
        metrics['overall'] = (
            metrics['keyword_overlap'] * 0.3 +
            metrics['context_support'] * 0.3 +
            metrics['length_score'] * 0.1 +
            metrics['retrieval_accuracy'] * 0.3
        )

        return metrics

    def evaluate_system(self, dataset: List[EvalSample],
                         system_func) -> Dict:
        """评估整个系统

        参数:
            dataset: 评估数据集
            system_func: 系统查询函数 (question -> {answer, docs})
        """
        results = []
        for sample in dataset:
            system_output = system_func(sample.question)
            metrics = self.evaluate_single(
                sample,
                prediction=system_output.get('answer', ''),
                retrieved_docs=system_output.get('docs', []),
            )
            results.append({
                'sample_id': sample.sample_id,
                'question': sample.question,
                'category': sample.category,
                **metrics,
            })

        return self._aggregate(results)

    def _aggregate(self, results: List[Dict]) -> Dict:
        """聚合评估结果"""
        if not results:
            return {}

        metric_names = [k for k in results[0].keys()
                        if k not in ('sample_id', 'question', 'category')]

        aggregated = {}
        for metric in metric_names:
            values = [r[metric] for r in results if metric in r]
            if values:
                aggregated[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'per_sample': values,
                }

        # 按类别聚合
        by_category = {}
        for r in results:
            cat = r.get('category', 'other')
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(r)

        category_metrics = {}
        for cat, cat_results in by_category.items():
            category_metrics[cat] = {
                metric: np.mean([r[metric] for r in cat_results if metric in r])
                for metric in metric_names if metric != 'per_sample'
                if any(metric in r for r in cat_results)
            }

        aggregated['by_category'] = category_metrics
        aggregated['total_samples'] = len(results)

        return aggregated


# ============================================================
# 4. 对比评估器
# ============================================================

class ComparisonEvaluator:
    """系统配置对比评估"""

    def __init__(self):
        self.evaluator = AutoEvaluator()
        self.comparisons: Dict[str, Dict] = {}

    def add_system(self, name: str, system_func):
        """添加待对比的系统"""
        self.comparisons[name] = system_func

    def run_comparison(self, dataset: List[EvalSample]) -> Dict:
        """运行对比评估"""
        results = {}
        for name, func in self.comparisons.items():
            print(f"  评估系统: {name}...")
            results[name] = self.evaluator.evaluate_system(dataset, func)
        return results

    def print_comparison(self, results: Dict):
        """打印对比结果"""
        print(f"\n{'='*70}")
        print("系统对比评估报告")
        print(f"{'='*70}")

        # 获取所有指标名
        metric_names = []
        for sys_result in results.values():
            for key in sys_result:
                if key not in ('by_category', 'total_samples', 'per_sample'):
                    if isinstance(sys_result[key], dict) and 'mean' in sys_result[key]:
                        metric_names.append(key)
            break

        # 表头
        header = f"{'指标':<20}" + "".join(f"{name:<15}" for name in results.keys())
        print(header)
        print("-" * len(header))

        for metric in metric_names:
            row = f"{metric:<20}"
            for name, sys_result in results.items():
                if metric in sys_result and isinstance(sys_result[metric], dict):
                    mean = sys_result[metric].get('mean', 0)
                    row += f"{mean:<15.4f}"
                else:
                    row += f"{'N/A':<15}"
            print(row)

        # 按类别
        print(f"\n[按类别对比]")
        categories = set()
        for sys_result in results.values():
            if 'by_category' in sys_result:
                categories.update(sys_result['by_category'].keys())

        for cat in sorted(categories):
            print(f"\n  类别: {cat}")
            for name, sys_result in results.items():
                if 'by_category' in sys_result and cat in sys_result['by_category']:
                    overall = sys_result['by_category'][cat].get('overall', 0)
                    print(f"    {name:<15}: {overall:.4f}")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("W33-D5 自动化评估套件")
    print("=" * 60)

    dataset = create_eval_dataset()
    print(f"评估数据集: {len(dataset)}个样本")
    for s in dataset:
        print(f"  [{s.category}] {s.question}")

    # --- 模拟不同系统 ---
    np.random.seed(42)

    def system_basic(question):
        """基础系统"""
        return {
            'answer': f"关于{question}的回答 (基础版)",
            'docs': ['doc_rag'] if 'RAG' in question else [],
        }

    def system_improved(question):
        """改进系统"""
        return {
            'answer': f"关于{question}的详细回答, 基于检索增强生成技术 (改进版)",
            'docs': ['doc_rag'] if 'RAG' in question else ['doc_other'],
        }

    def system_advanced(question):
        """高级系统"""
        return {
            'answer': f"{question}: 这是一个很好的问题。基于检索增强生成技术的原理和实践经验 (高级版)",
            'docs': ['doc_rag', 'doc_bm25'] if 'RAG' in question else ['doc_vector'],
        }

    # --- 对比评估 ---
    print(f"\n{'='*60}")
    print("--- 系统对比评估 ---")
    print(f"{'='*60}")

    comparator = ComparisonEvaluator()
    comparator.add_system("基础系统", system_basic)
    comparator.add_system("改进系统", system_improved)
    comparator.add_system("高级系统", system_advanced)

    results = comparator.run_comparison(dataset)
    comparator.print_comparison(results)

    # --- 可视化 ---
    if HAS_PLT:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        systems = list(results.keys())
        metrics_to_plot = ['keyword_overlap', 'context_support', 'retrieval_accuracy', 'overall']

        # 指标对比
        ax = axes[0]
        x = np.arange(len(metrics_to_plot))
        width = 0.25
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        for i, (sys_name, color) in enumerate(zip(systems, colors)):
            values = []
            for m in metrics_to_plot:
                if m in results[sys_name] and isinstance(results[sys_name][m], dict):
                    values.append(results[sys_name][m]['mean'])
                else:
                    values.append(0)
            ax.bar(x + i * width, values, width, label=sys_name, color=color)

        ax.set_xticks(x + width)
        ax.set_xticklabels(metrics_to_plot, rotation=15)
        ax.set_ylabel('分数')
        ax.set_title('各系统评估指标对比', fontweight='bold')
        ax.legend()
        ax.set_ylim(0, 1.2)

        # 雷达图(综合分)
        ax = axes[1]
        overall_scores = []
        for sys_name in systems:
            if 'overall' in results[sys_name] and isinstance(results[sys_name]['overall'], dict):
                overall_scores.append(results[sys_name]['overall']['mean'])
            else:
                overall_scores.append(0)

        bars = ax.bar(systems, overall_scores, color=colors[:len(systems)])
        for bar, val in zip(bars, overall_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center')
        ax.set_ylabel('综合分数')
        ax.set_title('系统综合评分', fontweight='bold')
        ax.set_ylim(0, 1.0)

        plt.tight_layout()
        plt.savefig('D:/code/big-model-learn/code/q_01/W33/d5_evaluation_suite.png', dpi=150)
        print("\n图表已保存为 d5_evaluation_suite.png")
        plt.close()

    print("\n完成!")
