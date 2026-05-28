"""
W32-D4 错误分析
================
分析RAG系统的失败模式, 包括:
- 失败模式分类
- 错误根因分析
- 改进建议生成

错误分析是持续改进系统的关键步骤。
"""

import json
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter

try:
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

import numpy as np


# ============================================================
# 1. 失败模式分类
# ============================================================

class FailureMode:
    """失败模式枚举"""
    NO_RELEVANT_DOC = "无相关文档"         # 知识库中缺少相关信息
    POOR_RETRIEVAL = "检索质量差"           # 相关文档存在但未被检索到
    LOST_IN_MIDDLE = "中间位置丢失"         # 相关信息在长上下文中间被忽略
    HALLUCINATION = "幻觉"                  # 编造了不存在的信息
    IRRELEVANT_ANSWER = "回答不相关"         # 回答偏离了问题
    INCOMPLETE = "回答不完整"               # 遗漏了重要信息
    WRONG_EXTRACTION = "信息提取错误"        # 从上下文中提取了错误信息
    FORMAT_ERROR = "格式错误"               # 输出格式不符合预期
    OVER_CONFIDENT = "过度自信"             # 对错误信息表示高确信度
    CONTEXT_OVERFLOW = "上下文溢出"          # 超出上下文窗口限制


@dataclass
class ErrorCase:
    """错误案例"""
    case_id: str
    question: str
    expected_answer: str
    actual_answer: str
    context: str = ""
    failure_mode: str = ""
    severity: str = "medium"  # low, medium, high, critical
    root_cause: str = ""
    improvement: str = ""


# ============================================================
# 2. 错误检测器
# ============================================================

class ErrorDetector:
    """错误检测器"""

    def detect_failures(self, question: str, expected: str,
                         actual: str, context: str = "") -> List[ErrorCase]:
        """检测一个问答对中的失败模式"""
        cases = []

        # 1. 检查无相关文档
        if not actual or len(actual) < 10:
            cases.append(ErrorCase(
                case_id=f"err_{hash(question) % 10000:04d}",
                question=question,
                expected_answer=expected,
                actual_answer=actual,
                context=context,
                failure_mode=FailureMode.NO_RELEVANT_DOC,
                severity="high",
            ))

        # 2. 检查幻觉
        if context and actual:
            hallucination_score = self._check_hallucination(actual, context)
            if hallucination_score > 0.5:
                cases.append(ErrorCase(
                    case_id=f"hall_{hash(question) % 10000:04d}",
                    question=question,
                    expected_answer=expected,
                    actual_answer=actual,
                    context=context,
                    failure_mode=FailureMode.HALLUCINATION,
                    severity="critical",
                    root_cause=f"回答中{hallucination_score:.0%}的内容无上下文支撑",
                ))

        # 3. 检查相关性
        relevance = self._check_relevance(question, actual)
        if relevance < 0.3:
            cases.append(ErrorCase(
                case_id=f"irr_{hash(question) % 10000:04d}",
                question=question,
                expected_answer=expected,
                actual_answer=actual,
                context=context,
                failure_mode=FailureMode.IRRELEVANT_ANSWER,
                severity="high",
                root_cause=f"回答与问题的相关性仅{relevance:.0%}",
            ))

        # 4. 检查完整性
        if expected and actual:
            completeness = self._check_completeness(expected, actual)
            if completeness < 0.5:
                cases.append(ErrorCase(
                    case_id=f"inc_{hash(question) % 10000:04d}",
                    question=question,
                    expected_answer=expected,
                    actual_answer=actual,
                    context=context,
                    failure_mode=FailureMode.INCOMPLETE,
                    severity="medium",
                    root_cause=f"回答仅覆盖了期望内容的{completeness:.0%}",
                ))

        return cases

    def _check_hallucination(self, answer: str, context: str) -> float:
        """检查幻觉程度"""
        ans_words = set(re.findall(r'\w+', answer.lower()))
        ctx_words = set(re.findall(r'\w+', context.lower()))

        if not ans_words:
            return 0.0

        unsupported = ans_words - ctx_words
        return len(unsupported) / len(ans_words)

    def _check_relevance(self, question: str, answer: str) -> float:
        """检查相关性"""
        q_words = set(re.findall(r'\w+', question.lower()))
        a_words = set(re.findall(r'\w+', answer.lower()))

        if not q_words:
            return 0.5

        overlap = q_words & a_words
        return len(overlap) / len(q_words)

    def _check_completeness(self, expected: str, actual: str) -> float:
        """检查完整性"""
        exp_words = set(re.findall(r'\w+', expected.lower()))
        act_words = set(re.findall(r'\w+', actual.lower()))

        if not exp_words:
            return 1.0

        covered = exp_words & act_words
        return len(covered) / len(exp_words)


# ============================================================
# 3. 错误分析器
# ============================================================

class ErrorAnalyzer:
    """错误分析器"""

    def __init__(self):
        self.detector = ErrorDetector()
        self.all_cases: List[ErrorCase] = []

    def analyze_batch(self, qa_pairs: List[Dict]) -> List[ErrorCase]:
        """批量分析问答对"""
        for pair in qa_pairs:
            cases = self.detector.detect_failures(
                question=pair['question'],
                expected=pair.get('expected', ''),
                actual=pair['answer'],
                context=pair.get('context', ''),
            )
            self.all_cases.extend(cases)
        return self.all_cases

    def get_failure_distribution(self) -> Dict[str, int]:
        """获取失败模式分布"""
        dist = Counter(case.failure_mode for case in self.all_cases)
        return dict(dist)

    def get_severity_distribution(self) -> Dict[str, int]:
        """获取严重度分布"""
        dist = Counter(case.severity for case in self.all_cases)
        return dict(dist)

    def get_root_cause_analysis(self) -> Dict:
        """根因分析"""
        causes = defaultdict(list)
        for case in self.all_cases:
            if case.root_cause:
                causes[case.failure_mode].append(case.root_cause)

        return dict(causes)

    def generate_improvement_suggestions(self) -> List[Dict]:
        """生成改进建议"""
        dist = self.get_failure_distribution()

        suggestions = {
            FailureMode.NO_RELEVANT_DOC: {
                'description': '知识库缺少相关信息',
                'actions': [
                    '扩充知识库, 补充相关领域的文档',
                    '优化文档分块策略, 确保关键信息不被截断',
                    '添加数据源, 覆盖更多用户可能提问的主题',
                ],
            },
            FailureMode.POOR_RETRIEVAL: {
                'description': '检索质量不足',
                'actions': [
                    '使用更好的embedding模型(如BGE-large)',
                    '尝试混合检索(BM25+向量)',
                    '添加重排序器提高检索精度',
                    '调整chunk_size和overlap参数',
                ],
            },
            FailureMode.HALLUCINATION: {
                'description': '模型编造信息',
                'actions': [
                    '加强Prompt中的约束: "只基于上下文回答"',
                    '添加忠实度检查后处理步骤',
                    '降低temperature参数',
                    '使用更强的模型或增加上下文长度',
                ],
            },
            FailureMode.IRRELEVANT_ANSWER: {
                'description': '回答不切题',
                'actions': [
                    '优化Query改写, 使查询更明确',
                    '在Prompt中要求直接回答问题',
                    '添加问题分类模块, 区分不同类型的问题',
                ],
            },
            FailureMode.INCOMPLETE: {
                'description': '回答不完整',
                'actions': [
                    '增加top_k, 检索更多相关文档',
                    '在Prompt中要求全面回答',
                    '使用Chain-of-Thought引导多角度分析',
                ],
            },
        }

        result = []
        for mode, count in dist.items():
            if mode in suggestions:
                result.append({
                    'failure_mode': mode,
                    'count': count,
                    **suggestions[mode],
                })

        return result

    def print_report(self):
        """打印分析报告"""
        print(f"\n{'='*60}")
        print("错误分析报告")
        print(f"{'='*60}")

        print(f"\n总错误案例: {len(self.all_cases)}")

        # 失败模式分布
        print(f"\n[失败模式分布]")
        for mode, count in self.get_failure_distribution().items():
            bar = '*' * count
            print(f"  {mode:<15s}: {count:3d} {bar}")

        # 严重度分布
        print(f"\n[严重度分布]")
        for sev, count in self.get_severity_distribution().items():
            print(f"  {sev:<10s}: {count}")

        # 改进建议
        print(f"\n[改进建议]")
        for suggestion in self.generate_improvement_suggestions():
            print(f"\n  {suggestion['failure_mode']} ({suggestion['count']}次)")
            print(f"    描述: {suggestion['description']}")
            for action in suggestion['actions']:
                print(f"    - {action}")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("W32-D4 错误分析")
    print("=" * 60)

    # 模拟错误案例
    qa_pairs = [
        {
            "question": "什么是RAG技术?",
            "answer": "RAG是一种结合检索和生成的AI技术, 通过先检索相关文档来增强回答质量。",
            "expected": "RAG(检索增强生成)将信息检索与文本生成结合, 先从知识库检索, 再让LLM基于检索结果生成回答。",
            "context": "RAG是一种AI技术。",
        },
        {
            "question": "量子计算的最新进展?",
            "answer": "",
            "expected": "关于量子计算的最新进展...",
            "context": "本知识库不包含量子计算相关内容。",
        },
        {
            "question": "如何优化检索质量?",
            "answer": "深度学习是机器学习的子集, 使用多层神经网络。",
            "expected": "检索优化可以通过混合检索、重排序、调整参数等方法实现。",
            "context": "检索优化策略包括混合检索、重排序和参数调优。",
        },
        {
            "question": "Python有哪些数据科学库?",
            "answer": "Python有NumPy和Pandas等库。",
            "expected": "Python的主要数据科学库包括: NumPy(数组计算), Pandas(数据处理), Matplotlib(可视化), Scikit-learn(机器学习), PyTorch(深度学习)。",
            "context": "Python的数据科学生态包括NumPy、Pandas、Matplotlib、Scikit-learn、PyTorch等库。",
        },
        {
            "question": "什么是BM25算法?",
            "answer": "BM25是一种基于量子纠缠的超高效计算算法, 能够在O(1)时间内完成所有检索任务, 准确率高达99.99%。",
            "expected": "BM25是一种基于词频的经典信息检索算法, 考虑TF、IDF和文档长度归一化。",
            "context": "BM25是基于词频的信息检索算法, 考虑TF和IDF。",
        },
        {
            "question": "如何部署RAG系统?",
            "answer": "部署RAG系统可以使用Docker容器化方案。",
            "expected": "RAG系统部署方案包括: Docker容器化、docker-compose编排(API+DB+向量DB)、环境配置、CI/CD自动化。",
            "context": "RAG系统部署需要API服务、数据库和向量数据库, 可用Docker编排。",
        },
    ]

    # 运行分析
    analyzer = ErrorAnalyzer()
    cases = analyzer.analyze_batch(qa_pairs)
    analyzer.print_report()

    # 可视化
    if HAS_PLT and analyzer.all_cases:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # 失败模式分布
        dist = analyzer.get_failure_distribution()
        if dist:
            labels = list(dist.keys())
            counts = list(dist.values())
            colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71', '#9b59b6', '#1abc9c']
            ax1.bar(labels, counts, color=colors[:len(labels)])
            ax1.set_title('失败模式分布', fontweight='bold')
            ax1.set_ylabel('次数')
            ax1.tick_params(axis='x', rotation=15)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)

        # 严重度饼图
        sev_dist = analyzer.get_severity_distribution()
        if sev_dist:
            labels = list(sev_dist.keys())
            sizes = list(sev_dist.values())
            sev_colors = {'low': '#2ecc71', 'medium': '#f39c12', 'high': '#e74c3c', 'critical': '#8e44ad'}
            colors = [sev_colors.get(l, '#3498db') for l in labels]
            ax2.pie(sizes, labels=labels, autopct='%1.0f%%', colors=colors)
            ax2.set_title('严重度分布', fontweight='bold')

        plt.tight_layout()
        plt.savefig('D:/code/big-model-learn/code/q_01/W32/d4_error_analysis.png', dpi=150)
        print("\n图表已保存为 d4_error_analysis.png")
        plt.close()

    print("\n完成!")
