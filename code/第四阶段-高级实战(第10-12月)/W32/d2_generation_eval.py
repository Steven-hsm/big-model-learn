"""
W32-D2 生成评估
================
实现RAG系统生成质量的评估, 包括:
- Faithfulness (忠实度)
- Relevance (相关性)
- Fluency (流畅度)
- 人工评估模板

生成评估衡量LLM回答的质量。
"""

import re
import math
from typing import List, Dict, Tuple, Optional
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
# 1. 忠实度评估 (Faithfulness)
# ============================================================

class FaithfulnessEvaluator:
    """忠实度评估器

    忠实度衡量回答是否基于提供的上下文, 而非编造信息。
    方法: 检查回答中的声明是否能在上下文中找到支撑。
    """

    def evaluate(self, answer: str, context: str) -> Dict:
        """评估回答的忠实度"""
        # 将回答分解为声明(sentences)
        claims = self._extract_claims(answer)
        if not claims:
            return {'score': 1.0, 'supported': 0, 'total': 0, 'details': []}

        # 检查每个声明是否有上下文支撑
        supported_count = 0
        details = []

        for claim in claims:
            support_score = self._check_support(claim, context)
            details.append({
                'claim': claim,
                'supported': support_score > 0.3,
                'score': support_score,
            })
            if support_score > 0.3:
                supported_count += 1

        faithfulness_score = supported_count / len(claims) if claims else 1.0

        return {
            'score': faithfulness_score,
            'supported': supported_count,
            'total': len(claims),
            'details': details,
        }

    def _extract_claims(self, text: str) -> List[str]:
        """从文本中提取声明(句子)"""
        sentences = re.split(r'[。！？\n]', text)
        return [s.strip() for s in sentences if len(s.strip()) > 5]

    def _check_support(self, claim: str, context: str) -> float:
        """检查声明是否有上下文支撑(基于关键词重叠)"""
        claim_words = set(re.findall(r'\w+', claim.lower()))
        context_words = set(re.findall(r'\w+', context.lower()))

        if not claim_words:
            return 0.0

        overlap = claim_words & context_words
        return len(overlap) / len(claim_words)


# ============================================================
# 2. 相关性评估 (Relevance)
# ============================================================

class RelevanceEvaluator:
    """相关性评估器

    相关性衡量回答是否切题, 是否直接回答了用户的问题。
    """

    def evaluate(self, question: str, answer: str) -> Dict:
        """评估回答的相关性"""
        q_words = set(re.findall(r'\w+', question.lower()))
        a_words = set(re.findall(r'\w+', answer.lower()))

        if not q_words:
            return {'score': 0.5, 'overlap': 0, 'question_words': 0}

        # 关键词重叠度
        overlap = q_words & a_words
        keyword_coverage = len(overlap) / len(q_words)

        # 回答长度合理性(太短或太长都扣分)
        answer_len = len(answer)
        if answer_len < 20:
            length_score = answer_len / 20
        elif answer_len > 1000:
            length_score = max(0.5, 1.0 - (answer_len - 1000) / 2000)
        else:
            length_score = 1.0

        # 直接性(回答开头是否直接回应问题)
        first_sentence = answer.split('。')[0] if '。' in answer else answer[:50]
        first_overlap = set(re.findall(r'\w+', first_sentence.lower())) & q_words
        directness = len(first_overlap) / max(len(q_words), 1)

        # 综合分数
        relevance_score = (
            keyword_coverage * 0.4 +
            length_score * 0.2 +
            directness * 0.4
        )

        return {
            'score': min(relevance_score, 1.0),
            'keyword_coverage': keyword_coverage,
            'length_score': length_score,
            'directness': directness,
            'overlap_words': len(overlap),
        }


# ============================================================
# 3. 流畅度评估 (Fluency)
# ============================================================

class FluencyEvaluator:
    """流畅度评估器

    流畅度衡量回答的可读性和语言质量。
    """

    def evaluate(self, answer: str) -> Dict:
        """评估回答的流畅度"""
        if not answer:
            return {'score': 0.0, 'length': 0}

        # 句子数量和平均长度
        sentences = re.split(r'[。！？\n]', answer)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return {'score': 0.0, 'length': len(answer)}

        avg_sentence_len = np.mean([len(s) for s in sentences])

        # 句子长度合理性(太长或太短都扣分)
        if avg_sentence_len < 5:
            sentence_score = avg_sentence_len / 5
        elif avg_sentence_len > 80:
            sentence_score = max(0.5, 1.0 - (avg_sentence_len - 80) / 100)
        else:
            sentence_score = 1.0

        # 结构化程度(有列表、标题等加分)
        has_structure = bool(re.search(r'[一二三四五六七八九十]、|首先|其次|最后|1\.|2\.', answer))
        structure_score = 1.0 if has_structure else 0.7

        # 标点使用
        punctuation_count = len(re.findall(r'[，。！？；：""''、]', answer))
        punct_ratio = punctuation_count / max(len(answer), 1)
        punct_score = min(1.0, punct_ratio * 20)  # 约5%的标点密度为最佳

        fluency_score = (
            sentence_score * 0.4 +
            structure_score * 0.3 +
            punct_score * 0.3
        )

        return {
            'score': min(fluency_score, 1.0),
            'sentence_count': len(sentences),
            'avg_sentence_length': avg_sentence_len,
            'has_structure': has_structure,
            'punctuation_density': punct_ratio,
        }


# ============================================================
# 4. 综合评估器
# ============================================================

class GenerationEvaluator:
    """生成质量综合评估器"""

    def __init__(self):
        self.faithfulness = FaithfulnessEvaluator()
        self.relevance = RelevanceEvaluator()
        self.fluency = FluencyEvaluator()

    def evaluate(self, question: str, answer: str,
                  context: str = None) -> Dict:
        """综合评估"""
        result = {
            'question': question,
            'answer_length': len(answer),
        }

        # 忠实度
        if context:
            result['faithfulness'] = self.faithfulness.evaluate(answer, context)
        else:
            result['faithfulness'] = {'score': None, 'note': '未提供上下文'}

        # 相关性
        result['relevance'] = self.relevance.evaluate(question, answer)

        # 流畅度
        result['fluency'] = self.fluency.evaluate(answer)

        # 综合分数
        scores = []
        if result['faithfulness']['score'] is not None:
            scores.append(result['faithfulness']['score'] * 0.4)
        scores.append(result['relevance']['score'] * 0.4)
        scores.append(result['fluency']['score'] * 0.2)

        result['overall_score'] = sum(scores) / len(scores) if scores else 0

        return result

    def batch_evaluate(self, eval_data: List[Dict]) -> List[Dict]:
        """批量评估"""
        results = []
        for item in eval_data:
            result = self.evaluate(
                question=item['question'],
                answer=item['answer'],
                context=item.get('context'),
            )
            results.append(result)
        return results


# ============================================================
# 5. 人工评估模板
# ============================================================

def generate_human_eval_template():
    """生成人工评估模板"""
    return """
# 人工评估表
# 评分标准: 1(差) - 5(优)

## 评估维度

### 1. 准确性 (1-5)
回答中的事实是否正确?
评分: ___

### 2. 完整性 (1-5)
回答是否完整地覆盖了问题的各个方面?
评分: ___

### 3. 相关性 (1-5)
回答是否直接针对用户的问题?
评分: ___

### 4. 忠实度 (1-5)
回答是否基于提供的上下文, 没有编造信息?
评分: ___

### 5. 可读性 (1-5)
回答的语言是否流畅、结构是否清晰?
评分: ___

### 6. 引用准确 (1-5)
引用的来源是否与回答内容一致?
评分: ___

## 总分: ___/30

## 具体意见
优点:
-
-

需改进:
-
-
"""


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("W32-D2 生成评估")
    print("=" * 60)

    # 示例数据
    eval_data = [
        {
            "question": "什么是RAG?",
            "context": "RAG(检索增强生成)是一种结合信息检索和文本生成的AI技术。"
                       "它通过先检索相关文档来增强LLM的回答质量, 减少幻觉问题。",
            "answer": "RAG(检索增强生成)是一种结合信息检索和文本生成的AI技术。"
                      "它通过先从知识库中检索相关文档, 然后将检索到的内容传递给LLM来生成回答。"
                      "这种方法有效减少了LLM的幻觉问题, 让回答更加准确可靠。"
        },
        {
            "question": "Python有哪些优势?",
            "context": "Python是一种广泛使用的高级编程语言, 拥有丰富的第三方库生态。",
            "answer": "Python的主要优势包括: 一、语法简洁易学; 二、拥有丰富的库如NumPy和Pandas;"
                      " 三、社区活跃, 资源丰富; 四、跨平台兼容性好。"
        },
        {
            "question": "如何优化检索?",
            "context": "检索优化可以通过调整chunk_size、使用重排序等方法。",
            "answer": "量子计算是未来技术的重要方向。"  # 不相关的回答
        },
    ]

    # 评估
    evaluator = GenerationEvaluator()
    results = evaluator.batch_evaluate(eval_data)

    for i, result in enumerate(results):
        print(f"\n{'='*50}")
        print(f"评估样本 {i+1}: {eval_data[i]['question']}")
        print(f"{'='*50}")
        print(f"  回答长度: {result['answer_length']}字")

        if result['faithfulness']['score'] is not None:
            f = result['faithfulness']
            print(f"  忠实度: {f['score']:.2f} ({f['supported']}/{f['total']}声明有支撑)")

        r = result['relevance']
        print(f"  相关性: {r['score']:.2f} (关键词覆盖={r['keyword_coverage']:.2f})")

        fl = result['fluency']
        print(f"  流畅度: {fl['score']:.2f} (句子数={fl['sentence_count']})")

        print(f"  综合分: {result['overall_score']:.2f}")

    # 人工评估模板
    print(f"\n{'='*60}")
    print("--- 人工评估模板 ---")
    print(generate_human_eval_template()[:500])

    # 可视化
    if HAS_PLT and results:
        fig, ax = plt.subplots(figsize=(10, 6))

        samples = [f"样本{i+1}" for i in range(len(results))]
        metrics = ['忠实度', '相关性', '流畅度', '综合']

        data = []
        for r in results:
            row = []
            row.append(r['faithfulness']['score'] or 0)
            row.append(r['relevance']['score'])
            row.append(r['fluency']['score'])
            row.append(r['overall_score'])
            data.append(row)

        x = np.arange(len(samples))
        width = 0.2
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

        for i, (metric, color) in enumerate(zip(metrics, colors)):
            values = [d[i] for d in data]
            ax.bar(x + i * width, values, width, label=metric, color=color)

        ax.set_xlabel('评估样本')
        ax.set_ylabel('分数')
        ax.set_title('生成质量评估结果', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(samples)
        ax.legend()
        ax.set_ylim(0, 1.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig('D:/code/big-model-learn/code/q_01/W32/d2_generation_eval.png', dpi=150)
        print("\n图表已保存为 d2_generation_eval.png")
        plt.close()

    print("\n完成!")
