"""
W19-D4 查询变换 (Query Transformation)
======================================
查询改写技术, HyDE实现, Step-back prompting,
Multi-query retrieval, 查询路由, 不同策略对比
"""

import sys
import re
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from collections import defaultdict

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W19-D4 查询变换 (Query Transformation)")
print("=" * 60)

# ============================================================
# 1. 查询变换概述
# ============================================================
print("\n--- 1. 查询变换概述 ---")
print("""
  用户的原始查询往往不理想:
    - 太短: 缺少关键词, 检索不到相关文档
    - 太模糊: 语义不清, 检索结果分散
    - 口语化: 与文档的专业表达不匹配
    - 太复杂: 多个问题混在一起

  查询变换技术:
    1) Query Rewriting:    改写查询使其更清晰
    2) HyDE:               用假设答案检索
    3) Step-back:          先问更抽象的问题
    4) Multi-query:        生成多个角度的查询
    5) Query Routing:      路由到合适的检索策略
    6) Query Decomposition: 分解复杂查询为子问题
""")

# ============================================================
# 2. 查询改写 (Query Rewriting)
# ============================================================
print("\n--- 2. 查询改写 ---")
print("""
  查询改写: 将原始查询转换为更适合检索的表达

  改写策略:
    1) 补充关键词: "RAG" -> "RAG 检索增强生成 rag retrieval augmented"
    2) 去除停用词: "什么是深度学习" -> "深度学习"
    3) 同义词扩展: "AI" -> "AI 人工智能 artificial intelligence"
    4) 专业化表达: "怎么训练模型" -> "模型训练方法 监督学习 微调"
""")


class QueryRewriter:
    """查询改写器"""

    def __init__(self):
        self.synonyms = {
            "AI": ["人工智能", "AI", "artificial intelligence"],
            "ML": ["机器学习", "ML", "machine learning"],
            "DL": ["深度学习", "DL", "deep learning", "神经网络"],
            "NLP": ["自然语言处理", "NLP", "文本处理", "natural language processing"],
            "LLM": ["大语言模型", "LLM", "大模型", "large language model"],
            "RAG": ["检索增强生成", "RAG", "retrieval augmented generation"],
        }

        self.stop_words = {"什么", "怎么", "如何", "的", "是", "了", "吗", "呢", "啊", "有哪些", "能", "可以"}

    def expand_synonyms(self, query: str) -> str:
        """同义词扩展"""
        expanded = query
        for term, syns in self.synonyms.items():
            if term in query or any(s in query for s in syns if len(s) > 2):
                for s in syns:
                    if s not in expanded:
                        expanded += " " + s
        return expanded

    def remove_stop_words(self, query: str) -> str:
        """去除停用词"""
        words = re.findall(r'[一-鿿]+|[a-zA-Z]+|[0-9]+', query)
        filtered = [w for w in words if w not in self.stop_words]
        return ' '.join(filtered)

    def rewrite(self, query: str) -> Dict[str, str]:
        """完整改写流程"""
        return {
            'original': query,
            'no_stop': self.remove_stop_words(query),
            'expanded': self.expand_synonyms(query),
            'combined': self.expand_synonyms(self.remove_stop_words(query)),
        }


rewriter = QueryRewriter()
test_queries = ["什么是RAG技术", "怎么训练深度学习模型", "NLP有哪些应用", "LLM怎么用"]
for q in test_queries:
    rewritten = rewriter.rewrite(q)
    print(f"\n  原始: {rewritten['original']}")
    print(f"  去停用词: {rewritten['no_stop']}")
    print(f"  同义词扩展: {rewritten['expanded']}")
    print(f"  组合: {rewritten['combined']}")

# ============================================================
# 3. HyDE 实现
# ============================================================
print("\n--- 3. HyDE 实现 ---")
print("""
  HyDE (Hypothetical Document Embeddings):
    Step 1: LLM为查询生成假设性答案
    Step 2: 用假设答案的嵌入进行检索
    Step 3: 返回检索到的真实文档

  为什么有效?
    - 答案比问题在向量空间中更接近真实文档
    - 假设答案即使不完全准确, 也提供了有用的语义信号
""")


class HyDEImplementer:
    """HyDE实现"""

    def __init__(self):
        self.mock_answers = {
            "机器学习": "机器学习是人工智能的一个子领域，它使用算法来从数据中学习模式。"
                        "主要方法包括监督学习（使用标注数据）、无监督学习（发现隐藏结构）和强化学习。",
            "深度学习": "深度学习使用多层神经网络进行特征学习。卷积神经网络(CNN)用于图像处理，"
                        "循环神经网络(RNN)用于序列数据，Transformer用于自然语言处理。",
            "RAG":     "RAG（检索增强生成）是一种结合信息检索和文本生成的技术。它通过从外部知识库"
                        "检索相关文档来增强大语言模型的回答，减少幻觉问题。",
            "Transformer": "Transformer是由Google在2017年提出的深度学习架构，基于自注意力机制。"
                           "它包括多头注意力、位置编码和前馈网络等组件。",
        }

    def generate_hypothesis(self, query: str) -> str:
        """生成假设性答案 (模拟LLM)"""
        for keyword, answer in self.mock_answers.items():
            if keyword in query:
                return answer
        return f"关于{query}的解释：这是一个重要的技术概念，在AI领域有广泛应用。"

    def multi_hypothesis(self, query: str, n: int = 3) -> List[str]:
        """生成多个假设答案"""
        base = self.generate_hypothesis(query)
        variants = [
            base,
            f"从技术角度来看，{base}",
            f"简单来说，{base}这项技术在近年来得到了快速发展。",
        ]
        return variants[:n]


hyde = HyDEImplementer()
for q in ["什么是机器学习", "RAG技术"]:
    hypothesis = hyde.generate_hypothesis(q)
    multi = hyde.multi_hypothesis(q)
    print(f"\n  查询: {q}")
    print(f"  假设答案: {hypothesis[:60]}...")
    print(f"  多假设数: {len(multi)}")

# ============================================================
# 4. Step-back Prompting
# ============================================================
print("\n--- 4. Step-back Prompting ---")
print("""
  Step-back Prompting (Google DeepMind, 2023):
    不直接回答具体问题, 先"后退一步"问更抽象/底层���问题

  示例:
    原始问题: "GPT-4在MMLU基准上的得分是多少?"
    Step-back: "大语言模型的评估方法和常用基准有哪些?"

    原始问题: "Python 3.11的字典性能提升多少?"
    Step-back: "Python字典的底层实现原理是什么?"

  优势:
    - 抽象问题能检索到更多相关背景知识
    - 避免过于具体的问题检索不到结果
    - 提供更全面的上下文
""")


class StepBackPrompter:
    """Step-back prompting"""

    def __init__(self):
        self.step_back_templates = {
            "具体数据": "关于{topic}的基本概念和原理是什么",
            "特定方法": "{field}领域常用的方法和技术有哪些",
            "特定模型": "{field}领域的主要模型架构有哪些",
            "特定工具": "{field}领域常用的工具和框架有哪些",
        }

    def generate_step_back(self, query: str) -> str:
        """生成step-back问题 (模拟)"""
        if any(w in query for w in ["得分", "性能", "效果", "分数"]):
            return "大语言模型的评估方法和常用基准有哪些"
        elif any(w in query for w in ["怎么用", "如何使用", "用法"]):
            return "这个技术的基本原理和核心概念是什么"
        elif any(w in query for w in ["训练", "微调", "优化"]):
            return "模型训练的基本方法和最佳实践有哪些"
        else:
            return f"与'{query}'相关的领域背景知识和核心概念有哪些"


stepback = StepBackPrompter()
test_qs = ["GPT-4的MMLU得分是多少", "怎么训练BERT模型", "RAG怎么用"]
for q in test_qs:
    sb = stepback.generate_step_back(q)
    print(f"\n  原始: {q}")
    print(f"  Step-back: {sb}")

# ============================================================
# 5. Multi-Query Retrieval
# ============================================================
print("\n--- 5. Multi-Query Retrieval ---")
print("""
  Multi-Query: 从不同角度生成多个查询, 分别检索后合并

  示例:
    原始: "机器学习"
    Query 1: "机器学习的基本概念和原理"
    Query 2: "机器学习的应用场景"
    Query 3: "机器学习的主要算法"
    Query 4: "机器学习和深度学习的区别"
    Query 5: "如何入门机器学习"
""")


class MultiQueryGenerator:
    """多查询生成器"""

    def __init__(self):
        self.angle_templates = [
            "{query}的基本概念和定义",
            "{query}的主要方法和算法",
            "{query}的应用场景和案例",
            "{query}的优缺点和局限性",
            "{query}的最新发展和趋势",
        ]

    def generate(self, query: str, n: int = 5) -> List[str]:
        """生成多角度查询"""
        queries = [query]  # 包含原始查询
        templates = self.angle_templates[:n-1]
        for template in templates:
            queries.append(template.format(query=query))
        return queries

    def merge_results(self, all_results: List[List[Tuple]], method='rrf', k=60) -> List[Tuple]:
        """合并多个查询的检索结果"""
        if method == 'rrf':
            return self._rrf_merge(all_results, k)
        else:
            return self._union_merge(all_results)

    def _rrf_merge(self, all_results, k=60):
        rrf_scores = defaultdict(float)
        for results in all_results:
            for rank, (doc_id, _) in enumerate(results):
                rrf_scores[doc_id] += 1.0 / (k + rank + 1)
        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    def _union_merge(self, all_results):
        scores = defaultdict(float)
        counts = defaultdict(int)
        for results in all_results:
            for doc_id, score in results:
                scores[doc_id] += score
                counts[doc_id] += 1
        # 投票 + 分数
        merged = [(did, scores[did] * (1 + counts[did] * 0.1)) for did in scores]
        return sorted(merged, key=lambda x: x[1], reverse=True)


mq = MultiQueryGenerator()
multi_queries = mq.generate("机器学习")
print(f"  原始查询: 机器学习")
print(f"  多角度查询:")
for i, q in enumerate(multi_queries):
    print(f"    {i+1}. {q}")

# ============================================================
# 6. 查询路由
# ============================================================
print("\n--- 6. 查询路由 ---")
print("""
  查询路由: 根据查询类型选择最合适的检索策略

  路由类型:
    1) 知识库路由: 根据主题路由到不同的知识库
    2) 策略路由:   根据查询特征选择检索方法
    3) 工具路由:   决定是否需要调用外部工具
""")


class QueryRouter:
    """查询路由器"""

    def __init__(self):
        self.routes = {
            'factual':    {'strategy': 'hybrid+rerank', 'top_k': 5,  'description': '事实性问题'},
            'conceptual': {'strategy': 'multi-query',   'top_k': 8,  'description': '概念性问题'},
            'procedural': {'strategy': 'hybrid',        'top_k': 10, 'description': '操作性问题'},
            'comparison': {'strategy': 'multi-query',   'top_k': 10, 'description': '比较性问题'},
            'casual':     {'strategy': 'dense',         'top_k': 3,  'description': '闲聊'},
        }

    def classify(self, query: str) -> str:
        """分类查询类型"""
        if any(w in query for w in ["是什么", "什么是", "定义"]):
            return 'factual'
        elif any(w in query for w in ["怎么", "如何", "步骤", "方法"]):
            return 'procedural'
        elif any(w in query for w in ["区别", "对比", "比较", "vs"]):
            return 'comparison'
        elif any(w in query for w in ["为什么", "原理", "概念"]):
            return 'conceptual'
        else:
            return 'casual'

    def route(self, query: str) -> Dict:
        """路由到合适的策略"""
        query_type = self.classify(query)
        config = self.routes[query_type]
        return {'query': query, 'type': query_type, 'config': config}


router = QueryRouter()
routing_tests = [
    "什么是Transformer架构",
    "怎么部署RAG系统",
    "BERT和GPT有什么区别",
    "深度学习的原理是什么",
    "今天天气怎么样",
]
for q in routing_tests:
    result = router.route(q)
    print(f"  '{q}' -> 类型: {result['type']}, 策略: {result['config']['strategy']}")

# ============================================================
# 7. 可视化
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 子图1: 不同查询变换策略效果对比
ax1 = axes[0]
strategies = ['原始查询', '同义词\n扩展', 'HyDE', 'Step-back', 'Multi-Query\n(5个)']
recall = [0.60, 0.72, 0.80, 0.75, 0.88]
precision = [0.85, 0.80, 0.78, 0.73, 0.70]

x = np.arange(len(strategies))
width = 0.35
ax1.bar(x - width/2, recall, width, label='Recall@10', color='#3498db', alpha=0.85, edgecolor='black')
ax1.bar(x + width/2, precision, width, label='Precision@5', color='#e74c3c', alpha=0.85, edgecolor='black')
ax1.set_xticks(x)
ax1.set_xticklabels(strategies, fontsize=10)
ax1.set_ylabel('分数', fontsize=12)
ax1.set_title('查询变换策略效果对比', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim(0.5, 1.0)

# 子图2: 查询路由决策树
ax2 = axes[1]
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_title('查询路由决策流程', fontsize=14, fontweight='bold')

nodes = [
    (5, 9.0, "用户查询", '#3498db'),
    (5, 7.5, "查询分类器", '#e67e22'),
    (2, 6.0, "事实型\nhybrid+rerank", '#2ecc71'),
    (4, 6.0, "概念型\nmulti-query", '#9b59b6'),
    (6, 6.0, "操作型\nhybrid", '#e74c3c'),
    (8, 6.0, "比较型\nmulti-query", '#1abc9c'),
    (5, 4.0, "检索执行", '#f39c12'),
    (5, 2.5, "结果合并", '#3498db'),
    (5, 1.0, "返回答案", '#2ecc71'),
]

for x_pos, y_pos, text, color in nodes:
    w, h = (2.5, 0.9) if y_pos >= 6 else (2.0, 0.8)
    ax2.add_patch(plt.Rectangle((x_pos - w/2, y_pos - h/2), w, h,
                                  facecolor=color, edgecolor='black', alpha=0.3, linewidth=2))
    ax2.text(x_pos, y_pos, text, ha='center', va='center', fontsize=9, fontweight='bold')

# 箭头
connections = [
    ((5, 8.5), (5, 8.0)),
    ((5, 7.0), (2, 6.5)), ((5, 7.0), (4, 6.5)), ((5, 7.0), (6, 6.5)), ((5, 7.0), (8, 6.5)),
    ((2, 5.5), (5, 4.4)), ((4, 5.5), (5, 4.4)), ((6, 5.5), (5, 4.4)), ((8, 5.5), (5, 4.4)),
    ((5, 3.6), (5, 2.9)),
    ((5, 2.1), (5, 1.4)),
]
for start, end in connections:
    ax2.annotate('', xy=end, xytext=start,
                 arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W19/query_transformation.png', dpi=150, bbox_inches='tight')
print("\n  图表已保存: query_transformation.png")
plt.close()

# ============================================================
# 8. 策略选择总结
# ============================================================
print("\n--- 8. 查询变换策略选择 ---")
print("""
  +------------------+------------------------+------------------+
  |      策略        |       适用场景         |     效果提升     |
  +------------------+------------------------+------------------+
  | 同义词扩展       | 查询词与文档不匹配    | Recall +10-15%   |
  | HyDE             | 短查询, 抽象问题      | Recall +15-20%   |
  | Step-back        | 过于具体的问题        | Context +20%     |
  | Multi-Query      | 模糊查询, 多义查询    | Recall +20-30%   |
  | 查询路由         | 混合类型查询          | 综合 +15%        |
  | 查询分解         | 复杂多部分问题        | 综合 +25%        |
  +------------------+------------------------+------------------+

  推荐组合:
    通用场景: 查询路由 + (HyDE或Multi-Query)
    高质量:   查询路由 + Multi-Query + Rerank
""")

print("\n" + "=" * 60)
print("W19-D4 完成! 本节学习了多种查询变换技术")
print("=" * 60)
