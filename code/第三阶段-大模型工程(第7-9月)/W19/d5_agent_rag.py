"""
W19-D5 Agent RAG (RAG with Agent)
==================================
工具调用模式, 自适应检索(判断是否需要检索),
ReAct模式实现, 检索+推理结合
"""

import sys
import re
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Callable
from collections import defaultdict

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W19-D5 Agent RAG (RAG with Agent)")
print("=" * 60)

# ============================================================
# 1. Agent RAG 概念
# ============================================================
print("\n--- 1. Agent RAG 概念 ---")
print("""
  传统RAG: 固定流程 (检索 -> 生成)
  Agent RAG: Agent自主决定何时检索、检索什么、如何使用

  Agent的核心能力:
    1) 判断: 这个问题是否需要检索?
    2) 规划: 需要检索什么内容? 检索几次?
    3) 工具使用: 调用检索、计算、数据库等工具
    4) 推理: 基于检索结果进行推理
    5) 自我修正: 检查答案, 如果不满意则重新检索

  Agent框架:
    - ReAct (Reasoning + Acting)
    - Toolformer
    - HuggingGPT
    - AutoGPT / LangChain Agent
""")

# ============================================================
# 2. 工具定义
# ============================================================
print("\n--- 2. 工具定义 ---")


class Tool:
    """工具基类"""
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def run(self, *args, **kwargs):
        raise NotImplementedError


class KnowledgeBaseTool(Tool):
    """知识库检索工具"""

    def __init__(self, documents: List[str]):
        super().__init__(
            name="knowledge_search",
            description="搜索内部知识库。输入: 搜索查询字符串。输出: 相关文档列表。"
        )
        self.documents = documents
        self.index = defaultdict(list)
        self._build_index()

    def _build_index(self):
        for i, doc in enumerate(self.documents):
            for word in re.findall(r'[一-鿿]+|[a-zA-Z]+', doc.lower()):
                self.index[word].append(i)

    def run(self, query: str, top_k: int = 3) -> List[Dict]:
        """执行检索"""
        query_words = re.findall(r'[一-鿿]+|[a-zA-Z]+', query.lower())
        doc_scores = Counter()
        for word in query_words:
            for doc_id in self.index.get(word, []):
                doc_scores[doc_id] += 1

        results = []
        for doc_id, score in doc_scores.most_common(top_k):
            results.append({
                'doc_id': doc_id,
                'content': self.documents[doc_id],
                'score': score,
            })
        return results


class CalculatorTool(Tool):
    """计算器工具"""

    def __init__(self):
        super().__init__(
            name="calculator",
            description="执行数学计算。输入: 数学表达式字符串。输出: 计算结果。"
        )

    def run(self, expression: str) -> Dict:
        """安全计算"""
        try:
            # 只允许安全字符
            safe_expr = re.sub(r'[^0-9+\-*/().%\s]', '', expression)
            result = eval(safe_expr)
            return {'expression': expression, 'result': result}
        except Exception as e:
            return {'expression': expression, 'error': str(e)}


class LLMTool(Tool):
    """模拟LLM工具"""

    def __init__(self):
        super().__init__(
            name="llm_generate",
            description="使用LLM生成回答。输入: 提示词。输出: 生成的文本。"
        )

    def run(self, prompt: str) -> str:
        """模拟LLM生成"""
        # 简单的关键词匹配模拟
        if "机器学习" in prompt:
            return "机器学习是AI的子领域，通过数据训练模型来学习模式。"
        elif "深度学习" in prompt:
            return "深度学习使用多层神经网络，在CV和NLP领域取得突破。"
        elif "计算" in prompt or "多少" in prompt:
            return "让我帮你计算一下..."
        return f"基于已有信息，这个问题可以从多个角度来回答。"


# 初始化工具
knowledge_docs = [
    "机器学习是人工智能的核心子领域，通过算法从数据中学习。监督学习使用标注数据，无监督学习发现隐藏模式。",
    "深度学习使用多层神经网络进行特征学习。CNN用于图像处理，RNN用于序列数据，Transformer用于NLP。",
    "RAG检索增强生成通过检索外部知识来增强LLM的回答质量，减少幻觉问题。",
    "LoRA低秩适应是一种参数高效微调方法，通过低秩分解减少可训练参数。",
    "Transformer架构基于自注意力机制，由Google在2017年提出，是现代NLP的基础。",
    "向量数据库使用ANN算法实现高效相似度搜索。FAISS是最流行的向量检索库。",
    "Python是AI开发最常用的语言。主要库包括NumPy、Pandas、PyTorch、TensorFlow。",
    "大语言模型的参数量从7B到1.8T不等。更大的模型通常性能更好但推理成本更高。",
]

tools = {
    'search': KnowledgeBaseTool(knowledge_docs),
    'calculator': CalculatorTool(),
    'llm': LLMTool(),
}

for name, tool in tools.items():
    print(f"  工具 [{name}]: {tool.name} - {tool.description[:40]}...")

# ============================================================
# 3. 自适应检索
# ============================================================
print("\n--- 3. 自适应检索 (判断是否需要检索) ---")
print("""
  自适应检索: Agent判断是否需要检索外部知识

  判断标准:
    1) 涉及具体事实/数据 -> 需要检索
    2) 常识性问题 -> 不需要检索
    3) 数学计算 -> 不需要检索, 用计算器
    4) 最新信息 -> 需要检索
    5) 闲聊 -> 不需要检索
""")


class RetrievalJudge:
    """判断是否需要检索"""

    def should_retrieve(self, query: str) -> Dict:
        """判断查询是否需要检索"""
        # 需要检索的信号词
        retrieval_signals = ["最新", "目前", "数据", "统计", "报告", "论文",
                            "具体", "多少", "哪个", "谁", "何时", "哪里"]
        # 不需要检索的信号词
        no_retrieve_signals = ["你好", "谢谢", "帮我算", "计算", "+", "-",
                              "你好吗", "什么是", "定义"]

        # 计算得分
        score = 0
        reasons = []
        for signal in retrieval_signals:
            if signal in query:
                score += 1
                reasons.append(f"包含检索信号词'{signal}'")

        for signal in no_retrieve_signals:
            if signal in query:
                score -= 1
                reasons.append(f"包含非检索信号词'{signal}'")

        should = score > 0

        return {
            'query': query,
            'should_retrieve': should,
            'confidence': min(1.0, abs(score) / 3),
            'reasons': reasons,
        }


judge = RetrievalJudge()
test_judgments = [
    "2024年最新的LLM评测报告",
    "帮我计算 (128 + 256) * 2",
    "你好，今天天气怎么样",
    "Transformer架构的具体细节",
    "什么是深度学习的定义",
    "最新的机器学习论文有哪些",
]

print("\n  检索判断测试:")
for q in test_judgments:
    result = judge.should_retrieve(q)
    action = "检索" if result['should_retrieve'] else "跳过检索"
    print(f"  '{q}' -> {action} (置信度={result['confidence']:.2f})")

# ============================================================
# 4. ReAct 模式实现
# ============================================================
print("\n--- 4. ReAct 模式实现 ---")
print("""
  ReAct (Reasoning + Acting) 模式:

  循环:
    Thought: 思考当前情况, 决定下一步
    Action:  执行一个动作 (调用工具)
    Observation: 观察动作结果
    ... 重复直到可以给出最终答案 ...
    Answer: 给出最终回答

  示例:
    Query: "LoRA微调的参数量和全量微调的对比"

    Thought: 这是一个技术对比问题, 我需要检索LoRA相关资料
    Action: search("LoRA微调参数量对比")
    Observation: [找到3篇相关文档, 包含参数量信息]

    Thought: 我已经有了基本信息, 可以回答了
    Answer: LoRA只训练0.1%-1%的参数, 而全量微调训练100%参数...
""")


class ReActAgent:
    """ReAct模式Agent"""

    def __init__(self, tools: Dict, max_iterations: int = 5):
        self.tools = tools
        self.max_iterations = max_iterations
        self.judge = RetrievalJudge()

    def think(self, query: str, observations: List[str]) -> Dict:
        """思考: 决定下一步行动"""
        if not observations:
            # 第一步: 判断是否需要检索
            judgment = self.judge.should_retrieve(query)
            if judgment['should_retrieve']:
                return {
                    'thought': f'问题涉及具体信息, 需要检索: {judgment["reasons"]}',
                    'action': 'search',
                    'action_input': query,
                }
            else:
                return {
                    'thought': '这是通用问题, 直接用LLM回答',
                    'action': 'llm',
                    'action_input': query,
                }
        else:
            # 有观察结果: 决定是否继续检索或生成答案
            return {
                'thought': '已收集足够信息, 生成最终答案',
                'action': 'answer',
                'action_input': self._synthesize(query, observations),
            }

    def _synthesize(self, query: str, observations: List[str]) -> str:
        """综合观察结果生成答案"""
        context = '\n'.join(observations)
        return f"基于检索到的信息:\n{context}\n\n对于'{query}'的回答: 综合以上资料, 这是一个重要技术概念。"

    def run(self, query: str) -> Dict:
        """执行ReAct循环"""
        print(f"\n  === ReAct Agent 处理: '{query}' ===")

        observations = []
        trace = []

        for iteration in range(self.max_iterations):
            # Think
            thought_result = self.think(query, observations)
            trace.append({
                'step': iteration + 1,
                'type': 'thought',
                'content': thought_result['thought'],
            })
            print(f"  [Step {iteration+1}] Thought: {thought_result['thought']}")

            # Act
            action = thought_result['action']
            action_input = thought_result['action_input']

            if action == 'search':
                result = self.tools['search'].run(action_input)
                obs = f"检索到{len(result)}篇文档: " + "; ".join([r['content'][:30] + '...' for r in result])
                print(f"  [Step {iteration+1}] Action: search('{action_input}')")
                print(f"  [Step {iteration+1}] Observation: {obs[:80]}...")
                observations.append(obs)

            elif action == 'calculator':
                result = self.tools['calculator'].run(action_input)
                obs = f"计算结果: {result}"
                print(f"  [Step {iteration+1}] Action: calculate('{action_input}')")
                print(f"  [Step {iteration+1}] Observation: {obs}")
                observations.append(obs)

            elif action == 'llm':
                result = self.tools['llm'].run(action_input)
                print(f"  [Step {iteration+1}] Action: llm_generate")
                print(f"  [Step {iteration+1}] Observation: {result[:60]}...")
                observations.append(result)
                # LLM直接回答, 可以结束
                action = 'answer'
                action_input = result

            elif action == 'answer':
                print(f"  [Step {iteration+1}] Answer: {action_input[:80]}...")
                return {
                    'query': query,
                    'answer': action_input,
                    'trace': trace,
                    'iterations': iteration + 1,
                    'observations': observations,
                }

        # 达到最大迭代
        return {
            'query': query,
            'answer': observations[-1] if observations else "无法回答",
            'trace': trace,
            'iterations': self.max_iterations,
            'observations': observations,
        }


# 测试Agent
agent = ReActAgent(tools, max_iterations=4)

test_queries = [
    "LoRA微调方法的原理是什么",
    "帮我计算 1024 * 768",
    "什么是深度学习",
]

for q in test_queries:
    result = agent.run(q)
    print(f"\n  最终答案: {str(result['answer'])[:100]}...")
    print(f"  迭代次数: {result['iterations']}")

# ============================================================
# 5. 可视化
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# 子图1: Agent决策流程
ax1 = axes[0]
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('Agent RAG 决策流程', fontsize=14, fontweight='bold')

flow_steps = [
    (5, 9.0, "用户问题", '#3498db'),
    (5, 7.8, "是否需要检索?", '#e67e22'),
    (2, 6.6, "直接LLM\n回答", '#2ecc71'),
    (8, 6.6, "检索\n知识库", '#e74c3c'),
    (8, 5.2, "信息\n足够?", '#e67e22'),
    (5, 4.0, "综合推理\n生成答案", '#9b59b6'),
    (8, 3.0, "继续检索\n/改写查询", '#f39c12'),
    (5, 2.0, "返回答案+来源", '#1abc9c'),
]

for x_pos, y_pos, text, color in flow_steps:
    w, h = (2.5, 0.8)
    shape = 'round' if '?' in text else 'rect'
    ax1.add_patch(plt.Rectangle((x_pos - w/2, y_pos - h/2), w, h,
                                  facecolor=color, edgecolor='black', alpha=0.3, linewidth=2,
                                  linestyle='--' if '?' in text else '-'))
    ax1.text(x_pos, y_pos, text, ha='center', va='center', fontsize=9, fontweight='bold')

connections = [
    ((5, 8.6), (5, 8.2)),
    ((4, 7.4), (2, 7.0), "否"),
    ((6, 7.4), (8, 7.0), "是"),
    ((2, 6.2), (5, 4.4)),
    ((8, 6.2), (8, 5.6)),
    ((7, 5.2), (5, 4.4), "是"),
    ((9, 5.2), (8, 3.4), "否"),
    ((8, 2.6), (8, 5.6)),
    ((5, 3.6), (5, 2.4)),
]
for conn in connections:
    start, end = conn[0], conn[1]
    label = conn[2] if len(conn) > 2 else None
    ax1.annotate('', xy=end, xytext=start,
                 arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    if label:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax1.text(mid_x + 0.2, mid_y, label, fontsize=10, color='red', fontweight='bold')

# 子图2: 传统RAG vs Agent RAG对比
ax2 = axes[1]
categories = ['检索精度', '复杂问题', '简单问题', '工具调用', '成本效率']
traditional_rag = [0.75, 0.50, 0.90, 0.20, 0.85]
agent_rag = [0.82, 0.80, 0.88, 0.90, 0.60]

x = np.arange(len(categories))
width = 0.35
ax2.bar(x - width/2, traditional_rag, width, label='传统RAG', color='#3498db', alpha=0.85, edgecolor='black')
ax2.bar(x + width/2, agent_rag, width, label='Agent RAG', color='#e74c3c', alpha=0.85, edgecolor='black')

ax2.set_xticks(x)
ax2.set_xticklabels(categories, fontsize=10)
ax2.set_ylabel('评分', fontsize=12)
ax2.set_title('传统RAG vs Agent RAG 对比', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W19/agent_rag.png', dpi=150, bbox_inches='tight')
print("\n  图表已保存: agent_rag.png")
plt.close()

# ============================================================
# 6. Agent RAG 框架对比
# ============================================================
print("\n--- 6. Agent RAG 框架对比 ---")
print("""
  +------------------+------------------+------------------+
  |    框架          |    特点          |    适用场景      |
  +------------------+------------------+------------------+
  | LangChain Agent  | 灵活, 工具丰富   | 通用场景         |
  | LlamaIndex       | RAG专用, 简单    | 文档问答         |
  | AutoGen          | Multi-Agent      | 复杂协作任务     |
  | CrewAI           | 角色扮演Agent    | 团队协作场景     |
  | Haystack         | Pipeline式       | 生产环境         |
  +------------------+------------------+------------------+

  选择建议:
    快速原型:  LlamaIndex
    灵活定制:  LangChain Agent
    生产部署:  Haystack / LangServe
""")

print("\n" + "=" * 60)
print("W19-D5 完成! 本节实现了ReAct模式的Agent RAG")
print("=" * 60)
