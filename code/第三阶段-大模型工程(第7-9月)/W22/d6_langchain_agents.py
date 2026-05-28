### Day 6（周六）：LangChain Agent框架
# AgentType对比, Chain组合, 自定义Tool, AgentExecutor配置

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. LangChain框架概念
# ============================================================
print("=" * 60)
print("1. LangChain框架核心概念")
print("=" * 60)

print("""
  LangChain核心组件:

  1. Model I/O     - LLM接口, Prompt模板, 输出解析器
  2. Retrieval     - 文档加载, 向量存储, 检索器
  3. Chains        - 调用链 (顺序/分支/并行)
  4. Agents        - 自主决策的工具调用
  5. Memory        - 对话记忆管理
  6. Callbacks     - 日志和监控

  架构图:
    User → Agent → [LLM] → 工具选择 → Tool执行 → 观察结果 → 继续或回答
""")


# ============================================================
# 2. 模拟LangChain核心组件
# ============================================================

# 2.1 Prompt模板
class PromptTemplate:
    """模拟LangChain PromptTemplate"""

    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def format(self, **kwargs):
        result = self.template
        for var in self.input_variables:
            result = result.replace(f"{{{var}}}", str(kwargs.get(var, '')))
        return result


print("=" * 60)
print("2. Prompt模板系统")
print("=" * 60)

# 创建Prompt模板
qa_template = PromptTemplate(
    template="请根据以下上下文回答问题:\n\n上下文: {context}\n\n问题: {question}\n\n回答:",
    input_variables=['context', 'question'],
)

agent_template = PromptTemplate(
    template="""你是一个AI助手, 可以使用以下工具:
{tools}

请回答用户的问题: {input}
思考过程:
1. 分析问题需要什么信息
2. 选择合适的工具
3. 执行并返回结果

回答:""",
    input_variables=['tools', 'input'],
)

# 使用模板
print("\n  QA模板示例:")
print(qa_template.format(
    context="Python是一种高级编程语言",
    question="Python是什么?"
))

print("\n  Agent模板示例:")
print(agent_template.format(
    tools="1. calculator - 数学计算\n2. search - 搜索",
    input="计算10+20",
)[:200] + "...")


# 2.2 Chain组合
class Chain:
    """模拟LangChain Chain"""

    def __init__(self, name):
        self.name = name
        self.steps = []

    def add_step(self, step_name, step_func):
        self.steps.append({'name': step_name, 'func': step_func})
        return self

    def run(self, input_data):
        """顺序执行所有步骤"""
        print(f"\n  [Chain: {self.name}] 开始执行")
        result = input_data
        for i, step in enumerate(self.steps):
            print(f"    Step {i + 1}: {step['name']}")
            result = step['func'](result)
            print(f"    → 输出: {str(result)[:60]}")
        return result


class SequentialChain:
    """顺序链: 多个Chain依次执行"""

    def __init__(self):
        self.chains = []

    def add_chain(self, chain):
        self.chains.append(chain)
        return self

    def run(self, input_data):
        result = input_data
        for chain in self.chains:
            result = chain.run(result)
        return result


print("\n" + "=" * 60)
print("3. Chain组合")
print("=" * 60)

# 创建处理链
def parse_input(text):
    return {'parsed': text.strip(), 'type': 'question'}

def retrieve_context(data):
    mock_context = "这是检索到的相关上下文信息。"
    data['context'] = mock_context
    return data

def generate_answer(data):
    return f"根据'{data['context'][:15]}...' 回答: {data['parsed']}的答案是..."

chain = Chain("QA Chain")
chain.add_step("解析输入", parse_input)
chain.add_step("检索上下文", retrieve_context)
chain.add_step("生成回答", generate_answer)

result = chain.run("什么是深度学习?")


# 2.3 自定义Tool
class CustomTool:
    """模拟LangChain BaseTool"""

    def __init__(self, name, description, func, return_direct=False):
        self.name = name
        self.description = description
        self.func = func
        self.return_direct = return_direct

    def _run(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __repr__(self):
        return f"Tool(name='{self.name}', desc='{self.description}')"


print("\n" + "=" * 60)
print("4. 自定义Tool")
print("=" * 60)

# 定义自定义工具
def python_repl(code):
    """模拟Python REPL"""
    try:
        # 只模拟, 不实际执行
        return f"[REPL模拟] 代码: {code[:50]}"
    except Exception as e:
        return f"错误: {e}"

def wikipedia_search(query):
    """模拟Wikipedia搜索"""
    mock_articles = {
        'AI': '人工智能(Artificial Intelligence)是计算机科学的分支...',
        'Python': 'Python编程语言由Guido van Rossum创建...',
    }
    for key, content in mock_articles.items():
        if key.lower() in query.lower():
            return content
    return f"未找到关于'{query}'的条目"

def json_extractor(json_str, key):
    """JSON数据提取"""
    try:
        import json as json_module
        data = json_module.loads(json_str)
        return str(data.get(key, f"键'{key}'不存在"))
    except Exception:
        return "无效的JSON"

tools = [
    CustomTool("python_repl", "执行Python代码", python_repl),
    CustomTool("wikipedia", "搜索Wikipedia", wikipedia_search),
    CustomTool("json_extractor", "从JSON中提取数据", json_extractor),
]

print("\n  已注册的自定义工具:")
for tool in tools:
    print(f"    {tool}")


# ============================================================
# 3. AgentType对比
# ============================================================
print("\n" + "=" * 60)
print("5. LangChain AgentType对比")
print("=" * 60)

agent_types = {
    'Zero-shot ReAct': {
        '描述': '基于工具描述动态选择, 无记忆',
        '优点': '简单灵活, 无需示例',
        '缺点': '可能选择错误工具',
        '适用': '简单工具调用',
    },
    'Conversational': {
        '描述': '带对话记忆的ReAct Agent',
        '优点': '支持多轮对话, 有上下文',
        '缺点': '记忆管理复杂',
        '适用': '对话式助手',
    },
    'OpenAI Functions': {
        '描述': '利用OpenAI Function Calling',
        '优点': '结构化输出, 可靠性高',
        '缺点': '依赖OpenAI API',
        '适用': 'OpenAI模型',
    },
    'Plan-and-Execute': {
        '描述': '先规划后执行',
        '优点': '适合复杂多步任务',
        '缺点': '规划可能需要调整',
        '适用': '复杂工作流',
    },
    'Self-Ask': {
        '描述': '自问自答模式',
        '优点': '适合需要搜索的问题',
        '缺点': '可能产生多余子问题',
        '适用': '知识密集型问答',
    },
}

print(f"\n  {'Agent类型':<20} {'描述':<30} {'适用场景':<15}")
print(f"  {'-' * 65}")
for name, info in agent_types.items():
    print(f"  {name:<20} {info['描述'][:28]:<30} {info['适用']:<15}")


# ============================================================
# 4. AgentExecutor模拟
# ============================================================
class SimulatedAgentExecutor:
    """模拟LangChain AgentExecutor"""

    def __init__(self, agent_type, tools, max_iterations=10, verbose=True):
        self.agent_type = agent_type
        self.tools = {t.name: t for t in tools}
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.history = []
        self.iteration_count = 0

    def _select_tool(self, task):
        """工具选择逻辑"""
        task_lower = task.lower()
        for name, tool in self.tools.items():
            if any(kw in task_lower for kw in name.split('_')):
                return name
            if any(kw in task_lower for kw in tool.description.lower().split()):
                return name
        return None

    def run(self, input_text):
        """运行AgentExecutor"""
        if self.verbose:
            print(f"\n  [AgentExecutor] 类型: {self.agent_type}")
            print(f"  [AgentExecutor] 输入: {input_text}")

        for iteration in range(self.max_iterations):
            self.iteration_count += 1

            # Agent决策
            tool_name = self._select_tool(input_text)

            if tool_name is None:
                answer = f"直接回答: {input_text}"
                self.history.append({'iter': iteration, 'type': 'answer', 'content': answer})
                if self.verbose:
                    print(f"  [迭代{iteration + 1}] 直接回答")
                return answer

            # 执行工具
            tool = self.tools[tool_name]
            result = tool._run(input_text)
            self.history.append({
                'iter': iteration, 'type': 'tool_call',
                'tool': tool_name, 'result': result,
            })

            if self.verbose:
                print(f"  [迭代{iteration + 1}] 调用 {tool_name} → {str(result)[:50]}")

            if tool.return_direct:
                return result

            # 构造最终回答
            answer = f"基于{tool_name}的结果: {result}"
            self.history.append({'iter': iteration, 'type': 'answer', 'content': answer})
            return answer

        return "达到最大迭代次数"

    def get_stats(self):
        return {
            'agent_type': self.agent_type,
            'iterations': self.iteration_count,
            'tools_available': list(self.tools.keys()),
        }


print("\n" + "=" * 60)
print("6. AgentExecutor测试")
print("=" * 60)

executor = SimulatedAgentExecutor(
    agent_type="Zero-shot ReAct",
    tools=tools,
    max_iterations=5,
)

test_inputs = [
    "搜索AI相关信息",
    "执行Python代码: print('hello')",
    "什么是深度学习?",
]

for inp in test_inputs:
    result = executor.run(inp)
    print(f"  最终结果: {str(result)[:60]}\n")


# ============================================================
# 5. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 5.1 LangChain组件架构图
ax1 = axes[0, 0]
components = ['Model I/O', 'Retrieval', 'Chains', 'Agents', 'Memory', 'Callbacks']
component_sizes = [3, 2, 4, 5, 3, 2]
colors_comp = plt.cm.Set2(np.linspace(0, 1, len(components)))

for i, (comp, size, color) in enumerate(zip(components, component_sizes, colors_comp)):
    rect = plt.Rectangle((0.05 + i * 0.15, 0.3), 0.13, size * 0.1,
                          facecolor=color, alpha=0.7, edgecolor='black')
    ax1.add_patch(rect)
    ax1.text(0.05 + i * 0.15 + 0.065, 0.3 + size * 0.05, comp,
             ha='center', va='center', fontsize=7, fontweight='bold')

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 0.9)
ax1.set_title('LangChain核心组件')
ax1.axis('off')

# 5.2 AgentType特性对比
ax2 = axes[0, 1]
categories_at = ['简单性', '灵活性', '可靠性', '上下文', '效率']
agent_type_scores = {
    'Zero-shot ReAct': [4, 4, 3, 1, 4],
    'Conversational':  [3, 3, 3, 5, 3],
    'OpenAI Functions': [4, 3, 5, 3, 5],
    'Plan-Execute':    [2, 5, 4, 3, 2],
}

x = np.arange(len(categories_at))
width = 0.2
for i, (at_name, scores) in enumerate(agent_type_scores.items()):
    ax2.bar(x + i * width, scores, width, label=at_name, edgecolor='black')

ax2.set_ylabel('分数 (1-5)')
ax2.set_title('AgentType特性对比')
ax2.set_xticks(x + width * 1.5)
ax2.set_xticklabels(categories_at)
ax2.legend(fontsize=7)
ax2.set_ylim(0, 6)

# 5.3 Chain执行时间分析
ax3 = axes[1, 0]
chain_types = ['SimpleChain', 'SequentialChain', 'RouterChain', 'MapReduceChain']
execution_times = [0.5, 1.2, 0.8, 2.5]
bar_colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']
ax3.bar(chain_types, execution_times, color=bar_colors, edgecolor='black')
ax3.set_ylabel('执行时间 (s)')
ax3.set_title('不同Chain类型执行时间')
for i, v in enumerate(execution_times):
    ax3.text(i, v + 0.05, f'{v}s', ha='center')

# 5.4 Agent迭代次数分布
ax4 = axes[1, 1]
np.random.seed(42)
iterations_simple = np.random.poisson(1.5, 1000) + 1
iterations_complex = np.random.poisson(3, 1000) + 1
ax4.hist(iterations_simple, bins=range(1, 10), alpha=0.6, label='简单任务', color='#4CAF50', edgecolor='black')
ax4.hist(iterations_complex, bins=range(1, 10), alpha=0.6, label='复杂任务', color='#FF9800', edgecolor='black')
ax4.set_xlabel('迭代次数')
ax4.set_ylabel('频率')
ax4.set_title('Agent迭代次数分布')
ax4.legend()

plt.suptitle('W22-D6: LangChain Agent框架', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W22/d6_langchain_agents.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存!")
