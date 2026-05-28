### Day 5（周五）：多工具Agent
# 注册多种工具, 自主选择工具, 错误处理/重试, 工具调用链, Agent循环

import numpy as np
import matplotlib.pyplot as plt
import time

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 增强版工具系统
# ============================================================
class Tool:
    """增强版工具类"""

    def __init__(self, name, func, description, param_schema, max_retries=3, timeout=30):
        self.name = name
        self.func = func
        self.description = description
        self.param_schema = param_schema
        self.max_retries = max_retries
        self.timeout = timeout
        self.call_count = 0
        self.error_count = 0
        self.total_time = 0

    def execute(self, **kwargs):
        """执行工具, 带重试机制"""
        self.call_count += 1
        for attempt in range(self.max_retries):
            try:
                start = time.time()
                result = self.func(**kwargs)
                elapsed = time.time() - start
                self.total_time += elapsed
                return {'success': True, 'result': result, 'attempts': attempt + 1}
            except Exception as e:
                self.error_count += 1
                if attempt == self.max_retries - 1:
                    return {'success': False, 'error': str(e), 'attempts': attempt + 1}
                time.sleep(0.01 * (attempt + 1))  # 指数退避模拟

    def get_stats(self):
        return {
            'name': self.name,
            'calls': self.call_count,
            'errors': self.error_count,
            'avg_time': self.total_time / max(self.call_count, 1),
        }


# 定义工具函数
def calculator(expression):
    """计算器"""
    clean = ''.join(c for c in str(expression) if c in '0123456789+-*/(). ')
    if not clean:
        raise ValueError("空表达式")
    return eval(clean)

def search(query):
    """知识搜索"""
    db = {
        'python': 'Python是Guido van Rossum创建的高级编程语言',
        'ai': '人工智能是模拟人类智能的技术',
        'llm': '大语言模型是基于Transformer的大规模模型',
        'agent': 'AI Agent是能自主决策的智能系统',
        'weather': '天气是指大气层在短时间内的状态',
    }
    for key, val in db.items():
        if key in query.lower():
            return val
    raise ValueError(f"未找到: {query}")

def unit_converter(value, from_unit, to_unit):
    """单位转换"""
    conversions = {
        ('km', 'mile'): 0.621371,
        ('mile', 'km'): 1.60934,
        ('kg', 'lb'): 2.20462,
        ('lb', 'kg'): 0.453592,
        ('c', 'f'): lambda x: x * 9 / 5 + 32,
        ('f', 'c'): lambda x: (x - 32) * 5 / 9,
    }
    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        factor = conversions[key]
        if callable(factor):
            return factor(float(value))
        return float(value) * factor
    raise ValueError(f"不支持的转换: {from_unit}->{to_unit}")

def text_analyzer(text):
    """文本分析"""
    return {
        'char_count': len(text),
        'word_count': len(text.split()),
        'sentence_count': text.count('.') + text.count('。'),
        'has_chinese': any('一' <= c <= '鿿' for c in text),
    }

def data_processor(data, operation):
    """数据处理"""
    arr = np.array(data)
    ops = {
        'sum': np.sum, 'mean': np.mean, 'std': np.std,
        'max': np.max, 'min': np.min, 'sort': lambda x: sorted(x.tolist()),
    }
    if operation in ops:
        return ops[operation](arr)
    raise ValueError(f"未知操作: {operation}")


# ============================================================
# 2. 多工具Agent实现
# ============================================================
class MultiToolAgent:
    """多工具Agent: 自主选择和使用工具"""

    def __init__(self, name="MultiToolAgent"):
        self.name = name
        self.tools = {}
        self.conversation_history = []
        self.tool_trace = []
        self.max_iterations = 10

    def register_tool(self, tool):
        """注册工具"""
        self.tools[tool.name] = tool
        print(f"  [{self.name}] 注册工具: {tool.name} - {tool.description}")

    def select_tool(self, task):
        """智能选择工具"""
        task_lower = task.lower()
        tool_scores = {}

        # 关键词匹配评分
        scoring_rules = {
            'calculator': ['计算', '加', '减', '乘', '除', '=', '+', '-', '*', '/'],
            'search': ['搜索', '查询', '什么是', '告诉我', '了解'],
            'unit_converter': ['转换', '换算', '公里', '英里', '千克', '磅', '摄氏', '华氏'],
            'text_analyzer': ['分析文本', '字数', '统计', '字符数'],
            'data_processor': ['数据处理', '求和', '平均值', '排序', '最大', '最小'],
        }

        for tool_name, keywords in scoring_rules.items():
            if tool_name in self.tools:
                score = sum(1 for kw in keywords if kw in task_lower)
                tool_scores[tool_name] = score

        best_tool = max(tool_scores, key=tool_scores.get) if tool_scores else None
        return best_tool if tool_scores.get(best_tool, 0) > 0 else None

    def extract_params(self, tool_name, task):
        """从任务中提取工具参数"""
        if tool_name == 'calculator':
            clean = ''.join(c for c in task if c in '0123456789+-*/(). ')
            return {'expression': clean.strip()}
        elif tool_name == 'search':
            return {'query': task}
        elif tool_name == 'unit_converter':
            import re
            nums = re.findall(r'\d+\.?\d*', task)
            value = float(nums[0]) if nums else 0
            return {'value': value, 'from_unit': 'km', 'to_unit': 'mile'}
        elif tool_name == 'text_analyzer':
            return {'text': task}
        elif tool_name == 'data_processor':
            import re
            nums = [float(x) for x in re.findall(r'\d+\.?\d*', task)]
            if not nums:
                nums = [1, 2, 3, 4, 5]
            op = 'mean' if '平均' in task else 'sum' if '求和' in task else 'sort'
            return {'data': nums, 'operation': op}
        return {}

    def run(self, task, verbose=True):
        """运行Agent循环"""
        if verbose:
            print(f"\n  [{self.name}] 收到任务: {task}")

        self.conversation_history.append({'role': 'user', 'content': task})
        iteration_results = []

        for iteration in range(self.max_iterations):
            # Step 1: 选择工具
            tool_name = self.select_tool(task)

            if tool_name is None:
                answer = f"直接回答: 关于'{task}', 这超出了我当前工具的能力范围。"
                iteration_results.append({'type': 'direct', 'answer': answer})
                break

            # Step 2: 提取参数
            params = self.extract_params(tool_name, task)

            # Step 3: 执行工具
            tool = self.tools[tool_name]
            result = tool.execute(**params)

            # 记录追踪
            trace_entry = {
                'iteration': iteration + 1,
                'tool': tool_name,
                'params': params,
                'success': result['success'],
                'attempts': result.get('attempts', 1),
            }

            if result['success']:
                trace_entry['result'] = result['result']
                if verbose:
                    print(f"  [{self.name}] Step {iteration + 1}: 调用 {tool_name} → {result['result']}")
                iteration_results.append({
                    'type': 'tool_call',
                    'tool': tool_name,
                    'result': result['result'],
                })
                # 简化: 一步完成
                break
            else:
                trace_entry['error'] = result['error']
                if verbose:
                    print(f"  [{self.name}] Step {iteration + 1}: {tool_name} 失败 → {result['error']}")
                iteration_results.append({
                    'type': 'error',
                    'tool': tool_name,
                    'error': result['error'],
                })

            self.tool_trace.append(trace_entry)

        self.conversation_history.append({
            'role': 'assistant',
            'content': str(iteration_results),
        })
        return iteration_results

    def run_chain(self, tasks):
        """运行工具调用链"""
        print(f"\n  [{self.name}] === 工具调用链 ===")
        chain_results = []
        for i, task in enumerate(tasks):
            print(f"\n  --- 链步骤 {i + 1}/{len(tasks)} ---")
            result = self.run(task, verbose=True)
            chain_results.append({
                'step': i + 1,
                'task': task,
                'result': result,
            })
        return chain_results

    def get_tool_stats(self):
        """获取工具使用统计"""
        stats = {}
        for name, tool in self.tools.items():
            stats[name] = tool.get_stats()
        return stats


# ============================================================
# 3. 测试多工具Agent
# ============================================================
print("=" * 60)
print("1. 多工具Agent测试")
print("=" * 60)

agent = MultiToolAgent("智能助手")
agent.register_tool(Tool('calculator', calculator, '数学计算器',
                         {'expression': 'string'}))
agent.register_tool(Tool('search', search, '知识搜索',
                         {'query': 'string'}))
agent.register_tool(Tool('unit_converter', unit_converter, '单位转换',
                         {'value': 'float', 'from_unit': 'string', 'to_unit': 'string'}))
agent.register_tool(Tool('text_analyzer', text_analyzer, '文本分析',
                         {'text': 'string'}))
agent.register_tool(Tool('data_processor', data_processor, '数据处理',
                         {'data': 'array', 'operation': 'string'}))

# 单任务测试
single_tasks = [
    "计算 (15 + 27) * 3",
    "搜索什么是Agent",
    "分析文本 这是一段测试文本, 用来分析。",
]

print("\n  === 单任务测试 ===")
for task in single_tasks:
    agent.run(task)

# 工具调用链测试
chain_tasks = [
    "计算 100 * 5",
    "搜索什么是AI",
    "计算 200 + 300",
]

print("\n  === 工具调用链 ===")
chain_results = agent.run_chain(chain_tasks)


# ============================================================
# 4. 错误处理与重试分析
# ============================================================
print("\n" + "=" * 60)
print("2. 错误处理与重试分析")
print("=" * 60)

# 模拟不稳定工具
def unstable_tool(data):
    """模拟不稳定的工具"""
    if np.random.random() < 0.4:
        raise ConnectionError("网络超时")
    return f"处理完成: {data}"

unstable = Tool('unstable', unstable_tool, '不稳定工具', {'data': 'string'}, max_retries=5)
agent.register_tool(unstable)

# 运行多次
successes = 0
total_attempts = 0
for i in range(20):
    result = unstable.execute(data=f"test_{i}")
    total_attempts += result['attempts']
    if result['success']:
        successes += 1

print(f"  不稳定工具测试 (20次调用):")
print(f"    成功率: {successes / 20:.1%}")
print(f"    平均重试次数: {total_attempts / 20:.1f}")


# ============================================================
# 5. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 5.1 工具调用统计
ax1 = axes[0, 0]
stats = agent.get_tool_stats()
tool_names = [s['name'] for n, s in stats.items()]
call_counts = [s['calls'] for n, s in stats.items()]
error_counts = [s['errors'] for n, s in stats.items()]
x = np.arange(len(tool_names))
width = 0.35
ax1.bar(x - width / 2, call_counts, width, label='调用次数', color='#2196F3', edgecolor='black')
ax1.bar(x + width / 2, error_counts, width, label='错误次数', color='#F44336', edgecolor='black')
ax1.set_xticks(x)
ax1.set_xticklabels([n[:8] for n in tool_names], rotation=30)
ax1.set_ylabel('次数')
ax1.set_title('工具调用统计')
ax1.legend()

# 5.2 Agent执行流程
ax2 = axes[0, 1]
if chain_results:
    steps = [f"Step {r['step']}" for r in chain_results]
    task_labels = [r['task'][:20] for r in chain_results]
    for i, (step, label) in enumerate(zip(steps, task_labels)):
        color = '#4CAF50' if r['result'] and r['result'][0].get('success', True) else '#FF9800'
        ax2.barh(i, 1, color=f'C{i}', alpha=0.7, edgecolor='black')
        ax2.text(0.5, i, label, ha='center', va='center', fontsize=9)
        if i < len(steps) - 1:
            ax2.annotate('', xy=(0.5, i + 0.85), xytext=(0.5, i + 0.15),
                         arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax2.set_yticks(range(len(steps)))
    ax2.set_yticklabels(steps)
    ax2.set_xlim(0, 1)
    ax2.set_title('工具调用链流程')
    ax2.set_xticks([])

# 5.3 工具选择准确率模拟
ax3 = axes[1, 0]
test_data = {
    '计算类': {'正确': 18, '错误': 2},
    '搜索类': {'正确': 16, '错误': 4},
    '转换类': {'正确': 15, '错误': 5},
    '分析类': {'正确': 17, '错误': 3},
    '处理类': {'正确': 14, '错误': 6},
}
categories = list(test_data.keys())
correct = [test_data[c]['正确'] for c in categories]
incorrect = [test_data[c]['错误'] for c in categories]
x = np.arange(len(categories))
ax3.bar(x, correct, label='选择正确', color='#4CAF50', edgecolor='black')
ax3.bar(x, incorrect, bottom=correct, label='选择错误', color='#F44336', edgecolor='black')
ax3.set_xticks(x)
ax3.set_xticklabels(categories)
ax3.set_ylabel('次数')
ax3.set_title('工具选择准确率 (20次测试)')
ax3.legend()

# 5.4 重试策略效果对比
ax4 = axes[1, 1]
retry_strategies = ['无重试', '重试1次', '重试3次', '重试5次', '指数退避']
base_success = 0.6
success_rates = [
    base_success,
    1 - (1 - base_success) ** 2,
    1 - (1 - base_success) ** 4,
    1 - (1 - base_success) ** 6,
    1 - (1 - base_success) ** 6 * 0.8,  # 指数退避更优
]
colors = ['#F44336', '#FF9800', '#FFC107', '#8BC34A', '#4CAF50']
bars = ax4.bar(retry_strategies, success_rates, color=colors, edgecolor='black')
ax4.set_ylabel('最终成功率')
ax4.set_title('不同重试策略的效果 (基础成功率60%)')
ax4.set_ylim(0, 1.1)
for bar, rate in zip(bars, success_rates):
    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
             f'{rate:.1%}', ha='center', fontsize=10)
ax4.set_xticklabels(retry_strategies, rotation=20)

plt.suptitle('W22-D5: 多工具Agent', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W22/d5_multi_tool_agent.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存!")
