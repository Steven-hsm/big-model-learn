### Day 1（周一）：Agent概念与简单实现
# 感知→规划→行动, ReAct模式, 工具调用, Agent vs 传统程序

import numpy as np
import matplotlib.pyplot as plt
import json

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. Agent核心概念
# ============================================================
print("=" * 60)
print("1. Agent核心概念: 感知→规划→行动")
print("=" * 60)

print("""
  Agent = LLM + 记忆 + 规划 + 工具使用

  核心循环 (Agent Loop):
    1. 感知 (Perception): 接收用户输入/环��状态
    2. 规划 (Planning): 分析任务, 制定行动计划
    3. 行动 (Action): 调用工具/生成回答
    4. 观察 (Observation): 获取行动结果
    5. 回到步骤1, 直到任务完成

  Agent vs 传统程序:
    传统程序: 固定逻辑流程, if-else分支
    Agent:    动态决策, 自主选择工具和策略
""")


# ============================================================
# 2. ReAct模式实现
# ============================================================
class SimpleAgent:
    """简化版ReAct Agent"""

    def __init__(self, name="Agent"):
        self.name = name
        self.tools = {}
        self.memory = []
        self.trace = []  # 记录推理过程

    def add_tool(self, name, func, description):
        """注册工具"""
        self.tools[name] = {
            'func': func,
            'description': description,
        }

    def think(self, task):
        """思考: 分析任务并决定下一步"""
        self.trace.append({'type': 'thought', 'content': f'需要完成: {task}'})

        # 简化: 根据关键词选择工具
        task_lower = task.lower()

        if any(kw in task_lower for kw in ['计算', '加', '减', '乘', '除', '+', '-', '*', '/']):
            return 'calculator', {'expression': task}
        elif any(kw in task_lower for kw in ['搜索', '查找', '查询', '什么是']):
            return 'search', {'query': task}
        elif any(kw in task_lower for kw in ['天气']):
            return 'weather', {'location': task}
        elif any(kw in task_lower for kw in ['翻译', 'translate']):
            return 'translator', {'text': task}
        else:
            return 'general', {'prompt': task}

    def act(self, tool_name, params):
        """行动: 调用工具"""
        self.trace.append({'type': 'action', 'tool': tool_name, 'params': params})

        if tool_name in self.tools:
            result = self.tools[tool_name]['func'](**params)
            self.trace.append({'type': 'observation', 'content': str(result)})
            return result
        else:
            fallback = f"直接回答: 关于'{params}', 这是一个通用问题, 我将尽力回答。"
            self.trace.append({'type': 'observation', 'content': fallback})
            return fallback

    def observe(self, result):
        """观察: 处理结果"""
        self.memory.append({'result': result})
        return result

    def run(self, task, max_steps=5):
        """运行Agent循环"""
        self.trace = []
        self.trace.append({'type': 'task', 'content': task})

        for step in range(max_steps):
            # Think
            tool_name, params = self.think(task)
            # Act
            result = self.act(tool_name, params)
            # Observe
            final = self.observe(result)

            # 判断是否完成 (简化: 一步完成)
            self.trace.append({'type': 'answer', 'content': str(final)})
            return final

        return "未能完成任务"


# 定义工具函数
def calculator(expression):
    """计算器工具"""
    # 提取表达式中的数字和运算符
    try:
        # 安全评估简单数学表达式
        clean = ''.join(c for c in expression if c in '0123456789+-*/(). ')
        if clean:
            result = eval(clean)
            return f"计算结果: {clean} = {result}"
        return "无法解析表达式"
    except Exception as e:
        return f"计算错误: {e}"


def search(query):
    """搜索工具 (模拟)"""
    knowledge = {
        'python': 'Python是一种高级编程语言, 由Guido van Rossum于1991年创建',
        'ai': 'AI(人工智能)是计算机科学的一个分支, 致力于创建智能机器',
        'llm': 'LLM(大语言模型)是基于Transformer架构的大规模语言模型',
        'agent': 'AI Agent是能够自主感知、规划和行动的智能系统',
    }
    for key, value in knowledge.items():
        if key in query.lower():
            return f"搜索结果: {value}"
    return f"搜索结果: 未找到与'{query}'相关的信息"


def weather(location):
    """天气工具 (模拟)"""
    return f"[模拟] {location}当前天气: 晴, 25°C, 湿度60%"


def translator(text):
    """翻译工具 (模拟)"""
    return f"[模拟翻译] {text} → This is a simulated translation"


# 创建并配置Agent
print("=" * 60)
print("2. 简单ReAct Agent演示")
print("=" * 60)

agent = SimpleAgent("助手Agent")
agent.add_tool('calculator', calculator, '数学计算器')
agent.add_tool('search', search, '知识搜索')
agent.add_tool('weather', weather, '天气查询')
agent.add_tool('translator', translator, '翻译工具')

# 测试不同任务
tasks = [
    "计算 25 * 4 + 10",
    "搜索什么是LLM",
    "查询北京的天气",
]

for task in tasks:
    print(f"\n  任务: {task}")
    result = agent.run(task)
    print(f"  结果: {result}")
    print(f"  推理过程:")
    for step in agent.trace:
        print(f"    [{step['type']}] {step.get('content', step.get('tool', ''))}")


# ============================================================
# 3. Agent vs 传统程序对比
# ============================================================
print("\n" + "=" * 60)
print("3. Agent vs 传统程序对比")
print("=" * 60)

class TraditionalProgram:
    """传统程序: 固定流程"""

    def process(self, task):
        if '计算' in task:
            return calculator(task)
        elif '搜索' in task:
            return search(task)
        elif '天气' in task:
            return weather(task)
        else:
            return "无法处理此请求"

class AgentProgram:
    """Agent程序: 动态决策"""

    def __init__(self):
        self.agent = SimpleAgent("动态Agent")
        self.agent.add_tool('calculator', calculator, '计算器')
        self.agent.add_tool('search', search, '搜索')
        self.agent.add_tool('weather', weather, '天气')
        self.agent.add_tool('translator', translator, '翻译')

    def process(self, task):
        return self.agent.run(task)


comparison_tasks = [
    ("计算 100 / 4", "简单匹配即可"),
    ("帮我算一下 3 * 7 的结果", "需要理解'算一下'等于'计算'"),
    ("今天天气怎么样", "隐式天气查询, 无'天气'关键词"),
    ("翻译 Hello World", "传统程序可能无法处理"),
]

trad = TraditionalProgram()
agent_prog = AgentProgram()

print(f"\n  {'任务':<25} {'传统程序':<30} {'Agent':<30}")
print(f"  {'-' * 85}")
for task, note in comparison_tasks:
    t_result = trad.process(task)
    a_result = agent_prog.process(task)
    print(f"  {task:<25} {str(t_result)[:28]:<30} {str(a_result)[:28]:<30}")
    print(f"  {'':25} 注: {note}")


# ============================================================
# 4. Agent架构模式分析
# ============================================================
print("\n" + "=" * 60)
print("4. Agent架构模式")
print("=" * 60)

patterns = {
    'ReAct': {
        '流程': 'Thought → Action → Observation → ... → Answer',
        '优点': '推理过程可解释, 灵活',
        '缺点': '可能陷入循环, 效率较低',
    },
    'Plan-and-Execute': {
        '流程': 'Plan(完整计划) ��� Execute(逐步执行) → Replan(如有需要)',
        '优点': '全局视角, 适合复杂任务',
        '缺点': '计划可能需要频繁调整',
    },
    'Reflexion': {
        '流程': 'Act → Evaluate → Reflect → Improve',
        '优点': '自我改进能力强',
        '缺点': '迭代成本高',
    },
    'LATS': {
        '流程': '树搜索 + 价值评估 + 回溯',
        '优点': '探索多种可能性',
        '缺点': '计算量大',
    },
}

for pattern, info in patterns.items():
    print(f"\n  {pattern}:")
    print(f"    流程: {info['流程']}")
    print(f"    优点: {info['优点']}")
    print(f"    缺点: {info['缺点']}")


# ============================================================
# 5. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 5.1 Agent循环示意图
ax1 = axes[0, 0]
steps = ['感知\n(Perception)', '规划\n(Planning)', '行动\n(Action)', '观察\n(Observation)']
positions = [(0.5, 0.8), (0.2, 0.3), (0.5, 0.0), (0.8, 0.3)]
colors_steps = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']

for i, (step, pos) in enumerate(zip(steps, positions)):
    ax1.add_patch(plt.Circle(pos, 0.12, color=colors_steps[i], alpha=0.7))
    ax1.text(pos[0], pos[1], step, ha='center', va='center', fontsize=8, fontweight='bold')

# 箭头
for i in range(len(positions)):
    start = positions[i]
    end = positions[(i + 1) % len(positions)]
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dist = np.sqrt(dx ** 2 + dy ** 2)
    ax1.annotate('', xy=end, xytext=start,
                 arrowprops=dict(arrowstyle='->', color='black', lw=2))

ax1.set_xlim(-0.1, 1.1)
ax1.set_ylim(-0.2, 1.0)
ax1.set_title('Agent核心循环')
ax1.set_aspect('equal')
ax1.axis('off')

# 5.2 ReAct推理过程可视化
ax2 = axes[0, 1]
example_trace = [
    ('Thought', '需要查询北京的天气'),
    ('Action', '调用weather工具'),
    ('Observation', '北京: 晴, 25°C'),
    ('Thought', '获得了天气信息'),
    ('Answer', '北京今天是晴天, 25°C'),
]
y_positions = range(len(example_trace))
colors_trace = {'Thought': '#2196F3', 'Action': '#FF9800',
                'Observation': '#9C27B0', 'Answer': '#4CAF50'}

for i, (step_type, content) in enumerate(example_trace):
    color = colors_trace.get(step_type, 'gray')
    ax2.barh(i, 1, color=color, alpha=0.7, edgecolor='black')
    ax2.text(0.5, i, f'{step_type}: {content}', ha='center', va='center', fontsize=9)

ax2.set_yticks([])
ax2.set_xlim(0, 1)
ax2.set_title('ReAct推理过程示例')
ax2.set_xticks([])

# 5.3 Agent vs 传统程序能力对比
ax3 = axes[1, 0]
categories = ['精确匹配', '语义理解', '灵活性', '可扩展', '可解释']
trad_scores = [5, 2, 1, 2, 4]
agent_scores = [3, 4, 5, 5, 4]

x = np.arange(len(categories))
width = 0.35
ax3.bar(x - width / 2, trad_scores, width, label='传统程序', color='#FF9800', edgecolor='black')
ax3.bar(x + width / 2, agent_scores, width, label='Agent', color='#2196F3', edgecolor='black')
ax3.set_ylabel('能力分数 (1-5)')
ax3.set_title('Agent vs 传统程序')
ax3.set_xticks(x)
ax3.set_xticklabels(categories)
ax3.legend()
ax3.set_ylim(0, 6)

# 5.4 Agent架构模式适用场景
ax4 = axes[1, 1]
pattern_names = ['ReAct', 'Plan-Execute', 'Reflexion', 'LATS']
task_complexity = [3, 5, 4, 5]  # 适合的任务复杂度
flexibility = [4, 3, 5, 5]      # 灵活性
efficiency = [3, 4, 2, 2]       # 效率

x = np.arange(len(pattern_names))
width = 0.25
ax4.bar(x - width, task_complexity, width, label='任务复杂度', color='#4CAF50', edgecolor='black')
ax4.bar(x, flexibility, width, label='灵活性', color='#2196F3', edgecolor='black')
ax4.bar(x + width, efficiency, width, label='效率', color='#FF9800', edgecolor='black')
ax4.set_ylabel('分数 (1-5)')
ax4.set_title('Agent架构模式对比')
ax4.set_xticks(x)
ax4.set_xticklabels(pattern_names)
ax4.legend()
ax4.set_ylim(0, 6)

plt.suptitle('W22-D1: Agent概念与简单实现', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W22/d1_agent_concept.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存!")
