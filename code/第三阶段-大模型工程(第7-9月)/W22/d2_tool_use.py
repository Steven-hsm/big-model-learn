### Day 2（周二）：工具定义与调用
# JSON Schema工具定义, 工具注册器, 常用工具实现, Function Calling概念

import numpy as np
import matplotlib.pyplot as plt
import json

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 工具定义模式 (JSON Schema)
# ============================================================
print("=" * 60)
print("1. 工具定义模式 (JSON Schema)")
print("=" * 60)

# OpenAI Function Calling风格的工具定义
tool_definitions = {
    "calculator": {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "执行数学计算, 支持加减乘除",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式, 如 '2 + 3 * 4'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    "search": {
        "type": "function",
        "function": {
            "name": "search",
            "description": "搜索互联网获取信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "返回结果数量, 默认5",
                    },
                },
                "required": ["query"],
            },
        },
    },
    "code_executor": {
        "type": "function",
        "function": {
            "name": "code_executor",
            "description": "执行Python代码并返回结果",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "要执行的Python代码",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "超时时间(秒), 默认30",
                    },
                },
                "required": ["code"],
            },
        },
    },
}

for name, definition in tool_definitions.items():
    func = definition["function"]
    print(f"\n  工具: {func['name']}")
    print(f"    描述: {func['description']}")
    params = func['parameters']['properties']
    required = func['parameters'].get('required', [])
    for pname, pinfo in params.items():
        req = " (必填)" if pname in required else " (可选)"
        print(f"    参数: {pname} [{pinfo['type']}]{req} - {pinfo['description']}")


# ============================================================
# 2. 工具注册器实现
# ============================================================
class ToolRegistry:
    """工具注册与管理器"""

    def __init__(self):
        self.tools = {}

    def register(self, name, func, description, parameters_schema):
        """注册工具"""
        self.tools[name] = {
            'func': func,
            'description': description,
            'parameters': parameters_schema,
        }
        print(f"  已注册工具: {name}")

    def unregister(self, name):
        """注销工具"""
        if name in self.tools:
            del self.tools[name]
            print(f"  已注销工具: {name}")

    def get_tool(self, name):
        """获取工具"""
        return self.tools.get(name)

    def list_tools(self):
        """列出所有工具"""
        print("\n  已注册工具列表:")
        for name, info in self.tools.items():
            params = info['parameters'].get('properties', {})
            param_str = ', '.join(params.keys())
            print(f"    {name}({param_str}) - {info['description']}")

    def execute(self, name, **kwargs):
        """执行工具"""
        if name not in self.tools:
            return {'error': f'工具 {name} 不存在'}
        try:
            result = self.tools[name]['func'](**kwargs)
            return {'success': True, 'result': result, 'tool': name}
        except Exception as e:
            return {'success': False, 'error': str(e), 'tool': name}

    def select_tool(self, task):
        """根据任务选择合适的工具 (简化版)"""
        task_lower = task.lower()
        scores = {}

        for name, info in self.tools.items():
            score = 0
            desc = info['description'].lower()
            # 关键词匹配
            keywords = desc.replace('执行', '').replace('搜索', '').replace('获取', '').split()
            for kw in keywords:
                if kw in task_lower:
                    score += 1
            # 参数名匹配
            for param in info['parameters'].get('properties', {}):
                if param in task_lower:
                    score += 0.5
            scores[name] = score

        best_tool = max(scores, key=scores.get)
        if scores[best_tool] > 0:
            return best_tool
        return None


# ============================================================
# 3. 实现常用工具
# ============================================================
print("\n" + "=" * 60)
print("2. 工具注册器与常用工具")
print("=" * 60)

def calculator(expression):
    """计算器工具"""
    clean = ''.join(c for c in expression if c in '0123456789+-*/(). ')
    if clean:
        return eval(clean)
    raise ValueError("无法解析表达式")

def search(query):
    """搜索工具 (模拟)"""
    mock_results = {
        'python': ['Python官方网站', 'Python教程 - 菜鸟教程', 'Python最佳实践'],
        'machine learning': ['机器学习入门', 'Scikit-learn文档', 'ML课程 - Coursera'],
        'agent': ['AI Agent开发指南', 'LangChain文档', 'AutoGPT项目'],
    }
    results = []
    for key, values in mock_results.items():
        if key in query.lower():
            results = values
            break
    return results if results else [f"搜索 '{query}' - 未找到相关结果"]

def code_executor(code):
    """代码执行器 (模拟 - 不实际执行)"""
    return f"[模拟执行] 代码长度: {len(code)}字符, 预期输出: OK"


# 注册工具
registry = ToolRegistry()
registry.register('calculator', calculator, '执行数学计算',
                  {'properties': {'expression': {'type': 'string'}}})
registry.register('search', search, '搜索互联网获取信息',
                  {'properties': {'query': {'type': 'string'}}})
registry.register('code_executor', code_executor, '执行Python代码',
                  {'properties': {'code': {'type': 'string'}}})

registry.list_tools()

# 测试工具调用
print("\n  工具调用测试:")
test_calls = [
    ('calculator', {'expression': '(10 + 5) * 3'}),
    ('search', {'query': 'python tutorial'}),
    ('code_executor', {'code': 'print("Hello")'}),
]

for tool_name, params in test_calls:
    result = registry.execute(tool_name, **params)
    print(f"\n    调用: {tool_name}({params})")
    print(f"    结果: {result}")

# 测试工具选择
print("\n  工具自动选择测试:")
test_tasks = [
    "帮我计算 25 * 4",
    "搜索关于machine learning的资料",
    "执行这段代码: print('hello')",
    "今天天气怎么样",  # 无匹配工具
]

for task in test_tasks:
    selected = registry.select_tool(task)
    print(f"    任务: '{task}' → 选择工具: {selected}")


# ============================================================
# 4. Function Calling流程模拟
# ============================================================
print("\n" + "=" * 60)
print("3. Function Calling流程模拟")
print("=" * 60)

class FunctionCallingSimulator:
    """模拟OpenAI Function Calling流程"""

    def __init__(self, registry):
        self.registry = registry
        self.conversation = []

    def chat(self, user_message):
        """模拟一次对话"""
        print(f"\n  用户: {user_message}")

        # Step 1: 模型判断是否需要调用函数
        tool_name = self.registry.select_tool(user_message)

        if tool_name:
            # Step 2: 模型决定调用哪个函数及参数
            print(f"  模型决策: 需要调用工具 '{tool_name}'")

            # 模拟参数提取
            params = self._extract_params(tool_name, user_message)
            print(f"  提取参数: {params}")

            # Step 3: 执行工具
            result = self.registry.execute(tool_name, **params)
            print(f"  工具返回: {result}")

            # Step 4: 模型基于工具结果生成最终回答
            answer = f"根据工具调用结果, {user_message}的答案是: {result.get('result', result)}"
            print(f"  模型回答: {answer}")
        else:
            answer = f"我可以直接回答: 关于'{user_message}', 这是一个通用问题。"
            print(f"  模型回答: {answer}")

        self.conversation.append({
            'user': user_message,
            'tool_call': tool_name,
            'answer': answer,
        })
        return answer

    def _extract_params(self, tool_name, message):
        """模拟参数提取"""
        import re
        if tool_name == 'calculator':
            clean = ''.join(c for c in message if c in '0123456789+-*/(). ')
            return {'expression': clean.strip()}
        elif tool_name == 'search':
            return {'query': message}
        elif tool_name == 'code_executor':
            return {'code': message}
        return {}


simulator = FunctionCallingSimulator(registry)
simulator.chat("帮我计算 100 + 200 * 3")
simulator.chat("搜索python相关资料")


# ============================================================
# 5. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 5.1 工具调用统计
ax1 = axes[0, 0]
tool_names_list = list(registry.tools.keys())
call_counts = [5, 8, 3]  # 模拟调用次数
colors = ['#4CAF50', '#2196F3', '#FF9800']
ax1.bar(tool_names_list, call_counts, color=colors, edgecolor='black')
ax1.set_ylabel('调用次数')
ax1.set_title('工具调用频率统计')
for i, v in enumerate(call_counts):
    ax1.text(i, v + 0.2, str(v), ha='center', fontsize=12)

# 5.2 Function Calling流程图
ax2 = axes[0, 1]
steps_fc = ['用户输入', '模型判断\n(是否调用工具)', '工具调用', '结果返回', '最终回答']
y_pos = list(range(len(steps_fc)))
step_colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#4CAF50']

for i, (step, color) in enumerate(zip(steps_fc, step_colors)):
    ax2.barh(i, 1, color=color, alpha=0.7, edgecolor='black')
    ax2.text(0.5, i, step, ha='center', va='center', fontsize=10, fontweight='bold')

ax2.set_yticks([])
ax2.set_xlim(0, 1)
ax2.set_title('Function Calling 流程')
ax2.set_xticks([])

# 5.3 工具选择准确率
ax3 = axes[1, 0]
selection_tasks = {
    '正确选择': 12,
    '错误选择': 2,
    '无法选择': 3,
}
labels = list(selection_tasks.keys())
sizes = list(selection_tasks.values())
colors_pie = ['#4CAF50', '#F44336', '#FF9800']
ax3.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.0f%%',
        startangle=90, textprops={'fontsize': 11})
ax3.set_title('工具选择准确率')

# 5.4 工具参数类型分布
ax4 = axes[1, 1]
param_types = ['string', 'integer', 'float', 'boolean', 'array', 'object']
type_counts = [15, 8, 5, 3, 4, 2]
ax4.bar(param_types, type_counts, color='#9C27B0', edgecolor='black', alpha=0.7)
ax4.set_ylabel('出现次数')
ax4.set_title('工具参数类型分布 (常见API)')
ax4.set_xticklabels(param_types, rotation=45)

plt.suptitle('W22-D2: 工具定义与调用', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W22/d2_tool_use.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存!")
