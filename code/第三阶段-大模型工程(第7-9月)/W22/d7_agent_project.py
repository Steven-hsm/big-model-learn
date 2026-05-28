### Day 7（周日）：Agent项目 - 智能助手
# 搜索+计算+文档查询, 多轮对话, 工具调用追踪, 错误恢复

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import deque

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 智能助手系统
# ============================================================
class SmartAssistant:
    """智能助手: 集成多种工具, 支持多轮对话"""

    def __init__(self, name="AI助手"):
        self.name = name
        self.tools = {}
        self.conversation = deque(maxlen=20)
        self.tool_trace = []
        self.error_log = []
        self.session_stats = {
            'total_queries': 0,
            'tool_calls': 0,
            'errors': 0,
            'recoveries': 0,
        }

    # ---------- 工具管理 ----------
    def register_tool(self, name, func, description, keywords):
        self.tools[name] = {
            'func': func,
            'description': description,
            'keywords': keywords,
        }

    def _select_tool(self, query):
        """智能工具选择"""
        query_lower = query.lower()
        best_tool = None
        best_score = 0

        for name, info in self.tools.items():
            score = sum(1 for kw in info['keywords'] if kw in query_lower)
            if score > best_score:
                best_score = score
                best_tool = name

        return best_tool if best_score > 0 else None

    # ---------- 工具函数 ----------
    @staticmethod
    def _calculator(expression):
        clean = ''.join(c for c in str(expression) if c in '0123456789+-*/(). ')
        if clean:
            return {'result': eval(clean), 'expression': clean}
        raise ValueError("无法解析表达式")

    @staticmethod
    def _search(query):
        knowledge_base = {
            'python': 'Python是一种广泛使用的高级编程语言',
            '机器学习': '机器学习是AI的分支, 让计算机从数据中学习模式',
            '深度学习': '深度学习使用多层神经网络进行特征学习',
            'transformer': 'Transformer是基于自注意力机制的序列模型架构',
            'llm': '大语言模型(LLM)是基于Transformer的大规模预训练模型',
            'agent': 'AI Agent是能够自主感知、规划和行动的智能体',
            'rag': 'RAG(检索增强生成)通过外部知识增强LLM的回答质量',
        }
        results = []
        for key, value in knowledge_base.items():
            if key in query.lower():
                results.append({'topic': key, 'content': value})
        if not results:
            results.append({'topic': 'general', 'content': f'关于"{query}"的通用回答'})
        return results

    @staticmethod
    def _doc_query(query):
        """文档查询 (模拟RAG)"""
        docs = [
            {'title': 'Python入门指南', 'content': 'Python基础语法包括变量、循环、函数等'},
            {'title': 'NumPy教程', 'content': 'NumPy是Python的科学计算库, 支持多维数组'},
            {'title': 'Pandas教程', 'content': 'Pandas提供DataFrame数据结构, 用于数据分析'},
            {'title': '机器学习概述', 'content': '机器学习分为监督学习、无监督学习和强化学习'},
            {'title': '深度学习框架', 'content': '主流框架包括PyTorch、TensorFlow和JAX'},
        ]
        results = []
        for doc in docs:
            if any(kw in query.lower() for kw in doc['title'].lower().split()):
                results.append(doc)
        if not results:
            results = docs[:2]  # 默认返回前两个
        return results

    @staticmethod
    def _unit_convert(params):
        conversions = {
            ('km', 'mile'): 0.621371, ('mile', 'km'): 1.60934,
            ('kg', 'lb'): 2.20462, ('lb', 'kg'): 0.453592,
            ('c', 'f'): lambda x: x * 9 / 5 + 32,
        }
        return {'value': params.get('value', 0), 'converted': '模拟转换结果'}

    # ---------- 核心: 对话处理 ----------
    def chat(self, user_input):
        """处理用户输入"""
        self.session_stats['total_queries'] += 1
        self.conversation.append({
            'role': 'user',
            'content': user_input,
            'time': datetime.now().isoformat(),
        })

        # 选择工具
        tool_name = self._select_tool(user_input)

        response = {
            'query': user_input,
            'tool_used': tool_name,
            'result': None,
            'error': None,
            'recovered': False,
        }

        if tool_name:
            try:
                tool_result = self._execute_tool(tool_name, user_input)
                response['result'] = tool_result
                self.session_stats['tool_calls'] += 1

                # 记录追踪
                self.tool_trace.append({
                    'query': user_input,
                    'tool': tool_name,
                    'success': True,
                    'time': datetime.now().isoformat(),
                })
            except Exception as e:
                response['error'] = str(e)
                self.session_stats['errors'] += 1

                # 错误恢复: 尝试备用方案
                recovery_result = self._recover(tool_name, user_input)
                if recovery_result:
                    response['result'] = recovery_result
                    response['recovered'] = True
                    self.session_stats['recoveries'] += 1

                self.error_log.append({
                    'tool': tool_name,
                    'error': str(e),
                    'recovered': response['recovered'],
                    'time': datetime.now().isoformat(),
                })

                self.tool_trace.append({
                    'query': user_input,
                    'tool': tool_name,
                    'success': False,
                    'recovered': response['recovered'],
                })
        else:
            response['result'] = f"我理解您的问题: '{user_input}'。让我直接回答..."

        # 生成最终回复
        final_answer = self._format_response(response)

        self.conversation.append({
            'role': 'assistant',
            'content': final_answer,
            'tool': tool_name,
            'time': datetime.now().isoformat(),
        })

        return final_answer

    def _execute_tool(self, tool_name, query):
        """执行工具"""
        tool = self.tools[tool_name]
        return tool['func'](query)

    def _recover(self, failed_tool, query):
        """错误恢复策略"""
        # 策略1: 使用备用工具
        fallback_map = {
            'search': 'doc_query',
            'doc_query': 'search',
        }
        fallback = fallback_map.get(failed_tool)
        if fallback and fallback in self.tools:
            try:
                return self._execute_tool(fallback, query)
            except Exception:
                pass

        # 策略2: 返回通用回答
        return f"[备用回答] 关于'{query}', 请参考相关文档获取详细信息。"

    def _format_response(self, response):
        """格式化回复"""
        if response['result'] is not None:
            if isinstance(response['result'], dict):
                return f"结果: {response['result']}"
            elif isinstance(response['result'], list):
                items = '\n'.join(f"  - {r}" for r in response['result'][:3])
                return f"找到以下信息:\n{items}"
            return str(response['result'])
        elif response['error']:
            prefix = "[已恢复] " if response['recovered'] else "[失败] "
            return f"{prefix}{response['error']}"
        return "我理解您的问题。"

    def get_session_report(self):
        """生成会话报告"""
        stats = self.session_stats
        print(f"\n{'=' * 50}")
        print(f"  会话报告 - {self.name}")
        print(f"{'=' * 50}")
        print(f"  总查询数: {stats['total_queries']}")
        print(f"  工具调用: {stats['tool_calls']}")
        print(f"  错误次数: {stats['errors']}")
        print(f"  成功恢复: {stats['recoveries']}")
        if stats['total_queries'] > 0:
            print(f"  工具使用率: {stats['tool_calls'] / stats['total_queries']:.1%}")
        if stats['errors'] > 0:
            print(f"  错误恢复率: {stats['recoveries'] / stats['errors']:.1%}")

        print(f"\n  对话历史 ({len(self.conversation)}条):")
        for msg in list(self.conversation)[-6:]:
            role = msg['role']
            content = msg['content'][:50]
            tool = msg.get('tool', '')
            tool_str = f" [{tool}]" if tool else ""
            print(f"    [{role:>9}] {content}...{tool_str}")

        print(f"\n  工具调用追踪 ({len(self.tool_trace)}条):")
        for trace in self.tool_trace:
            status = "OK" if trace['success'] else "FAIL"
            print(f"    [{status}] {trace['tool']}: {trace['query'][:30]}")

        return stats


# ============================================================
# 2. 运行智能助手
# ============================================================
print("=" * 60)
print("1. 智能助手运行测试")
print("=" * 60)

assistant = SmartAssistant("学习助手")

# 注册工具
assistant.register_tool('calculator', assistant._calculator, '数学计算器',
                        ['计算', '加', '减', '乘', '除', '=', '+', '-', '*', '/'])
assistant.register_tool('search', assistant._search, '知识搜索',
                        ['搜索', '查询', '什么是', '了��', '解释'])
assistant.register_tool('doc_query', assistant._doc_query, '文档查询',
                        ['文档', '教程', '怎么用', '如何使用'])
assistant.register_tool('unit_convert', assistant._unit_convert, '单位转换',
                        ['转换', '换算', '公里', '英里'])

# 多轮对话测试
print("\n  === 多轮对话测试 ===\n")
conversation_flow = [
    "你好, 我想学习Python",
    "搜索什么是机器学习",
    "计算 (128 + 256) * 2",
    "查询Pandas教程文档",
    "搜索transformer是什么",
    "帮我计算 1024 / 8",
    "什么是RAG技术?",
    "文档里有没有深度学习的教程?",
]

for user_input in conversation_flow:
    print(f"  用户: {user_input}")
    response = assistant.chat(user_input)
    print(f"  助手: {str(response)[:80]}")
    print()


# ============================================================
# 3. 错误恢复测试
# ============================================================
print("=" * 60)
print("2. 错误恢复测试")
print("=" * 60)

# 模拟错误场景
error_scenarios = [
    "搜索一个完全不存在的话题xyz",
    "计算空表达式",
    "查询不存在的文档类型",
]

for scenario in error_scenarios:
    print(f"\n  输入: {scenario}")
    response = assistant.chat(scenario)
    print(f"  回复: {str(response)[:80]}")

# 生成会话报告
assistant.get_session_report()


# ============================================================
# 4. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 4.1 工具调用分布
ax1 = axes[0, 0]
tool_names = ['calculator', 'search', 'doc_query', 'unit_convert']
tool_calls = [sum(1 for t in assistant.tool_trace if t['tool'] == name) for name in tool_names]
colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']
ax1.pie(tool_calls if any(tool_calls) else [1, 1, 1, 1],
        labels=tool_names, colors=colors, autopct='%1.0f%%',
        startangle=90, textprops={'fontsize': 9})
ax1.set_title('工具调用分布')

# 4.2 对话流程时间线
ax2 = axes[0, 1]
messages = list(assistant.conversation)
if messages:
    user_msgs = [(i, m['content'][:25]) for i, m in enumerate(messages) if m['role'] == 'user']
    asst_msgs = [(i, m.get('tool', 'direct')) for i, m in enumerate(messages) if m['role'] == 'assistant']

    for i, (idx, text) in enumerate(user_msgs):
        ax2.barh(idx, 1, color='#4CAF50', alpha=0.7, edgecolor='black')
        ax2.text(0.5, idx, f'U: {text}', ha='center', va='center', fontsize=7)

    for i, (idx, tool) in enumerate(asst_msgs):
        ax2.barh(idx, 1, color='#2196F3', alpha=0.7, edgecolor='black')
        ax2.text(0.5, idx, f'A: [{tool}]', ha='center', va='center', fontsize=7)

ax2.set_xlabel('消息')
ax2.set_ylabel('消息序号')
ax2.set_title('对话流程 (绿=用户, 蓝=助手)')
ax2.set_xticks([])

# 4.3 会话统计
ax3 = axes[1, 0]
stats = assistant.session_stats
stat_names = ['总查询', '工具调用', '错误', '恢复']
stat_values = [stats['total_queries'], stats['tool_calls'], stats['errors'], stats['recoveries']]
colors_stat = ['#2196F3', '#4CAF50', '#F44336', '#FF9800']
bars = ax3.bar(stat_names, stat_values, color=colors_stat, edgecolor='black')
ax3.set_ylabel('次数')
ax3.set_title('会话统计')
for bar, val in zip(bars, stat_values):
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
             str(val), ha='center', fontsize=12)

# 4.4 工具成功率
ax4 = axes[1, 1]
if assistant.tool_trace:
    tools_in_trace = list(set(t['tool'] for t in assistant.tool_trace))
    success_rates = []
    for name in tools_in_trace:
        tool_traces = [t for t in assistant.tool_trace if t['tool'] == name]
        success_rate = sum(1 for t in tool_traces if t['success']) / len(tool_traces)
        success_rates.append(success_rate)

    colors_sr = ['#4CAF50' if r >= 0.8 else '#FF9800' if r >= 0.5 else '#F44336'
                 for r in success_rates]
    ax4.bar(tools_in_trace, success_rates, color=colors_sr, edgecolor='black')
    ax4.set_ylabel('成功率')
    ax4.set_title('各工具调用成功率')
    ax4.set_ylim(0, 1.1)
    for i, (name, rate) in enumerate(zip(tools_in_trace, success_rates)):
        ax4.text(i, rate + 0.03, f'{rate:.0%}', ha='center')
else:
    ax4.text(0.5, 0.5, '无工具调用数据', ha='center', va='center', transform=ax4.transAxes)
    ax4.set_title('工具调用成功率')

plt.suptitle('W22-D7: Agent项目 - 智能助手', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W22/d7_agent_project.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存!")
