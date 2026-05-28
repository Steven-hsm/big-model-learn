### Day 1（周一）：Multi-Agent系统概念
# 层级/协作/辩论架构, 通信协议, 角色定义, 与单Agent对比

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. Multi-Agent架构模式
# ============================================================
print("=" * 60)
print("1. Multi-Agent架构模式")
print("=" * 60)

architectures = {
    '层级式 (Hierarchical)': {
        '描述': 'Manager分配任务给Worker, 汇总结果',
        '优点': '结构清晰, 易于管理',
        '缺点': '单点瓶颈, Manager故障则系统瘫痪',
        '适用': '任务明确的工作流',
    },
    '协作式 (Collaborative)': {
        '描述': 'Agent之间平等协作, 共享信息',
        '优点': '灵活, 去中心化',
        '缺点': '协调复杂, 可能冲突',
        '适用': '创意性任务, 头脑风暴',
    },
    '辩论式 (Debate)': {
        '描述': 'Agent从不同角度辩论, Judge评判',
        '优点': '多角度分析, 减少偏见',
        '缺点': '耗时, 可能无结论',
        '适用': '决策分析, 代码审查',
    },
    '流水线 (Pipeline)': {
        '描述': '任务按顺序流经多个Agent',
        '优点': '高效, 每个Agent专注一件事',
        '缺点': '串行瓶颈, 错误传播',
        '适用': '内容生产, 数据处理',
    },
}

for arch, info in architectures.items():
    print(f"\n  {arch}:")
    print(f"    描述: {info['描述']}")
    print(f"    优点: {info['优点']}")
    print(f"    缺点: {info['缺点']}")
    print(f"    适用: {info['适用']}")


# ============================================================
# 2. Agent角色定义
# ============================================================
class AgentRole:
    """Agent角色定义"""

    def __init__(self, name, role_description, capabilities, system_prompt):
        self.name = name
        self.role_description = role_description
        self.capabilities = capabilities
        self.system_prompt = system_prompt
        self.message_history = []

    def receive(self, message):
        """接收消息"""
        self.message_history.append(message)

    def respond(self, task):
        """模拟响应"""
        # 简化: 根据角色返回预设回答
        response = f"[{self.name}] 处理: {task[:30]}..."
        self.message_history.append({'role': 'self', 'content': response})
        return response

    def __repr__(self):
        return f"AgentRole({self.name}, caps={self.capabilities})"


print("\n" + "=" * 60)
print("2. Agent角色定义")
print("=" * 60)

# 定义常见角色
roles = [
    AgentRole("Manager", "任务分配和结果汇总",
              ["规划", "分配", "审核"], "你是项目经理, 负责任务分配和质量把控"),
    AgentRole("Researcher", "信息搜索和研究",
              ["搜索", "分析", "总结"], "你是研究员, 负责收集和分析信息"),
    AgentRole("Writer", "内容创作",
              ["写作", "编辑", "校对"], "你是作家, 负责撰写高质量内容"),
    AgentRole("Reviewer", "质量审核",
              ["审核", "评分", "反馈"], "你是审核员, 负责检查质量并给出改进建议"),
    AgentRole("Coder", "代码开发和调试",
              ["编码", "调试", "测试"], "你是程序员, 负责编写和调试代码"),
]

for role in roles:
    print(f"\n  {role}")
    print(f"    描述: {role.role_description}")
    print(f"    能力: {', '.join(role.capabilities)}")


# ============================================================
# 3. 通信协议
# ============================================================
class Message:
    """Agent间通信消息"""

    def __init__(self, sender, receiver, content, msg_type='info', metadata=None):
        self.sender = sender
        self.receiver = receiver
        self.content = content
        self.msg_type = msg_type  # info, task, result, feedback, broadcast
        self.metadata = metadata or {}
        self.timestamp = 0  # 简化时间戳

    def __repr__(self):
        return f"Msg({self.sender}→{self.receiver}: {self.msg_type})"


class CommunicationBus:
    """Agent间通信总线"""

    def __init__(self):
        self.message_queue = []
        self.message_log = []
        self.agent_registry = {}

    def register(self, agent):
        """注册Agent"""
        self.agent_registry[agent.name] = agent

    def send(self, message):
        """发送消息"""
        self.message_queue.append(message)
        self.message_log.append(message)

    def broadcast(self, sender, content, msg_type='info'):
        """广播消息"""
        for name in self.agent_registry:
            if name != sender:
                msg = Message(sender, name, content, msg_type)
                self.message_queue.append(msg)
                self.message_log.append(msg)

    def deliver(self):
        """投递消息"""
        delivered = 0
        while self.message_queue:
            msg = self.message_queue.pop(0)
            if msg.receiver in self.agent_registry:
                self.agent_registry[msg.receiver].receive(msg)
                delivered += 1
        return delivered


print("\n" + "=" * 60)
print("3. 通信协议测试")
print("=" * 60)

bus = CommunicationBus()
for role in roles:
    bus.register(role)

# 模拟通信
bus.send(Message("Manager", "Researcher", "请搜索AI发展趋势", "task"))
bus.send(Message("Manager", "Writer", "请准备AI报告大纲", "task"))
bus.broadcast("Manager", "项目启动: AI行业分析报告", "info")

delivered = bus.deliver()
print(f"  已投递 {delivered} 条消息")
print(f"  消息日志:")
for msg in bus.message_log:
    print(f"    {msg.sender} → {msg.receiver} [{msg.msg_type}]: {msg.content[:30]}")


# ============================================================
# 4. Multi-Agent vs Single-Agent对比
# ============================================================
print("\n" + "=" * 60)
print("4. Multi-Agent vs Single-Agent对比")
print("=" * 60)

# 模拟任务完成质量
np.random.seed(42)
task_types = ['代码开发', '文案创作', '数据分析', '研究调研', '综合项目']
single_agent_quality = []
multi_agent_quality = []
single_agent_time = []
multi_agent_time = []

for task in task_types:
    # 单Agent: 质量波动大, 时间短
    s_quality = np.random.uniform(0.5, 0.85)
    s_time = np.random.uniform(2, 8)
    # Multi-Agent: 质量稳定, 时间长
    m_quality = np.random.uniform(0.7, 0.95)
    m_time = np.random.uniform(5, 15)

    single_agent_quality.append(s_quality)
    multi_agent_quality.append(m_quality)
    single_agent_time.append(s_time)
    multi_agent_time.append(m_time)

comparison_data = {
    '维度': ['质量', '效率', '鲁棒性', '成本', '可扩展性', '复杂任务处理'],
    '单Agent': [3, 5, 2, 5, 3, 2],
    'Multi-Agent': [5, 2, 5, 2, 5, 5],
}

print(f"\n  {'维度':<15} {'单Agent':>10} {'Multi-Agent':>12}")
print(f"  {'-' * 40}")
for i, dim in enumerate(comparison_data['维度']):
    print(f"  {dim:<15} {comparison_data['单Agent'][i]:>10} {comparison_data['Multi-Agent'][i]:>12}")


# ============================================================
# 5. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 5.1 架构模式对比
ax1 = axes[0, 0]
arch_names = ['层级式', '协作式', '辩论式', '流水线']
complexity = [2, 4, 3, 5]
flexibility = [3, 5, 4, 2]
scalability = [4, 3, 2, 4]
x = np.arange(len(arch_names))
width = 0.25
ax1.bar(x - width, complexity, width, label='实现复杂度', color='#F44336', edgecolor='black')
ax1.bar(x, flexibility, width, label='灵活性', color='#4CAF50', edgecolor='black')
ax1.bar(x + width, scalability, width, label='可扩展性', color='#2196F3', edgecolor='black')
ax1.set_xticks(x)
ax1.set_xticklabels(arch_names)
ax1.set_ylabel('分数 (1-5)')
ax1.set_title('Multi-Agent架构模式对比')
ax1.legend()

# 5.2 通信模式图
ax2 = axes[0, 1]
agents_pos = {
    'Manager': (0.5, 0.85),
    'Researcher': (0.15, 0.5),
    'Writer': (0.5, 0.5),
    'Reviewer': (0.85, 0.5),
    'Coder': (0.5, 0.15),
}
colors_agents = ['#E91E63', '#4CAF50', '#2196F3', '#FF9800', '#9C27B0']

for i, (name, pos) in enumerate(agents_pos.items()):
    ax2.add_patch(plt.Circle(pos, 0.08, color=colors_agents[i], alpha=0.7))
    ax2.text(pos[0], pos[1], name[:4], ha='center', va='center', fontsize=8, fontweight='bold')

# 通信线
connections = [
    ('Manager', 'Researcher'), ('Manager', 'Writer'),
    ('Manager', 'Reviewer'), ('Manager', 'Coder'),
    ('Researcher', 'Writer'), ('Writer', 'Reviewer'),
]
for sender, receiver in connections:
    s = agents_pos[sender]
    r = agents_pos[receiver]
    ax2.annotate('', xy=r, xytext=s,
                 arrowprops=dict(arrowstyle='->', color='gray', lw=1, alpha=0.5))

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_title('Agent通信拓扑')
ax2.set_aspect('equal')
ax2.axis('off')

# 5.3 单Agent vs Multi-Agent质量对比
ax3 = axes[1, 0]
x = np.arange(len(task_types))
width = 0.35
ax3.bar(x - width / 2, single_agent_quality, width, label='单Agent', color='#FF9800', edgecolor='black')
ax3.bar(x + width / 2, multi_agent_quality, width, label='Multi-Agent', color='#2196F3', edgecolor='black')
ax3.set_xticks(x)
ax3.set_xticklabels(task_types, fontsize=9)
ax3.set_ylabel('完成质量')
ax3.set_title('任务完成质量对比')
ax3.legend()
ax3.set_ylim(0, 1.1)

# 5.4 综合能力雷达图
ax4 = axes[1, 1]
dims = comparison_data['维度']
num_dims = len(dims)
angles = np.linspace(0, 2 * np.pi, num_dims, endpoint=False).tolist()
angles += angles[:1]

single_vals = comparison_data['单Agent'] + comparison_data['单Agent'][:1]
multi_vals = comparison_data['Multi-Agent'] + comparison_data['Multi-Agent'][:1]

ax4.plot(angles, single_vals, 'o-', linewidth=2, label='单Agent', color='#FF9800')
ax4.fill(angles, single_vals, alpha=0.15, color='#FF9800')
ax4.plot(angles, multi_vals, 'o-', linewidth=2, label='Multi-Agent', color='#2196F3')
ax4.fill(angles, multi_vals, alpha=0.15, color='#2196F3')
ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(dims)
ax4.set_ylim(0, 6)
ax4.set_title('综合能力对比')
ax4.legend()

plt.suptitle('W23-D1: Multi-Agent系统概念', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W23/d1_multi_agent_concept.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存!")
