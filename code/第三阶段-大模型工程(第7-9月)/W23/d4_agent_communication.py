### Day 4（周四）：Agent间通信机制
# 消息格式, 共享状态, 黑板模式, 发布-订阅模式

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 消息格式设计
# ============================================================
print("=" * 60)
print("1. Agent间消息格式")
print("=" * 60)

class AgentMessage:
    """标准化的Agent消息格式"""

    def __init__(self, sender, receiver, msg_type, content, metadata=None, priority=0):
        self.sender = sender
        self.receiver = receiver
        self.msg_type = msg_type  # request, response, broadcast, notification, error
        self.content = content
        self.metadata = metadata or {}
        self.priority = priority
        self.timestamp = datetime.now().isoformat()
        self.msg_id = id(self)

    def to_dict(self):
        return {
            'msg_id': self.msg_id,
            'sender': self.sender,
            'receiver': self.receiver,
            'type': self.msg_type,
            'content': self.content,
            'metadata': self.metadata,
            'priority': self.priority,
            'timestamp': self.timestamp,
        }

    def __repr__(self):
        return (f"Msg({self.sender}→{self.receiver} "
                f"[{self.msg_type}]: {str(self.content)[:40]})")


# 消息类型说明
msg_types = {
    'request':     '请求其他Agent执行任务',
    'response':    '回复请求的结果',
    'broadcast':   '广播消息给所有Agent',
    'notification': '状态通知',
    'error':       '错误报告',
    'sync':        '同步状态',
}

print("\n  消息类型:")
for mtype, desc in msg_types.items():
    print(f"    {mtype:<15} - {desc}")

# 示例消息
print("\n  消息示例:")
example_msgs = [
    AgentMessage("Agent-A", "Agent-B", "request", {"task": "搜索", "query": "AI"}),
    AgentMessage("Agent-B", "Agent-A", "response", {"result": "AI是...", "status": "success"}),
    AgentMessage("Manager", "all", "broadcast", {"event": "任务开始", "project": "AI报告"}),
]
for msg in example_msgs:
    print(f"    {msg}")


# ============================================================
# 2. 共享状态机制
# ============================================================
class SharedState:
    """Agent间共享状态"""

    def __init__(self):
        self.state = {}
        self.version = 0
        self.history = []
        self.locks = {}

    def set(self, key, value, agent_name):
        """设置状态"""
        old_value = self.state.get(key)
        self.state[key] = value
        self.version += 1
        self.history.append({
            'action': 'set',
            'key': key,
            'old': old_value,
            'new': value,
            'agent': agent_name,
            'version': self.version,
        })

    def get(self, key, default=None):
        """获取状态"""
        return self.state.get(key, default)

    def update(self, updates, agent_name):
        """批量更新"""
        for key, value in updates.items():
            self.set(key, value, agent_name)

    def subscribe(self, key):
        """订阅状态变化 (模拟)"""
        if key not in self.locks:
            self.locks[key] = []
        return len(self.locks[key])

    def get_history(self, key=None):
        """获取变更历史"""
        if key:
            return [h for h in self.history if h['key'] == key]
        return self.history


print("\n" + "=" * 60)
print("2. 共享状态机制")
print("=" * 60)

shared = SharedState()
shared.set('project_status', 'started', 'Manager')
shared.set('current_task', 'research', 'Manager')
shared.update({'progress': 0.3, 'active_agents': 3}, 'System')

print(f"  当前状态: {shared.state}")
print(f"  变更历史:")
for h in shared.history:
    print(f"    v{h['version']}: {h['agent']} set {h['key']} = {h['new']}")


# ============================================================
# 3. 黑板模式 (Blackboard Pattern)
# ============================================================
class Blackboard:
    """黑板模式: 多Agent共享信息空间"""

    def __init__(self):
        self.sections = defaultdict(dict)
        self.contributors = defaultdict(set)
        self.read_log = []

    def write(self, section, key, value, agent_name):
        """写入黑板"""
        self.sections[section][key] = {
            'value': value,
            'author': agent_name,
            'timestamp': datetime.now().isoformat(),
        }
        self.contributors[section].add(agent_name)

    def read(self, section, key=None):
        """读取黑板"""
        if key:
            entry = self.sections[section].get(key)
            if entry:
                self.read_log.append({'section': section, 'key': key})
                return entry['value']
            return None
        return {k: v['value'] for k, v in self.sections[section].items()}

    def get_sections(self):
        """获取所有板块"""
        return dict(self.sections)

    def get_contributors(self, section):
        """获取板块贡献者"""
        return self.contributors.get(section, set())


print("\n" + "=" * 60)
print("3. 黑板模式 (Blackboard)")
print("=" * 60)

blackboard = Blackboard()

# 多个Agent向黑板写入信息
blackboard.write('requirements', '功能需求', ['登录', '搜索', '导出'], 'PM-Agent')
blackboard.write('requirements', '非功能需求', ['性能<1s', '可用性>99%'], 'PM-Agent')
blackboard.write('design', '架构方案', '微服务架构', 'Architect-Agent')
blackboard.write('design', '技术栈', 'Python + FastAPI + PostgreSQL', 'Architect-Agent')
blackboard.write('progress', '进度', '30%', 'Dev-Agent')
blackboard.write('issues', '阻塞项', '需要数据库权限', 'Dev-Agent')

print(f"  黑板内容:")
for section, entries in blackboard.get_sections().items():
    print(f"\n  [{section}] (贡献者: {blackboard.get_contributors(section)}):")
    for key, entry in entries.items():
        print(f"    {key}: {entry['value']} (by {entry['author']})")

# 读取测试
print(f"\n  读取需求: {blackboard.read('requirements')}")
print(f"  读取架构: {blackboard.read('design', '架构方案')}")


# ============================================================
# 4. 发布-订阅模式 (Pub/Sub)
# ============================================================
class PubSubBus:
    """发布-订阅消息总线"""

    def __init__(self):
        self.subscribers = defaultdict(list)  # topic → [callbacks]
        self.message_count = defaultdict(int)
        self.message_log = []

    def subscribe(self, topic, agent_name, callback=None):
        """订阅主题"""
        self.subscribers[topic].append({
            'agent': agent_name,
            'callback': callback,
        })
        print(f"  [{agent_name}] 订阅了主题: {topic}")

    def unsubscribe(self, topic, agent_name):
        """取消订阅"""
        self.subscribers[topic] = [
            s for s in self.subscribers[topic] if s['agent'] != agent_name
        ]

    def publish(self, topic, message, publisher):
        """发布消息"""
        self.message_count[topic] += 1
        log_entry = {
            'topic': topic,
            'publisher': publisher,
            'message': message,
            'subscribers': [s['agent'] for s in self.subscribers[topic]],
        }
        self.message_log.append(log_entry)

        print(f"  [{publisher}] 发布到 '{topic}': {str(message)[:40]}")

        # 通知所有订阅者
        for sub in self.subscribers[topic]:
            if sub['callback']:
                sub['callback'](message)
            print(f"    → [{sub['agent']}] 收到消息")

    def get_stats(self):
        return {
            'topics': list(self.subscribers.keys()),
            'message_counts': dict(self.message_count),
            'total_messages': sum(self.message_count.values()),
        }


print("\n" + "=" * 60)
print("4. 发布-订阅模式 (Pub/Sub)")
print("=" * 60)

bus = PubSubBus()

# 订阅
bus.subscribe('task_update', 'Manager-Agent')
bus.subscribe('task_update', 'Monitor-Agent')
bus.subscribe('error', 'Error-Handler-Agent')
bus.subscribe('error', 'Manager-Agent')
bus.subscribe('result', 'Aggregator-Agent')

# 发布消息
print()
bus.publish('task_update', {'task': 'research', 'status': 'completed'}, 'Researcher-Agent')
print()
bus.publish('error', {'task': 'coding', 'error': '编译失败'}, 'Coder-Agent')
print()
bus.publish('result', {'agent': 'Researcher', 'output': 'AI趋势报告'}, 'Researcher-Agent')

print(f"\n  统计: {bus.get_stats()}")


# ============================================================
# 5. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 5.1 通信模式对比
ax1 = axes[0, 0]
patterns = ['直接通信', '共享状态', '黑板模式', '发布-订阅']
latency = [1, 3, 5, 2]       # 延迟(ms)
scalability = [2, 4, 5, 5]   # 可扩展性
decoupling = [1, 3, 4, 5]    # 解耦程度

x = np.arange(len(patterns))
width = 0.25
ax1.bar(x - width, latency, width, label='延迟(低优)', color='#F44336', edgecolor='black')
ax1.bar(x, scalability, width, label='可扩展性', color='#4CAF50', edgecolor='black')
ax1.bar(x + width, decoupling, width, label='解耦程度', color='#2196F3', edgecolor='black')
ax1.set_xticks(x)
ax1.set_xticklabels(patterns)
ax1.set_ylabel('分数')
ax1.set_title('通信模式对比')
ax1.legend()

# 5.2 黑板模式可视化
ax2 = axes[0, 1]
sections = list(blackboard.sections.keys())
entries_count = [len(blackboard.sections[s]) for s in sections]
contributors_count = [len(blackboard.contributors[s]) for s in sections]
colors_bb = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']

ax2_twin = ax2.twinx()
l1 = ax2.bar(sections, entries_count, color=colors_bb, alpha=0.7, edgecolor='black', label='条目数')
l2 = ax2_twin.plot(sections, contributors_count, 'ro-', linewidth=2, markersize=10, label='贡献者数')
ax2.set_ylabel('条目数', color='blue')
ax2_twin.set_ylabel('贡献者数', color='red')
ax2.set_title('黑板模式: 各板块信息量')
lines = [l1] + l2
ax2.legend(lines, [l.get_label() for l in [l1] + l2])

# 5.3 Pub/Sub消息流量
ax3 = axes[1, 0]
if bus.message_log:
    topics = list(bus.message_count.keys())
    counts = list(bus.message_count.values())
    colors_ps = ['#4CAF50', '#F44336', '#2196F3'][:len(topics)]
    ax3.bar(topics, counts, color=colors_ps, edgecolor='black')
    ax3.set_ylabel('消息数')
    ax3.set_title('Pub/Sub各主题消息量')

# 5.4 通信效率随Agent数量变化
ax4 = axes[1, 1]
agent_counts = [2, 5, 10, 15, 20, 30]
direct_messages = [n * (n - 1) / 2 for n in agent_counts]
broadcast_messages = [n for n in agent_counts]
blackboard_operations = [np.log(n) * 2 for n in agent_counts]

ax4.plot(agent_counts, direct_messages, 'r-o', linewidth=2, label='直接通信 (P2P)')
ax4.plot(agent_counts, broadcast_messages, 'g-s', linewidth=2, label='广播')
ax4.plot(agent_counts, blackboard_operations, 'b-^', linewidth=2, label='黑板 (对数)')
ax4.set_xlabel('Agent数量')
ax4.set_ylabel('通信次数')
ax4.set_title('通信复杂度 vs Agent数量')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.suptitle('W23-D4: Agent间通信机制', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W23/d4_agent_communication.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存!")
