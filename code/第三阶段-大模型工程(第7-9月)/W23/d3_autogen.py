### Day 3（周三）：AutoGen概念与简化实现
# 对话模式(两Agent/群聊), 人类代理介入, 代码执行Agent

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. AutoGen核心概念
# ============================================================
print("=" * 60)
print("1. AutoGen核心概念")
print("=" * 60)

print("""
  AutoGen (微软) 核心组件:
    - AssistantAgent: AI助手, 可以写代码、回答问题
    - UserProxyAgent: 人类代理, 可以执行代码、提供反馈
    - GroupChat: 群聊模式, 多Agent讨论
    - GroupChatManager: 群聊管理器

  对话模式:
    1. 两Agent对话: Assistant ↔ UserProxy
    2. 群聊: 多Agent在一个聊天室中讨论
    3. 嵌套聊天: Agent内部再启动子对话

  特点:
    - 支持代码自动执行
    - 人类可以随时介入
    - 可自定义Agent行为
""")


# ============================================================
# 2. 简化版AutoGen实现
# ============================================================
class AutoGenAgent:
    """简化版AutoGen Agent"""

    def __init__(self, name, role, system_message="", max_consecutive_auto_reply=3):
        self.name = name
        self.role = role
        self.system_message = system_message
        self.max_consecutive_auto_reply = max_consecutive_auto_reply
        self.chat_history = []
        self.auto_reply_count = 0

    def send(self, message, recipient):
        """发送消息"""
        msg = {
            'sender': self.name,
            'recipient': recipient.name,
            'content': message,
        }
        self.chat_history.append(msg)
        recipient.chat_history.append(msg)
        print(f"  [{self.name} → {recipient.name}]: {message[:60]}...")

        # 触发接收者的自动回复
        response = recipient.auto_reply(msg)
        return response

    def auto_reply(self, received_msg):
        """自动回复"""
        if self.auto_reply_count >= self.max_consecutive_auto_reply:
            return None

        self.auto_reply_count += 1
        sender = received_msg['sender']
        content = received_msg['content']

        # 根据角色生成回复
        reply = self._generate_reply(sender, content)

        if reply:
            msg = {
                'sender': self.name,
                'recipient': sender,
                'content': reply,
            }
            self.chat_history.append(msg)
            print(f"  [{self.name} → {sender}]: {reply[:60]}...")

        return reply

    def _generate_reply(self, sender, content):
        """根据角色生成回复"""
        content_lower = content.lower()

        if self.role == 'assistant':
            if '代码' in content or '编程' in content:
                return "我来帮你写代码。这是一个Python实现示例: def solve(): ..."
            elif '分析' in content:
                return "我来分析这个问题。首先需要收集数据, 然后进行统计分析..."
            elif '完成' in content or '谢谢' in content:
                return "很高兴能帮到你! 如果还有其他问题, 随时告诉我。"
            else:
                return "我来帮你处理这个问题。让我先分析一下需求..."

        elif self.role == 'user_proxy':
            if '```python' in content or 'def ' in content:
                return "[代码执行成功] 输出: 结果正确"
            elif '?' in content or '？' in content:
                return "请继续。"
            else:
                return "收到, 请继续。"

        elif self.role == 'critic':
            return "让我审查一下。整体不错, 但有几个改进点: 1. 可以添加错误处理 2. 建议增加文档注释"

        return None

    def initiate_chat(self, recipient, message, max_rounds=5):
        """发起对话"""
        print(f"\n  === {self.name} 与 {recipient.name} 的对话 ===")

        # 重置计数
        self.auto_reply_count = 0
        recipient.auto_reply_count = 0

        current_msg = message
        for round_num in range(max_rounds):
            # 发送
            msg = {
                'sender': self.name,
                'recipient': recipient.name,
                'content': current_msg,
            }
            self.chat_history.append(msg)
            recipient.chat_history.append(msg)
            print(f"\n  Round {round_num + 1}:")
            print(f"  [{self.name} → {recipient.name}]: {current_msg[:80]}")

            # 接收方回复
            reply = recipient.auto_reply(msg)
            if reply is None:
                break
            current_msg = reply

            # 自己回复
            my_reply = self.auto_reply({'sender': recipient.name, 'content': reply})
            if my_reply is None:
                break
            current_msg = my_reply

        print(f"\n  === 对话结束 (共{round_num + 1}轮) ===")


class GroupChat:
    """简化版群聊"""

    def __init__(self, agents, messages=None, max_rounds=10):
        self.agents = agents
        self.messages = messages or []
        self.max_rounds = max_rounds
        self.chat_log = []

    def run(self, initial_message):
        """运行群聊"""
        print(f"\n  === 群聊开始 ===")
        print(f"  参与者: {[a.name for a in self.agents]}")

        current_msg = initial_message
        speaker_idx = 0

        for round_num in range(self.max_rounds):
            speaker = self.agents[speaker_idx]
            print(f"\n  Round {round_num + 1} - {speaker.name}:")
            print(f"    {current_msg[:80]}")

            self.chat_log.append({
                'round': round_num + 1,
                'speaker': speaker.name,
                'content': current_msg,
            })

            # 下一个发言者
            next_idx = (speaker_idx + 1) % len(self.agents)
            next_speaker = self.agents[next_idx]

            reply = next_speaker.auto_reply({
                'sender': speaker.name,
                'content': current_msg,
            })

            if reply is None or 'TERMINATE' in str(reply):
                break

            current_msg = reply
            speaker_idx = next_idx

        print(f"\n  === 群聊结束 (共{round_num + 1}轮) ===")
        return self.chat_log


# ============================================================
# 3. 两Agent对话演示
# ============================================================
print("=" * 60)
print("2. 两Agent对话: 编程助手")
print("=" * 60)

assistant = AutoGenAgent("AI助手", "assistant",
                         "你是一个编程助手", max_consecutive_auto_reply=5)
user_proxy = AutoGenAgent("用户代理", "user_proxy",
                          "你代表用户, 可以执行代码", max_consecutive_auto_reply=5)

assistant.initiate_chat(user_proxy, "我需要一段代码来计算斐波那契数列", max_rounds=4)


# ============================================================
# 4. 群聊演示
# ============================================================
print("\n" + "=" * 60)
print("3. 群聊模式: 代码审查")
print("=" * 60)

writer_agent = AutoGenAgent("Writer", "assistant", "撰写代码", 3)
critic_agent = AutoGenAgent("Critic", "critic", "审查代码", 3)
user_agent = AutoGenAgent("User", "user_proxy", "用户", 3)

group = GroupChat([writer_agent, critic_agent, user_agent], max_rounds=6)
group.run("请帮我写一个排序算法, 并进行代码审查")


# ============================================================
# 5. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 5.1 两Agent对话流程
ax1 = axes[0, 0]
two_agent_msgs = [
    ('User', 'AI助手', '需要计算斐波那契'),
    ('AI助手', 'User', '我来写代码: def fib(n)...'),
    ('User', 'AI助手', '[执行成功] 继续优化'),
    ('AI助手', 'User', '优化后的版本...'),
]
y_pos = list(range(len(two_agent_msgs)))
for i, (s, r, msg) in enumerate(two_agent_msgs):
    color = '#2196F3' if s == 'User' else '#4CAF50'
    ax1.barh(i, 1, color=color, alpha=0.6, edgecolor='black')
    ax1.text(0.5, i, f'{s}→{r}: {msg[:30]}', ha='center', va='center', fontsize=8)
ax1.set_yticks([])
ax1.set_title('两Agent对话流程')
ax1.set_xticks([])

# 5.2 群聊参与度
ax2 = axes[0, 1]
if group.chat_log:
    speakers = [log['speaker'] for log in group.chat_log]
    from collections import Counter
    speaker_counts = Counter(speakers)
    names = list(speaker_counts.keys())
    counts = list(speaker_counts.values())
    colors_gc = ['#4CAF50', '#FF9800', '#2196F3'][:len(names)]
    ax2.bar(names, counts, color=colors_gc, edgecolor='black')
    ax2.set_ylabel('发言次数')
    ax2.set_title('群聊各Agent发言次数')

# 5.3 AutoGen vs CrewAI对比
ax3 = axes[1, 0]
frameworks = ['AutoGen', 'CrewAI', 'LangGraph']
features = ['灵活性', '易用性', '代码执行', '人机协作', '可扩展']
scores = {
    'AutoGen': [5, 3, 5, 5, 4],
    'CrewAI':  [3, 5, 2, 3, 4],
    'LangGraph': [5, 3, 3, 4, 5],
}

x = np.arange(len(features))
width = 0.25
for i, (fw, sc) in enumerate(scores.items()):
    ax3.bar(x + i * width, sc, width, label=fw, edgecolor='black')
ax3.set_xticks(x + width)
ax3.set_xticklabels(features, fontsize=9)
ax3.set_ylabel('分数')
ax3.set_title('Multi-Agent框架对比')
ax3.legend()
ax3.set_ylim(0, 6)

# 5.4 对话轮次 vs 任务复杂度
ax4 = axes[1, 1]
complexities = ['简单问答', '代码编写', '代码审查', '项目规划', '系统设计']
rounds_needed = [2, 4, 6, 8, 10]
quality_improvement = [0.1, 0.25, 0.35, 0.45, 0.5]

ax4_twin = ax4.twinx()
l1 = ax4.plot(range(len(complexities)), rounds_needed, 'bo-', linewidth=2, label='所需轮次')
l2 = ax4_twin.plot(range(len(complexities)), quality_improvement, 'rs-', linewidth=2, label='质量提升')
ax4.set_xticks(range(len(complexities)))
ax4.set_xticklabels(complexities, fontsize=9)
ax4.set_ylabel('所需轮次', color='blue')
ax4_twin.set_ylabel('质量提升', color='red')
ax4.set_title('对话轮次 vs 任务复杂度')
lines = l1 + l2
ax4.legend(lines, [l.get_label() for l in lines])

plt.suptitle('W23-D3: AutoGen概念与实现', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W23/d3_autogen.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存!")
