### Day 4（周四）：Agent记忆系统
# 短期记忆, 长期记忆, 工作记忆, 记忆检索和更新

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 记忆系统架构
# ============================================================
print("=" * 60)
print("1. Agent记忆系统架构")
print("=" * 60)

print("""
  三层记忆架构:

  ┌─────────────────────────────────────┐
  │  工作记忆 (Working Memory)          │  ← 当前任务上下文
  │  容量小, 速度最快                    │
  ├─────────────────────────────────────┤
  │  短期记忆 (Short-term Memory)       │  ← 对话历史
  │  最近N轮对话, FIFO队列              │
  ├─────────────────────────────────────┤
  │  长期记忆 (Long-term Memory)        │  ← 向量数据库
  │  大容量, 需要检索                    │
  └─────────────────────────────────────┘
""")


# ============================================================
# 2. 短期记忆: 对话历史管理
# ============================================================
class ShortTermMemory:
    """短期记忆: 管理最近的对话历史"""

    def __init__(self, max_length=10):
        self.max_length = max_length
        self.history = deque(maxlen=max_length)
        self.token_count = 0

    def add(self, role, content):
        """添加一条消息"""
        tokens = len(content) // 2  # 简化token计数
        self.history.append({
            'role': role,
            'content': content,
            'tokens': tokens,
            'timestamp': datetime.now().isoformat(),
        })
        self.token_count = sum(m['tokens'] for m in self.history)

    def get_history(self, max_tokens=None):
        """获取历史消息"""
        if max_tokens is None:
            return list(self.history)

        # 从最新的消息开始, 不超过token限制
        result = []
        total = 0
        for msg in reversed(self.history):
            if total + msg['tokens'] > max_tokens:
                break
            result.append(msg)
            total += msg['tokens']
        return list(reversed(result))

    def summarize_old(self):
        """总结旧消息 (模拟)"""
        if len(self.history) > self.max_length * 0.8:
            old_count = len(self.history) // 3
            print(f"    [短期记忆] 总结了{old_count}条旧消息")
            for _ in range(old_count):
                if self.history:
                    self.history.popleft()
            self.token_count = sum(m['tokens'] for m in self.history)

    def get_stats(self):
        return {
            'length': len(self.history),
            'max_length': self.max_length,
            'token_count': self.token_count,
        }


print("=" * 60)
print("2. 短期记忆 (对话历史)")
print("=" * 60)

stm = ShortTermMemory(max_length=5)

# 模拟对话
conversations = [
    ('user', '你好, 我想了解Python'),
    ('assistant', 'Python是一种流行的编程语言, 您想了解哪个方面?'),
    ('user', 'Python的数据类型有哪些?'),
    ('assistant', 'Python主要有以下数据类型: int, float, str, list, dict, tuple, set, bool'),
    ('user', 'list和tuple有什么区别?'),
    ('assistant', 'list是可变的, tuple是不可变的。list用[], tuple用()'),
    ('user', '能举一个dict的例子吗?'),
    ('assistant', '当然: person = {"name": "Tom", "age": 25}'),
]

for role, content in conversations:
    stm.add(role, content)
    stats = stm.get_stats()
    print(f"  [{role:>9}] {content[:40]}... (记忆: {stats['length']}/{stats['max_length']}, "
          f"tokens: {stats['token_count']})")


# ============================================================
# 3. 长期记忆: 向量存储模拟
# ============================================================
class LongTermMemory:
    """长期记忆: 基于向量的知识存储与检索"""

    def __init__(self, embedding_dim=64):
        self.embedding_dim = embedding_dim
        self.memories = []  # 存储记忆
        self.embeddings = []  # 存储向量

    def _simple_embed(self, text):
        """简化版文本向量化 (用哈希模拟)"""
        np.random.seed(hash(text) % (2 ** 31))
        embedding = np.random.randn(self.embedding_dim)
        embedding = embedding / np.linalg.norm(embedding)  # 归一化
        return embedding

    def store(self, content, metadata=None):
        """存储一条记忆"""
        embedding = self._simple_embed(content)
        self.memories.append({
            'content': content,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat(),
            'access_count': 0,
        })
        self.embeddings.append(embedding)

    def retrieve(self, query, top_k=3):
        """检索相关记忆"""
        if not self.embeddings:
            return []

        query_embedding = self._simple_embed(query)

        # 计算余弦相似度
        embeddings_matrix = np.array(self.embeddings)
        similarities = np.dot(embeddings_matrix, query_embedding)

        # 获取top_k
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            self.memories[idx]['access_count'] += 1
            results.append({
                'content': self.memories[idx]['content'],
                'score': similarities[idx],
                'metadata': self.memories[idx]['metadata'],
            })
        return results

    def forget(self, threshold_days=30):
        """遗忘旧记忆 (模拟)"""
        # 简化: 随机遗忘访问次数少的记忆
        to_keep = []
        to_keep_emb = []
        forgotten = 0
        for i, mem in enumerate(self.memories):
            if mem['access_count'] > 0:
                to_keep.append(mem)
                to_keep_emb.append(self.embeddings[i])
            else:
                forgotten += 1
        self.memories = to_keep
        self.embeddings = to_keep_emb
        return forgotten


print("\n" + "=" * 60)
print("3. 长期记忆 (向量存储)")
print("=" * 60)

ltm = LongTermMemory(embedding_dim=32)

# 存储知识
knowledge = [
    ("Python由Guido van Rossum在1991年创建", {"category": "编程"}),
    ("机器学习是AI的子领域, 让计算机从数据中学习", {"category": "AI"}),
    ("Transformer架构在2017年的论文中提出", {"category": "深度学习"}),
    ("GPT是Generative Pre-trained Transformer的缩写", {"category": "LLM"}),
    ("向量数据库用于存储和检索高维向量", {"category": "数据库"}),
    ("RAG通过检索外部知识增强生成质量", {"category": "LLM"}),
    ("Python的list是可变序列, 支持增删改查", {"category": "编程"}),
    ("梯度下降是深度学习的核心优化算法", {"category": "深度学习"}),
]

for content, metadata in knowledge:
    ltm.store(content, metadata)

print(f"  已存储 {len(ltm.memories)} 条记忆")

# 检索测试
queries = ["什么是Transformer?", "Python的列表是什么?", "如何优化神经网络?"]
for query in queries:
    print(f"\n  查询: '{query}'")
    results = ltm.retrieve(query, top_k=2)
    for r in results:
        print(f"    [{r['score']:.3f}] {r['content']} ({r['metadata'].get('category', '')})")


# ============================================================
# 4. 工作记忆: 当前任务上下文
# ============================================================
class WorkingMemory:
    """工作记忆: 管理当前任务的关键信息"""

    def __init__(self, capacity=7):
        self.capacity = capacity  # Miller's Law: 7±2
        self.slots = {}
        self.importance = {}

    def set(self, key, value, importance=0.5):
        """设置工作记忆"""
        if len(self.slots) >= self.capacity and key not in self.slots:
            # 移除最不重要的
            least_important = min(self.importance, key=self.importance.get)
            del self.slots[least_important]
            del self.importance[least_important]

        self.slots[key] = value
        self.importance[key] = importance

    def get(self, key):
        """获取工作记忆"""
        return self.slots.get(key)

    def update(self, key, value):
        """更新工作记忆"""
        if key in self.slots:
            self.slots[key] = value

    def clear(self):
        """清空工作记忆"""
        self.slots.clear()
        self.importance.clear()

    def get_context_string(self):
        """生成上下文字符串"""
        items = sorted(self.slots.items(), key=lambda x: -self.importance.get(x[0], 0))
        return "\n".join(f"  - {k}: {v}" for k, v in items)


print("\n" + "=" * 60)
print("4. 工作记忆 (任务上下文)")
print("=" * 60)

wm = WorkingMemory(capacity=5)

# 模拟一个数据分析任务
wm.set('current_task', '分析销售数据', importance=0.9)
wm.set('data_source', 'sales_2024.csv', importance=0.8)
wm.set('user_goal', '找出销售趋势和异常', importance=0.9)
wm.set('tools_used', 'pandas, matplotlib', importance=0.5)
wm.set('intermediate_result', 'Q1销售增长15%', importance=0.7)

print(f"  当前工作记忆 (容量: {len(wm.slots)}/{wm.capacity}):")
print(wm.get_context_string())

# 添加新信息, 超过容量
print(f"\n  添加新信息...")
wm.set('new_finding', 'Q3出现异常下降', importance=0.85)
print(f"  工作记忆 (添加后):")
print(wm.get_context_string())


# ============================================================
# 5. 综合记忆管理器
# ============================================================
class MemoryManager:
    """综合记忆管理器"""

    def __init__(self):
        self.working = WorkingMemory(capacity=5)
        self.short_term = ShortTermMemory(max_length=10)
        self.long_term = LongTermMemory(embedding_dim=32)
        self.stats = {
            'stm_reads': 0, 'stm_writes': 0,
            'ltm_reads': 0, 'ltm_writes': 0,
            'wm_updates': 0,
        }

    def process_input(self, user_input):
        """处理用户输入"""
        # 1. 写入短期记忆
        self.short_term.add('user', user_input)
        self.stats['stm_writes'] += 1

        # 2. 从长期记忆检索相关信息
        relevant = self.long_term.retrieve(user_input, top_k=3)
        self.stats['ltm_reads'] += 1

        # 3. 更新工作记忆
        self.working.set('last_input', user_input, importance=0.8)
        if relevant:
            self.working.set('relevant_context', relevant[0]['content'], importance=0.7)
        self.stats['wm_updates'] += 1

        # 4. 检查是否需要总结短期记忆
        if self.short_term.get_stats()['length'] > 8:
            # 将重要信息存入长期记忆
            for msg in list(self.short_term.history)[:3]:
                self.long_term.store(msg['content'])
                self.stats['ltm_writes'] += 1

    def get_full_context(self, query):
        """获取完整的上下文"""
        context = {
            'working': dict(self.working.slots),
            'recent_history': [m['content'] for m in list(self.short_term.history)[-3:]],
            'relevant_memories': self.long_term.retrieve(query, top_k=2),
        }
        return context

    def get_stats(self):
        return self.stats


print("\n" + "=" * 60)
print("5. 综合记忆管理器测试")
print("=" * 60)

mm = MemoryManager()

# 预存一些知识
mm.long_term.store("用户偏好中文回答", {"type": "preference"})
mm.long_term.store("用户是Java开发者", {"type": "profile"})
mm.long_term.store("用户正在学习AI", {"type": "profile"})

# 模拟多轮对话
interactions = [
    "你好, 我想学习Python",
    "Python有什么优点?",
    "推荐一些学习资源",
    "如何在Python中实现单例模式?",
    "Python和Java哪个更适合AI开发?",
]

for user_input in interactions:
    mm.process_input(user_input)
    context = mm.get_full_context(user_input)
    print(f"\n  用户: {user_input}")
    print(f"    工作记忆: {list(context['working'].keys())}")
    print(f"    最近历史: {[h[:20] for h in context['recent_history']]}")
    if context['relevant_memories']:
        print(f"    相关记忆: {context['relevant_memories'][0]['content'][:30]}")

print(f"\n  记忆统计:")
for key, val in mm.get_stats().items():
    print(f"    {key}: {val}")


# ============================================================
# 6. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 6.1 记忆容量随时间变化
ax1 = axes[0, 0]
time_steps = range(20)
stm_usage = [min(t, stm.max_length) for t in range(1, 21)]
ltm_usage = [min(t * 2, 50) for t in range(1, 21)]
wm_usage = [min(t % 5 + 1, 5) for t in range(1, 21)]
ax1.plot(time_steps, stm_usage, 'b-', linewidth=2, label='短期记忆')
ax1.plot(time_steps, ltm_usage, 'g-', linewidth=2, label='长期记忆')
ax1.plot(time_steps, wm_usage, 'r-', linewidth=2, label='工作记忆')
ax1.set_xlabel('时间步')
ax1.set_ylabel('记忆条目数')
ax1.set_title('记忆使用量随时间变化')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 6.2 检索相似度分布
ax2 = axes[0, 1]
query_emb = ltm._simple_embed("什么是深度学习?")
all_sims = [np.dot(ltm._simple_embed(m['content']), query_emb)
            for m in ltm.memories]
ax2.hist(all_sims, bins=10, color='#9C27B0', edgecolor='black', alpha=0.7)
ax2.axvline(x=np.mean(all_sims), color='red', linestyle='--', label=f'均值={np.mean(all_sims):.2f}')
ax2.set_xlabel('余弦相似度')
ax2.set_ylabel('频次')
ax2.set_title('记忆检索相似度分布')
ax2.legend()

# 6.3 记忆访问频率
ax3 = axes[1, 0]
mem_labels = [m['content'][:15] + '...' for m in ltm.memories]
access_counts = [m['access_count'] for m in ltm.memories]
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(mem_labels)))
ax3.barh(range(len(mem_labels)), access_counts, color=colors, edgecolor='black')
ax3.set_yticks(range(len(mem_labels)))
ax3.set_yticklabels(mem_labels, fontsize=8)
ax3.set_xlabel('访问次数')
ax3.set_title('长期记忆访问频率')

# 6.4 记忆系统架构图
ax4 = axes[1, 1]
layers = ['工作记忆\n(当前上下文)', '短期记忆\n(对话历史)', '长期记忆\n(知识库)']
sizes = [5, 10, 50]
colors_arch = ['#FF9800', '#2196F3', '#4CAF50']
y_positions = [0.7, 0.4, 0.1]

for i, (layer, size, color, y) in enumerate(zip(layers, sizes, colors_arch, y_positions)):
    box_width = 0.3 + size * 0.015
    rect = plt.Rectangle((0.5 - box_width / 2, y - 0.08), box_width, 0.16,
                          facecolor=color, alpha=0.6, edgecolor='black', linewidth=2)
    ax4.add_patch(rect)
    ax4.text(0.5, y, f'{layer}\n容量: {size}', ha='center', va='center', fontsize=9)
    if i < 2:
        ax4.annotate('', xy=(0.5, y - 0.12), xytext=(0.5, y - 0.03),
                     arrowprops=dict(arrowstyle='->', color='black', lw=2))

ax4.set_xlim(0, 1)
ax4.set_ylim(-0.05, 0.9)
ax4.set_title('记忆系统三层架构')
ax4.axis('off')

plt.suptitle('W22-D4: Agent记忆系统', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W22/d4_memory.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存!")
