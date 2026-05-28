"""
W30-D4 多轮对话管理
====================
实现多轮对话功能, 包括:
- 对话历史管理
- 上下文压缩策略
- 对话摘要生成
- 会话存储与恢复

多轮对话是RAG系统的重要功能, 需要在有限的上下文窗口中
保持对话连贯性。
"""

import json
import hashlib
import time
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict
from collections import OrderedDict

try:
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    HAS_PLT = True
except ImportError:
    HAS_PLT = False


# ============================================================
# 1. 对话消息与历史管理
# ============================================================

@dataclass
class Message:
    """对话消息"""
    role: str          # 'system', 'user', 'assistant'
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict) -> 'Message':
        return Message(**data)

    def token_count(self) -> int:
        """估算token数"""
        chinese_chars = sum(1 for c in self.content if '一' <= c <= '鿿')
        other_chars = len(self.content) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4) + 10  # 加10为角色标签开销


class ConversationHistory:
    """对话历史管理器"""

    def __init__(self, max_messages: int = 50, max_tokens: int = 4096):
        self.messages: List[Message] = []
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.system_message: Optional[Message] = None

    def set_system_message(self, content: str):
        """设置系统消息"""
        self.system_message = Message(role='system', content=content)

    def add_user_message(self, content: str, metadata: Dict = None):
        """添加用户消息"""
        msg = Message(role='user', content=content, metadata=metadata or {})
        self.messages.append(msg)
        self._trim_history()
        return msg

    def add_assistant_message(self, content: str, metadata: Dict = None):
        """添加助手消息"""
        msg = Message(role='assistant', content=content, metadata=metadata or {})
        self.messages.append(msg)
        self._trim_history()
        return msg

    def _trim_history(self):
        """裁剪历史, 保持不超过限制"""
        # 先按消息数裁剪
        while len(self.messages) > self.max_messages:
            self.messages.pop(0)

        # 再按token数裁剪
        while self._total_tokens() > self.max_tokens and len(self.messages) > 2:
            self.messages.pop(0)

    def _total_tokens(self) -> int:
        total = sum(msg.token_count() for msg in self.messages)
        if self.system_message:
            total += self.system_message.token_count()
        return total

    def get_context(self, max_tokens: int = None) -> List[Dict]:
        """获取用于API调用的上下文消息列表"""
        max_t = max_tokens or self.max_tokens
        result = []

        if self.system_message:
            result.append({'role': 'system', 'content': self.system_message.content})
            max_t -= self.system_message.token_count()

        # 从最新消息开始, 逆序添加
        selected = []
        used = 0
        for msg in reversed(self.messages):
            if used + msg.token_count() > max_t:
                break
            selected.append(msg)
            used += msg.token_count()

        # 恢复正序
        selected.reverse()
        for msg in selected:
            result.append({'role': msg.role, 'content': msg.content})

        return result

    def get_turn_count(self) -> int:
        """获取对话轮数"""
        return sum(1 for m in self.messages if m.role == 'user')

    def get_last_n_turns(self, n: int) -> List[Message]:
        """获取最近n轮对话"""
        turns = [m for m in self.messages if m.role in ('user', 'assistant')]
        # 每轮包含user+assistant两条消息
        return turns[-(n*2):]

    def clear(self):
        """清空对话历史(保留系统消息)"""
        self.messages.clear()

    def to_dict(self) -> Dict:
        return {
            'system_message': self.system_message.to_dict() if self.system_message else None,
            'messages': [m.to_dict() for m in self.messages],
        }

    @staticmethod
    def from_dict(data: Dict) -> 'ConversationHistory':
        history = ConversationHistory()
        if data.get('system_message'):
            history.system_message = Message.from_dict(data['system_message'])
        history.messages = [Message.from_dict(m) for m in data.get('messages', [])]
        return history


# ============================================================
# 2. 上下文压缩策略
# ============================================================

class ContextCompressor:
    """上下文压缩器

    当对话历史过长时, 需要压缩以适应上下文窗口。
    策略:
    1. 滑动窗口: 只保留最近N轮
    2. 摘要压缩: 将早期对话压缩为摘要
    3. 关键信息提取: 提取实体和关键事实
    """

    def __init__(self, strategy: str = "summary"):
        self.strategy = strategy

    def compress(self, history: ConversationHistory,
                 keep_recent_turns: int = 3) -> ConversationHistory:
        """压缩对话历史"""
        if self.strategy == "sliding_window":
            return self._sliding_window(history, keep_recent_turns)
        elif self.strategy == "summary":
            return self._summary_compress(history, keep_recent_turns)
        elif self.strategy == "key_info":
            return self._key_info_extract(history, keep_recent_turns)
        else:
            return history

    def _sliding_window(self, history: ConversationHistory,
                         keep_turns: int) -> ConversationHistory:
        """滑动窗口: 只保留最近N轮"""
        recent = history.get_last_n_turns(keep_turns)
        new_history = ConversationHistory(
            max_messages=history.max_messages,
            max_tokens=history.max_tokens
        )
        new_history.system_message = history.system_message
        new_history.messages = recent
        return new_history

    def _summary_compress(self, history: ConversationHistory,
                           keep_turns: int) -> ConversationHistory:
        """摘要压缩: 将早期对话压缩为摘要"""
        all_turns = [m for m in history.messages if m.role in ('user', 'assistant')]

        if len(all_turns) <= keep_turns * 2:
            return history

        # 分离早期和近期
        split_idx = len(all_turns) - keep_turns * 2
        early_turns = all_turns[:split_idx]
        recent_turns = all_turns[split_idx:]

        # 生成早期对话的摘要
        summary = self._generate_summary(early_turns)

        # 构建新的历史
        new_history = ConversationHistory(
            max_messages=history.max_messages,
            max_tokens=history.max_tokens
        )
        new_history.system_message = history.system_message

        # 添加摘要作为系统上下文
        summary_msg = Message(
            role='system',
            content=f"[对话摘要] {summary}",
            metadata={'type': 'summary'}
        )
        new_history.messages = [summary_msg] + recent_turns

        return new_history

    def _generate_summary(self, turns: List[Message]) -> str:
        """生成对话摘要(简化版)"""
        user_queries = [m.content for m in turns if m.role == 'user']
        assistant_topics = []

        for m in turns:
            if m.role == 'assistant' and len(m.content) > 20:
                # 提取每条回答的第一句话作为主题
                first_sentence = m.content.split('。')[0]
                if len(first_sentence) > 5:
                    assistant_topics.append(first_sentence)

        summary_parts = []
        if user_queries:
            summary_parts.append(f"用户询问了{len(user_queries)}个问题")
        if assistant_topics:
            topics_str = "; ".join(assistant_topics[:5])
            summary_parts.append(f"讨论的主题包括: {topics_str}")

        return "。".join(summary_parts) if summary_parts else "之前有过简短的对话"

    def _key_info_extract(self, history: ConversationHistory,
                           keep_turns: int) -> ConversationHistory:
        """关键信息提取"""
        all_turns = [m for m in history.messages if m.role in ('user', 'assistant')]

        if len(all_turns) <= keep_turns * 2:
            return history

        split_idx = len(all_turns) - keep_turns * 2
        early_turns = all_turns[:split_idx]
        recent_turns = all_turns[split_idx:]

        # 提取关键实体和信息
        key_info = self._extract_key_information(early_turns)

        new_history = ConversationHistory(
            max_messages=history.max_messages,
            max_tokens=history.max_tokens
        )
        new_history.system_message = history.system_message

        key_info_msg = Message(
            role='system',
            content=f"[关键信息] {key_info}",
            metadata={'type': 'key_info'}
        )
        new_history.messages = [key_info_msg] + recent_turns

        return new_history

    def _extract_key_information(self, turns: List[Message]) -> str:
        """提取关键信息(简化版)"""
        # 在实际应用中应使用NER模型
        key_points = []
        for m in turns:
            if m.role == 'user' and '?' in m.content or '？' in m.content:
                key_points.append(f"用户关注: {m.content[:50]}")

        return "; ".join(key_points[:5]) if key_points else "无关键信息提取"


# ============================================================
# 3. 会话存储
# ============================================================

class SessionStore:
    """会话存储管理器

    支持将对话会话持久化存储, 以便用户可以恢复之前的对话。
    """

    def __init__(self, storage_backend: str = "memory"):
        self.backend = storage_backend
        self.sessions: OrderedDict[str, Dict] = OrderedDict()
        self.max_sessions = 100

    def create_session(self, user_id: str = "default") -> str:
        """创建新会话"""
        session_id = hashlib.md5(
            f"{user_id}_{time.time()}".encode()
        ).hexdigest()[:12]

        self.sessions[session_id] = {
            'session_id': session_id,
            'user_id': user_id,
            'created_at': time.time(),
            'updated_at': time.time(),
            'history': ConversationHistory(),
        }

        # 清理过多的会话
        while len(self.sessions) > self.max_sessions:
            self.sessions.popitem(last=False)

        print(f"[会话] 创建新会话: {session_id}")
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict]:
        """获取会话"""
        return self.sessions.get(session_id)

    def get_history(self, session_id: str) -> Optional[ConversationHistory]:
        """获取会话的对话历史"""
        session = self.sessions.get(session_id)
        if session:
            session['updated_at'] = time.time()
            return session['history']
        return None

    def save_message(self, session_id: str, role: str, content: str):
        """保存消息到会话"""
        session = self.sessions.get(session_id)
        if not session:
            return

        history = session['history']
        if role == 'user':
            history.add_user_message(content)
        elif role == 'assistant':
            history.add_assistant_message(content)

        session['updated_at'] = time.time()

    def list_sessions(self, user_id: str = None) -> List[Dict]:
        """列出所有会话"""
        sessions = []
        for sid, session in self.sessions.items():
            if user_id and session['user_id'] != user_id:
                continue
            sessions.append({
                'session_id': sid,
                'user_id': session['user_id'],
                'created_at': session['created_at'],
                'turns': session['history'].get_turn_count(),
                'messages': len(session['history'].messages),
            })
        return sessions

    def export_session(self, session_id: str) -> Optional[str]:
        """导出会话为JSON"""
        session = self.sessions.get(session_id)
        if not session:
            return None
        return json.dumps({
            'session_id': session_id,
            'user_id': session['user_id'],
            'created_at': session['created_at'],
            'history': session['history'].to_dict(),
        }, ensure_ascii=False, indent=2)

    def import_session(self, json_data: str) -> str:
        """从JSON导入会话"""
        data = json.loads(json_data)
        session_id = data['session_id']
        history = ConversationHistory.from_dict(data['history'])

        self.sessions[session_id] = {
            'session_id': session_id,
            'user_id': data['user_id'],
            'created_at': data['created_at'],
            'updated_at': time.time(),
            'history': history,
        }
        return session_id


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("W30-D4 多轮对话管理")
    print("=" * 60)

    # --- 1. 对话历史管理 ---
    print("\n--- 1. 对话历史管理 ---")
    history = ConversationHistory(max_tokens=2000)
    history.set_system_message("你是一个AI助手, 请用中文回答问题。")

    # 模拟多轮对话
    conversations = [
        ("什么是机器学习?", "机器学习是AI的分支, 让计算机从数据中学习模式。"),
        ("它和深度学习有什么关系?", "深度学习是机器学习的子集, 使用多层神经网络。"),
        ("能举个例子吗?", "比如图像识别、语音识别和自然语言处理都是深度学习的应用。"),
        ("Python适合做机器学习吗?", "Python是最流行的机器学习语言, 有丰富的库如scikit-learn。"),
        ("推荐几个学习资源?", "推荐Andrew Ng的Coursera课程和《动手学深度学习》。"),
    ]

    for user_msg, assistant_msg in conversations:
        history.add_user_message(user_msg)
        history.add_assistant_message(assistant_msg)

    print(f"对话轮数: {history.get_turn_count()}")
    print(f"总消息数: {len(history.messages)}")
    print(f"总token数: {history._total_tokens()}")

    # 获取上下文
    context = history.get_context()
    print(f"\nAPI上下文消息数: {len(context)}")
    for msg in context[:3]:
        print(f"  [{msg['role']}] {msg['content'][:40]}...")

    # --- 2. 上下文压缩 ---
    print(f"\n{'='*60}")
    print("--- 2. 上下文压缩 ---")
    print(f"{'='*60}")

    for strategy in ["sliding_window", "summary", "key_info"]:
        compressor = ContextCompressor(strategy=strategy)
        compressed = compressor.compress(history, keep_recent_turns=2)
        original_tokens = history._total_tokens()
        compressed_tokens = compressed._total_tokens()
        ratio = compressed_tokens / original_tokens if original_tokens > 0 else 0

        print(f"\n策略: {strategy}")
        print(f"  压缩前: {original_tokens} tokens, {len(history.messages)} 条消息")
        print(f"  压缩后: {compressed_tokens} tokens, {len(compressed.messages)} 条消息")
        print(f"  压缩率: {ratio:.1%}")

    # --- 3. 会话存储 ---
    print(f"\n{'='*60}")
    print("--- 3. 会话存储 ---")
    print(f"{'='*60}")

    store = SessionStore()

    # 创建会话并添加消息
    session_id = store.create_session("user_001")
    for user_msg, assistant_msg in conversations[:3]:
        store.save_message(session_id, 'user', user_msg)
        store.save_message(session_id, 'assistant', assistant_msg)

    # 列出会话
    sessions = store.list_sessions()
    print(f"活跃会话数: {len(sessions)}")
    for s in sessions:
        print(f"  会话{s['session_id']}: {s['turns']}轮, {s['messages']}条消息")

    # 导出和导入
    exported = store.export_session(session_id)
    print(f"\n导出会话JSON (前200字符):\n{exported[:200]}...")

    new_session_id = store.import_session(exported)
    print(f"导入成功, 新会话ID: {new_session_id}")

    # --- 可视化 ---
    if HAS_PLT:
        fig, ax = plt.subplots(figsize=(10, 5))

        turns = list(range(1, len(conversations) + 1))
        token_counts = []
        temp_history = ConversationHistory(max_tokens=10000)
        temp_history.set_system_message("你是一个AI助手")

        for user_msg, assistant_msg in conversations:
            temp_history.add_user_message(user_msg)
            temp_history.add_assistant_message(assistant_msg)
            token_counts.append(temp_history._total_tokens())

        ax.plot(turns, token_counts, 'o-', color='#3498db', linewidth=2, markersize=8)
        ax.axhline(y=2000, color='#e74c3c', linestyle='--', label='Token上限(2000)')
        ax.fill_between(turns, 2000, max(token_counts) * 1.1, alpha=0.1, color='red')
        ax.set_xlabel('对话轮数', fontsize=12)
        ax.set_ylabel('Token数', fontsize=12)
        ax.set_title('对话历史Token增长趋势', fontsize=14, fontweight='bold')
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig('D:/code/big-model-learn/code/q_01/W30/d4_conversation.png', dpi=150)
        print("\n图表已保存为 d4_conversation.png")
        plt.close()

    print("\n完成!")
