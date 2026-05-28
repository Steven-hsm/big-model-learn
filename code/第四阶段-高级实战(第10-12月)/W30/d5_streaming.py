"""
W30-D5 流式输出
================
实现流式输出功能, 包括:
- SSE(Server-Sent Events)协议
- 流式生成器实现
- FastAPI StreamingResponse集成示例

流式输出让用户在LLM生成回答时就能逐步看到内容,
大幅改善用户体验。
"""

import time
import json
import random
from typing import Generator, List, Dict, Any
from dataclasses import dataclass, asdict


# ============================================================
# 1. SSE(Server-Sent Events)协议
# ============================================================

class SSEMessage:
    """SSE消息封装

    SSE协议格式:
    event: 事件类型
    data: 数据内容
    id: 消息ID(可选)
    retry: 重连间隔(可选)

    每条消息以两个换行符结尾。
    """

    def __init__(self, data: str, event: str = "message",
                 msg_id: str = None, retry: int = None):
        self.data = data
        self.event = event
        self.msg_id = msg_id
        self.retry = retry

    def encode(self) -> str:
        """编码为SSE格式字符串"""
        lines = []
        if self.event:
            lines.append(f"event: {self.event}")
        if self.msg_id:
            lines.append(f"id: {self.msg_id}")
        if self.retry:
            lines.append(f"retry: {self.retry}")
        lines.append(f"data: {self.data}")
        return "\n".join(lines) + "\n\n"

    @staticmethod
    def parse(raw: str) -> 'SSEMessage':
        """解析SSE格式字符串"""
        event = "message"
        data = ""
        msg_id = None
        retry = None

        for line in raw.strip().split("\n"):
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data = line[6:]
            elif line.startswith("id: "):
                msg_id = line[4:]
            elif line.startswith("retry: "):
                retry = int(line[7:])

        return SSEMessage(data=data, event=event, msg_id=msg_id, retry=retry)


class SSEClient:
    """SSE客户端(模拟), 演示如何接收和处理SSE事件"""

    def __init__(self):
        self.event_handlers = {}

    def on(self, event: str, handler):
        """注册事件处理器"""
        self.event_handlers[event] = handler

    def handle_message(self, sse_raw: str):
        """处理接收到的SSE消息"""
        msg = SSEMessage.parse(sse_raw)
        handler = self.event_handlers.get(msg.event)
        if handler:
            handler(msg.data)
        else:
            print(f"[未处理事件] {msg.event}: {msg.data}")


# ============================================================
# 2. 流式生成器
# ============================================================

class StreamingGenerator:
    """流式文本生成器

    模拟LLM的流式输出行为。
    实际应用中应替换为真实的LLM API调用。
    """

    def __init__(self, chunk_size: int = 3, delay: float = 0.05):
        """
        参数:
            chunk_size: 每次输出的字符数
            delay: 模拟生成延迟(秒)
        """
        self.chunk_size = chunk_size
        self.delay = delay

    def generate(self, text: str) -> Generator[str, None, None]:
        """将完整文本按chunk_size逐步输出"""
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i:i + self.chunk_size]
            if self.delay:
                time.sleep(self.delay)
            yield chunk

    def generate_with_metadata(self, text: str,
                                metadata: Dict = None) -> Generator[Dict, None, None]:
        """流式输出, 附带元数据"""
        total_chars = len(text)
        for i, chunk in enumerate(self.generate(text)):
            yield {
                'chunk': chunk,
                'index': i,
                'progress': min((i + 1) * self.chunk_size / total_chars, 1.0),
                'metadata': metadata or {},
            }

    def generate_sse(self, text: str,
                      session_id: str = None) -> Generator[str, None, None]:
        """生成SSE格式的流式输出"""
        # 发送开始事件
        start_data = json.dumps({
            'type': 'start',
            'session_id': session_id,
            'total_length': len(text),
        }, ensure_ascii=False)
        yield SSEMessage(data=start_data, event="start").encode()

        # 逐步发送内容
        for i, chunk in enumerate(self.generate(text)):
            content_data = json.dumps({
                'type': 'content',
                'content': chunk,
                'index': i,
            }, ensure_ascii=False)
            yield SSEMessage(data=content_data, event="content").encode()

        # 发送结束事件
        end_data = json.dumps({
            'type': 'end',
            'total_chunks': (len(text) + self.chunk_size - 1) // self.chunk_size,
        }, ensure_ascii=False)
        yield SSEMessage(data=end_data, event="end").encode()


# ============================================================
# 3. 模拟LLM响应
# ============================================================

class MockLLM:
    """模拟LLM, 生成不同风格的回答"""

    RESPONSES = {
        "什么是RAG": "RAG(检索增强生成, Retrieval-Augmented Generation)是一种将信息检索与文本生成相结合的技术框架。"
                      "其核心思想是: 在生成回答前, 先从知识库中检索相关文档, 然后将检索到的内容作为上下文传递给大语言模型, "
                      "从而生成更准确、更有依据的回答。RAG有效解决了LLM的幻觉问题, 让AI的回答有据可依。",

        "Python优势": "Python之所以成为数据科学和AI领域的首选语言, 主要有以下几个优势: "
                      "第一, 语法简洁优雅, 学习门槛低; 第二, 拥有丰富的第三方库生态, 如NumPy、Pandas、PyTorch等; "
                      "第三, 社区活跃, 遇到问题容易找到解决方案; 第四, 跨平台兼容性好。",

        "default": "这是一个很好的问题。让我从几个方面来分析: "
                   "首先, 从技术角度来看, 这个问题涉及到多个核心概念的理解。"
                   "其次, 从实践角度来看, 需要结合具体场景来应用。"
                   "最后, 持续学习和实践是掌握任何技术的关键。"
    }

    def get_response(self, query: str) -> str:
        """获取回答"""
        for key, response in self.RESPONSES.items():
            if key in query:
                return response
        return self.RESPONSES["default"]


# ============================================================
# 4. RAG流式Pipeline
# ============================================================

class StreamingRAGPipeline:
    """支持流式输出的RAG Pipeline"""

    def __init__(self):
        self.generator = StreamingGenerator(chunk_size=5, delay=0.03)
        self.llm = MockLLM()

    def _mock_retrieve(self, query: str) -> List[Dict]:
        """模拟检索"""
        return [
            {"text": "RAG系统通过检索增强生成质量", "score": 0.95},
            {"text": "向量检索是RAG的核心组件", "score": 0.88},
        ]

    def stream_query(self, query: str) -> Generator[str, None, None]:
        """流式查询(纯文本流)"""
        # 1. 检索
        docs = self._mock_retrieve(query)
        context = "\n".join(d['text'] for d in docs)

        # 2. 生成(流式)
        response = self.llm.get_response(query)

        # 前缀(来源信息)
        yield f"📚 检索到{len(docs)}篇相关文档\n\n"

        # 流式输出回答
        for chunk in self.generator.generate(response):
            yield chunk

        # 后缀
        yield f"\n\n---\n📎 参考: {len(docs)}篇文档"

    def stream_query_sse(self, query: str,
                          session_id: str = None) -> Generator[str, None, None]:
        """流式查询(SSE格式)"""
        docs = self._mock_retrieve(query)
        response = self.llm.get_response(query)

        # SSE流
        yield from self.generator.generate_sse(response, session_id=session_id)


# ============================================================
# 5. FastAPI集成示例(伪代码)
# ============================================================

def print_fastapi_example():
    """打印FastAPI集成示例代码"""
    print("\n--- FastAPI StreamingResponse 集成示例 ---")
    print("""
# pip install fastapi uvicorn sse-starlette

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

app = FastAPI()
pipeline = StreamingRAGPipeline()

@app.get("/chat/stream")
async def chat_stream(query: str):
    '''SSE流式接口'''
    def event_generator():
        for sse_msg in pipeline.stream_query_sse(query):
            yield sse_msg
    return EventSourceResponse(event_generator())

@app.get("/chat/text")
async def chat_text(query: str):
    '''纯文本流式接口'''
    def text_generator():
        for chunk in pipeline.stream_query(query):
            yield chunk
    return StreamingResponse(
        text_generator(),
        media_type="text/plain; charset=utf-8"
    )

# 启动: uvicorn main:app --reload
""")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("W30-D5 流式输出")
    print("=" * 60)

    # --- 1. SSE协议演示 ---
    print("\n--- 1. SSE协议 ---")

    msg = SSEMessage(
        data='{"content": "你好"}',
        event="content",
        msg_id="1"
    )
    encoded = msg.encode()
    print(f"SSE编码:\n{encoded}")

    parsed = SSEMessage.parse(encoded)
    print(f"SSE解析: event={parsed.event}, data={parsed.data}")

    # --- 2. 流式生成器 ---
    print(f"\n{'='*60}")
    print("--- 2. 流式文本生成 ---")
    print(f"{'='*60}")

    generator = StreamingGenerator(chunk_size=5, delay=0.02)
    sample_text = "Python是一门优雅的编程语言, 广泛用于AI开发。"

    print("纯文本流:")
    full_response = ""
    for chunk in generator.generate(sample_text):
        print(chunk, end="", flush=True)
        full_response += chunk
    print(f"\n完整输出: {full_response}")

    print("\n带元数据流:")
    for item in generator.generate_with_metadata(sample_text, metadata={'source': 'test'}):
        bar = '█' * int(item['progress'] * 20)
        print(f"\r  进度: [{bar:<20}] {item['progress']:.0%}", end="", flush=True)
    print()

    # --- 3. SSE流式输出 ---
    print(f"\n\n{'='*60}")
    print("--- 3. SSE流式输出 ---")
    print(f"{'='*60}")

    generator2 = StreamingGenerator(chunk_size=8, delay=0.01)
    sse_output = ""
    for sse_msg in generator2.generate_sse("RAG是检索增强生成的缩写", session_id="abc123"):
        sse_output += sse_msg

    print("SSE完整输出:")
    print(sse_output[:300] + "...")

    # --- 4. 完整RAG流式Pipeline ---
    print(f"\n{'='*60}")
    print("--- 4. RAG流式Pipeline演示 ---")
    print(f"{'='*60}")

    pipeline = StreamingRAGPipeline()

    print("\n[纯文本流式输出]")
    for chunk in pipeline.stream_query("什么是RAG?"):
        print(chunk, end="", flush=True)
    print()

    print("\n\n[SSE格式流式输出]")
    for sse_chunk in pipeline.stream_query_sse("什么是RAG?", session_id="sess_001"):
        msg = SSEMessage.parse(sse_chunk)
        if msg.event == "content":
            data = json.loads(msg.data)
            print(data['content'], end="", flush=True)
    print()

    # --- 5. FastAPI示例 ---
    print_fastapi_example()

    print("\n完成!")
