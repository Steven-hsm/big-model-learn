"""
W31-D2 Gradio完整应用
======================
实现一个完整的Gradio RAG应用, 包括:
- 聊天界面
- 文档上传与管理
- 参数调节面板
- 示例问题

安装: pip install gradio
"""

import time
import json
import hashlib
from typing import List, Dict, Tuple, Optional

try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False

try:
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    HAS_PLT = True
except ImportError:
    HAS_PLT = False


# ============================================================
# 1. RAG后端系统
# ============================================================

class SimpleRAGBackend:
    """简化的RAG后端"""

    def __init__(self):
        self.documents: Dict[str, Dict] = {}
        self.chat_history: List[Dict] = []
        self.stats = {'queries': 0, 'uploads': 0}

    def upload_document(self, title: str, content: str, tags: List[str] = None) -> str:
        """上传并索引文档"""
        doc_id = hashlib.md5(f"{title}_{time.time()}".encode()).hexdigest()[:10]
        self.documents[doc_id] = {
            'doc_id': doc_id,
            'title': title,
            'content': content,
            'tags': tags or [],
            'chunks': self._split(content),
            'uploaded_at': time.time(),
        }
        self.stats['uploads'] += 1
        return doc_id

    def _split(self, text: str, size: int = 200) -> List[str]:
        """简单分块"""
        chunks = []
        for i in range(0, max(len(text), 1), size):
            chunk = text[i:i+size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks if chunks else [text]

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """关键词搜索"""
        q_words = set(query.lower().split())
        results = []
        for doc_id, doc in self.documents.items():
            best_score = 0
            for chunk in doc['chunks']:
                c_words = set(chunk.lower().split())
                overlap = q_words & c_words
                if overlap:
                    score = len(overlap) / max(len(q_words), 1)
                    best_score = max(best_score, score)
            if best_score > 0:
                results.append({
                    'doc_id': doc_id,
                    'title': doc['title'],
                    'score': best_score,
                    'preview': doc['chunks'][0][:80] if doc['chunks'] else "",
                })
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

    def query(self, question: str, top_k: int = 5, temperature: float = 0.7) -> Dict:
        """查询RAG系统"""
        self.stats['queries'] += 1
        start = time.time()

        results = self.search(question, top_k)

        if results:
            answer = f"基于{len(results)}篇相关文档, 关于「{question}」:\n\n"
            for i, r in enumerate(results[:3], 1):
                answer += f"{i}. {r['preview']}...\n"
                answer += f"   (来源: {r['title']}, 相关度: {r['score']:.2f})\n\n"
        else:
            answer = f"未找到与「{question}」相关的文档。请先上传相关文档, 或尝试不同的关键词。"

        self.chat_history.append({
            'question': question,
            'answer': answer,
            'sources': len(results),
            'latency_ms': (time.time() - start) * 1000,
        })

        return {
            'answer': answer,
            'sources': results,
            'latency_ms': (time.time() - start) * 1000,
        }

    def get_document_list(self) -> str:
        """获取文档列表"""
        if not self.documents:
            return "暂无文档, 请上传。"
        lines = []
        for doc in self.documents.values():
            lines.append(f"- {doc['title']} ({len(doc['chunks'])}块, 标签: {', '.join(doc['tags'])})")
        return "\n".join(lines)

    def get_stats(self) -> Dict:
        return {
            'documents': len(self.documents),
            'queries': self.stats['queries'],
            'uploads': self.stats['uploads'],
        }


# ============================================================
# 2. Gradio界面构建
# ============================================================

def create_gradio_app():
    """创建Gradio应用"""
    backend = SimpleRAGBackend()

    # 预加载示例文档
    sample_docs = [
        ("RAG技术介绍", "RAG(检索增强生成)是一种结合信息检索和文本生成的人工智能技术。它通过先从知识库中检索相关文档, 再让大语言模型基于检索结果生成回答。", ["RAG", "AI"]),
        ("Python编程指南", "Python是一种优雅且强大的编程语言, 广泛用于数据科学、AI开发和Web开发。其简洁的语法和丰富的库使其成为初学者的理想选择。", ["Python", "编程"]),
        ("向量数据库原理", "向量数据库专为存储和检索高维向量而设计, 是现代RAG系统的核心基础设施。常见产品包括Milvus、Pinecone和Weaviate。", ["数据库", "向量"]),
    ]
    for title, content, tags in sample_docs:
        backend.upload_document(title, content, tags)

    # --- 回调函数 ---
    def chat_respond(message, history, top_k, temperature):
        """处理聊天消息"""
        if not message.strip():
            return history
        result = backend.query(message, top_k=int(top_k), temperature=temperature)
        history.append((message, result['answer']))
        return history

    def upload_files(files):
        """处理文件上传"""
        if not files:
            return "未选择文件"
        results = []
        for f in files:
            try:
                with open(f.name, 'r', encoding='utf-8') as fh:
                    content = fh.read()
                backend.upload_document(f.name.split('/')[-1], content)
                results.append(f"已索引: {f.name.split('/')[-1]}")
            except Exception as e:
                results.append(f"失败: {f.name} ({e})")
        return "\n".join(results)

    def refresh_docs():
        """刷新文档列表"""
        return backend.get_document_list()

    def get_system_info():
        """获取系统信息"""
        stats = backend.get_stats()
        return f"文档数: {stats['documents']}\n查询次数: {stats['queries']}"

    # --- 构建界面 ---
    with gr.Blocks(title="RAG知识库问答系统", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # RAG知识库问答系统
        上传文档构建知识库, 然后提问获取AI回答。
        """)

        with gr.Row():
            # 主聊天区域
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=450, show_copy_button=True,
                                      bubble_full_width=False)
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="输入你的问题...",
                        show_label=False,
                        scale=4
                    )
                    send_btn = gr.Button("发送", variant="primary", scale=1)
                    clear_btn = gr.Button("清除", scale=1)

                # 示例问题
                gr.Examples(
                    examples=[
                        "什么是RAG技术?",
                        "Python有哪些应用领域?",
                        "向量数据库是什么?",
                        "如何优化检索质量?",
                    ],
                    inputs=msg_input,
                    label="示例问题"
                )

            # 右侧面板
            with gr.Column(scale=1):
                gr.Markdown("### 参数设置")
                top_k = gr.Slider(1, 20, value=5, step=1,
                                   label="检索文档数 (Top-K)")
                temperature = gr.Slider(0, 2, value=0.7, step=0.1,
                                         label="生成温度")

                gr.Markdown("### 系统信息")
                info_display = gr.Textbox(
                    value=get_system_info(),
                    label="状态",
                    interactive=False,
                    lines=3
                )
                refresh_btn = gr.Button("刷新状态")

                gr.Markdown("### 反馈")
                gr.Markdown("回答质量如何? (在Gradio中可添加点赞按钮)")

        # 文档管理区域
        with gr.Accordion("文档管理", open=True):
            with gr.Row():
                file_upload = gr.File(
                    label="上传文档(.txt, .md)",
                    file_types=[".txt", ".md"],
                    file_count="multiple"
                )
            with gr.Row():
                upload_btn = gr.Button("处理并索引", variant="secondary")
                doc_list = gr.Textbox(
                    label="已索引文档",
                    value=backend.get_document_list(),
                    interactive=False,
                    lines=5
                )

        # --- 事件绑定 ---
        send_btn.click(
            chat_respond,
            inputs=[msg_input, chatbot, top_k, temperature],
            outputs=[chatbot]
        ).then(lambda: "", outputs=[msg_input])

        msg_input.submit(
            chat_respond,
            inputs=[msg_input, chatbot, top_k, temperature],
            outputs=[chatbot]
        ).then(lambda: "", outputs=[msg_input])

        clear_btn.click(lambda: [], outputs=[chatbot])
        upload_btn.click(upload_files, [file_upload], [doc_list])
        refresh_btn.click(get_system_info, outputs=[info_display])

    return app


# ============================================================
# 3. 纯Python模拟(不需要Gradio)
# ============================================================

def simulate_gradio_app():
    """在命令行模拟Gradio应用"""
    backend = SimpleRAGBackend()

    # 预加载文档
    sample_docs = [
        ("RAG技术介绍", "RAG是一种结合检索和生成的AI技术", ["RAG"]),
        ("Python入门", "Python是数据科学的首选语言", ["Python"]),
    ]
    for title, content, tags in sample_docs:
        backend.upload_document(title, content, tags)

    print("  Gradio应用模拟")
    print("=" * 50)
    print(f"  已加载 {len(backend.documents)} 篇文档")

    # 模拟对话
    queries = ["什么是RAG?", "Python有什么优势?"]
    for q in queries:
        result = backend.query(q, top_k=3)
        print(f"\n  用户: {q}")
        print(f"  AI: {result['answer'][:100]}...")
        print(f"  延迟: {result['latency_ms']:.1f}ms")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("W31-D2 Gradio完整应用")
    print("=" * 60)

    if HAS_GRADIO:
        print("\nGradio已安装! 创建应用...")
        app = create_gradio_app()
        print("应用创建成功!")
        print("启动方式: app.launch(server_name='0.0.0.0', server_port=7860)")
        print("\n功能:")
        print("  - 聊天界面: 支持多轮对话, 显示参考来源")
        print("  - 文档管理: 上传.txt/.md文件, 自动索引")
        print("  - 参数调节: Top-K和温度滑块")
        print("  - 示例问题: 一键提问")
        print("  - 系统状态: 实时显示文档数和查询次数")
    else:
        print("\nGradio未安装, 运行命令行模拟...")
        print("安装: pip install gradio")

    # 命令行模拟
    simulate_gradio_app()

    # 生成应用代码
    print(f"\n{'='*60}")
    print("Gradio应用结构:")
    print(f"{'='*60}")
    print("""
    +-------------------------------------------+
    |         RAG知识库问答系统                    |
    +-------------------------------------------+
    | [聊天区域]              | [控制面板]        |
    | 用户: 什么是RAG?        | Top-K: [===5===]  |
    | AI: RAG是一种...        | 温度: [=0.7===]   |
    |   参考: [1] 文档A       |                   |
    | 用户: 如何优化?         | [系统状态]        |
    | AI: 优化检索可以从...   | 文档数: 3         |
    |                        | 查询次数: 5       |
    +-------------------------------------------+
    | [文档管理]                                  |
    | [选择文件] [处理并索引]                      |
    | 已索引: RAG技术介绍, Python入门, ...         |
    +-------------------------------------------+
    """)

    print("完成!")
