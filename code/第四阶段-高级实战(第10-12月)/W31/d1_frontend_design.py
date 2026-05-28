"""
W31-D1 前端设计模板
====================
演示RAG系统的前端界面设计, 包括:
- Gradio聊天界面模板
- Streamlit界面模板
- 文档上传组件
- UI设计原则

前端是用户与RAG系统交互的窗口, 需要简洁直观。
"""

import time
import json
from typing import List, Dict, Optional


# ============================================================
# 1. UI组件模型
# ============================================================

class ChatMessage:
    """聊天消息模型"""

    def __init__(self, role: str, content: str, sources: List[Dict] = None,
                 timestamp: float = None):
        self.role = role          # 'user' 或 'assistant'
        self.content = content
        self.sources = sources or []
        self.timestamp = timestamp or time.time()
        self.feedback = None      # 'like' / 'dislike' / None

    def to_dict(self) -> Dict:
        return {
            'role': self.role,
            'content': self.content,
            'sources': self.sources,
            'timestamp': self.timestamp,
            'feedback': self.feedback,
        }

    def format_display(self) -> str:
        """格式化为显示文本"""
        text = f"[{'用户' if self.role == 'user' else 'AI'}] {self.content}"
        if self.sources:
            text += f"\n  参考: {len(self.sources)}篇文档"
        return text


class DocumentInfo:
    """文档信息模型"""

    def __init__(self, doc_id: str, title: str, status: str = "已索引",
                 chunk_count: int = 0, tags: List[str] = None):
        self.doc_id = doc_id
        self.title = title
        self.status = status
        self.chunk_count = chunk_count
        self.tags = tags or []
        self.upload_time = time.time()

    def to_dict(self) -> Dict:
        return {
            'doc_id': self.doc_id,
            'title': self.title,
            'status': self.status,
            'chunk_count': self.chunk_count,
            'tags': self.tags,
        }


# ============================================================
# 2. Gradio模板
# ============================================================

def gradio_chat_template():
    """生成Gradio聊天应用模板代码"""
    return '''
# Gradio RAG聊天界面
# 安装: pip install gradio

import gradio as gr

# --- 模拟RAG后端 ---
def rag_query(message, history, top_k, temperature):
    """RAG查询函数"""
    # 替换为实际的RAG系统调用
    sources = [
        {"title": "技术文档A", "score": 0.95},
        {"title": "技术文档B", "score": 0.82},
    ]
    answer = f"关于「{message}」的回答: 根据检索到的{len(sources)}篇文档..."
    return answer

# --- Gradio界面 ---
with gr.Blocks(title="RAG知识库问答", theme=gr.themes.Soft()) as app:
    gr.Markdown("# RAG知识库问答系统")
    gr.Markdown("上传文档, 提问获取AI回答")

    with gr.Row():
        # 左侧: 聊天区域
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=500, show_copy_button=True)
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="输入问题...",
                    show_label=False,
                    scale=4
                )
                send_btn = gr.Button("发送", variant="primary", scale=1)

        # 右侧: 控制面板
        with gr.Column(scale=1):
            gr.Markdown("### 参数设置")
            top_k = gr.Slider(1, 20, value=5, step=1, label="检索文档数")
            temperature = gr.Slider(0, 2, value=0.7, step=0.1, label="温度")

            gr.Markdown("### 示例问题")
            examples = gr.Examples(
                examples=["什么是RAG?", "如何优化检索质量?", "推荐的学习路径"],
                inputs=msg_input
            )

    # 文档上传区域(底部)
    with gr.Accordion("文档管理", open=False):
        with gr.Row():
            file_upload = gr.File(
                label="上传文档",
                file_types=[".txt", ".pdf", ".md"],
                file_count="multiple"
            )
            upload_btn = gr.Button("处理并索引")

    # 事件绑定
    def respond(message, chat_history):
        response = rag_query(message, chat_history, top_k=5, temperature=0.7)
        chat_history.append((message, response))
        return "", chat_history

    send_btn.click(respond, [msg_input, chatbot], [msg_input, chatbot])
    msg_input.submit(respond, [msg_input, chatbot], [msg_input, chatbot])

app.launch(server_name="0.0.0.0", server_port=7860)
'''


def gradio_full_template():
    """完整的Gradio应用模板(含文档上传)"""
    return '''
# 完整Gradio应用
# pip install gradio

import gradio as gr
import time

class RAGApp:
    def __init__(self):
        self.documents = []

    def upload_files(self, files):
        results = []
        for f in files:
            self.documents.append(f.name)
            results.append(f"已索引: {f.name}")
        return "\\n".join(results)

    def query(self, message, history, top_k, temp):
        if not self.documents:
            return "请先上传文档"
        # 模拟检索和生成
        return f"基于{len(self.documents)}篇文档, 关于「{message}」的回答..."

    def get_docs(self):
        return "\\n".join(self.documents) if self.documents else "暂无文档"

app_instance = RAGApp()

with gr.Blocks(title="RAG系统", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# RAG知识库问答系统")

    with gr.Tab("聊天"):
        chatbot = gr.Chatbot(height=400)
        with gr.Row():
            inp = gr.Textbox(placeholder="输入问题...", scale=4)
            btn = gr.Button("发送", variant="primary")
        with gr.Row():
            top_k = gr.Slider(1, 20, value=5, label="Top-K")
            temp = gr.Slider(0, 2, value=0.7, label="温度")

    with gr.Tab("文档管理"):
        file_input = gr.File(file_count="multiple", label="上传文档")
        upload_btn = gr.Button("处理文档")
        doc_list = gr.Textbox(label="已索引文档", lines=10)

    with gr.Tab("设置"):
        gr.Markdown("### 系统设置")
        gr.Textbox(value="v1.0.0", label="版本")

    # 事件
    def respond(msg, hist):
        resp = app_instance.query(msg, hist, 5, 0.7)
        hist.append((msg, resp))
        return "", hist

    btn.click(respond, [inp, chatbot], [inp, chatbot])
    inp.submit(respond, [inp, chatbot], [inp, chatbot])
    upload_btn.click(app_instance.upload_files, [file_input], [doc_list])

demo.launch()
'''


# ============================================================
# 3. Streamlit模板
# ============================================================

def streamlit_template():
    """生成Streamlit应用模板代码"""
    return '''
# Streamlit RAG应用模板
# 安装: pip install streamlit
# 运行: streamlit run app.py

import streamlit as st

# --- 页面配置 ---
st.set_page_config(
    page_title="RAG知识库",
    page_icon="📚",
    layout="wide"
)

# --- 初始化Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents" not in st.session_state:
    st.session_state.documents = []

# --- 侧边栏 ---
with st.sidebar:
    st.header("设置")

    # 参数设置
    top_k = st.slider("检索文档数", 1, 20, 5)
    temperature = st.slider("温度", 0.0, 2.0, 0.7, 0.1)

    st.header("文档管理")
    uploaded_files = st.file_uploader(
        "上传文档",
        type=["txt", "pdf", "md"],
        accept_multiple_files=True
    )
    if uploaded_files:
        for f in uploaded_files:
            st.session_state.documents.append(f.name)
        st.success(f"已上传 {len(uploaded_files)} 个文件")

    # 已索引文档
    if st.session_state.documents:
        st.subheader("已索引文档")
        for doc in st.session_state.documents:
            st.text(doc)

    # 清除对话
    if st.button("清除对话"):
        st.session_state.messages = []

# --- 主区域 ---
st.title("📚 RAG知识库问答")

# 显示对话历史
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 输入框
if prompt := st.chat_input("输入你的问题..."):
    # 用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI回答
    with st.chat_message("assistant"):
        response = f"关于「{prompt}」的回答(基于{len(st.session_state.documents)}篇文档)..."
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
'''


# ============================================================
# 4. UI设计原则
# ============================================================

def print_ui_principles():
    """打印UI设计原则"""
    print("""
RAG系统前端设计原则:
========================================

1. 简洁性
   - 聊天界面为核心, 减少不必要的元素
   - 一目了然的操作流程

2. 可追溯性
   - 显示回答的参考来源
   - 支持点击查看原始文档

3. 反馈机制
   - 点赞/点踩按钮收集用户反馈
   - 显示"正在思考"等加载状态

4. 参数透明
   - 让用户可以调节Top-K、温度等参数
   - 显示检索到的文档数量和相关性

5. 错误处理
   - 友好的错误提示
   - 网络断开时的重连机制

6. 响应式设计
   - 适配桌面和移动设备
   - 支持深色模式

7. 文档管理
   - 支持拖拽上传
   - 显示文档处理进度
   - 文档列表和删除功能
""")


# ============================================================
# 5. 界面组件模拟
# ============================================================

def simulate_chat_interface():
    """模拟聊天界面(命令行版本)"""
    print("\n--- 聊天界面模拟 ---")
    print("=" * 50)

    messages = []
    sample_conversation = [
        ("user", "什么是RAG?"),
        ("assistant", "RAG(检索增强生成)是一种结合信息检索和文本生成的AI技术。它通过先检索相关文档, 再让LLM基于检索结果生成回答。\n\n参考来源:\n[1] RAG技术白皮书 (相关度: 0.95)\n[2] AI系统设计指南 (相关度: 0.82)"),
        ("user", "它和微调有什么区别?"),
        ("assistant", "RAG和微调是两种不同的LLM增强方式:\n\n1. RAG: 通过检索外部知识来增强回答, 无需重新训练模型\n2. 微调: 在特定数据上训练模型, 修改模型参数\n\nRAG适合知识频繁更新的场景, 微调适合需要特定风格或领域深度的场景。\n\n参考来源:\n[1] LLM优化策略对比 (相关度: 0.91)"),
    ]

    for role, content in sample_conversation:
        if role == "user":
            print(f"\n  你: {content}")
        else:
            print(f"\n  AI: {content}")
        messages.append(ChatMessage(role, content))

    return messages


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("W31-D1 前端设计模板")
    print("=" * 60)

    # --- 1. Gradio模板 ---
    print("\n--- 1. Gradio聊天界面模板 ---")
    print("Gradio是快速构建ML应用UI的Python框架")
    print("特点: 代码简洁, 内置聊天组件, 支持文件上传")
    print("\n模板代码已生成, 保存为 .py 文件后运行即可")

    # --- 2. Streamlit模板 ---
    print("\n--- 2. Streamlit界面模板 ---")
    print("Streamlit是数据科学应用的快速开发框架")
    print("特点: 声明式语法, 自动响应更新, 丰富的组件")

    # --- 3. UI设计原则 ---
    print_ui_principles()

    # --- 4. 界面模拟 ---
    messages = simulate_chat_interface()

    # --- 5. 模板文件输出 ---
    print(f"\n{'='*60}")
    print("--- 5. 模板代码 ---")
    print(f"{'='*60}")

    # 输出Gradio模板路径提示
    print("\nGradio模板代码(约60行):")
    print("  包含: 聊天界面, 文档上传, 参数调节, 示例问题")
    print("\nStreamlit模板代码(约70行):")
    print("  包含: 聊天界面, 侧边栏设置, 文档管理, 对话历史")

    # 保存模板
    templates = {
        'gradio_chat': gradio_chat_template(),
        'gradio_full': gradio_full_template(),
        'streamlit': streamlit_template(),
    }

    for name, code in templates.items():
        print(f"\n  模板 '{name}': {len(code)} 字符")

    print("\n提示: 将模板代码复制到 .py 文件中运行")
    print("  Gradio: python app.py")
    print("  Streamlit: streamlit run app.py")

    print("\n完成!")
