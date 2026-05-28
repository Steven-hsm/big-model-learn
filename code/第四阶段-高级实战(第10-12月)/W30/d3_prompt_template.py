"""
W30-D3 Prompt模板管理
======================
实现Prompt模板系统, 包括:
- 模板定义与渲染
- 系统Prompt设计
- 上下文窗口管理
- 引用溯源格式化

良好的Prompt管理是RAG系统的核心组件之一。
"""

import json
import time
from typing import List, Dict, Optional
from string import Template


# ============================================================
# 1. Prompt模板系统
# ============================================================

class PromptTemplate:
    """Prompt模板管理器

    支持:
    - 变量插值 ({{variable}})
    - 条件渲染
    - 默认值
    - 模板继承
    """

    def __init__(self, template_str: str, name: str = "unnamed"):
        self.template_str = template_str
        self.name = name
        self.variables = self._extract_variables()

    def _extract_variables(self):
        """提取模板中的变量名"""
        import re
        return set(re.findall(r'\{\{(\w+)\}\}', self.template_str))

    def render(self, **kwargs) -> str:
        """渲染模板, 替换变量"""
        result = self.template_str
        for var in self.variables:
            placeholder = '{{' + var + '}}'
            value = kwargs.get(var, '')
            result = result.replace(placeholder, str(value))
        return result.strip()

    def validate(self, **kwargs) -> List[str]:
        """验证渲染所需的变量是否齐全"""
        missing = []
        for var in self.variables:
            if var not in kwargs and f'default_{var}' not in kwargs:
                missing.append(var)
        return missing

    def __repr__(self):
        return f"PromptTemplate(name='{self.name}', variables={self.variables})"


class PromptManager:
    """Prompt模板集合管理"""

    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = {}
        self._init_default_templates()

    def _init_default_templates(self):
        """初始化常用模板"""
        # RAG问答模板
        self.register('rag_qa', PromptTemplate(
            """你是一个专业的知识助手。请根据以下参考资料回答用户的问题。

## 参考资料
{{context}}

## 用户问题
{{question}}

## 回答要求
1. 基于参考资料回答, 不要编造信息
2. 如果参考资料不足以回答问题, 请明确说明
3. 在回答中标注信息来源[1][2]等
4. 用清晰的结构化格式回答

## 回答""",
            name='rag_qa'
        ))

        # 系统Prompt模板
        self.register('system', PromptTemplate(
            """你是一个{{role}}。你的职责是:
{{responsibilities}}

回答风格:
- 语气: {{tone}}
- 语言: {{language}}
- 详细程度: {{detail_level}}""",
            name='system'
        ))

        # 摘要模板
        self.register('summary', PromptTemplate(
            """请对以下内容生成{{length}}字的摘要。

## 原文
{{content}}

## 摘要要求
- 保留关键信息
- 语言简洁
- 逻辑清晰

## 摘要""",
            name='summary'
        ))

        # 翻译模板
        self.register('translate', PromptTemplate(
            """请将以下文本从{{source_lang}}翻译为{{target_lang}}。

## 原文
{{content}}

## 翻译
""",
            name='translate'
        ))

    def register(self, name: str, template: PromptTemplate):
        """注册模板"""
        self.templates[name] = template

    def get(self, name: str) -> Optional[PromptTemplate]:
        """获取模板"""
        return self.templates.get(name)

    def render(self, name: str, **kwargs) -> str:
        """渲染指定模板"""
        template = self.get(name)
        if template is None:
            raise ValueError(f"模板 '{name}' 不存在")
        return template.render(**kwargs)

    def list_templates(self) -> List[str]:
        """列出所有模板名"""
        return list(self.templates.keys())


# ============================================================
# 2. 系统Prompt设计
# ============================================================

class SystemPromptBuilder:
    """系统Prompt构建器

    一个好的系统Prompt应该包含:
    1. 角色定义 (你是谁)
    2. 能力边界 (你能做什么/不能做什么)
    3. 回答规范 (格式、语气、长度)
    4. 安全约束 (不回答什么)
    """

    def __init__(self):
        self.role = ""
        self.capabilities = []
        self.limitations = []
        self.guidelines = []
        self.safety_rules = []
        self.output_format = ""

    def set_role(self, role: str, description: str = ""):
        self.role = role
        if description:
            self.capabilities.append(description)
        return self

    def add_capability(self, capability: str):
        self.capabilities.append(capability)
        return self

    def add_limitation(self, limitation: str):
        self.limitations.append(limitation)
        return self

    def add_guideline(self, guideline: str):
        self.guidelines.append(guideline)
        return self

    def add_safety_rule(self, rule: str):
        self.safety_rules.append(rule)
        return self

    def set_output_format(self, fmt: str):
        self.output_format = fmt
        return self

    def build(self) -> str:
        """构建完整的系统Prompt"""
        parts = []

        # 角色
        if self.role:
            parts.append(f"## 角色\n你是{self.role}。")

        # 能力
        if self.capabilities:
            parts.append("## 能力")
            for cap in self.capabilities:
                parts.append(f"- {cap}")

        # 限制
        if self.limitations:
            parts.append("## 限制")
            for lim in self.limitations:
                parts.append(f"- {lim}")

        # 指南
        if self.guidelines:
            parts.append("## 回答指南")
            for guide in self.guidelines:
                parts.append(f"- {guide}")

        # 安全规则
        if self.safety_rules:
            parts.append("## 安全规则")
            for rule in self.safety_rules:
                parts.append(f"- {rule}")

        # 输出格式
        if self.output_format:
            parts.append(f"## 输出格式\n{self.output_format}")

        return "\n\n".join(parts)


# ============================================================
# 3. 上下文窗口管理
# ============================================================

class ContextWindowManager:
    """上下文窗口管理器

    职责:
    1. 控制总token数不超过模型限制
    2. 优先保留最相关的上下文
    3. 在上下文和生成空间之间平衡
    """

    def __init__(self, max_tokens: int = 4096,
                 reserved_for_system: int = 500,
                 reserved_for_generation: int = 1000):
        self.max_tokens = max_tokens
        self.reserved_for_system = reserved_for_system
        self.reserved_for_generation = reserved_for_generation

    @property
    def available_for_context(self):
        """可用于上下文的token数"""
        return self.max_tokens - self.reserved_for_system - self.reserved_for_generation

    def estimate_tokens(self, text: str) -> int:
        """简单估算token数(中文约1.5字/token, 英文约4字符/token)"""
        chinese_chars = sum(1 for c in text if '一' <= c <= '鿿')
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4)

    def select_contexts(self, query: str, documents: List[Dict],
                        strategy: str = "relevance") -> List[Dict]:
        """选择要包含在上下文中的文档

        参数:
            query: 用户查询
            documents: 候选文档列表, 每个包含 'text' 和 'score'
            strategy: 选择策略
                - 'relevance': 按相关性依次添加
                - 'diverse': 优先选不同来源的文档
                - 'recent': 优先选最近的文档
        """
        budget = self.available_for_context
        selected = []
        used_tokens = 0

        if strategy == 'relevance':
            # 按分数降序排列
            sorted_docs = sorted(documents, key=lambda x: x.get('score', 0), reverse=True)
            for doc in sorted_docs:
                doc_tokens = self.estimate_tokens(doc['text'])
                if used_tokens + doc_tokens <= budget:
                    selected.append(doc)
                    used_tokens += doc_tokens
                else:
                    # 尝试截断
                    remaining = budget - used_tokens
                    if remaining > 100:  # 至少保留100 token
                        truncated = self._truncate_text(doc['text'], remaining)
                        doc_copy = doc.copy()
                        doc_copy['text'] = truncated
                        doc_copy['truncated'] = True
                        selected.append(doc_copy)
                        used_tokens += self.estimate_tokens(truncated)
                    break

        elif strategy == 'diverse':
            # 优先选不同来源
            sources_seen = set()
            remaining_docs = list(documents)
            while remaining_docs and used_tokens < budget:
                for doc in remaining_docs[:]:
                    source = doc.get('source', 'unknown')
                    if source not in sources_seen:
                        doc_tokens = self.estimate_tokens(doc['text'])
                        if used_tokens + doc_tokens <= budget:
                            selected.append(doc)
                            used_tokens += doc_tokens
                            sources_seen.add(source)
                            remaining_docs.remove(doc)
                            break
                else:
                    # 所有来源都已选过, 选最相关的
                    best = max(remaining_docs, key=lambda x: x.get('score', 0))
                    doc_tokens = self.estimate_tokens(best['text'])
                    if used_tokens + doc_tokens <= budget:
                        selected.append(best)
                        used_tokens += doc_tokens
                    remaining_docs.remove(best)

        return selected

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """截断文本到指定token数"""
        # 按句子截断
        sentences = text.replace('。', '。\n').replace('！', '！\n').replace('？', '？\n').split('\n')
        result = ""
        for sent in sentences:
            if self.estimate_tokens(result + sent) <= max_tokens:
                result += sent
            else:
                break
        return result if result else text[:int(max_tokens * 1.5)]

    def format_context(self, documents: List[Dict]) -> str:
        """格式化上下文文档"""
        parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.get('source', '未知来源')
            text = doc['text']
            truncated = " [内容已截断]" if doc.get('truncated') else ""
            parts.append(f"[{i}] (来源: {source}){truncated}\n{text}")
        return "\n\n".join(parts)


# ============================================================
# 4. 引用溯源格式化
# ============================================================

class CitationFormatter:
    """引用溯源格式化器

    在RAG回答中标注信息来源, 让用户知道每个事实的出处。
    """

    def __init__(self):
        self.citation_style = "inline"  # inline / footnote / endnote

    def format_answer_with_citations(self, answer: str,
                                      sources: List[Dict]) -> str:
        """为回答添加引用标注

        参数:
            answer: 生成的回答
            sources: 来源文档列表, 包含 text, source, page 等
        """
        if self.citation_style == "inline":
            return self._inline_citation(answer, sources)
        elif self.citation_style == "footnote":
            return self._footnote_citation(answer, sources)
        else:
            return self._endnote_citation(answer, sources)

    def _inline_citation(self, answer: str, sources: List[Dict]) -> str:
        """行内引用: 在事实后直接标注[1][2]"""
        # 在实际应用中, 应通过NLP匹配回答中的事实与来源
        # 这里简化处理: 在每个句子的结尾添加引用
        result = answer
        for i, source in enumerate(sources[:3], 1):
            source_info = f" [{i}]({source.get('source', '')})"
            # 在第一个句号后插入
            if i == 1 and '。' in result:
                pos = result.index('。')
                result = result[:pos+1] + source_info + result[pos+1:]
        return result

    def _footnote_citation(self, answer: str, sources: List[Dict]) -> str:
        """脚注引用: 回答末尾列出来源"""
        footnotes = []
        for i, source in enumerate(sources, 1):
            src = source.get('source', '未知')
            page = source.get('page', '')
            page_info = f", 第{page}页" if page else ""
            confidence = source.get('score', 0)
            footnotes.append(f"[{i}] {src}{page_info} (相关度: {confidence:.2f})")

        return f"{answer}\n\n---\n**参考来源:**\n" + "\n".join(footnotes)

    def _endnote_citation(self, answer: str, sources: List[Dict]) -> str:
        """尾注引用"""
        endnotes = []
        for i, source in enumerate(sources, 1):
            text_preview = source.get('text', '')[:60]
            endnotes.append(f"[{i}] \"{text_preview}...\" - {source.get('source', '未知')}")

        return f"{answer}\n\n**参考来源:**\n" + "\n".join(endnotes)


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("W30-D3 Prompt模板管理")
    print("=" * 60)

    # --- 1. 模板管理演示 ---
    print("\n--- 1. Prompt模板管理 ---")
    manager = PromptManager()
    print(f"可用模板: {manager.list_templates()}")

    # 渲染RAG模板
    rag_prompt = manager.render('rag_qa',
        context="RAG(检索增强生成)是一种结合信息检索和文本生成的技术[1]。",
        question="什么是RAG?")
    print(f"\nRAG问答模板渲染结果:\n{rag_prompt}")

    # --- 2. 系统Prompt构建 ---
    print(f"\n{'='*60}")
    print("--- 2. 系统Prompt设计 ---")
    print(f"{'='*60}")

    builder = SystemPromptBuilder()
    system_prompt = (builder
        .set_role("AI知识库助手", "根据知识库内容准确回答用户问题")
        .add_capability("回答关于公司产品和政策的问题")
        .add_capability("提供技术文档的查询和解释")
        .add_limitation("不回答知识库之外的问题")
        .add_limitation("不提供投资建议或医疗诊断")
        .add_guideline("回答时引用具体的文档来源")
        .add_guideline("如果不确定, 主动说明而非猜测")
        .add_safety_rule("不泄露用户隐私信息")
        .set_output_format("使用Markdown格式, 包含标题和列表")
        .build()
    )
    print(system_prompt)

    # --- 3. 上下文窗口管理 ---
    print(f"\n{'='*60}")
    print("--- 3. 上下文窗口管理 ---")
    print(f"{'='*60}")

    ctx_manager = ContextWindowManager(max_tokens=1000)
    print(f"总token预算: {ctx_manager.max_tokens}")
    print(f"系统Prompt预留: {ctx_manager.reserved_for_system}")
    print(f"生成预留: {ctx_manager.reserved_for_generation}")
    print(f"可用于上下文: {ctx_manager.available_for_context}")

    documents = [
        {"text": "Python是一种解释型高级编程语言, 支持多种编程范式。" * 5,
         "score": 0.95, "source": "Python官方文档"},
        {"text": "机器学习通过数据训练模型, 实现智能预测和决策。" * 3,
         "score": 0.85, "source": "ML入门指南"},
        {"text": "深度学习使用多层神经网络, 可以自动学习特征表示。" * 4,
         "score": 0.75, "source": "DL教程"},
        {"text": "自然语言处理研究计算机理解和生成人类语言的技术。" * 6,
         "score": 0.65, "source": "NLP手册"},
    ]

    selected = ctx_manager.select_contexts("Python编程", documents, strategy='relevance')
    print(f"\n选择的文档数: {len(selected)}")
    formatted = ctx_manager.format_context(selected)
    print(f"格式化上下文 (约{ctx_manager.estimate_tokens(formatted)} tokens):\n{formatted[:200]}...")

    # --- 4. 引用溯源 ---
    print(f"\n{'='*60}")
    print("--- 4. 引用溯源格式化 ---")
    print(f"{'='*60}")

    formatter = CitationFormatter()

    answer = "RAG是一种结合信息检索和文本生成的技术。它通过检索相关文档来增强大语言模型的回答质量。"
    sources = [
        {"text": "RAG系统通过检索增强生成", "source": "AI技术报告", "page": 12, "score": 0.95},
        {"text": "大语言模型的局限性及改进", "source": "LLM综述", "page": 45, "score": 0.88},
        {"text": "信息检索系统设计", "source": "IR教科书", "page": 78, "score": 0.72},
    ]

    for style in ['inline', 'footnote', 'endnote']:
        formatter.citation_style = style
        result = formatter.format_answer_with_citations(answer, sources)
        print(f"\n[{style}引用风格]:\n{result}")

    print("\n完成!")
