# W22 - AI Agent开发

> 第22周学习计划 | Java开发工程师转AI开发 | 工作日每晚2小时 + 周末6-8小时

---

## 一、本周目标

1. 理解Agent的核心概念、与Chatbot的区别、四大核心组件
2. 掌握ReAct推理模式的原理和手动实现
3. 掌握Function Calling/Tool Use的完整流程和通用工具框架
4. 理解Agent记忆管理机制（短期记忆+长期记忆）
5. 完成AI智能助手实战项目（项目10）

---

## 二、时间安排

### 工作日（周一至周五，每晚2小时）

| 日期 | 主题 | 时间分配 |
|------|------|----------|
| Day 1 (周一) | Agent概念 | 理论学习90分钟 + 笔记30分钟 |
| Day 2 (周二) | ReAct模式 | 理论60分钟 + 手动实现60分钟 |
| Day 3 (周三) | Function Calling/Tool Use | 学习60分钟 + 实现工具框架60分钟 |
| Day 4 (周四) | Agent记忆管理 | 理论40分钟 + 实现AgentMemory 80分钟 |
| Day 5 (周五) | LangGraph基础 | 学习60分钟 + 代码练习60分钟 |

### 周末（周六至周日，每天6-8小时）

| 日期 | 主题 | 时间分配 |
|------|------|----------|
| Day 6 (周六) | Agent框架实战 | 上午CrewAI/Anthropic SDK(3h) + 下午实践(3h) + 练习(1h) |
| Day 7 (周日) | 实战项目10 - AI智能助手 | 全天实现项目(7-8h) |

---

## 三、详细学习内容

### Day 1: Agent概念

#### 3.1.1 什么是Agent

```
Agent = 感知环境 → 思考决策 → 执行动作 的自主系统

核心定义：
  Agent是一个能够自主感知环境、做出决策并执行动作以实现目标的AI系统。

与Chatbot的核心区别：
  Chatbot: 用户提问 → LLM生成文本回答 → 结束
  Agent:   用户提问 → LLM思考需要做什么 → 选择工具 → 执行 → 观察结果 → 继续思考 → ...
           （可以多轮思考和行动，直到达成目标）
```

#### 3.1.2 Agent核心组件

```
Agent = LLM（大脑）+ Memory（记忆）+ Tools（工具）+ Planning（规划）

┌─────────────────────────────────────────────┐
│                   Agent                      │
│                                              │
│   ┌─────────┐  ┌─────────┐  ┌──────────┐   │
│   │  LLM    │  │ Memory  │  │  Tools   │   │
│   │ (大脑)   │  │ (记忆)  │  │  (工具)  │   │
│   │         │  │         │  │          │   │
│   │推理/决策 │  │ 短期记忆 │  │ 搜索     │   │
│   │理解/生成 │  │ 长期记忆 │  │ 数据库   │   │
│   │         │  │         │  │ API调用  │   │
│   └────┬────┘  └────┬────┘  └────┬─────┘   │
│        │            │            │          │
│        └────────────┼────────────┘          │
│                     │                        │
│              ┌──────┴──────┐                 │
│              │  Planning   │                 │
│              │  (规划)     │                 │
│              │             │                 │
│              │ 任务分解    │                 │
│              │ 步骤编排    │                 │
│              │ 反思改进    │                 │
│              └─────────────┘                 │
└─────────────────────────────────────────────┘
```

#### 3.1.3 应用场景

```
| 场景       | 描述                        | 需要的工具         |
|-----------|-----------------------------|-------------------|
| 个人助理   | 帮用户安排日程、查询信息       | 日历API、搜索      |
| 数据分析   | 自动获取数据、分析、生成报告    | SQL、Python、图表  |
| 编程助手   | 理解需求、生成代码、执行测试    | 终端、文件系统      |
| 自动化运维 | 监控系统、诊断问题、执行修复    | SSH、监控API       |
| 客服机器人 | 理解问题、查询知识库、处理工单  | RAG、工单系统      |
```

---

### Day 2: ReAct模式

#### 3.2.1 ReAct推理循环

```
ReAct = Reasoning + Acting 交替循环

循环流程：
  Thought (思考): 分析当前状态，决定下一步
      ↓
  Action (行动): 选择并执行一个工具
      ↓
  Observation (观察): 获取工具执行结果
      ↓
  Thought (思考): 基于观察结果继续推理
      ↓
  ... 重复直到 ...
      ↓
  Answer (回答): 得出最终答案
```

#### 3.2.2 手动实现ReAct循环

```python
import json
from openai import OpenAI

class SimpleReActAgent:
    """手动实现ReAct Agent（不用框架）"""

    def __init__(self, tools):
        self.client = OpenAI()
        self.tools = tools
        self.tool_map = {t["name"]: t for t in tools}
        self.max_iterations = 5

    def run(self, question):
        """运行ReAct推理循环"""
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": question},
        ]

        for i in range(self.max_iterations):
            # Step 1: LLM思考
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0,
            )
            thought = response.choices[0].message.content
            messages.append({"role": "assistant", "content": thought})

            print(f"\n--- 第{i+1}轮 ---")
            print(f"思考: {thought}")

            # Step 2: 解析Action
            action = self._parse_action(thought)
            if action is None:
                # 没有Action，说明LLM认为已经可以回答了
                print("最终回答已生成")
                return thought

            # Step 3: 执行工具
            tool_name = action["name"]
            tool_args = action["args"]
            print(f"动作: {tool_name}({tool_args})")

            observation = self._execute_tool(tool_name, tool_args)
            print(f"观察: {observation}")

            # Step 4: 将观察结果返回给LLM
            messages.append({
                "role": "user",
                "content": f"Observation: {observation}"
            })

        return "达到最大迭代次数，未能完成任务。"

    def _build_system_prompt(self):
        """构建ReAct系统提示"""
        tool_descriptions = "\n".join([
            f"- {t['name']}: {t['description']}"
            for t in self.tools
        ])
        return f"""你是一个使用ReAct模式推理的AI助手。

你可以使用以下工具：
{tool_descriptions}

请严格按以下格式回答：

Thought: [你的思考过程]
Action: {{"name": "工具名", "args": {{"参数名": "参数值"}}}}

或者当你可以直接回答时：

Thought: [你的思考过程]
Answer: [你的最终答案]

每次只使用一个工具。"""

    def _parse_action(self, text):
        """从LLM输出中解析Action"""
        try:
            # 查找Action行
            for line in text.split("\n"):
                line = line.strip()
                if line.startswith("Action:"):
                    action_str = line.replace("Action:", "").strip()
                    action = json.loads(action_str)
                    return action
        except (json.JSONDecodeError, KeyError):
            pass
        return None

    def _execute_tool(self, tool_name, tool_args):
        """执行工具"""
        tool = self.tool_map.get(tool_name)
        if tool and "execute" in tool:
            try:
                return tool["execute"](**tool_args)
            except Exception as e:
                return f"工具执行错误: {str(e)}"
        return f"未知工具: {tool_name}"


# 定义工具
tools = [
    {
        "name": "search",
        "description": "搜索网络信息，返回相关结果",
        "execute": lambda query: json.dumps({
            "results": [f"关于'{query}'的搜索结果1", f"关于'{query}'的搜索结果2"]
        }),
    },
    {
        "name": "calculator",
        "description": "执行数学计算",
        "execute": lambda expression: str(eval(expression, {"__builtins__": {}}, {})),
    },
    {
        "name": "get_weather",
        "description": "获取指定城市的天气",
        "execute": lambda city: json.dumps({
            "北京": "晴天 25°C", "上海": "多云 28°C"
        }.get(city, "未知城市")),
    },
]

# 运行
agent = SimpleReActAgent(tools)
result = agent.run("北京今天天气怎么样？如果室外活动的话适合吗？")
print(f"\n最终结果: {result}")
```

---

### Day 3: Function Calling/Tool Use

#### 3.3.1 工具定义格式

```python
# OpenAI Function Calling 工具定义
tool_schema = {
    "type": "function",
    "function": {
        "name": "search_database",
        "description": "在数据库中搜索符合条件的记录",
        "parameters": {
            "type": "object",
            "properties": {
                "table": {
                    "type": "string",
                    "description": "表名",
                    "enum": ["users", "orders", "products"]
                },
                "conditions": {
                    "type": "object",
                    "description": "查询条件",
                    "properties": {
                        "field": {"type": "string"},
                        "operator": {"type": "string", "enum": ["=", ">", "<", "LIKE"]},
                        "value": {"type": "string"}
                    }
                },
                "limit": {
                    "type": "integer",
                    "description": "返回记录数上限",
                    "default": 10
                }
            },
            "required": ["table"]
        }
    }
}
```

#### 3.3.2 通用工具注册和调用框架

```python
import json
from typing import Callable, Dict, Any, List
from openai import OpenAI

class ToolRegistry:
    """通用工具注册和调用框架"""

    def __init__(self):
        self._tools: Dict[str, dict] = {}      # 工具定义
        self._handlers: Dict[str, Callable] = {}  # 工具处理函数

    def register(self, name: str, description: str, parameters: dict):
        """注册工具的装饰器"""
        def decorator(func: Callable):
            self._tools[name] = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                }
            }
            self._handlers[name] = func
            return func
        return decorator

    def get_openai_tools(self) -> List[dict]:
        """获取OpenAI格式的工具定义列表"""
        return list(self._tools.values())

    def execute(self, tool_name: str, tool_args: dict) -> str:
        """执行工具"""
        handler = self._handlers.get(tool_name)
        if handler is None:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        try:
            result = handler(**tool_args)
            return json.dumps(result, ensure_ascii=False) if not isinstance(result, str) else result
        except Exception as e:
            return json.dumps({"error": str(e)})

    def get_tool_names(self) -> List[str]:
        return list(self._tools.keys())


# 使用示例
registry = ToolRegistry()

@registry.register(
    name="search_web",
    description="搜索网络信息",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词"}
        },
        "required": ["query"]
    }
)
def search_web(query: str):
    # 实际实现接入搜索API
    return {"results": [f"关于'{query}'的结果1"]}

@registry.register(
    name="query_sqlite",
    description="查询SQLite数据库",
    parameters={
        "type": "object",
        "properties": {
            "sql": {"type": "string", "description": "SQL查询语句"}
        },
        "required": ["sql"]
    }
)
def query_sqlite(sql: str):
    import sqlite3
    conn = sqlite3.connect("app.db")
    cursor = conn.execute(sql)
    columns = [desc[0] for desc in cursor.description]
    rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
    conn.close()
    return {"columns": columns, "rows": rows}

@registry.register(
    name="get_weather",
    description="获取城市天气信息",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "城市名称"}
        },
        "required": ["city"]
    }
)
def get_weather(city: str):
    # 实际实现调用天气API
    weather_db = {
        "北京": {"temp": 25, "condition": "晴", "humidity": 45},
        "上海": {"temp": 28, "condition": "多云", "humidity": 65},
    }
    return weather_db.get(city, {"error": "城市未找到"})


# 使用工具注册表运行Agent
class ToolAgent:
    """基于Function Calling的Agent"""

    def __init__(self, registry: ToolRegistry):
        self.client = OpenAI()
        self.registry = registry

    def run(self, user_message: str, max_turns: int = 5) -> str:
        messages = [
            {"role": "system", "content": "你是一个有帮助的AI助手，可以使用工具来获取信息。"},
            {"role": "user", "content": user_message},
        ]

        for turn in range(max_turns):
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=self.registry.get_openai_tools(),
                tool_choice="auto",
            )

            msg = response.choices[0].message
            messages.append(msg)

            # 检查是否需要调用工具
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)

                    print(f"调用工具: {tool_name}({tool_args})")
                    result = self.registry.execute(tool_name, tool_args)
                    print(f"工具结果: {result}")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    })
            else:
                # 没有工具调用，返回最终回答
                return msg.content

        return "达到最大轮次限制。"


# 运行
agent = ToolAgent(registry)
answer = agent.run("北京和上海今天的天气怎么样？哪个更适合户外活动？")
print(f"\n回答: {answer}")
```

---

### Day 4: Agent记忆管理

#### 3.4.1 记忆类型

```
Agent记忆层次：

1. 短期记忆 (Short-term Memory)
   - 当前对话的messages列表
   - 存储在内存中
   - 有token长度限制

2. 长期记忆 (Long-term Memory)
   - 跨对话的重要信息
   - 存储在向量数据库中
   - 需要时检索

3. 工作记忆 (Working Memory)
   - 当前任务的状态和中间结果
   - 存储在内存中
```

#### 3.4.2 实现AgentMemory类

```python
import json
import hashlib
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class AgentMemory:
    """Agent记忆管理：短期记忆 + 长期记忆"""

    def __init__(self, max_short_term_tokens=4000, embedding_model="paraphrase-multilingual-MiniLM-L12-v2"):
        # 短期记忆：对话历史
        self.messages: List[Dict] = []
        self.max_tokens = max_short_term_tokens

        # 长期记忆：向量存储
        self.embedder = SentenceTransformer(embedding_model)
        self.dimension = self.embedder.get_sentence_embedding_dimension()
        self.long_term_index = faiss.IndexFlatIP(self.dimension)
        self.long_term_texts: List[str] = []
        self.long_term_metadata: List[Dict] = []

        # 摘要
        self.summary: Optional[str] = None

    # ========== 短期记忆管理 ==========

    def add_message(self, role: str, content: str):
        """添加消息到短期记忆"""
        self.messages.append({"role": role, "content": content})
        # 检查是否超过token限制
        if self._estimate_tokens() > self.max_tokens:
            self._compress_memory()

    def get_messages(self) -> List[Dict]:
        """获取当前对话历史"""
        result = []
        if self.summary:
            result.append({
                "role": "system",
                "content": f"之前对话的摘要：{self.summary}"
            })
        result.extend(self.messages)
        return result

    def clear_short_term(self):
        """清空短期记忆"""
        self.messages = []

    def _estimate_tokens(self) -> int:
        """估算当前消息的token数"""
        total = 0
        for msg in self.messages:
            # 粗略估算：中文1字≈1.5token，英文1词≈1.3token
            total += len(msg["content"]) * 1.5
        return int(total)

    def _compress_memory(self, llm_client=None):
        """压缩记忆：将早期对话摘要化"""
        if len(self.messages) <= 4:
            return

        # 保留最近的几条消息
        old_messages = self.messages[:-4]
        recent_messages = self.messages[-4:]

        if llm_client:
            # 用LLM生成摘要
            old_text = "\n".join([f"{m['role']}: {m['content']}" for m in old_messages])
            summary_prompt = f"请用2-3句话总结以下对话的关键信息：\n{old_text}"
            response = llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0,
            )
            self.summary = response.choices[0].message.content
        else:
            # 简单拼接摘要
            self.summary = "对话摘要: " + " ".join([m["content"][:50] for m in old_messages])

        self.messages = recent_messages

    # ========== 长期记忆管理 ==========

    def save_to_long_term(self, content: str, metadata: Dict = None):
        """保存重要信息到长期记忆"""
        embedding = self.embedder.encode([content], normalize_embeddings=True)
        self.long_term_index.add(embedding.astype('float32'))
        self.long_term_texts.append(content)
        self.long_term_metadata.append(metadata or {})

    def retrieve_from_long_term(self, query: str, top_k: int = 3) -> List[Dict]:
        """从长期记忆中检索相关信息"""
        if self.long_term_index.ntotal == 0:
            return []

        query_embedding = self.embedder.encode([query], normalize_embeddings=True)
        scores, indices = self.long_term_index.search(query_embedding.astype('float32'), top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.long_term_texts):
                results.append({
                    "content": self.long_term_texts[idx],
                    "score": float(scores[0][i]),
                    "metadata": self.long_term_metadata[idx],
                })
        return results

    def auto_save_important(self, message: str, llm_client=None):
        """自动判断并保存重要信息"""
        if llm_client:
            prompt = f"""判断以下对话内容是否包含值得长期记住的重要信息。
例如：用户的偏好、重要事实、关键决策等。

内容：{message}

是否重要(YES/NO)："""
            response = llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            if "YES" in response.choices[0].message.content:
                self.save_to_long_term(message)

    # ========== 工作记忆 ==========

    def get_context_for_prompt(self, current_query: str) -> str:
        """构建包含短期和长期记忆的上下文"""
        context_parts = []

        # 添加长期记忆中的相关信息
        long_term_results = self.retrieve_from_long_term(current_query, top_k=3)
        if long_term_results:
            context_parts.append("相关历史信息：")
            for r in long_term_results:
                context_parts.append(f"- {r['content']}")

        # 添加摘要
        if self.summary:
            context_parts.append(f"\n之前对话摘要：{self.summary}")

        return "\n".join(context_parts)
```

---

### Day 5: LangGraph基础

#### 3.5.1 核心概念

```
LangGraph = 状态图(StateGraph) + 节点(Node) + 边(Edge)

核心概念：
1. State: 用TypedDict定义的状态对象，在整个工作流中传递和更新
2. Node: 处理函数，接收State，返回State更新
3. Edge: 定义节点之间的流转关系
4. Conditional Edge: 条件分支，根据State决定下一步去哪个节点
```

#### 3.5.2 实现简单的多步Agent工作流

```bash
pip install langgraph langchain-openai
```

```python
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import json

# 1. 定义State
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    current_tool: str
    tool_result: str
    iteration: int
    final_answer: str

# 2. 定义工具
def search_web(query: str) -> str:
    return json.dumps({"results": [f"搜索结果: {query}"]})

def calculate(expression: str) -> str:
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"计算错误: {e}"

def get_weather(city: str) -> str:
    return json.dumps({"北京": "晴 25°C", "上海": "多云 28°C"}.get(city, "未知"))

tools = {
    "search_web": {"func": search_web, "desc": "搜索网络信息"},
    "calculate": {"func": calculate, "desc": "数学计算"},
    "get_weather": {"func": get_weather, "desc": "获取天气"},
}

# 3. 定义节点函数
llm = ChatOpenAI(model="gpt-4o")

def think_node(state: AgentState) -> dict:
    """思考节点：决定是否需要调用工具"""
    tool_descriptions = "\n".join([f"- {k}: {v['desc']}" for k, v in tools.items()])

    system_prompt = f"""你是一个AI助手。你可以使用以下工具：
{tool_descriptions}

如果你想使用工具，请回复：
TOOL: 工具名
ARGS: {{"参数名": "参数值"}}

如果你已经有足够信息回答，请回复：
ANSWER: 你的最终回答"""

    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages)

    content = response.content
    update = {"iteration": state.get("iteration", 0) + 1}

    if content.startswith("TOOL:"):
        lines = content.strip().split("\n")
        tool_name = lines[0].replace("TOOL:", "").strip()
        tool_args = json.loads(lines[1].replace("ARGS:", "").strip())
        update["current_tool"] = tool_name
        update["tool_result"] = ""
        # 将工具调用意图添加到消息
        update["messages"] = [response]
        # 保存参数供execute使用
        update["_tool_args"] = tool_args
    elif content.startswith("ANSWER:"):
        answer = content.replace("ANSWER:", "").strip()
        update["final_answer"] = answer
        update["current_tool"] = ""
    else:
        update["final_answer"] = content
        update["current_tool"] = ""

    return update

def execute_node(state: AgentState) -> dict:
    """执行节点：调用工具"""
    tool_name = state.get("current_tool", "")
    if tool_name and tool_name in tools:
        # 获取参数（从上一轮保存的）
        tool_args = state.get("_tool_args", {})
        result = tools[tool_name]["func"](**tool_args)
        return {
            "tool_result": result,
            "messages": [HumanMessage(content=f"工具 {tool_name} 的结果: {result}")]
        }
    return {"tool_result": ""}

# 4. 定义条件边
def should_continue(state: AgentState) -> Literal["execute", "end"]:
    """判断是否需要继续调用工具"""
    if state.get("final_answer"):
        return "end"
    if state.get("current_tool"):
        return "execute"
    return "end"

# 5. 构建工作流图
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("think", think_node)
workflow.add_node("execute", execute_node)

# 设置入口
workflow.set_entry_point("think")

# 添加边
workflow.add_conditional_edges(
    "think",
    should_continue,
    {"execute": "execute", "end": END}
)
workflow.add_edge("execute", "think")  # 执行完回到思考

# 编译
app = workflow.compile()

# 6. 运行
result = app.invoke({
    "messages": [HumanMessage(content="北京今天天气怎么样？")],
    "current_tool": "",
    "tool_result": "",
    "iteration": 0,
    "final_answer": "",
})

print(f"最终回答: {result['final_answer']}")
print(f"迭代次数: {result['iteration']}")
```

---

### Day 6: Agent框架实战

#### 3.6.1 CrewAI实战

```bash
pip install crewai crewai-tools
```

```python
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, tool

# 自定义工具
@tool("数据库查询工具")
def query_database(sql: str) -> str:
    """执行SQL查询并返回结果"""
    import sqlite3
    conn = sqlite3.connect("app.db")
    cursor = conn.execute(sql)
    result = cursor.fetchall()
    conn.close()
    return str(result)

# 定义Agent
researcher = Agent(
    role="研究员",
    goal="搜索和收集相关信息",
    backstory="你是一个经验丰富的研究员，擅长从各种来源搜索和分析信息。",
    tools=[SerperDevTool()],
    verbose=True,
)

analyst = Agent(
    role="数据分析师",
    goal="分析数据并得出结论",
    backstory="你是一个数据专家，擅长SQL查询和数据分析。",
    tools=[query_database],
    verbose=True,
)

writer = Agent(
    role="报告撰写者",
    goal="基于研究结果撰写清晰的报告",
    backstory="你是一个专业写手，擅长将复杂信息整理成易懂的报告。",
    verbose=True,
)

# 定义Task
research_task = Task(
    description="搜索关于{topic}的最新信息和趋势",
    expected_output="包含关键发现和数据的结构化信息摘要",
    agent=researcher,
)

analysis_task = Task(
    description="分析收集到的关于{topic}的数据",
    expected_output="数据分析结果和关键洞察",
    agent=analyst,
)

writing_task = Task(
    description="基于研究和分析结果，撰写关于{topic}的综合报告",
    expected_output="结构清晰的综合报告，包含摘要、分析、结论",
    agent=writer,
)

# 组建Crew
crew = Crew(
    agents=[researcher, analyst, writer],
    tasks=[research_task, analysis_task, writing_task],
    process=Process.sequential,  # 顺序执行
    verbose=True,
)

# 运行
result = crew.kickoff(inputs={"topic": "AI Agent在企业中的应用"})
print(result)
```

#### 3.6.2 Anthropic SDK Tool Use实战

```python
import anthropic
import json

client = anthropic.Anthropic()

# 定义工具
tools = [
    {
        "name": "search",
        "description": "搜索网络信息",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_weather",
        "description": "获取天气信息",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称"}
            },
            "required": ["city"]
        }
    }
]

def run_agent(user_message: str, max_turns: int = 5) -> str:
    """Anthropic SDK多轮工具调用"""
    messages = [{"role": "user", "content": user_message}]

    for turn in range(max_turns):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            tools=tools,
            messages=messages,
        )

        # 检查是否需要调用工具
        if response.stop_reason == "tool_use":
            # 添加助手消息（包含工具调用）
            messages.append({"role": "assistant", "content": response.content})

            # 处理所有工具调用
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"调用工具: {block.name}({block.input})")

                    # 执行工具
                    if block.name == "search":
                        result = {"results": [f"搜索: {block.input['query']}"]}
                    elif block.name == "get_weather":
                        result = {"北京": "晴25°C", "上海": "多云28°C"}.get(
                            block.input['city'], "未知"
                        )
                    else:
                        result = {"error": "Unknown tool"}

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    })

            # 添加工具结果
            messages.append({"role": "user", "content": tool_results})
        else:
            # 返回最终文本回答
            return response.content[0].text

    return "达到最大轮次限制。"

# 运行
answer = run_agent("帮我查一下北京和上海的天气，然后比较哪个城市更适合今天户外运动。")
print(f"\n回答: {answer}")
```

---

### Day 7: 实战项目10 - AI智能助手

#### 项目功能

```
AI智能助手功能清单：
1. 搜索网页（DuckDuckGo搜索）
2. 查询数据库（SQLite）
3. 调用API（天气API）
4. 检索文档（RAG集成）
5. 记忆管理（短期+长期）
```

#### 项目架构

```
smart_assistant/
├── main.py               # 主入口
├── agent/
│   ├── __init__.py
│   ├── core.py           # Agent核心逻辑
│   └── memory.py         # 记忆管理
├── tools/
│   ├── __init__.py
│   ├── registry.py       # 工具注册框架
│   ├── search.py         # DuckDuckGo搜索
│   ├── database.py       # SQLite查询
│   ├── weather.py        # 天气API
│   └── rag.py            # RAG检索
├── data/
│   ├── documents/        # RAG文档
│   └── app.db            # SQLite数据库
└── requirements.txt
```

#### 核心实现

```python
# main.py
from agent.core import SmartAssistant
from tools.registry import ToolRegistry
from tools.search import setup_search_tool
from tools.database import setup_database_tool
from tools.weather import setup_weather_tool
from tools.rag import setup_rag_tool

def main():
    # 注册所有工具
    registry = ToolRegistry()
    setup_search_tool(registry)
    setup_database_tool(registry)
    setup_weather_tool(registry)
    setup_rag_tool(registry)

    # 创建智能助手
    assistant = SmartAssistant(registry)

    print("AI智能助手已启动（输入'quit'退出）")
    print("可用工具:", ", ".join(registry.get_tool_names()))

    while True:
        user_input = input("\n你: ").strip()
        if user_input.lower() == 'quit':
            break
        if not user_input:
            continue

        response = assistant.run(user_input)
        print(f"\n助手: {response}")

if __name__ == "__main__":
    main()
```

```python
# tools/search.py - DuckDuckGo搜索工具
from duckduckgo_search import DDGS

def setup_search_tool(registry):
    @registry.register(
        name="search_web",
        description="搜索互联网获取最新信息",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"}
            },
            "required": ["query"]
        }
    )
    def search_web(query: str):
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        return [{"title": r["title"], "body": r["body"][:200]} for r in results]
```

```python
# tools/weather.py - 天气API工具
import requests

def setup_weather_tool(registry):
    @registry.register(
        name="get_weather",
        description="获取指定城市的天气信息",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称"}
            },
            "required": ["city"]
        }
    )
    def get_weather(city: str):
        # 使用wttr.in免费天气API
        try:
            response = requests.get(
                f"https://wttr.in/{city}?format=j1",
                timeout=10
            )
            data = response.json()
            current = data["current_condition"][0]
            return {
                "temp": current["temp_C"] + "°C",
                "condition": current["weatherDesc"][0]["value"],
                "humidity": current["humidity"] + "%",
                "wind": current["windspeedKmph"] + "km/h",
            }
        except Exception as e:
            return {"error": str(e)}
```

```python
# tools/rag.py - RAG检索工具
def setup_rag_tool(registry):
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np

    # 初始化RAG组件
    embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    index = faiss.IndexFlatIP(embedder.get_sentence_embedding_dimension())
    doc_texts = []

    # 索引文档（启动时加载）
    import os
    doc_dir = "./data/documents"
    if os.path.exists(doc_dir):
        for filename in os.listdir(doc_dir):
            filepath = os.path.join(doc_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
                # 简单切分
                chunks = [text[i:i+500] for i in range(0, len(text), 450)]
                for chunk in chunks:
                    embedding = embedder.encode([chunk], normalize_embeddings=True)
                    index.add(embedding.astype('float32'))
                    doc_texts.append(chunk)

    @registry.register(
        name="search_documents",
        description="从本地文档库中检索相关信息",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索关键词"}
            },
            "required": ["query"]
        }
    )
    def search_documents(query: str):
        if index.ntotal == 0:
            return {"error": "文档库为空"}

        query_embedding = embedder.encode([query], normalize_embeddings=True)
        scores, indices = index.search(query_embedding.astype('float32'), 3)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(doc_texts):
                results.append({
                    "content": doc_texts[idx][:200],
                    "score": float(scores[0][i]),
                })
        return results
```

---

## 四、代码练习

### Day 2 练习：手动实现ReAct循环

```
任务：
1. 不使用任何框架，从零实现SimpleReActAgent
2. 实现3个工具（搜索/计算器/天气）
3. 测试5个需要多步推理的问题
4. 输出：react_agent.py
```

### Day 3 练习：实现通用工具注册和调用框架

```
任务：
1. 实现ToolRegistry类（注册/调用/OpenAI格式输出）
2. 注册至少5个工具
3. 实现ToolAgent类（自动工具调用循环）
4. 输出：tool_framework.py
```

### Day 5 练习：用LangGraph实现3步工作流

```
任务：
1. 定义State和3个节点（分析→检索→生成）
2. 实现条件边
3. 测试3种不同类型的查询
4. 输出：langgraph_workflow.py
```

### Day 7 练习：完成项目10智能助手

```
任务：
1. 完成所有工具实现（搜索/数据库/天气/RAG）
2. 集成记忆管理
3. 实现交互式对话界面
4. 测试至少10个不同类型的查询
5. 输出：smart_assistant/ 完整项目
```

---

## 五、本周产出

| 产出物 | 说明 | 完成标准 |
|--------|------|----------|
| react_agent.py | ReAct Agent实现 | 手动推理循环 + 3个工具 |
| tool_framework.py | 通用工具框架 | 注册/调用/Agent循环 |
| langgraph_workflow.py | LangGraph工作流 | 3节点+条件边 |
| smart_assistant/ | 智能助手项目 | 4个工具+记忆+交互界面 |

---

## 六、自测题

### 题目

1. **Agent和Chatbot的核心区别？**

<details>
<summary>参考答案</summary>

Chatbot是被动式的问答系统：用户提问→LLM生成文本回答→结束。它只能输出文字，不能执行任何实际操作。Agent是主动式的自主系统：用户提出目标→LLM思考需要做什么→选择并调用工具→观察执行结果→继续思考决策→直到达成目标。核心区别在于Agent具有三个能力Chatbot没有的：(1) 工具使用——能调用外部API、数据库、搜索引擎等；(2) 自主决策——能根据中间结果灵活调整策略；(3) 多步执行——不是一次性回答，而是多轮"思考-行动-观察"循环。
</details>

2. **ReAct模式的推理流程？**

<details>
<summary>参考答案</summary>

ReAct（Reasoning + Acting）交替执行思考和行动：(1) Thought：LLM分析当前状态和问题，推理下一步需要做什么；(2) Action：LLM选择一个工具并提供参数，执行工具获得结果；(3) Observation：将工具执行结果反馈给LLM；(4) 重复Thought→Action→Observation循环，直到LLM认为信息足够；(5) Answer：LLM基于所有收集到的信息给出最终回答。每个步骤都有明确的文本标记（Thought/Action/Observation），使推理过程可追溯。
</details>

3. **Function Calling的完整流程？**

<details>
<summary>参考答案</summary>

(1) 开发者定义工具的JSON Schema（名称、描述、参数类型）；(2) 用户提问，将问题+工具定义一起发送给LLM；(3) LLM判断是否需要调用工具——如果需要，输出工具名称和参数（JSON格式）；(4) 开发者的代码解析LLM输出，调用实际的工具函数；(5) 将工具执行结果按格式返回给LLM；(6) LLM根据工具结果继续推理，可能再次调用工具或生成最终回答。整个过程是LLM和开发者代码的协作循环。
</details>

4. **Agent记忆有哪些类型？**

<details>
<summary>参考答案</summary>

三种记忆类型：(1) 短期记忆（对话历史）：存储当前对话的所有消息，即messages列表，有token长度限制，超出时需要压缩（摘要化）或截断；(2) 长期记忆（向量存储）：将跨对话的重要信息存入向量数据库，需要时通过语义检索取回。例如用户的偏好、之前讨论的关键决策等；(3) 工作记忆（任务状态）：存储当前任务的中间结果和状态，如已经完成的步骤、待处理的数据等。类似人类的"工作记忆"，用于复杂多步骤任务的追踪。
</details>

5. **LangGraph的StateGraph如何工作？**

<details>
<summary>参考答案</summary>

StateGraph是LangGraph的核心抽象：(1) 用TypedDict定义State数据结构，包含工作流中需要传递的所有信息；(2) 每个Node是一个处理函数，接收当前State，返回State的部分更新；(3) Edge定义Node之间的固定流转（A→B）；(4) Conditional Edge根据State的值动态决定下一个Node（类似if-else）；(5) 工作流从entry_point开始，沿Edge/Conditional Edge流转，直到到达END节点。本质是一个有限状态机（FSM），State在每个节点被更新和传递。
</details>

---

## 七、Java开发者提示

### Agent类比

```
Agent ≈ Java中的命令模式 + 策略模式

// Java命令模式
interface Command {
    Result execute(Map<String, Object> args);
}

class SearchCommand implements Command {
    public Result execute(Map<String, Object> args) {
        return searchService.search((String) args.get("query"));
    }
}

// Agent = LLM(决策器) + 工具(Command集合)
// LLM扮演"智能路由"的角色，决定调用哪个Command
```

### ReAct类比

```
ReAct循环 ≈ Java中的while循环处理链

// Java中的多步处理
while (!task.isComplete()) {
    Decision decision = analyzer.decide(task);  // Thought
    Result result = executor.execute(decision); // Action
    task.update(result);                        // Observation
}

// ReAct
while not has_answer:
    thought = llm.think(state)       # Thought
    result = tool.execute(action)    # Action
    state.update(result)             # Observation
```

### 工具注册类比

```
ToolRegistry ≈ Java中的Spring IoC容器

// Spring：通过注解注册Bean
@Service
public class SearchService { ... }

// ToolRegistry：通过装饰器注册工具
@registry.register(name="search", ...)
def search(query): ...

// Spring：通过@Autowired注入
@Autowired private SearchService searchService;

// ToolRegistry：通过名称查找和执行
registry.execute("search", {"query": "AI"})
```

### LangGraph类比

```
StateGraph ≈ Java中的状态机（Spring StateMachine）

// Spring StateMachine
builder.configureStates()
    .withStates().initial("THINK").states("EXECUTE", "END");
builder.configureTransitions()
    .withExternal().source("THINK").target("EXECUTE").guard(needsTool())
    .withExternal().source("EXECUTE").target("THINK")
    .withExternal().source("THINK").target("END").guard(hasAnswer());

// LangGraph
workflow.add_node("think", think_fn)
workflow.add_node("execute", execute_fn)
workflow.add_conditional_edges("think", route, {"execute": "execute", "end": END})
workflow.add_edge("execute", "think")

概念完全对应：状态 → 节点，转换 → 边，条件转换 → 条件边
```

### 记忆管理类比

```
Agent记忆 ≈ Java Web应用的会话管理

短期记忆（对话历史）:
  Java: HttpSession存储当前会话数据
  Agent: messages列表存储当前对话

长期记忆（向量数据库）:
  Java: MySQL/Redis存储跨会话用户数据
  Agent: FAISS/Chroma存储跨对话重要信息

记忆压缩（摘要）:
  Java: session超时清理或数据压缩
  Agent: 用LLM摘要化早期对话，节省token
```
