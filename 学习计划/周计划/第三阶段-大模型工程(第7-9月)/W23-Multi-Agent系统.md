# W23 - Multi-Agent系统

> 第23周学习计划 | Java开发工程师转AI开发 | 工作日每晚2小时 + 周末6-8小时

---

## 一、本周目标

1. 掌握Multi-Agent的4种架构模式（主从/对等/层级/路由）及设计原则
2. 理解Agent间通信机制（消息传递/共享状态/任务分配）
3. 掌握LangGraph进阶功能（子图/Human-in-the-loop/并行/错误恢复）
4. 理解规划与反思机制（Plan-and-Execute/Self-Reflection/Reflexion）
5. 完成Multi-Agent系统实战项目（项目11）

---

## 二、时间安排

### 工作日（周一至周五，每晚2小时）

| 日期 | 主题 | 时间分配 |
|------|------|----------|
| Day 1 (周一) | Multi-Agent架构 | 理论90分钟 + 设计练习30分钟 |
| Day 2 (周二) | Agent间通信 | 理论60分钟 + 实现通信框架60分钟 |
| Day 3 (周三) | LangGraph进阶 | 学习60分钟 + 代码练习60分钟 |
| Day 4 (周四) | 规划与反思 | 理论60分钟 + 实现反思循环60分钟 |
| Day 5 (周五) | 代码生成Agent | 学习60分钟 + 实现CodeAgent 60分钟 |

### 周末（周六至周日，每天6-8小时）

| 日期 | 主题 | 时间分配 |
|------|------|----------|
| Day 6 (周六) | Multi-Agent系统设计 | 上午设计系统(3h) + 下午定义Agent(3h) + 练习(1h) |
| Day 7 (周日) | 实战项目11 - Multi-Agent系统 | 全天实现项目(7-8h) |

---

## 三、详细学习内容

### Day 1: Multi-Agent架构

#### 3.1.1 主从模式 (Manager-Worker)

```
用户请求
    ↓
┌──────────────┐
│ Manager Agent │ ← 负责任务分解和分配
└──────┬───────┘
       │ 分配任务
  ┌────┼────┐
  ↓    ↓    ↓
Worker1 Worker2 Worker3  ← 各自执行具体任务
  ↓    ↓    ↓
  └────┼────┘
       ↓
┌──────────────┐
│ Manager Agent │ ← 汇总结果
└──────────────┘
       ↓
   最终输出
```

```python
class ManagerWorkerPattern:
    """主从模式"""

    def __init__(self, llm_client):
        self.llm = llm_client
        self.workers = {}

    def register_worker(self, name, agent):
        self.workers[name] = agent

    def run(self, task):
        # Step 1: Manager分解任务
        decomposition = self._decompose_task(task)

        # Step 2: 分配给Worker执行
        results = {}
        for subtask in decomposition["subtasks"]:
            worker_name = subtask["assigned_to"]
            if worker_name in self.workers:
                result = self.workers[worker_name].execute(subtask["task"])
                results[worker_name] = result

        # Step 3: Manager汇总
        final_result = self._synthesize(task, results)
        return final_result

    def _decompose_task(self, task):
        prompt = f"""将以下任务分解为子任务，并指定负责的Agent。

可用Agent: {list(self.workers.keys())}

任务：{task}

以JSON格式输出子任务列表，每个子任务包含：
- task: 子任务描述
- assigned_to: 负责的Agent名称"""
        response = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        import json
        return json.loads(response.choices[0].message.content)
```

#### 3.1.2 对等模式 (Peer-to-Peer)

```
Agent A ←→ Agent B
    ↕           ↕
Agent C ←→ Agent D

Agent之间直接通信，没有中心控制节点。
适合：协作讨论、互相审查、轮流改进。
```

#### 3.1.3 层级模式 (Hierarchical)

```
        CEO Agent
       /         \
  Manager A    Manager B
   /    \       /    \
Worker1 Worker2 Worker3 Worker4

适合：复杂任务的多级分解，每层有不同的管理粒度。
```

#### 3.1.4 路由模式 (Router)

```
用户请求
    ↓
Router Agent（判断请求类型）
    ↓
┌─────┼─────┐
↓     ↓     ↓
编程   翻译   分析
Agent  Agent  Agent
```

```python
class RouterPattern:
    """路由模式：根据请求类型分发到专家Agent"""

    def __init__(self, llm_client):
        self.llm = llm_client
        self.experts = {}

    def register_expert(self, name, agent, expertise):
        self.experts[name] = {"agent": agent, "expertise": expertise}

    def run(self, user_request):
        # Step 1: 路由判断
        expert_name = self._route(user_request)

        # Step 2: 分发给专家Agent
        if expert_name in self.experts:
            return self.experts[expert_name]["agent"].execute(user_request)
        return "无法处理此类请求"

    def _route(self, request):
        expert_list = "\n".join([
            f"- {name}: {info['expertise']}"
            for name, info in self.experts.items()
        ])
        prompt = f"""判断以下请求应该由哪个专家处理。

专家列表：
{expert_list}

请求：{request}

请只返回专家名称。"""
        response = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip()
```

#### 3.1.5 设计原则

```
Multi-Agent设计原则：

1. 单一职责：每个Agent只负责一个明确的任务
   好的做法：SearchAgent只负责搜索，AnalysisAgent只负责分析
   坏的做法：一个Agent既搜索又分析又写作

2. 接口清晰：Agent之间的输入输出格式要明确
   定义标准的Message格式
   使用TypedDict或dataclass约束数据结构

3. 错误隔离：一个Agent失败不应导致整个系统崩溃
   每个Agent有try-except包裹
   失败时有降级策略

4. 可观测性：能够追踪每个Agent的执行过程
   记录每步的输入输出
   使用日志或链式追踪

5. 幂等性：同一输入应该产生一致的输出
   Agent执行应该是确定性的（控制temperature）
```

---

### Day 2: Agent间通信

#### 3.2.1 消息传递

```python
from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime
from enum import Enum

class MessageType(Enum):
    TASK_ASSIGN = "task_assign"       # 分配任务
    TASK_RESULT = "task_result"       # 任务结果
    QUERY = "query"                   # 查询
    RESPONSE = "response"             # 响应
    BROADCAST = "broadcast"           # 广播
    ERROR = "error"                   # 错误

@dataclass
class Message:
    """Agent间通信的消息类"""
    sender: str                       # 发送者ID
    receiver: str                     # 接收者ID
    msg_type: MessageType             # 消息类型
    content: Any                      # 消息内容
    metadata: dict = field(default_factory=dict)  # 元数据
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    reply_to: Optional[str] = None    # 回复的消息ID
    msg_id: str = field(default_factory=lambda: str(id(object())))


class AgentCommunicator:
    """Agent间通信管理器"""

    def __init__(self):
        self.mailboxes: dict[str, list[Message]] = {}  # 每个Agent的消息队列
        self.history: list[Message] = []                 # 消息历史

    def register_agent(self, agent_id: str):
        """注册Agent"""
        self.mailboxes[agent_id] = []

    def send(self, message: Message):
        """发送消息"""
        if message.receiver not in self.mailboxes:
            raise ValueError(f"未知接收者: {message.receiver}")
        self.mailboxes[message.receiver].append(message)
        self.history.append(message)

    def receive(self, agent_id: str) -> list[Message]:
        """接收消息（清空邮箱）"""
        messages = self.mailboxes.get(agent_id, [])
        self.mailboxes[agent_id] = []
        return messages

    def broadcast(self, sender: str, content: Any):
        """广播消息给所有Agent"""
        for agent_id in self.mailboxes:
            if agent_id != sender:
                msg = Message(
                    sender=sender,
                    receiver=agent_id,
                    msg_type=MessageType.BROADCAST,
                    content=content,
                )
                self.mailboxes[agent_id].append(msg)
                self.history.append(msg)

    def get_history(self, agent_id: str = None) -> list[Message]:
        """获取消息历史"""
        if agent_id:
            return [m for m in self.history if m.sender == agent_id or m.receiver == agent_id]
        return self.history
```

#### 3.2.2 共享状态

```python
from typing import TypedDict, List, Optional
import copy

class SharedState(TypedDict):
    """多个Agent共享的状态"""
    task: str
    subtasks: List[dict]
    results: dict
    status: str
    errors: List[str]

class StateManager:
    """共享状态管理器"""

    def __init__(self, initial_state: SharedState):
        self.state = initial_state
        self.listeners = []

    def get_state(self) -> SharedState:
        """获取当前状态（深拷贝）"""
        return copy.deepcopy(self.state)

    def update_state(self, updates: dict, updated_by: str):
        """更新状态"""
        self.state.update(updates)
        # 通知监听者
        for listener in self.listeners:
            listener(self.state, updated_by)

    def register_listener(self, listener):
        """注册状态变更监听器"""
        self.listeners.append(listener)
```

#### 3.2.3 任务分配

```python
class TaskDistributor:
    """Manager Agent的任务分配器"""

    def __init__(self, llm_client, agents_info: dict):
        self.llm = llm_client
        self.agents_info = agents_info  # {name: {capabilities, current_load}}

    def distribute(self, task: str) -> list[dict]:
        """智能任务分配"""
        agents_desc = "\n".join([
            f"- {name}: 能力={info['capabilities']}, 当前负载={info['current_load']}"
            for name, info in self.agents_info.items()
        ])

        prompt = f"""将以下任务分配给最合适的Agent。

任务：{task}

可用Agent：
{agents_desc}

请输出任务分配方案（JSON格式）：
[{{"subtask": "子任务描述", "assigned_to": "Agent名称"}}]"""

        response = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        import json
        return json.loads(response.choices[0].message.content)
```

---

### Day 3: LangGraph进阶

#### 3.3.1 子图 Subgraph

```python
from typing import TypedDict
from langgraph.graph import StateGraph, END

# 定义子图：文档处理子流程
class DocProcessState(TypedDict):
    documents: list
    processed_documents: list
    errors: list

def load_docs(state: DocProcessState) -> dict:
    """加载文档"""
    docs = state["documents"]
    return {"processed_documents": []}

def parse_docs(state: DocProcessState) -> dict:
    """解析文档"""
    processed = []
    for doc in state["documents"]:
        processed.append({"content": doc, "parsed": True})
    return {"processed_documents": processed}

def validate_docs(state: DocProcessState) -> dict:
    """验证文档"""
    errors = []
    for doc in state["processed_documents"]:
        if not doc.get("content"):
            errors.append("空文档")
    return {"errors": errors}

# 构建子图
doc_workflow = StateGraph(DocProcessState)
doc_workflow.add_node("load", load_docs)
doc_workflow.add_node("parse", parse_docs)
doc_workflow.add_node("validate", validate_docs)
doc_workflow.set_entry_point("load")
doc_workflow.add_edge("load", "parse")
doc_workflow.add_edge("parse", "validate")
doc_workflow.add_edge("validate", END)
doc_subgraph = doc_workflow.compile()

# 在主图中使用子图
class MainState(TypedDict):
    user_request: str
    documents: list
    analysis_result: str
    report: str

def doc_process_node(state: MainState) -> dict:
    """主图中的文档处理节点，调用子图"""
    sub_result = doc_subgraph.invoke({
        "documents": state["documents"],
        "processed_documents": [],
        "errors": [],
    })
    return {"documents": sub_result["processed_documents"]}

def analyze_node(state: MainState) -> dict:
    return {"analysis_result": "分析完成"}

def report_node(state: MainState) -> dict:
    return {"report": "报告生成完成"}

main_workflow = StateGraph(MainState)
main_workflow.add_node("process_docs", doc_process_node)
main_workflow.add_node("analyze", analyze_node)
main_workflow.add_node("report", report_node)
main_workflow.set_entry_point("process_docs")
main_workflow.add_edge("process_docs", "analyze")
main_workflow.add_edge("analyze", "report")
main_workflow.add_edge("report", END)
main_app = main_workflow.compile()
```

#### 3.3.2 Human-in-the-loop

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

class ReviewState(TypedDict):
    user_request: str
    draft: str
    human_feedback: str
    final_output: str
    revision_count: int

def generate_draft(state: ReviewState) -> dict:
    """生成初稿"""
    return {"draft": "这是初稿内容...", "revision_count": 0}

def human_review(state: ReviewState) -> dict:
    """人工审查节点（会被中断）"""
    # 这个函数在interrupt之前执行
    # 实际的审查由人工完成
    return {}

def apply_feedback(state: ReviewState) -> dict:
    """根据人工反馈修改"""
    return {
        "final_output": f"修改后的内容（基于反馈: {state.get('human_feedback', '')}）",
        "revision_count": state.get("revision_count", 0) + 1,
    }

def should_continue(state: ReviewState) -> str:
    """判断是否还需要修改"""
    if state.get("revision_count", 0) >= 3:
        return "end"
    if state.get("human_feedback"):
        return "revise"
    return "end"

# 构建带Human-in-the-loop的工作流
workflow = StateGraph(ReviewState)
workflow.add_node("generate", generate_draft)
workflow.add_node("review", human_review)
workflow.add_node("revise", apply_feedback)

workflow.set_entry_point("generate")
workflow.add_edge("generate", "review")
# review节点之前中断，等待人工输入
workflow.add_edge("review", "revise")
workflow.add_conditional_edges("revise", should_continue, {"revise": "review", "end": END})

# 使用checkpointer支持中断和恢复
checkpointer = MemorySaver()
app = workflow.compile(
    checkpointer=checkpointer,
    interrupt_before=["review"],  # 在review节点之前中断
)

# 运行
config = {"configurable": {"thread_id": "thread-1"}}
result = app.invoke({"user_request": "写一篇关于AI的文章"}, config)

# 此时被中断，获取当前状态
state = app.get_state(config)
print(f"当前初稿: {state.values.get('draft')}")

# 人工输入反馈
app.update_state(config, {"human_feedback": "请增加更多技术细节"}, as_node="review")

# 继续执行
result = app.invoke(None, config)
```

#### 3.3.3 并行节点

```python
from langgraph.graph import StateGraph, END
import asyncio

class ParallelState(TypedDict):
    query: str
    web_results: list
    db_results: list
    doc_results: list
    combined_results: list

def search_web(state: ParallelState) -> dict:
    """搜索网络（并行执行）"""
    return {"web_results": [f"网络搜索结果: {state['query']}"]}

def search_db(state: ParallelState) -> dict:
    """搜索数据库（并行执行）"""
    return {"db_results": [f"数据库查询结果: {state['query']}"]}

def search_docs(state: ParallelState) -> dict:
    """搜索文档（并行执行）"""
    return {"doc_results": [f"文档检索结果: {state['query']}"]}

def merge_results(state: ParallelState) -> dict:
    """合并所有结果"""
    combined = (
        state.get("web_results", []) +
        state.get("db_results", []) +
        state.get("doc_results", [])
    )
    return {"combined_results": combined}

# 构建并行工作流
workflow = StateGraph(ParallelState)
workflow.add_node("search_web", search_web)
workflow.add_node("search_db", search_db)
workflow.add_node("search_docs", search_docs)
workflow.add_node("merge", merge_results)

workflow.set_entry_point("search_web")  # 入口（3个并行节点）
# 三个节点都连接到merge，LangGraph会自动并行执行
workflow.add_edge("search_web", "merge")
workflow.add_edge("search_db", "merge")
workflow.add_edge("search_docs", "merge")
workflow.add_edge("merge", END)

app = workflow.compile()
result = app.invoke({"query": "AI Agent", "web_results": [], "db_results": [], "doc_results": [], "combined_results": []})
```

#### 3.3.4 错误恢复

```python
from langgraph.checkpoint.memory import MemorySaver

class ResilientState(TypedDict):
    task: str
    step_results: list
    current_step: int
    errors: list

def step_with_retry(state: ResilientState) -> dict:
    """带重试的步骤执行"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # 执行步骤
            result = f"步骤{state['current_step']}完成"
            new_results = state.get("step_results", []) + [result]
            return {
                "step_results": new_results,
                "current_step": state["current_step"] + 1,
            }
        except Exception as e:
            if attempt == max_retries - 1:
                new_errors = state.get("errors", []) + [str(e)]
                return {"errors": new_errors}
            continue

# 使用MemorySaver实现状态快照
checkpointer = MemorySaver()

# 从断点恢复
def resume_from_checkpoint(thread_id: str, app):
    """从checkpointer恢复执行"""
    config = {"configurable": {"thread_id": thread_id}}
    state = app.get_state(config)

    if state.next:  # 还有未执行的节点
        print(f"从断点恢复，待执行节点: {state.next}")
        result = app.invoke(None, config)
        return result
    else:
        print("已完成")
        return state.values
```

---

### Day 4: 规划与反思

#### 3.4.1 Plan-and-Execute

```
Plan-and-Execute流程：

用户任务 → Planner Agent制定计划
              ↓
         [步骤1] → Executor Agent执行 → 结果
              ↓
         [步骤2] → Executor Agent执行 → 结果
              ↓
         [步骤3] → Executor Agent执行 → 结果
              ↓
         汇总所有结果 → 最终输出

与ReAct的区别：
  ReAct: 逐步思考，每步决定下一步做什么（动态）
  Plan-and-Execute: 先制定完整计划，再逐步执行（静态）
  Plan-and-Execute适合：步骤明确、可以提前规划的任务
  ReAct适合：需要根据中间结果灵活调整的任务
```

```python
class PlanAndExecute:
    """Plan-and-Execute模式"""

    def __init__(self, llm_client, executor_agent):
        self.llm = llm_client
        self.executor = executor_agent

    def run(self, task: str) -> str:
        # Step 1: 制定计划
        plan = self._create_plan(task)
        print(f"执行计划: {plan}")

        # Step 2: 逐步执行
        results = []
        for i, step in enumerate(plan):
            print(f"\n执行步骤 {i+1}/{len(plan)}: {step}")
            result = self.executor.execute(step)
            results.append(result)
            print(f"结果: {result}")

        # Step 3: 汇总
        final = self._synthesize(task, plan, results)
        return final

    def _create_plan(self, task: str) -> list:
        prompt = f"""为以下任务制定执行计划，分解为具体步骤。

任务：{task}

请输出步骤列表（JSON数组）：
["步骤1", "步骤2", "步骤3", ...]"""

        response = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        import json
        return json.loads(response.choices[0].message.content)

    def _synthesize(self, task, plan, results):
        prompt = f"""基于以下执行结果，汇总最终答案。

任务：{task}
计划：{plan}
执行结果：{results}

请给出完整的最终输出。"""
        response = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
```

#### 3.4.2 Self-Reflection / Reflexion

```python
class ReflexionAgent:
    """反思循环：执行→评估→反思→改进→重新执行"""

    def __init__(self, llm_client, executor, evaluator, max_attempts=3):
        self.llm = llm_client
        self.executor = executor
        self.evaluator = evaluator
        self.max_attempts = max_attempts

    def run(self, task: str) -> str:
        reflections = []

        for attempt in range(self.max_attempts):
            print(f"\n=== 第{attempt+1}次尝试 ===")

            # Step 1: 执行
            result = self.executor.execute(task, reflections=reflections)
            print(f"执行结果: {result[:100]}...")

            # Step 2: 评估
            evaluation = self.evaluator.evaluate(task, result)
            print(f"评估: {evaluation}")

            if evaluation.get("passed", False):
                return result

            # Step 3: 反思
            reflection = self._reflect(task, result, evaluation)
            reflections.append(reflection)
            print(f"反思: {reflection[:100]}...")

        return result  # 返回最后一次的结果

    def _reflect(self, task, result, evaluation):
        prompt = f"""任务执行失败，请分析原因并提出改进策略。

任务：{task}
执行结果：{result}
评估反馈：{evaluation}

请分析：
1. 失败的原因是什么？
2. 下次执行应该如何改进？
3. 具体的改进策略是什么？"""

        response = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
```

---

### Day 5: 代码生成Agent

#### 3.5.1 代码生成→执行→验证→修复闭环

```
CodeAgent流程：

1. LLM生成代码
      ↓
2. 沙箱执行(subprocess)
      ↓
3. 检查执行结果/测试
      ↓ 成功 → 返回结果
      ↓ 失败
4. LLM分析错误原因
      ↓
5. LLM修复代码
      ↓
6. 重新执行 → 重复2-5直到成功或达到重试上限
```

```python
import subprocess
import tempfile
import os

class CodeAgent:
    """代码生成Agent：生成→执行→验证→修复"""

    def __init__(self, llm_client, max_retries=3):
        self.llm = llm_client
        self.max_retries = max_retries

    def run(self, task: str, language: str = "python") -> dict:
        """完成代码生成任务"""
        code = None
        errors_history = []

        for attempt in range(self.max_retries):
            print(f"\n=== 第{attempt+1}次尝试 ===")

            # Step 1: 生成代码
            code = self._generate_code(task, errors_history, language)
            print(f"生成的代码:\n{code}")

            # Step 2: 执行代码
            exec_result = self._execute_code(code, language)
            print(f"执行结果: {exec_result}")

            if exec_result["success"]:
                return {
                    "success": True,
                    "code": code,
                    "output": exec_result["output"],
                    "attempts": attempt + 1,
                }

            # Step 3: 记录错误
            errors_history.append({
                "code": code,
                "error": exec_result["error"],
                "attempt": attempt + 1,
            })

        return {
            "success": False,
            "code": code,
            "error": errors_history[-1]["error"] if errors_history else "Unknown error",
            "attempts": self.max_retries,
        }

    def _generate_code(self, task, errors_history, language):
        """生成代码（如果有历史错误，附带修复指令）"""
        if errors_history:
            # 修复模式
            last_error = errors_history[-1]
            prompt = f"""之前的代码执行出错，请修复。

任务：{task}

之前的代码：
```
{last_error['code']}
```

错误信息：
{last_error['error']}

请分析错误原因并修复代码。只输出修复后的完整代码，不要解释。"""
        else:
            # 首次生成
            prompt = f"""请编写{language}代码完成以下任务。

任务：{task}

要求：
1. 只输出代码，不要解释
2. 代码要包含错误处理
3. 代码要能独立运行"""

        response = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return self._extract_code(response.choices[0].message.content)

    def _execute_code(self, code, language="python"):
        """在沙箱中执行代码"""
        try:
            with tempfile.NamedTemporaryFile(
                mode='w', suffix=f'.{language == "python" and "py" or "js"}',
                delete=False, encoding='utf-8'
            ) as f:
                f.write(code)
                temp_path = f.name

            # 执行代码（带超时）
            result = subprocess.run(
                [language, temp_path],
                capture_output=True,
                text=True,
                timeout=30,
            )

            os.unlink(temp_path)

            if result.returncode == 0:
                return {"success": True, "output": result.stdout}
            else:
                return {"success": False, "error": result.stderr}

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "执行超时（30秒）"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _extract_code(self, text):
        """从LLM输出中提取代码块"""
        import re
        # 尝试提取```code```块
        pattern = r"```(?:python)?\s*\n(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # 如果没有代码块标记，返回全部文本
        return text.strip()
```

---

### Day 6: Multi-Agent系统设计

#### 3.6.1 研究报告自动生成系统

```
系统架构：

用户请求（研究主题）
        ↓
┌───────────────────────────────────┐
│        Orchestrator (LangGraph)    │
│                                    │
│  ┌──────────┐                     │
│  │Researcher │ ← 搜索工具 + RAG    │
│  │  Agent    │                     │
│  └─────┬────┘                      │
│        ↓                           │
│  ┌──────────┐                     │
│  │ Writer   │ ← 写作工具           │
│  │  Agent   │                     │
│  └─────┬────┘                      │
│        ↓                           │
│  ┌──────────┐                     │
│  │Reviewer  │ ← 评估工具           │
│  │  Agent   │                     │
│  └─────┬────┘                      │
│        ↓  (如果不合格)              │
│  ┌──────────┐                     │
│  │ Writer   │ ← 根据反馈修改       │
│  │  Agent   │                     │
│  └─────┬────┘                      │
│        ↓  (合格)                    │
│    最终报告                         │
└───────────────────────────────────┘
```

#### 3.6.2 定义Agent角色

```python
# Researcher Agent
researcher_config = {
    "name": "researcher",
    "role": "信息研究员",
    "goal": "收集和整理与研究主题相关的信息",
    "backstory": "你是一个经验丰富的研究员，擅长从多种来源搜索和整理信息。",
    "tools": ["search_web", "search_documents"],
    "output_format": "结构化的研究发现列表",
}

# Writer Agent
writer_config = {
    "name": "writer",
    "role": "报告撰写者",
    "goal": "基于研究发现撰写高质量的研究报告",
    "backstory": "你是一个专业写手，擅长将研究信息整理成结构清晰的报告。",
    "tools": [],
    "output_format": "Markdown格式的完整报告",
}

# Reviewer Agent
reviewer_config = {
    "name": "reviewer",
    "role": "质量审核员",
    "goal": "审核报告质量并提出改进建议",
    "backstory": "你是一个严格的审稿人，注重准确性、完整性和可读性。",
    "tools": [],
    "output_format": "审核意见 + 评分 + 具体修改建议",
}
```

---

### Day 7: 实战项目11 - Multi-Agent系统

#### 完整实现

```python
"""
项目11：研究报告自动生成系统
使用LangGraph编排Multi-Agent工作流
"""
from typing import TypedDict, List, Literal
from langgraph.graph import StateGraph, END
from openai import OpenAI

# 1. 定义状态
class ResearchState(TypedDict):
    topic: str
    research_findings: List[str]
    draft_report: str
    review_comments: str
    review_score: float
    final_report: str
    revision_count: int

# 2. 初始化LLM
client = OpenAI()

# 3. 定义节点函数
def research_node(state: ResearchState) -> dict:
    """Researcher Agent: 搜索和收集信息"""
    topic = state["topic"]

    prompt = f"""你是一个研究员。请为以下主题收集和整理关键信息。

主题：{topic}

请提供5-8个关键发现，每个发现包含：
- 标题
- 内容（2-3句话）
- 信息来源类型（搜索/文档）"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    findings = response.choices[0].message.content

    return {"research_findings": [findings]}

def write_node(state: ResearchState) -> dict:
    """Writer Agent: 撰写报告"""
    topic = state["topic"]
    findings = "\n".join(state.get("research_findings", []))
    review_comments = state.get("review_comments", "")

    if state.get("revision_count", 0) > 0 and review_comments:
        # 修改模式
        prompt = f"""你是一个专业写手。请根据审核意见修改报告。

主题：{topic}
研究发现：{findings}

当前报告：
{state.get('draft_report', '')}

审核意见：
{review_comments}

请修改报告，解决审核中提到的问题。"""
    else:
        # 初稿模式
        prompt = f"""你是一个专业写手。请基于研究发现撰写一份研究报告。

主题：{topic}

研究发现：
{findings}

报告要求：
1. 包含摘要、引言、主体、结论
2. 使用Markdown格式
3. 结构清晰，逻辑严谨
4. 800-1500字"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return {
        "draft_report": response.choices[0].message.content,
        "revision_count": state.get("revision_count", 0),
    }

def review_node(state: ResearchState) -> dict:
    """Reviewer Agent: 审核报告"""
    topic = state["topic"]
    report = state.get("draft_report", "")

    prompt = f"""你是一个严格的审稿人。请审核以下研究报告的质量。

主题：{topic}
报告：
{report}

请从以下维度评分(1-10)：
1. 内容准确性
2. 结构完整性
3. 逻辑清晰度
4. 可读性
5. 与主题的相关性

总评分(1-10)：
具体修改建议（如果分数<8）："""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    review = response.choices[0].message.content

    # 提取评分
    import re
    score_match = re.search(r'总评分[：:]\s*(\d+)', review)
    score = float(score_match.group(1)) if score_match else 7.0

    return {
        "review_comments": review,
        "review_score": score,
    }

def finalize_node(state: ResearchState) -> dict:
    """最终确认"""
    return {"final_report": state.get("draft_report", "")}

# 4. 条件判断
def should_revise(state: ResearchState) -> Literal["revise", "finalize"]:
    """判断是否需要修改"""
    score = state.get("review_score", 0)
    revisions = state.get("revision_count", 0)

    if score >= 8 or revisions >= 2:
        return "finalize"
    return "revise"

# 5. 构建工作流
workflow = StateGraph(ResearchState)

workflow.add_node("research", research_node)
workflow.add_node("write", write_node)
workflow.add_node("review", review_node)
workflow.add_node("finalize", finalize_node)

workflow.set_entry_point("research")
workflow.add_edge("research", "write")
workflow.add_edge("write", "review")
workflow.add_conditional_edges(
    "review",
    should_revise,
    {"revise": "write", "finalize": "finalize"}
)
workflow.add_edge("finalize", END)

app = workflow.compile()

# 6. 运行
result = app.invoke({
    "topic": "大语言模型在企业数字化转型中的应用",
    "research_findings": [],
    "draft_report": "",
    "review_comments": "",
    "review_score": 0.0,
    "final_report": "",
    "revision_count": 0,
})

print("=" * 60)
print("最终报告：")
print(result["final_report"])
print(f"\n修改次数: {result['revision_count']}")
print(f"最终评分: {result['review_score']}")
```

---

## 四、代码练习

### Day 3 练习：用LangGraph实现带Human-in-the-loop的工作流

```
任务：
1. 实现"文档生成→人工审核→根据反馈修改"的工作流
2. 使用interrupt_before暂停等待人工输入
3. 使用MemorySaver保存状态
4. 测试完整的审核-修改-确认流程
5. 输出：human_in_loop_workflow.py
```

### Day 5 练习：实现CodeAgent

```
任务：
1. 实现CodeAgent（代码生成→执行→修复闭环）
2. 测试5个Python代码生成任务
3. 记录每次的尝试次数和最终成功率
4. 输出：code_agent.py + 测试结果
```

### Day 7 练习：完成项目11 Multi-Agent系统

```
任务：
1. 完成研究报告自动生成系统
2. 包含Researcher + Writer + Reviewer三个Agent
3. 使用LangGraph编排工作流
4. 测试3个不同主题的报告生成
5. 输出：multi_agent_report/ 完整项目
```

---

## 五、本周产出

| 产出物 | 说明 | 完成标准 |
|--------|------|----------|
| human_in_loop_workflow.py | Human-in-the-loop工作流 | 审核-修改-确认流程 |
| code_agent.py | 代码生成Agent | 生成→执行→修复闭环 |
| multi_agent_report/ | Multi-Agent报告系统 | 3个Agent + LangGraph编排 |

---

## 六、自测题

### 题目

1. **Multi-Agent的4种架构模式？**

<details>
<summary>参考答案</summary>

(1) 主从模式（Manager-Worker）：一个Manager Agent负责接收任务、分解子任务、分配给Worker Agent执行、汇总结果。适合任务可以明确分解的场景。(2) 对等模式（Peer-to-Peer）：Agent之间直接通信，没有中心控制。适合协作讨论、互相审查。(3) 层级模式（Hierarchical）：多层管理结构，如CEO→Manager→Worker，适合复杂任务的多级分解。(4) 路由模式（Router）：Router Agent根据请求类型分发到专家Agent。适合不同类型任务需要不同专长Agent的场景。
</details>

2. **Agent间通信的几种方式？**

<details>
<summary>参考答案</summary>

(1) 消息传递（Message Passing）：Agent通过标准化的Message对象通信，每个Agent有邮箱，发送者指定接收者。适合松耦合的Agent系统。(2) 共享状态（Shared State）：多个Agent读写同一个State对象，通过StateManager管理状态更新和监听。适合需要紧密协作的场景。(3) 任务分配（Task Assignment）：Manager Agent根据Worker的能力和负载智能分配任务，类似工作队列。三种方式可以组合使用，如共享状态+消息传递。
</details>

3. **Plan-and-Execute和ReAct的区别？**

<details>
<summary>参考答案</summary>

ReAct是动态推理模式：每一步先思考（Thought），然后决定下一步做什么（Action），根据结果再思考。每步决策都基于最新信息，灵活但可能不够系统。Plan-and-Execute是先规划再执行模式：先用Planner Agent制定完整的步骤计划，然后Executor Agent按计划逐步执行。步骤提前确定，系统性强但灵活性不足。选择建议：任务步骤明确、可以提前规划时用Plan-and-Execute；需要根据中间结果灵活调整时用ReAct。两者也可以结合：先制定粗略计划，执行中用ReAct灵活调整。
</details>

4. **Human-in-the-loop在什么场景下必要？**

<details>
<summary>参考答案</summary>

(1) 高风险决策：AI Agent做出的决策可能产生重大影响时（如财务审批、医疗诊断建议），需要人工确认。(2) 质量把关：AI生成内容需要满足特定标准时（如法律文件、营销文案），需要人工审核。(3) 训练反馈：通过人工反馈来改进Agent的行为，类似RLHF中的人类偏好标注。(4) 异常处理：Agent遇到无法处理的情况时，转交给人工处理。(5) 合规要求：某些行业规定必须有人工参与关键决策环节。
</details>

5. **如何处理Agent执行失败的情况？**

<details>
<summary>参考答案</summary>

(1) 重试机制：设定最大重试次数，失败后自动重试。(2) 反思改进：Reflexion模式，失败后分析原因并调整策略再重试。(3) 降级策略：主Agent失败后切换到备用Agent或简化流程。(4) 错误隔离：一个Agent的失败不影响其他Agent，使用try-except包裹。(5) Checkpoint恢复：使用状态快照保存进度，失败后从断点恢复。(6) 人工接管：失败达到阈值后通知人工介入。关键是设计时考虑失败场景，而不是假设一切顺利。
</details>

---

## 七、Java开发者提示

### Multi-Agent类比

```
Multi-Agent系统 ≈ Java微服务架构

主从模式 → API Gateway + 微服务
  Gateway(MBA)路由请求到具体的微服务(Worker)

对等模式 → 微服务间直接调用
  ServiceA → ServiceB → ServiceC

层级模式 → 组织架构
  总部 → 分公司 → 部门 → 团队

路由模式 → Service Discovery + Load Balancer
  Nginx根据URL路由到不同的后端服务
```

### Agent通信类比

```
Agent通信 ≈ Java微服务间通信

消息传递 → 消息队列（RabbitMQ/Kafka）
  Agent A → Queue → Agent B
  解耦、异步、可靠

共享状态 → 分布式缓存（Redis）
  多个Agent读写同一个State
  需要考虑一致性问题

任务分配 → 任务调度器（Quartz/XXL-Job）
  Manager分发任务到Worker
  考虑负载均衡和能力匹配
```

### LangGraph类比

```
LangGraph ≈ Java工作流引擎（Camunda/Activiti）

Camunda BPMN流程：
  Start → Service Task → User Task → Gateway → End
  用XML定义流程，引擎执行

LangGraph StateGraph：
  Entry → Node → Conditional Edge → Node → END
  用Python代码定义，更灵活

核心概念对应：
  BPMN Task    → LangGraph Node
  BPMN Gateway → LangGraph Conditional Edge
  BPMN Event   → LangGraph interrupt/checkpoint
  BPMN Data    → LangGraph State
```

### 代码Agent类比

```
CodeAgent ≈ Java中的CI/CD Pipeline

CI/CD流程：
  代码提交 → 编译 → 测试 → 部署
  失败 → 分析日志 → 修复 → 重新执行

CodeAgent流程：
  生成代码 → 执行 → 测试 → 返回结果
  失败 → 分析错误 → 修复 → 重新执行

都是自动化的"执行→验证→修复"循环
```
