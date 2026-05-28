### Day 2（周二）：CrewAI框架与简化版Multi-Agent
# Agent角色定义, Task分配, Process模式, 简化版Multi-Agent实现

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. CrewAI核心概念
# ============================================================
print("=" * 60)
print("1. CrewAI核心概念")
print("=" * 60)

print("""
  CrewAI三大核心:
    1. Agent  - 具有角色、目标和工具的智能体
    2. Task   - 需要完成的具体任务
    3. Crew   - 管理Agent和Task的团队

  Process模式:
    - Sequential:  按顺序执行Task
    - Hierarchical: Manager分配Task给Worker

  配置示例 (伪代码):
    researcher = Agent(role='研究员', goal='搜集信息', tools=[search])
    writer = Agent(role='作家', goal='撰写文章', tools=[write])
    task1 = Task(description='研究AI趋势', agent=researcher)
    task2 = Task(description='写AI报告', agent=writer)
    crew = Crew(agents=[researcher, writer], tasks=[task1, task2], process='sequential')
    crew.kickoff()
""")


# ============================================================
# 2. 简化版CrewAI实现
# ============================================================
class CrewAgent:
    """简化版CrewAI Agent"""

    def __init__(self, role, goal, backstory, tools=None, verbose=True):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.tools = tools or []
        self.verbose = verbose
        self.context = ""
        self.output = None

    def execute_task(self, task):
        """执行任务"""
        if self.verbose:
            print(f"  [{self.role}] 执行任务: {task.description[:50]}...")

        # 模拟任务执行
        result = self._simulate_execution(task)

        if self.verbose:
            print(f"  [{self.role}] 完成任务: {str(result)[:60]}...")

        self.output = result
        return result

    def _simulate_execution(self, task):
        """模拟任务执行结果"""
        task_lower = task.description.lower()

        if '研究' in task_lower or '调研' in task_lower:
            return {
                'type': 'research',
                'findings': [
                    'AI市场规模预计2025年达到5000亿美元',
                    '大语言模型是当前最热门的技术方向',
                    '多模态AI正在快速发展',
                ],
                'sources': ['报告A', '论文B', '新闻C'],
            }
        elif '写' in task_lower or '撰写' in task_lower:
            return {
                'type': 'writing',
                'title': task.description,
                'content': '基于研究发现, AI行业正在经历前所未有的增长...',
                'word_count': 500,
            }
        elif '审核' in task_lower or '审查' in task_lower:
            return {
                'type': 'review',
                'quality_score': 0.85,
                'issues': ['需要补充数据支撑', '结论部分可以更详细'],
                'approved': False,
            }
        elif '代码' in task_lower or '编程' in task_lower:
            return {
                'type': 'code',
                'files': ['main.py', 'utils.py', 'tests/test_main.py'],
                'lines': 150,
                'test_pass_rate': 0.9,
            }
        else:
            return {'type': 'general', 'result': f"完成: {task.description[:30]}"}

    def delegate(self, task, other_agent):
        """委托任务给另一个Agent"""
        if self.verbose:
            print(f"  [{self.role}] 委托任务给 [{other_agent.role}]")
        other_agent.context = self.output
        return other_agent.execute_task(task)


class CrewTask:
    """简化版CrewAI Task"""

    def __init__(self, description, agent=None, expected_output=None, context=None):
        self.description = description
        self.agent = agent
        self.expected_output = expected_output
        self.context = context or []
        self.result = None
        self.status = 'pending'


class Crew:
    """简化版CrewAI Crew"""

    def __init__(self, agents, tasks, process='sequential', verbose=True):
        self.agents = agents
        self.tasks = tasks
        self.process = process
        self.verbose = verbose
        self.results = []
        self.execution_log = []

    def kickoff(self):
        """启动Crew执行"""
        if self.verbose:
            print(f"\n  === Crew启动 ===")
            print(f"  Agents: {[a.role for a in self.agents]}")
            print(f"  Tasks: {len(self.tasks)}个")
            print(f"  Process: {self.process}")
            print(f"  {'=' * 40}")

        if self.process == 'sequential':
            return self._run_sequential()
        elif self.process == 'hierarchical':
            return self._run_hierarchical()
        else:
            raise ValueError(f"未知Process: {self.process}")

    def _run_sequential(self):
        """顺序执行"""
        for i, task in enumerate(self.tasks):
            if self.verbose:
                print(f"\n  --- Task {i + 1}/{len(self.tasks)} ---")

            agent = task.agent or self.agents[i % len(self.agents)]
            result = agent.execute_task(task)
            task.result = result
            task.status = 'completed'
            self.results.append(result)
            self.execution_log.append({
                'task': task.description,
                'agent': agent.role,
                'status': 'completed',
            })

        return self.results

    def _run_hierarchical(self):
        """层级式执行 (Manager分配)"""
        manager = self.agents[0]
        workers = self.agents[1:]

        if self.verbose:
            print(f"\n  Manager [{manager.role}] 分配任务:")

        for i, task in enumerate(self.tasks):
            worker = workers[i % len(workers)]
            if self.verbose:
                print(f"    Task {i + 1} → {worker.role}")

            result = manager.delegate(task, worker)
            task.result = result
            task.status = 'completed'
            self.results.append(result)
            self.execution_log.append({
                'task': task.description,
                'manager': manager.role,
                'worker': worker.role,
                'status': 'completed',
            })

        return self.results


# ============================================================
# 3. 运行示例
# ============================================================
print("=" * 60)
print("2. Sequential模式示例: AI报告生成")
print("=" * 60)

# 创建Agents
researcher = CrewAgent(
    role="研究员",
    goal="搜集和分析AI行业信息",
    backstory="你是一位资深AI行业研究员, 擅长数据分析和趋势预测",
)

writer = CrewAgent(
    role="作家",
    goal="撰写高质量的AI行业报告",
    backstory="你是一位技术作家, 擅长将复杂技术概念转化为易懂的文章",
)

reviewer = CrewAgent(
    role="审核员",
    goal="确保报告质量达标",
    backstory="你是一位严格的编辑, 注重事实准确性和逻辑清晰度",
)

# 创建Tasks
tasks = [
    CrewTask("研究2024年AI行业发展趋势", agent=researcher,
             expected_output="研究报告, 包含关键数据和趋势"),
    CrewTask("撰写AI行业分析报告", agent=writer,
             expected_output="1000字以上的结构化报告"),
    CrewTask("审核报告质量并给出修改意见", agent=reviewer,
             expected_output="质量评分和改进建议"),
]

crew = Crew(
    agents=[researcher, writer, reviewer],
    tasks=tasks,
    process='sequential',
)
results = crew.kickoff()

# 打印结果
print("\n  === 执行结果 ===")
for i, result in enumerate(results):
    print(f"\n  Task {i + 1}: {tasks[i].description[:30]}")
    for key, value in result.items():
        print(f"    {key}: {str(value)[:60]}")


# ============================================================
# 4. Hierarchical模式示例
# ============================================================
print("\n" + "=" * 60)
print("3. Hierarchical模式示例: 项目开发")
print("=" * 60)

manager = CrewAgent(role="项目经理", goal="管理项目进度和质量",
                    backstory="你是一位经验丰富的项目经理")
coder = CrewAgent(role="程序员", goal="编写高质量代码",
                  backstory="你是一位全栈工程师")
tester = CrewAgent(role="测试员", goal="确保代码质量",
                   backstory="你是一位QA工程师")

dev_tasks = [
    CrewTask("开发用户登录模块", expected_output="可运行的代码"),
    CrewTask("编写登录模块的单元测试", expected_output="测试用例"),
    CrewTask("开发数据导出功能", expected_output="可运行的代码"),
]

dev_crew = Crew(
    agents=[manager, coder, tester],
    tasks=dev_tasks,
    process='hierarchical',
)
dev_results = dev_crew.kickoff()


# ============================================================
# 5. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 5.1 Sequential流程图
ax1 = axes[0, 0]
seq_steps = ['研究员\n(研究)', '作家\n(撰写)', '审核员\n(审核)']
for i, step in enumerate(seq_steps):
    ax1.add_patch(plt.Rectangle((0.1, 0.8 - i * 0.3), 0.8, 0.2,
                                 facecolor=f'C{i}', alpha=0.6, edgecolor='black', linewidth=2))
    ax1.text(0.5, 0.9 - i * 0.3, step, ha='center', va='center', fontsize=10)
    if i < 2:
        ax1.annotate('', xy=(0.5, 0.8 - i * 0.3), xytext=(0.5, 0.82 - i * 0.3),
                     arrowprops=dict(arrowstyle='->', color='black', lw=2))
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1.1)
ax1.set_title('Sequential模式流程')
ax1.axis('off')

# 5.2 Hierarchical流程图
ax2 = axes[0, 1]
ax2.add_patch(plt.Rectangle((0.3, 0.75), 0.4, 0.15,
                              facecolor='#E91E63', alpha=0.6, edgecolor='black', linewidth=2))
ax2.text(0.5, 0.825, '项目经理', ha='center', va='center', fontsize=10)

for i, (label, x_pos) in enumerate([('程序员', 0.15), ('测试员', 0.85)]):
    ax2.add_patch(plt.Rectangle((x_pos - 0.15, 0.3), 0.3, 0.15,
                                 facecolor=f'C{i}', alpha=0.6, edgecolor='black', linewidth=2))
    ax2.text(x_pos, 0.375, label, ha='center', va='center', fontsize=10)
    ax2.annotate('', xy=(x_pos, 0.45), xytext=(0.5, 0.75),
                 arrowprops=dict(arrowstyle='->', color='black', lw=2))

ax2.set_xlim(0, 1)
ax2.set_ylim(0.1, 1)
ax2.set_title('Hierarchical模式流程')
ax2.axis('off')

# 5.3 任务完成质量对比
ax3 = axes[1, 0]
processes = ['Sequential', 'Hierarchical']
quality_scores = {
    '代码质量': [0.82, 0.88],
    '文档质量': [0.85, 0.80],
    '完成速度': [0.70, 0.75],
    '错误率(低)': [0.75, 0.85],
}
metrics = list(quality_scores.keys())
x = np.arange(len(processes))
width = 0.18
for i, metric in enumerate(metrics):
    ax3.bar(x + i * width - 0.27, quality_scores[metric], width,
            label=metric, edgecolor='black')
ax3.set_xticks(x)
ax3.set_xticklabels(processes)
ax3.set_ylabel('分数')
ax3.set_title('Process模式质量对比')
ax3.legend(fontsize=8)
ax3.set_ylim(0, 1.1)

# 5.4 Agent数量 vs 效率
ax4 = axes[1, 1]
agent_counts = [1, 2, 3, 4, 5, 6, 7, 8]
throughput = [1, 1.8, 2.5, 3.0, 3.3, 3.5, 3.6, 3.7]
coordination_cost = [0, 0.1, 0.25, 0.4, 0.6, 0.8, 1.0, 1.3]
net_efficiency = [t - c for t, c in zip(throughput, coordination_cost)]

ax4.plot(agent_counts, throughput, 'g-o', linewidth=2, label='总产出')
ax4.plot(agent_counts, coordination_cost, 'r-s', linewidth=2, label='协调成本')
ax4.plot(agent_counts, net_efficiency, 'b-^', linewidth=2, label='净效率')
ax4.set_xlabel('Agent数量')
ax4.set_ylabel('效率')
ax4.set_title('Agent数量 vs 效率')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.suptitle('W23-D2: CrewAI框架与Multi-Agent', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W23/d2_crewai.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存!")
