### Day 5（周五）：工作流Agent
# DAG任务编排, 条件分支, 并行执行, 错误处理, 工作流可视化

import numpy as np
import matplotlib.pyplot as plt
from collections import deque

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. DAG任务编排
# ============================================================
print("=" * 60)
print("1. DAG任务编排 (有向无环图)")
print("=" * 60)

class WorkflowNode:
    """工作流节点"""

    def __init__(self, name, func=None, node_type='task'):
        self.name = name
        self.func = func
        self.node_type = node_type  # task, condition, parallel, merge
        self.dependencies = []
        self.children = []
        self.status = 'pending'  # pending, running, completed, failed, skipped
        self.result = None
        self.error = None
        self.execution_time = 0

    def add_child(self, child):
        self.children.append(child)
        child.dependencies.append(self)


class Workflow:
    """工作流编排引擎"""

    def __init__(self, name):
        self.name = name
        self.nodes = {}
        self.start_node = None
        self.execution_order = []
        self.results = {}

    def add_node(self, node):
        self.nodes[node.name] = node
        if self.start_node is None:
            self.start_node = node

    def connect(self, parent_name, child_name):
        parent = self.nodes[parent_name]
        child = self.nodes[child_name]
        parent.add_child(child)

    def topological_sort(self):
        """拓扑排序"""
        in_degree = {name: 0 for name in self.nodes}
        for name, node in self.nodes.items():
            for child in node.children:
                in_degree[child.name] += 1

        queue = deque([name for name, deg in in_degree.items() if deg == 0])
        order = []

        while queue:
            current = queue.popleft()
            order.append(current)
            for child in self.nodes[current].children:
                in_degree[child.name] -= 1
                if in_degree[child.name] == 0:
                    queue.append(child)

        if len(order) != len(self.nodes):
            raise ValueError("检测到循环依赖!")

        return order

    def execute(self, context=None):
        """执行工作流"""
        context = context or {}
        print(f"\n  执行工作流: {self.name}")
        print(f"  {'=' * 40}")

        order = self.topological_sort()
        print(f"  拓扑排序: {' → '.join(order)}")

        for node_name in order:
            node = self.nodes[node_name]
            node.status = 'running'

            # 检查依赖是否完成
            deps_ok = all(
                self.nodes[dep.name].status == 'completed'
                for dep in node.dependencies
            )
            if not deps_ok:
                node.status = 'skipped'
                continue

            # 执行节点
            try:
                result = self._execute_node(node, context)
                node.result = result
                node.status = 'completed'
                self.results[node_name] = result
                print(f"  [OK] {node_name}: {str(result)[:50]}")
            except Exception as e:
                node.status = 'failed'
                node.error = str(e)
                print(f"  [FAIL] {node_name}: {e}")
                # 简化: 继续执行其他节点

        return self.results

    def _execute_node(self, node, context):
        """执行单个节点"""
        if node.func:
            return node.func(context)
        return f"完成 {node.name}"


# 定义工作流函数
def collect_data(ctx):
    return {'status': '数据收集完成', 'records': 1000}

def clean_data(ctx):
    return {'status': '数据清洗完成', 'valid_records': 950}

def feature_engineering(ctx):
    return {'status': '特征工程完成', 'features': 25}

def train_model(ctx):
    return {'status': '模型训练完成', 'accuracy': 0.92}

def evaluate_model(ctx):
    return {'status': '评估完成', 'f1_score': 0.89}

def generate_report(ctx):
    return {'status': '报告生成完成', 'pages': 5}

# 构建ML工作流
ml_workflow = Workflow("机器学习Pipeline")
ml_workflow.add_node(WorkflowNode("数据收集", collect_data))
ml_workflow.add_node(WorkflowNode("数据清洗", clean_data))
ml_workflow.add_node(WorkflowNode("特征工程", feature_engineering))
ml_workflow.add_node(WorkflowNode("模型训练", train_model))
ml_workflow.add_node(WorkflowNode("模型评估", evaluate_model))
ml_workflow.add_node(WorkflowNode("生成报告", generate_report))

ml_workflow.connect("数据收集", "数据清洗")
ml_workflow.connect("数据清洗", "特征工程")
ml_workflow.connect("特征工程", "模型训练")
ml_workflow.connect("模型训练", "模型评估")
ml_workflow.connect("模型评估", "生成报告")

results = ml_workflow.execute()


# ============================================================
# 2. 条件分支工作流
# ============================================================
print("\n" + "=" * 60)
print("2. 条件分支工作流")
print("=" * 60)

class ConditionalWorkflow(Workflow):
    """支持条件分支的工作流"""

    def __init__(self, name):
        super().__init__(name)
        self.conditions = {}

    def add_condition(self, parent_name, true_child, false_child, condition_func):
        """添加条件分支"""
        self.conditions[parent_name] = {
            'true': true_child,
            'false': false_child,
            'condition': condition_func,
        }

    def execute(self, context=None):
        """执行带条件的工作流"""
        context = context or {}
        print(f"\n  执行条件工作流: {self.name}")

        # 依次执行, 遇到条件节点时分支
        for node_name, node in self.nodes.items():
            if node.status == 'completed':
                continue

            # 检查条件
            if node_name in self.conditions:
                cond = self.conditions[node_name]
                result = cond['condition'](context)
                print(f"  [条件] {node_name}: {'真' if result else '假'}")
                if result:
                    target = cond['true']
                else:
                    target = cond['false']
                self.nodes[target].status = 'completed'
                self.nodes[target].result = f"条件分支选择: {target}"
                print(f"  → 选择分支: {target}")
            else:
                try:
                    result = self._execute_node(node, context)
                    node.result = result
                    node.status = 'completed'
                    print(f"  [OK] {node_name}: {str(result)[:50]}")
                except Exception as e:
                    node.status = 'failed'
                    print(f"  [FAIL] {node_name}: {e}")

        return {n: self.nodes[n].result for n in self.nodes if self.nodes[n].result}


# 构建条件工作流
cond_workflow = ConditionalWorkflow("客户服务工作流")
cond_workflow.add_node(WorkflowNode("接收请求", lambda ctx: "请求已接收"))
cond_workflow.add_node(WorkflowNode("分类请求", lambda ctx: "分类完成"))
cond_workflow.add_node(WorkflowNode("技术支持处理", lambda ctx: "技术问题已解决"))
cond_workflow.add_node(WorkflowNode("销售咨询处理", lambda ctx: "销售信息已提供"))

cond_workflow.add_condition(
    "分类请求", "技术支持处理", "销售咨询处理",
    lambda ctx: ctx.get('request_type') == 'technical'
)

cond_workflow.execute({'request_type': 'technical'})


# ============================================================
# 3. 并行执行模拟
# ============================================================
print("\n" + "=" * 60)
print("3. 并行执行分析")
print("=" * 60)

# 模拟串行 vs 并行执行时间
tasks_execution = {
    '数据收集': 5,
    '数据清洗': 3,
    '特征A': 4,
    '特征B': 4,
    '特征C': 3,
    '模型训练': 10,
    '模型评估': 2,
    '报告生成': 3,
}

serial_time = sum(tasks_execution.values())
print(f"  串行总时间: {serial_time}s")

# 并行: 特征A/B/C可以并行
parallel_time = (tasks_execution['数据收集'] + tasks_execution['数据清洗'] +
                 max(tasks_execution['特征A'], tasks_execution['特征B'], tasks_execution['特征C']) +
                 tasks_execution['模型训练'] + tasks_execution['模型评估'] +
                 tasks_execution['报告生成'])
print(f"  并行总时间: {parallel_time}s")
print(f"  加速比: {serial_time / parallel_time:.2f}x")


# ============================================================
# 4. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 5.1 DAG工作流图
ax1 = axes[0, 0]
node_positions = {
    '数据收集': (0.5, 0.9),
    '数据清洗': (0.5, 0.72),
    '特征工程': (0.5, 0.54),
    '模型训练': (0.5, 0.36),
    '模型评估': (0.5, 0.18),
    '生成报告': (0.5, 0.0),
}

status_colors = {
    'completed': '#4CAF50', 'running': '#FF9800',
    'failed': '#F44336', 'pending': '#9E9E9E',
}

for name, pos in node_positions.items():
    node = ml_workflow.nodes[name]
    color = status_colors.get(node.status, '#9E9E9E')
    ax1.add_patch(plt.Rectangle((pos[0] - 0.18, pos[1] - 0.06), 0.36, 0.12,
                                 facecolor=color, alpha=0.7, edgecolor='black', linewidth=2))
    ax1.text(pos[0], pos[1], name, ha='center', va='center', fontsize=9, fontweight='bold')

# 连接线
edges = [
    ('数据收集', '数据清洗'), ('数据清洗', '特征工程'),
    ('特征工程', '模型训练'), ('模型训练', '模型评估'),
    ('模型评估', '生成报告'),
]
for parent, child in edges:
    p = node_positions[parent]
    c = node_positions[child]
    ax1.annotate('', xy=(c[0], c[1] + 0.06), xytext=(p[0], p[1] - 0.06),
                 arrowprops=dict(arrowstyle='->', color='black', lw=2))

ax1.set_xlim(0, 1)
ax1.set_ylim(-0.1, 1.0)
ax1.set_title('ML Pipeline工作流DAG')
ax1.axis('off')

# 5.2 串行 vs 并行执行时间
ax2 = axes[0, 1]
labels = ['数据收集', '数据清洗', '特征A|B|C', '模型训练', '评估', '报告']
serial_times = [5, 3, 11, 10, 2, 3]  # 特征A+B+C串行
parallel_times = [5, 3, 4, 10, 2, 3]  # 特征A/B/C并行取最大

x = np.arange(len(labels))
width = 0.35
ax2.bar(x - width / 2, serial_times, width, label='串行', color='#FF9800', edgecolor='black')
ax2.bar(x + width / 2, parallel_times, width, label='并行', color='#4CAF50', edgecolor='black')
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=9)
ax2.set_ylabel('执行时间 (s)')
ax2.set_title(f'串行 vs 并行 (加速比: {serial_time / parallel_time:.1f}x)')
ax2.legend()

# 5.3 条件分支示意图
ax3 = axes[1, 0]
branch_steps = [
    ('接收请求', 0.5, 0.9),
    ('分类请求', 0.5, 0.65),
    ('技术支持', 0.2, 0.35),
    ('销售咨询', 0.8, 0.35),
]
for name, x, y in branch_steps:
    color = '#4CAF50' if name != '销售咨询' else '#9E9E9E'
    ax3.add_patch(plt.Rectangle((x - 0.15, y - 0.07), 0.3, 0.14,
                                 facecolor=color, alpha=0.6, edgecolor='black', linewidth=2))
    ax3.text(x, y, name, ha='center', va='center', fontsize=9)

ax3.annotate('', xy=(0.5, 0.72), xytext=(0.5, 0.83),
             arrowprops=dict(arrowstyle='->', color='black', lw=2))
ax3.annotate('', xy=(0.2, 0.42), xytext=(0.4, 0.58),
             arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax3.text(0.22, 0.55, '技术问题', fontsize=8, color='green')
ax3.annotate('', xy=(0.8, 0.42), xytext=(0.6, 0.58),
             arrowprops=dict(arrowstyle='->', color='gray', lw=2, linestyle='dashed'))
ax3.text(0.62, 0.55, '销售问题', fontsize=8, color='gray')

ax3.set_xlim(0, 1)
ax3.set_ylim(0.1, 1)
ax3.set_title('条件分支工作流')
ax3.axis('off')

# 5.4 工作流节点数 vs 执行时间
ax4 = axes[1, 1]
node_counts = [3, 5, 8, 12, 16, 20]
serial_exec = [n * 2.5 for n in node_counts]
parallel_exec = [n * 1.5 + np.log(n + 1) * 2 for n in node_counts]
overhead = [n * 0.3 for n in node_counts]

ax4.plot(node_counts, serial_exec, 'r-o', linewidth=2, label='串行执行')
ax4.plot(node_counts, parallel_exec, 'g-s', linewidth=2, label='并行执行')
ax4.plot(node_counts, overhead, 'b-^', linewidth=2, label='协调开销')
ax4.fill_between(node_counts, serial_exec, parallel_exec, alpha=0.1, color='green')
ax4.set_xlabel('节点数')
ax4.set_ylabel('执行时间 (s)')
ax4.set_title('工作流规模 vs 执行时间')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.suptitle('W23-D5: 工作流Agent', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W23/d5_workflow_agent.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存!")
