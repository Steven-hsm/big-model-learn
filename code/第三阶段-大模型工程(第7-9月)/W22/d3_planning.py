### Day 3（周三）：任务规划与多步推理
# 任务分解, 思维链规划, Plan-and-Execute, 规划效果可视化

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 任务分解 (Task Decomposition)
# ============================================================
print("=" * 60)
print("1. 任务分解 (Task Decomposition)")
print("=" * 60)

class TaskDecomposer:
    """任务分解器: 将复杂任务拆解为子任务"""

    def __init__(self):
        self.task_tree = {}

    def decompose(self, task, max_depth=3, current_depth=0):
        """递归分解任务"""
        if current_depth >= max_depth:
            return {'name': task, 'subtasks': [], 'depth': current_depth}

        # 简化: 根据任务类型分解
        subtasks_map = {
            '写一篇关于AI的论文': [
                '调研AI发展历史',
                '整理核心技术要点',
                '撰写论文大纲',
                '完成各章节内容',
                '审校和修改',
            ],
            '开发一个Web应用': [
                '需求分析',
                '数据库设计',
                '���端API开发',
                '前端页面开发',
                '测试与部署',
            ],
            '分析销售数据': [
                '数据收集与清洗',
                '探索性数据分析',
                '趋势分析',
                '生成可视化报告',
                '提出优化建议',
            ],
            '调研AI发展历史': [
                '搜索早期AI研究',
                '整理神经网络发展',
                '总结深度学习突破',
            ],
        }

        subtasks = subtasks_map.get(task, [])
        result = {
            'name': task,
            'subtasks': [],
            'depth': current_depth,
        }

        for subtask in subtasks:
            child = self.decompose(subtask, max_depth, current_depth + 1)
            result['subtasks'].append(child)

        return result

    def flatten(self, task_tree):
        """将任务树展平为列表"""
        tasks = []
        if task_tree['subtasks']:
            for sub in task_tree['subtasks']:
                tasks.extend(self.flatten(sub))
        else:
            tasks.append(task_tree['name'])
        return tasks

    def estimate_complexity(self, task_tree):
        """估算任务复杂度"""
        flat = self.flatten(task_tree)
        return len(flat)


decomposer = TaskDecomposer()

# 分解复杂任务
complex_task = "写一篇关于AI的论文"
task_tree = decomposer.decompose(complex_task, max_depth=2)

print(f"\n  复杂任务: {complex_task}")
print(f"  子任务数: {decomposer.estimate_complexity(task_tree)}")
print(f"\n  任务分解结果:")

def print_tree(node, indent=0):
    prefix = "  " * indent + ("├─ " if indent > 0 else "")
    status = f" (复杂度: {len(node['subtasks'])}子任务)" if node['subtasks'] else " [叶子任务]"
    print(f"  {prefix}{node['name']}{status}")
    for sub in node['subtasks']:
        print_tree(sub, indent + 1)

print_tree(task_tree)


# ============================================================
# 2. 思维链规划 (Chain-of-Thought Planning)
# ============================================================
print("\n" + "=" * 60)
print("2. 思维链规划 (Chain-of-Thought)")
print("=" * 60)

class ChainOfThoughtPlanner:
    """思维链规划器"""

    def plan(self, task):
        """生成思维链规划"""
        # 模拟思维链推理过程
        cot_examples = {
            '计算20件商品的总价, 每件15元, 打8折': [
                ('Step 1', '理解问题', '需要计算20件商品在打8折后的总价'),
                ('Step 2', '计算原价', '20件 × 15元/件 = 300元'),
                ('Step 3', '应用折扣', '300元 × 0.8 = 240元'),
                ('Step 4', '得出答案', '总价为240元'),
            ],
            '判断378是否能被9整除': [
                ('Step 1', '理解问题', '需要判断378 ÷ 9是否为整数'),
                ('Step 2', '方法一: 直接除', '378 ÷ 9 = 42, 是整数'),
                ('Step 3', '方法二: 数位和', '3+7+8=18, 18÷9=2, 可整除'),
                ('Step 4', '得出答案', '378能被9整除'),
            ],
        }

        # 通用思维链模板
        generic_cot = [
            ('Step 1', '理解问题', f'分析任务: {task}'),
            ('Step 2', '制定计划', '确定解决步骤和所需信息'),
            ('Step 3', '执行步骤', '按计划逐步执行'),
            ('Step 4', '验证结果', '检查结果是否正确'),
            ('Step 5', '得出答案', '总结最终答案'),
        ]

        return cot_examples.get(task, generic_cot)

# 测试
planner = ChainOfThoughtPlanner()
tasks_cot = [
    '计算20件商品的总价, 每件15元, 打8折',
    '判断378是否能被9整除',
    '设计一个推荐系统',
]

for task in tasks_cot:
    print(f"\n  任务: {task}")
    steps = planner.plan(task)
    for step_id, step_name, content in steps:
        print(f"    {step_id} [{step_name}]: {content}")


# ============================================================
# 3. Plan-and-Execute模式
# ============================================================
print("\n" + "=" * 60)
print("3. Plan-and-Execute模式")
print("=" * 60)

class PlanAndExecuteAgent:
    """Plan-and-Execute Agent"""

    def __init__(self):
        self.plan = []
        self.results = []
        self.execution_log = []

    def create_plan(self, task):
        """创建完整计划"""
        # 预定义计划模板
        plan_templates = {
            '数据分析项目': [
                {'step': 1, 'action': '收集数据', 'tool': 'data_loader', 'status': 'pending'},
                {'step': 2, 'action': '数据清洗', 'tool': 'data_cleaner', 'status': 'pending'},
                {'step': 3, 'action': '统计分析', 'tool': 'analyzer', 'status': 'pending'},
                {'step': 4, 'action': '可视化', 'tool': 'visualizer', 'status': 'pending'},
                {'step': 5, 'action': '生成报告', 'tool': 'reporter', 'status': 'pending'},
            ],
        }

        self.plan = plan_templates.get(task, [
            {'step': i + 1, 'action': f'执行步骤{i + 1}', 'tool': 'general', 'status': 'pending'}
            for i in range(4)
        ])
        return self.plan

    def execute_step(self, step_index):
        """执行单个步骤"""
        if step_index >= len(self.plan):
            return None

        step = self.plan[step_index]
        step['status'] = 'running'

        # 模拟执行
        success_prob = 0.85
        success = np.random.random() < success_prob

        if success:
            step['status'] = 'completed'
            result = f"{step['action']}完成"
        else:
            step['status'] = 'failed'
            result = f"{step['action']}失败"

        self.execution_log.append({
            'step': step['step'],
            'action': step['action'],
            'success': success,
            'result': result,
        })
        self.results.append(result)
        return result

    def replan(self, from_step):
        """重新规划 (从某步开始)"""
        for i in range(from_step, len(self.plan)):
            if self.plan[i]['status'] == 'failed':
                self.plan[i]['status'] = 'pending'
                self.plan[i]['retry'] = self.plan[i].get('retry', 0) + 1

    def run(self, task):
        """运行完整的Plan-and-Execute"""
        print(f"\n  任务: {task}")
        print(f"  {'=' * 40}")

        # 创建计划
        self.create_plan(task)
        print(f"\n  初始计划:")
        for step in self.plan:
            print(f"    Step {step['step']}: {step['action']} (工具: {step['tool']})")

        # 执行
        print(f"\n  执行过程:")
        for i in range(len(self.plan)):
            result = self.execute_step(i)
            step = self.plan[i]
            status_icon = "OK" if step['status'] == 'completed' else "FAIL"
            print(f"    Step {step['step']}: [{status_icon}] {result}")

            # 如果失败, 尝试重新规划
            if step['status'] == 'failed':
                print(f"    → 重新规划...")
                self.replan(i)
                result = self.execute_step(i)
                print(f"    Step {step['step']} (重试): [{status_icon}] {result}")

        completed = sum(1 for s in self.plan if s['status'] == 'completed')
        print(f"\n  完成: {completed}/{len(self.plan)} 步")
        return self.results


np.random.seed(42)
agent = PlanAndExecuteAgent()
agent.run("数据分析项目")


# ============================================================
# 4. 多步推理实现
# ============================================================
print("\n" + "=" * 60)
print("4. 多步推理性能分析")
print("=" * 60)

# 模拟不同任务复杂度的规划效果
np.random.seed(42)
task_complexities = [3, 5, 8, 10, 15, 20]
planning_results = {
    'success_rate': [],
    'avg_steps': [],
    'replan_count': [],
    'time_cost': [],
}

for complexity in task_complexities:
    success_count = 0
    total_replans = 0
    for _ in range(100):
        steps_needed = complexity
        success = True
        replans = 0
        for _ in range(steps_needed):
            if np.random.random() > 0.85:
                success = False
                replans += 1
                # 重试
                if np.random.random() > 0.5:
                    success = True
        if success:
            success_count += 1
        total_replans += replans

    planning_results['success_rate'].append(success_count / 100)
    planning_results['avg_steps'].append(complexity)
    planning_results['replan_count'].append(total_replans / 100)
    planning_results['time_cost'].append(complexity * 0.5 + total_replans / 100 * 0.3)

print(f"  {'复杂度':>8} {'成功率':>8} {'平均重规划':>10} {'时间成本':>8}")
print(f"  {'-' * 40}")
for i, c in enumerate(task_complexities):
    print(f"  {c:>8} {planning_results['success_rate'][i]:>8.2%} "
          f"{planning_results['replan_count'][i]:>10.2f} "
          f"{planning_results['time_cost'][i]:>8.2f}s")


# ============================================================
# 5. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 5.1 任务分解树状图
ax1 = axes[0, 0]
def draw_tree(ax, node, x, y, x_spread, level=0):
    ax.text(x, y, node['name'][:10], ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor=plt.cm.Set3(level / 3), alpha=0.8),
            fontsize=8)
    n_children = len(node['subtasks'])
    if n_children > 0:
        child_spread = x_spread / max(n_children, 1)
        start_x = x - x_spread / 2 + child_spread / 2
        for i, child in enumerate(node['subtasks']):
            child_x = start_x + i * child_spread
            child_y = y - 0.25
            ax.annotate('', xy=(child_x, child_y + 0.03), xytext=(x, y - 0.03),
                        arrowprops=dict(arrowstyle='->', color='gray'))
            draw_tree(ax, child, child_x, child_y, child_spread, level + 1)

draw_tree(ax1, task_tree, 0.5, 0.9, 0.9)
ax1.set_xlim(-0.1, 1.1)
ax1.set_ylim(0, 1)
ax1.set_title('任务分解树状图')
ax1.axis('off')

# 5.2 Plan-and-Execute执行过程
ax2 = axes[0, 1]
if agent.execution_log:
    steps = [f"S{log['step']}" for log in agent.execution_log]
    success = [1 if log['success'] else 0 for log in agent.execution_log]
    colors = ['#4CAF50' if s else '#F44336' for s in success]
    ax2.bar(steps, [1] * len(steps), color=colors, edgecolor='black')
    ax2.set_ylabel('执行状态')
    ax2.set_title('Plan-and-Execute执行过程')
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['', '成功'])
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#4CAF50', label='成功'),
                       Patch(facecolor='#F44336', label='失败')]
    ax2.legend(handles=legend_elements)

# 5.3 任务复杂度 vs 成功率
ax3 = axes[1, 0]
ax3.plot(task_complexities, planning_results['success_rate'], 'go-', linewidth=2, markersize=8)
ax3.set_xlabel('任务复杂度 (步骤数)')
ax3.set_ylabel('成功率')
ax3.set_title('任务复杂度 vs 执行成功率')
ax3.set_ylim(0, 1.05)
ax3.grid(True, alpha=0.3)

# 5.4 复杂度 vs 时间成本和重规划
ax4 = axes[1, 1]
ax4_twin = ax4.twinx()
l1 = ax4.plot(task_complexities, planning_results['time_cost'], 'b^-', linewidth=2,
              label='时间成本 (s)', markersize=8)
l2 = ax4_twin.plot(task_complexities, planning_results['replan_count'], 'rs-', linewidth=2,
                   label='平均重规划次数', markersize=8)
ax4.set_xlabel('任务复杂度 (步骤数)')
ax4.set_ylabel('时间成本 (s)', color='blue')
ax4_twin.set_ylabel('重规划次数', color='red')
ax4.set_title('复杂度对规划效率的影响')
lines = l1 + l2
ax4.legend(lines, [l.get_label() for l in lines])

plt.suptitle('W22-D3: 任务规划与多步推理', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W22/d3_planning.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存!")
