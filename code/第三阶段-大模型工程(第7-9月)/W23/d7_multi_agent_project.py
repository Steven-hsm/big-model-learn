### Day 7（周日）：Multi-Agent项目 - 代码审查系统
# Writer+Reviewer+Refiner, 迭代改进, 结果评估

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 代码审查系统架构
# ============================================================
print("=" * 60)
print("1. Multi-Agent代码审查系统")
print("=" * 60)

print("""
  架构: Writer → Reviewer → Refiner 循环

  Writer (编写者):  根据需求编写代码
  Reviewer (审查者): 审查代码质量, 指出问题
  Refiner (改进者): 根据审查意见改进代码
  Manager (管理者): 协调整个流程, 决定是否通过

  迭代流程:
    1. Writer编写初始代码
    2. Reviewer审查并给出评分和改进建议
    3. 如果分数 < 阈值, Refiner改进代码
    4. 回到步骤2, 直到代码质量达标或达到最大迭代次数
""")


# ============================================================
# 2. Agent实现
# ============================================================
class CodeReviewAgent:
    """代码审查Agent基类"""

    def __init__(self, name, role):
        self.name = name
        self.role = role
        self.history = []

    def log(self, action, detail):
        self.history.append({
            'time': datetime.now().isoformat(),
            'action': action,
            'detail': detail,
        })


class WriterAgent(CodeReviewAgent):
    """代码编写Agent"""

    def __init__(self):
        super().__init__("Writer", "代码编写")
        self.code_templates = {
            'sort': {
                'code': '''def bubble_sort(arr):
    for i in range(len(arr)):
        for j in range(len(arr) - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr''',
                'language': 'Python',
            },
            'search': {
                'code': '''def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1''',
                'language': 'Python',
            },
            'fibonacci': {
                'code': '''def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)''',
                'language': 'Python',
            },
        }

    def write_code(self, requirement):
        """根据需求编写代码"""
        req_lower = requirement.lower()
        if '排序' in req_lower or 'sort' in req_lower:
            template = self.code_templates['sort']
        elif '搜索' in req_lower or 'search' in req_lower:
            template = self.code_templates['search']
        else:
            template = self.code_templates['fibonacci']

        result = {
            'code': template['code'],
            'language': template['language'],
            'requirement': requirement,
        }
        self.log('write', f"编写了{requirement}的代码")
        return result


class ReviewerAgent(CodeReviewAgent):
    """代码审查Agent"""

    def __init__(self):
        super().__init__("Reviewer", "代码审查")
        self.review_criteria = {
            '正确性': 0.3,
            '可读性': 0.2,
            '性能': 0.2,
            '边界处理': 0.15,
            '代码风格': 0.15,
        }

    def review(self, code_result, iteration=1):
        """审查代码"""
        code = code_result['code']
        np.random.seed(hash(code) % (2 ** 31))

        # 基础分数随迭代提升
        base_bonus = min(iteration * 0.08, 0.3)

        scores = {}
        issues = []
        suggestions = []

        for criterion, weight in self.review_criteria.items():
            base_score = np.random.uniform(0.4, 0.85)
            score = min(base_score + base_bonus, 1.0)
            scores[criterion] = score

            if score < 0.7:
                issues.append(f"{criterion}: {self._get_issue(criterion, score)}")
                suggestions.append(f"建议改进{criterion}: {self._get_suggestion(criterion)}")

        # 加权总分
        total_score = sum(
            scores[c] * self.review_criteria[c] for c in scores
        )

        review_result = {
            'scores': scores,
            'total_score': total_score,
            'issues': issues,
            'suggestions': suggestions,
            'approved': total_score >= 0.75,
            'iteration': iteration,
        }

        self.log('review', f"审查得分: {total_score:.2f}, "
                            f"通过: {review_result['approved']}")
        return review_result

    def _get_issue(self, criterion, score):
        issues = {
            '正确性': '存在逻辑错误',
            '可读性': '变量命名不清晰',
            '性能': '时间复杂度可以优化',
            '边界处理': '缺少空输入处理',
            '代码风格': '不符合PEP8规范',
        }
        return issues.get(criterion, '需要改进')

    def _get_suggestion(self, criterion):
        suggestions = {
            '正确性': '添加单元测试验证',
            '可读性': '使用更有意义的变量名和注释',
            '性能': '考虑使用更高效的算法',
            '边界处理': '添加输入验证和异常处理',
            '代码风格': '遵循PEP8编码规范',
        }
        return suggestions.get(criterion, '请参考最佳实践')


class RefinerAgent(CodeReviewAgent):
    """代码改进Agent"""

    def __init__(self):
        super().__init__("Refiner", "代码改进")
        self.improvements = [
            '添加了类型注解',
            '添加了边界条件检查',
            '优化了算法复杂度',
            '添加了文档字符串',
            '改善了变量命名',
            '添加了异常处理',
            '使用内置函数替代手动实现',
        ]

    def refine(self, code_result, review_result):
        """根据审查意见改进代码"""
        applied = []
        code = code_result['code']

        # 模拟代码改进
        improvements_needed = len(review_result['issues'])

        for i in range(min(improvements_needed, 3)):
            improvement = self.improvements[np.random.randint(len(self.improvements))]
            applied.append(improvement)

        # 在代码中添加改进标记
        refined_code = f"# [改进] {', '.join(applied)}\n{code}"

        result = {
            'code': refined_code,
            'language': code_result['language'],
            'improvements': applied,
            'based_on': review_result['suggestions'],
        }

        self.log('refine', f"应用了{len(applied)}项改进")
        return result


class ReviewManager:
    """审查流程管理器"""

    def __init__(self, writer, reviewer, refiner, max_iterations=5, threshold=0.75):
        self.writer = writer
        self.reviewer = reviewer
        self.refiner = refiner
        self.max_iterations = max_iterations
        self.threshold = threshold
        self.review_history = []

    def run_review(self, requirement):
        """运行完整的代码审查流程"""
        print(f"\n  {'=' * 50}")
        print(f"  需求: {requirement}")
        print(f"  通过阈值: {self.threshold}")
        print(f"  最大迭代: {self.max_iterations}")

        # Step 1: Writer编写代码
        code_result = self.writer.write_code(requirement)
        print(f"\n  [Writer] 初始代码:")
        for line in code_result['code'].split('\n')[:5]:
            print(f"    {line}")
        if len(code_result['code'].split('\n')) > 5:
            print(f"    ... ({len(code_result['code'].split(chr(10)))}行)")

        # 迭代审查
        for iteration in range(1, self.max_iterations + 1):
            print(f"\n  --- 迭代 {iteration} ---")

            # Step 2: Reviewer审查
            review = self.reviewer.review(code_result, iteration)
            print(f"  [Reviewer] 总分: {review['total_score']:.2f}")
            print(f"  [Reviewer] 各项: {', '.join(f'{k}={v:.2f}' for k, v in review['scores'].items())}")

            if review['issues']:
                print(f"  [Reviewer] 问题:")
                for issue in review['issues']:
                    print(f"    - {issue}")

            self.review_history.append({
                'iteration': iteration,
                'scores': review['scores'],
                'total_score': review['total_score'],
                'approved': review['approved'],
            })

            # 判断是否通过
            if review['approved']:
                print(f"\n  [通过] 代码审查通过! (得分: {review['total_score']:.2f})")
                break

            # Step 3: Refiner改进
            if iteration < self.max_iterations:
                print(f"  [Refiner] 正在改进代码...")
                code_result = self.refiner.refine(code_result, review)
                print(f"  [Refiner] 改进: {', '.join(code_result['improvements'])}")
        else:
            print(f"\n  [未通过] 达到最大迭代次数 ({self.max_iterations})")

        # 生成审查报告
        self._generate_report(requirement)
        return code_result, review

    def _generate_report(self, requirement):
        """生成审查报告"""
        print(f"\n  {'=' * 50}")
        print(f"  审查报告: {requirement}")
        print(f"  {'=' * 50}")

        for record in self.review_history:
            status = "通过" if record['approved'] else "未通过"
            print(f"  迭代{record['iteration']}: 得分={record['total_score']:.2f} [{status}]")

        print(f"  总迭代次数: {len(self.review_history)}")


# ============================================================
# 3. 运行代码审查系统
# ============================================================
print("=" * 60)
print("2. 运行代码审查系统")
print("=" * 60)

writer = WriterAgent()
reviewer = ReviewerAgent()
refiner = RefinerAgent()
manager = ReviewManager(writer, reviewer, refiner, max_iterations=5, threshold=0.75)

requirements = [
    "编写一个排序算法",
    "实现二分搜索",
]

for req in requirements:
    final_code, final_review = manager.run_review(req)


# ============================================================
# 4. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 5.1 分数随迭代变化
ax1 = axes[0, 0]
if manager.review_history:
    # 按需求分组
    req1_records = manager.review_history[:3]  # 前3次是第一个需求
    req2_records = manager.review_history[3:]  # 后面是第二个需求

    for records, label, color in [
        (req1_records, '排序算法', '#2196F3'),
        (req2_records, '二分搜索', '#4CAF50'),
    ]:
        if records:
            iters = [r['iteration'] for r in records]
            scores = [r['total_score'] for r in records]
            ax1.plot(iters, scores, f'{color[0]}o-', linewidth=2, markersize=10,
                     label=label, color=color)

    ax1.axhline(y=0.75, color='red', linestyle='--', label='通过阈值')
    ax1.set_xlabel('迭代次数')
    ax1.set_ylabel('总得分')
    ax1.set_title('代码质量随迭代变化')
    ax1.legend()
    ax1.set_ylim(0.4, 1.0)
    ax1.grid(True, alpha=0.3)

# 5.2 各维度分数雷达图
ax2 = axes[0, 1]
if manager.review_history:
    first = manager.review_history[0]
    last = manager.review_history[-1]
    dims = list(first['scores'].keys())
    num_dims = len(dims)
    angles = np.linspace(0, 2 * np.pi, num_dims, endpoint=False).tolist()
    angles += angles[:1]

    first_vals = list(first['scores'].values()) + list(first['scores'].values())[:1]
    last_vals = list(last['scores'].values()) + list(last['scores'].values())[:1]

    ax2.plot(angles, first_vals, 'ro-', linewidth=2, label='初始版本')
    ax2.fill(angles, first_vals, alpha=0.1, color='red')
    ax2.plot(angles, last_vals, 'go-', linewidth=2, label='最终版本')
    ax2.fill(angles, last_vals, alpha=0.1, color='green')
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(dims)
    ax2.set_ylim(0, 1)
    ax2.set_title('代码质量雷达图')
    ax2.legend()

# 5.3 Agent工作量统计
ax3 = axes[1, 0]
agent_names = ['Writer', 'Reviewer', 'Refiner']
actions = [
    len(writer.history),
    len(reviewer.history),
    len(refiner.history),
]
colors_agents = ['#4CAF50', '#2196F3', '#FF9800']
ax3.bar(agent_names, actions, color=colors_agents, edgecolor='black')
ax3.set_ylabel('操作次数')
ax3.set_title('各Agent工作量统计')
for i, v in enumerate(actions):
    ax3.text(i, v + 0.1, str(v), ha='center', fontsize=12)

# 5.4 改进效果对比
ax4 = axes[1, 1]
if manager.review_history:
    criteria = list(manager.review_history[0]['scores'].keys())
    initial_scores = list(manager.review_history[0]['scores'].values())
    final_scores = list(manager.review_history[-1]['scores'].values())

    x = np.arange(len(criteria))
    width = 0.35
    ax4.bar(x - width / 2, initial_scores, width, label='初始', color='#FF9800', edgecolor='black')
    ax4.bar(x + width / 2, final_scores, width, label='最终', color='#4CAF50', edgecolor='black')
    ax4.set_xticks(x)
    ax4.set_xticklabels(criteria, fontsize=9)
    ax4.set_ylabel('分数')
    ax4.set_title('代码质量改进对比')
    ax4.legend()
    ax4.set_ylim(0, 1.1)

plt.suptitle('W23-D7: Multi-Agent代码审查系统', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W23/d7_multi_agent_project.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存!")
