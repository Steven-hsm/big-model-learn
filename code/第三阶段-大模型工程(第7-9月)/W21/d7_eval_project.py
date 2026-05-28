### Day 7（周日）：综合评估项目
# 多维度模型能力评估, 自动+人工评估, 评估报告, 雷达图可视化

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 综合评估框架
# ============================================================
class ModelEvaluationSuite:
    """多维度模型评估套件"""

    def __init__(self, model_name, model_version="1.0"):
        self.model_name = model_name
        self.model_version = model_version
        self.eval_date = datetime.now().strftime("%Y-%m-%d")
        self.dimensions = {}
        self.detail_results = {}

    def add_dimension(self, name, weight=1.0):
        """添加评估维度"""
        self.dimensions[name] = {
            'weight': weight,
            'auto_score': None,
            'human_score': None,
            'sub_scores': {},
        }

    def auto_evaluate(self, dimension, sub_tests):
        """自动评估"""
        scores = {}
        for test_name, test_func in sub_tests.items():
            score = test_func()
            scores[test_name] = score

        avg_score = np.mean(list(scores.values()))
        self.dimensions[dimension]['auto_score'] = avg_score
        self.dimensions[dimension]['sub_scores'].update(scores)
        self.detail_results[dimension] = scores
        return avg_score

    def set_human_score(self, dimension, score):
        """设置人工评估分数"""
        self.dimensions[dimension]['human_score'] = score

    def compute_composite_score(self, method='weighted_avg', auto_weight=0.6, human_weight=0.4):
        """计算综合得分"""
        scores = []
        weights = []
        for dim_name, dim_info in self.dimensions.items():
            auto = dim_info['auto_score'] or 0
            human = dim_info['human_score'] or auto  # 如果没有人工评分, 用自动评分
            combined = auto * auto_weight + human * human_weight
            scores.append(combined)
            weights.append(dim_info['weight'])

        scores = np.array(scores)
        weights = np.array(weights)
        composite = np.average(scores, weights=weights)
        return composite

    def generate_report(self):
        """生成评估报告"""
        print("\n" + "=" * 70)
        print(f"  模型评估报告")
        print(f"  {'=' * 66}")
        print(f"  模型名称: {self.model_name} v{self.model_version}")
        print(f"  评估日期: {self.eval_date}")
        print(f"  {'=' * 66}")

        composite = self.compute_composite_score()

        print(f"\n  {'维度':<12} {'自动评分':>8} {'人工评分':>8} {'综合':>8} {'权重':>6}")
        print(f"  {'-' * 50}")

        for dim_name, dim_info in self.dimensions.items():
            auto_s = dim_info['auto_score'] or 0
            human_s = dim_info['human_score'] or auto_s
            combined = auto_s * 0.6 + human_s * 0.4
            print(f"  {dim_name:<12} {auto_s:>8.2f} {human_s:>8.2f} {combined:>8.2f} "
                  f"{dim_info['weight']:>6.1f}")

            if dim_info['sub_scores']:
                for sub, score in dim_info['sub_scores'].items():
                    print(f"    └─ {sub}: {score:.2f}")

        print(f"\n  {'综合得分':<12} {'':>8} {'':>8} {composite:>8.2f}")
        print(f"  {'=' * 66}")

        # 能力等级
        if composite >= 4.5:
            level = "卓越 (Excellent)"
        elif composite >= 4.0:
            level = "优秀 (Good)"
        elif composite >= 3.0:
            level = "良好 (Fair)"
        elif composite >= 2.0:
            level = "一般 (Below Average)"
        else:
            level = "需改进 (Needs Improvement)"

        print(f"  能力等级: {level}")
        print("=" * 70)

        return composite


# ============================================================
# 2. 定义各维度的测试函数
# ============================================================
print("=" * 60)
print("1. 多维度模型评估系统")
print("=" * 60)

# 知识与推理能力测试
def test_knowledge_accuracy():
    """知识准确性测试"""
    # 模拟: 在不同领域回答问题的准确率
    domains = {'科学': 0.88, '历史': 0.82, '地理': 0.85, '文学': 0.90, '技术': 0.87}
    return np.mean(list(domains.values()))

def test_logical_reasoning():
    """逻辑推理测试"""
    tasks = {'演绎推理': 0.85, '归纳推理': 0.78, '类比推理': 0.82, '因果推理': 0.80}
    return np.mean(list(tasks.values()))

def test_math_ability():
    """数学能力测试"""
    levels = {'算术': 0.95, '代数': 0.85, '几何': 0.80, '概率统计': 0.82, '微积分': 0.75}
    return np.mean(list(levels.values()))

# 语言能力测试
def test_reading_comprehension():
    """阅读理解测试"""
    return 0.88

def test_writing_quality():
    """写作质量测试"""
    return 0.85

def test_translation():
    """翻译能力测试"""
    return 0.83

def test_summarization():
    """摘要能力测试"""
    return 0.86

# 代码能力测试
def test_code_generation():
    """代码生成测试"""
    return 0.90

def test_code_debugging():
    """代码调试测试"""
    return 0.82

def test_code_explanation():
    """代码解释测试"""
    return 0.88

# 安全性测试
def test_harmful_refusal():
    """有害内容拒绝率"""
    return 0.92

def test_bias_fairness():
    """偏见与公平性"""
    return 0.85

def test_privacy_protection():
    """隐私保护"""
    return 0.90

# 实用性测试
def test_instruction_following():
    """指令遵循"""
    return 0.88

def test_creativity():
    """创造性"""
    return 0.82

def test_multi_turn_coherence():
    """多轮对话连贯性"""
    return 0.84


# ============================================================
# 3. 运行评估
# ============================================================
print("\n正在运行自动评估...")

suite = ModelEvaluationSuite("MyLLM", "2.0")

# 添加维度
suite.add_dimension('知识推理', weight=3)
suite.add_dimension('语言能力', weight=2)
suite.add_dimension('代码能力', weight=2)
suite.add_dimension('安全性', weight=3)
suite.add_dimension('实用性', weight=2)

# 运行自动评估
suite.auto_evaluate('知识推理', {
    '知识准确性': test_knowledge_accuracy,
    '逻辑推理': test_logical_reasoning,
    '数学能力': test_math_ability,
})

suite.auto_evaluate('语言能力', {
    '阅读理解': test_reading_comprehension,
    '写作质量': test_writing_quality,
    '翻译能力': test_translation,
    '摘要能力': test_summarization,
})

suite.auto_evaluate('代码能力', {
    '代码生成': test_code_generation,
    '代码调试': test_code_debugging,
    '代码解释': test_code_explanation,
})

suite.auto_evaluate('安全性', {
    '有害拒绝率': test_harmful_refusal,
    '偏见公平性': test_bias_fairness,
    '隐私保护': test_privacy_protection,
})

suite.auto_evaluate('实用性', {
    '指令遵循': test_instruction_following,
    '创造性': test_creativity,
    '多轮连���性': test_multi_turn_coherence,
})

# 设置人工评分 (模拟)
np.random.seed(42)
for dim in suite.dimensions:
    auto_score = suite.dimensions[dim]['auto_score']
    # 人工评分在自动评分附近波动
    human_score = np.clip(auto_score + np.random.normal(0, 0.05), 0, 5)
    suite.set_human_score(dim, human_score)

# 生成报告
composite = suite.generate_report()


# ============================================================
# 4. 多模型对比评估
# ============================================================
print("\n" + "=" * 60)
print("2. 多模型对比评估")
print("=" * 60)

np.random.seed(42)
all_models = ['GPT-4', 'Claude-3', 'MyLLM-v2', 'LLaMA-2-70B', 'Mistral-7B']
all_model_scores = {}

for model in all_models:
    model_suite = ModelEvaluationSuite(model)
    model_suite.add_dimension('知识推理', weight=3)
    model_suite.add_dimension('语言能力', weight=2)
    model_suite.add_dimension('代码能力', weight=2)
    model_suite.add_dimension('安全性', weight=3)
    model_suite.add_dimension('实用性', weight=2)

    for dim in model_suite.dimensions:
        if 'GPT' in model or 'Claude' in model:
            base = np.random.uniform(0.82, 0.95)
        elif 'MyLLM' in model:
            base = np.random.uniform(0.78, 0.90)
        elif '70B' in model:
            base = np.random.uniform(0.70, 0.85)
        else:
            base = np.random.uniform(0.60, 0.80)
        model_suite.dimensions[dim]['auto_score'] = base

    composite = model_suite.compute_composite_score()
    all_model_scores[model] = {
        'composite': composite,
        'dimensions': {d: model_suite.dimensions[d]['auto_score']
                       for d in model_suite.dimensions},
    }
    print(f"  {model}: 综合得分 = {composite:.3f}")


# ============================================================
# 5. 可视化
# ============================================================
fig = plt.figure(figsize=(16, 14))

# 5.1 雷达图: 单模型多维度
ax1 = fig.add_subplot(2, 2, 1, polar=True)
dim_names = list(suite.dimensions.keys())
num_dims = len(dim_names)
angles = np.linspace(0, 2 * np.pi, num_dims, endpoint=False).tolist()
angles += angles[:1]

auto_values = [suite.dimensions[d]['auto_score'] for d in dim_names]
human_values = [suite.dimensions[d]['human_score'] for d in dim_names]
auto_values += auto_values[:1]
human_values += human_values[:1]

ax1.plot(angles, auto_values, 'ro-', linewidth=2, label='自动评估')
ax1.fill(angles, auto_values, alpha=0.15, color='red')
ax1.plot(angles, human_values, 'bo-', linewidth=2, label='人工评估')
ax1.fill(angles, human_values, alpha=0.15, color='blue')
ax1.set_xticks(angles[:-1])
ax1.set_xticklabels(dim_names)
ax1.set_ylim(0, 1)
ax1.set_title(f'{suite.model_name} 多维度评估', pad=20)
ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

# 5.2 雷达图: 多模型对比
ax2 = fig.add_subplot(2, 2, 2, polar=True)
colors = ['#E91E63', '#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
for i, model in enumerate(all_models):
    values = list(all_model_scores[model]['dimensions'].values())
    values += values[:1]
    ax2.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
    ax2.fill(angles, values, alpha=0.05, color=colors[i])

ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(dim_names)
ax2.set_ylim(0, 1)
ax2.set_title('多模型能力对比雷达图', pad=20)
ax2.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))

# 5.3 综合得分对比
ax3 = fig.add_subplot(2, 2, 3)
models_sorted = sorted(all_model_scores.keys(),
                       key=lambda m: all_model_scores[m]['composite'], reverse=True)
scores_sorted = [all_model_scores[m]['composite'] for m in models_sorted]
bar_colors = ['#4CAF50' if s >= 0.85 else '#FF9800' if s >= 0.75 else '#F44336'
              for s in scores_sorted]
bars = ax3.barh(models_sorted, scores_sorted, color=bar_colors, edgecolor='black')
ax3.set_xlabel('综合得分')
ax3.set_title('模型综合得分排名')
ax3.set_xlim(0, 1)
for bar, score in zip(bars, scores_sorted):
    ax3.text(score + 0.01, bar.get_y() + bar.get_height() / 2,
             f'{score:.3f}', va='center')

# 5.4 子维度热力图
ax4 = fig.add_subplot(2, 2, 4)
dim_names_all = list(suite.dimensions.keys())
models_top = all_models
score_matrix = np.array(
    [[all_model_scores[m]['dimensions'][d] for d in dim_names_all] for m in models_top]
)
im = ax4.imshow(score_matrix, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=1.0)
ax4.set_xticks(range(len(dim_names_all)))
ax4.set_xticklabels(dim_names_all)
ax4.set_yticks(range(len(models_top)))
ax4.set_yticklabels(models_top)
ax4.set_title('各维度评分热力图')
for i in range(len(models_top)):
    for j in range(len(dim_names_all)):
        ax4.text(j, i, f'{score_matrix[i, j]:.2f}', ha='center', va='center',
                 fontsize=10, fontweight='bold')
plt.colorbar(im, ax=ax4)

plt.suptitle('W21-D7: 综合模型评估项目报告', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W21/d7_eval_project.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存!")
