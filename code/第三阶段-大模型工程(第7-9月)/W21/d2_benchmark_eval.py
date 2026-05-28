### Day 2（周二）：常见Benchmark与自动评估Pipeline
# MMLU/HumanEval/GSM8K概念, 评估pipeline实现, 结果分析可视化

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 常见Benchmark介绍与模拟
# ============================================================
print("=" * 60)
print("1. 常见LLM评估Benchmark")
print("=" * 60)

benchmarks = {
    'MMLU': {
        'description': 'Massive Multitask Language Understanding',
        'subjects': ['数学', '历史', '法律', '医学', '计算机', '哲学', '经济', '物理'],
        'format': '多项选择题 (4选1)',
        'metrics': '准确率 (Accuracy)',
    },
    'HumanEval': {
        'description': '代码生成能力评估',
        'subjects': ['Python编程'],
        'format': '函数补全 + 单元测试',
        'metrics': 'Pass@k (代码通过率)',
    },
    'GSM8K': {
        'description': '小学数学应用题',
        'subjects': ['数学推理'],
        'format': '自由生成的数学题',
        'metrics': '准确率 (Accuracy)',
    },
}

for name, info in benchmarks.items():
    print(f"\n  {name}: {info['description']}")
    print(f"    领域: {', '.join(info['subjects'])}")
    print(f"    格式: {info['format']}")
    print(f"    指标: {info['metrics']}")


# ============================================================
# 2. 模拟MMLU评估
# ============================================================
def simulate_mmlu_eval(model_name, num_questions=100, seed=None):
    """模拟MMLU评估: 多选题准确率"""
    if seed is not None:
        np.random.seed(seed)

    subjects = ['STEM', '人文', '社科', '其他']
    results = {}
    for subject in subjects:
        # 不同模型在不同领域的表现不同
        if 'GPT' in model_name:
            base_acc = np.random.uniform(0.70, 0.90)
        elif 'LLaMA' in model_name:
            base_acc = np.random.uniform(0.55, 0.80)
        else:
            base_acc = np.random.uniform(0.40, 0.65)

        correct = int(num_questions * base_acc)
        results[subject] = {
            'correct': correct,
            'total': num_questions,
            'accuracy': correct / num_questions,
        }
    return results


print("\n" + "=" * 60)
print("2. 模拟MMLU评估结果")
print("=" * 60)

models_mmlu = ['GPT-4', 'LLaMA-2-70B', 'LLaMA-2-7B']
mmlu_results = {}
for model in models_mmlu:
    mmlu_results[model] = simulate_mmlu_eval(model, seed=42)
    avg = np.mean([v['accuracy'] for v in mmlu_results[model].values()])
    print(f"\n  {model} (平均准确率: {avg:.2%})")
    for subj, res in mmlu_results[model].items():
        print(f"    {subj}: {res['accuracy']:.2%} ({res['correct']}/{res['total']})")


# ============================================================
# 3. 模拟HumanEval评估 (Pass@k)
# ============================================================
def compute_pass_at_k(n, c, k):
    """计算Pass@k: n个问题中c个通过, k次尝试中至少通过一次的概率"""
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def simulate_humaneval(model_name, num_problems=164, seed=None):
    """模拟HumanEval评估"""
    if seed is not None:
        np.random.seed(seed)

    pass_rates = {'pass@1': 0, 'pass@5': 0, 'pass@10': 0}
    if 'GPT' in model_name:
        pass_rates['pass@1'] = np.random.uniform(0.65, 0.88)
    elif '70B' in model_name:
        pass_rates['pass@1'] = np.random.uniform(0.45, 0.65)
    else:
        pass_rates['pass@1'] = np.random.uniform(0.15, 0.35)

    pass_rates['pass@5'] = min(pass_rates['pass@1'] + np.random.uniform(0.10, 0.20), 1.0)
    pass_rates['pass@10'] = min(pass_rates['pass@1'] + np.random.uniform(0.15, 0.30), 1.0)

    return pass_rates


print("\n" + "=" * 60)
print("3. 模拟HumanEval评估 (Pass@k)")
print("=" * 60)

humaneval_results = {}
for model in models_mmlu:
    humaneval_results[model] = simulate_humaneval(model, seed=42)
    print(f"\n  {model}:")
    for k, v in humaneval_results[model].items():
        print(f"    {k}: {v:.2%}")


# ============================================================
# 4. 模拟GSM8K评估
# ============================================================
def simulate_gsm8k(model_name, num_problems=1319, seed=None):
    """模拟GSM8K数学推理评估"""
    if seed is not None:
        np.random.seed(seed)

    if 'GPT' in model_name:
        acc = np.random.uniform(0.85, 0.95)
    elif '70B' in model_name:
        acc = np.random.uniform(0.55, 0.70)
    else:
        acc = np.random.uniform(0.15, 0.40)

    correct = int(num_problems * acc)
    return {
        'correct': correct,
        'total': num_problems,
        'accuracy': correct / num_problems,
        'error_types': {
            '计算错误': np.random.randint(30, 80),
            '推理错误': np.random.randint(40, 100),
            '理解错误': np.random.randint(10, 40),
        },
    }


print("\n" + "=" * 60)
print("4. 模拟GSM8K数学推理评估")
print("=" * 60)

gsm8k_results = {}
for model in models_mmlu:
    gsm8k_results[model] = simulate_gsm8k(model, seed=42)
    r = gsm8k_results[model]
    print(f"\n  {model}: 准确率={r['accuracy']:.2%} ({r['correct']}/{r['total']})")
    print(f"    错误分布:")
    for err_type, count in r['error_types'].items():
        print(f"      {err_type}: {count}")


# ============================================================
# 5. 自动评估Pipeline
# ============================================================
class EvaluationPipeline:
    """自动评估Pipeline框架"""

    def __init__(self, model_name):
        self.model_name = model_name
        self.results = {}

    def add_benchmark(self, name, eval_func, **kwargs):
        """添加benchmark评估任务"""
        print(f"  [{self.model_name}] 正在评估 {name}...")
        self.results[name] = eval_func(self.model_name, **kwargs)
        return self

    def run_all(self):
        """运行所有评估"""
        print(f"\n{'=' * 50}")
        print(f"运行评估Pipeline: {self.model_name}")
        print(f"{'=' * 50}")
        return self

    def generate_report(self):
        """生成评估报告"""
        print(f"\n{'=' * 50}")
        print(f"评估报告: {self.model_name}")
        print(f"{'=' * 50}")
        for name, result in self.results.items():
            print(f"  {name}: {result}")
        return self.results


print("\n" + "=" * 60)
print("5. 自动评估Pipeline示例")
print("=" * 60)

pipeline = (
    EvaluationPipeline("GPT-4")
    .run_all()
    .add_benchmark("MMLU", simulate_mmlu_eval, seed=42)
    .add_benchmark("HumanEval", simulate_humaneval, seed=42)
    .add_benchmark("GSM8K", simulate_gsm8k, seed=42)
)

report = pipeline.generate_report()


# ============================================================
# 6. 可视化: Benchmark评估结果
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 6.1 MMLU各领域准确率
ax1 = axes[0, 0]
subjects = list(mmlu_results['GPT-4'].keys())
x = np.arange(len(subjects))
width = 0.25
for i, model in enumerate(models_mmlu):
    accs = [mmlu_results[model][s]['accuracy'] for s in subjects]
    ax1.bar(x + i * width, accs, width, label=model, edgecolor='black')
ax1.set_ylabel('准确率')
ax1.set_title('MMLU各领域准确率对比')
ax1.set_xticks(x + width)
ax1.set_xticklabels(subjects)
ax1.legend()
ax1.set_ylim(0, 1.05)

# 6.2 HumanEval Pass@k
ax2 = axes[0, 1]
pass_k = ['pass@1', 'pass@5', 'pass@10']
for i, model in enumerate(models_mmlu):
    values = [humaneval_results[model][k] for k in pass_k]
    ax2.plot(pass_k, values, 'o-', label=model, linewidth=2, markersize=8)
ax2.set_ylabel('通过率')
ax2.set_title('HumanEval Pass@k 对比')
ax2.legend()
ax2.set_ylim(0, 1.05)

# 6.3 GSM8K错误类型分布
ax3 = axes[1, 0]
error_types = list(gsm8k_results['GPT-4']['error_types'].keys())
for i, model in enumerate(models_mmlu):
    errors = [gsm8k_results[model]['error_types'][e] for e in error_types]
    ax3.bar(np.arange(len(error_types)) + i * width, errors, width,
            label=model, edgecolor='black')
ax3.set_ylabel('错误数量')
ax3.set_title('GSM8K错误类型分布')
ax3.set_xticks(np.arange(len(error_types)) + width)
ax3.set_xticklabels(error_types)
ax3.legend()

# 6.4 综合能力对比雷达图
ax4 = axes[1, 1]
categories = ['MMLU-STEM', 'MMLU-人文', 'MMLU-社科', 'HumanEval', 'GSM8K', '综合']
num_cats = len(categories)
angles = np.linspace(0, 2 * np.pi, num_cats, endpoint=False).tolist()
angles += angles[:1]

for model in models_mmlu:
    values = [
        mmlu_results[model]['STEM']['accuracy'],
        mmlu_results[model]['人文']['accuracy'],
        mmlu_results[model]['社科']['accuracy'],
        humaneval_results[model]['pass@1'],
        gsm8k_results[model]['accuracy'],
        np.mean([
            np.mean([v['accuracy'] for v in mmlu_results[model].values()]),
            humaneval_results[model]['pass@1'],
            gsm8k_results[model]['accuracy'],
        ]),
    ]
    values += values[:1]
    ax4.plot(angles, values, 'o-', label=model, linewidth=2)
    ax4.fill(angles, values, alpha=0.1)

ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(categories)
ax4.set_ylim(0, 1)
ax4.set_title('模型综合能力雷达图')
ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

plt.suptitle('W21-D2: Benchmark评估结果分析', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W21/d2_benchmark_eval.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存!")
