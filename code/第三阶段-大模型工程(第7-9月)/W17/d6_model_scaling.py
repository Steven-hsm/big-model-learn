"""
W17-D6 模型缩放定律 (Model Scaling Laws)
==========================================
模型缩放定律(Chinchilla), 计算量估算(FLOPs),
参数量与数据量的平衡, 不同规模模型对比, 常见开源模型列表
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W17-D6 模型缩放定律 (Model Scaling Laws)")
print("=" * 60)

# ============================================================
# 1. 缩放定律概述
# ============================================================
print("\n--- 1. 缩放定律概述 ---")
print("""
  核心发现: 模型性能可预测地随三个因素增长
    1) 模型参数量 (N)
    2) 训练数据量 (D, tokens数)
    3) 计算预算 (C, FLOPs)

  重要论文:
    - Kaplan et al. (2020): "Scaling Laws for Neural Language Models"
      (OpenAI, 最早的缩放定律研究)
    - Hoffmann et al. (2022): "Training Compute-Optimal Large Language Models"
      (DeepMind, Chinchilla论文)

  关键结论:
    损失函数 L(N, D) 近似满足幂律关系:
      L(N) ∝ N^(-α_N)    (α_N ≈ 0.076)
      L(D) ∝ D^(-α_D)    (α_D ≈ 0.095)
      L(C) ∝ C^(-α_C)    (α_C ≈ 0.050)
""")

# ============================================================
# 2. Chinchilla 缩放定律
# ============================================================
print("\n--- 2. Chinchilla 缩放定律 ---")
print("""
  Chinchilla 的核心发现:

  给定计算预算 C (FLOPs), 最优的参数量 N 和数据量 D 应满足:
    N_optimal ∝ C^0.50
    D_optimal ∝ C^0.50

  即: 参数量和数据量应该等比例增长!

  这意味着之前很多模型(GPT-3, GPT-4等)都是"过大而数据不足"
  Chinchilla (70B参数, 1.4T tokens) 超过了 Gopher (280B参数, 300B tokens)

  最优训练token数估算:
    D_optimal ≈ 20 * N    (约20倍参数量的训练数据)
""")

# ============================================================
# 3. FLOPs 计算量估算
# ============================================================
print("\n--- 3. FLOPs 计算量估算 ---")
print("""
  Transformer 前向传播的 FLOPs 估算:

  单层Transformer的 FLOPs ≈ 6 * N * L
    (N = 参数量, L = 序列长度)

  完整训练 FLOPs ≈ 6 * N * D
    (N = 参数量, D = 训练总token数)
    其中:
      6 = 2(前向) + 4(反向, 约为前向的2倍)

  示例:
    GPT-3 (175B参数, 300B tokens):
      FLOPs = 6 * 175e9 * 300e9 = 3.15e23

    LLaMA-7B (7B参数, 1T tokens):
      FLOPs = 6 * 7e9 * 1e12 = 4.2e22

  GPU训练时间估算:
    A100 GPU: ~312 TFLOPS (bf16)
    训练天数 = FLOPs / (GPU数 * TFLOPS * 86400)
""")


def estimate_flops(params_billion, tokens_billion):
    """估算训练FLOPs"""
    return 6 * params_billion * 1e9 * tokens_billion * 1e9


def estimate_training_days(flops, num_gpus, tflops_per_gpu=312):
    """估算训练天数"""
    total_tflops = num_gpus * tflops_per_gpu * 1e12
    seconds = flops / total_tflops
    return seconds / 86400


# 常见模型训练估算
models = [
    ("LLaMA-7B",   7,   1000, 1024),
    ("LLaMA-13B",  13,  1000, 1024),
    ("LLaMA-30B",  30,  1200, 1024),
    ("LLaMA-65B",  65,  1400, 2048),
    ("LLaMA-3-8B", 8,   15000, 2048),
    ("GPT-3",      175, 300,  10240),
]

print(f"\n  {'模型':<15s} {'参数(B)':>8s} {'数据(B tok)':>12s} {'FLOPs':>12s} {'GPU数':>6s} {'训练天数':>10s}")
print(f"  {'-'*15} {'-'*8} {'-'*12} {'-'*12} {'-'*6} {'-'*10}")
for name, params_b, tokens_b, gpus in models:
    flops = estimate_flops(params_b, tokens_b)
    days = estimate_training_days(flops, gpus)
    print(f"  {name:<15s} {params_b:>8d} {tokens_b:>12d} {flops:>12.2e} {gpus:>6d} {days:>10.0f}")

# ============================================================
# 4. 参数量与数据量的平衡
# ============================================================
print("\n--- 4. 参数量与数据量的平衡 ---")
print("""
  根据 Chinchilla 定律:
    最优 token 数 = 20 * 参数量

  过大模型 + 不足数据 = 浪费计算 (如 Gopher: 280B参数, 300B tokens)
  适中模型 + 充足数据 = 计算最优 (如 Chinchilla: 70B参数, 1.4T tokens)
  小模型 + 大量数据 = 数据过拟合风险 (需要更多参数)

  实际中的考虑:
    1) 推理成本: 大模型推理更贵, 所以有些场景倾向小模型+多数据
    2) 数据质量: 高质量数据可以减少数据量需求
    3) 训练稳定性: 大模型训练更不稳定, 需要更多工程优化
""")

# ============================================================
# 5. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 子图1: 缩放定律 - 损失 vs 参数量
ax1 = axes[0, 0]
params_range = np.logspace(7, 12, 100)  # 10M 到 1T
# 模拟损失曲线 (Kaplan缩放定律)
loss_with_data = 2.5 * (params_range / 1e9) ** (-0.076)
# Chinchilla最优
chinchilla_params = np.array([0.7, 1.4, 2.8, 7, 13, 30, 70, 175, 540]) * 1e9
chinchilla_loss = 2.5 * (chinchilla_params / 1e9) ** (-0.076)
model_labels = ['0.7B', '1.4B', '2.8B', '7B', '13B', '30B', '70B', '175B', '540B']

ax1.plot(np.log10(params_range), loss_with_data, 'b-', linewidth=2, label='缩放定律预测')
ax1.scatter(np.log10(chinchilla_params), chinchilla_loss, c='red', s=100, zorder=5, edgecolors='black')
for i, label in enumerate(model_labels):
    ax1.annotate(label, (np.log10(chinchilla_params[i]), chinchilla_loss[i]),
                 fontsize=8, ha='left', va='bottom', xytext=(5, 5), textcoords='offset points')
ax1.set_xlabel('参数量 (log10)', fontsize=12)
ax1.set_ylabel('损失 (Loss)', fontsize=12)
ax1.set_title('Kaplan缩放定律: 损失 vs 参数量', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 子图2: Chinchilla 最优比例
ax2 = axes[0, 1]
compute_budgets = np.logspace(18, 25, 50)  # FLOPs
N_opt = 0.6 * compute_budgets ** 0.50  # 最优参数量
D_opt = 0.3 * compute_budgets ** 0.50  # 最优数据量

ax2.plot(np.log10(compute_budgets), np.log10(N_opt), 'r-', linewidth=2, label='最优参数量 N')
ax2.plot(np.log10(compute_budgets), np.log10(D_opt), 'b-', linewidth=2, label='最优数据量 D (tokens)')

# 标注一些实际模型
actual_models = {
    'GPT-3': (3.15e23, 1.75e11, 3e11),
    'Chinchilla': (5.76e23, 7e10, 1.4e12),
    'LLaMA-7B': (4.2e22, 7e9, 1e12),
}
for name, (c, n, d) in actual_models.items():
    ax2.scatter(np.log10(c), np.log10(n), c='red', s=80, zorder=5, marker='*')
    ax2.scatter(np.log10(c), np.log10(d), c='blue', s=80, zorder=5, marker='*')

ax2.set_xlabel('计算预算 (FLOPs, log10)', fontsize=12)
ax2.set_ylabel('参数量/数据量 (log10)', fontsize=12)
ax2.set_title('Chinchilla缩放定律: 最优N和D', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# 子图3: 不同规模模型性能对比
ax3 = axes[1, 0]
model_sizes = ['1B', '7B', '13B', '30B', '65B', '70B', '175B']
benchmarks = {
    'MMLU (5-shot)':    [25, 35, 46, 56, 64, 68, 52],
    'HumanEval':        [12, 18, 26, 35, 42, 48, 30],
    'GSM8K':            [15, 22, 32, 45, 55, 60, 35],
}

x = np.arange(len(model_sizes))
width = 0.25
for i, (bench, scores) in enumerate(benchmarks.items()):
    ax3.bar(x + i * width, scores, width, label=bench, alpha=0.85, edgecolor='black')

ax3.set_xticks(x + width)
ax3.set_xticklabels(model_sizes, fontsize=10)
ax3.set_ylabel('分数', fontsize=12)
ax3.set_title('不同规模模型基准测试对比 (模拟数据)', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')

# 子图4: 训练成本估算
ax4 = axes[1, 1]
model_names_cost = ['LLaMA-7B', 'LLaMA-13B', 'LLaMA-30B', 'LLaMA-65B',
                     'LLaMA-3-8B', 'GPT-3']
gpu_hours = [
    82432,   # LLaMA-7B
    161792,  # LLaMA-13B
    409600,  # LLaMA-30B
    1023232, # LLaMA-65B
    1300000, # LLaMA-3-8B (估计)
    3499000, # GPT-3 (估计)
]
gpu_hours_k = [h / 1000 for h in gpu_hours]

colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(model_names_cost)))
bars = ax4.bar(range(len(model_names_cost)), gpu_hours_k, color=colors, edgecolor='black', alpha=0.85)
ax4.set_xticks(range(len(model_names_cost)))
ax4.set_xticklabels(model_names_cost, fontsize=9, rotation=15)
ax4.set_ylabel('A100 GPU 小时 (千)', fontsize=12)
ax4.set_title('模型训练成本估算 (A100 GPU小时)', fontsize=14, fontweight='bold')
for bar, val in zip(bars, gpu_hours_k):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
             f'{val:.0f}K', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W17/model_scaling.png', dpi=150, bbox_inches='tight')
print("\n  图表已保存: model_scaling.png")
plt.close()

# ============================================================
# 6. 常见开源模型列表
# ============================================================
print("\n--- 6. 常见开源模型列表 ---")

open_source_models = [
    ("LLaMA-3-8B",         "Meta",     "8B",    "15T",    "开源, 多语言",           "Apache 2.0"),
    ("LLaMA-3-70B",        "Meta",     "70B",   "15T",    "开源新标杆",              "Apache 2.0"),
    ("Mistral-7B",         "Mistral",  "7B",    "8T",     "滑动窗口注意力",           "Apache 2.0"),
    ("Mixtral-8x7B",       "Mistral",  "47B",   "~8T",    "MoE架构",                "Apache 2.0"),
    ("Qwen2.5-7B",         "阿里",     "7B",    "18T",    "中文优秀, 代码强",         "Apache 2.0"),
    ("Qwen2.5-72B",        "阿里",     "72B",   "18T",    "多语言, 推理强",           "Apache 2.0"),
    ("DeepSeek-V3",        "DeepSeek", "671B",  "14.8T",  "MoE, 推理能力突出",        "MIT"),
    ("DeepSeek-R1",        "DeepSeek", "671B",  "-",      "推理链模型",               "MIT"),
    ("Gemma-2-9B",         "Google",   "9B",    "8T",     "轻量高效",                "Gemma"),
    ("Yi-1.5-34B",         "零一万物",  "34B",   "3T",     "中英双语",               "Apache 2.0"),
    ("ChatGLM3-6B",        "智谱AI",   "6B",    "~1T",    "中文对话, 部署友好",        "Apache 2.0"),
    ("InternLM2-20B",      "上海AI",   "20B",   "~2T",    "长上下文",               "Apache 2.0"),
    ("Phi-3-mini",         "Microsoft","3.8B",  "3.3T",   "小模型高性能",             "MIT"),
]

print(f"  {'模型':<20s} {'机构':<10s} {'参数':<8s} {'训练数据':<8s} {'特点':<20s} {'许可证':<12s}")
print(f"  {'-'*20} {'-'*10} {'-'*8} {'-'*8} {'-'*20} {'-'*12}")
for name, org, params, data, feat, lic in open_source_models:
    print(f"  {name:<20s} {org:<10s} {params:<8s} {data:<8s} {feat:<20s} {lic:<12s}")

# ============================================================
# 7. 模型选择指南
# ============================================================
print("\n--- 7. 模型选择指南 ---")
print("""
  选择模型时需要考虑的因素:

  1) 任务类型:
     - 文本生成/对话:  Decoder-only (LLaMA, Qwen, Mistral)
     - 文本理解/分类:  Encoder-only (BERT) 或 Decoder-only
     - 翻译/摘要:      Encoder-Decoder (T5, BART)

  2) 硬件资源:
     - 8GB VRAM:    7B模型 (4-bit量化)
     - 24GB VRAM:   13B模型 (fp16) 或 70B (4-bit量化)
     - 80GB VRAM:   70B模型 (fp16)

  3) 语言需求:
     - 中文为主:    Qwen2.5, ChatGLM3, DeepSeek
     - 英文为主:    LLaMA-3, Mistral, Phi-3
     - 多语言:      LLaMA-3, Qwen2.5

  4) 推理 vs 训练:
     - 只推理: 选开源模型, 量化后部署
     - 要微调: 选支持好的模型 (PEFT/LoRA兼容)
""")

print("\n" + "=" * 60)
print("W17-D6 完成! 本节学习了模型缩放定律和开源模型生态")
print("=" * 60)
