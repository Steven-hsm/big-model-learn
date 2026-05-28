"""
W20-D1 微调方法概览 (Finetuning Overview)
==========================================
微调方法全景(全量微调/冻结/Adapter/Prompt Tuning/LoRA),
参数量对比, 各方法优劣分析
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W20-D1 微调方法概览 (Finetuning Overview)")
print("=" * 60)

# ============================================================
# 1. 为什么需要微调
# ============================================================
print("\n--- 1. 为什么需要微调 ---")
print("""
  预训练模型 vs 微调:
    预训练: 在大规模语料上学习通用知识 (如GPT, BERT)
    微调:   在特定任务数据上继续训练, 适应特定需求

  微调解决的问题:
    1) 领域适应: 让通用模型理解专业领域 (医疗/法律/金融)
    2) 任务定制: 让模型执行特定任务 (分类/生成/QA)
    3) 风格调整: 调整输出风格和格式
    4) 知识注入: 注入新知识 (如公司内部文档)

  挑战:
    - 全量微调需要大量GPU内存 (7B模型需~28GB fp16)
    - 灾难性遗忘: 微调可能损失预训练知识
    - 数据需求: 需要标注数据
""")

# ============================================================
# 2. 微调方法全景
# ============================================================
print("\n--- 2. 微调方法全景 ---")
print("""
  +--------------------+------------------+------------------+
  |      方法          |  可训练参数占比  |    核心思想      |
  +--------------------+------------------+------------------+
  | 全量微调(Full FT)  | 100%            | 更新所有参数      |
  | 冻结微调(Freeze)   | 10-30%          | 冻结底层, 只训练顶层|
  | Adapter            | 1-5%            | 在层间插入小网络  |
  | Prompt Tuning      | 0.01-0.1%       | 学习软提示向量    |
  | Prefix Tuning      | 0.1-1%          | 学习前缀向量      |
  | LoRA               | 0.1-2%          | 低秩分解适配      |
  | QLoRA              | 0.1-2%          | 量化+LoRA        |
  | IA3                | 0.01-0.1%       | 向量缩放适配      |
  +--------------------+------------------+------------------+
""")

# ============================================================
# 3. 各方法详解
# ============================================================
print("\n--- 3. 各方法详解 ---")

methods = {
    "全量微调 (Full Fine-tuning)": """
    所有参数都参与训练, 效果最好但成本最高
    - 参数量: 100%
    - 内存: 完整模型 + 梯度 + 优化器状态
    - 7B模型: ~28GB (fp16) + ~56GB (梯度+优化器)
    - 适用: 数据充足, GPU资源充足
    """,

    "冻结微调 (Feature Extraction / Freeze)": """
    冻结大部分层, 只训练最后几层或新加的分类头
    - 参数量: 10-30% (只训练顶层)
    - 内存: 显著减少
    - 适用: 小数据集, 快速实验
    """,

    "Adapter Tuning": """
    在Transformer每层中插入小型瓶颈网络(Adapter)
    Adapter结构: Linear(down) -> ReLU -> Linear(up) -> 残差连接
    - 参数量: 1-5% (取决于bottleneck大小)
    - 优势: 可以为每个任务训练不同的Adapter, 共享主模型
    - 缺点: 增加推理延迟 (额外的Adapter层)
    """,

    "Prompt Tuning": """
    在输入前添加可学习的连续向量(软提示)
    冻结模型参数, 只训练这些提示向量
    - 参数量: 0.01-0.1% (几十到几百个token)
    - 优势: 极少参数, 可以为每个任务保存不同的提示
    - 缺点: 随模型规模增大效果才变好
    """,

    "Prefix Tuning": """
    在Transformer每层的输入前添加可学习的前缀向量
    比Prompt Tuning更灵活 (每层都有)
    - 参数量: 0.1-1%
    - 优势: 比Prompt Tuning效果更好
    - 缺点: 略多于Prompt Tuning
    """,

    "LoRA (Low-Rank Adaptation)": """
    将权重更新分解为低秩矩阵: W_new = W + BA
    B (d x r), A (r x d), r << d
    - 参数量: 0.1-2% (取决于秩r)
    - 优势: 不增加推理延迟, 可合并回原权重
    - 缺点: 秩r的选择影响效果
    """,

    "QLoRA": """
    先将模型量化为4-bit, 然后应用LoRA
    结合了量化和LoRA的优势
    - 参数量: 0.1-2% (LoRA部分)
    - 内存: 大幅减少 (7B模型~6GB即可)
    - 量化: NF4量化 + 双重量化 + 分页优化
    - 适用: 消费级GPU微调大模型
    """,

    "IA3 (Injected Attention and Adapter)": """
    通过学习向量对Key/Value/FFN进行缩放
    比LoRA更少的参数
    - 参数量: 0.01-0.1%
    - 优势: 极少参数, 效果接近LoRA
    """,
}

for name, desc in methods.items():
    print(f"\n  {name}")
    print(desc)

# ============================================================
# 4. 参数量对比
# ============================================================
print("\n--- 4. 参数量对比 ---")

# 以LLaMA-7B为例
model_params = 7e9  # 7B参数
hidden_dim = 4096
num_layers = 32

print(f"  基准模型: LLaMA-7B")
print(f"  总参数: {model_params/1e9:.1f}B")
print(f"  隐藏维度: {hidden_dim}")
print(f"  层数: {num_layers}")

# 各方法的可训练参数估算
method_params = {
    "全量微调":       model_params,
    "冻结(顶层3层)":   model_params * 0.15,
    "Adapter (b=64)":  num_layers * 2 * (hidden_dim * 64 + 64 * hidden_dim),
    "Prompt Tuning":   20 * hidden_dim,  # 20个虚拟token
    "Prefix Tuning":   num_layers * 2 * 10 * hidden_dim,  # 10个前缀
    "LoRA (r=8)":      num_layers * 2 * hidden_dim * 8 * 2,  # Q,V都加
    "LoRA (r=16)":     num_layers * 2 * hidden_dim * 16 * 2,
    "LoRA (r=64)":     num_layers * 2 * hidden_dim * 64 * 2,
    "QLoRA (r=16)":    num_layers * 2 * hidden_dim * 16 * 2,  # 同LoRA
    "IA3":             num_layers * 3 * hidden_dim,
}

print(f"\n  {'方法':<22s} {'可训练参数':>15s} {'占总参数比例':>15s}")
print(f"  {'-'*22} {'-'*15} {'-'*15}")
for name, params in method_params.items():
    ratio = params / model_params * 100
    if params >= 1e9:
        print(f"  {name:<22s} {params/1e9:>12.2f}B {ratio:>12.4f}%")
    elif params >= 1e6:
        print(f"  {name:<22s} {params/1e6:>12.2f}M {ratio:>12.4f}%")
    else:
        print(f"  {name:<22s} {params:>12.0f}  {ratio:>12.4f}%")

# ============================================================
# 5. GPU内存需求对比
# ============================================================
print("\n--- 5. GPU内存需求对比 ---")
print("""
  模型参数存储:
    fp32: 4 bytes/参数
    fp16: 2 bytes/参数
    int8: 1 byte/参数
    int4: 0.5 bytes/参数

  训练额外开销:
    梯度:      与参数同大小
    优化器状态: Adam需要2倍参数大小 (fp32)
""")

# 内存估算
memory_estimates = {
    "全量微调 (fp16)":    7 * 2 + 7 * 2 + 7 * 4 * 2,  # 模型+梯度+优化器
    "冻结微调 (fp16)":    7 * 2 + 1 * 2 + 1 * 4 * 2,  # 只训练1B参数
    "LoRA r=16 (fp16)":  7 * 2 + 0.02 * 2 + 0.02 * 4 * 2,  # LoRA参数很少
    "QLoRA (4-bit)":     7 * 0.5 + 0.02 * 2 + 0.02 * 4 * 2,  # 4bit模型
}

print(f"\n  {'方法':<25s} {'估算内存 (GB)':>15s} {'所需GPU':>15s}")
print(f"  {'-'*25} {'-'*15} {'-'*15}")
for name, mem_gb in memory_estimates.items():
    if mem_gb > 40:
        gpu = "A100 80GB x2"
    elif mem_gb > 24:
        gpu = "A100 40GB / A6000"
    elif mem_gb > 16:
        gpu = "V100 32GB"
    elif mem_gb > 8:
        gpu = "RTX 3090/4090"
    else:
        gpu = "RTX 3060/4060"
    print(f"  {name:<25s} {mem_gb:>12.1f}GB {gpu:>15s}")

# ============================================================
# 6. 可视化
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 子图1: 可训练参数对比 (对数坐标)
ax1 = axes[0]
names_short = ['Full FT', 'Freeze', 'Adapter', 'Prompt\nTune', 'Prefix\nTune', 'LoRA\nr=8', 'LoRA\nr=16', 'IA3']
param_values = [method_params[k] for k in method_params]
param_ratios = [p / model_params * 100 for p in param_values]

colors = ['#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#3498db', '#2ecc71', '#27ae60', '#e67e22']
bars = ax1.bar(range(len(names_short)), [np.log10(max(p, 1)) for p in param_values],
               color=colors, edgecolor='black', alpha=0.85)
ax1.set_xticks(range(len(names_short)))
ax1.set_xticklabels(names_short, fontsize=9)
ax1.set_ylabel('参数量 (log10)', fontsize=12)
ax1.set_title('可训练参数量对比 (LLaMA-7B)', fontsize=14, fontweight='bold')
for bar, ratio in zip(bars, param_ratios):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{ratio:.3f}%', ha='center', fontsize=8, rotation=30)
ax1.grid(True, alpha=0.3, axis='y')

# 子图2: GPU内存对比
ax2 = axes[1]
mem_names = ['Full FT\n(fp16)', 'Freeze\n(fp16)', 'LoRA r=16\n(fp16)', 'QLoRA\n(4-bit)']
mem_values = list(memory_estimates.values())
gpu_limits = [8, 12, 16, 24, 40, 80]

colors_mem = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
bars = ax2.barh(range(len(mem_names)), mem_values, color=colors_mem, edgecolor='black', alpha=0.85)
ax2.set_yticks(range(len(mem_names)))
ax2.set_yticklabels(mem_names, fontsize=10)
ax2.set_xlabel('GPU内存 (GB)', fontsize=12)
ax2.set_title('GPU内存需求对比 (LLaMA-7B)', fontsize=14, fontweight='bold')
for bar, val in zip(bars, mem_values):
    ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}GB', va='center', fontsize=10)
# 添加GPU内存线
for limit in gpu_limits:
    ax2.axvline(x=limit, color='gray', linestyle=':', alpha=0.5)
ax2.grid(True, alpha=0.3, axis='x')

# 子图3: 效果 vs 成本散点图
ax3 = axes[2]
method_names_plot = ['Full FT', 'Freeze', 'Adapter', 'Prompt', 'Prefix', 'LoRA', 'QLoRA', 'IA3']
# 模拟数据: x=参数比例(对数), y=效果分数
cost_x = [100, 15, 3, 0.05, 0.5, 0.5, 0.5, 0.05]
quality_y = [95, 80, 88, 72, 85, 90, 89, 86]
bubble_size = [300, 150, 100, 50, 80, 120, 100, 60]
colors_scatter = plt.cm.Set2(np.linspace(0, 1, len(method_names_plot)))

for i, (name, x, y, s) in enumerate(zip(method_names_plot, cost_x, quality_y, bubble_size)):
    ax3.scatter(np.log10(x), y, s=s, c=[colors_scatter[i]], edgecolors='black',
                linewidth=1, alpha=0.8, zorder=5)
    ax3.annotate(name, (np.log10(x), y), fontsize=8, ha='center', va='bottom',
                 xytext=(0, 8), textcoords='offset points')

ax3.set_xlabel('可训练参数比例 (log10 %)', fontsize=12)
ax3.set_ylabel('任务效果分数', fontsize=12)
ax3.set_title('效果 vs 成本 散点图', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W20/finetuning_overview.png', dpi=150, bbox_inches='tight')
print("\n  图表已保存: finetuning_overview.png")
plt.close()

# ============================================================
# 7. 方法选择指南
# ============================================================
print("\n--- 7. 方法选择指南 ---")
print("""
  +------------------+------------------+------------------+
  |    GPU资源       |    推荐方法      |    预期效果      |
  +------------------+------------------+------------------+
  | < 8GB            | QLoRA r=8        | 良好             |
  | 8-16GB           | QLoRA r=16       | 很好             |
  | 16-24GB          | LoRA r=16        | 优秀             |
  | 24-48GB          | LoRA r=64        | 接近全量微调     |
  | > 48GB           | 全量微调         | 最佳             |
  +------------------+------------------+------------------+

  2024年推荐默认选择: LoRA 或 QLoRA
    - 效果接近全量微调
    - 内存需求大幅降低
    - 推理时无额外开销
    - 社区支持最广泛
""")

print("\n" + "=" * 60)
print("W20-D1 完成! 本节全面介绍了模型微调的各种方法")
print("=" * 60)
