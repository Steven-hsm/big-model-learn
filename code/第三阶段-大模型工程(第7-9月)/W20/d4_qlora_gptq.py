"""
W20-D4 QLoRA与量化 (QLoRA & GPTQ)
===================================
QLoRA概念(4-bit量化+LoRA), GPTQ量化原理,
bitsandbytes用法, 不同量化级别对比
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W20-D4 QLoRA与量化 (QLoRA & GPTQ)")
print("=" * 60)

# ============================================================
# 1. 量化概述
# ============================================================
print("\n--- 1. 量化概述 ---")
print("""
  量化 (Quantization): 降低模型参数的精度
    fp32 (32-bit) -> fp16 (16-bit) -> int8 (8-bit) -> int4 (4-bit)

  为什么需要量化?
    - 减少内存: 7B模型 fp32=28GB, int4=3.5GB
    - 加速推理: 整数运算比浮点快
    - 降低成本: 消费级GPU也能运行大模型

  量化类型:
    1) 训练后量化 (PTQ): 训练完成后量化, 不需要重新训练
    2) 量化感知训练 (QAT): 训练时模拟量化, 精度损失更小
    3) QLoRA量化: 边量化边LoRA训练
""")

# ============================================================
# 2. 量化原理
# ============================================================
print("\n--- 2. 量化原理 ---")
print("""
  线性量化公式:
    q = round(x / scale + zero_point)
    x_dequant = (q - zero_point) * scale

  scale = (x_max - x_min) / (q_max - q_min)
  zero_point = round(q_min - x_min / scale)

  示例 (float -> int8):
    范围 [-3.0, 3.0] -> [-128, 127]
    scale = 6.0 / 255 ≈ 0.0235
    value 1.5 -> round(1.5 / 0.0235) = 64
    dequant: 64 * 0.0235 = 1.504

  NF4 (NormalFloat4) - QLoRA使用:
    - 假设权重服从正态分布
    - 非均匀量化: 信息密度高的区域更多量化级别
    - 比 uniform int4 精度更高
""")


def quantize_linear(values, n_bits=8):
    """线性量化"""
    v_min, v_max = values.min(), values.max()
    q_min, q_max = 0, 2**n_bits - 1

    scale = (v_max - v_min) / (q_max - q_min)
    zero_point = round(q_min - v_min / scale)

    # 量化
    quantized = np.clip(np.round(values / scale + zero_point), q_min, q_max).astype(np.int32)

    # 反量化
    dequantized = (quantized - zero_point) * scale

    # 误差
    error = np.mean((values - dequantized) ** 2)

    return quantized, dequantized, scale, zero_point, error


# 演示不同位宽的量化
np.random.seed(42)
original = np.random.randn(1000).astype(np.float32)  # 模拟权重

print(f"\n  量化对比 (1000个权重值):")
print(f"  {'位宽':>6s} {'每参数字节':>10s} {'总大小':>10s} {'MSE误差':>12s} {'相对误差':>10s}")
print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*12} {'-'*10}")

for n_bits, name in [(32, "fp32"), (16, "fp16"), (8, "int8"), (4, "int4")]:
    if n_bits == 32:
        mse = 0
        size = original.nbytes
    elif n_bits == 16:
        q = original.astype(np.float16).astype(np.float32)
        mse = np.mean((original - q) ** 2)
        size = len(original) * 2
    else:
        _, _, _, _, mse = quantize_linear(original, n_bits)
        size = len(original) * n_bits // 8

    rel_error = mse / np.var(original) * 100
    print(f"  {name:>6s} {n_bits/8:>8.1f}B {size:>8d}B {mse:>12.6f} {rel_error:>9.3f}%")

# ============================================================
# 3. GPTQ 量化原理
# ============================================================
print("\n--- 3. GPTQ 量化原理 ---")
print("""
  GPTQ (Frantar et al., 2022): 训练后量化方法

  核心思想:
    逐列量化权重, 并调整未量化列来补偿量化误差

  算法流程:
    1) 计算Hessian矩阵 H = 2 * X^T * X (使用校准数据)
    2) 按列依次量化:
       - 量化当前列: w_q = quantize(w_col)
       - 计算误差: δ = w_col - dequantize(w_q)
       - 调整后续列: W_rest -= δ * (H_col_rest / H_col_col)
    3) 使用Cholesky分解加速Hessian逆的计算

  特点:
    - 需要128-1024条校准数据
    - 量化后精度接近fp16
    - 适合推理部署
    - 支持3-bit, 4-bit, 8-bit

  与QLoRA的区别:
    GPTQ:  纯量化, 用于推理加速
    QLoRA: 量化 + LoRA训练, 用于参数高效微调
""")


def simple_gptq_demo(weights, hessian, block_size=4, n_bits=4):
    """简化GPTQ演示"""
    d = weights.shape[0]
    quantized = np.zeros_like(weights)
    errors = np.zeros_like(weights)

    q_max = 2 ** (n_bits - 1) - 1
    q_min = -(2 ** (n_bits - 1))

    for i in range(0, d, block_size):
        # 量化当前block
        for j in range(i, min(i + block_size, d)):
            col = weights[:, j]
            scale = max(abs(col.max()), abs(col.min())) / q_max
            q = np.clip(np.round(col / scale), q_min, q_max).astype(np.int32)
            dq = q * scale
            quantized[:, j] = dq

            # 误差
            err = (col - dq) / hessian[j, j]
            errors[:, j] = col - dq

            # 调整后续列
            if j + 1 < d:
                weights[:, j+1:] -= err[:, np.newaxis] * hessian[j, j+1:]

    return quantized, errors


# 简单GPTQ演示
np.random.seed(42)
W = np.random.randn(16, 16) * 0.5
X = np.random.randn(32, 16)
H = 2 * X.T @ X + 1e-6 * np.eye(16)  # Hessian + 正则化

W_q, errs = simple_gptq_demo(W.copy(), H, n_bits=4)
print(f"\n  简化GPTQ演示:")
print(f"  原始权重范数: {np.linalg.norm(W):.4f}")
print(f"  量化误差范数: {np.linalg.norm(W - W_q):.4f}")
print(f"  相对误差: {np.linalg.norm(W - W_q) / np.linalg.norm(W) * 100:.2f}%")

# ============================================================
# 4. QLoRA 概念
# ============================================================
print("\n--- 4. QLoRA 概念 ---")
print("""
  QLoRA = 4-bit Quantization + LoRA (Dettmers et al., 2023)

  三项创新:
    1) NF4量化:      基于正态分布的4-bit量化
    2) 双重量化:     量化参数(scale/zero_point)本身也量化, 节省~0.4bit/param
    3) 分页优化器:   利用CPU内存处理优化器状态, 解决GPU内存溢出

  工作流程:
    1) 加载预训练模型, 量化为NF4
    2) 添加LoRA适配器 (fp16)
    3) 前向时: 反量化权重 -> 计算 -> 只更新LoRA参数
    4) LoRA参数保持高精度 (bf16/fp16)

  内存对比 (LLaMA-7B):
    全量微调 fp16:    ~28GB
    LoRA fp16:        ~16GB
    QLoRA 4-bit:      ~6GB  <-- 消费级GPU可用!

  精度对比:
    全量微调:         基准
    LoRA fp16:        接近基准 (-0.5%)
    QLoRA 4-bit:      接近LoRA (-0.3%)
""")

# ============================================================
# 5. bitsandbytes 用法 (可选)
# ============================================================
print("\n--- 5. bitsandbytes 用法 (可选) ---")
print("""
  bitsandbytes: 量化的核心库

  # 4-bit量化加载模型
  from transformers import BitsAndBytesConfig
  bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",        # NF4量化
      bnb_4bit_compute_dtype=torch.bfloat16,  # 计算精度
      bnb_4bit_use_double_quant=True,    # 双重量化
  )
  model = AutoModelForCausalLM.from_pretrained("model_name", quantization_config=bnb_config)

  # 8-bit量化
  model = AutoModelForCausalLM.from_pretrained("model_name", load_in_8bit=True)

  # QLoRA完整流程
  from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
  model = prepare_model_for_kbit_training(model)  # 准备量化模型
  lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
  model = get_peft_model(model, lora_config)
""")

# ============================================================
# 6. 可视化
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 子图1: 量化精度对比
ax1 = axes[0]
quant_methods = ['FP32', 'FP16', 'INT8', 'NF4\n(QLoRA)', 'INT4\n(uniform)']
accuracy = [100, 99.8, 99.5, 98.5, 96.0]
memory = [28, 14, 7, 3.5, 3.5]  # GB for 7B model

colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6']
bars = ax1.bar(range(len(quant_methods)), accuracy, color=colors, edgecolor='black', alpha=0.85)

ax1_twin = ax1.twinx()
ax1_twin.plot(range(len(quant_methods)), memory, 'r^-', linewidth=2, markersize=10, label='内存(GB)')
ax1_twin.set_ylabel('内存占用 (GB, 7B模型)', fontsize=12, color='red')

for bar, acc in zip(bars, accuracy):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{acc:.1f}%', ha='center', fontsize=10, fontweight='bold')

ax1.set_xticks(range(len(quant_methods)))
ax1.set_xticklabels(quant_methods, fontsize=10)
ax1.set_ylabel('精度保持率 (%)', fontsize=12)
ax1.set_title('量化方法: 精度 vs 内存', fontsize=14, fontweight='bold')
ax1.set_ylim(90, 102)
ax1_twin.legend(fontsize=10, loc='center right')
ax1.grid(True, alpha=0.3, axis='y')

# 子图2: NF4 vs 均匀量化
ax2 = axes[1]
# 生成正态分布权重
np.random.seed(42)
weights = np.random.randn(10000)

# NF4量化级别 (非均匀, 中间更密集)
nf4_levels = np.array([-1.0, -0.696, -0.525, -0.394, -0.284, -0.185, -0.093, 0.0,
                        0.093, 0.185, 0.284, 0.394, 0.525, 0.696, 1.0, 0.796])
nf4_levels = np.sort(nf4_levels)

# 均匀量化级别
uniform_levels = np.linspace(-1, 1, 16)

# 显示量化级别
ax2.hist(weights, bins=50, density=True, alpha=0.3, color='gray', label='权重分布')
for level in nf4_levels:
    ax2.axvline(x=level, color='#e74c3c', linewidth=1, alpha=0.7)
ax2.axvline(x=uniform_levels[8], color='#3498db', linewidth=1, alpha=0.5)
ax2.set_title('NF4量化级别 (红色线)', fontsize=14, fontweight='bold')
ax2.set_xlabel('权重值', fontsize=12)
ax2.set_ylabel('密度', fontsize=12)
ax2.set_xlim(-3, 3)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# 子图3: QLoRA内存分解
ax3 = axes[2]
components = ['模型权重\n(NF4)', 'LoRA参数\n(fp16)', '梯度\n(LoRA)', '优化器\n(LoRA)', '激活值']
qlora_sizes = [3.5, 0.02, 0.02, 0.04, 1.5]  # GB
lora_sizes = [14, 0.02, 0.02, 0.04, 1.5]
full_sizes = [14, 14, 14, 28, 2]

x = np.arange(len(components))
width = 0.25
ax3.bar(x - width, full_sizes, width, label='全量微调', color='#e74c3c', alpha=0.85, edgecolor='black')
ax3.bar(x, lora_sizes, width, label='LoRA', color='#3498db', alpha=0.85, edgecolor='black')
ax3.bar(x + width, qlora_sizes, width, label='QLoRA', color='#2ecc71', alpha=0.85, edgecolor='black')

ax3.set_xticks(x)
ax3.set_xticklabels(components, fontsize=9)
ax3.set_ylabel('内存 (GB)', fontsize=12)
ax3.set_title('内存分解对比 (7B模型)', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W20/qlora_gptq.png', dpi=150, bbox_inches='tight')
print("\n  图表已保存: qlora_gptq.png")
plt.close()

# ============================================================
# 7. 量化选择指南
# ============================================================
print("\n--- 7. 量化选择指南 ---")
print("""
  +------------------+------------------+------------------+
  |      方法        |    适用场景      |    推荐工具      |
  +------------------+------------------+------------------+
  | FP16/BF16        | 训练/推理       | 默认             |
  | INT8 (bitsandbytes) | 推理节省内存 | bitsandbytes     |
  | NF4 (QLoRA)      | LoRA微调        | bitsandbytes+PEFT|
  | GPTQ 4-bit       | 纯推理部署      | auto-gptq        |
  | AWQ              | 纯推理部署      | autoawq          |
  | GGUF (llama.cpp) | CPU推理         | llama.cpp        |
  +------------------+------------------+------------------+

  推荐:
    微调: QLoRA (NF4 + LoRA)
    GPU推理: GPTQ 或 AWQ
    CPU推理: GGUF (llama.cpp)
""")

print("\n" + "=" * 60)
print("W20-D4 完成! 本节学习了量化原理和QLoRA/GPTQ")
print("=" * 60)
