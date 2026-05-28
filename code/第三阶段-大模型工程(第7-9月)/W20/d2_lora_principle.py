"""
W20-D2 LoRA原理 (LoRA Principle)
=================================
LoRA数学原理(W = W0 + BA), 低秩分解, 秩r的选择,
Alpha参数, LoRA在不同层的应用
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W20-D2 LoRA原理 (LoRA Principle)")
print("=" * 60)

# ============================================================
# 1. LoRA 核心思想
# ============================================================
print("\n--- 1. LoRA 核心思想 ---")
print("""
  LoRA = Low-Rank Adaptation (Hu et al., 2021, Microsoft)

  核心观察:
    预训练模型的权重更新具有低秩特性 (即不需要改变所有方向)
    因此可以用低秩矩阵来近似权重更新

  数学原理:
    原始:   y = W * x      (W: d x d)
    LoRA:   y = (W + ΔW) * x = W * x + ΔW * x
    其中:   ΔW = B * A      (低秩分解)
            B: d x r
            A: r x d
            r << d (秩远小于维度)

  参数量对比:
    原始更新:  d x d = d² 个参数
    LoRA:      d x r + r x d = 2dr 个参数
    压缩比:    d² / (2dr) = d / (2r)

  示例 (d=4096, r=8):
    原始: 4096² = 16,777,216 参数
    LoRA: 2 * 4096 * 8 = 65,536 参数
    压缩比: 256x !
""")

# ============================================================
# 2. 低秩分解详解
# ============================================================
print("\n--- 2. 低秩分解详解 ---")
print("""
  矩阵的低秩分解: M ≈ B * A

  直观理解:
    一个 d×d 的矩阵有 d² 个自由度
    如果它的"有效秩"只有r, 那么只有 ~2dr 个重要自由度
    可以用 B(d×r) 和 A(r×d) 来近似

  数学基础 (SVD):
    M = U * S * V^T
    取前r个奇异值: M_r = U[:, :r] * S[:r, :r] * V^T[:r, :]

  LoRA vs SVD:
    SVD:  需要完整矩阵M, 然后压缩
    LoRA: 直接学习 B 和 A, 不需要先有 ΔW
""")

# 低秩分解演示
np.random.seed(42)
d = 64  # 原始维度
r = 4   # 低秩

# 创建一个"天然"低秩矩阵
A_true = np.random.randn(r, d)
B_true = np.random.randn(d, r)
W_delta = B_true @ A_true  # d x d, 秩为r

print(f"  ΔW 矩阵形状: {W_delta.shape}")
print(f"  ΔW 实际秩: {np.linalg.matrix_rank(W_delta)}")
print(f"  ΔW 参数量: {W_delta.size}")
print(f"  B*A 参数量: {B_true.size + A_true.size}")
print(f"  压缩比: {W_delta.size / (B_true.size + A_true.size):.1f}x")

# 验证低秩近似
U, S, Vt = np.linalg.svd(W_delta)
print(f"\n  奇异值: {S[:6].round(2)}")
print(f"  前{r}个奇异值能量占比: {S[:r].sum() / S.sum() * 100:.1f}%")

# ============================================================
# 3. LoRA 缩放因子 Alpha
# ============================================================
print("\n--- 3. LoRA 缩放因子 Alpha ---")
print("""
  LoRA的完整公式:
    y = W * x + (α/r) * B * A * x

  α (alpha): 缩放因子, 控制LoRA更新的强度
  r: 秩

  α/r 的作用:
    - 控制LoRA的贡献相对于原始权重的比例
    - 增大α/r → 更强的适应 (更接近微调效果)
    - 减小α/r → 更保守的适应 (更接近原始模型)

  常用设置:
    r=8,   α=16  → α/r = 2.0
    r=16,  α=32  → α/r = 2.0
    r=64,  α=16  → α/r = 0.25

  经验法则:
    α 通常设为 r 的 1-2 倍
    α/r = 1-2 是常见的起点
""")

# Alpha影响的可视化演示
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 子图1: 低秩分解可视化
ax1 = axes[0]
ax1.set_title('LoRA 低秩分解', fontsize=14, fontweight='bold')

# 绘制矩阵结构
# W (d x d)
ax1.add_patch(plt.Rectangle((0, 0), 4, 4, facecolor='#3498db', edgecolor='black', alpha=0.4))
ax1.text(2, 2, 'W\n(d×d)', ha='center', va='center', fontsize=14, fontweight='bold')
ax1.text(2, -0.5, '冻结', ha='center', fontsize=10, color='blue')

# + 号
ax1.text(5, 2, '+', ha='center', va='center', fontsize=20, fontweight='bold')

# B (d x r)
ax1.add_patch(plt.Rectangle((6, 0), 0.8, 4, facecolor='#e74c3c', edgecolor='black', alpha=0.4))
ax1.text(6.4, 2, 'B\n(d×r)', ha='center', va='center', fontsize=10, fontweight='bold')
ax1.text(6.4, -0.5, '可训练', ha='center', fontsize=10, color='red')

# * 号
ax1.text(7.5, 2, '×', ha='center', va='center', fontsize=20, fontweight='bold')

# A (r x d)
ax1.add_patch(plt.Rectangle((8.2, 3.2), 4, 0.8, facecolor='#2ecc71', edgecolor='black', alpha=0.4))
ax1.text(10.2, 3.6, 'A (r×d)', ha='center', va='center', fontsize=10, fontweight='bold')
ax1.text(10.2, 2.8, '可训练', ha='center', fontsize=10, color='green')

ax1.set_xlim(-1, 13)
ax1.set_ylim(-1.5, 5)
ax1.axis('off')

# 子图2: 不同秩r的效果
ax2 = axes[1]
ranks = [1, 2, 4, 8, 16, 32, 64, 128]
# 模拟: 秩越大效果越好但收益递减
quality = [65, 75, 82, 88, 91, 93, 94, 94.5]
params_ratio = [2 * r / (d * d) * 100 for r in ranks]  # 简化参数比例

ax2.plot(ranks, quality, 'o-', linewidth=2, markersize=8, color='#3498db', label='任务效果')
ax2.set_xlabel('秩 r', fontsize=12)
ax2.set_ylabel('效果分数', fontsize=12)
ax2.set_title('秩 r 对效果的影响', fontsize=14, fontweight='bold')
ax2.axvline(x=8, color='red', linestyle='--', label='推荐 r=8')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# 子图3: Alpha/r对训练的影响
ax3 = axes[2]
alpha_over_r = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
train_loss = [0.85, 0.72, 0.55, 0.42, 0.38, 0.40]  # 太大可能过拟合
overfit_score = [0.05, 0.08, 0.12, 0.15, 0.25, 0.35]

ax3.plot(alpha_over_r, train_loss, 'o-', linewidth=2, color='#2ecc71', label='训练损失')
ax3_twin = ax3.twinx()
ax3_twin.plot(alpha_over_r, overfit_score, 's-', linewidth=2, color='#e74c3c', label='过拟合风险')
ax3.set_xlabel('α/r', fontsize=12)
ax3.set_ylabel('训练损失', fontsize=12, color='#2ecc71')
ax3_twin.set_ylabel('过拟合风险', fontsize=12, color='#e74c3c')
ax3.set_title('α/r 对训练的影响', fontsize=14, fontweight='bold')
ax3.set_xscale('log')
ax3.legend(fontsize=10, loc='upper left')
ax3_twin.legend(fontsize=10, loc='upper right')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W20/lora_principle.png', dpi=150, bbox_inches='tight')
print("  图表已保存: lora_principle.png")
plt.close()

# ============================================================
# 4. LoRA 在不同层的应用
# ============================================================
print("\n--- 4. LoRA 在不同层的应用 ---")
print("""
  Transformer中的LoRA应用位置:

  每个Transformer Block包含:
    1) Self-Attention:
       - Wq (d x d): Query投影    ← 常用LoRA
       - Wk (d x d): Key投影      ← 可选LoRA
       - Wv (d x d): Value投影    ← 常用LoRA
       - Wo (d x d): Output投影   ← 可选LoRA

    2) Feed-Forward Network:
       - W1 (d x 4d): 升维        ← 可选LoRA
       - W2 (4d x d): 降维        ← 可选LoRA

  常见配置:
    最小配置: 只在 Q, V 上加LoRA (论文推荐)
    中等配置: 在 Q, V, K, O 上加LoRA
    最大配置: 在所有线性层加LoRA

  哪些层加LoRA效果最好?
    - 通常Q和V最有效
    - FFN层也可以加, 但收益较小
    - 低层(靠近输入)的更新较小, 高层更新较大
""")

# 不同配置的参数量对比
print("\n  --- LoRA配置对比 (LLaMA-7B, r=16) ---")
d = 4096
r = 16
num_layers = 32

configs = {
    "Q+V":      num_layers * 2 * (d * r + r * d),
    "Q+K+V+O":  num_layers * 4 * (d * r + r * d),
    "全部线性层": num_layers * (4 * (d * r + r * d) + d * r + r * d + 4*d * r + r * 4*d),
}

for name, params in configs.items():
    ratio = params / 7e9 * 100
    print(f"  {name:<12s}: {params:>12,} 参数 ({ratio:.3f}%)")

# ============================================================
# 5. LoRA 初始化策略
# ============================================================
print("\n--- 5. LoRA 初始化策略 ---")
print("""
  B 和 A 的初始化:
    A: 使用随机高斯初始化 (Kaiming)
    B: 初始化为全零

  为什么?
    W_new = W + B * A
    初始时 B * A = 0 (因为B全零)
    所以初始时 LoRA 不改变模型行为
    训练从原始模型的性能开始, 逐步调整

  推理时的优化:
    训练完成后: W_merged = W + (α/r) * B * A
    将LoRA合并回原始权重, 推理时无额外开销!

  多任务切换:
    保存不同的 B*A 对, 推理时切换:
    Task A: W + B_A * A_A
    Task B: W + B_B * A_B
""")

# 演示初始化和合并
d_demo = 8
r_demo = 2
np.random.seed(42)

W = np.random.randn(d_demo, d_demo)  # 原始权重
A = np.random.randn(r_demo, d_demo)  # A: 随机初始化
B = np.zeros((d_demo, r_demo))       # B: 全零初始化

x = np.random.randn(d_demo)

# 初始时 LoRA 无影响
y_original = W @ x
y_lora_init = (W + B @ A) @ x
print(f"\n  原始输出:     {y_original.round(3)}")
print(f"  LoRA初始输出: {y_lora_init.round(3)}")
print(f"  差异:         {np.linalg.norm(y_original - y_lora_init):.6f} (应该为0)")

# 模拟训练一步
B_trained = np.random.randn(d_demo, r_demo) * 0.1  # 模拟训练后的B
y_lora_trained = (W + B_trained @ A) @ x
print(f"\n  训练后输出:   {y_lora_trained.round(3)}")
print(f"  变化量:       {np.linalg.norm(y_lora_trained - y_original):.4f}")

# 合并权重
W_merged = W + B_trained @ A
y_merged = W_merged @ x
print(f"\n  合并后输出:   {y_merged.round(3)}")
print(f"  与LoRA输出一致: {np.allclose(y_lora_trained, y_merged)}")

# ============================================================
# 6. LoRA 与其他方法的数学关系
# ============================================================
print("\n--- 6. LoRA 与其他方法的数学关系 ---")
print("""
  LoRA 是更通用框架的特例:

  全量微调:   ΔW 无约束 (d²参数)
  Adapter:    ΔW = U * diag(h) * V  (瓶颈结构)
  Prefix:     ΔW 作用于输入序列的前缀位置
  LoRA:       ΔW = B * A  (低秩约束)
  IA3:        ΔW = diag(l)  (对角缩放, 秩=1)

  统一视角: 都是在参数空间中寻找高效的方向来更新模型

  LoRA的优势:
    1) 训练后可合并回原权重, 无推理开销
    2) 参数量与秩r线性相关, 容易控制
    3) 可以在不同任务间灵活切换
    4) 与量化兼容 (QLoRA)
""")

print("\n" + "=" * 60)
print("W20-D2 完成! 本节深入讲解了LoRA的数学原理")
print("=" * 60)
