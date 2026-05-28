"""
W13-D4: Positional Encoding (Sinusoidal + RoPE)
================================================
位置编码的实现和可视化。
"""

import torch
import torch.nn as nn
import numpy as np
import math


# ============================================================
# 1. 为什么需要位置编码？
# ============================================================
def demo_why_position():
    """
    演示自注意力对位置不敏感
    """
    print("=" * 60)
    print("1. 为什么需要位置编码？")
    print("=" * 60)

    print("自注意力的特性:")
    print("  - 对输入顺序不敏感 (permutation invariant)")
    print("  - 无论输入 ['我', '爱', '你'] 还是 ['你', '爱', '我']")
    print("    自注意力计算方式完全相同")
    print()

    # 演示
    d_model = 8
    W_q = torch.randn(d_model, d_model) * 0.1
    W_k = torch.randn(d_model, d_model) * 0.1

    # 原始序列
    x1 = torch.randn(3, d_model)
    # 交换位置
    x2 = x1[[2, 1, 0]]  # 反转序列

    Q1, K1 = x1 @ W_q, x1 @ W_k
    Q2, K2 = x2 @ W_q, x2 @ W_k

    scores1 = Q1 @ K1.T
    scores2 = Q2 @ K2.T

    print("原始序列的注意力分数:")
    print(scores1.detach().numpy().round(2))
    print("\n反转序列的注意力分数:")
    print(scores2.detach().numpy().round(2))
    print("\n注意: scores2 就是 scores1 的行和列都反转了")
    print("      模型无法区分'我爱你'和'你爱我'——需要位置编码!")
    print()


# ============================================================
# 2. Sinusoidal 位置编码
# ============================================================
class SinusoidalPositionalEncoding(nn.Module):
    """
    正弦位置编码 (原论文方法)

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 预计算位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度

        # 注册为 buffer（不参与训练，但会随模型移动设备）
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        """
        参数: x [batch, seq_len, d_model]
        返回: x + positional_encoding
        """
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)


def demo_sinusoidal_encoding():
    """
    演示正弦位置编码的计算过程
    """
    print("=" * 60)
    print("2. Sinusoidal 位置编码")
    print("=" * 60)

    d_model = 16
    max_len = 50
    pe_module = SinusoidalPositionalEncoding(d_model, max_len, dropout=0.0)

    pe = pe_module.pe[0]  # [max_len, d_model]

    print(f"位置编码矩阵 shape: {pe.shape}")
    print(f"  max_len = {max_len}, d_model = {d_model}")
    print()

    # 展示几个位置的前几个维度
    print("位置编码示例 (前8维):")
    print(f"{'Pos':>4} | " + " | ".join(f"dim{i}" for i in range(8)))
    print("-" * 70)
    for pos in [0, 1, 2, 5, 10, 20]:
        vals = " | ".join(f"{pe[pos, i]:>5.2f}" for i in range(8))
        print(f"{pos:>4} | {vals}")

    print()

    # 手动验证第一个位置的编码
    print("手动验证 position=0:")
    for i in range(min(4, d_model // 2)):
        angle = 0.0 / (10000 ** (2 * i / d_model))
        print(f"  dim {2*i}: sin({angle:.6f}) = {math.sin(angle):.6f}, "
              f"实际: {pe[0, 2*i]:.6f}")
        print(f"  dim {2*i+1}: cos({angle:.6f}) = {math.cos(angle):.6f}, "
              f"实际: {pe[0, 2*i+1]:.6f}")

    print()

    # 展示与 embedding 相加
    batch_size = 2
    seq_len = 10
    embedding = torch.randn(batch_size, seq_len, d_model)
    output = pe_module(embedding)
    print(f"Embedding shape: {list(embedding.shape)}")
    print(f"输出 shape:      {list(output.shape)}")
    print(f"输出 = embedding + positional_encoding")
    print()


# ============================================================
# 3. 位置编码可视化（字符画热力图）
# ============================================================
def visualize_positional_encoding():
    """
    用字符画可视化位置编码
    """
    print("=" * 60)
    print("3. 位置编码可视化")
    print("=" * 60)

    d_model = 32
    max_len = 20
    pe_module = SinusoidalPositionalEncoding(d_model, max_len, dropout=0.0)
    pe = pe_module.pe[0].numpy()  # [max_len, d_model]

    # 显示前 16 个维度的热力图
    display_dims = 16
    print(f"位置编码热力图 (位置 0-19, 维度 0-{display_dims - 1}):")
    print()

    header = "Pos  "
    for d in range(display_dims):
        header += f"d{d:>2} "
    print(header)
    print("-" * len(header))

    for pos in range(max_len):
        row = f"{pos:>3}  "
        for d in range(display_dims):
            val = pe[pos, d]
            if val > 0.7:
                ch = "+++ "
            elif val > 0.3:
                ch = "++  "
            elif val > 0.0:
                ch = "+   "
            elif val > -0.3:
                ch = "-   "
            elif val > -0.7:
                ch = "--  "
            else:
                ch = "--- "
            row += ch
        print(row)

    print()
    print("观察:")
    print("  - 低维度 (d0, d1): 波长短，相邻位置差异大")
    print("  - 高维度 (d14, d15): 波长长，变化缓慢")
    print("  - 每个位置的编码都是唯一的")
    print()


# ============================================================
# 4. 位置编码的特性
# ============================================================
def demo_encoding_properties():
    """
    演示正弦位置编码的重要特性
    """
    print("=" * 60)
    print("4. 位置编码的数学特性")
    print("=" * 60)

    d_model = 64
    max_len = 100
    pe_module = SinusoidalPositionalEncoding(d_model, max_len, dropout=0.0)
    pe = pe_module.pe[0]  # [max_len, d_model]

    # 特性 1: 每个位置编码是唯一的
    print("特性 1: 每个位置的编码唯一")
    cos_sims = torch.nn.functional.cosine_similarity(
        pe.unsqueeze(1), pe.unsqueeze(0), dim=-1
    )
    # 对角线应该全是 1
    diag = torch.diag(cos_sims)
    print(f"  自身相似度: min={diag.min():.4f}, max={diag.max():.4f}")

    # 任意两个不同位置的相似度 < 1
    mask = ~torch.eye(max_len, dtype=bool)
    off_diag = cos_sims[mask]
    print(f"  不同位置相似度: min={off_diag.min():.4f}, "
          f"max={off_diag.max():.4f}, mean={off_diag.mean():.4f}")

    # 特性 2: 相邻位置差异较小，远距离差异较大
    print("\n特性 2: 相邻位置差异小，远距离差异大")
    for offset in [1, 5, 10, 50]:
        sim = torch.nn.functional.cosine_similarity(pe[:-offset], pe[offset:], dim=-1)
        print(f"  距离 {offset:>2}: 平均余弦相似度 = {sim.mean():.4f}")

    # 特性 3: 线性关系 (可以表示为偏移位置的线性函数)
    print("\n特性 3: PE(pos+k) 可以用 PE(pos) 的线性变换表示")
    print("  (这是原始论文选择 sin/cos 的原因之一)")
    k = 3
    pe_pos = pe[0]   # 位置 0
    pe_posk = pe[k]  # 位置 k
    print(f"  PE(0) 和 PE({k}) 的关系是确定的数学函数")
    print()


# ============================================================
# 5. RoPE (Rotary Position Embedding) 说明
# ============================================================
def demo_rope():
    """
    解释 RoPE 的原理
    """
    print("=" * 60)
    print("5. RoPE (旋转位置编码)")
    print("=" * 60)

    print("RoPE 的核心思想:")
    print("  将位置信息编码为向量空间中的旋转")
    print("  Q*K 的点积只依赖于相对位置 (m-n)")
    print()
    print("原理:")
    print("  对于 2D 向量 [x1, x2] 和位置 m:")
    print("  RoPE(x, m) = [x1*cos(m*θ) - x2*sin(m*θ),")
    print("                x1*sin(m*θ) + x2*cos(m*θ)]")
    print()
    print("  这等价于旋转矩阵:")
    print("  R(m, θ) = [cos(m*θ), -sin(m*θ)]")
    print("            [sin(m*θ),  cos(m*θ)]")
    print()
    print("  关键性质: <R(m)x, R(n)y> = <R(m-n)x, y>")
    print("  即: 内积只依赖相对位置 m-n")
    print()

    # 简单实现
    def apply_rope(x, pos, theta=1.0):
        """对 2D 向量应用 RoPE"""
        cos_val = math.cos(pos * theta)
        sin_val = math.sin(pos * theta)
        x1, x2 = x[0], x[1]
        return torch.tensor([
            x1 * cos_val - x2 * sin_val,
            x1 * sin_val + x2 * cos_val,
        ])

    x = torch.tensor([1.0, 0.0])
    print(f"原始向量: {x.tolist()}")
    for pos in [0, 1, 2, 3, 4]:
        rotated = apply_rope(x, pos, theta=0.5)
        print(f"  位置 {pos}: {rotated.tolist()}")

    # 验证相对位置性质
    print("\n验证: <RoPE(q, m), RoPE(k, n)> = <RoPE(q, m-n), k>")
    q = torch.randn(2)
    k = torch.randn(2)
    m, n = 5, 3

    left = torch.dot(apply_rope(q, m), apply_rope(k, n))
    right = torch.dot(apply_rope(q, m - n), k)

    print(f"  <RoPE(q, {m}), RoPE(k, {n})> = {left:.6f}")
    print(f"  <RoPE(q, {m - n}), k>       = {right:.6f}")
    print(f"  相等: {abs(left - right) < 1e-5}")

    print()
    print("RoPE 优势:")
    print("  - 天然编码相对位置")
    print("  - 可外推到更长序列")
    print("  - 用于 LLaMA, GPT-NeoX 等现代模型")
    print()


# ============================================================
# 6. 学习式位置编码 vs 固定位置编码
# ============================================================
def demo_learned_vs_fixed():
    """
    比较两种位置编码
    """
    print("=" * 60)
    print("6. 学习式 vs 固定位置编码")
    print("=" * 60)

    d_model = 64
    max_len = 100

    # 固定: Sinusoidal
    fixed_pe = SinusoidalPositionalEncoding(d_model, max_len, dropout=0.0)

    # 学习式: nn.Embedding
    learned_pe = nn.Embedding(max_len, d_model)

    print(f"{'特征':>12} | {'Sinusoidal':>12} | {'Learned':>12}")
    print("-" * 45)
    print(f"{'参数量':>12} | {'0':>12} | {max_len * d_model:>12,}")
    print(f"{'灵活性':>12} | {'固定公式':>12} | {'数据驱动':>12}")
    print(f"{'外推性':>12} | {'可外推':>12} | {'不可外推':>12}")
    print(f"{'使用模型':>12} | {'原Transformer':>12} | {'BERT/GPT':>12}")

    # 使用方式
    seq_len = 10
    batch_size = 2
    x = torch.randn(batch_size, seq_len, d_model)

    # 固定编码
    out_fixed = fixed_pe(x)

    # 学习式编码
    positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    pos_emb = learned_pe(positions)
    out_learned = x + pos_emb

    print(f"\n输出 shape (固定):   {list(out_fixed.shape)}")
    print(f"输出 shape (学习式): {list(out_learned.shape)}")
    print()


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)

    demo_why_position()
    demo_sinusoidal_encoding()
    visualize_positional_encoding()
    demo_encoding_properties()
    demo_rope()
    demo_learned_vs_fixed()

    print("所有演示完成！")
