"""
W13-D2: Scaled Dot-Product Attention
=====================================
缩放点积注意力的公式、手动计算、为什么缩放、因果掩码。
"""

import torch
import torch.nn.functional as F
import numpy as np
import math


# ============================================================
# 1. 公式详解
# ============================================================
def demo_formula():
    """
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    """
    print("=" * 60)
    print("1. 缩放点积注意力公式")
    print("=" * 60)

    print("公式: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) @ V")
    print()
    print("维度分析:")
    print("  Q: [batch, seq_len_q, d_k]   — 查询矩阵")
    print("  K: [batch, seq_len_k, d_k]   — 键矩阵")
    print("  V: [batch, seq_len_k, d_v]   — 值矩阵")
    print()
    print("  Step 1: scores = Q @ K^T")
    print("          [batch, seq_len_q, d_k] @ [batch, d_k, seq_len_k]")
    print("          = [batch, seq_len_q, seq_len_k]")
    print()
    print("  Step 2: scaled_scores = scores / sqrt(d_k)")
    print("          缩放因子防止点积过大导致 softmax 饱和")
    print()
    print("  Step 3: weights = softmax(scaled_scores, dim=-1)")
    print("          沿最后一个维度归一化，每行和为1")
    print()
    print("  Step 4: output = weights @ V")
    print("          [batch, seq_len_q, seq_len_k] @ [batch, seq_len_k, d_v]")
    print("          = [batch, seq_len_q, d_v]")
    print()


# ============================================================
# 2. 手动计算（逐步数字演示）
# ============================================================
def demo_manual_calculation():
    """
    用具体数字手动计算注意力
    """
    print("=" * 60)
    print("2. 手动计算演示")
    print("=" * 60)

    # 使用简单的数字，2x2 矩阵
    Q = torch.tensor([[1.0, 0.0],
                       [0.0, 1.0]])
    K = torch.tensor([[1.0, 0.0],
                       [0.0, 1.0]])
    V = torch.tensor([[1.0, 2.0],
                       [3.0, 4.0]])

    d_k = Q.shape[-1]
    print(f"d_k = {d_k}")
    print(f"sqrt(d_k) = {math.sqrt(d_k):.4f}")
    print()

    # Step 1: Q @ K^T
    scores = Q @ K.T
    print(f"Step 1: scores = Q @ K^T")
    print(f"  Q = {Q.tolist()}")
    print(f"  K = {K.tolist()}")
    print(f"  scores = {scores.tolist()}")

    # Step 2: 缩放
    scaled_scores = scores / math.sqrt(d_k)
    print(f"\nStep 2: scaled = scores / sqrt({d_k})")
    print(f"  scaled_scores = {scaled_scores.tolist()}")

    # Step 3: Softmax
    weights = F.softmax(scaled_scores, dim=-1)
    print(f"\nStep 3: weights = softmax(scaled_scores)")
    print(f"  weights = ")
    for i, row in enumerate(weights):
        print(f"    行 {i}: [{', '.join(f'{v:.4f}' for v in row)}]")
    print(f"  每行和: {weights.sum(dim=-1).tolist()}")

    # Step 4: weights @ V
    output = weights @ V
    print(f"\nStep 4: output = weights @ V")
    print(f"  V = {V.tolist()}")
    print(f"  output = {output.tolist()}")

    # 解释
    print(f"\n解释:")
    print(f"  Q[0]=[1,0] 与 K[0]=[1,0] 完全匹配 -> weights[0][0] 最大")
    print(f"  Q[1]=[0,1] 与 K[1]=[0,1] 完全匹配 -> weights[1][1] 最大")
    print(f"  所以 output[0] 约等于 V[0]=[1,2], output[1] 约等于 V[1]=[3,4]")
    print()


# ============================================================
# 3. 为什么需要缩放 (scale by sqrt(d_k))
# ============================================================
def demo_why_scale():
    """
    演示不缩放时的问题
    """
    print("=" * 60)
    print("3. 为什么需要缩放 sqrt(d_k)?")
    print("=" * 60)

    print("原因: 当 d_k 较大时，Q @ K^T 的值方差也会增大")
    print("方差大的输入导致 softmax 输出接近 one-hot（梯度消失）")
    print()

    for d_k in [8, 64, 512, 1024]:
        Q = torch.randn(1, d_k)
        K = torch.randn(10, d_k)

        # 不缩放的分数
        raw_scores = (Q @ K.T).squeeze()
        raw_std = raw_scores.std().item()
        raw_weights = F.softmax(raw_scores, dim=-1)
        raw_max = raw_weights.max().item()

        # 缩放后的分数
        scaled_scores = raw_scores / math.sqrt(d_k)
        scaled_std = scaled_scores.std().item()
        scaled_weights = F.softmax(scaled_scores, dim=-1)
        scaled_max = scaled_weights.max().item()

        print(f"d_k = {d_k:>4}:")
        print(f"  不缩放: scores std={raw_std:>7.2f}, "
              f"softmax max={raw_max:.6f} (接近one-hot!)")
        print(f"  缩放后: scores std={scaled_std:>7.2f}, "
              f"softmax max={scaled_max:.6f} (更平滑)")
        print()

    print("关键洞察:")
    print("  Q 和 K 的每个元素独立 ~N(0,1)")
    print("  点积 Q·K = sum(q_i * k_i) 的方差 = d_k")
    print("  所以 std = sqrt(d_k)")
    print("  除以 sqrt(d_k) 让方差回到 1")
    print()


# ============================================================
# 4. NumPy 实现
# ============================================================
def scaled_dot_product_attention_numpy(Q, K, V, mask=None):
    """
    用 NumPy 实现缩放点积注意力

    参数:
        Q: shape (seq_len_q, d_k)
        K: shape (seq_len_k, d_k)
        V: shape (seq_len_k, d_v)
        mask: optional, shape (seq_len_q, seq_len_k)
    返回:
        output: shape (seq_len_q, d_v)
        weights: shape (seq_len_q, seq_len_k)
    """
    d_k = Q.shape[-1]

    # Step 1: 点积
    scores = Q @ K.T  # (seq_len_q, seq_len_k)

    # Step 2: 缩放
    scores = scores / np.sqrt(d_k)

    # Step 3: 掩码（可选）
    if mask is not None:
        scores = np.where(mask, scores, -1e9)

    # Step 4: Softmax
    # 数值稳定版本: 减去最大值
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # Step 5: 加权求和
    output = weights @ V  # (seq_len_q, d_v)

    return output, weights


def demo_numpy_implementation():
    print("=" * 60)
    print("4. NumPy 实现")
    print("=" * 60)

    np.random.seed(42)

    seq_len = 5
    d_k = 8
    d_v = 6

    Q = np.random.randn(seq_len, d_k).astype(np.float32)
    K = np.random.randn(seq_len, d_k).astype(np.float32)
    V = np.random.randn(seq_len, d_v).astype(np.float32)

    output, weights = scaled_dot_product_attention_numpy(Q, K, V)

    print(f"Q shape: {Q.shape}")
    print(f"K shape: {K.shape}")
    print(f"V shape: {V.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"Weights 每行和: {weights.sum(axis=-1)}")

    # 验证与 PyTorch 一致
    Q_t = torch.from_numpy(Q)
    K_t = torch.from_numpy(K)
    V_t = torch.from_numpy(V)
    ref_output = F.scaled_dot_product_attention(Q_t, K_t, V_t)

    diff = np.abs(output - ref_output.numpy()).max()
    print(f"\n与 PyTorch F.scaled_dot_product_attention 差异: {diff:.8f}")
    print()


# ============================================================
# 5. 因果掩码 (Causal Mask)
# ============================================================
def demo_causal_mask():
    """
    因果掩码: 确保位置 i 只能关注位置 <= i
    用于 Transformer 解码器
    """
    print("=" * 60)
    print("5. 因果掩码 (Causal / Look-ahead Mask)")
    print("=" * 60)

    seq_len = 5
    d_k = 8
    d_v = 8

    np.random.seed(42)
    Q = np.random.randn(seq_len, d_k).astype(np.float32)
    K = np.random.randn(seq_len, d_k).astype(np.float32)
    V = np.random.randn(seq_len, d_v).astype(np.float32)

    # 创建因果掩码（上三角为 False）
    causal_mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))
    print(f"因果掩码 (1=可见, 0=屏蔽):")
    print("     " + "  ".join(f"Pos{j}" for j in range(seq_len)))
    for i in range(seq_len):
        row = "  ".join("  T" if causal_mask[i, j] else "  F" for j in range(seq_len))
        print(f"Pos{i}: {row}")
    print()

    # 不带掩码
    output_no_mask, weights_no_mask = scaled_dot_product_attention_numpy(Q, K, V)
    # 带掩码
    output_masked, weights_masked = scaled_dot_product_attention_numpy(Q, K, V, mask=causal_mask)

    print("注意力权重 (无掩码):")
    for i in range(seq_len):
        row = " ".join(f"{w:.3f}" for w in weights_no_mask[i])
        print(f"  Pos{i}: [{row}]")

    print("\n注意力权重 (因果掩码):")
    for i in range(seq_len):
        row = " ".join(f"{w:.3f}" if causal_mask[i, j] else "  ---" for j, w in enumerate(weights_masked[i]))
        print(f"  Pos{i}: [{row}]")

    print("\n含义:")
    print("  位置 0 只能看到自己")
    print("  位置 1 可以看到位置 0 和 1")
    print("  位置 4 可以看到所有位置 0-4")
    print("  这保证了自回归生成时不会'偷看'未来信息")
    print()


# ============================================================
# 6. PyTorch 完整实现
# ============================================================
class ScaledDotProductAttention(torch.nn.Module):
    """缩放点积注意力模块"""

    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        """
        参数:
            Q: [batch, seq_len_q, d_k]
            K: [batch, seq_len_k, d_k]
            V: [batch, seq_len_k, d_v]
            mask: [batch, seq_len_q, seq_len_k] 或 [seq_len_q, seq_len_k]
        返回:
            output: [batch, seq_len_q, d_v]
            weights: [batch, seq_len_q, seq_len_k]
        """
        d_k = Q.shape[-1]

        # 计算注意力分数
        scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)

        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # 加权求和
        output = weights @ V

        return output, weights


def demo_pytorch_implementation():
    print("=" * 60)
    print("6. PyTorch 模块实现")
    print("=" * 60)

    batch_size = 2
    seq_len = 5
    d_k = 8
    d_v = 8

    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.randn(batch_size, seq_len, d_v)

    attn = ScaledDotProductAttention(dropout=0.0)

    # 无掩码
    output, weights = attn(Q, K, V)
    print(f"输入: Q{list(Q.shape)}, K{list(K.shape)}, V{list(V.shape)}")
    print(f"输出: {list(output.shape)}, 权重: {list(weights.shape)}")

    # 因果掩码
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
    causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)

    output_masked, weights_masked = attn(Q, K, V, mask=causal_mask)
    print(f"\n因果掩码输出: {list(output_masked.shape)}")

    # 验证位置 i 的输出只依赖于位置 0..i
    print("验证: 位置 0 的权重只在位置 0 有值:")
    print(f"  weights[0, 0] = {weights_masked[0, 0].tolist()}")
    print(f"  只有权重[0]非零: {(weights_masked[0, 0, 1:] == 0).all().item()}")
    print()


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    demo_formula()
    demo_manual_calculation()
    demo_why_scale()
    demo_numpy_implementation()
    demo_causal_mask()
    demo_pytorch_implementation()

    print("所有演示完成！")
