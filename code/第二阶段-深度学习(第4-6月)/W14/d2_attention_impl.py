"""
W14-D2: ScaledDotProductAttention + Causal Mask + MultiHeadAttention
=====================================================================
Transformer 从零实现 - 第2步: 注意力机制。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 1. Scaled Dot-Product Attention
# ============================================================
class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) @ V
    """

    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        """
        参数:
            Q: [batch, num_heads, seq_len_q, d_k]
            K: [batch, num_heads, seq_len_k, d_k]
            V: [batch, num_heads, seq_len_k, d_v]
            mask: [batch, 1, seq_len_q, seq_len_k] 或 [1, 1, seq_len_q, seq_len_k]
                  或 [seq_len_q, seq_len_k] (bool, True=有效)
        返回:
            output: [batch, num_heads, seq_len_q, d_v]
            attention_weights: [batch, num_heads, seq_len_q, seq_len_k]
        """
        d_k = Q.shape[-1]

        # QK^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 加权求和
        output = torch.matmul(attention_weights, V)

        return output, attention_weights


def test_scaled_attention():
    print("=" * 60)
    print("1. ScaledDotProductAttention 测试")
    print("=" * 60)

    batch = 2
    heads = 4
    seq_len = 5
    d_k = 16
    d_v = 16

    attn = ScaledDotProductAttention(dropout=0.0)

    Q = torch.randn(batch, heads, seq_len, d_k)
    K = torch.randn(batch, heads, seq_len, d_k)
    V = torch.randn(batch, heads, seq_len, d_v)

    output, weights = attn(Q, K, V)

    print(f"Q: {list(Q.shape)}")
    print(f"K: {list(K.shape)}")
    print(f"V: {list(V.shape)}")
    print(f"Output: {list(output.shape)}")
    print(f"Weights: {list(weights.shape)}")

    # 验证权重归一化
    row_sums = weights.sum(dim=-1)
    print(f"\n权重行和 (应为1.0): {row_sums[0, 0].tolist()}")

    # 验证无掩码时所有位置都有权重
    min_weight = weights.min().item()
    print(f"最小权重: {min_weight:.6f} (>0, 无掩码时所有位置有非零权重)")
    assert min_weight > 0, "无掩码时所有权重应 > 0"
    print("[PASS] 基本注意力测试")
    print()


# ============================================================
# 2. Causal Mask (因果掩码)
# ============================================================
def generate_causal_mask(seq_len):
    """
    生成因果掩码 (下三角)

    返回: [1, 1, seq_len, seq_len] bool tensor
    True = 可见, False = 屏蔽

    Example (seq_len=4):
    [[1, 0, 0, 0],
     [1, 1, 0, 0],
     [1, 1, 1, 0],
     [1, 1, 1, 1]]
    """
    mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]


def test_causal_mask():
    print("=" * 60)
    print("2. Causal Mask 测试")
    print("=" * 60)

    seq_len = 5
    mask = generate_causal_mask(seq_len)

    print(f"Causal Mask shape: {list(mask.shape)}")
    print(f"Causal Mask (位置 i 只能看到 <=i):")
    print("     " + " ".join(f"P{j}" for j in range(seq_len)))
    for i in range(seq_len):
        row = " ".join(" V" if mask[0, 0, i, j] else " X" for j in range(seq_len))
        print(f"  P{i}: {row}")

    # 使用因果掩码的注意力
    batch = 2
    heads = 4
    d_k = 16

    attn = ScaledDotProductAttention(dropout=0.0)
    Q = torch.randn(batch, heads, seq_len, d_k)
    K = torch.randn(batch, heads, seq_len, d_k)
    V = torch.randn(batch, heads, seq_len, d_k)

    output, weights = attn(Q, K, V, mask=mask)

    print(f"\n因果注意力权重 (batch=0, head=0):")
    for i in range(seq_len):
        row = " ".join(f"{weights[0, 0, i, j]:.3f}" if mask[0, 0, i, j] else "  ---" for j in range(seq_len))
        print(f"  P{i}: [{row}]")

    # 验证: 上三角部分权重为0
    upper_tri = ~mask[0, 0]
    assert (weights[0, 0][upper_tri] == 0).all(), "上三角权重应为0"
    print("[PASS] 因果掩码: 上三角权重全为0")
    print()


# ============================================================
# 3. Multi-Head Attention
# ============================================================
class MultiHeadAttention(nn.Module):
    """
    多头注意力

    1. 线性投影: Q, K, V -> Q', K', V'
    2. 分头: 拆分为 num_heads 个头
    3. 每个头独立计算注意力
    4. 拼接所有头的输出
    5. 最终线性投影
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model({d_model}) 必须能被 num_heads({num_heads}) 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 线性投影
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)

    def forward(self, query, key, value, mask=None):
        """
        参数:
            query: [batch, seq_len_q, d_model]
            key:   [batch, seq_len_k, d_model]
            value: [batch, seq_len_k, d_model]
            mask:  [seq_len_q, seq_len_k] 或可广播的形状
        返回:
            output: [batch, seq_len_q, d_model]
            attention_weights: [batch, num_heads, seq_len_q, seq_len_k]
        """
        batch_size = query.shape[0]

        # 1. 线性投影
        Q = self.W_q(query)  # [batch, seq_len_q, d_model]
        K = self.W_k(key)    # [batch, seq_len_k, d_model]
        V = self.W_v(value)  # [batch, seq_len_k, d_model]

        # 2. 分头: [batch, seq_len, d_model] -> [batch, num_heads, seq_len, d_k]
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 3. 注意力计算
        attn_out, attn_weights = self.attention(Q, K, V, mask)

        # 4. 拼接头: [batch, num_heads, seq_len_q, d_k] -> [batch, seq_len_q, d_model]
        attn_out = attn_out.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        # 5. 输出投影
        output = self.W_o(attn_out)

        return output, attn_weights


def test_multihead_attention():
    print("=" * 60)
    print("3. MultiHeadAttention 测试")
    print("=" * 60)

    d_model = 64
    num_heads = 8
    batch_size = 2
    seq_len = 5

    mha = MultiHeadAttention(d_model, num_heads, dropout=0.0)

    x = torch.randn(batch_size, seq_len, d_model)

    # 自注意力
    output, weights = mha(x, x, x)

    print(f"自注意力:")
    print(f"  输入: {list(x.shape)}")
    print(f"  输出: {list(output.shape)}")
    print(f"  权重: {list(weights.shape)}")
    assert output.shape == (batch_size, seq_len, d_model), f"Output shape 错误"
    print(f"  [PASS] 输出 shape 正确")

    # 交叉注意力
    src_len = 8
    tgt_len = 5
    query = torch.randn(batch_size, tgt_len, d_model)
    key = torch.randn(batch_size, src_len, d_model)
    value = torch.randn(batch_size, src_len, d_model)

    output_cross, weights_cross = mha(query, key, value)
    print(f"\n交叉注意力:")
    print(f"  Query: {list(query.shape)}")
    print(f"  Key:   {list(key.shape)}")
    print(f"  Value: {list(value.shape)}")
    print(f"  输出:  {list(output_cross.shape)}")
    print(f"  权重:  {list(weights_cross.shape)}")
    assert output_cross.shape == (batch_size, tgt_len, d_model), f"Cross output shape 错误"
    assert weights_cross.shape == (batch_size, num_heads, tgt_len, src_len)
    print(f"  [PASS] 交叉注意力 shape 正确")

    # 带因果掩码
    causal_mask = generate_causal_mask(seq_len)
    output_masked, weights_masked = mha(x, x, x, mask=causal_mask)
    assert output_masked.shape == (batch_size, seq_len, d_model)
    print(f"\n因果掩码注意力:")
    print(f"  输出: {list(output_masked.shape)}")
    print(f"  [PASS] 因果掩码 shape 正确")

    # 参数量
    param_count = sum(p.numel() for p in mha.parameters())
    print(f"\n参数量: {param_count:,}")
    print(f"  W_q: {d_model * d_model + d_model:,}")
    print(f"  W_k: {d_model * d_model + d_model:,}")
    print(f"  W_v: {d_model * d_model + d_model:,}")
    print(f"  W_o: {d_model * d_model + d_model:,}")
    print()


# ============================================================
# 4. 与 PyTorch nn.MultiheadAttention 对比
# ============================================================
def compare_with_pytorch():
    print("=" * 60)
    print("4. 与 nn.MultiheadAttention 对比")
    print("=" * 60)

    d_model = 64
    num_heads = 8
    batch_size = 2
    seq_len = 5

    custom_mha = MultiHeadAttention(d_model, num_heads, dropout=0.0)
    pytorch_mha = nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=0.0)

    x = torch.randn(batch_size, seq_len, d_model)

    # 自定义实现
    custom_out, custom_w = custom_mha(x, x, x)

    # PyTorch 实现
    pt_out, pt_w = pytorch_mha(x, x, x)

    print(f"自定义输出 shape: {list(custom_out.shape)}")
    print(f"PyTorch输出 shape: {list(pt_out.shape)}")
    print(f"Shape 一致: {custom_out.shape == pt_out.shape}")

    print(f"\n自定义权重 shape: {list(custom_w.shape)}")
    print(f"PyTorch权重 shape: {list(pt_w.shape)}")
    print(f"Shape 一致: {custom_w.shape == pt_w.shape}")

    # 参数量对比
    custom_params = sum(p.numel() for p in custom_mha.parameters())
    pt_params = sum(p.numel() for p in pytorch_mha.parameters())
    print(f"\n自定义参数量: {custom_params:,}")
    print(f"PyTorch参数量: {pt_params:,}")
    print(f"(PyTorch 多了 in_proj_bias 等额外参数)")
    print()


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)

    test_scaled_attention()
    test_causal_mask()
    test_multihead_attention()
    compare_with_pytorch()

    print("所有测试完成！")
