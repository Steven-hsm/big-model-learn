"""
W14-D4: EncoderLayer + Encoder (stack of N layers) + Shape Verification + Parameter Counting
===============================================================================================
Transformer 从零实现 - 第4步: 编码器。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 导入之前实现的组件
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d2_attention_impl import MultiHeadAttention, generate_causal_mask
from d3_ffn_norm import PositionWiseFFN, LayerNorm, ResidualConnection


# ============================================================
# 1. EncoderLayer (编码器单层)
# ============================================================
class EncoderLayer(nn.Module):
    """
    Transformer 编码器层

    结构 (Pre-LN):
        x -> LayerNorm -> Multi-Head Self-Attention -> Residual ->
        -> LayerNorm -> FFN -> Residual -> output
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # Self-Attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # FFN
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        """
        参数:
            x: [batch, seq_len, d_model]
            src_mask: 可选的源掩码
        返回:
            [batch, seq_len, d_model]
        """
        # 1. Self-Attention + Residual (Pre-LN)
        normed = self.norm1(x)
        attn_out, _ = self.self_attn(normed, normed, normed, src_mask)
        x = x + self.dropout1(attn_out)

        # 2. FFN + Residual (Pre-LN)
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + self.dropout2(ffn_out)

        return x


def test_encoder_layer():
    print("=" * 60)
    print("1. EncoderLayer 测试")
    print("=" * 60)

    d_model = 64
    num_heads = 8
    d_ff = 256
    batch_size = 2
    seq_len = 5

    layer = EncoderLayer(d_model, num_heads, d_ff, dropout=0.0)

    x = torch.randn(batch_size, seq_len, d_model)
    out = layer(x)

    # Shape 测试
    assert out.shape == (batch_size, seq_len, d_model), \
        f"EncoderLayer output shape 错误: {out.shape}"
    print(f"[PASS] Shape: {list(x.shape)} -> {list(out.shape)}")

    # 参数量
    params = sum(p.numel() for p in layer.parameters())
    print(f"参数量: {params:,}")

    # 参数分布
    attn_params = sum(p.numel() for n, p in layer.named_parameters() if "attn" in n)
    ffn_params = sum(p.numel() for n, p in layer.named_parameters() if "ffn" in n)
    norm_params = sum(p.numel() for n, p in layer.named_parameters() if "norm" in n)
    print(f"  Attention: {attn_params:,}")
    print(f"  FFN:       {ffn_params:,}")
    print(f"  LayerNorm: {norm_params:,}")

    # 梯度测试
    x_grad = x.clone().requires_grad_(True)
    out = layer(x_grad)
    out.sum().backward()
    assert x_grad.grad is not None, "梯度为 None"
    print(f"[PASS] 梯度流通 (grad_norm={x_grad.grad.norm():.4f})")
    print()


# ============================================================
# 2. Encoder (N 层堆叠)
# ============================================================
class Encoder(nn.Module):
    """
    Transformer 编码器: N 个 EncoderLayer 堆叠 + 最终 LayerNorm
    """

    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = LayerNorm(d_model)

    def forward(self, x, src_mask=None):
        """
        参数:
            x: [batch, seq_len, d_model]  (embedding 输出)
            src_mask: 可选掩码
        返回:
            [batch, seq_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, src_mask)

        return self.final_norm(x)


def test_encoder():
    print("=" * 60)
    print("2. Encoder (多层堆叠) 测试")
    print("=" * 60)

    d_model = 64
    num_heads = 8
    d_ff = 256
    num_layers = 3
    batch_size = 2
    seq_len = 5

    encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout=0.0)

    x = torch.randn(batch_size, seq_len, d_model)
    out = encoder(x)

    # Shape 测试
    assert out.shape == (batch_size, seq_len, d_model), \
        f"Encoder output shape 错误: {out.shape}"
    print(f"[PASS] Shape: {list(x.shape)} -> {list(out.shape)}")

    # 参数量
    params = sum(p.numel() for p in encoder.parameters())
    per_layer = params // num_layers
    print(f"参数量: {params:,} (每层约 {per_layer:,})")

    # 逐层输出 (验证中间 shape)
    print(f"\n逐层输出:")
    hidden = x
    for i, layer in enumerate(encoder.layers):
        hidden = layer(hidden)
        print(f"  Layer {i}: {list(hidden.shape)}, norm={hidden.norm():.4f}")

    # 不同层数对比
    print(f"\n不同层数参数量:")
    for n_layers in [1, 2, 4, 6, 12]:
        enc = Encoder(n_layers, d_model, num_heads, d_ff)
        p = sum(p.numel() for p in enc.parameters())
        print(f"  {n_layers:>2} 层: {p:>10,} 参数")
    print()


# ============================================================
# 3. Shape 验证 (全流程)
# ============================================================
def test_shape_verification():
    print("=" * 60)
    print("3. Encoder Shape 全流程验证")
    print("=" * 60)

    # 配置
    vocab_size = 100
    d_model = 64
    num_heads = 8
    d_ff = 256
    num_layers = 2
    batch_size = 3
    seq_len = 8

    # 构建
    embedding = nn.Embedding(vocab_size, d_model)
    pe = nn.Parameter(torch.zeros(1, seq_len, d_model))  # 简化 PE
    encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout=0.0)

    # 输入
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"输入 token ids: {list(input_ids.shape)}")

    # Embedding
    emb = embedding(input_ids) * math.sqrt(d_model)
    print(f"Token Embedding: {list(emb.shape)}")
    assert emb.shape == (batch_size, seq_len, d_model)

    # + Positional Encoding
    x = emb + pe[:, :seq_len, :]
    print(f"+ Position: {list(x.shape)}")
    assert x.shape == (batch_size, seq_len, d_model)

    # Encoder
    enc_out = encoder(x)
    print(f"Encoder 输出: {list(enc_out.shape)}")
    assert enc_out.shape == (batch_size, seq_len, d_model)

    # 变长序列 (padding mask)
    print(f"\nPadding Mask 测试:")
    lengths = torch.tensor([6, 8, 4])
    mask = torch.arange(seq_len).unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
    enc_out_masked = encoder(x, src_mask=mask)
    print(f"  Masked 输出: {list(enc_out_masked.shape)}")
    assert enc_out_masked.shape == (batch_size, seq_len, d_model)

    print(f"\n[PASS] 所有 Shape 验证通过!")
    print()


# ============================================================
# 4. 参数量分析 (原始 Transformer 配置)
# ============================================================
def analyze_parameters():
    print("=" * 60)
    print("4. 参数量分析 (不同配置)")
    print("=" * 60)

    configs = [
        ("Tiny",      64,  4,  256,  2),
        ("Small",    128,  4,  512,  4),
        ("Base",     512,  8, 2048,  6),
        ("Large",   1024, 16, 4096, 12),
    ]

    print(f"{'配置':>10} | {'d_model':>7} | {'heads':>5} | {'d_ff':>5} | "
          f"{'layers':>6} | {'参数量':>12} | {'每层参数':>10}")
    print("-" * 75)

    for name, d_model, num_heads, d_ff, num_layers in configs:
        encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout=0.0)
        params = sum(p.numel() for p in encoder.parameters())
        per_layer = params // num_layers
        print(f"{name:>10} | {d_model:>7} | {num_heads:>5} | {d_ff:>5} | "
              f"{num_layers:>6} | {params:>12,} | {per_layer:>10,}")

    print()


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)

    test_encoder_layer()
    test_encoder()
    test_shape_verification()
    analyze_parameters()

    print("所有测试完成！")
