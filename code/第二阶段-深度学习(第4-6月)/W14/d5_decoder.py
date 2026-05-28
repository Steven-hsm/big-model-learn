"""
W14-D5: DecoderLayer (masked self-attn + cross-attn + FFN) + Decoder (N layers)
=================================================================================
Transformer 从零实现 - 第5步: 解码器。
"""

import torch
import torch.nn as nn
import math

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d2_attention_impl import MultiHeadAttention, generate_causal_mask
from d3_ffn_norm import PositionWiseFFN, LayerNorm


# ============================================================
# 1. DecoderLayer (解码器单层)
# ============================================================
class DecoderLayer(nn.Module):
    """
    Transformer 解码器层

    结构 (Pre-LN):
        x -> LayerNorm -> Masked Self-Attention -> Residual ->
        -> LayerNorm -> Cross-Attention -> Residual ->
        -> LayerNorm -> FFN -> Residual -> output
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # 1. Masked Self-Attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # 2. Cross-Attention
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        # 3. FFN
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        """
        参数:
            x: [batch, tgt_len, d_model] 解码器输入
            enc_output: [batch, src_len, d_model] 编码器输出
            tgt_mask: 因果掩码 [1, 1, tgt_len, tgt_len]
            src_mask: 源 padding 掩码
        返回:
            [batch, tgt_len, d_model]
        """
        # 1. Masked Self-Attention + Residual
        normed = self.norm1(x)
        self_attn_out, self_attn_weights = self.self_attn(
            normed, normed, normed, tgt_mask
        )
        x = x + self.dropout1(self_attn_out)

        # 2. Cross-Attention + Residual
        # Q 来自解码器, K/V 来自编码器
        normed = self.norm2(x)
        cross_attn_out, cross_attn_weights = self.cross_attn(
            normed, enc_output, enc_output, src_mask
        )
        x = x + self.dropout2(cross_attn_out)

        # 3. FFN + Residual
        normed = self.norm3(x)
        ffn_out = self.ffn(normed)
        x = x + self.dropout3(ffn_out)

        return x


def test_decoder_layer():
    print("=" * 60)
    print("1. DecoderLayer 测试")
    print("=" * 60)

    d_model = 64
    num_heads = 8
    d_ff = 256
    batch_size = 2
    tgt_len = 5
    src_len = 8

    layer = DecoderLayer(d_model, num_heads, d_ff, dropout=0.0)

    x = torch.randn(batch_size, tgt_len, d_model)
    enc_output = torch.randn(batch_size, src_len, d_model)

    # 因果掩码
    tgt_mask = generate_causal_mask(tgt_len)

    out = layer(x, enc_output, tgt_mask=tgt_mask)

    # Shape 测试
    assert out.shape == (batch_size, tgt_len, d_model), \
        f"DecoderLayer output shape 错误: {out.shape}"
    print(f"[PASS] Shape: x{list(x.shape)} + enc{list(enc_output.shape)} -> {list(out.shape)}")

    # 参数量
    params = sum(p.numel() for p in layer.parameters())
    self_attn_p = sum(p.numel() for n, p in layer.named_parameters() if "self_attn" in n)
    cross_attn_p = sum(p.numel() for n, p in layer.named_parameters() if "cross_attn" in n)
    ffn_p = sum(p.numel() for n, p in layer.named_parameters() if "ffn" in n)
    print(f"参数量: {params:,}")
    print(f"  Self-Attention: {self_attn_p:,}")
    print(f"  Cross-Attention: {cross_attn_p:,}")
    print(f"  FFN: {ffn_p:,}")

    # 无因果掩码测试
    out_no_mask = layer(x, enc_output)
    assert out_no_mask.shape == (batch_size, tgt_len, d_model)
    print(f"[PASS] 无掩码也能正常工作")

    # 梯度测试
    x_grad = x.clone().requires_grad_(True)
    enc_grad = enc_output.clone().requires_grad_(True)
    out = layer(x_grad, enc_grad, tgt_mask=tgt_mask)
    out.sum().backward()
    assert x_grad.grad is not None, "x 梯度为 None"
    assert enc_grad.grad is not None, "enc_output 梯度为 None"
    print(f"[PASS] 梯度流通 (x: {x_grad.grad.norm():.4f}, enc: {enc_grad.grad.norm():.4f})")
    print()


# ============================================================
# 2. Decoder (N 层堆叠)
# ============================================================
class Decoder(nn.Module):
    """
    Transformer 解码器: N 个 DecoderLayer 堆叠 + 最终 LayerNorm
    """

    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = LayerNorm(d_model)

    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        """
        参数:
            x: [batch, tgt_len, d_model]
            enc_output: [batch, src_len, d_model]
            tgt_mask: 因果掩码
            src_mask: 源 padding 掩码
        返回:
            [batch, tgt_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, src_mask)

        return self.final_norm(x)


def test_decoder():
    print("=" * 60)
    print("2. Decoder (多层堆叠) 测试")
    print("=" * 60)

    d_model = 64
    num_heads = 8
    d_ff = 256
    num_layers = 3
    batch_size = 2
    tgt_len = 5
    src_len = 8

    decoder = Decoder(num_layers, d_model, num_heads, d_ff, dropout=0.0)

    x = torch.randn(batch_size, tgt_len, d_model)
    enc_output = torch.randn(batch_size, src_len, d_model)
    tgt_mask = generate_causal_mask(tgt_len)

    out = decoder(x, enc_output, tgt_mask=tgt_mask)

    # Shape 测试
    assert out.shape == (batch_size, tgt_len, d_model), \
        f"Decoder output shape 错误: {out.shape}"
    print(f"[PASS] Shape: {list(out.shape)}")

    # 参数量
    params = sum(p.numel() for p in decoder.parameters())
    per_layer = params // num_layers
    print(f"参数量: {params:,} (每层约 {per_layer:,})")

    # 逐层输出
    print(f"\n逐层输出:")
    hidden = x
    for i, layer in enumerate(decoder.layers):
        hidden = layer(hidden, enc_output, tgt_mask=tgt_mask)
        print(f"  Layer {i}: {list(hidden.shape)}, norm={hidden.norm():.4f}")

    # 不同层数对比
    print(f"\n不同层数参数量:")
    for n_layers in [1, 2, 4, 6]:
        dec = Decoder(n_layers, d_model, num_heads, d_ff)
        p = sum(p.numel() for p in dec.parameters())
        print(f"  {n_layers:>2} 层: {p:>10,} 参数")
    print()


# ============================================================
# 3. Shape 验证 (Encoder + Decoder 联合)
# ============================================================
def test_encoder_decoder_shapes():
    print("=" * 60)
    print("3. Encoder + Decoder 联合 Shape 验证")
    print("=" * 60)

    from d4_encoder import Encoder

    d_model = 64
    num_heads = 8
    d_ff = 256
    num_layers = 2
    batch_size = 2
    src_len = 10
    tgt_len = 6

    encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout=0.0)
    decoder = Decoder(num_layers, d_model, num_heads, d_ff, dropout=0.0)

    # 输入
    src_emb = torch.randn(batch_size, src_len, d_model)
    tgt_emb = torch.randn(batch_size, tgt_len, d_model)
    tgt_mask = generate_causal_mask(tgt_len)

    print(f"源 Embedding:   {list(src_emb.shape)}")
    print(f"目标 Embedding: {list(tgt_emb.shape)}")
    print(f"Causal Mask:    {list(tgt_mask.shape)}")

    # Encode
    enc_out = encoder(src_emb)
    print(f"\nEncoder 输出:   {list(enc_out.shape)}")
    assert enc_out.shape == (batch_size, src_len, d_model)

    # Decode
    dec_out = decoder(tgt_emb, enc_out, tgt_mask=tgt_mask)
    print(f"Decoder 输出:   {list(dec_out.shape)}")
    assert dec_out.shape == (batch_size, tgt_len, d_model)

    # Projection (模拟输出层)
    vocab_size = 50
    projection = nn.Linear(d_model, vocab_size)
    logits = projection(dec_out)
    print(f"Logits:         {list(logits.shape)}")
    assert logits.shape == (batch_size, tgt_len, vocab_size)

    print(f"\n[PASS] Encoder + Decoder 联合 Shape 全部正确!")
    print()


# ============================================================
# 4. 因果掩码验证
# ============================================================
def test_causal_mask_effect():
    print("=" * 60)
    print("4. 因果掩码效果验证")
    print("=" * 60)

    d_model = 32
    num_heads = 4
    d_ff = 128
    batch_size = 1
    tgt_len = 5

    layer = DecoderLayer(d_model, num_heads, d_ff, dropout=0.0)

    # 让编码器输出和目标输入有特定模式
    enc_output = torch.randn(batch_size, 8, d_model)

    # 使用全 1 输入 (这样注意力权重完全由位置决定)
    x = torch.ones(batch_size, tgt_len, d_model) * 0.1
    for i in range(tgt_len):
        x[0, i, 0] = float(i)  # 每个位置第一个维度不同

    # 无掩码
    out_no_mask = layer(x, enc_output, tgt_mask=None)

    # 有因果掩码
    tgt_mask = generate_causal_mask(tgt_len)
    out_with_mask = layer(x, enc_output, tgt_mask=tgt_mask)

    # 有掩码和无掩码应该不同
    diff = (out_no_mask - out_with_mask).abs().sum().item()
    print(f"有掩码 vs 无掩码输出差异: {diff:.4f}")
    print(f"[PASS] 因果掩码确实影响输出 (差异 > 0: {diff > 0})")

    # 验证: 位置 i 只依赖于位置 0..i
    # 修改位置 j > i 的输入，位置 i 的输出不应该变
    print(f"\n因果性验证:")
    x_orig = torch.randn(batch_size, tgt_len, d_model)
    out_orig = layer(x_orig, enc_output, tgt_mask=tgt_mask)

    for i in range(tgt_len):
        # 修改位置 i+1 之后的内容
        x_modified = x_orig.clone()
        x_modified[0, i + 1:, :] = torch.randn(tgt_len - i - 1, d_model) * 100
        out_modified = layer(x_modified, enc_output, tgt_mask=tgt_mask)

        # 位置 0..i 的输出应该不变
        diff = (out_orig[0, :i + 1] - out_modified[0, :i + 1]).abs().max().item()
        print(f"  修改位置 {i + 1}..{tgt_len - 1} 后, 位置 0..{i} 输出变化: {diff:.6f}")
        assert diff < 1e-4, f"位置 {i} 输出不应当变化: diff={diff}"

    print(f"[PASS] 因果性验证通过: 每个位置不受未来位置影响")
    print()


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)

    test_decoder_layer()
    test_decoder()
    test_encoder_decoder_shapes()
    test_causal_mask_effect()

    print("所有测试完成！")
