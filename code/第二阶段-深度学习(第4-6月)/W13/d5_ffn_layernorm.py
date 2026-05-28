"""
W13-D5: Feed-Forward Network, LayerNorm, Residual Connection, Transformer Block
================================================================================
Transformer 的核心组件和完整 Block。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 1. Feed-Forward Network (FFN)
# ============================================================
class PositionWiseFFN(nn.Module):
    """
    位置前馈网络: 两个线性层 + GELU 激活

    FFN(x) = W2 * GELU(W1 * x + b1) + b2
    中间维度通常是 d_model 的 4 倍
    """

    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 4

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        参数: x [batch, seq_len, d_model]
        返回: [batch, seq_len, d_model]
        """
        x = self.fc1(x)            # [batch, seq_len, d_ff]
        x = F.gelu(x)              # GELU 激活
        x = self.dropout(x)
        x = self.fc2(x)            # [batch, seq_len, d_model]
        return x


def demo_ffn():
    print("=" * 60)
    print("1. Feed-Forward Network (FFN)")
    print("=" * 60)

    d_model = 64
    d_ff = 256
    batch_size = 2
    seq_len = 5

    ffn = PositionWiseFFN(d_model, d_ff)

    x = torch.randn(batch_size, seq_len, d_model)
    out = ffn(x)

    print(f"输入:     {list(x.shape)}")
    print(f"中间层:   [{batch_size}, {seq_len}, {d_ff}]")
    print(f"输出:     {list(out.shape)}")
    print(f"参数量:   {sum(p.numel() for p in ffn.parameters()):,}")
    print()

    # 展示逐位置处理
    print("FFN 对每个位置独立处理 (相同权重):")
    pos0_out = ffn(x[:, 0:1, :])
    pos1_out = ffn(x[:, 1:2, :])
    full_out = ffn(x)

    diff0 = (full_out[:, 0:1, :] - pos0_out).abs().max().item()
    diff1 = (full_out[:, 1:2, :] - pos1_out).abs().max().item()
    print(f"  位置0 单独处理 vs 整体处理差异: {diff0:.8f}")
    print(f"  位置1 单独处理 vs 整体处理差异: {diff1:.08f}")
    print(f"  (差异为0说明确实是逐位置处理)")
    print()

    # ReLU vs GELU
    print("激活函数对比:")
    x_act = torch.linspace(-3, 3, 7)
    print(f"  x:     {x_act.tolist()}")
    print(f"  ReLU:  {F.relu(x_act).tolist()}")
    print(f"  GELU:  {F.gelu(x_act).tolist()}")
    print(f"  GELU 在 x<0 时有小的非零输出，更加平滑")
    print()


# ============================================================
# 2. LayerNorm 从零实现
# ============================================================
class LayerNorm(nn.Module):
    """
    Layer Normalization 实现

    与 BatchNorm 不同:
    - BatchNorm: 沿 batch 维度归一化 (每个特征)
    - LayerNorm: 沿 feature 维度归一化 (每个样本)

    对 [batch, seq_len, d_model] 的输入:
    - 在最后一个维度 d_model 上归一化
    """

    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        # 可学习参数
        self.gamma = nn.Parameter(torch.ones(d_model))  # 缩放
        self.beta = nn.Parameter(torch.zeros(d_model))   # 偏移

    def forward(self, x):
        """
        参数: x [..., d_model]
        返回: 归一化后的 x, shape 不变
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


def demo_layernorm():
    print("=" * 60)
    print("2. Layer Normalization")
    print("=" * 60)

    d_model = 64
    batch_size = 2
    seq_len = 5

    x = torch.randn(batch_size, seq_len, d_model) * 5 + 3  # 偏移和缩放

    # 自定义实现
    custom_ln = LayerNorm(d_model)
    out_custom = custom_ln(x)

    # PyTorch 实现
    pytorch_ln = nn.LayerNorm(d_model)
    # 复制参数
    pytorch_ln.weight = custom_ln.gamma
    pytorch_ln.bias = custom_ln.beta
    out_pytorch = pytorch_ln(x)

    print(f"输入:     {list(x.shape)}")
    print(f"  均值: {x.mean(dim=-1)[0, 0].item():.4f}")
    print(f"  方差: {x.var(dim=-1)[0, 0].item():.4f}")

    print(f"\n自定义 LN 输出:")
    print(f"  均值: {out_custom.mean(dim=-1)[0, 0].item():.6f} (接近0)")
    print(f"  方差: {out_custom.var(dim=-1)[0, 0].item():.6f} (接近1)")

    diff = (out_custom - out_pytorch).abs().max().item()
    print(f"\n与 nn.LayerNorm 差异: {diff:.10f} (应接近0)")
    print()

    # LayerNorm vs BatchNorm
    print("LayerNorm vs BatchNorm:")
    print(f"  {'':>12} | {'LayerNorm':>15} | {'BatchNorm':>15}")
    print(f"  {'归一化维度':>12} | {'Feature维度':>15} | {'Batch维度':>15}")
    print(f"  {'依赖batch':>12} | {'否':>15} | {'是':>15}")
    print(f"  {'推理行为':>12} | {'与训练一致':>15} | {'需要running_mean':>15}")
    print(f"  {'适用场景':>12} | {'NLP/Transformer':>15} | {'CNN':>15}")
    print()


# ============================================================
# 3. 残差连接 (Residual Connection)
# ============================================================
class ResidualConnection(nn.Module):
    """残差连接 + LayerNorm + Dropout"""

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_output):
        """
        Post-LN:  output = norm(x + dropout(sublayer(x)))
        Pre-LN:   output = x + dropout(sublayer(norm(x)))

        这里使用 Pre-LN (现代 Transformer 常用)
        """
        return x + self.dropout(sublayer_output)


def demo_residual():
    print("=" * 60)
    print("3. 残差连接")
    print("=" * 60)

    d_model = 64
    batch_size = 2
    seq_len = 5

    x = torch.randn(batch_size, seq_len, d_model)
    norm = nn.LayerNorm(d_model)
    sublayer = nn.Linear(d_model, d_model)

    # Pre-LN (GPT 风格)
    # output = x + sublayer(norm(x))
    sublayer_out = sublayer(norm(x))
    output_pre_ln = x + sublayer_out

    # Post-LN (原始 Transformer)
    # output = norm(x + sublayer(x))
    output_post_ln = norm(x + sublayer(x))

    print(f"输入: {list(x.shape)}")
    print(f"Pre-LN 输出:  {list(output_pre_ln.shape)}")
    print(f"Post-LN 输出: {list(output_post_ln.shape)}")

    print(f"\n残差连接的好处:")
    print(f"  1. 梯度可以直接流过 (缓解梯度消失)")
    print(f"  2. 模型可以学习恒等映射 (至少不比浅层差)")
    print(f"  3. 训练更稳定")

    # 演示梯度流
    x_param = x.clone().requires_grad_(True)
    sublayer_out = sublayer(norm(x_param))
    out = x_param + sublayer_out
    out.sum().backward()

    print(f"\n  残差路径的梯度 = 1 (直接传递)")
    print(f"  输入梯度范数: {x_param.grad.norm().item():.4f}")
    print()


# ============================================================
# 4. 完整 Transformer Block (Encoder)
# ============================================================
class TransformerEncoderBlock(nn.Module):
    """
    完整的 Transformer 编码器块

    结构 (Pre-LN):
        x -> LayerNorm -> Multi-Head Attention -> Residual ->
        -> LayerNorm -> FFN -> Residual -> output
    """

    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1):
        super().__init__()
        # 多头注意力
        self.attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        # FFN
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)
        # Layer Norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        参数: x [batch, seq_len, d_model]
        返回: [batch, seq_len, d_model]
        """
        # 1. Multi-Head Attention + Residual
        attn_out, _ = self.attention(
            self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=mask
        )
        x = x + self.dropout1(attn_out)

        # 2. FFN + Residual
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout2(ffn_out)

        return x


def demo_transformer_block():
    print("=" * 60)
    print("4. 完整 Transformer Block")
    print("=" * 60)

    d_model = 512
    num_heads = 8
    batch_size = 2
    seq_len = 10

    block = TransformerEncoderBlock(d_model, num_heads, dropout=0.1)

    x = torch.randn(batch_size, seq_len, d_model)
    out = block(x)

    print(f"输入:  {list(x.shape)}")
    print(f"输出:  {list(out.shape)}")
    print(f"参数量: {sum(p.numel() for p in block.parameters()):,}")

    # 参数分布
    print(f"\n参数分布:")
    attn_params = sum(p.numel() for n, p in block.named_parameters() if "attention" in n)
    ffn_params = sum(p.numel() for n, p in block.named_parameters() if "ffn" in n)
    norm_params = sum(p.numel() for n, p in block.named_parameters() if "norm" in n)
    print(f"  Attention: {attn_params:>10,} ({attn_params / (attn_params + ffn_params + norm_params):.1%})")
    print(f"  FFN:       {ffn_params:>10,} ({ffn_params / (attn_params + ffn_params + norm_params):.1%})")
    print(f"  LayerNorm: {norm_params:>10,} ({norm_params / (attn_params + ffn_params + norm_params):.1%})")
    print()

    # 堆叠多层
    num_layers = 6
    layers = nn.ModuleList([
        TransformerEncoderBlock(d_model, num_heads) for _ in range(num_layers)
    ])

    total_params = sum(p.numel() for p in layers.parameters())
    print(f"{num_layers} 层 Transformer Encoder:")
    print(f"  总参数量: {total_params:,}")
    print(f"  每层参数: {total_params // num_layers:,}")

    # 前向传播
    hidden = x
    for i, layer in enumerate(layers):
        hidden = layer(hidden)
        print(f"  Layer {i}: output shape {list(hidden.shape)}")

    print()


# ============================================================
# 5. Shape 全流程追踪
# ============================================================
def demo_shape_tracking():
    print("=" * 60)
    print("5. Shape 全流程追踪")
    print("=" * 60)

    d_model = 64
    num_heads = 4
    d_ff = 256
    batch_size = 2
    seq_len = 5

    x = torch.randn(batch_size, seq_len, d_model)
    print(f"输入 x: {list(x.shape)}")

    # Step 1: LayerNorm
    norm = nn.LayerNorm(d_model)
    x_normed = norm(x)
    print(f"\n1. LayerNorm: {list(x_normed.shape)}")

    # Step 2: Multi-Head Attention
    mha = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
    attn_out, attn_weights = mha(x_normed, x_normed, x_normed)
    print(f"2. MultiHead Attention:")
    print(f"   Q, K, V: {list(x_normed.shape)}")
    print(f"   注意力权重: {list(attn_weights.shape)}")
    print(f"   输出: {list(attn_out.shape)}")

    # Step 3: Residual
    x = x + attn_out
    print(f"3. Residual: {list(x.shape)}")

    # Step 4: LayerNorm
    x_normed = norm(x)
    print(f"4. LayerNorm: {list(x_normed.shape)}")

    # Step 5: FFN
    ffn = PositionWiseFFN(d_model, d_ff)
    ffn_out = ffn(x_normed)
    print(f"5. FFN:")
    print(f"   fc1 (d_model -> d_ff): [{batch_size}, {seq_len}, {d_ff}]")
    print(f"   fc2 (d_ff -> d_model): {list(ffn_out.shape)}")

    # Step 6: Residual
    output = x + ffn_out
    print(f"6. Residual: {list(output.shape)}")

    print(f"\n总结: 输入 {list(x.shape)} -> 输出 {list(output.shape)} (shape不变)")
    print()


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)

    demo_ffn()
    demo_layernorm()
    demo_residual()
    demo_transformer_block()
    demo_shape_tracking()

    print("所有演示完成！")
