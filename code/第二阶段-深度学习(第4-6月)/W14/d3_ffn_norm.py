"""
W14-D3: PositionWiseFFN + LayerNorm + ResidualConnection + Unit Tests
======================================================================
Transformer 从零实现 - 第3步: FFN、LayerNorm 和残差连接。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 1. Position-Wise Feed-Forward Network
# ============================================================
class PositionWiseFFN(nn.Module):
    """
    位置前馈网络

    FFN(x) = W2 * GELU(W1 * x + b1) + b2

    逐位置独立处理 (每个位置使用相同权重)
    中间维度 d_ff 通常是 d_model 的 4 倍
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        参数: x [batch, seq_len, d_model]
        返回: [batch, seq_len, d_model]
        """
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


def test_ffn():
    print("=" * 60)
    print("1. PositionWiseFFN 测试")
    print("=" * 60)

    d_model = 64
    d_ff = 256
    batch_size = 2
    seq_len = 5

    ffn = PositionWiseFFN(d_model, d_ff, dropout=0.0)

    x = torch.randn(batch_size, seq_len, d_model)
    out = ffn(x)

    # Shape 测试
    assert out.shape == (batch_size, seq_len, d_model), \
        f"FFN output shape 错误: {out.shape}"
    print(f"[PASS] Shape: {list(x.shape)} -> {list(out.shape)}")

    # 逐位置独立测试
    out_pos0 = ffn(x[:, 0:1, :])
    diff = (out[:, 0:1, :] - out_pos0).abs().max().item()
    assert diff < 1e-6, f"FFN 逐位置处理失败: diff={diff}"
    print(f"[PASS] 逐位置独立处理 (diff={diff:.2e})")

    # 参数量
    params = sum(p.numel() for p in ffn.parameters())
    expected = (d_model * d_ff + d_ff) + (d_ff * d_model + d_model)
    assert params == expected, f"参数量不符: {params} vs {expected}"
    print(f"[PASS] 参数量: {params:,} (fc1: {d_model}*{d_ff}+{d_ff}, fc2: {d_ff}*{d_model}+{d_model})")
    print()


# ============================================================
# 2. Layer Normalization
# ============================================================
class LayerNorm(nn.Module):
    """
    Layer Normalization

    对每个样本的 feature 维度归一化
    mean, var 沿最后一个维度计算
    可学习参数 gamma (缩放) 和 beta (偏移)
    """

    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        """
        参数: x [..., d_model]
        返回: 归一化后的 x, shape 不变
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


def test_layernorm():
    print("=" * 60)
    print("2. LayerNorm 测试")
    print("=" * 60)

    d_model = 64
    batch_size = 2
    seq_len = 5

    custom_ln = LayerNorm(d_model)
    pytorch_ln = nn.LayerNorm(d_model)

    # 同步参数
    pytorch_ln.weight = custom_ln.gamma
    pytorch_ln.bias = custom_ln.beta

    x = torch.randn(batch_size, seq_len, d_model) * 5 + 3

    # Shape 测试
    out = custom_ln(x)
    assert out.shape == x.shape, f"LayerNorm shape 错误: {out.shape}"
    print(f"[PASS] Shape: {list(x.shape)} -> {list(out.shape)}")

    # 归一化验证: 均值接近0，方差接近1
    mean = out.mean(dim=-1)
    var = out.var(dim=-1, unbiased=False)
    assert mean.abs().max() < 1e-4, f"均值不为0: {mean.abs().max()}"
    assert (var - 1.0).abs().max() < 1e-2, f"方差不为1: {(var - 1.0).abs().max()}"
    print(f"[PASS] 归一化验证: 均值={mean.abs().max():.6f}, "
          f"方差偏差={(var - 1.0).abs().max():.6f}")

    # 与 PyTorch 对比
    out_pt = pytorch_ln(x)
    diff = (out - out_pt).abs().max().item()
    assert diff < 1e-5, f"与 PyTorch LayerNorm 差异过大: {diff}"
    print(f"[PASS] 与 nn.LayerNorm 一致 (diff={diff:.2e})")

    # gamma, beta 效果
    ln_no_params = LayerNorm(d_model)
    ln_no_params.gamma.data.fill_(1.0)
    ln_no_params.beta.data.fill_(0.0)
    out_noparam = ln_no_params(x)
    # 无参数时就是纯归一化
    print(f"[PASS] gamma=1, beta=0 时输出均值为0")
    print()


# ============================================================
# 3. Residual Connection (残差连接)
# ============================================================
class ResidualConnection(nn.Module):
    """
    残差连接 + LayerNorm + Dropout

    支持 Pre-LN 和 Post-LN 两种模式:

    Pre-LN (GPT 风格, 更稳定):
        output = x + Dropout(sublayer(LayerNorm(x)))

    Post-LN (原始 Transformer):
        output = LayerNorm(x + Dropout(sublayer(x)))
    """

    def __init__(self, d_model, dropout=0.1, pre_norm=True):
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.pre_norm = pre_norm

    def forward(self, x, sublayer_output):
        """
        参数:
            x: 原始输入 [batch, seq_len, d_model]
            sublayer_output: 子层输出 (已应用或未应用 norm)
        返回: [batch, seq_len, d_model]
        """
        if self.pre_norm:
            # Pre-LN: x + Dropout(sublayer(LayerNorm(x)))
            # 注意: 调用方应传入 sublayer(norm(x)) 的结果
            return x + self.dropout(sublayer_output)
        else:
            # Post-LN: LayerNorm(x + Dropout(sublayer(x)))
            return self.norm(x + self.dropout(sublayer_output))


def test_residual():
    print("=" * 60)
    print("3. ResidualConnection 测试")
    print("=" * 60)

    d_model = 64
    batch_size = 2
    seq_len = 5

    # Pre-LN 模式
    residual_pre = ResidualConnection(d_model, dropout=0.0, pre_norm=True)
    norm = LayerNorm(d_model)
    sublayer = nn.Linear(d_model, d_model)

    x = torch.randn(batch_size, seq_len, d_model)

    # Pre-LN: sublayer 需要先 norm 再传入
    sublayer_input = norm(x)
    sublayer_out = sublayer(sublayer_input)
    output_pre = residual_pre(x, sublayer_out)

    assert output_pre.shape == (batch_size, seq_len, d_model), \
        f"Pre-LN shape 错误: {output_pre.shape}"
    print(f"[PASS] Pre-LN shape: {list(output_pre.shape)}")

    # 验证残差: output = x + sublayer(norm(x))
    expected = x + sublayer(norm(x))
    diff = (output_pre - expected).abs().max().item()
    assert diff < 1e-6, f"Pre-LN 残差计算错误: diff={diff}"
    print(f"[PASS] Pre-LN 残差验证 (diff={diff:.2e})")

    # Post-LN 模式
    residual_post = ResidualConnection(d_model, dropout=0.0, pre_norm=False)
    sublayer_out_post = sublayer(x)  # 不先 norm
    output_post = residual_post(x, sublayer_out_post)

    assert output_post.shape == (batch_size, seq_len, d_model)
    print(f"[PASS] Post-LN shape: {list(output_post.shape)}")

    # 梯度流测试
    x_param = x.clone().requires_grad_(True)
    sublayer_out_grad = sublayer(norm(x_param))
    out_grad = residual_pre(x_param, sublayer_out_grad)
    out_grad.sum().backward()

    assert x_param.grad is not None, "梯度为 None"
    grad_norm = x_param.grad.norm().item()
    assert grad_norm > 0, "梯度为零"
    print(f"[PASS] 梯度流通 (grad_norm={grad_norm:.4f})")
    print()


# ============================================================
# 4. 综合测试: FFN + LayerNorm + Residual
# ============================================================
def test_combined():
    print("=" * 60)
    print("4. FFN + LayerNorm + Residual 综合测试")
    print("=" * 60)

    d_model = 64
    d_ff = 256
    batch_size = 2
    seq_len = 5

    # 构建: x -> LayerNorm -> FFN -> Residual
    norm = LayerNorm(d_model)
    ffn = PositionWiseFFN(d_model, d_ff, dropout=0.0)
    residual = ResidualConnection(d_model, dropout=0.0, pre_norm=True)

    x = torch.randn(batch_size, seq_len, d_model)

    # 前向传播
    normed = norm(x)
    ffn_out = ffn(normed)
    output = residual(x, ffn_out)

    assert output.shape == (batch_size, seq_len, d_model)
    print(f"[PASS] 组合模块 shape: {list(output.shape)}")

    # 验证 output = x + FFN(LayerNorm(x))
    expected = x + ffn(norm(x))
    diff = (output - expected).abs().max().item()
    assert diff < 1e-5, f"组合计算错误: diff={diff}"
    print(f"[PASS] 组合计算正确 (diff={diff:.2e})")

    # 梯度测试
    x_param = x.clone().requires_grad_(True)
    out = residual(x_param, ffn(norm(x_param)))
    out.sum().backward()

    assert x_param.grad is not None
    print(f"[PASS] 梯度流通")
    print(f"  输入梯度范数: {x_param.grad.norm():.4f}")

    # 参数量
    total_params = (
        sum(p.numel() for p in norm.parameters()) +
        sum(p.numel() for p in ffn.parameters()) +
        sum(p.numel() for p in residual.parameters())
    )
    print(f"  总参数量: {total_params:,}")
    print()


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)

    test_ffn()
    test_layernorm()
    test_residual()
    test_combined()

    print("所有单元测试通过！")
