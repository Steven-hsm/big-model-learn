"""
W13-D3: Multi-Head Attention
=============================
多头注意力的动机、实现和参数分析。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 1. 多头注意力的动机
# ============================================================
def demo_motivation():
    """
    为什么需要多头注意力？
    """
    print("=" * 60)
    print("1. 多头注意力的动机")
    print("=" * 60)

    print("单个注意力头的问题:")
    print("  - 只能学习一种'关注模式'")
    print("  - 类似于只用一个特征来衡量相关性")
    print()
    print("多头注意力的想法:")
    print("  - 让模型同时从多个'角度'关注输入")
    print("  - 每个头学习不同的关注模式:")
    print("    * 头1: 关注语法关系 (主语-谓语)")
    print("    * 头2: 关注指代关系 (代词-名词)")
    print("    * 头3: 关注位置关系 (相邻词)")
    print("    * ...")
    print()
    print("实现方式:")
    print("  1. 将 d_model 维度拆分为 h 个头，每个头维度 d_k = d_model / h")
    print("  2. 每个头独立计算注意力")
    print("  3. 拼接所有头的输出")
    print("  4. 通过线性变换融合")
    print()
    print("公式:")
    print("  MultiHead(Q,K,V) = Concat(head_1, ..., head_h) @ W_o")
    print("  head_i = Attention(Q @ W_q_i, K @ W_k_i, V @ W_v_i)")
    print()


# ============================================================
# 2. 分头 (Split Heads) 过程
# ============================================================
def demo_split_heads():
    """
    演示如何将输入拆分为多个头
    """
    print("=" * 60)
    print("2. 分头过程")
    print("=" * 60)

    batch_size = 2
    seq_len = 5
    d_model = 64
    num_heads = 8
    d_k = d_model // num_heads  # 每个头的维度

    print(f"d_model = {d_model}, num_heads = {num_heads}, d_k = {d_k}")

    # 线性投影
    W_q = nn.Linear(d_model, d_model)
    W_k = nn.Linear(d_model, d_model)
    W_v = nn.Linear(d_model, d_model)

    x = torch.randn(batch_size, seq_len, d_model)
    print(f"\n输入 x: {list(x.shape)}")

    # 投影
    Q = W_q(x)  # [batch, seq_len, d_model]
    K = W_k(x)
    V = W_v(x)
    print(f"投影后 Q: {list(Q.shape)}")

    # 分头: [batch, seq_len, d_model] -> [batch, num_heads, seq_len, d_k]
    Q_heads = Q.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
    K_heads = K.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
    V_heads = V.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)

    print(f"分头后 Q: {list(Q_heads.shape)}")
    print(f"  含义: {batch_size}个batch x {num_heads}个头 x {seq_len}个位置 x {d_k}维")

    # 每个头独立计算注意力
    scores = Q_heads @ K_heads.transpose(-2, -1) / math.sqrt(d_k)
    weights = F.softmax(scores, dim=-1)
    head_outputs = weights @ V_heads  # [batch, num_heads, seq_len, d_k]
    print(f"\n每个头输出: {list(head_outputs.shape)}")

    # 拼接头
    concat = head_outputs.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
    print(f"拼接后: {list(concat.shape)}")

    # 最终投影
    W_o = nn.Linear(d_model, d_model)
    output = W_o(concat)
    print(f"最终输出: {list(output.shape)}")
    print()


# ============================================================
# 3. 完整 PyTorch 实现
# ============================================================
class MultiHeadAttention(nn.Module):
    """多头注意力实现"""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 线性投影层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        参数:
            query: [batch, seq_len_q, d_model]
            key:   [batch, seq_len_k, d_model]
            value: [batch, seq_len_k, d_model]
            mask:  [batch, seq_len_q, seq_len_k] 或 [seq_len_q, seq_len_k]
        返回:
            output: [batch, seq_len_q, d_model]
            attention_weights: [batch, num_heads, seq_len_q, seq_len_k]
        """
        batch_size = query.shape[0]

        # 1. 线性投影
        Q = self.W_q(query)  # [batch, seq_len_q, d_model]
        K = self.W_k(key)    # [batch, seq_len_k, d_model]
        V = self.W_v(value)  # [batch, seq_len_k, d_model]

        # 2. 分头
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 3. 缩放点积注意力
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        head_outputs = attention_weights @ V  # [batch, num_heads, seq_len_q, d_k]

        # 4. 拼接头
        concat = head_outputs.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        # 5. 输出投影
        output = self.W_o(concat)

        return output, attention_weights


def demo_multihead_implementation():
    print("=" * 60)
    print("3. 完整多头注意力实现测试")
    print("=" * 60)

    batch_size = 2
    seq_len = 6
    d_model = 64
    num_heads = 8

    mha = MultiHeadAttention(d_model, num_heads, dropout=0.0)

    x = torch.randn(batch_size, seq_len, d_model)

    # 自注意力: Q, K, V 都来自同一输入
    output, weights = mha(x, x, x)

    print(f"输入:    {list(x.shape)}")
    print(f"输出:    {list(output.shape)}")
    print(f"权重:    {list(weights.shape)}")
    print(f"权重和:  {weights[0, 0].sum(dim=-1).tolist()} (每行应为1.0)")

    # 交叉注意力: Q 来自一个序列, K/V 来自另一个
    x_encoder = torch.randn(batch_size, 10, d_model)
    x_decoder = torch.randn(batch_size, seq_len, d_model)

    output_cross, weights_cross = mha(x_decoder, x_encoder, x_encoder)
    print(f"\n交叉注意力:")
    print(f"  Q(解码器): {list(x_decoder.shape)}")
    print(f"  K,V(编码器): {list(x_encoder.shape)}")
    print(f"  输出: {list(output_cross.shape)}")
    print(f"  权重: {list(weights_cross.shape)}")
    print()


# ============================================================
# 4. 参数量计算
# ============================================================
def demo_parameter_count():
    """
    分析多头注意力的参数量
    """
    print("=" * 60)
    print("4. 参数量分析")
    print("=" * 60)

    configs = [
        (512, 8),    # 原始 Transformer base
        (768, 12),   # BERT-base
        (1024, 16),  # BERT-large
        (4096, 32),  # LLaMA-like
    ]

    print(f"{'d_model':>8} | {'num_heads':>10} | {'d_k':>4} | {'W_q':>8} | "
          f"{'W_k':>8} | {'W_v':>8} | {'W_o':>8} | {'总参数':>10}")
    print("-" * 85)

    for d_model, num_heads in configs:
        d_k = d_model // num_heads

        # 每个投影矩阵: d_model * d_model + d_model (bias)
        w_params = d_model * d_model + d_model
        total = 4 * w_params  # W_q, W_k, W_v, W_o

        print(f"{d_model:>8} | {num_heads:>10} | {d_k:>4} | "
              f"{w_params:>8,} | {w_params:>8,} | {w_params:>8,} | "
              f"{w_params:>8,} | {total:>10,}")

    print()
    print("注意:")
    print("  - 多头注意力的参数量与头数无关!")
    print("  - 无论 1 个头还是 8 个头, 4 个投影矩阵的总参数量相同")
    print("  - 头数只影响每个头的维度 d_k = d_model / num_heads")
    print()

    # 验证
    for d_model, num_heads in configs:
        mha = MultiHeadAttention(d_model, num_heads)
        param_count = sum(p.numel() for p in mha.parameters())
        print(f"  d_model={d_model}, heads={num_heads}: "
              f"实际参数 = {param_count:,}")


# ============================================================
# 5. 不同头数的对比
# ============================================================
def demo_head_comparison():
    """
    比较不同头数的注意力行为
    """
    print("=" * 60)
    print("5. 不同头数对比")
    print("=" * 60)

    d_model = 64
    batch_size = 1
    seq_len = 5
    torch.manual_seed(42)

    x = torch.randn(batch_size, seq_len, d_model)

    for num_heads in [1, 2, 4, 8]:
        d_k = d_model // num_heads
        mha = MultiHeadAttention(d_model, num_heads, dropout=0.0)

        output, weights = mha(x, x, x)

        # 分析权重多样性
        # 看不同头是否关注不同的位置
        if num_heads > 1:
            # 每个头的 top-1 关注位置
            top_positions = weights[0].argmax(dim=-1)  # [num_heads, seq_len]
            diversity = len(torch.unique(top_positions)) / (num_heads * seq_len)
        else:
            diversity = 0.0

        print(f"heads={num_heads}, d_k={d_k:>2}: "
              f"output{list(output.shape)}, "
              f"weights{list(weights.shape)}, "
              f"多样性={diversity:.2%}")

    print()


# ============================================================
# 6. 与 PyTorch nn.MultiheadAttention 对比
# ============================================================
def demo_compare_with_pytorch():
    print("=" * 60)
    print("6. 与 nn.MultiheadAttention 对比")
    print("=" * 60)

    d_model = 64
    num_heads = 8
    batch_size = 2
    seq_len = 5

    # PyTorch 内置
    pytorch_mha = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

    x = torch.randn(batch_size, seq_len, d_model)
    pytorch_out, pytorch_weights = pytorch_mha(x, x, x)

    print(f"nn.MultiheadAttention:")
    print(f"  输入:  {list(x.shape)}")
    print(f"  输出:  {list(pytorch_out.shape)}")
    print(f"  权重:  {list(pytorch_weights.shape)}")
    print(f"  参数量: {sum(p.numel() for p in pytorch_mha.parameters()):,}")

    # 我们的实现
    custom_mha = MultiHeadAttention(d_model, num_heads)
    custom_out, custom_weights = custom_mha(x, x, x)

    print(f"\n自定义 MultiHeadAttention:")
    print(f"  输出:  {list(custom_out.shape)}")
    print(f"  权重:  {list(custom_weights.shape)}")
    print(f"  参数量: {sum(p.numel() for p in custom_mha.parameters()):,}")

    print(f"\n两者接口一致，输出形状相同 (数值不同因为随机初始化)")
    print()


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)

    demo_motivation()
    demo_split_heads()
    demo_multihead_implementation()
    demo_parameter_count()
    demo_head_comparison()
    demo_compare_with_pytorch()

    print("所有演示完成！")
