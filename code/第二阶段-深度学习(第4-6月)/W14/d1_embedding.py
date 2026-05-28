"""
W14-D1: TokenEmbedding + PositionalEncoding + Combined Embedding + Shape Verification
======================================================================================
Transformer 从零实现 - 第1步: 嵌入层。
"""

import torch
import torch.nn as nn
import math


# ============================================================
# 1. Token Embedding
# ============================================================
class TokenEmbedding(nn.Module):
    """
    Token 嵌入层: 将 token id 映射为向量

    nn.Embedding 内部维护一个 [vocab_size, d_model] 的查找表
    输入 token id -> 输出对应的嵌入向量
    """

    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        """
        参数: x [batch, seq_len] — token id 序列
        返回: [batch, seq_len, d_model] — 嵌入向量
        """
        # 乘以 sqrt(d_model) 是原始论文的做法
        # 这样 positional encoding 的尺度不会淹没 embedding
        return self.embedding(x) * math.sqrt(self.d_model)


def test_token_embedding():
    print("=" * 60)
    print("1. TokenEmbedding 测试")
    print("=" * 60)

    vocab_size = 1000
    d_model = 512
    batch_size = 2
    seq_len = 10

    token_emb = TokenEmbedding(vocab_size, d_model)

    # 输入: token id 序列
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    output = token_emb(input_ids)

    print(f"输入 (token ids): {list(input_ids.shape)}")
    print(f"  示例: {input_ids[0, :5].tolist()}")
    print(f"输出 (embeddings): {list(output.shape)}")
    print(f"  嵌入向量范数: {output[0, 0].norm():.4f}")
    print(f"  sqrt(d_model) = {math.sqrt(d_model):.4f}")

    # 参数
    print(f"\n参数量: {sum(p.numel() for p in token_emb.parameters()):,}")
    print(f"  Embedding 矩阵: [{vocab_size}, {d_model}]")

    # 验证相同 token 得到相同嵌入
    same_ids = torch.tensor([[42, 42, 42]])
    same_out = token_emb(same_ids)
    diff = (same_out[0, 0] - same_out[0, 1]).abs().max().item()
    print(f"\n相同 token 嵌入一致: {diff == 0.0}")
    print()


# ============================================================
# 2. Positional Encoding (Sinusoidal)
# ============================================================
class PositionalEncoding(nn.Module):
    """
    正弦位置编码

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 预计算位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度用 sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度用 cos

        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        """
        参数: x [batch, seq_len, d_model]
        返回: [batch, seq_len, d_model]
        """
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)


def test_positional_encoding():
    print("=" * 60)
    print("2. PositionalEncoding 测试")
    print("=" * 60)

    d_model = 512
    max_len = 100
    batch_size = 2
    seq_len = 10

    pos_enc = PositionalEncoding(d_model, max_len, dropout=0.0)

    x = torch.randn(batch_size, seq_len, d_model)
    output = pos_enc(x)

    print(f"输入: {list(x.shape)}")
    print(f"输出: {list(output.shape)}")

    # 验证不同位置有不同的编码
    pe = pos_enc.pe[0]  # [max_len, d_model]
    for pos in [0, 1, 50]:
        norm = pe[pos].norm().item()
        print(f"  位置 {pos:>2} 编码范数: {norm:.4f}")

    # 验证相邻位置的差异
    diff_adjacent = (pe[1] - pe[0]).norm().item()
    diff_far = (pe[50] - pe[0]).norm().item()
    print(f"\n  位置 0 vs 1 差异: {diff_adjacent:.4f}")
    print(f"  位置 0 vs 50 差异: {diff_far:.4f}")
    print(f"  远距离差异更大: {diff_far > diff_adjacent}")
    print()


# ============================================================
# 3. 组合嵌入层
# ============================================================
class TransformerEmbedding(nn.Module):
    """
    组合嵌入: Token Embedding + Positional Encoding

    这是 Transformer 的输入层
    """

    def __init__(self, vocab_size, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)

    def forward(self, x):
        """
        参数: x [batch, seq_len] — token id 序列
        返回: [batch, seq_len, d_model]
        """
        token_emb = self.token_embedding(x)
        return self.positional_encoding(token_emb)


def test_combined_embedding():
    print("=" * 60)
    print("3. TransformerEmbedding 组合测试")
    print("=" * 60)

    vocab_size = 1000
    d_model = 512
    batch_size = 4
    seq_len = 20

    embedding = TransformerEmbedding(vocab_size, d_model, dropout=0.0)

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    output = embedding(input_ids)

    print(f"输入 token ids: {list(input_ids.shape)}")
    print(f"  示例: {input_ids[0, :8].tolist()}")
    print(f"输出 embeddings: {list(output.shape)}")
    print()

    # 验证不同位置的相同 token 有不同输出
    # 因为位置编码不同
    same_token_ids = torch.tensor([[42, 42, 42, 42, 42]])
    same_out = embedding(same_token_ids)
    diff_pos_0_1 = (same_out[0, 0] - same_out[0, 1]).norm().item()
    diff_pos_0_4 = (same_out[0, 0] - same_out[0, 4]).norm().item()
    print(f"相同 token 不同位置:")
    print(f"  位置 0 vs 1 差异: {diff_pos_0_1:.4f}")
    print(f"  位置 0 vs 4 差异: {diff_pos_0_4:.4f}")
    print(f"  差异来自位置编码 (embedding 本身相同)")
    print()

    # 参数量
    total_params = sum(p.numel() for p in embedding.parameters())
    print(f"总参数量: {total_params:,}")
    print(f"  Token Embedding: {vocab_size * d_model:,}")
    print(f"  Positional Encoding: 0 (固定计算)")
    print()


# ============================================================
# 4. Shape 验证汇总
# ============================================================
def test_all_shapes():
    print("=" * 60)
    print("4. Shape 验证汇总")
    print("=" * 60)

    vocab_size = 100
    d_model = 64
    batch_size = 3
    seq_len = 8

    token_emb = TokenEmbedding(vocab_size, d_model)
    pos_enc = PositionalEncoding(d_model, max_len=50, dropout=0.0)
    combined = TransformerEmbedding(vocab_size, d_model, max_len=50, dropout=0.0)

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    print(f"输入: {list(input_ids.shape)} [batch={batch_size}, seq_len={seq_len}]")
    print(f"d_model: {d_model}")
    print()

    # Token Embedding
    tok_out = token_emb(input_ids)
    assert tok_out.shape == (batch_size, seq_len, d_model), \
        f"Token Embedding shape 错误: {tok_out.shape}"
    print(f"[PASS] TokenEmbedding: {list(input_ids.shape)} -> {list(tok_out.shape)}")

    # Positional Encoding
    x = torch.randn(batch_size, seq_len, d_model)
    pos_out = pos_enc(x)
    assert pos_out.shape == (batch_size, seq_len, d_model), \
        f"Positional Encoding shape 错误: {pos_out.shape}"
    print(f"[PASS] PositionalEncoding: {list(x.shape)} -> {list(pos_out.shape)}")

    # Combined
    comb_out = combined(input_ids)
    assert comb_out.shape == (batch_size, seq_len, d_model), \
        f"Combined Embedding shape 错误: {comb_out.shape}"
    print(f"[PASS] TransformerEmbedding: {list(input_ids.shape)} -> {list(comb_out.shape)}")

    # 边界情况: seq_len = 1
    single_input = torch.randint(0, vocab_size, (batch_size, 1))
    single_out = combined(single_input)
    assert single_out.shape == (batch_size, 1, d_model)
    print(f"[PASS] 单 token 输入: {list(single_input.shape)} -> {list(single_out.shape)}")

    # 边界情况: batch_size = 1
    single_batch = torch.randint(0, vocab_size, (1, seq_len))
    single_batch_out = combined(single_batch)
    assert single_batch_out.shape == (1, seq_len, d_model)
    print(f"[PASS] 单 batch 输入: {list(single_batch.shape)} -> {list(single_batch_out.shape)}")

    print(f"\n所有 Shape 验证通过!")
    print()


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)

    test_token_embedding()
    test_positional_encoding()
    test_combined_embedding()
    test_all_shapes()

    print("所有测试完成！")
