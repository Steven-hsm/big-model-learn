"""
W13-D6: Full Encoder-Decoder Architecture, Masks, Cross-Attention, Shape Tracking
==================================================================================
完整的 Transformer 编码器-解码器架构。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 组件定义
# ============================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        B = query.shape[0]
        Q = self.W_q(query).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        out = (weights @ V).transpose(1, 2).contiguous().view(B, -1, self.num_heads * self.d_k)
        return self.W_o(out), weights


class PositionWiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.shape[1], :])


# ============================================================
# 1. 掩码机制
# ============================================================
def demo_masks():
    print("=" * 60)
    print("1. 掩码机制 (Padding Mask + Causal Mask)")
    print("=" * 60)

    batch_size = 2
    seq_len = 6
    d_model = 8

    # --- Padding Mask ---
    # 假设序列长度不一，需要 pad 到统一长度
    # [1, 1, 1, 1, 0, 0] 表示前4个位置有效，后2个是padding
    # [1, 1, 1, 1, 1, 0] 表示前5个位置有效
    lengths = torch.tensor([4, 5])
    padding_mask = torch.arange(seq_len).unsqueeze(0) < lengths.unsqueeze(1)
    print(f"Padding Mask (1=有效, 0=padding):")
    for i, (l, mask_row) in enumerate(zip(lengths, padding_mask)):
        print(f"  样本 {i} (len={l}): {mask_row.int().tolist()}")

    # 扩展为注意力掩码: [batch, 1, 1, seq_len]
    #广播到 [batch, num_heads, seq_len_q, seq_len_k]
    attn_padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
    print(f"\n注意力 Padding Mask shape: {list(attn_padding_mask.shape)}")

    # --- Causal Mask ---
    # 下三角矩阵，确保位置 i 只能看到 <= i 的位置
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
    print(f"\nCausal Mask (1=可见, 0=屏蔽):")
    print("     " + " ".join(f"P{j}" for j in range(seq_len)))
    for i in range(seq_len):
        row = " ".join(" T" if causal_mask[i, j] else " F" for j in range(seq_len))
        print(f"  P{i}: {row}")

    # --- Combined Mask ---
    # 解码器的自注意力需要同时应用 causal mask 和 padding mask
    # combined = causal_mask AND padding_mask
    combined = causal_mask.unsqueeze(0) & padding_mask.unsqueeze(1).unsqueeze(2)
    print(f"\nCombined Mask (Causal + Padding) shape: {list(combined.shape)}")
    print(f"含义: 位置 i 只能看到位置 0..i 中有效的位置")
    print()


# ============================================================
# 2. Encoder
# ============================================================
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # Self-Attention + Residual + Norm
        attn_out, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), src_mask)
        x = x + self.dropout1(attn_out)

        # FFN + Residual + Norm
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout2(ffn_out)

        return x


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


# ============================================================
# 3. Decoder
# ============================================================
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # Masked Self-Attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        # Cross-Attention
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        # FFN
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)
        # Norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        # Dropouts
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        # 1. Masked Self-Attention (只看已生成的位置)
        attn_out, _ = self.self_attn(
            self.norm1(x), self.norm1(x), self.norm1(x), tgt_mask
        )
        x = x + self.dropout1(attn_out)

        # 2. Cross-Attention (关注编码器输出)
        cross_out, _ = self.cross_attn(
            self.norm2(x), enc_output, enc_output, src_mask
        )
        x = x + self.dropout2(cross_out)

        # 3. FFN
        ffn_out = self.ffn(self.norm3(x))
        x = x + self.dropout3(ffn_out)

        return x


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, src_mask)
        return self.norm(x)


# ============================================================
# 4. 完整 Transformer
# ============================================================
class Transformer(nn.Module):
    def __init__(
        self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
        num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, dropout=0.1
    ):
        super().__init__()
        self.d_model = d_model

        # Embeddings
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        # Encoder & Decoder
        self.encoder = Encoder(num_encoder_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(num_decoder_layers, d_model, num_heads, d_ff, dropout)

        # Output projection
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        参数:
            src: [batch, src_len] 源语言 token ids
            tgt: [batch, tgt_len] 目标语言 token ids
        返回:
            logits: [batch, tgt_len, tgt_vocab_size]
        """
        # Embedding + Positional Encoding
        src_emb = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))

        # Encode
        enc_output = self.encoder(src_emb, src_mask)

        # Decode
        dec_output = self.decoder(tgt_emb, enc_output, tgt_mask, src_mask)

        # Project
        logits = self.output_proj(dec_output)

        return logits


# ============================================================
# 5. 完整前向传播 Shape 追踪
# ============================================================
def demo_full_forward_pass():
    print("=" * 60)
    print("5. 完整前向传播 Shape 追踪")
    print("=" * 60)

    # 小模型配置
    src_vocab = 100
    tgt_vocab = 80
    d_model = 64
    num_heads = 4
    num_layers = 2
    d_ff = 256
    batch_size = 2
    src_len = 8
    tgt_len = 6

    model = Transformer(
        src_vocab, tgt_vocab, d_model, num_heads,
        num_layers, num_layers, d_ff, dropout=0.1
    )

    src = torch.randint(1, src_vocab, (batch_size, src_len))
    tgt = torch.randint(1, tgt_vocab, (batch_size, tgt_len))

    # 创建 causal mask
    tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len)).bool()
    tgt_mask = tgt_mask.unsqueeze(0).expand(batch_size, -1, -1)

    print(f"源输入:      {list(src.shape)} (token ids)")
    print(f"目标输入:    {list(tgt.shape)} (token ids)")
    print(f"Causal mask: {list(tgt_mask.shape)}")

    # Step by step
    src_emb = model.src_embed(src) * math.sqrt(d_model)
    print(f"\n源 Embedding: {list(src_emb.shape)}")
    src_emb = model.pos_enc(src_emb)
    print(f"源 + PosEnc:  {list(src_emb.shape)}")

    enc_out = model.encoder(src_emb)
    print(f"编码器输出:   {list(enc_out.shape)}")

    tgt_emb = model.tgt_embed(tgt) * math.sqrt(d_model)
    tgt_emb = model.pos_enc(tgt_emb)
    print(f"\n目标 Embedding + PosEnc: {list(tgt_emb.shape)}")

    dec_out = model.decoder(tgt_emb, enc_out, tgt_mask=tgt_mask)
    print(f"解码器输出:   {list(dec_out.shape)}")

    logits = model.output_proj(dec_out)
    print(f"Logits:       {list(logits.shape)}")

    print(f"\n最终: 输入 [{batch_size}, {src_len}] -> 输出 [{batch_size}, {tgt_len}, {tgt_vocab}]")
    print(f"总参数量: {sum(p.numel() for p in model.parameters()):,}")
    print()


# ============================================================
# 6. Cross-Attention 详解
# ============================================================
def demo_cross_attention():
    print("=" * 60)
    print("6. Cross-Attention 详解")
    print("=" * 60)

    d_model = 64
    num_heads = 4
    batch_size = 1

    # 编码器输出 (源语言 "I love you")
    src_len = 3
    enc_output = torch.randn(batch_size, src_len, d_model)

    # 解码器当前状态 (目标语言 "我 爱 ?")
    tgt_len = 2
    dec_state = torch.randn(batch_size, tgt_len, d_model)

    cross_attn = MultiHeadAttention(d_model, num_heads, dropout=0.0)

    # Cross-Attention: Q 来自解码器, K/V 来自编码器
    output, weights = cross_attn(dec_state, enc_output, enc_output)

    print(f"编码器输出 (K,V): {list(enc_output.shape)} — 源语言 {src_len} 个位置")
    print(f"解码器状态 (Q):   {list(dec_state.shape)} — 目标语言 {tgt_len} 个位置")
    print(f"Cross-Attention 输出: {list(output.shape)}")
    print(f"Attention 权重:       {list(weights.shape)}")

    print(f"\n注意力权重 (目标关注源):")
    print(f"  形状: [heads={num_heads}, tgt_len={tgt_len}, src_len={src_len}]")
    for h in range(num_heads):
        print(f"  头 {h}:")
        for t in range(tgt_len):
            row = " ".join(f"{weights[0, h, t, s]:.3f}" for s in range(src_len))
            print(f"    目标位置 {t} 关注源: [{row}]")

    print(f"\n含义:")
    print(f"  目标位置 0 ('我') 可能关注源位置 0 ('I')")
    print(f"  目标位置 1 ('爱') 可能关注源位置 1 ('love')")
    print(f"  这就是翻译对齐!")
    print()


# ============================================================
# 7. 推理过程 (自回归生成)
# ============================================================
def demo_autoregressive_generation():
    print("=" * 60)
    print("7. 自回归生成过程")
    print("=" * 60)

    src_vocab = 50
    tgt_vocab = 40
    d_model = 64
    num_heads = 4
    num_layers = 2
    d_ff = 128

    model = Transformer(
        src_vocab, tgt_vocab, d_model, num_heads,
        num_layers, num_layers, d_ff, dropout=0.0
    )
    model.eval()

    # 源序列
    src = torch.randint(1, src_vocab, (1, 5))  # "hello world !"
    print(f"源序列: {src.tolist()}")

    # 自回归生成
    # 开始 token = 1
    tgt = torch.tensor([[1]])  # [BOS]
    max_len = 10
    eos_token = 2

    print(f"\n自回归生成过程:")
    print(f"  Step 0: tgt = {tgt.tolist()}")

    generated = [1]
    with torch.no_grad():
        for step in range(max_len):
            tgt_mask = torch.tril(torch.ones(tgt.shape[1], tgt.shape[1])).bool()
            tgt_mask = tgt_mask.unsqueeze(0)

            logits = model(src, tgt, tgt_mask=tgt_mask)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(next_token.item())

            print(f"  Step {step + 1}: 预测 token {next_token.item()}, "
                  f"tgt = {generated}")

            tgt = torch.cat([tgt, next_token], dim=1)

            if next_token.item() == eos_token:
                print(f"  -> 遇到 EOS，生成结束")
                break

    print(f"\n最终生成序列: {generated}")
    print()


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)

    demo_masks()
    demo_full_forward_pass()
    demo_cross_attention()
    demo_autoregressive_generation()

    print("所有演示完成！")
