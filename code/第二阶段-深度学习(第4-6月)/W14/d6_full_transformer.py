"""
W14-D6: Complete Transformer (Encoder + Decoder + Projection)
==============================================================
从零实现的完整 Transformer，包括参数初始化和复制任务训练。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import math

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d1_embedding import TransformerEmbedding
from d2_attention_impl import MultiHeadAttention, generate_causal_mask
from d3_ffn_norm import PositionWiseFFN, LayerNorm
from d4_encoder import EncoderLayer, Encoder
from d5_decoder import DecoderLayer, Decoder


# ============================================================
# 1. 完整 Transformer 模型
# ============================================================
class Transformer(nn.Module):
    """
    完整的 Encoder-Decoder Transformer

    组件:
    - Source Embedding (token + position)
    - Target Embedding (token + position)
    - Encoder (N layers)
    - Decoder (N layers)
    - Output Projection (d_model -> tgt_vocab_size)
    """

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        dropout=0.1,
        max_len=5000,
    ):
        super().__init__()
        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

        # Embeddings
        self.src_embedding = TransformerEmbedding(
            src_vocab_size, d_model, max_len, dropout
        )
        self.tgt_embedding = TransformerEmbedding(
            tgt_vocab_size, d_model, max_len, dropout
        )

        # Encoder & Decoder
        self.encoder = Encoder(
            num_encoder_layers, d_model, num_heads, d_ff, dropout
        )
        self.decoder = Decoder(
            num_decoder_layers, d_model, num_heads, d_ff, dropout
        )

        # Output Projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        # 参数初始化
        self._init_parameters()

    def _init_parameters(self):
        """Xavier 均匀初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def generate_square_subsequent_mask(seq_len):
        """
        生成因果掩码 (下三角)

        返回: [1, 1, seq_len, seq_len]
        True = 可见, False = 屏蔽
        """
        return generate_causal_mask(seq_len)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        参数:
            src: [batch, src_len] 源语言 token ids
            tgt: [batch, tgt_len] 目标语言 token ids
            src_mask: 源 padding 掩码
            tgt_mask: 目标因果掩码
        返回:
            logits: [batch, tgt_len, tgt_vocab_size]
        """
        # Embedding + Positional Encoding
        src_emb = self.src_embedding(src)
        tgt_emb = self.tgt_embedding(tgt)

        # Encode
        enc_output = self.encoder(src_emb, src_mask)

        # Decode
        dec_output = self.decoder(tgt_emb, enc_output, tgt_mask, src_mask)

        # Project to vocabulary
        logits = self.output_projection(dec_output)

        return logits

    def encode(self, src, src_mask=None):
        """编码器前向传播"""
        src_emb = self.src_embedding(src)
        return self.encoder(src_emb, src_mask)

    def decode(self, tgt, enc_output, tgt_mask=None, src_mask=None):
        """解码器前向传播"""
        tgt_emb = self.tgt_embedding(tgt)
        dec_output = self.decoder(tgt_emb, enc_output, tgt_mask, src_mask)
        return self.output_projection(dec_output)


# ============================================================
# 2. 复制任务数据生成
# ============================================================
def generate_copy_task_data(batch_size, seq_len, vocab_size, pad_id=0):
    """
    生成复制任务数据: 输入序列 -> 输出相同序列

    返回: src [batch, seq_len], tgt [batch, seq_len+2] (含 BOS/EOS)
    """
    # 随机序列 (排除 pad_id=0)
    src = torch.randint(1, vocab_size, (batch_size, seq_len))

    # 目标: BOS + src + EOS
    bos = torch.full((batch_size, 1), 1)  # BOS token = 1
    eos = torch.full((batch_size, 1), 2)  # EOS token = 2
    tgt = torch.cat([bos, src, eos], dim=1)

    return src, tgt


# ============================================================
# 3. 训练循环
# ============================================================
def train_copy_task():
    """
    在复制任务上训练 Transformer
    """
    print("=" * 60)
    print("复制任务训练")
    print("=" * 60)

    # 配置 (小模型用于演示)
    config = {
        "src_vocab_size": 10,
        "tgt_vocab_size": 10,
        "d_model": 64,
        "num_heads": 4,
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "d_ff": 128,
        "dropout": 0.0,
        "max_len": 100,
        "seq_len": 6,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "num_epochs": 30,
    }

    # 创建模型
    model = Transformer(
        src_vocab_size=config["src_vocab_size"],
        tgt_vocab_size=config["tgt_vocab_size"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_encoder_layers=config["num_encoder_layers"],
        num_decoder_layers=config["num_decoder_layers"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        max_len=config["max_len"],
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    print(f"配置: d_model={config['d_model']}, heads={config['num_heads']}, "
          f"layers={config['num_encoder_layers']}+{config['num_decoder_layers']}")

    # 优化器和损失
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略 padding
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # 训练
    print(f"\n开始训练 ({config['num_epochs']} epochs)...")
    print(f"{'Epoch':>5} | {'Loss':>8} | {'Accuracy':>8}")
    print("-" * 30)

    for epoch in range(config["num_epochs"]):
        model.train()

        # 生成数据
        src, tgt = generate_copy_task_data(
            config["batch_size"],
            config["seq_len"],
            config["src_vocab_size"],
        )

        # Decoder 输入: tgt[:, :-1] (不含最后的 EOS)
        # 目标: tgt[:, 1:] (不含开头的 BOS)
        tgt_input = tgt[:, :-1]
        tgt_target = tgt[:, 1:]

        # 生成 causal mask
        tgt_mask = model.generate_square_subsequent_mask(tgt_input.shape[1])

        # 前向传播
        logits = model(src, tgt_input, tgt_mask=tgt_mask)

        # 计算损失
        loss = criterion(
            logits.reshape(-1, config["tgt_vocab_size"]),
            tgt_target.reshape(-1),
        )

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        # 计算准确率
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            correct = (preds == tgt_target).float()
            # 排除 padding 位置
            mask = tgt_target != 0
            accuracy = correct[mask].mean().item()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"{epoch + 1:>5} | {loss.item():>8.4f} | {accuracy:>8.2%}")

    # 最终测试
    print(f"\n最终测试:")
    model.eval()
    with torch.no_grad():
        test_src, test_tgt = generate_copy_task_data(5, config["seq_len"], config["src_vocab_size"])
        tgt_input = test_tgt[:, :-1]
        tgt_target = test_tgt[:, 1:]
        tgt_mask = model.generate_square_subsequent_mask(tgt_input.shape[1])

        logits = model(test_src, tgt_input, tgt_mask=tgt_mask)
        preds = logits.argmax(dim=-1)

        for i in range(min(3, test_src.shape[0])):
            print(f"  输入:   {test_src[i].tolist()}")
            print(f"  目标:   {tgt_target[i].tolist()}")
            print(f"  预测:   {preds[i].tolist()}")
            match = (preds[i] == tgt_target[i]).all().item()
            print(f"  匹配:   {'Yes' if match else 'No'}")
            print()

    return model


# ============================================================
# 4. 参数初始化验证
# ============================================================
def test_parameter_init():
    print("=" * 60)
    print("参数初始化验证")
    print("=" * 60)

    model = Transformer(
        src_vocab_size=100,
        tgt_vocab_size=100,
        d_model=64,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=128,
    )

    # 检查 Xavier 初始化效果
    print("各层权重统计:")
    for name, param in model.named_parameters():
        if param.dim() > 1 and "embedding" not in name and "norm" not in name:
            fan_in = param.shape[0]
            fan_out = param.shape[1] if param.dim() == 2 else param.shape[0]
            xavier_std = math.sqrt(2.0 / (fan_in + fan_out))
            actual_std = param.std().item()
            ratio = actual_std / xavier_std
            if abs(ratio - 1.0) < 0.5:
                status = "OK"
            else:
                status = "CHECK"
            # 只打印部分
            if len(name) < 50:
                print(f"  {name}: std={actual_std:.4f}, "
                      f"xavier_std={xavier_std:.4f}, ratio={ratio:.2f} [{status}]")

    print("[PASS] 参数初始化验证完成")
    print()


# ============================================================
# 5. 模型功能测试
# ============================================================
def test_transformer():
    print("=" * 60)
    print("Transformer 完整功能测试")
    print("=" * 60)

    src_vocab = 50
    tgt_vocab = 40
    d_model = 64
    batch_size = 2
    src_len = 8
    tgt_len = 6

    model = Transformer(
        src_vocab, tgt_vocab, d_model=d_model,
        num_heads=4, num_encoder_layers=2, num_decoder_layers=2,
        d_ff=128, dropout=0.0,
    )

    src = torch.randint(1, src_vocab, (batch_size, src_len))
    tgt = torch.randint(1, tgt_vocab, (batch_size, tgt_len))
    tgt_mask = model.generate_square_subsequent_mask(tgt_len)

    # 前向传播
    logits = model(src, tgt, tgt_mask=tgt_mask)

    print(f"源输入: {list(src.shape)}")
    print(f"目标输入: {list(tgt.shape)}")
    print(f"Logits: {list(logits.shape)}")
    assert logits.shape == (batch_size, tgt_len, tgt_vocab)
    print(f"[PASS] Shape 正确")

    # encode + decode 分步执行
    enc_out = model.encode(src)
    print(f"\n编码器输出: {list(enc_out.shape)}")
    assert enc_out.shape == (batch_size, src_len, d_model)

    dec_logits = model.decode(tgt, enc_out, tgt_mask=tgt_mask)
    print(f"解码器 Logits: {list(dec_logits.shape)}")
    assert dec_logits.shape == (batch_size, tgt_len, tgt_vocab)

    # 分步和一步结果一致
    diff = (logits - dec_logits).abs().max().item()
    print(f"一步 vs 分步差异: {diff:.8f}")
    assert diff < 1e-5, f"一步和分步结果不一致: {diff}"
    print(f"[PASS] 分步执行结果一致")

    # 总参数量
    print(f"\n总参数量: {sum(p.numel() for p in model.parameters()):,}")
    print()


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)

    test_transformer()
    test_parameter_init()
    model = train_copy_task()

    print("所有测试完成！")
