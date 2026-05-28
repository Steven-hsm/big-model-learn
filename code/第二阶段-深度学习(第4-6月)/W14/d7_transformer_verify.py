"""
W14-D7: Train on Copy Task, Compare with nn.Transformer, Plot Training Loss
=============================================================================
验证从零实现的 Transformer 能否正确学习复制任务。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import math

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d6_full_transformer import Transformer


# ============================================================
# 1. 复制任务数据生成
# ============================================================
def generate_copy_data(batch_size, seq_len, vocab_size):
    """
    生成复制任务数据
    输入: 随机序列 [batch, seq_len]
    输出: 相同序列 (带 BOS/EOS)
    """
    src = torch.randint(3, vocab_size, (batch_size, seq_len))  # 0=pad, 1=BOS, 2=EOS
    bos = torch.full((batch_size, 1), 1)
    eos = torch.full((batch_size, 1), 2)
    tgt = torch.cat([bos, src, eos], dim=1)
    return src, tgt


# ============================================================
# 2. 训练自定义 Transformer
# ============================================================
def train_custom_transformer(config, num_epochs=50, log_interval=10):
    """训练自定义 Transformer 并返回损失历史"""

    model = Transformer(
        src_vocab_size=config["vocab_size"],
        tgt_vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_encoder_layers=config["num_layers"],
        num_decoder_layers=config["num_layers"],
        d_ff=config["d_ff"],
        dropout=0.0,
        max_len=100,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    losses = []
    accuracies = []

    for epoch in range(num_epochs):
        model.train()

        src, tgt = generate_copy_data(
            config["batch_size"], config["seq_len"], config["vocab_size"]
        )

        tgt_input = tgt[:, :-1]
        tgt_target = tgt[:, 1:]
        tgt_mask = model.generate_square_subsequent_mask(tgt_input.shape[1])

        logits = model(src, tgt_input, tgt_mask=tgt_mask)
        loss = criterion(
            logits.reshape(-1, config["vocab_size"]),
            tgt_target.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # 计算准确率
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            mask = tgt_target != 0
            acc = (preds[mask] == tgt_target[mask]).float().mean().item()

        losses.append(loss.item())
        accuracies.append(acc)

        if (epoch + 1) % log_interval == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:>3}: loss={loss.item():.4f}, acc={acc:.2%}")

    return model, losses, accuracies


# ============================================================
# 3. 训练 PyTorch nn.Transformer (对比)
# ============================================================
class PyTorchTransformer(nn.Module):
    """使用 PyTorch 内置 nn.Transformer 的模型"""

    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=d_ff,
            dropout=0.0,
            batch_first=True,
        )
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Xavier 初始化
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, tgt_mask=None):
        src_emb = self.embedding(src) * math.sqrt(self.d_model) + self.pos_encoding[:, :src.shape[1]]
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model) + self.pos_encoding[:, :tgt.shape[1]]

        out = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        return self.output_proj(out)


def train_pytorch_transformer(config, num_epochs=50, log_interval=10):
    """训练 PyTorch 内置 Transformer"""

    model = PyTorchTransformer(
        config["vocab_size"], config["d_model"],
        config["num_heads"], config["num_layers"], config["d_ff"],
    )

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    losses = []
    accuracies = []

    for epoch in range(num_epochs):
        model.train()

        src, tgt = generate_copy_data(
            config["batch_size"], config["seq_len"], config["vocab_size"]
        )

        tgt_input = tgt[:, :-1]
        tgt_target = tgt[:, 1:]

        # PyTorch nn.Transformer 使用 float mask (负无穷)
        tgt_len = tgt_input.shape[1]
        tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len) * float("-inf"), diagonal=1)

        logits = model(src, tgt_input, tgt_mask=tgt_mask)
        loss = criterion(
            logits.reshape(-1, config["vocab_size"]),
            tgt_target.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            mask = tgt_target != 0
            acc = (preds[mask] == tgt_target[mask]).float().mean().item()

        losses.append(loss.item())
        accuracies.append(acc)

        if (epoch + 1) % log_interval == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:>3}: loss={loss.item():.4f}, acc={acc:.2%}")

    return model, losses, accuracies


# ============================================================
# 4. 可视化训练损失
# ============================================================
def plot_training_comparison(custom_losses, pytorch_losses, custom_accs, pytorch_accs):
    """用字符画展示训练曲线"""
    print("\n" + "=" * 60)
    print("训练曲线对比")
    print("=" * 60)

    # 损失曲线
    print("\nLoss 曲线 (自定义 vs PyTorch nn.Transformer):")
    print(f"  {'Epoch':>5} | {'Custom Loss':>11} | {'PyTorch Loss':>12} | 曲线")
    print("  " + "-" * 55)

    n = len(custom_losses)
    step = max(1, n // 20)
    max_loss = max(max(custom_losses), max(pytorch_losses))

    for i in range(0, n, step):
        cl = custom_losses[i]
        pl = pytorch_losses[i]
        # 用 # 表示自定义, 用 * 表示 PyTorch
        bar_len = 40
        custom_bar = int(cl / max_loss * bar_len)
        pt_bar = int(pl / max_loss * bar_len)

        bar = "#" * custom_bar + " " * max(0, pt_bar - custom_bar) + "|" + "*" * max(0, pt_bar - custom_bar)
        # 简化: 只显示数值
        print(f"  {i + 1:>5} | {cl:>11.4f} | {pl:>12.4f}")

    # 准确率曲线
    print(f"\nAccuracy 曲线:")
    print(f"  {'Epoch':>5} | {'Custom Acc':>10} | {'PyTorch Acc':>11}")
    print("  " + "-" * 45)

    for i in range(0, n, step):
        ca = custom_accs[i]
        pa = pytorch_accs[i]
        bar_c = int(ca * 30)
        bar_p = int(pa * 30)
        print(f"  {i + 1:>5} | {ca:>10.2%} | {pa:>11.2%} | "
              f"{'#' * bar_c}({'*' * bar_p})")


# ============================================================
# 5. 最终验证
# ============================================================
def final_verification(model, config):
    """验证模型是否正确学会了复制任务"""
    print("\n" + "=" * 60)
    print("最终验证: 模型是否正确学会复制?")
    print("=" * 60)

    model.eval()

    # 测试不同长度的序列
    test_lengths = [3, 5, 8, 10]

    all_correct = True
    for seq_len in test_lengths:
        src, tgt = generate_copy_data(1, seq_len, config["vocab_size"])
        tgt_input = tgt[:, :-1]
        tgt_target = tgt[:, 1:]
        tgt_mask = model.generate_square_subsequent_mask(tgt_input.shape[1])

        with torch.no_grad():
            logits = model(src, tgt_input, tgt_mask=tgt_mask)
            preds = logits.argmax(dim=-1)

        correct = (preds == tgt_target)
        # 排除 EOS 位置 (模型可能不一定生成 EOS)
        src_part = correct[:, 1:seq_len + 1]  # 只看复制部分
        is_correct = src_part.all().item()
        all_correct = all_correct and is_correct

        print(f"\n  序列长度 {seq_len}:")
        print(f"    输入:   {src[0].tolist()}")
        print(f"    目标:   {tgt_target[0].tolist()}")
        print(f"    预测:   {preds[0].tolist()}")
        print(f"    复制正确: {'YES' if is_correct else 'NO'}")

    print(f"\n  所有长度复制正确: {'YES' if all_correct else 'NO'}")
    return all_correct


# ============================================================
# 6. 模型大小对比
# ============================================================
def compare_model_sizes(config):
    print("\n" + "=" * 60)
    print("模型大小对比")
    print("=" * 60)

    # 自定义
    custom_model = Transformer(
        config["vocab_size"], config["vocab_size"],
        d_model=config["d_model"], num_heads=config["num_heads"],
        num_encoder_layers=config["num_layers"],
        num_decoder_layers=config["num_layers"],
        d_ff=config["d_ff"], dropout=0.0,
    )
    custom_params = sum(p.numel() for p in custom_model.parameters())

    # PyTorch
    pt_model = PyTorchTransformer(
        config["vocab_size"], config["d_model"],
        config["num_heads"], config["num_layers"], config["d_ff"],
    )
    pt_params = sum(p.numel() for p in pt_model.parameters())

    print(f"  {'模型':>20} | {'参数量':>12}")
    print(f"  {'-'*20}-+-{'-'*12}")
    print(f"  {'自定义 Transformer':>20} | {custom_params:>12,}")
    print(f"  {'PyTorch nn.Transformer':>20} | {pt_params:>12,}")
    print(f"\n  参数量接近: {abs(custom_params - pt_params) / max(custom_params, pt_params) < 0.1}")


# ============================================================
# 主函数
# ============================================================
def main():
    torch.manual_seed(42)

    config = {
        "vocab_size": 10,
        "d_model": 64,
        "num_heads": 4,
        "num_layers": 2,
        "d_ff": 128,
        "seq_len": 6,
        "batch_size": 32,
        "lr": 5e-4,
    }

    num_epochs = 60
    log_interval = 10

    print("=" * 60)
    print("Transformer 从零实现 - 复制任务验证")
    print("=" * 60)
    print(f"配置: {config}")
    print(f"训练: {num_epochs} epochs")
    print()

    # 训练自定义模型
    print("--- 训练自定义 Transformer ---")
    custom_model, custom_losses, custom_accs = train_custom_transformer(
        config, num_epochs, log_interval
    )

    # 训练 PyTorch 模型
    print("\n--- 训练 PyTorch nn.Transformer ---")
    pt_model, pt_losses, pt_accs = train_pytorch_transformer(
        config, num_epochs, log_interval
    )

    # 对比曲线
    plot_training_comparison(custom_losses, pt_losses, custom_accs, pt_accs)

    # 最终验证
    print("\n--- 自定义模型最终验证 ---")
    final_verification(custom_model, config)

    print("\n--- PyTorch 模型最终验证 ---")
    final_verification(pt_model, config)

    # 模型大小对比
    compare_model_sizes(config)

    print("\n" + "=" * 60)
    print("验证完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
