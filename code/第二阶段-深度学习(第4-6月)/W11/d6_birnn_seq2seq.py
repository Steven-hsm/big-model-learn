"""
W11-d6: 双向RNN与Seq2Seq
- PyTorch双向LSTM（输出形状=hidden*2）
- 提取最终隐藏状态用于分类
- 实现简单Encoder-Decoder（LSTM）
- 展示encoder产生context vector, decoder逐步生成
- 打印各阶段形状
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 检查PyTorch是否可用
try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch未安装，部分功能将使用NumPy模拟")
    print("安装命令: pip install torch")


# ============================================================
# 1. 双向LSTM
# ============================================================
print("=" * 60)
print("1. 双向LSTM (Bidirectional LSTM)")
print("=" * 60)

if PYTORCH_AVAILABLE:
    # 超参数
    batch_size = 2
    seq_len = 6
    input_size = 4
    hidden_size = 8
    num_layers = 2

    # 创建双向LSTM
    bilstm = nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=True,
        bidirectional=True
    )

    # 输入
    x = torch.randn(batch_size, seq_len, input_size)
    h0 = torch.zeros(num_layers * 2, batch_size, hidden_size)  # *2 因为双向
    c0 = torch.zeros(num_layers * 2, batch_size, hidden_size)

    output, (hn, cn) = bilstm(x, (h0, c0))

    print(f"\n输入形状: {x.shape}")
    print(f"  (batch_size={batch_size}, seq_len={seq_len}, input_size={input_size})")
    print(f"\n输出形状: {output.shape}")
    print(f"  (batch_size={batch_size}, seq_len={seq_len}, hidden_size*2={hidden_size * 2})")
    print(f"  => 双向输出维度 = hidden_size * 2 = {hidden_size * 2}")
    print(f"\n最终隐藏状态形状: {hn.shape}")
    print(f"  (num_layers*2={num_layers * 2}, batch_size={batch_size}, hidden_size={hidden_size})")
    print(f"\n最终细胞状态形状: {cn.shape}")
    print(f"  (num_layers*2={num_layers * 2}, batch_size={batch_size}, hidden_size={hidden_size})")

    # 分离前向和后向输出
    output_forward = output[:, :, :hidden_size]    # 前向
    output_backward = output[:, :, hidden_size:]   # 后向
    print(f"\n前向输出形状: {output_forward.shape}")
    print(f"后向输出形状: {output_backward.shape}")

    # 分离前向和后向隐藏状态
    hn_forward = hn[:num_layers]       # 前向各层
    hn_backward = hn[num_layers:]      # 后向各层
    print(f"\n前向最终隐藏形状: {hn_forward.shape}")
    print(f"后向最终隐藏形状: {hn_backward.shape}")

    # 参数统计
    total_params = sum(p.numel() for p in bilstm.parameters())
    print(f"\n双向LSTM参数量: {total_params}")

    # 与单向LSTM对比
    uni_lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
    uni_params = sum(p.numel() for p in uni_lstm.parameters())
    print(f"单向LSTM参数量: {uni_params}")
    print(f"双向/单向参数比: {total_params / uni_params:.2f}x (约2倍)")

else:
    print("\nPyTorch不可用，展示双向LSTM概念:")
    print("  双向LSTM = 前向LSTM + 后向LSTM")
    print("  输出维度 = hidden_size * 2")
    print("  参数量约为单向的2倍")


# ============================================================
# 2. 提取最终隐藏状态用于分类
# ============================================================
print("\n" + "=" * 60)
print("2. 从双向LSTM提取特征用于分类")
print("=" * 60)

if PYTORCH_AVAILABLE:
    class BiLSTMClassifier(nn.Module):
        """双向LSTM文本分类器"""

        def __init__(self, vocab_size, embed_dim, hidden_size,
                     num_classes, num_layers=1, dropout=0.3):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers

            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.lstm = nn.LSTM(
                embed_dim, hidden_size, num_layers,
                batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0
            )
            self.dropout = nn.Dropout(dropout)
            # 分类头: 输入维度为 hidden_size * 2 (双向拼接)
            self.fc = nn.Linear(hidden_size * 2, num_classes)

        def forward(self, x):
            # x: (batch, seq_len)
            embedded = self.embedding(x)  # (batch, seq_len, embed_dim)

            output, (hn, cn) = self.lstm(embedded)

            # 方法1: 拼接最后一层的前向和后向最终隐藏状态
            hn_forward = hn[-2]    # 最后一层前向: (batch, hidden)
            hn_backward = hn[-1]   # 最后一层后向: (batch, hidden)
            hidden = torch.cat([hn_forward, hn_backward], dim=1)  # (batch, hidden*2)

            # 方法2 (替代): 使用所有时间步的输出取平均/最大池化
            # hidden = output.mean(dim=1)  # (batch, hidden*2)

            hidden = self.dropout(hidden)
            logits = self.fc(hidden)  # (batch, num_classes)
            return logits, output, hidden

    # 创建分类器
    classifier = BiLSTMClassifier(
        vocab_size=1000, embed_dim=32, hidden_size=64,
        num_classes=2, num_layers=2
    )

    # 模拟输入
    x_input = torch.randint(0, 1000, (4, 20))  # batch=4, seq_len=20

    logits, lstm_output, final_hidden = classifier(x_input)

    print(f"\n分类器结构:")
    print(f"  Embedding: vocab_size=1000, embed_dim=32")
    print(f"  BiLSTM: hidden=64, layers=2, bidirectional=True")
    print(f"  FC: input={64 * 2}, output=2")
    print(f"\n各阶段形状:")
    print(f"  输入:       {x_input.shape}")
    print(f"  LSTM输出:   {lstm_output.shape}  (batch, seq_len, hidden*2)")
    print(f"  拼接隐藏:   {final_hidden.shape}  (batch, hidden*2)")
    print(f"  分类logits: {logits.shape}  (batch, num_classes)")

    total_cls_params = sum(p.numel() for p in classifier.parameters())
    print(f"\n分类器总参数量: {total_cls_params:,}")

else:
    print("\nPyTorch不可用，展示分类器概念:")
    print("  1. Embedding: (batch, seq_len) -> (batch, seq_len, embed_dim)")
    print("  2. BiLSTM: (batch, seq_len, embed_dim) -> (batch, seq_len, hidden*2)")
    print("  3. 取最终隐藏: 拼接前向+后向 -> (batch, hidden*2)")
    print("  4. FC: (batch, hidden*2) -> (batch, num_classes)")


# ============================================================
# 3. Encoder-Decoder (Seq2Seq)
# ============================================================
print("\n" + "=" * 60)
print("3. Encoder-Decoder (Seq2Seq) 实现")
print("=" * 60)

if PYTORCH_AVAILABLE:
    class Encoder(nn.Module):
        """LSTM编码器"""

        def __init__(self, vocab_size, embed_dim, hidden_size, num_layers=1):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers

            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True)

        def forward(self, x):
            # x: (batch, src_len)
            embedded = self.embedding(x)  # (batch, src_len, embed_dim)
            outputs, (hidden, cell) = self.lstm(embedded)
            # outputs: (batch, src_len, hidden)
            # hidden, cell: (num_layers, batch, hidden)
            return outputs, hidden, cell

    class Decoder(nn.Module):
        """LSTM解码器"""

        def __init__(self, vocab_size, embed_dim, hidden_size, num_layers=1):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.vocab_size = vocab_size

            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True)
            self.fc_out = nn.Linear(hidden_size, vocab_size)

        def forward(self, x, hidden, cell):
            # x: (batch, 1) - 一次输入一个token
            embedded = self.embedding(x)  # (batch, 1, embed_dim)
            output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
            # output: (batch, 1, hidden)
            prediction = self.fc_out(output.squeeze(1))  # (batch, vocab_size)
            return prediction, hidden, cell

    class Seq2Seq(nn.Module):
        """完整的Seq2Seq模型"""

        def __init__(self, encoder, decoder, sos_token_id):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
            self.sos_token_id = sos_token_id

        def forward(self, src, trg, teacher_forcing_ratio=0.5):
            batch_size = src.shape[0]
            trg_len = trg.shape[1]
            trg_vocab_size = self.decoder.vocab_size

            # 存储解码器输出
            outputs = torch.zeros(batch_size, trg_len, trg_vocab_size)

            # 编码
            enc_outputs, hidden, cell = self.encoder(src)

            # 解码器第一个输入是<SOS>
            decoder_input = torch.full((batch_size, 1), self.sos_token_id, dtype=torch.long)

            for t in range(trg_len):
                # 解码一步
                prediction, hidden, cell = self.decoder(decoder_input, hidden, cell)
                outputs[:, t, :] = prediction

                # Teacher forcing: 用真实目标还是预测值作为下一步输入
                if np.random.random() < teacher_forcing_ratio:
                    decoder_input = trg[:, t:t + 1]  # 真实值
                else:
                    decoder_input = prediction.argmax(dim=1, keepdim=True)  # 预测值

            return outputs, enc_outputs

    # 参数
    SRC_VOCAB = 100
    TRG_VOCAB = 80
    EMBED_DIM = 16
    HIDDEN_SIZE = 32
    NUM_LAYERS = 2
    SOS_TOKEN = 1

    encoder = Encoder(SRC_VOCAB, EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS)
    decoder = Decoder(TRG_VOCAB, EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS)
    seq2seq = Seq2Seq(encoder, decoder, SOS_TOKEN)

    # 模拟输入
    batch_size = 3
    src_len = 8
    trg_len = 6
    src = torch.randint(2, SRC_VOCAB, (batch_size, src_len))
    trg = torch.randint(2, TRG_VOCAB, (batch_size, trg_len))

    print(f"\nEncoder-Decoder 结构:")
    print(f"  Encoder: vocab={SRC_VOCAB}, embed={EMBED_DIM}, hidden={HIDDEN_SIZE}, layers={NUM_LAYERS}")
    print(f"  Decoder: vocab={TRG_VOCAB}, embed={EMBED_DIM}, hidden={HIDDEN_SIZE}, layers={NUM_LAYERS}")

    # 编码阶段
    enc_outputs, enc_hidden, enc_cell = encoder(src)
    print(f"\n--- 编码阶段 ---")
    print(f"  源输入:          {src.shape}  (batch, src_len)")
    print(f"  编码器输出:      {enc_outputs.shape}  (batch, src_len, hidden)")
    print(f"  编码器隐藏状态:  {enc_hidden.shape}  (num_layers, batch, hidden)")
    print(f"  编码器细胞状态:  {enc_cell.shape}  (num_layers, batch, hidden)")
    print(f"  => Context Vector = (hidden, cell) 传递给解码器")

    # 完整Seq2Seq
    outputs, _ = seq2seq(src, trg, teacher_forcing_ratio=0.5)
    print(f"\n--- Seq2Seq 完整前向传播 ---")
    print(f"  源输入:  {src.shape}")
    print(f"  目标:    {trg.shape}")
    print(f"  输出:    {outputs.shape}  (batch, trg_len, trg_vocab_size)")
    print(f"  => 每个位置输出 {TRG_VOCAB} 个类别的概率分布")

    # 解码器逐步生成
    print(f"\n--- 解码器逐步生成 ---")
    dec_input = torch.full((batch_size, 1), SOS_TOKEN, dtype=torch.long)
    dec_hidden = enc_hidden
    dec_cell = enc_cell
    for t in range(trg_len):
        pred, dec_hidden, dec_cell = decoder(dec_input, dec_hidden, dec_cell)
        next_token = pred.argmax(dim=1)
        print(f"  步骤 {t}: 输入={dec_input.squeeze().tolist()}, "
              f"预测token={next_token.tolist()}, "
              f"hidden形状={list(dec_hidden.shape)}")
        dec_input = next_token.unsqueeze(1)

    # 参数统计
    enc_params = sum(p.numel() for p in encoder.parameters())
    dec_params = sum(p.numel() for p in decoder.parameters())
    total_params = enc_params + dec_params
    print(f"\n参数量: Encoder={enc_params:,}, Decoder={dec_params:,}, Total={total_params:,}")

else:
    print("\nPyTorch不可用，展示Seq2Seq概念:")
    print("  Encoder: 将源序列编码为 context vector (hidden, cell)")
    print("  Decoder: 基于 context vector，逐步生成目标序列")
    print("  Teacher Forcing: 训练时以一定概率使用真实目标作为解码器输入")


# ============================================================
# 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图1: 双向LSTM结构示意
ax = axes[0, 0]
ax.set_xlim(-0.5, 6.5)
ax.set_ylim(-1, 5)
ax.set_title('双向LSTM结构示意')
ax.axis('off')

# 前向
for i in range(6):
    ax.add_patch(plt.Rectangle((i - 0.2, 3), 0.4, 0.6,
                               facecolor='lightblue', edgecolor='navy'))
    ax.text(i, 3.3, f'h{i}', ha='center', va='center', fontsize=8)
for i in range(5):
    ax.annotate('', xy=(i + 0.5, 3.3), xytext=(i + 0.8, 3.3),
                arrowprops=dict(arrowstyle='->', color='blue'))
ax.text(-0.3, 3.3, '前向→', ha='right', fontsize=9, color='blue')

# 后向
for i in range(6):
    ax.add_patch(plt.Rectangle((i - 0.2, 1.5), 0.4, 0.6,
                               facecolor='lightyellow', edgecolor='orange'))
    ax.text(i, 1.8, f'h\'{i}', ha='center', va='center', fontsize=8)
for i in range(5, 0, -1):
    ax.annotate('', xy=(i - 0.5, 1.8), xytext=(i - 0.8, 1.8),
                arrowprops=dict(arrowstyle='->', color='orange'))
ax.text(-0.3, 1.8, '←后向', ha='right', fontsize=9, color='orange')

# 拼接输出
for i in range(6):
    ax.add_patch(plt.Rectangle((i - 0.2, -0.5), 0.4, 0.6,
                               facecolor='lightgreen', edgecolor='green'))
    ax.text(i, -0.2, f'y{i}', ha='center', va='center', fontsize=8)
    ax.annotate('', xy=(i, 0.1), xytext=(i, 1.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))
    ax.annotate('', xy=(i, 0.1), xytext=(i, 3.0),
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

ax.text(3, 4.5, 'y_t = concat(h_forward_t, h_backward_t)',
        ha='center', fontsize=10, style='italic', color='green')
ax.text(3, -1, '每个时间步的输出 = hidden_size * 2',
        ha='center', fontsize=9, color='darkgreen')

# 图2: Seq2Seq结构示意
ax = axes[0, 1]
ax.set_xlim(-1, 10)
ax.set_ylim(-0.5, 5)
ax.set_title('Encoder-Decoder (Seq2Seq) 结构')
ax.axis('off')

# Encoder
ax.text(2, 4.5, 'Encoder', ha='center', fontsize=12, fontweight='bold', color='navy')
for i in range(4):
    ax.add_patch(plt.Rectangle((i * 1.1, 2.5), 0.8, 1.2,
                               facecolor='lightblue', edgecolor='navy'))
    ax.text(i * 1.1 + 0.4, 3.1, f'E{i}', ha='center', fontsize=9)
ax.annotate('', xy=(2.2, 3.1), xytext=(1.8, 3.1),
            arrowprops=dict(arrowstyle='->', color='navy'))

# Context Vector
ax.add_patch(plt.Rectangle((4.8, 2.7), 1.2, 0.8,
                            facecolor='lightyellow', edgecolor='orange', linewidth=2))
ax.text(5.4, 3.1, 'CTX', ha='center', fontsize=10, fontweight='bold')
ax.annotate('', xy=(4.8, 3.1), xytext=(4.5, 3.1),
            arrowprops=dict(arrowstyle='->', color='orange', lw=2))
ax.text(5.4, 2.3, 'Context\nVector', ha='center', fontsize=8, color='orange')

# Decoder
ax.text(7.5, 4.5, 'Decoder', ha='center', fontsize=12, fontweight='bold', color='darkgreen')
for i in range(3):
    ax.add_patch(plt.Rectangle((6 + i * 1.1, 2.5), 0.8, 1.2,
                               facecolor='lightgreen', edgecolor='green'))
    ax.text(6 + i * 1.1 + 0.4, 3.1, f'D{i}', ha='center', fontsize=9)
    if i < 2:
        ax.annotate('', xy=(6 + (i + 1) * 1.1, 3.1), xytext=(6 + i * 1.1 + 0.8, 3.1),
                    arrowprops=dict(arrowstyle='->', color='green'))
ax.annotate('', xy=(6.0, 3.1), xytext=(6.0, 3.1),
            arrowprops=dict(arrowstyle='->', color='green'))
ax.annotate('', xy=(6, 3.1), xytext=(5.9, 3.1),
            arrowprops=dict(arrowstyle='<-', color='green', lw=2))

# 图3: Encoder隐藏状态变化
ax = axes[1, 0]
if PYTORCH_AVAILABLE:
    np_enc = enc_outputs[0].detach().numpy()
    for i in range(min(8, HIDDEN_SIZE)):
        ax.plot(range(src_len), np_enc[:, i], alpha=0.7, linewidth=1.2)
    ax.set_xlabel('时间步')
    ax.set_ylabel('隐藏状态值')
    ax.set_title(f'Encoder隐藏状态变化\n(第1个样本, {min(8, HIDDEN_SIZE)}个维度)')
else:
    np_enc = np.random.randn(8, 32)
    for i in range(8):
        ax.plot(range(8), np_enc[:, i], alpha=0.7)
    ax.set_xlabel('时间步')
    ax.set_ylabel('隐藏状态值')
    ax.set_title('Encoder隐藏状态变化 (模拟)')
ax.grid(True, alpha=0.3)

# 图4: Seq2Seq参数分布饼图
ax = axes[1, 1]
if PYTORCH_AVAILABLE:
    enc_embedding = sum(p.numel() for p in encoder.embedding.parameters())
    enc_lstm = sum(p.numel() for p in encoder.lstm.parameters())
    dec_embedding = sum(p.numel() for p in decoder.embedding.parameters())
    dec_lstm = sum(p.numel() for p in decoder.lstm.parameters())
    dec_fc = sum(p.numel() for p in decoder.fc_out.parameters())

    sizes = [enc_embedding, enc_lstm, dec_embedding, dec_lstm, dec_fc]
    labels = ['Enc Embed', 'Enc LSTM', 'Dec Embed', 'Dec LSTM', 'Dec FC']
else:
    sizes = [1600, 12000, 1280, 12000, 2560]
    labels = ['Enc Embed', 'Enc LSTM', 'Dec Embed', 'Dec LSTM', 'Dec FC']

colors = ['#4e79a7', '#f28e2b', '#59a14f', '#e15759', '#76b7b2']
ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
ax.set_title('Seq2Seq模型参数分布')

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W11/d6_birnn_seq2seq.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n图片已保存: d6_birnn_seq2seq.png")
