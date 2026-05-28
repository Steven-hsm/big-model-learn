"""
W11-d7: IMDB情感分析项目
- 生成合成情感数据（不依赖外部数据集）
- 构建SentimentRNN模型（支持 rnn_type='rnn'|'lstm'|'gru'）
- 完整训练流程：数据准备、模型构建、训练循环、评估
- 对比Simple RNN、LSTM、GRU
- 绘制3个模型的训练曲线
- 打印最终对比表
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from collections import Counter

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 检查PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch未安装! 请运行: pip install torch")
    print("将使用NumPy进行简化演示")


# ============================================================
# 1. 生成合成情感分析数据
# ============================================================
print("=" * 60)
print("1. 生成合成情感分析数据")
print("=" * 60)


def generate_sentiment_data(n_samples=1000, min_len=5, max_len=30, seed=42):
    """生成合成的情感分析数据"""
    np.random.seed(seed)

    positive_words = [
        'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
        'love', 'best', 'perfect', 'awesome', 'beautiful',
        'brilliant', 'outstanding', 'superb', 'enjoy', 'happy',
        'good', 'nice', 'impressive', 'remarkable', 'delightful',
        'recommend', 'enjoyed', 'loved', 'favorite', 'pleased'
    ]

    negative_words = [
        'terrible', 'awful', 'horrible', 'worst', 'bad',
        'hate', 'poor', 'boring', 'disappointing', 'waste',
        'ugly', 'dreadful', 'pathetic', 'mediocre', 'annoying',
        'terribly', 'hated', 'disgusting', 'unpleasant', 'rubbish',
        'fails', 'failed', 'worse', 'negative', 'dull'
    ]

    neutral_words = [
        'the', 'a', 'an', 'is', 'was', 'are', 'were', 'be',
        'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'can',
        'this', 'that', 'these', 'those', 'it', 'its',
        'i', 'you', 'we', 'they', 'he', 'she',
        'my', 'your', 'his', 'her', 'our', 'their',
        'and', 'but', 'or', 'not', 'so', 'if',
        'in', 'on', 'at', 'to', 'for', 'with',
        'movie', 'film', 'story', 'plot', 'scene', 'character',
        'actor', 'director', 'script', 'ending', 'beginning',
        'part', 'just', 'really', 'very', 'quite', 'also',
        'more', 'most', 'much', 'some', 'any', 'all'
    ]

    texts = []
    labels = []

    for _ in range(n_samples):
        label = np.random.randint(0, 2)
        length = np.random.randint(min_len, max_len + 1)

        words = []
        # 添加情感词（确保有足够的信号）
        if label == 1:  # 正面
            n_pos = np.random.randint(2, min(6, length))
            words.extend(np.random.choice(positive_words, n_pos).tolist())
        else:  # 负面
            n_neg = np.random.randint(2, min(6, length))
            words.extend(np.random.choice(negative_words, n_neg).tolist())

        # 填充中性词
        remaining = length - len(words)
        if remaining > 0:
            words.extend(np.random.choice(neutral_words, remaining).tolist())

        # 打乱顺序
        np.random.shuffle(words)
        texts.append(' '.join(words))
        labels.append(label)

    return texts, np.array(labels)


# 生成数据
texts, labels = generate_sentiment_data(n_samples=1000, seed=42)

print(f"\n数据集大小: {len(texts)}")
print(f"正面样本: {np.sum(labels == 1)}")
print(f"负面样本: {np.sum(labels == 0)}")
print(f"正/负比例: {np.sum(labels == 1) / np.sum(labels == 0):.2f}")

print(f"\n示例:")
for i in range(4):
    sentiment = "正面" if labels[i] == 1 else "负面"
    print(f"  [{sentiment}] {texts[i][:80]}...")


# ============================================================
# 2. 文本预处理
# ============================================================
print("\n" + "=" * 60)
print("2. 文本预处理")
print("=" * 60)


def simple_tokenize(text):
    """简单分词"""
    import re
    return re.findall(r'\w+', text.lower())


def build_vocab(texts, min_freq=2, max_size=2000):
    """构建词汇表"""
    counter = Counter()
    for text in texts:
        tokens = simple_tokenize(text)
        counter.update(tokens)

    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, freq in counter.most_common(max_size - 2):
        if freq >= min_freq:
            vocab[word] = len(vocab)

    return vocab


def encode_text(text, vocab, max_len=30):
    """编码文本"""
    tokens = simple_tokenize(text)
    indices = [vocab.get(t, 1) for t in tokens]
    # 截断
    if len(indices) > max_len:
        indices = indices[:max_len]
    # 填充
    while len(indices) < max_len:
        indices.append(0)
    return indices


# 构建词汇表
vocab = build_vocab(texts, min_freq=2, max_size=2000)
print(f"词汇表大小: {len(vocab)}")

# 编码所有文本
max_len = 30
encoded_texts = np.array([encode_text(t, vocab, max_len) for t in texts])
print(f"编码后形状: {encoded_texts.shape}")

# 划分训练/测试集
split_idx = int(0.8 * len(texts))
X_train = encoded_texts[:split_idx]
y_train = labels[:split_idx]
X_test = encoded_texts[split_idx:]
y_test = labels[split_idx:]

print(f"训练集: {len(X_train)} 样本")
print(f"测试集: {len(X_test)} 样本")


# ============================================================
# 3. PyTorch模型实现与训练
# ============================================================
if PYTORCH_AVAILABLE:
    print("\n" + "=" * 60)
    print("3. SentimentRNN 模型 (PyTorch)")
    print("=" * 60)

    class SentimentDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.LongTensor(X)
            self.y = torch.FloatTensor(y)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    class SentimentRNN(nn.Module):
        """可配置的情感分析RNN模型"""

        def __init__(self, vocab_size, embed_dim, hidden_size,
                     num_layers=1, rnn_type='lstm', dropout=0.3, bidirectional=False):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.rnn_type = rnn_type
            self.bidirectional = bidirectional
            self.num_directions = 2 if bidirectional else 1

            # Embedding层
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

            # RNN层
            rnn_input = embed_dim
            if rnn_type == 'rnn':
                self.rnn = nn.RNN(rnn_input, hidden_size, num_layers,
                                  batch_first=True, dropout=dropout if num_layers > 1 else 0,
                                  bidirectional=bidirectional)
            elif rnn_type == 'lstm':
                self.rnn = nn.LSTM(rnn_input, hidden_size, num_layers,
                                   batch_first=True, dropout=dropout if num_layers > 1 else 0,
                                   bidirectional=bidirectional)
            elif rnn_type == 'gru':
                self.rnn = nn.GRU(rnn_input, hidden_size, num_layers,
                                  batch_first=True, dropout=dropout if num_layers > 1 else 0,
                                  bidirectional=bidirectional)
            else:
                raise ValueError(f"不支持的rnn_type: {rnn_type}")

            # 分类头
            fc_input = hidden_size * self.num_directions
            self.dropout = nn.Dropout(dropout)
            self.fc1 = nn.Linear(fc_input, hidden_size // 2)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size // 2, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            # x: (batch, seq_len)
            embedded = self.embedding(x)  # (batch, seq_len, embed_dim)

            if self.rnn_type == 'lstm':
                output, (hidden, cell) = self.rnn(embedded)
            else:
                output, hidden = self.rnn(embedded)

            # 取最终隐藏状态
            if self.bidirectional:
                # 拼接最后一层的前向和后向
                hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
            else:
                hidden = hidden[-1]  # (batch, hidden_size)

            hidden = self.dropout(hidden)
            out = self.fc1(hidden)
            out = self.relu(out)
            out = self.dropout(out)
            out = self.fc2(out)
            out = self.sigmoid(out)
            return out.squeeze(1)

        def count_parameters(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # 训练函数
    def train_model(model, train_loader, val_loader, epochs=20, lr=0.001, device='cpu'):
        model = model.to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        train_losses = []
        val_losses = []
        val_accs = []

        for epoch in range(epochs):
            # 训练阶段
            model.train()
            total_loss = 0
            n_batches = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                # 梯度裁剪
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_train_loss = total_loss / n_batches
            train_losses.append(avg_train_loss)

            # 验证阶段
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            n_val = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
                    n_val += 1

                    preds = (outputs >= 0.5).float()
                    correct += (preds == y_batch).sum().item()
                    total += y_batch.size(0)

            avg_val_loss = val_loss / n_val
            val_acc = correct / total
            val_losses.append(avg_val_loss)
            val_accs.append(val_acc)

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch + 1:3d}/{epochs}: "
                      f"Train Loss={avg_train_loss:.4f}, "
                      f"Val Loss={avg_val_loss:.4f}, "
                      f"Val Acc={val_acc:.4f}")

        return train_losses, val_losses, val_accs

    # 准备数据
    train_dataset = SentimentDataset(X_train, y_train)
    test_dataset = SentimentDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 训练配置
    VOCAB_SIZE = len(vocab)
    EMBED_DIM = 64
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    EPOCHS = 25
    LR = 0.001

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n设备: {device}")

    # 训练3种模型
    results = {}
    for rnn_type in ['rnn', 'lstm', 'gru']:
        print(f"\n--- 训练 {rnn_type.upper()} 模型 ---")
        model = SentimentRNN(
            vocab_size=VOCAB_SIZE,
            embed_dim=EMBED_DIM,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            rnn_type=rnn_type,
            dropout=0.3,
            bidirectional=False
        )

        print(f"参数量: {model.count_parameters():,}")

        start_time = time.time()
        train_losses, val_losses, val_accs = train_model(
            model, train_loader, test_loader,
            epochs=EPOCHS, lr=LR, device=device
        )
        train_time = time.time() - start_time

        results[rnn_type] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accs': val_accs,
            'params': model.count_parameters(),
            'time': train_time,
            'final_acc': val_accs[-1],
            'best_acc': max(val_accs),
        }

        print(f"  训练时间: {train_time:.2f}s")
        print(f"  最佳验证准确率: {max(val_accs):.4f}")

    # ============================================================
    # 对比表
    # ============================================================
    print(f"\n{'=' * 70}")
    print(f"{'模型对比总结':^70}")
    print(f"{'=' * 70}")
    print(f"{'指标':<18} {'Simple RNN':>15} {'LSTM':>15} {'GRU':>15}")
    print(f"{'-' * 70}")
    print(f"{'参数量':<18} {results['rnn']['params']:>15,} {results['lstm']['params']:>15,} {results['gru']['params']:>15,}")
    print(f"{'训练时间(s)':<18} {results['rnn']['time']:>15.2f} {results['lstm']['time']:>15.2f} {results['gru']['time']:>15.2f}")
    print(f"{'最终验证准确率':<18} {results['rnn']['final_acc']:>15.4f} {results['lstm']['final_acc']:>15.4f} {results['gru']['final_acc']:>15.4f}")
    print(f"{'最佳验证准确率':<18} {results['rnn']['best_acc']:>15.4f} {results['lstm']['best_acc']:>15.4f} {results['gru']['best_acc']:>15.4f}")
    print(f"{'最终训练Loss':<18} {results['rnn']['train_losses'][-1]:>15.4f} {results['lstm']['train_losses'][-1]:>15.4f} {results['gru']['train_losses'][-1]:>15.4f}")
    print(f"{'最终验证Loss':<18} {results['rnn']['val_losses'][-1]:>15.4f} {results['lstm']['val_losses'][-1]:>15.4f} {results['gru']['val_losses'][-1]:>15.4f}")
    print(f"{'=' * 70}")

    best_model = max(results.keys(), key=lambda k: results[k]['best_acc'])
    print(f"\n最佳模型: {best_model.upper()} (准确率={results[best_model]['best_acc']:.4f})")

else:
    # NumPy fallback
    print("\n" + "=" * 60)
    print("3. NumPy简化实现 (PyTorch不可用)")
    print("=" * 60)

    def sigmoid_np(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    class NumPyRNNClassifier:
        """简化的NumPy RNN分类器"""

        def __init__(self, vocab_size, embed_dim, hidden_size, rnn_type='rnn', seed=42):
            np.random.seed(seed)
            self.vocab_size = vocab_size
            self.embed_dim = embed_dim
            self.hidden_size = hidden_size
            self.rnn_type = rnn_type

            # Embedding
            self.emb = np.random.randn(vocab_size, embed_dim) * 0.1

            scale = 0.1
            if rnn_type == 'rnn':
                self.W_ih = np.random.randn(embed_dim, hidden_size) * scale
                self.W_hh = np.random.randn(hidden_size, hidden_size) * scale
                self.b_h = np.zeros(hidden_size)
                self.n_rnn_params = self.W_ih.size + self.W_hh.size + self.b_h.size
            elif rnn_type == 'lstm':
                self.W = np.random.randn(embed_dim + hidden_size, hidden_size * 4) * scale
                self.b = np.zeros(hidden_size * 4)
                self.b[hidden_size:2 * hidden_size] = 1.0
                self.n_rnn_params = self.W.size + self.b.size
            elif rnn_type == 'gru':
                self.W_z = np.random.randn(embed_dim + hidden_size, hidden_size) * scale
                self.W_r = np.random.randn(embed_dim + hidden_size, hidden_size) * scale
                self.W_h = np.random.randn(embed_dim + hidden_size, hidden_size) * scale
                self.b_z = np.zeros(hidden_size)
                self.b_r = np.zeros(hidden_size)
                self.b_h = np.zeros(hidden_size)
                self.n_rnn_params = self.W_z.size + self.W_r.size + self.W_h.size + self.b_z.size + self.b_r.size + self.b_h.size

            # 分类头
            self.W_fc1 = np.random.randn(hidden_size, hidden_size // 2) * scale
            self.b_fc1 = np.zeros(hidden_size // 2)
            self.W_fc2 = np.random.randn(hidden_size // 2, 1) * scale
            self.b_fc2 = np.zeros(1)

        def _rnn_step(self, x_t, h):
            return np.tanh(x_t @ self.W_ih + h @ self.W_hh + self.b_h)

        def _lstm_step(self, x_t, h, c):
            concat = np.concatenate([x_t, h])
            gates = concat @ self.W + self.b
            hs = self.hidden_size
            i = sigmoid_np(gates[:hs])
            f = sigmoid_np(gates[hs:2 * hs])
            g = np.tanh(gates[2 * hs:3 * hs])
            o = sigmoid_np(gates[3 * hs:])
            c_new = f * c + i * g
            h_new = o * np.tanh(c_new)
            return h_new, c_new

        def _gru_step(self, x_t, h):
            concat = np.concatenate([x_t, h])
            z = sigmoid_np(concat @ self.W_z + self.b_z)
            r = sigmoid_np(concat @ self.W_r + self.b_r)
            concat_r = np.concatenate([x_t, r * h])
            h_tilde = np.tanh(concat_r @ self.W_h + self.b_h)
            return (1 - z) * h + z * h_tilde

        def forward(self, x_indices):
            # Embedding
            embedded = self.emb[x_indices]  # (seq_len, embed_dim)

            h = np.zeros(self.hidden_size)
            c = np.zeros(self.hidden_size)

            for t in range(len(embedded)):
                if self.rnn_type == 'rnn':
                    h = self._rnn_step(embedded[t], h)
                elif self.rnn_type == 'lstm':
                    h, c = self._lstm_step(embedded[t], h, c)
                elif self.rnn_type == 'gru':
                    h = self._gru_step(embedded[t], h)

            # 分类头
            hidden = np.maximum(0, h @ self.W_fc1 + self.b_fc1)
            logit = (hidden @ self.W_fc2 + self.b_fc2)[0]
            return sigmoid_np(logit)

        def count_parameters(self):
            fc_params = self.W_fc1.size + self.b_fc1.size + self.W_fc2.size + self.b_fc2.size
            return self.emb.size + self.n_rnn_params + fc_params

    # 训练简化版
    EPOCHS = 15
    results = {}

    for rnn_type in ['rnn', 'lstm', 'gru']:
        print(f"\n--- 训练 {rnn_type.upper()} (NumPy简化版) ---")
        model = NumPyRNNClassifier(len(vocab), 32, 32, rnn_type=rnn_type)
        print(f"参数量: {model.count_parameters():,}")

        lr = 0.005
        train_accs = []
        test_accs = []

        start_time = time.time()
        for epoch in range(EPOCHS):
            # 简单评估（不做完整梯度下降，只看前向传播准确率）
            # 随机扰动模拟训练
            noise_scale = max(0.01, 0.5 * (1 - epoch / EPOCHS))

            # 训练准确率
            train_correct = 0
            for i in range(len(X_train)):
                pred = model.forward(X_train[i])
                pred_label = 1 if pred >= 0.5 else 0
                if pred_label == y_train[i]:
                    train_correct += 1

                # 简单更新：根据误差方向调整
                error = pred - y_train[i]
                model.emb[X_train[i]] -= lr * error * noise_scale * 0.01

            train_acc = train_correct / len(X_train)
            train_accs.append(train_acc)

            # 测试准确率
            test_correct = 0
            for i in range(len(X_test)):
                pred = model.forward(X_test[i])
                pred_label = 1 if pred >= 0.5 else 0
                if pred_label == y_test[i]:
                    test_correct += 1

            test_acc = test_correct / len(X_test)
            test_accs.append(test_acc)

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch + 1:3d}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")

        train_time = time.time() - start_time
        results[rnn_type] = {
            'train_accs': train_accs,
            'test_accs': test_accs,
            'params': model.count_parameters(),
            'time': train_time,
            'final_acc': test_accs[-1],
            'best_acc': max(test_accs),
        }

    # NumPy版对比表
    print(f"\n{'=' * 60}")
    print(f"{'模型对比总结 (NumPy简化版)':^60}")
    print(f"{'=' * 60}")
    print(f"{'指标':<15} {'Simple RNN':>13} {'LSTM':>13} {'GRU':>13}")
    print(f"{'-' * 60}")
    print(f"{'参数量':<15} {results['rnn']['params']:>13,} {results['lstm']['params']:>13,} {results['gru']['params']:>13,}")
    print(f"{'训练时间(s)':<15} {results['rnn']['time']:>13.2f} {results['lstm']['time']:>13.2f} {results['gru']['time']:>13.2f}")
    print(f"{'最佳测试准确率':<15} {results['rnn']['best_acc']:>13.4f} {results['lstm']['best_acc']:>13.4f} {results['gru']['best_acc']:>13.4f}")
    print(f"{'最终测试准确率':<15} {results['rnn']['final_acc']:>13.4f} {results['lstm']['final_acc']:>13.4f} {results['gru']['final_acc']:>13.4f}")
    print(f"{'=' * 60}")


# ============================================================
# 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

colors = {'rnn': '#e74c3c', 'lstm': '#3498db', 'gru': '#2ecc71'}
names = {'rnn': 'Simple RNN', 'lstm': 'LSTM', 'gru': 'GRU'}

if PYTORCH_AVAILABLE:
    # 图1: 训练损失
    ax = axes[0, 0]
    for rnn_type in ['rnn', 'lstm', 'gru']:
        ax.plot(range(1, EPOCHS + 1), results[rnn_type]['train_losses'],
                color=colors[rnn_type], label=f'{names[rnn_type]}', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('训练损失对比')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 图2: 验证损失
    ax = axes[0, 1]
    for rnn_type in ['rnn', 'lstm', 'gru']:
        ax.plot(range(1, EPOCHS + 1), results[rnn_type]['val_losses'],
                color=colors[rnn_type], label=f'{names[rnn_type]}', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('验证损失对比')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 图3: 验证准确率
    ax = axes[1, 0]
    for rnn_type in ['rnn', 'lstm', 'gru']:
        ax.plot(range(1, EPOCHS + 1), results[rnn_type]['val_accs'],
                color=colors[rnn_type], label=f'{names[rnn_type]}', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('准确率')
    ax.set_title('验证准确率对比')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 图4: 参数量与准确率柱状图
    ax = axes[1, 1]
    x_pos = np.arange(3)
    bar_width = 0.35
    params = [results[k]['params'] for k in ['rnn', 'lstm', 'gru']]
    accs = [results[k]['best_acc'] for k in ['rnn', 'lstm', 'gru']]

    ax2 = ax.twinx()
    bars1 = ax.bar(x_pos - bar_width / 2, [p / 1000 for p in params],
                   bar_width, label='参数量(K)', color=list(colors.values()), alpha=0.6)
    bars2 = ax2.bar(x_pos + bar_width / 2, accs,
                    bar_width, label='最佳准确率', color=list(colors.values()), alpha=0.9,
                    edgecolor='black')

    ax.set_xlabel('模型')
    ax.set_ylabel('参数量 (K)')
    ax2.set_ylabel('准确率')
    ax.set_title('参数量 vs 最佳准确率')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([names[k] for k in ['rnn', 'lstm', 'gru']])
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

else:
    # NumPy版可视化
    # 图1: 训练准确率
    ax = axes[0, 0]
    for rnn_type in ['rnn', 'lstm', 'gru']:
        ax.plot(range(1, len(results[rnn_type]['train_accs']) + 1),
                results[rnn_type]['train_accs'],
                color=colors[rnn_type], label=f'{names[rnn_type]}', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('准确率')
    ax.set_title('训练准确率对比 (NumPy简化版)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 图2: 测试准确率
    ax = axes[0, 1]
    for rnn_type in ['rnn', 'lstm', 'gru']:
        ax.plot(range(1, len(results[rnn_type]['test_accs']) + 1),
                results[rnn_type]['test_accs'],
                color=colors[rnn_type], label=f'{names[rnn_type]}', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('准确率')
    ax.set_title('测试准确率对比 (NumPy简化版)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 图3: 参数量对比
    ax = axes[1, 0]
    x_pos = range(3)
    params = [results[k]['params'] for k in ['rnn', 'lstm', 'gru']]
    bar_colors = [colors[k] for k in ['rnn', 'lstm', 'gru']]
    bars = ax.bar(x_pos, params, color=bar_colors, edgecolor='black', alpha=0.7)
    ax.set_xlabel('模型')
    ax.set_ylabel('参数量')
    ax.set_title('参数量对比')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([names[k] for k in ['rnn', 'lstm', 'gru']])
    for bar, val in zip(bars, params):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:,}', ha='center', va='bottom', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # 图4: 最佳准确率对比
    ax = axes[1, 1]
    accs = [results[k]['best_acc'] for k in ['rnn', 'lstm', 'gru']]
    bars = ax.bar(x_pos, accs, color=bar_colors, edgecolor='black', alpha=0.7)
    ax.set_xlabel('模型')
    ax.set_ylabel('准确率')
    ax.set_title('最佳测试准确率对比')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([names[k] for k in ['rnn', 'lstm', 'gru']])
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W11/d7_imdb_sentiment.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n图片已保存: d7_imdb_sentiment.png")
print("\n完成!")
