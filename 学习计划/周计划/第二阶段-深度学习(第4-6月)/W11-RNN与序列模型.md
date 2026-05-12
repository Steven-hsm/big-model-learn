# 第11周 - RNN与序列模型

> 本周学习处理序列数据的神经网络架构。从基础RNN到LSTM/GRU，理解它们如何处理文本、语音等序列数据，以及如何解决长程依赖问题。

---

## 一、本周目标

1. 理解RNN的循环结构和参数共享机制，掌握BPTT反向传播
2. 深入理解LSTM的门控机制（遗忘门、输入门、输出门）和细胞状态更新
3. 掌握GRU的简化结构，能对比LSTM和GRU的优劣
4. 学会文本预处理Pipeline（分词、词表、编码、padding）
5. 完成IMDB情感分析项目，对比Simple RNN、LSTM、GRU三种模型

---

## 二、时间安排

### 工作日（每晚2小时）

| 日期 | 主题 | 时间分配 |
|------|------|----------|
| Day 1 (周一) | RNN基础 | 理论50min + 代码实现70min |
| Day 2 (周二) | RNN的问题 | 理论40min + 数学推导80min |
| Day 3 (周三) | LSTM | 理论60min + 手动推导60min |
| Day 4 (周四) | GRU | 理论40min + 代码实现80min |
| Day 5 (周五) | 文本预处理 | 理论30min + Pipeline实现90min |

### 周末（6-8小时）

| 日期 | 主题 | 时间分配 |
|------|------|----------|
| Day 6 (周六) | 双向RNN与Seq2Seq | 理论60min + 代码实现120min |
| Day 7 (周日) | IMDB情感分析项目 | 数据准备60min + 模型训练180min + 分析总结60min |

---

## 三、详细学习内容

### Day 1: RNN基础

#### 1. 序列数据特点

序列数据有时序/顺序关系，当前输出可能与之前的所有输入有关：
- **文本**：第i个词的含义依赖上下文
- **语音**：当前帧与前后帧相关
- **时序**：股票价格、天气数据等有时间依赖

与普通MLP的区别：MLP假设每个输入独立同分布，RNN利用序列的历史信息。

#### 2. RNN循环结构

RNN的核心公式：
```
h_t = f(W_hh · h_{t-1} + W_xh · x_t + b_h)
y_t = W_hy · h_t + b_y
```

其中：
- `x_t`：时间步t的输入
- `h_{t-1}`：上一个时间步的隐藏状态（记忆）
- `h_t`：当前时间步的隐藏状态
- `W_hh`：隐藏状态到隐藏状态的权重，shape `(hidden_dim, hidden_dim)`
- `W_xh`：输入到隐藏状态的权重，shape `(hidden_dim, input_dim)`
- `f`：激活函数，通常用tanh

#### 3. 时间步展开

将RNN按时间步展开：
```
x_1 → [RNN] → h_1 → [RNN] → h_2 → [RNN] → h_3 → ... → h_T
         ↑              ↑              ↑
        h_0           h_1           h_2
```

展开后看起来像一个很深的深层网络，每层共享相同的参数 `W_hh, W_xh, b_h`。

#### 4. 参数共享

RNN的所有时间步使用同一组参数 `W_hh, W_xh, W_hy, b_h, b_y`。

好处：
- 参数量不随序列长度增加
- 可以处理变长序列
- 类似卷积中的参数共享

参数量计算：
```
input_dim = vocab_size (或embedding_dim)
hidden_dim = 128

W_xh: input_dim × hidden_dim
W_hh: hidden_dim × hidden_dim
b_h:  hidden_dim
W_hy: hidden_dim × output_dim
b_y:  output_dim
```

#### 5. BPTT (Backpropagation Through Time)

RNN的反向传播需要在时间维度上展开：
```
∂L/∂W_hh = Σ_t ∂L/∂h_t · ∂h_t/∂h_{t-1} · ... · ∂h_1/∂W_hh
```

每个时间步都会贡献一个梯度项，这些梯度项通过链式法则相乘。

```python
# NumPy实现Simple RNN前向传播
import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        # 初始化参数
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        self.bh = np.zeros(hidden_size)
        self.by = np.zeros(output_size)

    def forward(self, x_seq):
        """x_seq: (seq_len, input_size)"""
        h = np.zeros(self.hidden_size)
        self.h_states = [h]
        self.outputs = []

        for t in range(len(x_seq)):
            h = np.tanh(self.Wxh @ x_seq[t] + self.Whh @ h + self.bh)
            y = self.Why @ h + self.by
            self.h_states.append(h)
            self.outputs.append(y)

        return self.outputs, self.h_states
```

---

### Day 2: RNN的问题

#### 1. 梯度消失（长程依赖问题）

在BPTT中，梯度经过多个时间步的乘法：
```
∂h_t/∂h_{t-k} = Π_{i=k}^{t} ∂h_i/∂h_{i-1}
               = Π_{i=k}^{t} W_hh · diag(tanh'(z_i))
```

tanh的导数范围是 `(0, 1)`，最大值为1。如果 `W_hh` 的特征值小于1，那么经过多次乘法，梯度会指数级衰减趋近于0。

**具体例子**：考虑句子 "我出生在___（很长的描述）___的中国城市___"
- RNN需要将"我出生在"的信息传递到最后来预测"北京"
- 但经过很多时间步后，梯度已经消失，模型无法学到这个远距离依赖

#### 2. 梯度爆炸

如果 `W_hh` 的特征值大于1，梯度会指数级增长：
```
梯度爆炸: ||∂L/∂h_0|| → ∞
```

解决方案——**梯度裁剪**：
```python
import torch

# PyTorch梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

# 原理：如果梯度范数超过max_norm，按比例缩放
# g = g * (max_norm / ||g||)  if ||g|| > max_norm
```

梯度裁剪是RNN训练的标配操作。

#### 3. 长期依赖问题总结

| 问题 | 表现 | 原因 | 解决方案 |
|------|------|------|----------|
| 梯度消失 | 无法学到远距离信息 | 多次小梯度相乘 | LSTM/GRU |
| 梯度爆炸 | 训练不稳定/NaN | 多次大梯度相乘 | 梯度裁剪 |
| 长期依赖 | 忘记早期信息 | 隐藏状态被不断覆盖 | 门控机制 |

---

### Day 3: LSTM (Long Short-Term Memory)

LSTM通过门控机制和细胞状态(cell state)解决了RNN的长期依赖问题。

#### 1. LSTM核心思想

LSTM引入了一个"传送带"——细胞状态 `C_t`，信息可以在上面几乎不变地流动。三个门控制信息的进出。

#### 2. 遗忘门 (Forget Gate)

```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
```

决定从细胞状态中丢弃什么信息：
- `f_t` 接近1：保留该信息
- `f_t` 接近0：丢弃该信息
- `σ` 是Sigmoid函数，输出范围 `(0, 1)`

#### 3. 输入门 (Input Gate)

```
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)     # 决定更新什么
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)  # 候选新值
```

输入门决定哪些新信息要写入细胞状态。

#### 4. 细胞状态更新

```
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
```

这是LSTM的关键：
- `f_t ⊙ C_{t-1}`：遗忘旧信息（逐元素乘法）
- `i_t ⊙ C̃_t`：添加新信息
- **加法更新**（不是乘法）——这是LSTM缓解梯度消失的核心！梯度可以通过 `C_t` 几乎无损地回传

#### 5. 输出门 (Output Gate)

```
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)  # 决定输出什么
h_t = o_t ⊙ tanh(C_t)                 # 细胞状态经过tanh后输出
```

#### 6. 完整LSTM计算流程

```
输入: x_t, h_{t-1}, C_{t-1}

合并输入: [h_{t-1}, x_t]

遗忘门: f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
输入门: i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
候选值: C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
细胞状态: C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
输出门: o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
隐藏状态: h_t = o_t ⊙ tanh(C_t)
```

#### 7. 为什么LSTM缓解梯度消失

细胞状态 `C_t` 的梯度传播：
```
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t

∂C_t/∂C_{t-1} = f_t  （遗忘门的值）
```

与Simple RNN不同，LSTM的梯度是**加法传播**的：
```
∂L/∂C_0 = ∂L/∂C_T · (f_T · f_{T-1} · ... · f_1) + 其他路径
```

遗忘门 `f_t` 是可学习的——模型可以学会在需要长程记忆时将 `f_t` 保持接近1。

```python
# PyTorch中使用LSTM
import torch.nn as nn

lstm = nn.LSTM(
    input_size=128,     # 输入特征维度
    hidden_size=256,    # 隐藏状态维度
    num_layers=2,       # LSTM层数
    batch_first=True,   # 输入shape为(batch, seq, feature)
    dropout=0.5,        # 层间Dropout
    bidirectional=True  # 双向LSTM
)

# 输入
x = torch.randn(32, 100, 128)  # (batch, seq_len, input_size)
output, (h_n, c_n) = lstm(x)
# output: (32, 100, 512) 最后一个维度 = hidden_size * 2 (双向)
# h_n: (4, 32, 256)  (num_layers * 2, batch, hidden_size)
# c_n: (4, 32, 256)
```

---

### Day 4: GRU (Gated Recurrent Unit)

GRU是LSTM的简化版本，参数更少，在很多任务上效果相近。

#### 1. GRU结构

```
重置门: r_t = σ(W_r · [h_{t-1}, x_t])
更新门: z_t = σ(W_z · [h_{t-1}, x_t])
候选状态: h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t])
最终状态: h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

#### 2. GRU vs LSTM对比

| 特性 | LSTM | GRU |
|------|------|-----|
| 门数量 | 3个（遗忘门、输入门、输出门） | 2个（重置门、更新门） |
| 细胞状态 | 有独立的C_t | 没有独立的细胞状态 |
| 参数量 | 约4×hidden² | 约3×hidden² |
| 训练速度 | 稍慢 | 稍快 |
| 记忆能力 | 更强（独立的细胞状态） | 稍弱 |
| 适用场景 | 需要更强记忆的任务 | 数据量较小或需要更快训练 |

**选择建议**：
- 数据量大、序列长 → 优先LSTM
- 数据量小、追求速度 → 优先GRU
- 实际项目中两者都要试

```python
# PyTorch GRU
gru = nn.GRU(
    input_size=128,
    hidden_size=256,
    num_layers=2,
    batch_first=True,
    dropout=0.5,
    bidirectional=True
)

output, h_n = gru(x)  # GRU没有细胞状态c_n
```

---

### Day 5: 文本预处理

#### 1. 分词 Tokenization

```python
# 中文分词
import jieba

text = "自然语言处理是人工智能的重要方向"
tokens = jieba.lcut(text)
# ['自然语言', '处理', '是', '人工智能', '的', '重要', '方向']

# 英文分词
from nltk.tokenize import word_tokenize

text = "Natural language processing is important."
tokens = word_tokenize(text)
# ['Natural', 'language', 'processing', 'is', 'important', '.']
```

#### 2. 构建词表 Vocabulary

```python
from collections import Counter

class Vocabulary:
    def __init__(self, freq_threshold=2):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = len(self.itos)

        for sentence in sentence_list:
            for word in sentence:
                frequencies[word] += 1

        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        return [self.stoi.get(word, self.stoi["<UNK>"]) for word in text]

    def __len__(self):
        return len(self.itos)
```

#### 3. 词嵌入 Word2Vec

**CBOW (Continuous Bag of Words)**：通过上下文预测中心词。
```
输入: [The, cat, sits, on, the] → 预测: mat
```

**Skip-gram**：通过中心词预测上下文。
```
输入: mat → 预测: [The, cat, sits, on, the]
```

```python
# 使用Gensim训练Word2Vec
from gensim.models import Word2Vec

sentences = [["the", "cat", "sat", "on", "the", "mat"],
             ["the", "dog", "lay", "on", "the", "rug"]]

model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# 获取词向量
vector = model.wv['cat']  # shape: (100,)

# 相似词
similar = model.wv.most_similar('cat', topn=5)
```

#### 4. Padding与Truncation

```python
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

# 不同长度的序列需要padding对齐
def collate_fn(batch):
    texts, labels = zip(*batch)
    # pad_sequence自动将不同长度的tensor padding到同一长度
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    lengths = torch.tensor([len(t) for t in texts])
    labels = torch.tensor(labels)
    return texts_padded, lengths, labels
```

#### 5. pack_padded_sequence

`pack_padded_sequence` 将padding后的序列"打包"，让RNN跳过padding部分：

```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# 假设已经padding的输入
# texts_padded: (batch, max_len)
# lengths: 每个样本的真实长度

embedded = embedding(texts_padded)  # (batch, max_len, embed_dim)

# 打包（按长度从长到短排序）
packed = pack_padded_sequence(embedded, lengths.cpu(),
                               batch_first=True, enforce_sorted=False)

# RNN只处理真实数据，跳过padding
output_packed, (h_n, c_n) = lstm(packed)

# 解包（如果需要每个时间步的输出）
output, _ = pad_packed_sequence(output_packed, batch_first=True)
```

为什么需要pack：RNN不需要处理padding的0，节省计算且不影响隐藏状态。

---

### Day 6: 双向RNN与Seq2Seq

#### 1. 双向BiLSTM

正向LSTM从左到右读取序列，反向LSTM从右到左读取序列，最后拼接：
```
h_t = [→h_t; ←h_t]  正向和反向隐藏状态拼接
```

优势：每个位置的信息同时包含前文和后文的上下文。

```python
bilstm = nn.LSTM(input_size=128, hidden_size=256,
                  bidirectional=True, batch_first=True)
# 输出维度 = 256 * 2 = 512

# 取最后一个时间步的输出（用于分类）
output, (h_n, c_n) = bilstm(x)
# h_n shape: (2, batch, 256)  2 = num_directions
# 拼接正向和反向的最后一个隐藏状态
final_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (batch, 512)
```

**注意**：双向RNN不能用于语言模型（生成任务），因为语言模型只能看到前文。但可以用于分类、NER等任务。

#### 2. 多层堆叠RNN

```python
# 2层LSTM
lstm = nn.LSTM(input_size=128, hidden_size=256,
                num_layers=2, batch_first=True, dropout=0.3)
```

第一层的输出作为第二层的输入，增加模型容量。

#### 3. Seq2Seq基础

Seq2Seq (Sequence to Sequence) 由两部分组成：

**Encoder（编码器）**：将输入序列编码为一个固定长度的上下文向量(context vector)
```
输入序列 → Encoder LSTM → 最后的隐藏状态 (h, c) = 上下文向量
```

**Decoder（解码器）**：根据上下文向量逐步生成输出序列
```
上下文向量 → Decoder LSTM → 逐步生成输出序列
```

```python
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, batch_first=True)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell  # 上下文向量

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(1)  # (batch, 1)
        embedded = self.embedding(input)
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden, cell
```

---

## 四、代码练习

### Day 3：手动推导LSTM一个时间步

```python
"""任务：手动计算LSTM一个时间步的全部计算"""
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 参数设置
input_size, hidden_size = 4, 3

# 随机初始化参数
np.random.seed(42)
W_f = np.random.randn(hidden_size, hidden_size + input_size)
W_i = np.random.randn(hidden_size, hidden_size + input_size)
W_C = np.random.randn(hidden_size, hidden_size + input_size)
W_o = np.random.randn(hidden_size, hidden_size + input_size)

b_f = np.zeros(hidden_size)
b_i = np.zeros(hidden_size)
b_C = np.zeros(hidden_size)
b_o = np.zeros(hidden_size)

# 输入
x_t = np.array([1.0, 0.5, -0.3, 0.8])  # (input_size,)
h_prev = np.array([0.1, -0.2, 0.3])     # (hidden_size,)
C_prev = np.array([0.5, -0.1, 0.2])     # (hidden_size,)

# 合并输入
combined = np.concatenate([h_prev, x_t])  # (hidden_size + input_size,)

# 遗忘门
f_t = sigmoid(W_f @ combined + b_f)
print(f"遗忘门 f_t: {f_t}")

# 输入门
i_t = sigmoid(W_i @ combined + b_i)
print(f"输入门 i_t: {i_t}")

# 候选细胞状态
C_tilde = np.tanh(W_C @ combined + b_C)
print(f"候选值 C̃_t: {C_tilde}")

# 新细胞状态
C_t = f_t * C_prev + i_t * C_tilde
print(f"细胞状态 C_t: {C_t}")

# 输出门
o_t = sigmoid(W_o @ combined + b_o)
print(f"输出门 o_t: {o_t}")

# 新隐藏状态
h_t = o_t * np.tanh(C_t)
print(f"隐藏状态 h_t: {h_t}")
```

### Day 5：文本预处理Pipeline

```python
"""任务：实现完整的文本预处理Pipeline"""
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter

class TextPipeline:
    def __init__(self, freq_threshold=2, max_len=200):
        self.freq_threshold = freq_threshold
        self.max_len = max_len
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        self.word_counts = Counter()

    def fit(self, texts):
        """构建词表"""
        for text in texts:
            words = text.lower().split()
            self.word_counts.update(words)

        idx = len(self.vocab)
        for word, count in self.word_counts.items():
            if count >= self.freq_threshold:
                self.vocab[word] = idx
                idx += 1

        self.idx2word = {v: k for k, v in self.vocab.items()}
        print(f"词表大小: {len(self.vocab)}")
        return self

    def encode(self, text):
        """文本→索引序列"""
        words = text.lower().split()
        # 截断
        words = words[:self.max_len]
        indices = [self.vocab.get(w, self.vocab['<UNK>']) for w in words]
        return torch.tensor(indices, dtype=torch.long)

    def decode(self, indices):
        """索引序列→文本"""
        return ' '.join([self.idx2word.get(idx, '<UNK>') for idx in indices])

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, pipeline):
        self.texts = texts
        self.labels = labels
        self.pipeline = pipeline

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.pipeline.encode(self.texts[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return encoded, label

def collate_fn(batch):
    """自定义batch组装，处理变长序列"""
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(t) for t in texts])
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return texts_padded, lengths, labels

# 使用示例
pipeline = TextPipeline(freq_threshold=5, max_len=200)
# pipeline.fit(train_texts)
# train_dataset = IMDBDataset(train_texts, train_labels, pipeline)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
```

### Day 7：IMDB情感分析完整项目

```python
"""项目5：IMDB情感分析 - 3模型对比"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset

# ==================== 数据准备 ====================
dataset = load_dataset("imdb")
train_texts = [x['text'] for x in dataset['train']]
train_labels = [x['label'] for x in dataset['train']]
test_texts = [x['text'] for x in dataset['test']]
test_labels = [x['label'] for x in dataset['test']]

# 构建词表和Pipeline（使用Day 5的实现）
pipeline = TextPipeline(freq_threshold=5, max_len=200)
pipeline.fit(train_texts)

train_dataset = IMDBDataset(train_texts, train_labels, pipeline)
test_dataset = IMDBDataset(test_texts, test_labels, pipeline)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

# ==================== 模型定义 ====================
class SentimentRNN(nn.Module):
    """通用RNN情感分类模型"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim,
                 rnn_type='lstm', n_layers=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        rnn_map = {
            'rnn': nn.RNN,
            'lstm': nn.LSTM,
            'gru': nn.GRU,
        }
        self.rnn = rnn_map[rnn_type](
            embed_dim, hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=True
        )

        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, lengths):
        embedded = self.dropout(self.embedding(text))

        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        packed_output, hidden = self.rnn(packed)

        # 拼接双向最后隐藏状态
        if isinstance(hidden, tuple):  # LSTM
            hidden = hidden[0]
        # hidden: (n_layers*2, batch, hidden_dim)
        hidden = self.dropout(torch.cat([hidden[-2], hidden[-1]], dim=1))

        return self.fc(hidden)

# ==================== 训练和评估 ====================
def train_and_evaluate(rnn_type, epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SentimentRNN(
        vocab_size=len(pipeline.vocab),
        embed_dim=128,
        hidden_dim=256,
        output_dim=1,
        rnn_type=rnn_type,
        n_layers=2,
        dropout=0.5
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print(f"\n===== {rnn_type.upper()} =====")
    for epoch in range(epochs):
        # 训练
        model.train()
        total_loss = 0
        for batch in train_loader:
            text, lengths, labels = [x.to(device) for x in batch]
            optimizer.zero_grad()
            predictions = model(text, lengths).squeeze(1)
            loss = criterion(predictions, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()

        # 评估
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in test_loader:
                text, lengths, labels = [x.to(device) for x in batch]
                predictions = model(text, lengths).squeeze(1)
                preds = (torch.sigmoid(predictions) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = 100 * correct / total
        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Acc={acc:.1f}%")

    return acc

# 运行对比
results = {}
for rnn_type in ['rnn', 'lstm', 'gru']:
    results[rnn_type] = train_and_evaluate(rnn_type, epochs=5)

print("\n===== 结果汇总 =====")
for name, acc in results.items():
    print(f"{name.upper()}: {acc:.1f}%")
```

---

## 五、本周产出

### 周末交付物

1. **文本预处理Pipeline**：完整的分词、词表构建、编码、padding工具类 `text_pipeline.py`
2. **IMDB情感分析项目**：3种RNN变体对比的完整代码 `imdb_sentiment.py`
3. **LSTM手动推导**：一个时间步的完整计算过程 `lstm_manual_calc.py`
4. **模型对比报告**：准确率、训练速度、梯度稳定性的对比分析

---

## 六、自测题

### 题1：RNN的梯度消失问题根本原因？

<details>
<summary>参考答案</summary>

RNN在BPTT中，梯度需要经过多个时间步的链式乘法：
```
∂L/∂h_0 = ∂L/∂h_T · Π_{t=1}^{T} (∂h_t/∂h_{t-1})
```

其中 `∂h_t/∂h_{t-1} = W_hh · diag(tanh'(z_t))`。tanh的导数范围是(0,1)，最大值为1。如果 `W_hh` 的特征值也小于1，那么多次乘法后梯度指数级衰减到0。

根本原因是RNN的隐藏状态更新使用**乘法**操作（`h_t = tanh(W_hh·h_{t-1} + ...)`），梯度必须通过这些乘法传递，导致指数衰减。
</details>

### 题2：LSTM的遗忘门、输入门、输出门分别控制什么？

<details>
<summary>参考答案</summary>

- **遗忘门 f_t**：控制从旧的细胞状态 `C_{t-1}` 中保留多少信息。f_t接近1表示保留，接近0表示遗忘。例如处理文本时，遇到新的主语时遗忘门可能学会遗忘旧主语的性别信息。

- **输入门 i_t**：控制多少新信息写入细胞状态。与候选值 `C̃_t` 配合，决定哪些新信息被存储。

- **输出门 o_t**：控制细胞状态 `C_t` 中多少信息作为当前时间步的输出 `h_t`。决定哪些记忆与当前计算相关。
</details>

### 题3：pack_padded_sequence的作用？为什么需要它？

<details>
<summary>参考答案</summary>

`pack_padded_sequence` 将padding后的序列"压缩"，去除padding部分，只保留真实数据。

需要它的原因：
1. **节省计算**：RNN不需要处理padding的0值
2. **保持隐藏状态正确**：如果RNN处理padding的0，最后的隐藏状态会被无意义的0值影响
3. **效率**：特别是当序列长度差异很大时，可以跳过大量padding

使用方式：先padding对齐，再pack压缩，RNN处理packed数据，最后用 `pad_packed_sequence` 解压。
</details>

### 题4：双向RNN为什么比单向好？能用于语言模型吗？

<details>
<summary>参考答案</summary>

双向RNN更好，因为它同时利用了前文和后文的信息。例如在句子"我喜欢苹果___很好吃"中，预测空白处时：
- 正向RNN只看到"我喜欢苹果"
- 双向RNN还能看到"很好吃"，能更准确地理解上下文

**不能用于语言模型**：语言模型的核心约束是"只能看到前文"（自回归生成），如果用了反向RNN，就相当于看到了未来信息，在生成时这些信息是不存在的。

双向RNN适用于：文本分类、NER、机器阅读理解等不需要生成的任务。
</details>

### 题5：GRU相比LSTM的优劣？

<details>
<summary>参考答案</summary>

GRU的优势：
1. **参数更少**：只有2个门vs LSTM的3个门，参数量约为LSTM的3/4
2. **训练更快**：更少的矩阵运算
3. **更不容易过拟合**：参数少，正则化效果好

GRU的劣势：
1. **记忆能力稍弱**：没有独立的细胞状态，所有信息通过隐藏状态传递
2. **在超长序列上可能不如LSTM**：LSTM的细胞状态更适合长期记忆

选择建议：
- 数据量小或需要快速迭代 → GRU
- 序列很长或需要更强的记忆能力 → LSTM
- 实际项目中两者都应该实验对比
</details>

---

## 七、Java开发者提示

### 1. RNN vs 流式处理

RNN处理序列数据的方式和Java中的流式处理(Stream Processing)非常相似：

```java
// RNN的循环结构类似Java的流处理
T state = initialState;  // h_0
for (T item : sequence) {  // 遍历序列
    state = process(item, state);  // h_t = f(x_t, h_{t-1})
}
// state 就是整个序列的表示
```

类比：
| RNN概念 | Java流式处理 |
|---------|-------------|
| 隐藏状态 h_t | 中间状态/累加器 |
| 输入 x_t | 流中的每个元素 |
| 参数共享 | 同一个处理函数 |
| 时间步展开 | for循环展开 |

### 2. LSTM门控 vs 访问控制

LSTM的门控机制类似Java中的权限控制系统：
- **遗忘门** → 清理过期Session/缓存（决定丢弃什么）
- **输入门** → 写权限检查（决定允许什么新信息写入）
- **输出门** → 读权限检查（决定对外暴露什么信息）
- **细胞状态** → 数据库（持久化存储，门控控制读写）

### 3. 梯度消失 vs 对象引用丢失

```java
// 梯度消失类似Java中的引用链断裂
class A { B b; }
class B { C c; }
class C { D d; }

// 如果中间某一层引用为null，后面的对象就无法访问了
// RNN中：如果中间梯度接近0，更前面的层就得不到更新
```

LSTM的解决方案类似给引用加上"弱引用+强引用"双重通道，即使弱引用断了（梯度消失），强引用（细胞状态）仍然保持连接。

### 4. Seq2Seq vs 翻译服务

Seq2Seq架构和微服务中的翻译服务完全对应：
- **Encoder** = 接收请求并解析（将输入序列编码为中间表示）
- **Context Vector** = 序列化的请求对象（固定长度的中间表示）
- **Decoder** = 生成响应（从中间表示逐步生成输出序列）

### 5. 文本预处理 vs ETL Pipeline

文本预处理的每一步都对应ETL中的操作：
```
原始文本 → 分词(Extract) → 词表映射(Transform) → Padding对齐(Load) → 模型输入
```

和你在Java中处理数据库数据到前端展示的流程一样：
```
数据库 → 查询(Extract) → DTO转换(Transform) → 格式化(Load) → JSON响应
```
