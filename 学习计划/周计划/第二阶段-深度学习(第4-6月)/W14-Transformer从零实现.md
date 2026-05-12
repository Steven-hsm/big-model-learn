# 第14周 - Transformer从零实现

> 本周将上周学到的Transformer理论逐步转化为完整代码。从Token Embedding开始，一个组件一个组件地实现，最终组装成完整的Transformer并训练一个复制任务来验证正确性。

---

## 一、本周目标

1. 实现Token Embedding和Positional Encoding模块
2. 从零实现Scaled Dot-Product Attention和Multi-Head Attention
3. 实现FFN、LayerNorm和残差连接的封装
4. 分别实现Encoder Layer/Encoder和Decoder Layer/Decoder
5. 组装完整Transformer并训练复制任务验证正确性

---

## 二、时间安排

### 工作日（每晚2小时）

| 日期 | 主题 | 时间分配 |
|------|------|----------|
| Day 1 (周一) | Token Embedding + Positional Encoding | 理论30min + 实现90min |
| Day 2 (周二) | Attention实现 | 理论20min + 编码100min |
| Day 3 (周三) | FFN + LayerNorm + 残差 | 实现60min + 测试60min |
| Day 4 (周四) | Encoder Layer + Encoder | 实现80min + 测试40min |
| Day 5 (周五) | Decoder Layer + Decoder | 实现80min + 测试40min |

### 周末（6-8小时）

| 日期 | 主题 | 时间分配 |
|------|------|----------|
| Day 6 (周六) | 完整Transformer + 训练 | 组装60min + 训练代码120min + 调试120min |
| Day 7 (周日) | 代码重构与验证 | 重构60min + 与PyTorch对比60min + 文档60min |

---

## 三、详细学习内容

### Day 1: Token Embedding + Positional Encoding

#### 1. Token Embedding

Token Embedding将离散的token ID映射为连续的稠密向量：

```python
class TokenEmbedding(nn.Module):
    """将token索引转换为稠密向量"""
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        """
        x: (batch_size, seq_len) — token索引
        返回: (batch_size, seq_len, d_model)
        """
        # 乘以sqrt(d_model)进行缩放，原论文的做法
        # 原因：在加入位置编码前放大embedding值，
        # 使embedding的尺度与位置编码匹配
        return self.embedding(x) * math.sqrt(self.d_model)
```

**为什么不用One-Hot**：
- One-Hot维度等于词表大小（如30000维），极度稀疏
- One-Hot任意两个词的距离相同，无法表达语义关系
- Embedding是稠密的（如512维），维度低、计算快
- Embedding中语义相近的词在向量空间中距离也近（通过学习得到）

#### 2. Positional Encoding

```python
class PositionalEncoding(nn.Module):
    """正弦位置编码"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 预计算位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        # register_buffer: 保存为模型的一部分但不参与梯度更新
        self.register_buffer('pe', pe)

    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
```

#### 3. Input Embedding组合

```python
class TransformerEmbedding(nn.Module):
    """Token Embedding + Positional Encoding"""
    def __init__(self, vocab_size, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)

    def forward(self, x):
        """
        x: (batch, seq_len) — token索引
        返回: (batch, seq_len, d_model) — 带位置信息的embedding
        """
        token_emb = self.token_embedding(x)       # (batch, seq_len, d_model)
        return self.positional_encoding(token_emb)  # 加上位置编码
```

---

### Day 2: Scaled Dot-Product Attention + Multi-Head Attention

#### 1. Scaled Dot-Product Attention

```python
def scaled_dot_product_attention(Q, K, V, mask=None, dropout=None):
    """
    Q: (batch, n_heads, seq_len_q, d_k)
    K: (batch, n_heads, seq_len_k, d_k)
    V: (batch, n_heads, seq_len_v, d_v)
    mask: (batch, 1, seq_len_q, seq_len_k) 或 (1, 1, seq_len_q, seq_len_k)

    返回: output (batch, n_heads, seq_len_q, d_v),
          attention_weights (batch, n_heads, seq_len_q, seq_len_k)
    """
    d_k = Q.size(-1)

    # Step 1: 计算注意力分数
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    # scores: (batch, n_heads, seq_len_q, seq_len_k)

    # Step 2: 应用mask
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # Step 3: Softmax归一化
    attention_weights = F.softmax(scores, dim=-1)

    if dropout is not None:
        attention_weights = dropout(attention_weights)

    # Step 4: 加权求和
    output = torch.matmul(attention_weights, V)
    # output: (batch, n_heads, seq_len_q, d_v)

    return output, attention_weights
```

#### 2. Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model必须能被n_heads整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Q, K, V的线性投影（所有头合并在一起）
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        # 输出投影
        self.W_O = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        """
        将最后一个维度拆分为n_heads个头
        (batch, seq_len, d_model) → (batch, n_heads, seq_len, d_k)
        """
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.n_heads, self.d_k)
        return x.transpose(1, 2)  # 交换seq_len和n_heads维度

    def merge_heads(self, x):
        """
        将多个头的输出合并
        (batch, n_heads, seq_len, d_k) → (batch, seq_len, d_model)
        """
        batch_size, _, seq_len, _ = x.size()
        x = x.transpose(1, 2).contiguous()  # (batch, seq_len, n_heads, d_k)
        return x.view(batch_size, seq_len, self.d_model)

    def forward(self, query, key, value, mask=None):
        """
        query: (batch, seq_len_q, d_model)
        key:   (batch, seq_len_k, d_model)
        value: (batch, seq_len_v, d_model)
        mask:  (batch, 1, seq_len_q, seq_len_k)
        """
        batch_size = query.size(0)

        # 线性投影
        Q = self.W_Q(query)  # (batch, seq_len_q, d_model)
        K = self.W_K(key)    # (batch, seq_len_k, d_model)
        V = self.W_V(value)  # (batch, seq_len_v, d_model)

        # 拆分为多头
        Q = self.split_heads(Q)  # (batch, n_heads, seq_len_q, d_k)
        K = self.split_heads(K)  # (batch, n_heads, seq_len_k, d_k)
        V = self.split_heads(V)  # (batch, n_heads, seq_len_v, d_k)

        # 计算注意力
        attn_output, attn_weights = scaled_dot_product_attention(
            Q, K, V, mask, self.dropout
        )

        # 合并多头
        attn_output = self.merge_heads(attn_output)  # (batch, seq_len_q, d_model)

        # 输出投影
        output = self.W_O(attn_output)

        return output, attn_weights
```

#### 3. causal_mask实现

```python
def create_causal_mask(seq_len):
    """创建Decoder的因果掩码"""
    # 下三角矩阵，位置i只能看到0~i
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    # shape: (1, 1, seq_len, seq_len)
    return mask

def create_padding_mask(seq, pad_idx=0):
    """创建padding掩码"""
    # seq: (batch, seq_len)
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    # shape: (batch, 1, 1, seq_len)
    return mask
```

---

### Day 3: FFN + LayerNorm + 残差

#### 1. Position-wise FFN

```python
class PositionWiseFFN(nn.Module):
    """位置级前馈网络"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        FFN(x) = ReLU(x·W1+b1)·W2+b2
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))
```

#### 2. LayerNorm

```python
class LayerNorm(nn.Module):
    """Layer Normalization"""
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
```

#### 3. 残差连接封装

```python
class SublayerConnection(nn.Module):
    """残差连接 + LayerNorm (Post-Norm)"""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        x: 输入
        sublayer: 子层函数（如attention或FFN）
        output = LayerNorm(x + Dropout(Sublayer(x)))
        """
        return self.norm(x + self.dropout(sublayer))
```

---

### Day 4: Encoder Layer + Encoder

#### 1. Encoder Layer

```python
class EncoderLayer(nn.Module):
    """Transformer Encoder的一层"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)

    def forward(self, x, src_mask=None):
        """
        x: (batch, seq_len, d_model)
        src_mask: padding mask
        """
        # 子层1: Self-Attention + 残差 + LayerNorm
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, src_mask)[0])

        # 子层2: FFN + 残差 + LayerNorm
        x = self.sublayer2(x, self.ffn)

        return x
```

#### 2. Encoder

```python
class Encoder(nn.Module):
    """Transformer Encoder: N个EncoderLayer堆叠"""
    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = LayerNorm(d_model)

    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)  # 最终归一化
```

---

### Day 5: Decoder Layer + Decoder

#### 1. Decoder Layer

```python
class DecoderLayer(nn.Module):
    """Transformer Decoder的一层"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        # 三个子层
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)
        self.sublayer3 = SublayerConnection(d_model, dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        """
        x: decoder输入 (batch, tgt_len, d_model)
        enc_output: encoder输出 (batch, src_len, d_model)
        src_mask: encoder的padding mask
        tgt_mask: decoder的因果mask + padding mask
        """
        # 子层1: Masked Self-Attention
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, tgt_mask)[0])

        # 子层2: Cross-Attention (Q=decoder, K=V=encoder)
        x = self.sublayer2(x, lambda x: self.cross_attn(x, enc_output, enc_output, src_mask)[0])

        # 子层3: FFN
        x = self.sublayer3(x, self.ffn)

        return x
```

#### 2. Decoder

```python
class Decoder(nn.Module):
    """Transformer Decoder: N个DecoderLayer堆叠"""
    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = LayerNorm(d_model)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return self.norm(x)
```

---

### Day 6: 完整Transformer + 训练

#### 1. 完整Transformer组装

```python
class Transformer(nn.Module):
    """完整的Transformer模型"""
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512,
                 n_heads=8, n_layers=6, d_ff=2048, dropout=0.1,
                 max_len=5000):
        super().__init__()

        # Embedding
        self.src_embedding = TransformerEmbedding(src_vocab_size, d_model, max_len, dropout)
        self.tgt_embedding = TransformerEmbedding(tgt_vocab_size, d_model, max_len, dropout)

        # Encoder & Decoder
        self.encoder = Encoder(n_layers, d_model, n_heads, d_ff, dropout)
        self.decoder = Decoder(n_layers, d_model, n_heads, d_ff, dropout)

        # 输出层
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        # 参数初始化
        self._init_weights()

    def _init_weights(self):
        """Xavier初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        src: (batch, src_len) — 源序列token索引
        tgt: (batch, tgt_len) — 目标序列token索引
        """
        # Encoder
        src_embedded = self.src_embedding(src)
        enc_output = self.encoder(src_embedded, src_mask)

        # Decoder
        tgt_embedded = self.tgt_embedding(tgt)
        dec_output = self.decoder(tgt_embedded, enc_output, src_mask, tgt_mask)

        # 输出投影
        logits = self.generator(dec_output)  # (batch, tgt_len, tgt_vocab_size)

        return logits

    def encode(self, src, src_mask=None):
        """仅编码"""
        return self.encoder(self.src_embedding(src), src_mask)

    def decode(self, tgt, enc_output, src_mask=None, tgt_mask=None):
        """仅解码"""
        return self.decoder(self.tgt_embedding(tgt), enc_output, src_mask, tgt_mask)
```

#### 2. 训练复制任务

复制任务是验证Transformer实现的经典测试——输入一个随机序列，模型需要输出相同的序列。

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class CopyTaskDataset(Dataset):
    """复制任务数据集：输入=目标"""
    def __init__(self, vocab_size, seq_len, n_samples=1000):
        self.data = torch.randint(1, vocab_size, (n_samples, seq_len))
        # 添加EOS token
        self.eos = vocab_size  # EOS的索引 = vocab_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src = self.data[idx]
        # 目标 = 源序列 + EOS
        tgt_input = torch.cat([src, torch.tensor([self.eos])])
        tgt_output = torch.cat([src, torch.tensor([self.eos])])
        return src, tgt_input, tgt_output

def train_copy_task():
    # 超参数
    vocab_size = 10       # 0=PAD, 1-9=有效token, 10=EOS
    d_model = 64
    n_heads = 4
    n_layers = 2
    d_ff = 256
    seq_len = 10
    batch_size = 32
    epochs = 50

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据
    dataset = CopyTaskDataset(vocab_size, seq_len, n_samples=2000)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 模型（tgt_vocab_size = vocab_size + 1 因为有EOS）
    model = Transformer(
        src_vocab_size=vocab_size + 1,
        tgt_vocab_size=vocab_size + 1,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=0.1
    ).to(device)

    # 标签平滑的交叉熵损失
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略PAD
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)

    # 学习率调度
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 训练
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for src, tgt_input, tgt_output in dataloader:
            src, tgt_input, tgt_output = src.to(device), tgt_input.to(device), tgt_output.to(device)

            # 创建masks
            src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, src_len)
            tgt_mask = create_causal_mask(tgt_input.size(1)).to(device)

            optimizer.zero_grad()
            logits = model(src, tgt_input, src_mask, tgt_mask)

            # 计算loss（忽略第一个位置）
            loss = criterion(
                logits[:, :-1].contiguous().view(-1, vocab_size + 1),
                tgt_output[:, 1:].contiguous().view(-1)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            # 计算准确率
            _, predicted = logits.max(-1)
            mask = tgt_output[:, 1:] != 0
            correct += (predicted[:, :-1] == tgt_output[:, 1:]).masked_select(mask).sum().item()
            total += mask.sum().item()

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total if total > 0 else 0

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.1f}%")

    return model

# 运行训练
model = train_copy_task()
```

---

### Day 7: 代码重构与对比验证

#### 1. 与PyTorch nn.Transformer对比

```python
def compare_with_pytorch():
    """对比自实现和PyTorch内置Transformer"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 超参数
    src_vocab_size = 11
    tgt_vocab_size = 11
    d_model = 64
    n_heads = 4
    n_layers = 2
    d_ff = 256

    # 自实现的模型
    custom_model = Transformer(
        src_vocab_size, tgt_vocab_size,
        d_model, n_heads, n_layers, d_ff
    ).to(device)

    # PyTorch内置模型
    pytorch_model = nn.Transformer(
        d_model=d_model,
        nhead=n_heads,
        num_encoder_layers=n_layers,
        num_decoder_layers=n_layers,
        dim_feedforward=d_ff,
        batch_first=True
    ).to(device)

    # 参数量对比
    custom_params = sum(p.numel() for p in custom_model.parameters())
    pytorch_params = sum(p.numel() for p in pytorch_model.parameters())
    print(f"自实现参数量: {custom_params:,}")
    print(f"PyTorch参数量: {pytorch_params:,}")

    # 功能验证：两者都能在复制任务上收敛
    # 如果自实现的模型能达到接近100%的准确率，说明实现正确
```

#### 2. 代码重构清单

```
transformer/
├── __init__.py
├── embeddings.py        # TokenEmbedding, PositionalEncoding, TransformerEmbedding
├── attention.py         # scaled_dot_product_attention, MultiHeadAttention
├── layers.py            # LayerNorm, PositionWiseFFN, SublayerConnection
├── encoder.py           # EncoderLayer, Encoder
├── decoder.py           # DecoderLayer, Decoder
├── transformer.py       # Transformer主模型
├── masks.py             # create_causal_mask, create_padding_mask
└── utils.py             # 初始化工具等
```

---

## 四、代码练习

### Day 1-5：逐步实现

每天的练习就是实现对应模块，并用单元测试验证：

```python
"""测试代码模板"""
import torch

def test_token_embedding():
    emb = TokenEmbedding(vocab_size=100, d_model=512)
    x = torch.randint(0, 100, (2, 10))  # (batch=2, seq_len=10)
    out = emb(x)
    assert out.shape == (2, 10, 512), f"Expected (2,10,512), got {out.shape}"
    print("TokenEmbedding: PASS")

def test_positional_encoding():
    pe = PositionalEncoding(d_model=512, max_len=100)
    x = torch.randn(2, 10, 512)
    out = pe(x)
    assert out.shape == (2, 10, 512)
    print("PositionalEncoding: PASS")

def test_multihead_attention():
    mha = MultiHeadAttention(d_model=512, n_heads=8)
    x = torch.randn(2, 10, 512)
    out, weights = mha(x, x, x)
    assert out.shape == (2, 10, 512)
    assert weights.shape == (2, 8, 10, 10)
    print("MultiHeadAttention: PASS")

def test_encoder_layer():
    layer = EncoderLayer(d_model=512, n_heads=8, d_ff=2048)
    x = torch.randn(2, 10, 512)
    out = layer(x)
    assert out.shape == (2, 10, 512)
    print("EncoderLayer: PASS")

def test_decoder_layer():
    layer = DecoderLayer(d_model=512, n_heads=8, d_ff=2048)
    x = torch.randn(2, 10, 512)
    enc_out = torch.randn(2, 15, 512)
    tgt_mask = create_causal_mask(10)
    out = layer(x, enc_out, tgt_mask=tgt_mask)
    assert out.shape == (2, 10, 512)
    print("DecoderLayer: PASS")

def test_transformer():
    model = Transformer(src_vocab_size=100, tgt_vocab_size=100,
                        d_model=512, n_heads=8, n_layers=6, d_ff=2048)
    src = torch.randint(0, 100, (2, 15))
    tgt = torch.randint(0, 100, (2, 10))
    out = model(src, tgt)
    assert out.shape == (2, 10, 100)
    print("Transformer: PASS")

# 运行所有测试
test_token_embedding()
test_positional_encoding()
test_multihead_attention()
test_encoder_layer()
test_decoder_layer()
test_transformer()
```

### Day 6：训练复制任务

（见Day 6详细内容中的完整训练代码）

### Day 7：完善代码和文档

- 代码重构为模块化结构
- 添加完整的docstring注释
- 编写使用示例
- 与PyTorch nn.Transformer对比

---

## 五、本周产出

### 周末交付物

1. **完整Transformer实现**：模块化的Transformer代码，包含所有组件 `transformer/`
2. **复制任务训练**：能成功训练复制任务的完整代码 `train_copy_task.py`
3. **单元测试**：每个组件的shape和功能测试 `tests/`
4. **对比验证**：与PyTorch nn.Transformer的参数量和功能对比

---

## 六、自测题

### 题1：Token Embedding的作用？为什么不用One-Hot？

<details>
<summary>参考答案</summary>

Token Embedding将离散的token ID映射为连续的稠密向量（如从整数5映射为512维浮点向量）。

不用One-Hot的原因：
1. **维度问题**：One-Hot维度=词表大小（如30000），每个向量30000维但只有1个非零元素。Embedding只需512维
2. **计算效率**：One-Hot的矩阵乘法等价于查表，但需要存储巨大的稀疏矩阵。Embedding直接查表
3. **语义表达**：One-Hot中任意两个词的余弦相似度都是0（正交）。Embedding中语义相近的词距离更近
4. **参数效率**：Embedding参数量=vocab_size×d_model，远小于One-Hot+线性层的参数量
</details>

### 题2：实现Multi-Head Attention时split_heads的维度变换？

<details>
<summary>参考答案</summary>

维度变换过程：
```
输入: (batch, seq_len, d_model)
↓ view
(batch, seq_len, n_heads, d_k)    其中d_k = d_model / n_heads
↓ transpose(1, 2)
(batch, n_heads, seq_len, d_k)
```

关键理解：
- view操作将d_model维度拆分为n_heads和d_k两个维度
- transpose将n_heads维度提到前面，方便后续批量计算
- 每个头独立计算注意力，但所有头的计算通过矩阵运算并行完成

合并时逆向操作：
```
(batch, n_heads, seq_len, d_k)
↓ transpose(1, 2)
(batch, seq_len, n_heads, d_k)
↓ contiguous().view
(batch, seq_len, d_model)
```
</details>

### 题3：残差连接在Transformer中的作用？

<details>
<summary>参考答案</summary>

1. **梯度流动**：反向传播时，梯度可以通过残差路径直接传递（+1恒等映射），不受子层（Attention/FFN）梯度的影响。在深层Transformer中尤其重要

2. **信息保留**：每一层的输出 = 原始输入 + 子层的增量信息。如果子层学不到有用的东西，至少不丢失信息

3. **训练稳定性**：防止深层网络中信息逐步退化。残差连接让每层只需要学习"差值"，比学习完整映射更容易

4. **恒等映射**：如果某层不需要，可以将权重学为0，等效于跳过该层
</details>

### 题4：Decoder的Mask有几种？分别是什么？

<details>
<summary>参考答案</summary>

Decoder中有两种mask：

1. **Causal Mask（因果掩码）**：
   - 用于Masked Self-Attention
   - 下三角矩阵，位置i只能看到位置0~i
   - 防止在生成时"偷看"未来信息
   - 实现：`mask = torch.tril(torch.ones(seq_len, seq_len))`

2. **Padding Mask（填充掩码）**：
   - 用于Self-Attention和Cross-Attention
   - 标记padding位置（值为0），使注意力不关注padding
   - 实现：`mask = (input != pad_idx)`
   - 在Cross-Attention中，padding mask基于源序列

实际使用时，两种mask可以通过逻辑与（AND）合并：
```python
combined_mask = causal_mask & padding_mask
```
</details>

### 题5：如何验证你的Transformer实现是正确的？

<details>
<summary>参考答案</summary>

验证方法：

1. **Shape测试**：确保每个组件的输入输出shape正确
2. **复制任务**：训练一个复制任务（输入=输出），如果模型能达到接近100%准确率，说明前向传播和反向传播都正确
3. **与PyTorch内置对比**：
   - 参数量对比（应该相近）
   - 固定随机种子，对比输出值
4. **梯度检查**：用数值梯度验证解析梯度（参考第9周内容）
5. **消融实验**：去掉某些组件（如mask、位置编码），观察性能下降，验证组件的作用
</details>

---

## 七、Java开发者提示

### 1. 从零实现 vs 使用框架

这周的工作类似于：
- Spring Boot的源码阅读 —— 理解框架内部机制
- 自己写一个简易版Spring —— 加深理解

```java
// 用Java的思维理解Transformer的组装
// 就像组装一个复杂的应用：

Transformer transformer = new Transformer.Builder()
    .srcEmbedding(new TokenEmbedding(vocabSize, dModel))
    .positionalEncoding(new SinusoidalPE(dModel))
    .encoder(new Encoder(nLayers, dModel, nHeads, dFF))
    .decoder(new Decoder(nLayers, dModel, nHeads, dFF))
    .generator(new Linear(dModel, tgtVocabSize))
    .build();

// 前向传播 = pipeline处理
Output output = transformer.forward(src, tgt);
// 等价于：
// srcEmbed → posEncode → encoder → [cross-attn with] → decoder → generator
```

### 2. 模块化设计 vs 分层架构

Transformer的模块化设计和Java的分层架构完全对应：
```
Controller (入口)          → Transformer (主模型)
  Service (业务逻辑)        → Encoder/Decoder (核心逻辑)
    DAO (数据访问)           → Embedding (数据编码)
      Util (工具类)          → LayerNorm, FFN (基础组件)
```

每个组件只做一件事，通过接口（forward方法）连接。这种设计理念在Java中是基本功。

### 3. 测试驱动开发

这周的开发流程非常适合TDD（Test-Driven Development）：
1. 先写测试（验证shape和功能）
2. 再实现功能
3. 确保测试通过
4. 重构

```java
// Java中的类比
@Test
void testEncoderLayer() {
    Tensor input = Tensor.randn(2, 10, 512);
    Tensor output = encoderLayer.forward(input);
    assertEquals(new int[]{2, 10, 512}, output.shape());
}
```

### 4. 参数初始化 vs 构造函数

Transformer的权重初始化类似Java中构造函数的初始化：
```java
// Java
public class UserService {
    private final UserRepository repo;  // 依赖注入
    private final CacheManager cache;   // 初始化

    public UserService(UserRepository repo) {
        this.repo = repo;
        this.cache = new CacheManager();  // 默认初始化
    }
}

// Transformer
class Transformer(nn.Module):
    def __init__(self):
        self.encoder = Encoder(...)     # 依赖组件
        self.decoder = Decoder(...)     # 依赖组件
        self._init_weights()            # 参数初始化（类似构造函数中的初始化）
```

### 5. 复制任务 vs 单元测试的Hello World

复制任务是Transformer的"Hello World"，就像Java程序员写的第一个测试：
```java
@Test
void testAddition() {
    assertEquals(4, calculator.add(2, 2));  // 最简单的验证
}
```

复制任务也是最简单的验证——模型能不能把输入原样输出。如果连这个都做不到，说明实现有问题。
