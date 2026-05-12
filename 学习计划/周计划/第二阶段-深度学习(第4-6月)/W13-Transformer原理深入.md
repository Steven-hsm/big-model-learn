# 第13周 - Transformer原理深入

> 本周深入学习Transformer的每一个组件。从注意力机制的直觉到数学公式，从位置编码到Encoder-Decoder架构，逐层拆解"Attention Is All You Need"论文。

---

## 一、本周目标

1. 深入理解注意力机制的数学原理（加性注意力、乘性注意力、Scaled Dot-Product）
2. 掌握Multi-Head Attention的设计动机和实现细节
3. 理解不同位置编码方案（正弦编码、RoPE、ALiBi）
4. 彻底理解Encoder-Decoder架构，包括Mask机制和Cross-Attention
5. 完成Transformer原论文的精读，理解其创新点和局限性

---

## 二、时间安排

### 工作日（每晚2小时）

| 日期 | 主题 | 时间分配 |
|------|------|----------|
| Day 1 (周一) | Attention机制 | 理论50min + 可视化70min |
| Day 2 (周二) | Scaled Dot-Product Attention | 理论40min + 手动计算80min |
| Day 3 (周三) | Multi-Head Attention | 理论50min + PyTorch实现70min |
| Day 4 (周四) | 位置编码 | 理论60min + 代码实现60min |
| Day 5 (周五) | FFN与Layer Norm | 理论40min + 代码实现80min |

### 周末（6-8小时）

| 日期 | 主题 | 时间分配 |
|------|------|----------|
| Day 6 (周六) | Encoder-Decoder架构 | 理论60min + 代码实现120min |
| Day 7 (周日) | 论文精读 | 论文阅读120min + 笔记整理120min |

---

## 三、详细学习内容

### Day 1: Attention机制

#### 1. 注意力的直觉

人类视觉系统在观察场景时，并不会同时处理所有信息，而是**聚焦于关键区域**。例如看一张照片，你会自动关注人脸、文字等重要区域，忽略背景。

注意力机制就是让模型学会"在哪里聚焦"——对输入的不同部分赋予不同的重要性权重。

#### 2. Query-Key-Value类比

注意力机制借用了信息检索的概念：
- **Query (Q)**：当前想查什么（我在找什么？）
- **Key (K)**：每条记录的标签/索引（每个信息有什么特征？）
- **Value (V)**：每条记录的实际内容（具体的信息内容）

类比图书馆检索：
```
Query = "我想找关于深度学习的书"
Key   = 每本书的标题和关键词
Value = 每本书的内容

计算 Query 和每个 Key 的相似度 → 得到注意力权重
根据权重加权求和 Value → 得到最终结果
```

#### 3. 加性注意力 (Bahdanau Attention, 2014)

```python
score(h_t, h_s) = v^T · tanh(W_1 · h_t + W_2 · h_s)
```

使用一个小型神经网络（MLP）计算Query和Key的相关性：
- `h_t`：当前decoder隐藏状态（Query）
- `h_s`：encoder的每个隐藏状态（Key）
- `v, W_1, W_2`：可学习参数

特点：灵活但计算量大（需要MLP前向传播）。

#### 4. 乘性注意力 (Luong Attention, 2015)

```python
score(h_t, h_s) = h_t^T · W · h_s
```

直接用点积计算Query和Key的相关性。计算更快，因为矩阵乘法在GPU上高度优化。

#### 5. 注意力权重可视化

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def attention_heatmap(Q, K, V):
    """可视化注意力权重"""
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)  # 缩放点积
    weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)  # softmax
    output = weights @ V

    # 绘制热力图
    plt.figure(figsize=(8, 6))
    sns.heatmap(weights, annot=True, fmt='.2f', cmap='YlOrRd')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.title('Attention Weights')
    plt.savefig('attention_heatmap.png', dpi=150)
    plt.show()

    return output, weights

# 示例
np.random.seed(42)
Q = np.random.randn(4, 8)  # 4个query，维度8
K = np.random.randn(6, 8)  # 6个key
V = np.random.randn(6, 8)  # 6个value
output, weights = attention_heatmap(Q, K, V)
print(f"Output shape: {output.shape}")  # (4, 8)
print(f"Weights shape: {weights.shape}")  # (4, 6)
```

---

### Day 2: Scaled Dot-Product Attention

#### 1. Q/K/V矩阵的来源

输入序列 `X` 通过三个不同的线性变换得到Q、K、V：

```
Q = X · W_Q    W_Q shape: (d_model, d_k)
K = X · W_K    W_K shape: (d_model, d_k)
V = X · W_V    W_V shape: (d_model, d_v)
```

其中：
- `X` 是输入矩阵，shape `(seq_len, d_model)` 或 `(batch, seq_len, d_model)`
- `W_Q, W_K, W_V` 是可学习的投影矩阵
- `d_model` 是模型维度（原论文为512）
- `d_k = d_v = d_model / h`（h是注意力头数）

#### 2. 完整计算公式

```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

分步计算：
```
步骤1: scores = Q · K^T              # (seq_len, seq_len) 点积
步骤2: scaled_scores = scores / √d_k # 缩放
步骤3: weights = softmax(scaled_scores) # (seq_len, seq_len) 归一化
步骤4: output = weights · V           # (seq_len, d_v) 加权求和
```

#### 3. 为什么要除以 √d_k

当 `d_k` 较大时，Q和K的点积值会变得很大（假设每个元素独立，点积的方差正比于d_k）。大的点积值经过softmax后：

```python
# softmax对大值的敏感性
x = np.array([1.0, 2.0, 3.0])
print(softmax(x))  # [0.09, 0.24, 0.67]  分布相对均匀

x_large = np.array([10.0, 20.0, 30.0])
print(softmax(x_large))  # [0.00, 0.00, 1.00]  几乎是one-hot
```

当输入值很大时，softmax输出接近one-hot，梯度接近0，导致训练困难。除以 `√d_k` 将点积值缩放到合理范围，使softmax的梯度保持有效。

数学推导：
- 假设Q和K的元素独立且均值为0、方差为1
- 点积 `q·k = Σ q_i·k_i` 的方差为 `d_k`
- 除以 `√d_k` 后方差变为1

#### 4. 手动计算3x4的Q/K/V

```python
import numpy as np

np.random.seed(0)

# 设置参数
seq_len = 3   # 序列长度3
d_model = 8   # 模型维度
d_k = 4       # key/query维度
d_v = 4       # value维度

# 输入
X = np.random.randn(seq_len, d_model)
print(f"X shape: {X.shape}")  # (3, 8)

# 投影矩阵
W_Q = np.random.randn(d_model, d_k)  # (8, 4)
W_K = np.random.randn(d_model, d_k)  # (8, 4)
W_V = np.random.randn(d_model, d_v)  # (8, 4)

# 得到Q, K, V
Q = X @ W_Q  # (3, 4)
K = X @ W_K  # (3, 4)
V = X @ W_V  # (3, 4)

print(f"Q: {Q.shape}\n{Q}")
print(f"K: {K.shape}\n{K}")
print(f"V: {V.shape}\n{V}")

# Step 1: 点积
scores = Q @ K.T  # (3, 3)
print(f"\nScores:\n{scores}")

# Step 2: 缩放
scaled_scores = scores / np.sqrt(d_k)
print(f"\nScaled scores:\n{scaled_scores}")

# Step 3: Softmax
def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

weights = softmax(scaled_scores)
print(f"\nAttention weights:\n{weights}")

# Step 4: 加权求和
output = weights @ V
print(f"\nOutput: {output.shape}\n{output}")
```

---

### Day 3: Multi-Head Attention

#### 1. 核心思想

与其使用一个大的注意力函数，不如将Q、K、V投影到h个不同的子空间，在每个子空间独立计算注意力，最后拼接结果。

类比：让h个"专家"从不同角度分析信息，每个专家关注不同的模式。

#### 2. 数学公式

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O

其中每个head:
head_i = Attention(Q·W_Q_i, K·W_K_i, V·W_V_i)
```

参数维度：
```
W_Q_i: (d_model, d_k), d_k = d_model / h
W_K_i: (d_model, d_k)
W_V_i: (d_model, d_v), d_v = d_model / h
W_O:   (h·d_v, d_model) = (d_model, d_model)
```

原论文参数：`d_model=512, h=8, d_k=d_v=64`

#### 3. 为什么多头比单头好

单头注意力只能学一种注意力模式，多头可以同时关注：
- 头1：关注语法关系（主谓一致）
- 头2：关注语义相似性
- 头3：关注位置关系（相邻词）
- ...

实验证据：可视化不同注意力头的权重矩阵，发现它们确实学到了不同的模式。

#### 4. PyTorch实现（不使用nn.MultiheadAttention）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # 线性投影（合并所有头的权重）
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Q, K, V: (batch, n_heads, seq_len, d_k)
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, V)
        return output, weights

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # 线性投影并split heads
        # (batch, seq_len, d_model) → (batch, seq_len, n_heads, d_k) → (batch, n_heads, seq_len, d_k)
        Q = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 注意力计算
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # 合并heads
        # (batch, n_heads, seq_len, d_k) → (batch, seq_len, n_heads, d_k) → (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 输出投影
        output = self.W_O(attn_output)
        return output, attn_weights

# 使用示例
mha = MultiHeadAttention(d_model=512, n_heads=8)
x = torch.randn(2, 10, 512)  # (batch=2, seq_len=10, d_model=512)
output, weights = mha(x, x, x)  # Self-Attention: Q=K=V=x
print(f"Output: {output.shape}")   # (2, 10, 512)
print(f"Weights: {weights.shape}") # (2, 8, 10, 10)
```

---

### Day 4: 位置编码

#### 1. 为什么需要位置编码

Transformer的核心操作是注意力：
```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

这个操作是**排列不变的（permutation invariant）**——打乱输入顺序，输出只是对应打乱，不会改变内容。也就是说，Transformer本身不知道"我爱你"和"你爱我"的区别。

对比：RNN按顺序处理，天然包含位置信息。CNN通过卷积核的局部感受野隐含位置信息。

解决方案：在输入Embedding中加入位置信息。

#### 2. 正弦位置编码 (Sinusoidal Positional Encoding)

原论文的方法：
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

其中 `pos` 是位置索引，`i` 是维度索引。

特点：
- 每个维度对应一个不同频率的正弦/余弦波
- 低维度频率高（快速变化），高维度频率低（缓慢变化）
- 理论上可以外推到训练时没见过的长度（通过线性变换表达相对位置）

```python
import torch
import math

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)  # 不参与梯度更新

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]
```

#### 3. RoPE (Rotary Position Embedding)

RoPE是目前LLM（LLaMA、GPT-NeoX等）最常用的位置编码。

核心思想：**通过旋转矩阵在注意力计算中编码相对位置**。

```
将位置m的向量 [x1, x2] 旋转角度 mθ:
[x1'] = [cos(mθ)  -sin(mθ)] [x1]
[x2']   [sin(mθ)   cos(mθ)] [x2]
```

优势：
- 在注意力计算中自然产生相对位置信息
- 不需要额外的可学习参数
- 外推性更好

```python
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        self.max_len = max_len

    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        seq_len = x.size(1)
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, d_model)
        return emb.cos(), emb.sin()

def apply_rotary_emb(x, cos, sin):
    """应用旋转位置编码"""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    # 旋转
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin
```

#### 4. ALiBi (Attention with Linear Biases)

ALiBi不在Embedding中加位置信息，而是在注意力分数上加线性偏置：
```
attention_score(i, j) = q_i · k_j - m · |i - j|
```

其中 `m` 是每个头不同的斜率（如 `2^(-8/n_heads)` 到 `2^(-8)`）。

特点：
- 不需要位置编码参数
- 外推性极好（训练短序列，推理长序列）
- 计算简单高效

#### 5. 位置编码方案对比

| 方案 | 类型 | 外推性 | 参数量 | 使用模型 |
|------|------|--------|--------|---------|
| 正弦编码 | 绝对位置 | 一般 | 0 | 原始Transformer |
| 可学习位置编码 | 绝对位置 | 差 | max_len×d_model | BERT, GPT-2 |
| RoPE | 相对位置 | 好 | 0 | LLaMA, PaLM |
| ALiBi | 相对位置 | 极好 | 0 | BLOOM, MPT |

---

### Day 5: FFN与Layer Norm

#### 1. Feed-Forward Network (FFN)

每个Transformer层包含一个位置级FFN：
```
FFN(x) = max(0, x·W_1 + b_1)·W_2 + b_2
```

即两层线性变换中间加ReLU激活：
```
Linear1: (d_model, d_ff)  升维（通常d_ff = 4 × d_model）
ReLU
Linear2: (d_ff, d_model)  降维回来
```

```python
class PositionWiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))
```

为什么先升维再降维：增加非线性表达能力。中间层的维度越大，模型可以学习越复杂的变换。

现代变体：
- **GLU变体**（LLaMA使用）：`FFN(x) = (xW_1 ⊙ σ(xV))W_2`，用门控替代ReLU
- **SwiGLU**：`FFN(x) = (xW_1 ⊙ Swish(xV))W_2`

#### 2. Layer Normalization

```
LayerNorm(x) = (x - μ) / σ · γ + β
```

其中 μ 和 σ 在**特征维度**上计算（不是batch维度）：

```python
class LayerNorm(nn.Module):
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

LayerNorm vs BatchNorm：
```
输入: (batch, seq_len, d_model)

BatchNorm: 对每个特征，在batch维度上归一化
  → 统计量: 沿batch计算均值和方差
  → 问题: batch_size变化时行为不同；序列长度变化时也不稳定

LayerNorm: 对每个样本，在特征维度上归一化
  → 统计量: 沿d_model计算均值和方差
  → 优势: 不依赖batch_size；适合变长序列
```

#### 3. 残差连接

```
output = LayerNorm(x + Sublayer(x))
```

残差连接的作用：
1. **梯度流动**：提供梯度的"高速公路"，缓解梯度消失
2. **信息保留**：至少保留原始信息，只需要学习增量
3. **训练稳定**：每一层的输出变化不会太大

#### 4. Pre-Norm vs Post-Norm

```
Post-Norm (原论文): output = LayerNorm(x + Sublayer(x))
Pre-Norm (现代做法): output = x + Sublayer(LayerNorm(x))
```

```python
class PostNormEncoderLayer(nn.Module):
    """Post-Norm: 先Sublayer后归一化"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Post-Norm: Sublayer → Add → Norm
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, mask)[0]))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x

class PreNormEncoderLayer(nn.Module):
    """Pre-Norm: 先归一化后Sublayer"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-Norm: Norm → Sublayer → Add
        x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask)[0])
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x
```

Pre-Norm的优势：
- 训练更稳定（不需要warmup也能收敛）
- 主路梯度直接传递，不受归一化影响
- 几乎所有现代Transformer（GPT、LLaMA）都用Pre-Norm

---

### Day 6: Encoder-Decoder架构

#### 1. Encoder

原始Transformer的Encoder由6个相同的层组成，每层包含两个子层：

```
Encoder Layer:
  1. Multi-Head Self-Attention (全局自注意力)
  2. Position-wise FFN
  每个子层都有残差连接和LayerNorm
```

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # Self-Attention + 残差 + LayerNorm
        attn_out, _ = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout1(attn_out))
        # FFN + 残差 + LayerNorm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x

class Encoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)  # 最终归一化

    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)
```

#### 2. Decoder

Decoder的每层包含三个子层：

```
Decoder Layer:
  1. Masked Multi-Head Self-Attention (因果注意力，不能看到未来)
  2. Multi-Head Cross-Attention (Q来自Decoder, K/V来自Encoder)
  3. Position-wise FFN
  每个子层都有残差连接和LayerNorm
```

#### 3. Mask机制

**Causal Mask（因果掩码）**：防止Decoder在生成第t个位置时看到第t+1及之后的信息。

```python
# 创建因果mask（上三角矩阵）
def create_causal_mask(seq_len):
    """创建因果mask，位置i只能看到位置0到i"""
    mask = torch.tril(torch.ones(seq_len, seq_len))  # 下三角矩阵
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

# 示例：seq_len=5
# tensor([[[[1, 0, 0, 0, 0],
#           [1, 1, 0, 0, 0],
#           [1, 1, 1, 0, 0],
#           [1, 1, 1, 1, 0],
#           [1, 1, 1, 1, 1]]]])
```

在注意力计算中：
```python
scores = scores.masked_fill(causal_mask == 0, -1e9)
# 将mask为0的位置（未来位置）的分数设为-1e9
# softmax后这些位置的权重接近0，相当于"看不到"
```

#### 4. Cross-Attention

Cross-Attention中，Q来自Decoder，K和V来自Encoder：
```
Q = Decoder输出 · W_Q
K = Encoder输出 · W_K
V = Encoder输出 · W_V
```

这让Decoder的每个位置都能"看到"Encoder的所有输出，关注源序列的不同部分。

```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        # Self-Attention (带因果mask)
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        # Cross-Attention (Q=decoder, K/V=encoder)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # 1. Masked Self-Attention
        attn_out, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_out))

        # 2. Cross-Attention (Q=decoder, K=V=encoder)
        cross_out, _ = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout2(cross_out))

        # 3. FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_out))
        return x
```

#### 5. 三种架构范式

| 类型 | 模型 | 特点 |
|------|------|------|
| Encoder-Only | BERT, ViT | 双向注意力，适合理解任务（分类、NER） |
| Decoder-Only | GPT, LLaMA | 因果注意力，适合生成任务（对话、写作） |
| Encoder-Decoder | T5, BART | 适合seq2seq任务（翻译、摘要） |

---

### Day 7: 论文精读

#### 精读指南：Attention Is All You Need (Vaswani et al., 2017)

**第1-2节：Introduction & Background**
- 之前的序列模型（RNN/LSTM）的局限：顺序计算无法并行、长程依赖困难
- 注意力机制已存在（Bahdanau 2014），但通常与RNN结合
- 本文的创新：完全用注意力替代循环和卷积

**第3节：Model Architecture**
- Encoder-Decoder架构，6层Encoder + 6层Decoder
- 每个位置的计算复杂度：
  - Self-Attention: O(n²·d) （n是序列长度，d是维度）
  - RNN: O(n·d²)
  - 当n < d时，Self-Attention更快

**第4节：训练细节**
- WMT 2014英德翻译：4.5M句对
- 8个P100 GPU训练
- Base模型：12h，Big模型：3.5天
- Adam优化器，β1=0.9, β2=0.98, ε=10⁻⁹
- 学习率调度：warmup + cosine decay
- Label smoothing: ε=0.1

**关键实验结果**
| 模型 | EN-DE BLEU | EN-FR BLEU |
|------|-----------|-----------|
| 之前最佳 | 26.10 | 40.27 |
| Transformer Big | **28.40** | **41.80** |

**思考题**
1. 为什么Self-Attention的复杂度是O(n²d)？
2. 为什么要用Label Smoothing？
3. 这篇论文发表时有什么被忽视但后来变得重要的点？

---

## 四、代码练习

### Day 2：手动推演Scaled Dot-Product Attention

（见Day 2详细内容中的完整手动计算代码）

### Day 3：实现Multi-Head Attention

```python
"""任务：从零实现Multi-Head Attention，不使用nn.MultiheadAttention"""
# 参考Day 3的完整实现
# 要求：
# 1. 实现split_heads和merge_heads
# 2. 支持mask参数
# 3. 返回注意力权重用于可视化
# 4. 与PyTorch的nn.MultiheadAttention对比验证
```

### Day 6：实现Encoder Layer和Decoder Layer

```python
"""任务：分别实现完整的Encoder Layer和Decoder Layer"""
# Encoder Layer: Self-Attention + FFN + 残差 + LayerNorm
# Decoder Layer: Masked Self-Attention + Cross-Attention + FFN + 残差 + LayerNorm
# 要求：支持Pre-Norm和Post-Norm两种模式

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, pre_norm=True):
        super().__init__()
        # ... 参考Day 6实现

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, pre_norm=True):
        super().__init__()
        # ... 参考Day 6实现
```

---

## 五、本周产出

### 周末交付物

1. **Attention手动计算**：完整的Scaled Dot-Product Attention计算过程 `attention_manual.py`
2. **Multi-Head Attention实现**：从零实现的MHA `multihead_attention.py`
3. **Encoder/Decoder Layer实现**：完整的Transformer层实现 `transformer_layers.py`
4. **论文精读笔记**：记录关键创新点、数学推导和思考

---

## 六、自测题

### 题1：Scaled Dot-Product Attention中为什么要除以√d_k？

<details>
<summary>参考答案</summary>

当d_k较大时，Q和K的点积值的方差正比于d_k（假设每个元素独立）。过大的点积值经过softmax后，输出分布会变得非常尖锐（接近one-hot），导致softmax的梯度几乎为0，训练无法有效进行。

除以√d_k将方差归一化为1，使softmax的输入保持在合理范围，确保梯度有效。

数学推导：假设q和k的元素~N(0,1)，则q·k的方差=d_k。除以√d_k后方差=1。
</details>

### 题2：Multi-Head Attention为什么比Single-Head好？

<details>
<summary>参考答案</summary>

1. **多模式学习**：每个头可以独立学习不同的注意力模式。实验可视化显示不同头关注不同类型的关系（如语法、语义、位置等）
2. **计算效率**：多头注意力的总计算量与单头相同（每个头的维度是d_model/h，h个头总共还是d_model）
3. **更强的表达能力**：多个子空间的组合比单个大空间的表达能力更强
4. **鲁棒性**：不依赖单一注意力模式，降低过拟合风险
</details>

### 题3：为什么Transformer需要位置编码？RNN不需要？

<details>
<summary>参考答案</summary>

Transformer的Self-Attention操作是排列不变的（permutation invariant）——打乱输入顺序，输出只是对应位置打乱。没有位置编码，Transformer无法区分"我爱你"和"你爱我"。

RNN不需要额外的位置编码，因为它按时间步顺序处理序列，隐藏状态天然包含了历史信息的位置顺序。每个时间步h_t都隐含了"这是第t个元素"的信息。

CNN也不需要显式位置编码，因为卷积核的局部感受野和滑动位置本身就编码了位置信息。
</details>

### 题4：Pre-Norm和Post-Norm的区别和各自优势？

<details>
<summary>参考答案</summary>

Post-Norm：`output = Norm(x + Sublayer(x))`
- 先计算子层，再加残差，最后归一化
- 原论文使用的方法
- 需要warmup才能稳定训练
- 可能性能上限更高（但训练难度更大）

Pre-Norm：`output = x + Sublayer(Norm(x))`
- 先归一化，再计算子层，最后加残差
- 现代模型（GPT、LLaMA）的做法
- 训练更稳定，不需要warmup
- 主路径上梯度直接传递（x → x + ... → x + ...），不经过归一化
- 训练更容易但理论上限可能略低
</details>

### 题5：Decoder的Mask机制是什么？为什么需要？

<details>
<summary>参考答案</summary>

Decoder使用因果掩码(Causal Mask)，确保位置i只能看到位置0到i-1的信息，不能看到未来位置。

实现方式：在注意力分数矩阵上叠加一个上三角mask，将未来位置的分数设为-∞（实际用-1e9），经过softmax后权重接近0。

为什么需要：
1. **训练时**：使用Teacher Forcing（输入完整序列），但必须防止"作弊"——不能让模型在预测第t个词时看到第t+1个词
2. **自回归生成**：推理时逐个生成token，每个token只能基于已生成的token。Mask确保训练和推理的行为一致
3. **信息泄露防护**：没有Mask就相当于开卷考试，模型无法学到真正的预测能力
</details>

---

## 七、Java开发者提示

### 1. 注意力机制 vs 数据库查询

| Attention概念 | Java/数据库类比 |
|-------------|---------------|
| Query (Q) | SQL查询条件 |
| Key (K) | 数据库索引 |
| Value (V) | 查询返回的数据 |
| 注意力权重 | 查询结果的相关度排序 |
| Softmax | 归一化为概率分布 |

```java
// 注意力机制 ≈ 带相关度排序的数据库查询
List<Result> search(String query, List<Record> database) {
    // query → Q
    // database的索引 → K
    // database的内容 → V
    List<Double> scores = database.stream()
        .map(record -> similarity(query, record.key))  // Q·K^T
        .collect(toList());
    List<Double> weights = softmax(scores);            // softmax
    return weightedSum(database, weights);             // weights · V
}
```

### 2. Multi-Head Attention vs 微服务

多头注意力类似于微服务架构中不同服务处理不同关注点：
- **头1**（语法分析）≈ 用户服务
- **头2**（语义分析）≈ 订单服务
- **头3**（位置关系）≈ 库存服务
- 最后的W_O投影 ≈ API网关聚合各服务结果

### 3. 位置编码 vs 时间戳

位置编码相当于给每条数据加时间戳：
```java
// 没有位置编码：无法区分顺序
records.sort(???);  // 不知道先后

// 有位置编码：清楚知道顺序
for (int i = 0; i < records.size(); i++) {
    records.get(i).setPosition(i);  // 加上位置编码
}
```

正弦位置编码的波状模式类似信号处理中的频域分析——低频捕获全局位置，高频捕获局部位置。

### 4. 残差连接 vs 中间件链

```java
// 残差连接类似Web中间件链
class MiddlewareChain {
    Response process(Request req) {
        Response original = req.toResponse();        // x
        Response processed = subLayer.process(req);  // Sublayer(x)
        return original.merge(processed);            // x + Sublayer(x)
    }
}
// 原始信息始终保留，不会被覆盖
```

### 5. FFN的升维降维 vs ETL中的数据转换

FFN先升维（d_model→4d_model）再降维（4d_model→d_model），类似ETL中先展开数据再聚合：
```java
// 展开阶段（升维）：将数据展开到更多列
List<ExpandedRecord> expanded = records.stream()
    .flatMap(r -> expandToColumns(r))  // 维度从d_model扩展到d_ff
    .collect(toList());

// 聚合阶段（降维）：将展开的数据压缩回原维度
List<Record> result = expanded.stream()
    .map(this::compress)  // 维度从d_ff压缩回d_model
    .collect(toList());
```
