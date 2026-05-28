"""
W17-D2 GPT架构详解 (GPT Architecture)
======================================
GPT架构详解(decoder-only), 自回归生成原理, KV Cache概念,
代码实现简化GPT(Embedding+TransformerBlocks+LMHead), 生成文本demo
"""

import sys
import math
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W17-D2 GPT架构详解 (GPT Architecture)")
print("=" * 60)

# ============================================================
# 1. GPT 架构详解
# ============================================================
print("\n--- 1. GPT 架构详解 (Decoder-Only) ---")
print("""
  GPT = Generative Pre-trained Transformer

  整体结构:
    输入 Token IDs
        |
    Token Embedding + Position Embedding
        |
    Transformer Block x N
      |-- Masked Multi-Head Self-Attention (因果掩码)
      |-- Add & LayerNorm
      |-- Feed-Forward Network
      |-- Add & LayerNorm
        |
    LayerNorm
        |
    Linear (隐藏维度 -> 词表大小)
        |
    Softmax -> 概率分布 -> 下一个Token

  关键特征:
    1) Decoder-Only: 只有Transformer解码器部分
    2) 因果掩码(Causal Mask): 只能看到当前位置之前的token
    3) 自回归生成: 逐个token生成, 每步依赖之前所有输出
""")

# ============================================================
# 2. 自回归生成原理
# ============================================================
print("\n--- 2. 自回归生成原理 ---")
print("""
  自回归 (Autoregressive) 生成:

    P(x_1, x_2, ..., x_n) = P(x_1) * P(x_2|x_1) * P(x_3|x_1,x_2) * ... * P(x_n|x_1,...,x_{n-1})

    即: 联合概率分解为条件概率的乘积

  生成过程:
    Step 1: 输入 [BOS]          -> 模型预测 "我"
    Step 2: 输入 [BOS] "我"     -> 模型预测 "喜欢"
    Step 3: 输入 [BOS] "我" "喜欢" -> 模型预测 "学习"
    ...

  因果掩码矩阵 (4x4 示例):
    [[1, 0, 0, 0],   <- 位置0只能看到位置0
     [1, 1, 0, 0],   <- 位置1能看到位置0,1
     [1, 1, 1, 0],   <- 位置2能看到位置0,1,2
     [1, 1, 1, 1]]   <- 位置3能看到位置0,1,2,3
""")

# 展示因果掩码
seq_len = 6
causal_mask = np.tril(np.ones((seq_len, seq_len)))
print("  因果掩码 (6x6):")
print(f"  {causal_mask.astype(int)}")

# ============================================================
# 3. KV Cache 概念
# ============================================================
print("\n--- 3. KV Cache 概念 ---")
print("""
  KV Cache 是大模型推理的核心优化:

  问题:
    自回归生成第 t 个token时, 需要计算 Q_t 与所有 K_0..t-1 的注意力
    如果每次都重新计算前面所有token的K和V, 计算量随长度线性增长

  解决方案:
    缓存已经计算过的 Key 和 Value 矩阵
    每生成一个新token, 只需计算新token的Q/K/V, 然后与缓存拼接

  对比:
    无KV Cache: 第t步需要计算 t 次 Attention
    有KV Cache: 第t步只需计算 1 次 Attention + 拼接

  内存占用:
    KV Cache大小 = 2 * num_layers * batch_size * seq_len * hidden_dim * dtype_size
    例: LLaMA-7B, seq_len=2048, fp16 -> 约 2 * 32 * 1 * 2048 * 4096 * 2 = ~1GB
""")

# KV Cache 内存估算可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax1 = axes[0]
seq_lengths = [512, 1024, 2048, 4096, 8192, 16384]
kv_cache_7b = [2 * 32 * sl * 4096 * 2 / (1024**3) for sl in seq_lengths]
kv_cache_13b = [2 * 40 * sl * 5120 * 2 / (1024**3) for sl in seq_lengths]
kv_cache_70b = [2 * 80 * sl * 8192 * 2 / (1024**3) for sl in seq_lengths]

ax1.plot(seq_lengths, kv_cache_7b, 'o-', label='LLaMA-7B', linewidth=2)
ax1.plot(seq_lengths, kv_cache_13b, 's-', label='LLaMA-13B', linewidth=2)
ax1.plot(seq_lengths, kv_cache_70b, '^-', label='LLaMA-70B', linewidth=2)
ax1.set_xlabel('序列长度 (tokens)', fontsize=12)
ax1.set_ylabel('KV Cache 大小 (GB, fp16)', fontsize=12)
ax1.set_title('KV Cache 内存占用', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 自回归生成演示
ax2 = axes[1]
tokens = ["[BOS]", "深", "度", "学", "习", "是"]
probs_history = {
    "[BOS]": {"深": 0.35, "机": 0.25, "人": 0.15, "大": 0.10, "自": 0.08, "其他": 0.07},
    "深":    {"度": 0.65, "入": 0.12, "刻": 0.08, "蓝": 0.05, "厚": 0.03, "其他": 0.07},
    "度":    {"学": 0.55, "神": 0.15, "理": 0.10, "思": 0.08, "情": 0.05, "其他": 0.07},
    "学":    {"习": 0.45, "习": 0.45, "者": 0.20, "术": 0.10, "校": 0.08, "其他": 0.17},
    "习":    {"是": 0.40, "的": 0.25, "和": 0.10, "方": 0.08, "很": 0.05, "其他": 0.12},
}

for i, (tok, probs) in enumerate(probs_history.items()):
    top_tokens = list(probs.keys())[:5]
    top_values = [probs[t] for t in top_tokens]
    colors = ['#e74c3c' if j == 0 else '#3498db' for j in range(len(top_tokens))]
    bars = ax2.barh(np.arange(len(top_tokens)) + i * 3.5, top_values, height=0.7,
                     color=colors, alpha=0.8)
    ax2.set_yticks(np.arange(len(top_tokens)) + i * 3.5)
    ax2.set_yticklabels(top_tokens, fontsize=9)
    ax2.text(-0.02, np.arange(len(top_tokens))[2] + i * 3.5, f'输入: {tok}',
             ha='right', va='center', fontsize=9, fontweight='bold')

ax2.set_xlabel('概率', fontsize=12)
ax2.set_title('自回归生成: 每步Top-5候选', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W17/gpt_architecture.png', dpi=150, bbox_inches='tight')
print("  图表已保存: gpt_architecture.png")
plt.close()

# ============================================================
# 4. 简化GPT实现 (纯NumPy)
# ============================================================
print("\n--- 4. 简化GPT实现 (纯NumPy) ---")


def softmax(x, axis=-1):
    """数值稳定的softmax"""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def layer_norm(x, eps=1e-5):
    """Layer Normalization"""
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + eps)


def self_attention(q, k, v, mask=None):
    """缩放点积注意力"""
    d_k = q.shape[-1]
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    if mask is not None:
        scores = scores + mask * (-1e9)
    weights = softmax(scores, axis=-1)
    return np.matmul(weights, v), weights


class SimpleGPT:
    """
    极简GPT实现 (仅演示前向传播, 不含训练)
    使用随机初始化权重
    """

    def __init__(self, vocab_size=1000, d_model=64, n_heads=4, n_layers=2, max_seq_len=32):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_head = d_model // n_heads

        # Token + Position Embeddings
        self.token_emb = np.random.randn(vocab_size, d_model) * 0.02
        self.pos_emb = np.random.randn(max_seq_len, d_model) * 0.02

        # Transformer Blocks
        self.blocks = []
        for _ in range(n_layers):
            block = {
                'Wq': np.random.randn(d_model, d_model) * 0.02,
                'Wk': np.random.randn(d_model, d_model) * 0.02,
                'Wv': np.random.randn(d_model, d_model) * 0.02,
                'Wo': np.random.randn(d_model, d_model) * 0.02,
                'W1': np.random.randn(d_model, d_model * 4) * 0.02,
                'b1': np.zeros(d_model * 4),
                'W2': np.random.randn(d_model * 4, d_model) * 0.02,
                'b2': np.zeros(d_model),
            }
            self.blocks.append(block)

        # LM Head
        self.lm_head = np.random.randn(d_model, vocab_size) * 0.02

    def forward(self, token_ids):
        """
        前向传播
        token_ids: (seq_len,) 整数数组
        返回: logits (seq_len, vocab_size)
        """
        seq_len = len(token_ids)

        # Embedding
        x = self.token_emb[token_ids] + self.pos_emb[:seq_len]  # (seq_len, d_model)
        x = x[np.newaxis, :]  # (1, seq_len, d_model) 添加batch维度

        # 因果掩码
        mask = 1 - np.tril(np.ones((seq_len, seq_len)))
        mask = mask[np.newaxis, np.newaxis, :, :]  # (1, 1, seq_len, seq_len)

        for block in self.blocks:
            # Multi-Head Self-Attention
            Q = x @ block['Wq']  # (1, seq_len, d_model)
            K = x @ block['Wk']
            V = x @ block['Wv']

            # 分头
            batch = 1
            Q = Q.reshape(batch, seq_len, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
            K = K.reshape(batch, seq_len, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
            V = V.reshape(batch, seq_len, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

            attn_out, _ = self_attention(Q, K, V, mask)  # (1, n_heads, seq_len, d_head)
            attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.d_model)
            attn_out = attn_out @ block['Wo']

            # 残差 + LayerNorm
            x = layer_norm(x + attn_out)

            # Feed-Forward
            ff = np.maximum(0, x @ block['W1'] + block['b1'])  # ReLU
            ff = ff @ block['W2'] + block['b2']
            x = layer_norm(x + ff)

        # LM Head
        x = x.squeeze(0)  # (seq_len, d_model)
        logits = x @ self.lm_head  # (seq_len, vocab_size)
        return logits

    def generate(self, start_tokens, max_new_tokens=10, temperature=1.0):
        """自回归生成"""
        tokens = list(start_tokens)
        for _ in range(max_new_tokens):
            logits = self.forward(np.array(tokens))
            next_logits = logits[-1] / temperature  # 取最后一个位置
            probs = softmax(next_logits)
            next_token = np.random.choice(self.vocab_size, p=probs)
            tokens.append(int(next_token))
        return tokens


# 演示
print("\n  初始化简化GPT模型...")
gpt = SimpleGPT(vocab_size=100, d_model=64, n_heads=4, n_layers=2, max_seq_len=32)
print(f"  模型参数: vocab_size={gpt.vocab_size}, d_model={gpt.d_model}, "
      f"n_heads={gpt.n_heads}, n_layers={gpt.n_layers}")

# 前向传播
input_ids = np.array([1, 5, 10, 15, 20])
print(f"\n  输入 token IDs: {input_ids}")
logits = gpt.forward(input_ids)
print(f"  输出 logits 形状: {logits.shape}")
print(f"  最后位置 top-5 预测: {np.argsort(logits[-1])[-5:][::-1]}")

# 自回归生成
start_tokens = [1, 5, 10]
generated = gpt.generate(start_tokens, max_new_tokens=8, temperature=0.8)
print(f"\n  从 {start_tokens} 开始生成: {generated}")

# ============================================================
# 5. 不同GPT模型配置对比
# ============================================================
print("\n--- 5. 不同GPT模型配置对比 ---")
configs = [
    ("GPT-1",     12,  768,   12, 117e6,  512),
    ("GPT-2",     48, 1600,   25, 1.5e9,  1024),
    ("GPT-3",     96, 12288,  96, 175e9,  2048),
    ("GPT-4*",   120, 16384, 120, 1.8e12, 8192),
]

print(f"  {'模型':<10s} {'层数':>6s} {'隐藏维度':>10s} {'头数':>6s} {'参数量':>12s} {'上下文':>8s}")
print(f"  {'-'*10} {'-'*6} {'-'*10} {'-'*6} {'-'*12} {'-'*8}")
for name, layers, d_model, heads, params, ctx in configs:
    print(f"  {name:<10s} {layers:>6d} {d_model:>10d} {heads:>6d} {params:>12.2e} {ctx:>8d}")

print("""
  * GPT-4 参数量为估计值, 可能采用 MoE (Mixture of Experts) 架构

  参数量估算公式:
    总参数 ~ vocab_size * d_model + n_layers * (
        4 * d_model^2          # Q,K,V,O 投影
        + 2 * d_model * 4*d_model  # FFN (W1, W2)
        + 4 * d_model          # LayerNorm
    ) + d_model * vocab_size   # LM Head
""")

# ============================================================
# 6. 使用HuggingFace加载GPT-2 (可选)
# ============================================================
print("\n--- 6. 使用 HuggingFace 加载 GPT-2 (可选) ---")

try:
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    print("  正在加载 GPT-2 small ...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  模型配置: {model.config}")

    # 生成文本
    input_text = "Artificial intelligence is"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    print(f"\n  输入: '{input_text}'")

    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=30, do_sample=True,
                                 temperature=0.7, top_k=50)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"  生成: '{generated_text}'")

except ImportError as e:
    print(f"  [!] transformers/torch 未安装, 跳过HuggingFace演示: {e}")
    print("  安装命令: pip install transformers torch")

print("\n" + "=" * 60)
print("W17-D2 完成! 本节详解了GPT架构和自回归生成原理")
print("=" * 60)
