"""
W13-D1: Attention Intuition, Q/K/V, Basic Attention, Dot-Product, Visualization
================================================================================
从直觉到实现，逐步理解注意力机制。
"""

import torch
import torch.nn.functional as F
import numpy as np


# ============================================================
# 1. 注意力直觉: 为什么需要注意力？
# ============================================================
def demo_attention_intuition():
    """
    用自然语言处理的例子解释注意力
    """
    print("=" * 60)
    print("1. 注意力机制直觉")
    print("=" * 60)

    print("例子: 'The animal didn't cross the street because IT was too tired'")
    print("问题: 'IT' 指的是什么？")
    print()
    print("人类会自然地关注上下文中相关的词:")
    print("  'IT' 与 'animal' 的关联度高 (tired 描述的是 animal)")
    print("  'IT' 与 'street' 的关联度低")
    print()
    print("注意力机制就是让模型学会:'关注'输入中与当前处理最相关的部分")
    print("核心思想: 用 Query 去 '查询' 所有的 Key，得到与每个 Key 的相关度，")
    print("         然后用相关度作为权重，对 Value 进行加权求和。")
    print()

    # 简单例子: 词向量相似度
    print("--- 词向量相似度模拟 ---")
    words = ["animal", "street", "it", "tired", "cross"]
    np.random.seed(42)
    # 模拟词向量（实际中来自 Embedding 层）
    embeddings = np.random.randn(len(words), 8).astype(np.float32)

    # "it" 与其他词的相似度（点积）
    it_vec = embeddings[2]  # "it"
    similarities = embeddings @ it_vec  # 点积相似度
    attention_weights = np.exp(similarities) / np.exp(similarities).sum()

    print("注意力权重 (it 关注每个词的程度):")
    for word, weight in zip(words, attention_weights):
        bar = "#" * int(weight * 50)
        print(f"  {word:>8}: {weight:.4f} {bar}")
    print()


# ============================================================
# 2. Q/K/V 概念
# ============================================================
def demo_qkv_concept():
    """
    解释 Query, Key, Value 的概念
    """
    print("=" * 60)
    print("2. Q/K/V 概念")
    print("=" * 60)

    print("类比: 图书馆搜索")
    print("  Query (Q):  你想搜索的内容 ('深度学习入门')")
    print("  Key (K):    每本书的标签/关键词 ('机器学习', '深度学习', '烹饪')")
    print("  Value (V):  每本书的实际内容")
    print()
    print("注意力计算步骤:")
    print("  1. 计算 Q 与每个 K 的相似度 -> attention_scores")
    print("  2. 对 scores 做 softmax -> attention_weights (归一化为概率)")
    print("  3. 用 weights 对 V 加权求和 -> output (注意力输出)")
    print()

    # 用矩阵演示
    print("--- 矩阵形式 ---")
    seq_len = 4    # 序列长度
    d_model = 8    # 模型维度

    # 原始输入
    X = torch.randn(seq_len, d_model)
    print(f"输入 X shape: {X.shape} (序列长度={seq_len}, 维度={d_model})")

    # 通过线性变换得到 Q, K, V
    W_q = torch.randn(d_model, d_model) * 0.1
    W_k = torch.randn(d_model, d_model) * 0.1
    W_v = torch.randn(d_model, d_model) * 0.1

    Q = X @ W_q  # [seq_len, d_model]
    K = X @ W_k
    V = X @ W_v

    print(f"Q shape: {Q.shape}  — 每个位置想要'查询'什么")
    print(f"K shape: {K.shape}  — 每个位置可以被'匹配'到什么")
    print(f"V shape: {V.shape}  — 每个位置提供的'内容'")
    print()


# ============================================================
# 3. 基础注意力: 加性注意力 (Bahdanau)
# ============================================================
def demo_additive_attention():
    """
    Bahdanau (加性) 注意力
    score(q, k) = v^T * tanh(W_q * q + W_k * k)
    """
    print("=" * 60)
    print("3. 加性注意力 (Bahdanau)")
    print("=" * 60)

    q = torch.randn(1, 6)   # 1个查询, 维度6
    K = torch.randn(5, 6)   # 5个Key, 维度6
    V = torch.randn(5, 8)   # 5个Value, 维度8

    # 参数
    W_q = torch.randn(6, 4) * 0.1
    W_k = torch.randn(6, 4) * 0.1
    v = torch.randn(4, 1) * 0.1

    # 计算分数
    scores = torch.tanh(q @ W_q + K @ W_k) @ v  # [5, 1]

    # Softmax 归一化
    weights = F.softmax(scores, dim=0)

    # 加权求和
    output = (weights * V).sum(dim=0, keepdim=True)

    print(f"Query:     {q.shape}")
    print(f"Keys:      {K.shape}")
    print(f"Values:    {V.shape}")
    print(f"Scores:    {scores.squeeze().tolist()}")
    print(f"Weights:   {weights.squeeze().tolist()}")
    print(f"Output:    {output.shape}")
    print()


# ============================================================
# 4. 点积注意力 (Dot-Product Attention)
# ============================================================
def demo_dot_product_attention():
    """
    点积注意力: score(Q, K) = Q @ K^T
    最简单的注意力形式
    """
    print("=" * 60)
    print("4. 点积注意力")
    print("=" * 60)

    torch.manual_seed(42)

    seq_len = 4
    d_k = 8

    Q = torch.randn(seq_len, d_k)
    K = torch.randn(seq_len, d_k)
    V = torch.randn(seq_len, d_k)

    # Step 1: 计算注意力分数
    scores = Q @ K.transpose(-2, -1)  # [seq_len, seq_len]
    print(f"1. Scores (Q @ K^T) shape: {scores.shape}")
    print(f"   Score 矩阵:")
    for i, row in enumerate(scores):
        row_str = " ".join(f"{v:>7.2f}" for v in row)
        print(f"   位置 {i}: [{row_str}]")

    # Step 2: Softmax 归一化
    attention_weights = F.softmax(scores, dim=-1)
    print(f"\n2. Attention Weights (softmax) shape: {attention_weights.shape}")
    print(f"   每行和为1: {attention_weights.sum(dim=-1).tolist()}")

    # Step 3: 加权求和
    output = attention_weights @ V  # [seq_len, d_k]
    print(f"\n3. Output (weights @ V) shape: {output.shape}")

    print(f"\n总结:")
    print(f"  attention(Q, K, V) = softmax(Q @ K^T) @ V")
    print(f"  输入: Q[{seq_len},{d_k}], K[{seq_len},{d_k}], V[{seq_len},{d_k}]")
    print(f"  输出: [{seq_len}, {d_k}]")
    print()


# ============================================================
# 5. 注意力权重可视化
# ============================================================
def visualize_attention():
    """
    用字符画可视化注意力权重矩阵
    """
    print("=" * 60)
    print("5. 注意力权重可视化")
    print("=" * 60)

    torch.manual_seed(42)

    # 模拟一个英法翻译的注意力
    src_words = ["Le", "chat", "est", "sur", "le", "tapis"]
    tgt_words = ["The", "cat", "is", "on", "the", "mat"]

    seq_len = len(src_words)
    d_k = 16

    Q = torch.randn(len(tgt_words), d_k) * 0.3
    K = torch.randn(len(src_words), d_k) * 0.3

    # 制造一些对齐模式
    # 让 "The" 关注 "Le", "cat" 关注 "chat" 等
    for i in range(min(len(tgt_words), len(src_words))):
        Q[i] = K[i] + torch.randn(d_k) * 0.1

    scores = Q @ K.transpose(-2, -1)
    weights = F.softmax(scores, dim=-1)

    # 字符画热力图
    print("注意力热力图 (目标词 关注 源词):")
    print()

    # 表头
    header = "          "
    for w in src_words:
        header += f" {w:>5}"
    print(header)
    print("          " + "-" * (len(src_words) * 6 + 2))

    for i, tgt_word in enumerate(tgt_words):
        row = f"{tgt_word:>8} |"
        for j in range(len(src_words)):
            w = weights[i, j].item()
            if w > 0.5:
                ch = "█████"
            elif w > 0.2:
                ch = "███  "
            elif w > 0.05:
                ch = "█    "
            else:
                ch = "     "
            row += f" {ch}"
        print(row)

    print()
    print("图例: █████ > 0.5 | ███ > 0.2 | █ > 0.05 | 空 < 0.05")
    print()


# ============================================================
# 6. 自注意力 vs 交叉注意力
# ============================================================
def demo_self_vs_cross_attention():
    print("=" * 60)
    print("6. 自注意力 vs 交叉注意力")
    print("=" * 60)

    d_k = 8

    # --- 自注意力: Q, K, V 来自同一个序列 ---
    print("--- 自注意力 ---")
    X = torch.randn(5, d_k)
    W_q = torch.randn(d_k, d_k) * 0.1
    W_k = torch.randn(d_k, d_k) * 0.1
    W_v = torch.randn(d_k, d_k) * 0.1

    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v

    scores = Q @ K.transpose(-2, -1)
    weights = F.softmax(scores, dim=-1)
    self_attn_out = weights @ V

    print(f"输入 X: {X.shape}")
    print(f"Q=X@Wq, K=X@Wk, V=X@Wv (都来自同一输入)")
    print(f"自注意力输出: {self_attn_out.shape}")
    print(f"含义: 每个位置关注同一序列中的其他位置")

    # --- 交叉注意力: Q 来自一个序列, K,V 来自另一个 ---
    print("\n--- 交叉注意力 ---")
    X_encoder = torch.randn(6, d_k)  # 编码器输出
    X_decoder = torch.randn(4, d_k)  # 解码器隐藏状态

    W_q2 = torch.randn(d_k, d_k) * 0.1
    W_k2 = torch.randn(d_k, d_k) * 0.1
    W_v2 = torch.randn(d_k, d_k) * 0.1

    Q2 = X_decoder @ W_q2    # 来自解码器
    K2 = X_encoder @ W_k2    # 来自编码器
    V2 = X_encoder @ W_v2    # 来自编码器

    scores2 = Q2 @ K2.transpose(-2, -1)
    weights2 = F.softmax(scores2, dim=-1)
    cross_attn_out = weights2 @ V2

    print(f"编码器输出: {X_encoder.shape}")
    print(f"解码器状态: {X_decoder.shape}")
    print(f"Q 来自解码器, K/V 来自编码器")
    print(f"交叉注意力输出: {cross_attn_out.shape}")
    print(f"含义: 解码器每个位置关注编码器输出中的相关部分")
    print()


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    demo_attention_intuition()
    demo_qkv_concept()
    demo_additive_attention()
    demo_dot_product_attention()
    visualize_attention()
    demo_self_vs_cross_attention()

    print("所有演示完成！")
