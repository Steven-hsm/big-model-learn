"""
W17-D3 BERT架构详解 (BERT Architecture)
========================================
BERT架构详解(encoder-only), MLM和NSP预训练任务,
[CLS]/[SEP] token, BERT微调范式, 代码实现简化BERT, 下游任务适配
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W17-D3 BERT架构详解 (BERT Architecture)")
print("=" * 60)

# ============================================================
# 1. BERT 架构详解
# ============================================================
print("\n--- 1. BERT 架构详解 (Encoder-Only) ---")
print("""
  BERT = Bidirectional Encoder Representations from Transformers

  核心思想: 双向编码器, 同时看左右两侧上下文

  整体结构:
    输入: Token A + [SEP] + Token B
        |
    Token Embedding + Segment Embedding + Position Embedding
        |
    Transformer Encoder Block x N
      |-- Multi-Head Self-Attention (双向, 无掩码)
      |-- Add & LayerNorm
      |-- Feed-Forward Network
      |-- Add & LayerNorm
        |
    输出: 每个token的上下文化表示

  特殊Token:
    [CLS] (101): 句子开头, 用于分类任务 (整个序列的表示)
    [SEP] (102): 句子分隔, 区分两个句子
    [MASK] (103): 掩码, 用于MLM预训练
    [PAD] (0):   填充, 统一批次长度
    [UNK] (100): 未知词, 词表外的token

  BERT两种规格:
    BERT-Base:  12层, 768维, 12头, 110M参数
    BERT-Large: 24层, 1024维, 16头, 340M参数
""")

# ============================================================
# 2. MLM 和 NSP 预训练任务
# ============================================================
print("\n--- 2. MLM 和 NSP 预训练任务 ---")
print("""
  BERT的两个预训练任务:

  1) Masked Language Model (MLM)
     ============================================
     随机遮盖15%的token, 让模型预测被遮盖的词
     - 80%替换为[MASK]
     - 10%替换为随机词
     - 10%保持不变

     示例:
       原句: 我 [CLS] 喜欢 机器 学习 [SEP] 深度 学习 很 有趣 [SEP]
       MLM:  我 [CLS] 喜欢 [MASK] 学习 [SEP] 深度 学习 很 [MASK] [SEP]
       目标:               机器                     有趣

  2) Next Sentence Prediction (NSP)
     ============================================
     判断句子B是否是句子A的下一句 (二分类)

     示例:
       正样本 (IsNext):
         A: "今天天气很好"  B: "我们去公园吧"
         标签: IsNext (1)

       负样本 (NotNext):
         A: "今天天气很好"  B: "量子计算很有趣"
         标签: NotNext (0)

  预训练数据:
    BooksCorpus (8亿词) + 英文维基百科 (25亿词)
""")

# MLM示例演示
print("\n  --- MLM 遮盖策略示例 ---")
sentence = ["[CLS]", "深", "度", "学", "习", "是", "人", "工", "智", "能", "的", "基", "础", "[SEP]"]
masked_indices = [2, 5, 11]  # 遮盖位置

original = sentence.copy()
masked = sentence.copy()

for idx in masked_indices:
    rand = np.random.random()
    if rand < 0.8:
        masked[idx] = "[MASK]"
    elif rand < 0.9:
        masked[idx] = "随机词"
    else:
        masked[idx] = sentence[idx]

print(f"  原句: {' '.join(original)}")
print(f"  遮盖: {' '.join(masked)}")
print(f"  目标: {', '.join([f'位置{i}={original[i]}' for i in masked_indices])}")

# ============================================================
# 3. BERT输入表示
# ============================================================
print("\n--- 3. BERT 输入表示 ---")
print("""
  BERT的输入由三部分Embedding相加组成:

  1) Token Embedding:    词本身的嵌入
  2) Segment Embedding:  句子标识 (句子A=0, 句子B=1)
  3) Position Embedding: 位置信息 (学习的, 非固定正弦)

  示例:
    Tokens:     [CLS] 我 喜欢 NLP [SEP] 深度 学习 有趣 [SEP]
    Token ID:   101   1  2    3   102  4    5    6    102
    Segment:     0    0  0    0    0   1    1    1    1
    Position:    0    1  2    3    4   5    6    7    8
""")

# 可视化Embedding组合
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 子图1: MLM概率可视化
ax1 = axes[0, 0]
vocab = ["机器", "学习", "深度", "自然", "语言", "处理", "人工", "智能", "数据", "模型"]
mlm_probs = np.random.dirichlet(np.ones(len(vocab)) * 2)
mlm_probs[0] += 0.3  # "机器" 概率最高
mlm_probs[1] += 0.2  # "学习" 其次
mlm_probs = mlm_probs / mlm_probs.sum()

colors = ['#e74c3c' if p > 0.15 else '#3498db' for p in mlm_probs]
bars = ax1.bar(range(len(vocab)), mlm_probs, color=colors, edgecolor='black', alpha=0.8)
ax1.set_xticks(range(len(vocab)))
ax1.set_xticklabels(vocab, fontsize=10)
ax1.set_ylabel('预测概率', fontsize=12)
ax1.set_title('MLM: [MASK]位置的Top预测概率', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# 子图2: NSP分类
ax2 = axes[0, 1]
sentences = [
    ("天气好/去公园", 0.92),
    ("天气好/量子计算", 0.08),
    ("学习AI/需要数学", 0.88),
    ("学习AI/今天吃什么", 0.12),
    ("BERT/GPT", 0.15),
    ("NLP/Transformer", 0.85),
]
labels = [s[0] for s in sentences]
is_next_probs = [s[1] for s in sentences]
colors = ['#2ecc71' if p > 0.5 else '#e74c3c' for p in is_next_probs]
ax2.barh(range(len(labels)), is_next_probs, color=colors, edgecolor='black', alpha=0.8)
ax2.set_yticks(range(len(labels)))
ax2.set_yticklabels(labels, fontsize=10)
ax2.set_xlabel('IsNext 概率', fontsize=12)
ax2.set_title('NSP: 句子对预测', fontsize=14, fontweight='bold')
ax2.axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
ax2.grid(True, alpha=0.3, axis='x')

# 子图3: BERT vs GPT 注意力模式
ax3 = axes[1, 0]
seq_len = 8
# BERT: 双向注意力
bert_attn = np.ones((seq_len, seq_len))
ax3.imshow(bert_attn, cmap='Greens', vmin=0, vmax=1)
ax3.set_title('BERT: 双向注意力 (Encoder)', fontsize=14, fontweight='bold')
ax3.set_xlabel('Key 位置')
ax3.set_ylabel('Query 位置')
for i in range(seq_len):
    for j in range(seq_len):
        ax3.text(j, i, '1', ha='center', va='center', fontsize=8)

# 子图4: GPT: 因果注意力
ax4 = axes[1, 1]
gpt_attn = np.tril(np.ones((seq_len, seq_len)))
ax4.imshow(gpt_attn, cmap='Blues', vmin=0, vmax=1)
ax4.set_title('GPT: 因果注意力 (Decoder)', fontsize=14, fontweight='bold')
ax4.set_xlabel('Key 位置')
ax4.set_ylabel('Query 位置')
for i in range(seq_len):
    for j in range(seq_len):
        ax4.text(j, i, str(int(gpt_attn[i, j])), ha='center', va='center', fontsize=8)

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W17/bert_architecture.png', dpi=150, bbox_inches='tight')
print("  图表已保存: bert_architecture.png")
plt.close()

# ============================================================
# 4. 简化BERT实现
# ============================================================
print("\n--- 4. 简化BERT实现 (纯NumPy) ---")


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def layer_norm(x, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + eps)


def gelu(x):
    """GELU激活函数"""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


class SimpleBERT:
    """极简BERT实现"""

    def __init__(self, vocab_size=1000, d_model=64, n_heads=4, n_layers=2, max_seq_len=32):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_head = d_model // n_heads

        np.random.seed(42)

        # Embeddings
        self.token_emb = np.random.randn(vocab_size, d_model) * 0.02
        self.segment_emb = np.random.randn(2, d_model) * 0.02
        self.pos_emb = np.random.randn(max_seq_len, d_model) * 0.02
        self.emb_norm = np.random.randn(d_model) * 0.02

        # Transformer Encoder Blocks
        self.blocks = []
        for _ in range(n_layers):
            self.blocks.append({
                'Wq': np.random.randn(d_model, d_model) * 0.02,
                'Wk': np.random.randn(d_model, d_model) * 0.02,
                'Wv': np.random.randn(d_model, d_model) * 0.02,
                'Wo': np.random.randn(d_model, d_model) * 0.02,
                'W1': np.random.randn(d_model, d_model * 4) * 0.02,
                'b1': np.zeros(d_model * 4),
                'W2': np.random.randn(d_model * 4, d_model) * 0.02,
                'b2': np.zeros(d_model),
            })

        # MLM Head
        self.mlm_head = np.random.randn(d_model, vocab_size) * 0.02

        # NSP Head
        self.nsp_head_w = np.random.randn(d_model, 2) * 0.02
        self.nsp_head_b = np.zeros(2)

    def forward(self, token_ids, segment_ids):
        """
        BERT前向传播
        token_ids: (seq_len,)
        segment_ids: (seq_len,) 0或1
        返回: mlm_logits (seq_len, vocab_size), nsp_logits (2,)
        """
        seq_len = len(token_ids)

        # 三种Embedding相加
        x = self.token_emb[token_ids] + self.segment_emb[segment_ids] + self.pos_emb[:seq_len]
        x = x[np.newaxis, :]  # (1, seq_len, d_model)

        # Encoder Blocks (无掩码, 双向注意力)
        for block in self.blocks:
            Q = (x @ block['Wq']).reshape(1, seq_len, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
            K = (x @ block['Wk']).reshape(1, seq_len, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
            V = (x @ block['Wv']).reshape(1, seq_len, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

            scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_head)
            attn_weights = softmax(scores)
            attn_out = np.matmul(attn_weights, V)
            attn_out = attn_out.transpose(0, 2, 1, 3).reshape(1, seq_len, self.d_model)
            attn_out = attn_out @ block['Wo']
            x = layer_norm(x + attn_out)

            ff = gelu(x @ block['W1'] + block['b1'])
            ff = ff @ block['W2'] + block['b2']
            x = layer_norm(x + ff)

        x = x.squeeze(0)  # (seq_len, d_model)

        # MLM: 每个位置预测词
        mlm_logits = x @ self.mlm_head

        # NSP: [CLS]位置(索引0)做二分类
        cls_output = x[0]
        nsp_logits = cls_output @ self.nsp_head_w + self.nsp_head_b

        return mlm_logits, nsp_logits


# 演示
print("\n  初始化简化BERT模型...")
bert = SimpleBERT(vocab_size=100, d_model=64, n_heads=4, n_layers=2)

token_ids = np.array([1, 5, 10, 15, 20, 25])  # [CLS] A B C [SEP] D
segment_ids = np.array([0, 0, 0, 0, 0, 1])     # 句子A=0, 句子B=1

mlm_logits, nsp_logits = bert.forward(token_ids, segment_ids)
print(f"  输入 tokens: {token_ids}")
print(f"  输入 segments: {segment_ids}")
print(f"  MLM logits 形状: {mlm_logits.shape}")
print(f"  NSP logits: {nsp_logits}")
print(f"  NSP 预测: {'IsNext' if np.argmax(nsp_logits) == 1 else 'NotNext'}")

# ============================================================
# 5. BERT微调范式
# ============================================================
print("\n--- 5. BERT 微调范式 ---")
print("""
  BERT微调的四种典型范式:

  1) 单句分类 (Sentence Classification)
     [CLS] 句子A [SEP] -> [CLS]输出 -> Linear -> 情感/类别
     例: 情感分析, 文本分类

  2) 句子对分类 (Sentence Pair Classification)
     [CLS] 句子A [SEP] 句子B [SEP] -> [CLS]输出 -> Linear -> 关系/类别
     例: 自然语言推理(NLI), 语义相似度

  3) 问答任务 (Question Answering)
     [CLS] 问题 [SEP] 文章 [SEP] -> 每个token输出 -> 起始/结束位置
     例: SQuAD, 阅读理解

  4) 序列标注 (Token Classification)
     [CLS] 词1 词2 ... [SEP] -> 每个token输出 -> Linear -> 标签
     例: NER(命名实体识别), 词性标注
""")

# ============================================================
# 6. 使用HuggingFace BERT (可选)
# ============================================================
print("\n--- 6. 使用 HuggingFace BERT (可选) ---")

try:
    import torch
    from transformers import BertTokenizer, BertModel, BertForMaskedLM

    print("  正在加载 BERT-base-chinese ...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    model = BertForMaskedLM.from_pretrained("bert-base-chinese")

    print(f"  词表大小: {tokenizer.vocab_size}")
    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # MLM演示
    text = "深度学习是人工智能的[MASK]要方向。"
    inputs = tokenizer(text, return_tensors="pt")
    print(f"\n  输入: {text}")

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits

    mask_pos = (inputs.input_ids == tokenizer.mask_token_id).nonzero()
    if len(mask_pos) > 0:
        mask_idx = mask_pos[0][1].item()
        top_5 = torch.topk(predictions[0, mask_idx], 5)
        print("  [MASK] Top-5 预测:")
        for prob, idx in zip(top_5.values, top_5.indices):
            token = tokenizer.decode([idx])
            print(f"    {token}: {prob.item():.4f}")

except ImportError as e:
    print(f"  [!] transformers/torch 未安装, 跳过: {e}")
    print("  安装命令: pip install transformers torch")
except Exception as e:
    print(f"  [!] 加载模型出错: {e}")

print("\n" + "=" * 60)
print("W17-D3 完成! 本节详解了BERT架构、预训练任务和微调范式")
print("=" * 60)
