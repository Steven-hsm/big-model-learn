"""
W17-D4 分词算法深入 (Tokenization Deep Dive)
=============================================
BPE算法完整实现, WordPiece算法实现, SentencePiece概念,
tiktoken用法, 不同tokenizer对比(词表大小/编码效率)
"""

import sys
import re
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W17-D4 分词算法深入 (Tokenization Deep Dive)")
print("=" * 60)

# ============================================================
# 1. 分词概述
# ============================================================
print("\n--- 1. 分词概述 ---")
print("""
  分词 (Tokenization) 是NLP的第一步:
    原始文本 -> Token序列 -> Token ID -> 模型输入

  常见分词方法:
    1) 字符级 (Character): 词表小(~256), 序列长, 语义弱
    2) 词级 (Word):        词表大(>100K), OOV问题, 序列短
    3) 子词级 (Subword):   词表适中(30K-100K), 兼顾效率和语义
       - BPE (Byte-Pair Encoding):      GPT系列
       - WordPiece:                     BERT
       - Unigram:                       T5/ALBERT
       - SentencePiece:                 多语言, 语言无关
""")

# ============================================================
# 2. BPE 算法完整实现
# ============================================================
print("\n--- 2. BPE 算法完整实现 ---")
print("""
  BPE (Byte-Pair Encoding) 核心思想:
    1. 初始化: 每个字符/字节作为一个token
    2. 统计所有相邻token对的频率
    3. 合并频率最高的token对, 作为新token加入词表
    4. 重复步骤2-3直到达到目标词表大小
""")


class SimpleBPE:
    """从零实现 BPE 分词器"""

    def __init__(self):
        self.merges = []       # 合并规则列表
        self.vocab = {}        # 最终词表

    def _get_pairs(self, word_freqs):
        """统计相邻token对频率"""
        pairs = Counter()
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def _merge_pair(self, pair, word_freqs):
        """合并指定的token对"""
        new_word_freqs = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        for word, freq in word_freqs.items():
            new_word = word.replace(bigram, replacement)
            new_word_freqs[new_word] = freq
        return new_word_freqs

    def train(self, text, num_merges=50):
        """训练BPE: 学习合并规则"""
        # 初始化: 每个字符作为一个token, 用空格分隔
        word_freqs = Counter()
        for word in text.split():
            # 在字符间加空格, 末尾加 </w> 表示词尾
            processed = ' '.join(list(word)) + ' </w>'
            word_freqs[processed] += 1

        # 初始词表: 所有出现的字符
        initial_vocab = set()
        for word in word_freqs:
            for char in word.split():
                initial_vocab.add(char)
        print(f"  初始词表大小: {len(initial_vocab)}")
        print(f"  初始词表: {sorted(initial_vocab)[:20]}...")

        # 迭代合并
        self.merges = []
        for i in range(num_merges):
            pairs = self._get_pairs(word_freqs)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            word_freqs = self._merge_pair(best_pair, word_freqs)
            self.merges.append(best_pair)

            if i < 10 or i == num_merges - 1:
                print(f"  合并 #{i+1}: {best_pair[0]} + {best_pair[1]} -> {best_pair[0]+best_pair[1]} (频率: {pairs[best_pair]})")

        # 构建最终词表
        self.vocab = set(initial_vocab)
        for pair in self.merges:
            self.vocab.add(pair[0] + pair[1])

        print(f"\n  最终词表大小: {len(self.vocab)}")
        print(f"  合并规则数: {len(self.merges)}")
        return self

    def tokenize(self, word):
        """使用学到的规则对单词分词"""
        symbols = list(word) + ['</w>']
        symbols_str = symbols[:]

        for pair in self.merges:
            i = 0
            while i < len(symbols_str) - 1:
                if symbols_str[i] == pair[0] and symbols_str[i + 1] == pair[1]:
                    symbols_str = symbols_str[:i] + [pair[0] + pair[1]] + symbols_str[i + 2:]
                else:
                    i += 1
        return symbols_str


# 训练BPE
print("\n  --- 训练BPE ---")
corpus = """
low lower newest widest low low lower lowest newest newest newest
newest widest lowest low lower low deep learning learn learner
machine learning deep neural network transformer attention mechanism
natural language processing computer vision speech recognition
artificial intelligence model training data feature engineering
"""
corpus = ' '.join(corpus.split())  # 清理空白
bpe = SimpleBPE()
bpe.train(corpus, num_merges=30)

# 测试分词
test_words = ["lowest", "learning", "lower", "newest", "transformer"]
print("\n  --- 测试BPE分词 ---")
for word in test_words:
    tokens = bpe.tokenize(word)
    print(f"  '{word}' -> {tokens}")

# ============================================================
# 3. WordPiece 算法实现
# ============================================================
print("\n--- 3. WordPiece 算法实现 ---")
print("""
  WordPiece 与 BPE 的区别:
    - BPE:       选择频率最高的对进行合并
    - WordPiece: 选择使似然增益最大的对进行合并
    - WordPiece 使用 ## 前缀标记非词首子词

  示例:
    "unwanted" -> ["un", "##want", "##ed"]
    "learning" -> ["learn", "##ing"]
""")


class SimpleWordPiece:
    """简化 WordPiece 分词器"""

    def __init__(self, vocab=None):
        self.vocab = vocab or set()

    def train(self, text, min_freq=2):
        """简化训练: 从语料中提取子词"""
        word_freqs = Counter(text.split())

        # 收集所有字符和子词
        subword_freqs = Counter()
        for word, freq in word_freqs.items():
            # 词首不加前缀, 非词首加 ##
            chars = list(word)
            for i, char in enumerate(chars):
                token = char if i == 0 else '##' + char
                subword_freqs[token] += freq
            # 添加整个词
            subword_freqs[word] += freq

        # 过滤低频
        self.vocab = {token for token, freq in subword_freqs.items() if freq >= min_freq}
        print(f"  WordPiece 词表大小: {len(self.vocab)}")
        return self

    def tokenize(self, word):
        """贪婪最长匹配分词"""
        tokens = []
        start = 0
        while start < len(word):
            end = len(word)
            found = False
            while start < end:
                substr = word[start:end]
                if start > 0:
                    substr = '##' + substr
                if substr in self.vocab:
                    tokens.append(substr)
                    found = True
                    break
                end -= 1
            if not found:
                tokens.append(word[start] if start == 0 else '##' + word[start])
                start += 1
            else:
                start = end
        return tokens


# 训练WordPiece
wp = SimpleWordPiece()
wp.train(corpus)

test_words_wp = ["learning", "lower", "newest", "transformer", "machine"]
print("\n  --- 测试WordPiece分词 ---")
for word in test_words_wp:
    tokens = wp.tokenize(word)
    print(f"  '{word}' -> {tokens}")

# ============================================================
# 4. 不同Tokenizer对比
# ============================================================
print("\n--- 4. 不同Tokenizer对比 ---")

tokenizer_comparison = [
    ("GPT-2 BPE",          50257, "Byte-level BPE",      "~1.3 tokens/word"),
    ("GPT-4 (cl100k)",    100277, "Byte-level BPE",      "~1.1 tokens/word"),
    ("BERT WordPiece",      30522, "WordPiece",           "~1.3 tokens/word"),
    ("BERT-chinese",        21128, "WordPiece(字级)",      "~1.0 tokens/char"),
    ("T5 SentencePiece",   32000, "Unigram + SentencePiece", "~1.2 tokens/word"),
    ("LLaMA SentencePiece", 32000, "BPE + SentencePiece", "~1.2 tokens/word"),
    ("Gemma",              256000, "BPE",                 "~1.0 tokens/word"),
]

print(f"  {'模型Tokenizer':<22s} {'词表大小':>10s} {'算法':<25s} {'编码效率':>18s}")
print(f"  {'-'*22} {'-'*10} {'-'*25} {'-'*18}")
for name, vocab_size, algo, eff in tokenizer_comparison:
    print(f"  {name:<22s} {vocab_size:>10d} {algo:<25s} {eff:>18s}")

# ============================================================
# 5. 可视化对比
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 子图1: 词表大小对比
ax1 = axes[0]
names = [t[0].split()[0] + '\n' + t[0].split()[1] if len(t[0].split()) > 1 else t[0]
         for t in tokenizer_comparison]
vocab_sizes = [t[1] for t in tokenizer_comparison]
colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
bars = ax1.barh(range(len(names)), vocab_sizes, color=colors, edgecolor='black', alpha=0.85)
ax1.set_yticks(range(len(names)))
ax1.set_yticklabels(names, fontsize=9)
ax1.set_xlabel('词表大小', fontsize=12)
ax1.set_title('不同Tokenizer词表大小对比', fontsize=14, fontweight='bold')
for bar, size in zip(bars, vocab_sizes):
    ax1.text(bar.get_width() + 500, bar.get_y() + bar.get_height()/2,
             f'{size:,}', ha='left', va='center', fontsize=9)
ax1.grid(True, alpha=0.3, axis='x')

# 子图2: 分词结果可视化
ax2 = axes[1]
test_sentence = "unbelievably"
tokenizers_demo = {
    "字符级": list(test_sentence),
    "BPE (模拟)": ["un", "bel", "iev", "ably"],
    "WordPiece": ["un", "##believ", "##ably"],
    "词级": ["unbelievably"],
}

y_pos = 0
for tok_name, tokens in tokenizers_demo.items():
    x_start = 0
    token_colors = plt.cm.Pastel1(np.linspace(0, 1, max(len(tokens), 1)))
    for j, token in enumerate(tokens):
        width = max(len(token) * 0.8, 0.5)
        ax2.barh(y_pos, width, left=x_start, height=0.6,
                  color=token_colors[j % len(token_colors)], edgecolor='black')
        ax2.text(x_start + width/2, y_pos, token, ha='center', va='center', fontsize=9)
        x_start += width + 0.1
    ax2.text(-0.5, y_pos, tok_name, ha='right', va='center', fontsize=10, fontweight='bold')
    y_pos += 1

ax2.set_xlim(-3, 12)
ax2.set_title(f'分词对比: "{test_sentence}"', fontsize=14, fontweight='bold')
ax2.axis('off')

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W17/tokenization_comparison.png', dpi=150, bbox_inches='tight')
print("  图表已保存: tokenization_comparison.png")
plt.close()

# ============================================================
# 6. tiktoken 用法 (可选)
# ============================================================
print("\n--- 6. tiktoken 用法 (可选) ---")

try:
    import tiktoken

    # GPT-4 使用的 cl100k_base 编码器
    enc = tiktoken.get_encoding("cl100k_base")
    print(f"  编码器: cl100k_base")
    print(f"  词表大小: {enc.n_vocab}")

    # 编码
    text = "Hello, world! 深度学习是人工智能的重要方向。"
    tokens = enc.encode(text)
    print(f"\n  原文: {text}")
    print(f"  Token IDs: {tokens}")
    print(f"  Token数量: {len(tokens)}")

    # 解码
    decoded = enc.decode(tokens)
    print(f"  解码: {decoded}")

    # 逐token解码
    print("  逐Token解码:")
    for tid in tokens:
        print(f"    ID={tid:>6d} -> '{enc.decode([tid])}'")

    # 不同编码器对比
    print("\n  不同编码器对比:")
    for encoding_name in ["r50k_base", "p50k_base", "cl100k_base", "o200k_base"]:
        try:
            e = tiktoken.get_encoding(encoding_name)
            n_tokens = len(e.encode(text))
            print(f"    {encoding_name:<15s}: 词表={e.n_vocab:>7d}, tokens={n_tokens}")
        except Exception:
            print(f"    {encoding_name:<15s}: 不可用")

except ImportError:
    print("  [!] tiktoken 未安装, 跳过")
    print("  安装命令: pip install tiktoken")

# ============================================================
# 7. SentencePiece 概念
# ============================================================
print("\n--- 7. SentencePiece 概念 ---")
print("""
  SentencePiece 特点:
    - 语言无关: 直接处理原始文本, 不需要预分词
    - 支持BPE和Unigram两种算法
    - 可处理任何语言(中文/日文/韩文等无需分词预处理)
    - 训练和推理速度很快

  使用方式:
    import sentencepiece as spm

    # 训练
    spm.SentencePieceTrainer.train(
        input='corpus.txt',
        model_prefix='mymodel',
        vocab_size=32000,
        model_type='bpe'  # 或 'unigram'
    )

    # 加载和分词
    sp = spm.SentencePieceProcessor()
    sp.load('mymodel.model')
    tokens = sp.encode('你好世界', out_type=str)
    ids = sp.encode('你好世界', out_type=int)
""")

print("\n" + "=" * 60)
print("W17-D4 完成! 本节深入实现了BPE和WordPiece分词算法")
print("=" * 60)
