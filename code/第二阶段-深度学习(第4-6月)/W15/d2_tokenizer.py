"""
W15-D2 分词器 (Tokenizer)
==========================
BPE 算法手动实现, WordPiece / SentencePiece 概念,
使用 transformers AutoTokenizer, 展示分词步骤, 特殊标记
"""

import re
from collections import Counter, defaultdict

print("=" * 60)
print("W15-D2 分词器 (Tokenizer)")
print("=" * 60)

# ============================================================
# 1. 分词的意义
# ============================================================
print("\n--- 1. 分词的意义 ---")
print("""
  文本 → Token → ID → 模型输入

  理想分词器应该:
    - 保持语义完整性 (unfriendly → un + friend + ly)
    - 处理未见过的词 (OOV, Out-of-Vocabulary)
    - 高效编码 (短序列, 小词表)
    - 多语言支持

  主流算法: BPE, WordPiece, SentencePiece (Unigram)
""")

# ============================================================
# 2. BPE (Byte Pair Encoding) 手动实现
# ============================================================
print("\n--- 2. BPE 算法手动实现 ---")
print("""
  BPE 思路:
    1) 初始化: 将每个字符作为单个 token
    2) 统计所有相邻 token 对的频率
    3) 合并频率最高的 token 对
    4) 重复 2-3 直到达到目标词表大小或合并次数
""")


class SimpleBPE:
    """简化的 BPE 分词器"""

    def __init__(self, num_merges=50):
        self.num_merges = num_merges
        self.merges = []       # 合并规则列表
        self.vocab = {}        # 最终词表

    def _get_word_freqs(self, corpus):
        """将语料拆成词, 并统计频率"""
        word_freqs = Counter()
        for text in corpus:
            words = text.lower().split()
            for word in words:
                # 将词拆成字符序列, 末尾加 </w> 标记词结束
                chars = tuple(list(word) + ["</w>"])
                word_freqs[chars] += 1
        return word_freqs

    def _get_pair_freqs(self, word_freqs):
        """统计相邻 token 对的频率"""
        pair_freqs = Counter()
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs

    def _merge_pair(self, word_freqs, pair):
        """合并指定 token 对"""
        new_word_freqs = {}
        bigram = pair[0] + pair[1]
        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                    new_word.append(bigram)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_freqs[tuple(new_word)] = freq
        return new_word_freqs

    def fit(self, corpus):
        """训练 BPE"""
        word_freqs = self._get_word_freqs(corpus)

        # 构建初始词表 (所有单个字符)
        self.vocab = set()
        for word in word_freqs:
            for char in word:
                self.vocab.add(char)

        # 迭代合并
        for step in range(self.num_merges):
            pair_freqs = self._get_pair_freqs(word_freqs)
            if not pair_freqs:
                break
            best_pair = max(pair_freqs, key=pair_freqs.get)
            word_freqs = self._merge_pair(word_freqs, best_pair)
            self.merges.append(best_pair)
            self.vocab.add(best_pair[0] + best_pair[1])

        self.vocab = sorted(self.vocab)
        return self

    def tokenize(self, word):
        """对单个词进行 BPE 分词"""
        tokens = list(word.lower()) + ["</w>"]
        for merge_pair in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == merge_pair[0] and tokens[i + 1] == merge_pair[1]:
                    tokens = tokens[:i] + [merge_pair[0] + merge_pair[1]] + tokens[i + 2:]
                else:
                    i += 1
        return tokens


# 训练 BPE
corpus = [
    "low lower newest widest",
    "low low low low low",
    "newest newest newest newest newest widest",
    "the cat sat on the mat",
    "the dog played with the cat",
    "natural language processing is fun",
    "deep learning models process language",
]

bpe = SimpleBPE(num_merges=30)
bpe.fit(corpus)

print(f"  BPE 词表大小: {len(bpe.vocab)}")
print(f"  合并规则 (前 10 条): {bpe.merges[:10]}")

test_words = ["lowest", "newest", "language", "processing"]
print("\n  分词测试:")
for word in test_words:
    tokens = bpe.tokenize(word)
    print(f"    {word:15s} → {tokens}")

# ============================================================
# 3. WordPiece 概念
# ============================================================
print("\n--- 3. WordPiece 概念 ---")
print("""
  WordPiece (用于 BERT):
    - 类似 BPE, 但选择合并对的标准不同
    - BPE 选频率最高的对
    - WordPiece 选 似然增益 最大的对
    - 词表用 '##' 标记子词后缀
      例: "unfriendly" → ["un", "##friend", "##ly"]
    - 用于: BERT, DistilBERT, Electra

  SentencePiece / Unigram (用于 T5, LLaMA):
    - 不依赖预分词, 直接在原始文本上操作
    - Unigram 语言模型: 从大词表逐步删减
    - 支持字节回退 (byte-fallback)
    - 多语言友好
    - 用于: T5, mT5, ALBERT, LLaMA 系列
""")

# ============================================================
# 4. 使用 AutoTokenizer
# ============================================================
print("\n--- 4. 使用 transformers AutoTokenizer ---")

from transformers import AutoTokenizer

# 加载 BERT tokenizer (WordPiece)
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print(f"  BERT tokenizer 类型:   {type(bert_tokenizer).__name__}")
print(f"  词表大小:              {bert_tokenizer.vocab_size}")
print(f"  模型最大输入长度:       {bert_tokenizer.model_max_length}")

# 加载 DistilBERT tokenizer
distil_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
print(f"\n  DistilBERT tokenizer:   {type(distil_tokenizer).__name__}")

# ============================================================
# 5. 分词步骤详解: text → tokens → ids
# ============================================================
print("\n--- 5. 分词步骤: text → tokens → ids ---")

text = "HuggingFace makes NLP easy and accessible!"

print(f"  原始文本: {text}")

# Step 1: 分词
tokens = bert_tokenizer.tokenize(text)
print(f"  Step 1 - 分词:    {tokens}")

# Step 2: 转为 ID
ids = bert_tokenizer.convert_tokens_to_ids(tokens)
print(f"  Step 2 - Token IDs: {ids}")

# Step 3: 解码回文本
decoded = bert_tokenizer.decode(ids)
print(f"  Step 3 - 解码:    {decoded}")

# 一步完成: encode
encoded = bert_tokenizer.encode(text)
print(f"\n  encode (含特殊标记): {encoded}")

# 使用 __call__ (推荐)
output = bert_tokenizer(text, return_tensors="pt")
print(f"\n  tokenizer(text) 返回:")
for key, val in output.items():
    print(f"    {key}: {val.shape} → {val[0].tolist()[:15]}...")

# ============================================================
# 6. 特殊标记 (Special Tokens)
# ============================================================
print("\n--- 6. 特殊标记 (Special Tokens) ---")

special = bert_tokenizer.special_tokens_map
print(f"  特殊标记映射: {special}")
print(f"""
  特殊标记说明:
    [CLS] (ID={bert_tokenizer.cls_token_id})
      - 分类标记, 放在序列开头
      - 其对应的隐藏状态用于分类任务
    [SEP] (ID={bert_tokenizer.sep_token_id})
      - 分隔标记, 分隔不同句子
      - 也放在序列末尾
    [PAD] (ID={bert_tokenizer.pad_token_id})
      - 填充标记, 使同一 batch 内序列等长
    [UNK] (ID={bert_tokenizer.unk_token_id})
      - 未知标记, 词表中不存在的 token
    [MASK] (ID={bert_tokenizer.mask_token_id})
      - 掩码标记, 用于 MLM (Masked Language Model)
""")

# 演示 batch 编码 (padding + truncation)
texts = [
    "Short text.",
    "This is a somewhat longer sentence for demonstration.",
    "Hi!"
]

batch = bert_tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=20,
    return_tensors="pt"
)

print("  Batch 编码 (自动 padding):")
print(f"    input_ids shape:    {batch['input_ids'].shape}")
print(f"    attention_mask shape: {batch['attention_mask'].shape}")
for i, text in enumerate(texts):
    ids = batch["input_ids"][i].tolist()
    toks = bert_tokenizer.convert_ids_to_tokens(ids)
    print(f"    [{i}] {text[:40]:40s} → {toks}")

# 处理中文
print("\n  中文分词示例:")
chinese_text = "自然语言处理是人工智能的重要方向"
cn_tokens = bert_tokenizer.tokenize(chinese_text)
cn_ids = bert_tokenizer.encode(chinese_text)
print(f"    原文:   {chinese_text}")
print(f"    tokens: {cn_tokens}")
print(f"    ids:    {cn_ids}")

# ============================================================
# 7. 总结
# ============================================================
print("\n--- 7. 总结 ---")
print("""
  本节学习了:
  1) BPE 算法原理和手动实现 (频率驱动的子词合并)
  2) WordPiece 和 SentencePiece 概念
  3) AutoTokenizer 加载和使用
  4) 分词完整流程: text → tokens → ids → tensors
  5) 特殊标记 [CLS], [SEP], [PAD], [UNK], [MASK]

  下一步: d3_pipeline_inference.py - Pipeline 推理实战
""")
