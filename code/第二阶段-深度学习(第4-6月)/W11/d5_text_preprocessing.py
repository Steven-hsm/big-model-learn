"""
W11-d5: 文本预处理
- Vocabulary类（fit, encode, decode, PAD/UNK tokens）
- 文本分词（简单split + NLTK可选）
- 填充与截断实现
- 自定义文本Dataset
- 变长序列的collate_fn
- 演示完整流程
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. Vocabulary类
# ============================================================
print("=" * 60)
print("1. Vocabulary 类实现")
print("=" * 60)


class Vocabulary:
    """文本词汇表类，支持构建、编码、解码"""

    # 特殊token
    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'
    BOS_TOKEN = '<BOS>'
    EOS_TOKEN = '<EOS>'

    def __init__(self, min_freq=1, max_size=None):
        self.min_freq = min_freq
        self.max_size = max_size
        self.token2idx = {}
        self.idx2token = {}
        self.token_freq = Counter()
        self._built = False

    def fit(self, tokenized_texts):
        """从分词后的文本构建词汇表"""
        # 统计词频
        self.token_freq = Counter()
        for tokens in tokenized_texts:
            self.token_freq.update(tokens)

        # 构建映射（先加入特殊token）
        self.token2idx = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.BOS_TOKEN: 2,
            self.EOS_TOKEN: 3,
        }

        # 按频率排序加入
        sorted_tokens = sorted(
            self.token_freq.items(),
            key=lambda x: (-x[1], x[0])  # 频率降序，同频按字母
        )

        max_vocab = self.max_size - len(self.token2idx) if self.max_size else None
        count = 0
        for token, freq in sorted_tokens:
            if freq < self.min_freq:
                continue
            if max_vocab is not None and count >= max_vocab:
                break
            if token not in self.token2idx:
                self.token2idx[token] = len(self.token2idx)
                count += 1

        self.idx2token = {idx: tok for tok, idx in self.token2idx.items()}
        self._built = True
        return self

    @property
    def vocab_size(self):
        return len(self.token2idx)

    @property
    def pad_idx(self):
        return self.token2idx[self.PAD_TOKEN]

    @property
    def unk_idx(self):
        return self.token2idx[self.UNK_TOKEN]

    def encode(self, tokens, add_special=False):
        """将token列表编码为索引列表"""
        if not self._built:
            raise RuntimeError("词汇表未构建，请先调用fit()")

        result = []
        if add_special:
            result.append(self.token2idx[self.BOS_TOKEN])

        for tok in tokens:
            result.append(self.token2idx.get(tok, self.unk_idx))

        if add_special:
            result.append(self.token2idx[self.EOS_TOKEN])

        return result

    def decode(self, indices, remove_special=True):
        """将索引列表解码为token列表"""
        special_tokens = {self.PAD_TOKEN, self.UNK_TOKEN,
                          self.BOS_TOKEN, self.EOS_TOKEN}
        tokens = []
        for idx in indices:
            tok = self.idx2token.get(idx, self.UNK_TOKEN)
            if remove_special and tok in special_tokens:
                continue
            tokens.append(tok)
        return tokens

    def __len__(self):
        return self.vocab_size

    def __repr__(self):
        return f"Vocabulary(size={self.vocab_size}, min_freq={self.min_freq})"


# 演示
texts = [
    "I love machine learning and deep learning",
    "deep learning is a subset of machine learning",
    "natural language processing uses deep learning",
    "recurrent neural networks process sequences",
    "LSTM and GRU are types of recurrent networks",
]

# 简单分词
tokenized = [text.lower().split() for text in texts]

vocab = Vocabulary(min_freq=1)
vocab.fit(tokenized)

print(f"\n词汇表大小: {vocab.vocab_size}")
print(f"PAD index: {vocab.pad_idx}")
print(f"UNK index: {vocab.unk_idx}")
print(f"\ntoken2idx (前20个):")
for tok, idx in list(vocab.token2idx.items())[:20]:
    print(f"  '{tok}': {idx}")

# 编码示例
sample = "deep learning is great".split()
encoded = vocab.encode(sample, add_special=True)
decoded = vocab.decode(encoded, remove_special=False)
print(f"\n编码演示:")
print(f"  原始: {sample}")
print(f"  编码: {encoded}")
print(f"  解码: {decoded}")
print(f"  去特殊: {vocab.decode(encoded)}")

# UNK处理
unk_sample = "transformer is new".split()
encoded_unk = vocab.encode(unk_sample)
print(f"\nUNK处理:")
print(f"  原始: {unk_sample}")
print(f"  编码: {encoded_unk}")
print(f"  解码: {vocab.decode(encoded_unk)}")


# ============================================================
# 2. 文本分词
# ============================================================
print("\n" + "=" * 60)
print("2. 文本分词")
print("=" * 60)


def simple_tokenize(text):
    """简单分词：小写 + 按空格/标点分割"""
    import re
    text = text.lower()
    # 保留字母、数字、常见标点
    tokens = re.findall(r'\w+|[^\w\s]', text)
    return tokens


# 尝试NLTK分词
try:
    from nltk.tokenize import word_tokenize
    nltk_available = True
    print("\nNLTK可用，使用word_tokenize")
except ImportError:
    nltk_available = False
    print("\nNLTK不可用，使用简单分词")

sample_text = "Hello, world! This is a test. Deep learning is amazing, isn't it?"

print(f"\n原文: {sample_text}")
print(f"简单分词: {simple_tokenize(sample_text)}")
if nltk_available:
    print(f"NLTK分词:  {word_tokenize(sample_text.lower())}")


# ============================================================
# 3. 填充与截断
# ============================================================
print("\n" + "=" * 60)
print("3. 填充(Padding)与截断(Truncation)")
print("=" * 60)


def pad_sequence(indices, max_len, pad_value=0, truncation='pre'):
    """
    对索引序列进行填充或截断
    truncation: 'pre' 截断前面的, 'post' 截断后面的
    """
    if len(indices) > max_len:
        if truncation == 'pre':
            return indices[-max_len:]
        else:
            return indices[:max_len]
    elif len(indices) < max_len:
        padding = [pad_value] * (max_len - len(indices))
        return indices + padding  # 后填充
    return indices


def pad_sequences(sequences, max_len=None, pad_value=0, truncation='pre'):
    """批量填充"""
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)

    result = []
    for seq in sequences:
        result.append(pad_sequence(seq, max_len, pad_value, truncation))
    return np.array(result)


# 演示
sequences = [
    [1, 2, 3],
    [4, 5, 6, 7, 8, 9],
    [10, 11],
    [12, 13, 14, 15],
]

print("\n原始序列:")
for i, seq in enumerate(sequences):
    print(f"  序列{i}: {seq} (长度={len(seq)})")

# 固定长度填充
padded = pad_sequences(sequences, max_len=5, pad_value=0)
print(f"\n填充到长度5 (PAD=0):")
print(padded)

# 截断
truncated = pad_sequences(sequences, max_len=3, pad_value=0, truncation='pre')
print(f"\n截断到长度3 (截断前面):")
print(truncated)

truncated_post = pad_sequences(sequences, max_len=3, pad_value=0, truncation='post')
print(f"\n截断到长度3 (截断后面):")
print(truncated_post)


# ============================================================
# 4. 自定义文本Dataset
# ============================================================
print("\n" + "=" * 60)
print("4. 自定义文本 Dataset")
print("=" * 60)


class TextDataset:
    """自定义文本数据集"""

    def __init__(self, texts, labels, vocab, max_len=50,
                 tokenizer=None, add_special=False):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
        self.tokenizer = tokenizer or simple_tokenize
        self.add_special = add_special

        # 预处理所有文本
        self.encoded_texts = []
        for text in texts:
            tokens = self.tokenizer(text)
            indices = vocab.encode(tokens, add_special=self.add_special)
            indices = pad_sequence(indices, max_len,
                                   pad_value=vocab.pad_idx,
                                   truncation='pre')
            self.encoded_texts.append(indices)

        self.encoded_texts = np.array(self.encoded_texts)
        self.labels = np.array(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.encoded_texts[idx], self.labels[idx]

    def get_text(self, idx):
        """获取原始文本"""
        return self.texts[idx]

    def decode_item(self, idx):
        """解码某个样本"""
        return self.vocab.decode(self.encoded_texts[idx])


# 构建完整训练数据
train_texts = [
    "this movie is great and I love it",
    "terrible film waste of time and money",
    "an excellent masterpiece of cinema",
    "boring and dull not worth watching",
    "wonderful acting and great story",
    "worst movie ever completely awful",
    "highly recommended for everyone",
    "disappointing and poorly directed",
    "a beautiful and touching story",
    "bad acting terrible plot and dialogue",
    "absolutely fantastic best film this year",
    "not good very disappointing experience",
]

train_labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1=正面, 0=负面

# 构建词汇表
tokenized_train = [simple_tokenize(t) for t in train_texts]
full_vocab = Vocabulary(min_freq=1)
full_vocab.fit(tokenized_train)

# 创建Dataset
dataset = TextDataset(train_texts, train_labels, full_vocab, max_len=10)

print(f"\n词汇表大小: {full_vocab.vocab_size}")
print(f"数据集大小: {len(dataset)}")
print(f"编码后形状: {dataset.encoded_texts.shape}")
print(f"\n数据集示例:")
for i in range(4):
    encoded, label = dataset[i]
    print(f"  [{i}] 标签={label} | 编码={encoded} | 原文='{train_texts[i]}'")
    print(f"       解码={dataset.decode_item(i)}")


# ============================================================
# 5. collate_fn for 变长序列
# ============================================================
print("\n" + "=" * 60)
print("5. collate_fn 变长序列批处理")
print("=" * 60)


def collate_fn(batch, pad_value=0):
    """
    自定义collate函数：将变长序列整理成batch
    batch: list of (sequence, label) tuples
    """
    sequences, labels = zip(*batch)
    labels = np.array(labels)

    # 找到batch内最大长度
    max_len = max(len(seq) for seq in sequences)

    # 填充到最大长度
    padded = []
    lengths = []
    for seq in sequences:
        seq_list = list(seq)
        length = len(seq_list)
        padded.append(seq_list + [pad_value] * (max_len - length))
        lengths.append(length)

    padded = np.array(padded)
    lengths = np.array(lengths)

    return padded, labels, lengths


# 模拟变长序列batch
raw_batch = [
    ([1, 2, 3], 0),
    ([4, 5, 6, 7, 8], 1),
    ([9, 10], 0),
    ([11, 12, 13, 14], 1),
]

padded_batch, labels_batch, lengths_batch = collate_fn(raw_batch, pad_value=0)

print(f"\n原始batch:")
for seq, label in raw_batch:
    print(f"  序列={seq}, 标签={label}")

print(f"\ncollate后:")
print(f"  padded:\n{padded_batch}")
print(f"  labels: {labels_batch}")
print(f"  lengths: {lengths_batch}")


# ============================================================
# 6. 完整流程演示
# ============================================================
print("\n" + "=" * 60)
print("6. 完整文本预处理流程演示")
print("=" * 60)

# Step 1: 原始文本
raw_documents = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning algorithms learn from data",
    "Natural language processing is a fascinating field",
    "Deep learning models can understand complex patterns",
    "Text classification is an important NLP task",
    "Recurrent neural networks handle sequential data well",
]

print(f"\nStep 1: 原始文本 ({len(raw_documents)} 条)")
for i, doc in enumerate(raw_documents):
    print(f"  [{i}] {doc}")

# Step 2: 分词
print(f"\nStep 2: 分词")
tokenized_docs = [simple_tokenize(doc) for doc in raw_documents]
for i, tokens in enumerate(tokenized_docs):
    print(f"  [{i}] {tokens[:8]}... ({len(tokens)} tokens)")

# Step 3: 构建词汇表
print(f"\nStep 3: 构建词汇表")
pipeline_vocab = Vocabulary(min_freq=1)
pipeline_vocab.fit(tokenized_docs)
print(f"  词汇表大小: {pipeline_vocab.vocab_size}")
print(f"  前15个词: {list(pipeline_vocab.token2idx.keys())[:15]}")

# Step 4: 编码
print(f"\nStep 4: 编码")
encoded_docs = []
for tokens in tokenized_docs:
    encoded = pipeline_vocab.encode(tokens, add_special=True)
    encoded_docs.append(encoded)
    print(f"  长度={len(encoded):>2d}: {encoded[:10]}{'...' if len(encoded) > 10 else ''}")

# Step 5: 填充
print(f"\nStep 5: 填充到统一长度")
max_len = max(len(e) for e in encoded_docs) + 2
padded_docs = pad_sequences(encoded_docs, max_len=max_len,
                            pad_value=pipeline_vocab.pad_idx)
print(f"  填充后形状: {padded_docs.shape}")
print(f"  PAD index: {pipeline_vocab.pad_idx}")
for i in range(len(padded_docs)):
    print(f"  [{i}] {padded_docs[i]}")

# Step 6: 解码验证
print(f"\nStep 6: 解码验证")
for i in range(3):
    decoded = pipeline_vocab.decode(padded_docs[i])
    original = simple_tokenize(raw_documents[i])
    match = decoded == original
    print(f"  [{i}] 匹配={match}")
    if not match:
        print(f"       原始: {original}")
        print(f"       解码: {decoded}")


# ============================================================
# 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图1: 词频分布
ax = axes[0, 0]
word_freqs = Counter()
for tokens in tokenized_train:
    word_freqs.update(tokens)
top_words = word_freqs.most_common(15)
words, freqs = zip(*top_words)
ax.barh(range(len(words)), freqs, color='steelblue')
ax.set_yticks(range(len(words)))
ax.set_yticklabels(words, fontsize=9)
ax.set_xlabel('频率')
ax.set_title('词频分布 (Top 15)')
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

# 图2: 序列长度分布
ax = axes[0, 1]
lengths = [len(tokens) for tokens in tokenized_docs]
ax.hist(lengths, bins=10, color='coral', edgecolor='black', alpha=0.7)
ax.axvline(x=np.mean(lengths), color='red', linestyle='--',
           label=f'均值={np.mean(lengths):.1f}')
ax.set_xlabel('序列长度')
ax.set_ylabel('频次')
ax.set_title('文本序列长度分布')
ax.legend()
ax.grid(True, alpha=0.3)

# 图3: 编码矩阵热力图
ax = axes[1, 0]
im = ax.imshow(padded_docs, aspect='auto', cmap='YlOrRd')
ax.set_xlabel('位置')
ax.set_ylabel('样本')
ax.set_title('编码后的文本矩阵\n(0=PAD, 数字越大颜色越深)')
plt.colorbar(im, ax=ax, label='Token ID')

# 图4: 填充比例
ax = axes[1, 1]
pad_ratios = []
for row in padded_docs:
    pad_count = np.sum(row == pipeline_vocab.pad_idx)
    pad_ratios.append(pad_count / len(row))
x_pos = range(len(raw_documents))
colors = ['coral' if r > 0.3 else 'steelblue' for r in pad_ratios]
ax.bar(x_pos, pad_ratios, color=colors, edgecolor='black', alpha=0.7)
ax.set_xlabel('样本')
ax.set_ylabel('PAD比例')
ax.set_title('各样本的填充比例\n(红色 > 30%, 蓝色 <= 30%)')
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W11/d5_text_preprocessing.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n图片已保存: d5_text_preprocessing.png")
