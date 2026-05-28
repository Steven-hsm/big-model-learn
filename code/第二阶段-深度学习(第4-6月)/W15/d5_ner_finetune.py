"""
W15-D5 命名实体识别 (NER) 微调
================================
NER 任务介绍, 合成 NER 数据, tokenize + label alignment,
微调 BERT 进行 token classification, seqeval 评估
"""

import numpy as np
import torch
from torch.utils.data import Dataset, random_split

print("=" * 60)
print("W15-D5 命名实体识别 (NER) 微调")
print("=" * 60)

# ============================================================
# 1. NER 任务介绍
# ============================================================
print("\n--- 1. NER 任务介绍 ---")
print("""
  命名实体识别 (Named Entity Recognition, NER):
    输入: "Apple was founded by Steve Jobs in California"
    输出: Apple→ORG, Steve Jobs→PER, California→LOC

  BIO 标注体系:
    B-PER: 人名开头    I-PER: 人名内部
    B-ORG: 组织开头    I-ORG: 组织内部
    B-LOC: 地点开头    I-LOC: 地点内部
    O:    非实体

  示例标注:
    Apple  was   founded  by    Steve  Jobs   in   California
    B-ORG  O     O        O     B-PER  I-PER  O    B-LOC

  常见 NER 数据集: CoNLL-2003, OntoNotes, WNUT 2017
""")

# ============================================================
# 2. 构造合成 NER 数据
# ============================================================
print("\n--- 2. 构造合成 NER 数据 ---")

# 合成数据: 每个样本是 (tokens, ner_tags) 的元组
# 标签映射: O=0, B-PER=1, I-PER=2, B-ORG=3, I-ORG=4, B-LOC=5, I-LOC=6
label_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for i, l in enumerate(label_list)}

synthetic_ner_data = [
    (["John", "Smith", "works", "at", "Google", "in", "New", "York"],
     ["B-PER", "I-PER", "O", "O", "B-ORG", "O", "B-LOC", "I-LOC"]),
    (["Microsoft", "was", "founded", "by", "Bill", "Gates"],
     ["B-ORG", "O", "O", "O", "B-PER", "I-PER"]),
    (["Elon", "Musk", "leads", "Tesla", "and", "SpaceX"],
     ["B-PER", "I-PER", "O", "B-ORG", "O", "B-ORG"]),
    (["Paris", "is", "the", "capital", "of", "France"],
     ["B-LOC", "O", "O", "O", "O", "B-LOC"]),
    (["Amazon", "opened", "a", "new", "office", "in", "London"],
     ["B-ORG", "O", "O", "O", "O", "O", "B-LOC"]),
    (["Tim", "Cook", "announced", "new", "products", "at", "Apple", "Park"],
     ["B-PER", "I-PER", "O", "O", "O", "O", "B-ORG", "I-ORG"]),
    (["The", "United", "Nations", "met", "in", "Geneva", "last", "week"],
     ["O", "B-ORG", "I-ORG", "O", "O", "B-LOC", "O", "O"]),
    (["Sarah", "Johnson", "traveled", "from", "Berlin", "to", "Tokyo"],
     ["B-PER", "I-PER", "O", "O", "B-LOC", "O", "B-LOC"]),
    (["IBM", "and", "Intel", "partnered", "with", "Stanford", "University"],
     ["B-ORG", "O", "B-ORG", "O", "O", "B-ORG", "I-ORG"]),
    (["David", "Beckham", "visited", "Los", "Angeles", "last", "summer"],
     ["B-PER", "I-PER", "O", "B-LOC", "I-LOC", "O", "O"]),
    (["Netflix", "is", "headquartered", "in", "Los", "Gatos", "California"],
     ["B-ORG", "O", "O", "O", "B-LOC", "I-LOC", "B-LOC"]),
    (["Mark", "Zuckerberg", "created", "Facebook", "at", "Harvard"],
     ["B-PER", "I-PER", "O", "B-ORG", "O", "B-ORG"]),
    (["The", "European", "Union", "headquarters", "are", "in", "Brussels"],
     ["O", "B-ORG", "I-ORG", "O", "O", "O", "B-LOC"]),
    (["Emma", "Watson", "spoke", "at", "the", "United", "Nations", "in", "New", "York"],
     ["B-PER", "I-PER", "O", "O", "O", "B-ORG", "I-ORG", "O", "B-LOC", "I-LOC"]),
    (["Samsung", "shipped", "products", "to", "South", "Korea", "and", "Japan"],
     ["B-ORG", "O", "O", "O", "B-LOC", "I-LOC", "O", "B-LOC"]),
    (["Jeff", "Bezos", "founded", "Amazon", "in", "Seattle"],
     ["B-PER", "I-PER", "O", "B-ORG", "O", "B-LOC"]),
    (["Tesla", "moved", "its", "headquarters", "to", "Austin", "Texas"],
     ["B-ORG", "O", "O", "O", "O", "B-LOC", "B-LOC"]),
    (["Barack", "Obama", "visited", "London", "and", "Paris"],
     ["B-PER", "I-PER", "O", "B-LOC", "O", "B-LOC"]),
    (["Google", "DeepMind", "is", "based", "in", "London"],
     ["B-ORG", "I-ORG", "O", "O", "O", "B-LOC"]),
    (["Alice", "Chen", "works", "for", "Microsoft", "in", "Beijing"],
     ["B-PER", "I-PER", "O", "O", "B-ORG", "O", "B-LOC"]),
]

print(f"  样本数量: {len(synthetic_ner_data)}")
print(f"  标签列表: {label_list}")
print(f"  示例:")
for i in range(3):
    tokens, tags = synthetic_ner_data[i]
    print(f"    tokens: {tokens}")
    print(f"    tags:   {tags}")
    print()

# ============================================================
# 3. Tokenize + Label Alignment
# ============================================================
print("\n--- 3. Tokenize + Label Alignment ---")
print("""
  NER 分词的特殊处理:
    - 一个词可能被拆成多个 subword token
    - 只有第一个 subword 保持原标签, 其余标记为 -100 (忽略)
    - [CLS], [SEP] 等特殊标记也标记为 -100

  例: "Jobs" → ["jo", "##bs"]
      B-PER  → [1, -100]  (只有 "jo" 保留 B-PER)
""")

from transformers import AutoTokenizer

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_and_align_labels(tokens, ner_tags):
    """分词并对齐 NER 标签"""
    # 对每个词进行分词 (is_split_into_words=True)
    tokenized = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=32,
        return_tensors="pt"
    )

    word_ids = tokenized.word_ids(batch_index=0)
    aligned_labels = []

    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            # 特殊标记 ([CLS], [SEP], [PAD])
            aligned_labels.append(-100)
        elif word_idx != previous_word_idx:
            # 词的第一个 subword → 保留原标签
            tag = ner_tags[word_idx]
            aligned_labels.append(label2id[tag])
        else:
            # 词的后续 subword → -100 (忽略)
            aligned_labels.append(-100)
        previous_word_idx = word_idx

    tokenized["labels"] = torch.tensor(aligned_labels)
    return tokenized


# 演示 label alignment
demo_tokens, demo_tags = synthetic_ner_data[0]
demo_result = tokenize_and_align_labels(demo_tokens, demo_tags)
print(f"  原始词:  {demo_tokens}")
print(f"  原始标签: {demo_tags}")
subword_tokens = tokenizer.convert_ids_to_tokens(demo_result["input_ids"][0])
labels = demo_result["labels"].tolist()
print(f"  Subword: {[t for t in subword_tokens if t != '[PAD]']}")
print(f"  对齐标签: {[id2label[l] if l != -100 else -100 for l in labels if l != -100 or labels.index(l) < 15]}")

# ============================================================
# 4. 创建 Dataset
# ============================================================
print("\n--- 4. 创建 NER Dataset ---")


class NERDataset(Dataset):
    """NER 数据集"""

    def __init__(self, data, tokenizer, max_length=32):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.processed = []
        for tokens, tags in data:
            self.processed.append(tokenize_and_align_labels(tokens, tags))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "input_ids": self.processed[idx]["input_ids"].squeeze(0),
            "attention_mask": self.processed[idx]["attention_mask"].squeeze(0),
            "labels": self.processed[idx]["labels"]
        }


full_dataset = NERDataset(synthetic_ner_data, tokenizer)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"  总样本数: {len(full_dataset)}")
print(f"  训练集:   {len(train_dataset)}")
print(f"  验证集:   {len(val_dataset)}")

# ============================================================
# 5. 微调 DistilBERT 进行 Token Classification
# ============================================================
print("\n--- 5. 微调 DistilBERT ---")

from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer
)

model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

print(f"  模型: {model_name}")
print(f"  分类头输出维度: {model.classifier.out_features}")


def compute_ner_metrics(eval_pred):
    """计算 NER 评估指标 (简化版, 不依赖 seqeval)"""
    predictions, labels_data = eval_pred
    predictions = np.argmax(predictions, axis=-1)

    # 过滤掉 -100 (忽略的标签)
    true_labels = []
    pred_labels = []
    for pred_seq, label_seq in zip(predictions, labels_data):
        for p, l in zip(pred_seq, label_seq):
            if l != -100:
                true_labels.append(l)
                pred_labels.append(p)

    # 准确率
    correct = sum(t == p for t, p in zip(true_labels, pred_labels))
    total = len(true_labels)
    accuracy = correct / total if total > 0 else 0

    # 简化的精确率/召回率 (对实体标签)
    entity_labels = set(range(1, len(label_list)))  # 非 O 标签
    tp = sum(1 for t, p in zip(true_labels, pred_labels) if t in entity_labels and t == p)
    fp = sum(1 for t, p in zip(true_labels, pred_labels) if t not in entity_labels and p in entity_labels and p != t)
    fn = sum(1 for t, p in zip(true_labels, pred_labels) if t in entity_labels and p != t)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


training_args = TrainingArguments(
    output_dir="./results_ner",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_steps=3,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none",
    save_total_limit=1,
    no_cuda=True,
    seed=42,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_ner_metrics,
)

print("  开始训练 (CPU, 5 epochs)...")
trainer.train()

# ============================================================
# 6. 评估结果
# ============================================================
print("\n--- 6. 评估结果 ---")

eval_results = trainer.evaluate()
print(f"  验证集评估:")
for key, val in eval_results.items():
    print(f"    {key}: {val:.4f}" if isinstance(val, float) else f"    {key}: {val}")

# 推理演示
print("\n  推理演示:")
test_sentences = [
    ["Larry", "Page", "founded", "Google", "in", "Silicon", "Valley"],
    ["NASA", "launched", "a", "mission", "from", "Houston", "Texas"],
]

model.eval()
for tokens in test_sentences:
    inputs = tokenizer(
        tokens, is_split_into_words=True,
        return_tensors="pt", truncation=True, padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=-1)[0]
    word_ids = inputs.word_ids(batch_index=0)

    print(f"    句子: {' '.join(tokens)}")
    print(f"    实体: ", end="")
    prev_word = None
    for idx, (pred, wid) in enumerate(zip(predictions, word_ids)):
        if wid is not None and wid != prev_word:
            tag = id2label[pred.item()]
            if tag != "O":
                print(f"{tokens[wid]}({tag})", end=" ")
        prev_word = wid
    print()

# ============================================================
# 7. seqeval 库说明
# ============================================================
print("\n--- 7. seqeval 评估库说明 ---")
print("""
  在实际项目中, 推荐使用 seqeval 库进行 NER 评估:

    pip install seqeval

    from seqeval.metrics import classification_report
    from seqeval.scheme import IOB2

    # 输入格式: List[List[str]] (BIO 标签列表)
    y_true = [["O", "B-PER", "I-PER", "O", "B-LOC"]]
    y_pred = [["O", "B-PER", "I-PER", "O", "B-ORG"]]

    print(classification_report(y_true, y_pred, mode='strict', scheme=IOB2))

  seqeval 会自动处理 BIO 标签, 计算实体级别的:
    - Precision (精确率)
    - Recall (召回率)
    - F1-score
    - Support (各类别数量)
""")

# ============================================================
# 8. 总结
# ============================================================
print("\n--- 8. 总结 ---")
print("""
  本节学习了:
  1) NER 任务和 BIO 标注体系
  2) 构造合成 NER 数据
  3) Tokenize + Label Alignment (处理 subword)
  4) 微调 DistilBERT 进行 Token Classification
  5) NER 评估指标 (实体级别 precision/recall/F1)
  6) seqeval 库介绍

  关键点:
    - NER 是 Token Classification 任务, 每个词一个标签
    - subword tokenization 需要对齐标签 (第一个 subword 保留, 其余 -100)
    - 评估使用实体级别的 F1, 而非 token 级别准确率

  下一步: d6_qa_system.py - 问答系统微调
""")
