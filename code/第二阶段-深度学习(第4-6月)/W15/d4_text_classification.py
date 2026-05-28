"""
W15-D4 文本分类微调实战
========================
使用合成数据集, tokenize, 创建 torch Dataset,
用 Trainer API 微调 DistilBERT 进行文本分类, 评估结果
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

print("=" * 60)
print("W15-D4 文本分类微调实战")
print("=" * 60)

# ============================================================
# 1. 准备数据集 (合成 IMDB 风格数据)
# ============================================================
print("\n--- 1. 准备数据集 ---")

# 合成的电影评论数据 (正面=1, 负面=0)
synthetic_reviews = [
    # 正面评论
    ("This movie was absolutely fantastic and I loved every minute of it", 1),
    ("A brilliant performance by the lead actor highly recommended", 1),
    ("One of the best films I have ever seen truly a masterpiece", 1),
    ("Wonderful story with great characters and beautiful cinematography", 1),
    ("I really enjoyed this film it was entertaining from start to finish", 1),
    ("An excellent movie with a compelling plot and outstanding acting", 1),
    ("This is a heartwarming film that the whole family can enjoy", 1),
    ("Amazing special effects and a gripping storyline loved it", 1),
    ("A delightful comedy that had me laughing throughout the entire movie", 1),
    ("Superb direction and writing make this a must see film", 1),
    ("The performances were top notch and the script was incredibly well written", 1),
    ("A visually stunning movie with a powerful emotional core", 1),
    ("I was thoroughly impressed by this film it exceeded all expectations", 1),
    ("Beautiful music and gorgeous scenery make this film unforgettable", 1),
    ("A perfect blend of action and drama that keeps you engaged", 1),
    ("This charming indie film deserves all the praise it has received", 1),
    ("Fantastic writing and brilliant performances throughout", 1),
    ("An inspiring story that will stay with you long after watching", 1),
    ("The chemistry between the leads was electric and wonderful to watch", 1),
    ("A thrilling adventure from beginning to end highly recommended", 1),
    # 负面评论
    ("This movie was terrible and a complete waste of time", 0),
    ("Awful acting and a predictable boring storyline", 0),
    ("One of the worst films I have ever seen do not watch", 0),
    ("Disappointing plot with poor character development and bad dialogue", 0),
    ("I found this movie incredibly boring and dull throughout", 0),
    ("The script was weak and the performances were uninspired", 0),
    ("A complete disaster of a film with no redeeming qualities", 0),
    ("Terrible special effects and a nonsensical plot avoid at all costs", 0),
    ("I fell asleep during this movie it was so incredibly dull", 0),
    ("Poor direction and terrible writing ruin what could have been good", 0),
    ("The acting was wooden and the story made absolutely no sense", 0),
    ("A forgettable movie that offers nothing new or interesting", 0),
    ("I regret spending money on this truly awful movie experience", 0),
    ("Overlong and tedious this film was a chore to sit through", 0),
    ("Bad acting terrible script and horrible editing avoid this", 0),
    ("This sequel was completely unnecessary and terribly executed", 0),
    ("A misguided attempt at filmmaking that fails on every level", 0),
    ("The plot was confusing and the ending was deeply unsatisfying", 0),
    ("Worst movie of the year avoid at all costs truly terrible", 0),
    ("An incoherent mess with no entertainment value whatsoever", 0),
    # 额外正面
    ("What a pleasant surprise this film exceeded my expectations", 1),
    ("Great performances all around especially from the supporting cast", 1),
    ("A thought provoking film that raises important questions", 1),
    ("The soundtrack perfectly complements the stunning visuals", 1),
    ("A triumphant return to form for this acclaimed director", 1),
    # 额外负面
    ("I cannot believe how bad this movie was truly disappointing", 0),
    ("Nothing about this movie works the plot the acting the direction", 0),
    ("A boring and pretentious film with no real substance", 0),
    ("The dialogue was cringe worthy and the plot was nonsensical", 0),
    ("Save your time and money and skip this terrible movie", 0),
]

np.random.seed(42)
np.random.shuffle(synthetic_reviews)

texts = [r[0] for r in synthetic_reviews]
labels = [r[1] for r in synthetic_reviews]

print(f"  样本总数:   {len(texts)}")
print(f"  正面评论:   {sum(labels)} 条")
print(f"  负面评论:   {len(labels) - sum(labels)} 条")
print(f"  示例 (正面): {texts[0][:60]}... → label={labels[0]}")
print(f"  示例 (负面): {texts[1][:60]}... → label={labels[1]}")

# ============================================================
# 2. 使用 AutoTokenizer 分词
# ============================================================
print("\n--- 2. Tokenizer 分词 ---")

from transformers import AutoTokenizer

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 分词演示
sample = texts[0]
tokens = tokenizer(sample, truncation=True, padding="max_length", max_length=64)
print(f"  原文: {sample}")
print(f"  input_ids 长度: {len(tokens['input_ids'])}")
print(f"  前 20 个 tokens: {tokenizer.convert_ids_to_tokens(tokens['input_ids'][:20])}")

# ============================================================
# 3. 创建 PyTorch Dataset
# ============================================================
print("\n--- 3. 创建 PyTorch Dataset ---")


class TextClassificationDataset(Dataset):
    """文本分类数据集"""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


full_dataset = TextClassificationDataset(texts, labels, tokenizer, max_length=64)

# 划分训练/验证集 (80/20)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"  训练集大小: {len(train_dataset)}")
print(f"  验证集大小: {val_size}")
print(f"  样本特征: input_ids {train_dataset[0]['input_ids'].shape}, "
      f"attention_mask {train_dataset[0]['attention_mask'].shape}")

# ============================================================
# 4. 加载预训练模型 & 微调 (Trainer API)
# ============================================================
print("\n--- 4. 微调 DistilBERT ---")

from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label={0: "NEGATIVE", 1: "POSITIVE"},
    label2id={"NEGATIVE": 0, "POSITIVE": 1}
)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  模型总参数: {total_params:,}")
print(f"  可训练参数: {trainable_params:,}")


# 评估函数
def compute_metrics(eval_pred):
    """计算评估指标"""
    from sklearn.metrics import accuracy_score, f1_score
    logits, labels_data = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels_data, predictions)
    f1 = f1_score(labels_data, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}


# 训练参数
training_args = TrainingArguments(
    output_dir="./results_text_cls",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=5,
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none",         # 不上传 wandb 等
    save_total_limit=1,
    no_cuda=True,             # CPU 训练 (小数据集足够)
    seed=42,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print("  开始训练...")
print("  (CPU 模式, 小数据集, 约 3 epochs)")

train_result = trainer.train()

print(f"\n  训练完成!")
print(f"  总训练时间: {train_result.metrics['train_runtime']:.1f} 秒")
print(f"  训练样本数: {train_result.metrics['total_flos']:.0f} FLOPs")

# ============================================================
# 5. 评估结果
# ============================================================
print("\n--- 5. 评估结果 ---")

eval_results = trainer.evaluate()
print(f"  验证集结果:")
for key, val in eval_results.items():
    print(f"    {key}: {val:.4f}" if isinstance(val, float) else f"    {key}: {val}")

# 推理测试
print("\n  推理测试:")
test_texts = [
    "This movie is absolutely wonderful I loved it so much",
    "Terrible film complete waste of time do not watch",
    "A decent film with some good moments but nothing special",
]

model.eval()
for text in test_texts:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    pred = torch.argmax(probs, dim=-1).item()
    label = "正面" if pred == 1 else "负面"
    print(f"    文本: {text}")
    print(f"    预测: {label} (正面概率: {probs[0][1]:.4f}, 负面概率: {probs[0][0]:.4f})")
    print()

# ============================================================
# 6. 手动训练循环 (不用 Trainer)
# ============================================================
print("\n--- 6. 手动训练循环示例 ---")
print("""
  如果不用 Trainer API, 手动训练循环如下:

  from torch.optim import AdamW
  from torch.nn import CrossEntropyLoss

  optimizer = AdamW(model.parameters(), lr=2e-5)
  criterion = CrossEntropyLoss()

  for epoch in range(num_epochs):
      model.train()
      for batch in train_loader:
          optimizer.zero_grad()
          outputs = model(
              input_ids=batch['input_ids'],
              attention_mask=batch['attention_mask'],
              labels=batch['labels']
          )
          loss = outputs.loss
          loss.backward()
          optimizer.step()
""")

# ============================================================
# 7. 总结
# ============================================================
print("\n--- 7. 总结 ---")
print("""
  本节学习了:
  1) 构造文本分类数据集 (合成 IMDB 风格数据)
  2) 使用 AutoTokenizer 分词
  3) 创建 PyTorch Dataset 和 DataLoader
  4) 用 Trainer API 微调 DistilBERT
  5) 评估准确率和 F1 分数
  6) 推理新文本

  关键步骤:
    数据 → tokenize → Dataset → Trainer.train() → evaluate → predict

  下一步: d5_ner_finetune.py - 命名实体识别微调
""")
