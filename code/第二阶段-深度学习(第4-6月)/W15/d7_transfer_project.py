"""
W15-D7 完整迁移学习项目
=========================
完整流程: 选择任务 → 加载数据 → 预处理 → 微调 → 评估 → 保存 → 推理
以新闻主题分类为例
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, random_split

print("=" * 60)
print("W15-D7 完整迁移学习项目")
print("=" * 60)

# ============================================================
# 1. 项目概述: 新闻主题分类
# ============================================================
print("\n--- 1. 项目概述 ---")
print("""
  任务: 新闻文本主题分类 (4 类别)
    - Sports (体育)
    - Technology (科技)
    - Business (商业)
    - Entertainment (娱乐)

  流程:
    1) 构造数据集
    2) 加载预训练 DistilBERT
    3) 预处理 (tokenize)
    4) 微调 (Trainer API)
    5) 评估 (accuracy, F1, confusion matrix)
    6) 保存模型
    7) 加载模型 & 推理
""")

# ============================================================
# 2. 构造数据集
# ============================================================
print("\n--- 2. 构造数据集 ---")

label_names = ["sports", "technology", "business", "entertainment"]
label2id = {name: i for i, name in enumerate(label_names)}
id2label = {i: name for i, name in enumerate(label_names)}

data = [
    # Sports
    ("The team won the championship after an incredible season", "sports"),
    ("He scored three goals in the final match of the tournament", "sports"),
    ("The Olympic games will be held in a new city next year", "sports"),
    ("The basketball player signed a record breaking contract", "sports"),
    ("Tennis star advances to the semifinals after a tough match", "sports"),
    ("The football coach announced his retirement after twenty years", "sports"),
    ("Swimming records were broken at the world championships", "sports"),
    ("The marathon runner finished in under two hours for the first time", "sports"),
    ("Cricket world cup draws millions of viewers worldwide", "sports"),
    ("The baseball team traded their star pitcher before the deadline", "sports"),
    ("Golf tournament sees unexpected winner from qualifying rounds", "sports"),
    ("The hockey team made it to the playoffs after a strong season", "sports"),
    ("Boxing champion defends title in a twelve round decision", "sports"),
    ("The ski resort opened early due to heavy snowfall this winter", "sports"),
    ("Soccer fans celebrated the victory with a massive parade", "sports"),
    # Technology
    ("The new smartphone features an advanced AI powered camera", "technology"),
    ("Cloud computing continues to transform how businesses operate", "technology"),
    ("The tech startup raised millions in series B funding", "technology"),
    ("Artificial intelligence is revolutionizing healthcare diagnostics", "technology"),
    ("The new chip delivers faster processing with lower power consumption", "technology"),
    ("Quantum computing research achieves a major breakthrough", "technology"),
    ("The software update fixes several critical security vulnerabilities", "technology"),
    ("Self driving cars are being tested on public roads in several cities", "technology"),
    ("The social media platform introduced new privacy controls", "technology"),
    ("Robotics company unveils humanoid robot for warehouse operations", "technology"),
    ("The new programming language is gaining popularity among developers", "technology"),
    ("Virtual reality headsets are becoming more affordable and accessible", "technology"),
    ("Blockchain technology is being adopted by major financial institutions", "technology"),
    ("The cybersecurity firm detected a new type of malware attack", "technology"),
    ("Open source software continues to drive innovation in the industry", "technology"),
    # Business
    ("The company reported record profits for the third quarter", "business"),
    ("Stock markets rose sharply after the trade deal was announced", "business"),
    ("The merger will create the largest company in the industry", "business"),
    ("Interest rates were raised by the central bank this morning", "business"),
    ("The startup went public with an impressive initial offering", "business"),
    ("Corporate earnings exceeded analyst expectations this quarter", "business"),
    ("The retail chain announced plans to close fifty stores nationwide", "business"),
    ("Global supply chain disruptions continue to affect manufacturing", "business"),
    ("The airline reported a significant increase in passenger revenue", "business"),
    ("Real estate prices surged in major metropolitan areas", "business"),
    ("The pharmaceutical company acquired a smaller biotech firm", "business"),
    ("Oil prices dropped following the OPEC meeting in Vienna", "business"),
    ("The bank announced a new round of layoffs affecting thousands", "business"),
    ("Venture capital funding reached new heights in the technology sector", "business"),
    ("The automotive industry is shifting toward electric vehicle production", "business"),
    # Entertainment
    ("The new movie broke box office records on its opening weekend", "entertainment"),
    ("The popular TV series was renewed for another season", "entertainment"),
    ("The Grammy Awards ceremony featured incredible live performances", "entertainment"),
    ("The bestselling novel is being adapted into a major film", "entertainment"),
    ("The video game sold millions of copies within the first week", "entertainment"),
    ("The music festival attracted record crowds this summer", "entertainment"),
    ("The streaming service added several new original series", "entertainment"),
    ("A famous actor announced his retirement from the film industry", "entertainment"),
    ("The Broadway show received multiple Tony Award nominations", "entertainment"),
    ("The comedian is launching a new podcast next month", "entertainment"),
    ("The reality TV show has become a cultural phenomenon", "entertainment"),
    ("A classic rock band announced their farewell world tour", "entertainment"),
    ("The animated film received critical acclaim for its stunning visuals", "entertainment"),
    ("The celebrity couple announced their engagement on social media", "entertainment"),
    ("The theater production sold out all tickets within hours", "entertainment"),
]

np.random.seed(42)
np.random.shuffle(data)

texts = [d[0] for d in data]
labels = [label2id[d[1]] for d in data]

print(f"  总样本数: {len(texts)}")
for name in label_names:
    count = labels.count(label2id[name])
    print(f"    {name:20s}: {count} 条")

# ============================================================
# 3. 预处理 (Tokenize)
# ============================================================
print("\n--- 3. 预处理 ---")

from transformers import AutoTokenizer

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)


class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


full_dataset = NewsDataset(texts, labels, tokenizer)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"  训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")

# ============================================================
# 4. 加载预训练模型
# ============================================================
print("\n--- 4. 加载预训练模型 ---")

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label_names),
    id2label=id2label,
    label2id=label2id
)

total_params = sum(p.numel() for p in model.parameters())
print(f"  模型: {model_name}")
print(f"  参数量: {total_params:,}")
print(f"  分类类别数: {len(label_names)}")

# ============================================================
# 5. 微调
# ============================================================
print("\n--- 5. 微调 ---")

from transformers import TrainingArguments, Trainer

try:
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def compute_metrics(eval_pred):
    logits, labels_data = eval_pred
    predictions = np.argmax(logits, axis=-1)
    result = {}
    if HAS_SKLEARN:
        result["accuracy"] = accuracy_score(labels_data, predictions)
        result["f1"] = f1_score(labels_data, predictions, average="weighted")
    else:
        correct = (predictions == labels_data).mean()
        result["accuracy"] = correct
    return result


training_args = TrainingArguments(
    output_dir="./results_news_cls",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=5,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
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
    compute_metrics=compute_metrics,
)

print("  开始训练...")
trainer.train()

# ============================================================
# 6. 评估
# ============================================================
print("\n--- 6. 评估 ---")

eval_results = trainer.evaluate()
print("  验证集结果:")
for key, val in eval_results.items():
    print(f"    {key}: {val:.4f}" if isinstance(val, float) else f"    {key}: {val}")

# 详细分类报告
print("\n  分类报告:")
model.eval()
all_preds = []
all_labels = []
for i in range(len(val_dataset)):
    sample = val_dataset[i]
    with torch.no_grad():
        input_ids = sample["input_ids"].unsqueeze(0)
        attention_mask = sample["attention_mask"].unsqueeze(0)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    pred = torch.argmax(outputs.logits, dim=-1).item()
    all_preds.append(pred)
    all_labels.append(sample["labels"].item())

if HAS_SKLEARN:
    print(classification_report(all_labels, all_preds, target_names=label_names))
else:
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    print(f"  准确率: {correct}/{len(all_labels)} = {correct/len(all_labels):.4f}")

# 混淆矩阵 (手动)
print("  混淆矩阵:")
print(f"  {'':>15s}", end="")
for name in label_names:
    print(f"  {name[:8]:>8s}", end="")
print()
for i, true_name in enumerate(label_names):
    print(f"  {true_name:>15s}", end="")
    for j in range(len(label_names)):
        count = sum(1 for p, l in zip(all_preds, all_labels) if l == i and p == j)
        print(f"  {count:>8d}", end="")
    print()

# ============================================================
# 7. 保存模型
# ============================================================
print("\n--- 7. 保存模型 ---")

save_dir = "./saved_news_classifier"
trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)

# 保存标签映射
label_info = {
    "label_names": label_names,
    "label2id": label2id,
    "id2label": {str(k): v for k, v in id2label.items()},
    "model_name": model_name
}
with open(os.path.join(save_dir, "label_info.json"), "w", encoding="utf-8") as f:
    json.dump(label_info, f, indent=2, ensure_ascii=False)

print(f"  模型已保存到: {save_dir}")
print(f"  文件列表: {os.listdir(save_dir)}")

# ============================================================
# 8. 加载模型 & 推理
# ============================================================
print("\n--- 8. 加载模型 & 推理 ---")

# 加载
loaded_tokenizer = AutoTokenizer.from_pretrained(save_dir)
loaded_model = AutoModelForSequenceClassification.from_pretrained(save_dir)

with open(os.path.join(save_dir, "label_info.json"), "r", encoding="utf-8") as f:
    loaded_info = json.load(f)

loaded_id2label = loaded_info["id2label"]

print("  模型加载成功!")
print(f"  标签映射: {loaded_id2label}")

# 推理
test_news = [
    "The quarterback threw four touchdown passes in the superbowl",
    "Apple unveiled their newest laptop with a revolutionary chip",
    "The stock market experienced a significant rally today",
    "The highly anticipated sequel opens in theaters this weekend",
    "A new AI model can generate realistic images from text descriptions",
    "The soccer match ended in a dramatic penalty shootout",
]

print("\n  推理结果:")
loaded_model.eval()
for news in test_news:
    inputs = loaded_tokenizer(news, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = loaded_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    pred = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][pred].item()
    label = loaded_id2label[str(pred)]

    # 显示所有类别概率
    prob_str = ", ".join(
        f"{loaded_id2label[str(i)]}: {probs[0][i]:.3f}"
        for i in range(len(loaded_id2label))
    )
    print(f"    新闻: {news}")
    print(f"    预测: {label} (置信度: {confidence:.4f})")
    print(f"    分布: [{prob_str}]")
    print()

# ============================================================
# 9. 项目总结
# ============================================================
print("\n--- 9. 项目总结 ---")
print("""
  完整迁移学习流程:
    1) 任务定义: 4 类新闻分类 (sports, technology, business, entertainment)
    2) 数据准备: 构造/加载数据集
    3) 预处理:   AutoTokenizer 分词
    4) 加载模型: AutoModelForSequenceClassification (DistilBERT)
    5) 微调训练: Trainer API (5 epochs)
    6) 评估:     accuracy, F1, classification report, confusion matrix
    7) 保存:     trainer.save_model() + tokenizer.save_pretrained()
    8) 加载推理: AutoModel.from_pretrained(save_dir)

  迁移学习的优势:
    - 不需要海量数据 (小数据集即可)
    - 不需要从头训练 (利用预训练知识)
    - 训练速度快 (只微调几轮)
    - 效果好 (预训练模型已学到丰富语言知识)

  本周 (W15) 学习内容回顾:
    D1: HuggingFace 生态概览
    D2: 分词器 (BPE, WordPiece, SentencePiece)
    D3: Pipeline 推理 (6 种 NLP 任务)
    D4: 文本分类微调
    D5: NER 微调
    D6: 问答系统微调
    D7: 完整迁移学习项目

  下周 (W16): 高级训练技术
""")
