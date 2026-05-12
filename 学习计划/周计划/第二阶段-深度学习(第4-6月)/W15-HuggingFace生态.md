# 第15周 - HuggingFace生态

> 本周学习HuggingFace生态系统——深度学习领域的"GitHub"。掌握Transformers、Datasets、Tokenizers库的使用，以及微调预训练模型完成实际任务。

---

## 一、本周目标

1. 了解HuggingFace生态的全貌（Transformers/Datasets/Tokenizers/Hub/Accelerate）
2. 深入理解BPE/WordPiece/SentencePiece分词算法的区别和原理
3. 掌握预训练模型的加载和pipeline一键推理
4. 完成文本分类微调（BERT）和NER微调的完整流程
5. 独立完成一个迁移学习实战项目

---

## 二、时间安排

### 工作日（每晚2小时）

| 日期 | 主题 | 时间分配 |
|------|------|----------|
| Day 1 (周一) | HuggingFace生态概览 | 理论40min + 环境搭建80min |
| Day 2 (周二) | Tokenizer深入 | 理论60min + 代码实现60min |
| Day 3 (周三) | 预训练模型加载与pipeline | 理论30min + 代码实践90min |
| Day 4 (周四) | 文本分类微调 | 理论40min + 代码实现80min |
| Day 5 (周五) | NER微调/问答微调 | 理论50min + 代码实现70min |

### 周末（6-8小时）

| 日期 | 主题 | 时间分配 |
|------|------|----------|
| Day 6 (周六) | 问答系统微调 | 理论60min + 代码实现120min |
| Day 7 (周日) | 实战项目7 - 迁移学习 | 项目实现240min |

---

## 三、详细学习内容

### Day 1: HuggingFace生态概览

#### 1. 生态全景

```
HuggingFace生态系统:
├── Transformers    # 模型+分词器库（核心）
├── Datasets        # 数据集加载和处理
├── Tokenizers      # 高性能分词器
├── Hub             # 模型托管和共享平台
├── Accelerate      # 分布式训练抽象层
├── PEFT            # 参数高效微调（LoRA等）
├── Diffusers       # 扩散模型（图像生成）
├── Gradio/Spaces   # 模型Demo部署
└── Safetensors     # 安全的模型存储格式
```

#### 2. 环境安装

```bash
# 基础安装
pip install transformers datasets tokenizers accelerate

# 完整安装
pip install transformers[torch] datasets tokenizers accelerate
pip install evaluate seqeval  # 评估指标
pip install wandb tensorboard  # 实验管理

# 验证安装
python -c "from transformers import pipeline; print('OK')"
```

#### 3. Transformers库核心概念

```python
from transformers import (
    AutoModel,           # 自动选择模型类
    AutoTokenizer,       # 自动选择分词器
    AutoModelForSequenceClassification,  # 文本分类模型
    AutoModelForTokenClassification,     # Token分类(NER)模型
    AutoModelForQuestionAnswering,       # 问答模型
    AutoModelForCausalLM,                # 语言生成模型
    pipeline,            # 一键推理pipeline
    Trainer,             # 训练器
    TrainingArguments,   # 训练参数
)
```

#### 4. Datasets库

```python
from datasets import load_dataset, DatasetDict

# 加载数据集
dataset = load_dataset("imdb")  # IMDB情感分析
# DatasetDict({train: Dataset({...}), test: Dataset({...})})

# 常用数据集
# load_dataset("squad")     # 问答
# load_dataset("glue", "sst2")  # 文本分类
# load_dataset("conll2003") # NER
# load_dataset("wmt16", "de-en")  # 翻译

# 数据处理
dataset = dataset.map(lambda x: {'text_len': len(x['text'])})
dataset = dataset.filter(lambda x: x['text_len'] > 100)
dataset = dataset.shuffle(seed=42)
dataset = dataset.train_test_split(test_size=0.2)
```

---

### Day 2: Tokenizer深入

#### 1. BPE (Byte Pair Encoding) 算法

BPE是最常用的子词分词算法，GPT系列使用。

**算法流程**：
```
初始: 将文本拆为字符序列，统计相邻字符对的频率
循环: 合并频率最高的字符对 → 形成新的子词
直到: 达到目标词表大小

示例:
初始词表: {l, o, w, e, r, n, w, s, t, o, i, d}
文本:    l o w, l o w e r, n e w e s t, w i d e s t

第1轮: 最频繁对 'e'+'s'→'es'，词表加入'es'
第2轮: 最频繁对 'es'+'t'→'est'，词表加入'est'
...
```

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# 训练BPE分词器
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(
    vocab_size=30000,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)

# 在文本上训练
files = ["train.txt"]
tokenizer.train(files, trainer)

# 使用
output = tokenizer.encode("Hello, how are you?")
print(output.tokens)    # ['Hello', ',', 'how', 'are', 'you', '?']
print(output.ids)       # [1234, 56, 789, 101, 112, 134]
```

#### 2. WordPiece (BERT使用)

WordPiece与BPE类似，但选择合并对的标准不同：
- BPE：选择频率最高的对
- WordPiece：选择使语言模型似然增加最大的对

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 分词
tokens = tokenizer.tokenize("I love natural language processing")
print(tokens)
# ['i', 'love', 'natural', 'language', 'processing']

# 带##前缀的子词（非首字）
tokens = tokenizer.tokenize("unhappiness")
print(tokens)
# ['un', '##ha', '##pp', '##in', '##ess']
# ## 表示这是被切分的子词（非第一个子词）
```

#### 3. SentencePiece (LLaMA使用)

SentencePiece是语言无关的分词工具，直接从原始文本训练，不需要预分词。

```python
from transformers import LlamaTokenizer

# LLaMA使用的SentencePiece分词器
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 特点
print(tokenizer.vocab_size)  # 32000
print(tokenizer.bos_token)   # <s> (开始)
print(tokenizer.eos_token)   # </s> (结束)
print(tokenizer.unk_token)   # <unk> (未知)
```

#### 4. Tokenizer核心API

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# 编码（文本→ID）
# 单条文本
encoded = tokenizer("Hello world")
print(encoded.keys())  # dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])
print(encoded['input_ids'])       # [101, 7592, 2088, 102]
print(encoded['attention_mask'])  # [1, 1, 1, 1]

# ���量编码
batch = tokenizer(
    ["Hello world", "How are you?"],
    padding=True,          # 填充到相同长度
    truncation=True,       # 截断到最大长度
    max_length=128,        # 最大长度
    return_tensors='pt'    # 返回PyTorch tensor
)
print(batch['input_ids'].shape)       # (2, 4) 或更长
print(batch['attention_mask'].shape)  # (2, 4) 或更长

# 解码（ID→文本）
decoded = tokenizer.decode(encoded['input_ids'])
print(decoded)  # "[CLS] hello world [SEP]"
```

---

### Day 3: 预训练模型加载

#### 1. AutoModel系列

```python
from transformers import AutoModel, AutoTokenizer
import torch

# 加载模型和分词器
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 基础推理
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# 输出
print(outputs.last_hidden_state.shape)  # (1, seq_len, 768)
print(outputs.pooler_output.shape)      # (1, 768) — [CLS]的表示
```

AutoModel vs 具体模型类：
```python
# AutoModel：自动推断模型类型
model = AutoModel.from_pretrained("bert-base-uncased")

# 具体类：明确知道模型类型
from transformers import BertModel, BertForSequenceClassification
model = BertModel.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

# AutoModel会自动根据配置选择正确的类
# 好处：换模型时只改名字，代码不用改
```

#### 2. pipeline一键推理

```python
from transformers import pipeline

# 1. 情感分析
sentiment = pipeline("sentiment-analysis")
result = sentiment("I love this movie!")
# [{'label': 'POSITIVE', 'score': 0.9998}]

# 2. 命名实体识别(NER)
ner = pipeline("ner", grouped_entities=True)
result = ner("Hugging Face is based in New York City")
# [{'entity_group': 'ORG', 'word': 'Hugging Face', ...},
#  {'entity_group': 'LOC', 'word': 'New York City', ...}]

# 3. 问答
qa = pipeline("question-answering")
result = qa(
    question="What is the capital of France?",
    context="France is a country. Its capital is Paris."
)
# {'answer': 'Paris', 'score': 0.97, ...}

# 4. 文本生成
generator = pipeline("text-generation", model="gpt2")
result = generator("Once upon a time", max_length=50)
# [{'generated_text': 'Once upon a time...'}]

# 5. 翻译
translator = pipeline("translation_en_to_fr")
result = translator("Hello, how are you?")
# [{'translation_text': 'Bonjour, comment allez-vous?'}]

# 6. 文本摘要
summarizer = pipeline("summarization")
result = summarizer(long_text, max_length=130, min_length=30)
# [{'summary_text': '...'}]

# 7. 零样本分类
classifier = pipeline("zero-shot-classification")
result = classifier(
    "This is a tutorial about deep learning",
    candidate_labels=["education", "politics", "technology"]
)
# {'labels': ['technology', 'education', 'politics'], 'scores': [...]}
```

#### 3. 模型配置和参数量

```python
# 查看配置
from transformers import AutoConfig
config = AutoConfig.from_pretrained("bert-base-uncased")
print(config)
# BertConfig {
#   "hidden_size": 768,        # d_model
#   "num_attention_heads": 12,  # n_heads
#   "num_hidden_layers": 12,    # n_layers
#   "intermediate_size": 3072,  # d_ff
#   "vocab_size": 30522,
#   "max_position_embeddings": 512
# }

# 查看参数量
model = AutoModel.from_pretrained("bert-base-uncased")
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params:,}")  # 约 109M
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"可训练参数: {trainable_params:,}")
```

---

### Day 4: 文本分类微调

#### 1. 完整微调流程

```python
"""BERT文本分类微调 - 以IMDB情感分析为例"""
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline,
)
from datasets import load_dataset
import evaluate
import numpy as np

# ==================== 1. 准备数据 ====================
model_name = "bert-base-uncased"
dataset = load_dataset("imdb")

# 子采样加速实验（正式训练用全部数据）
train_dataset = dataset["train"].shuffle(seed=42).select(range(5000))
test_dataset = dataset["test"].shuffle(seed=42).select(range(1000))

# ==================== 2. Tokenize ====================
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# 设置格式
train_dataset = train_dataset.remove_columns(["text"])
test_dataset = test_dataset.remove_columns(["text"])
train_dataset.set_format("torch")
test_dataset.set_format("torch")

# ==================== 3. 加载模型 ====================
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2  # 二分类
)

# ==================== 4. 评估指标 ====================
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

# ==================== 5. 训练配置 ====================
training_args = TrainingArguments(
    output_dir="./results/imdb",
    learning_rate=2e-5,          # 微调学习率，比从头训练小很多
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,          # 微调通常3-5个epoch
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    warmup_ratio=0.1,
    logging_dir="./logs",
    logging_steps=50,
    fp16=torch.cuda.is_available(),  # 自动使用混合精度
)

# ==================== 6. 训练 ====================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# ==================== 7. 评估 ====================
results = trainer.evaluate()
print(f"测试准确率: {results['eval_accuracy']:.4f}")

# ==================== 8. 保存模型 ====================
trainer.save_model("./models/imdb-bert")
tokenizer.save_pretrained("./models/imdb-bert")
```

#### 2. 微调学习率选择

为什么微调学习率(2e-5)比从头训练(1e-3)小很多？

- 预训练模型已经学到了很好的特征表示
- 大学习率会破坏已经学到的知识（灾难性遗忘）
- 微调只需要小幅调整，不需要大幅改变参数

常见学习率范围：
- 微调BERT: 2e-5 ~ 5e-5
- 微调GPT: 1e-5 ~ 5e-5
- 从头训练: 1e-4 ~ 1e-3

---

### Day 5: Token分类/NER微调

#### 1. 命名实体识别任务

NER (Named Entity Recognition) 是Token级别的分类任务——为每个token预测一个实体标签。

BIO标注格式：
```
文本:   John  lives  in  New   York  City
标签:   B-PER O      O   B-LOC I-LOC I-LOC
```

- **B-X**：实体X的开始(Begin)
- **I-X**：实体X的内部(Inside)
- **O**：不属于任何实体(Outside)

#### 2. NER微调流程

```python
"""BERT NER微调 - 以CoNLL-2003为例"""
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
import evaluate
import numpy as np

# 加载数据
dataset = load_dataset("conll2003")
label_list = dataset["train"].features["ner_tags"].feature.names
# ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

# Tokenize + 标签对齐
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_and_align_labels(examples):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True  # 输入已经是分好词的
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                # 特殊token ([CLS], [SEP]) 用 -100 忽略
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # 每个词的第一个子token使用原始标签
                label_ids.append(label[word_idx])
            else:
                # 被切分的子词使用原始标签（也可以用-100忽略）
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized["labels"] = labels
    return tokenized

tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# 加载模型
model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label_list)
)

# 评估指标
seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # 移除-100（特殊token）
    true_predictions = [
        [label_list[p] for p, l in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for p, l in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# 训练
training_args = TrainingArguments(
    output_dir="./results/ner",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
```

---

### Day 6: 问答系统微调

#### 1. SQuAD数据集格式

```json
{
  "context": "The Normans were a people who ...",
  "question": "In what country is Normandy located?",
  "answers": {
    "text": ["France", "France"],
    "answer_start": [159, 159]
  }
}
```

#### 2. 抽取式问答

抽取式问答预测答案在原文中的起始和结束位置：

```python
"""BERT问答微调 - SQuAD"""
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
)
from datasets import load_dataset

# 加载数据
dataset = load_dataset("squad")

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_squad(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",  # 只截断context
        stride=128,                # 重叠窗口
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # 获取答案的起始和结束位置
    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        # 找到token级别的start和end
        sequence_ids = inputs.sequence_ids(i)
        idx = 0
        while sequence_ids[idx] != 1:  # 找到context开始位置
            idx += 1
        context_start = idx
        while idx < len(sequence_ids) and sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # 如果答案不在当前窗口中
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            token_start = context_start
            while token_start <= context_end and offset[token_start][0] <= start_char:
                token_start += 1
            start_positions.append(token_start - 1)

            token_end = context_end
            while token_end >= context_start and offset[token_end][1] >= end_char:
                token_end -= 1
            end_positions.append(token_end + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

tokenized_dataset = dataset.map(preprocess_squad, batched=True,
                                  remove_columns=dataset["train"].column_names)

# 模型和训练
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

training_args = TrainingArguments(
    output_dir="./results/qa",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=DefaultDataCollator(),
)

trainer.train()

# 推理测试
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
result = qa_pipeline(
    question="What is the capital of France?",
    context="France is a country in Europe. The capital of France is Paris, "
            "which is known for the Eiffel Tower."
)
print(f"答案: {result['answer']}, 置信度: {result['score']:.4f}")
```

---

### Day 7: 实战项目7 - 迁移学习

#### 项目选择：BERT文本分类（推荐）或 ViT图像分类

以下以BERT文本分类为例，完整流程：

```python
"""项目7：BERT文本分类迁移学习 - 完整流程"""
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset, DatasetDict
import evaluate
import numpy as np

# ==================== 配置 ====================
CONFIG = {
    "model_name": "bert-base-uncased",
    "dataset_name": "imdb",
    "max_length": 256,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "epochs": 3,
    "output_dir": "./models/imdb-bert-final",
}

# ==================== Step 1: 数据准备 ====================
print("Step 1: 加载数据...")
raw_dataset = load_dataset(CONFIG["dataset_name"])

# 划分验证集
split = raw_dataset["train"].train_test_split(test_size=0.2, seed=42)
dataset = DatasetDict({
    "train": split["train"],
    "validation": split["test"],
    "test": raw_dataset["test"],
})
print(f"训练集: {len(dataset['train'])}, 验证集: {len(dataset['validation'])}, 测试集: {len(dataset['test'])}")

# ==================== Step 2: 分词 ====================
print("Step 2: 分词处理...")
tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

def tokenize(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=CONFIG["max_length"]
    )

tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
tokenized.set_format("torch")

# ==================== Step 3: Baseline ====================
print("Step 3: Baseline评估（预训练模型直接推理）...")
baseline_classifier = pipeline(
    "sentiment-analysis",
    model=CONFIG["model_name"]
)
# 预训练BERT没有针对IMDB微调，直接推理效果有限
# 这个baseline可以帮助理解微调的价值

# ==================== Step 4: 微调 ====================
print("Step 4: 微调BERT...")
model = AutoModelForSequenceClassification.from_pretrained(
    CONFIG["model_name"],
    num_labels=2
)

accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir=CONFIG["output_dir"],
    learning_rate=CONFIG["learning_rate"],
    per_device_train_batch_size=CONFIG["batch_size"],
    per_device_eval_batch_size=CONFIG["batch_size"] * 2,
    num_train_epochs=CONFIG["epochs"],
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    warmup_ratio=0.1,
    fp16=torch.cuda.is_available(),
    logging_dir=f"{CONFIG['output_dir']}/logs",
    report_to="tensorboard",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# ==================== Step 5: 评估 ====================
print("Step 5: 测试集评估...")
test_results = trainer.evaluate(tokenized["test"])
print(f"测试集准确率: {test_results['eval_accuracy']:.4f}")

# ==================== Step 6: 与Baseline对比 ====================
print("\n===== 结果对比 =====")
print(f"微调BERT准确率: {test_results['eval_accuracy']:.4f}")
print(f"Baseline(未微调): 需要运行baseline推理对比")

# ==================== Step 7: 保存模型 ====================
print("Step 7: 保存模型...")
trainer.save_model(CONFIG["output_dir"])
tokenizer.save_pretrained(CONFIG["output_dir"])
print(f"模型已保存到 {CONFIG['output_dir']}")

# ==================== Step 8: 加载模型推理 ====================
print("Step 8: 加载模型并推理...")
loaded_model = AutoModelForSequenceClassification.from_pretrained(CONFIG["output_dir"])
loaded_tokenizer = AutoTokenizer.from_pretrained(CONFIG["output_dir"])

classifier = pipeline("sentiment-analysis", model=loaded_model, tokenizer=loaded_tokenizer)

test_texts = [
    "This movie was absolutely wonderful! Great acting and story.",
    "Terrible film. Waste of time and money.",
    "An average movie with some good moments.",
]

for text in test_texts:
    result = classifier(text)
    label = "正面" if result[0]['label'] == 'LABEL_1' else "负面"
    print(f"文本: {text[:50]}... → {label} ({result[0]['score']:.4f})")
```

---

## 四、代码练习

### Day 3：5种pipeline推理

```python
"""任务：用pipeline做5种不同任务的推理"""
from transformers import pipeline

# 1. 情感分析
print("=== 情感分析 ===")
sentiment = pipeline("sentiment-analysis")
print(sentiment("This course is incredibly helpful!"))

# 2. NER
print("\n=== 命名实体识别 ===")
ner = pipeline("ner", grouped_entities=True)
print(ner("Elon Musk founded SpaceX in Hawthorne, California."))

# 3. 问答
print("\n=== 问答 ===")
qa = pipeline("question-answering")
print(qa(question="Who founded SpaceX?",
         context="Elon Musk founded SpaceX in 2002."))

# 4. 文本生成
print("\n=== 文本生成 ===")
generator = pipeline("text-generation", model="gpt2")
print(generator("The future of AI is", max_length=30, num_return_sequences=2))

# 5. 零样本分类
print("\n=== 零样本分类 ===")
classifier = pipeline("zero-shot-classification")
print(classifier("This is a great Python programming tutorial.",
                 candidate_labels=["education", "entertainment", "technology", "sports"]))
```

### Day 4：微调BERT二分类

（见Day 4详细内容中的完整代码）

### Day 7：完成项目7

（见Day 7详细内容中的完整项目代码）

---

## 五、本周产出

### 周末交付物

1. **BERT文本分类微调**：完整的IMDB情感分析微调代码和模型 `project7_bert_classification/`
2. **NER微调**：CoNLL-2003 NER微调代码 `ner_finetune.py`
3. **Pipeline练习**：5种任务的pipeline推理代码 `pipeline_demo.py`
4. **模型对比报告**：微调前后的性能对比

---

## 六、自测题

### 题1：BPE和WordPiece的区别？

<details>
<summary>参考答案</summary>

相同点：
- 都是子词分词算法
- 都通过合并操作逐步构建词表
- 都能处理未登录词（OOV）

不同点：
1. **合并策略不同**：
   - BPE：选择频率最高的相邻字符对合并
   - WordPiece：选择使语言模型似然增加最大的对合并

2. **标记方式不同**：
   - BPE：子词直接拼接，无特殊标记
   - WordPiece：非首子词用`##`前缀标记（如 `un##ha##ppiness`）

3. **使用模型**：
   - BPE：GPT-2/3/4、LLaMA（变体）
   - WordPiece：BERT、DistilBERT

4. **实际差异**：在大多数任务上两者性能相近，选择更多取决于模型设计者的偏好。
</details>

### 题2：AutoModel和具体模型类（BertModel）的区别？

<details>
<summary>参考答案</summary>

- **AutoModel**：根据模型名称自动选择正确的类。内部通过读取config.json来确定模型类型。
  ```python
  # 这两行代码效果完全相同
  model = AutoModel.from_pretrained("bert-base-uncased")
  model = BertModel.from_pretrained("bert-base-uncased")
  ```

- **具体模型类**：明确指定模型类型。

选择建议：
- 写通用代码、做实验对比时用AutoModel（换模型只改名字）
- 确定模型类型后可以用具体类（IDE提示更好、类型检查更方便）
- 生产环境推荐具体类（避免自动推断的不确定性）
</details>

### 题3：Trainer API的核心参数有哪些？

<details>
<summary>参考答案</summary>

TrainingArguments核心参数：

**学习相关**：
- `learning_rate`：学习率（微调常用2e-5）
- `num_train_epochs`：训练轮数（微调常用3-5）
- `weight_decay`：权重衰减（0.01）
- `warmup_ratio`：预热比例（0.1）

**批次相关**：
- `per_device_train_batch_size`：每GPU训练batch
- `per_device_eval_batch_size`：每GPU评估batch
- `gradient_accumulation_steps`：梯度累积步数

**保存和评估**：
- `evaluation_strategy`：评估策略（"epoch"/"steps"）
- `save_strategy`：保存策略
- `load_best_model_at_end`：训练结束加载最优模型
- `metric_for_best_model`：最优模型的标准

**加速相关**：
- `fp16`：是否使用混合精度
- `dataloader_num_workers`：数据加载进程数

Trainer还支持自定义compute_metrics回调函数来计算评估指标。
</details>

### 题4：微调时学习率一般设多少？为什么比从头训练小？

<details>
<summary>参考答案</summary>

微调学习率通常为2e-5到5e-5，比从头训练的1e-3到1e-4小约100倍。

原因：
1. **预训练模型已学到了通用知识**：特征提取能力已经很强，微调只需要小幅调整
2. **防止灾难性遗忘**：大学习率会大幅改变参数，破坏预训练学到的通用特征
3. **下游数据量通常较小**：大数据+大学习率容易过拟合
4. **微调是精细调整**：类似于"雕刻"而非"粗加工"

经验规则：
- 微调epoch越少，学习率可以稍大
- 数据量越大，学习率可以稍大
- 底层（靠近输入的层）学习率可以更小，顶层（靠近输出的层）可以稍大（分层学习率）
</details>

### 题5：如何判断微调是否过拟合？

<details>
<summary>参考答案</summary>

过拟合的信号：
1. **训练loss持续下降，验证loss开始上升**：最经典的信号
2. **训练准确率很高，验证准确率停滞或下降**
3. **训练集和验证集的指标差距越来越大**

应对措施：
1. **Early Stopping**：设置 `load_best_model_at_end=True`，监控验证集指标
2. **减小epoch数**：微调通常3-5个epoch足够
3. **增加正则化**：增大weight_decay（0.01→0.1），增加dropout
4. **数据增强**：回译、同义词替换等
5. **减小模型**：用DistilBERT代替BERT，或冻结部分层
6. **增加数据量**：最有效的办法

```python
# 在Trainer中设置Early Stopping
from transformers import EarlyStoppingCallback

trainer = Trainer(
    ...,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)
```
</details>

---

## 七、Java开发者提示

### 1. HuggingFace Hub vs Maven Central

| HuggingFace概念 | Java生态类比 |
|---------------|-------------|
| Hub | Maven Central / NPM |
| 模型仓库 | Maven artifact |
| from_pretrained() | Maven依赖下载 |
| 模型版本(tag) | 版本号(1.0.0) |
| README/Model Card | README/Javadoc |
| pipeline | Spring Boot Starter |

```java
// Java中引入依赖
<dependency>
    <groupId>com.example</groupId>
    <artifactId>bert-base-uncased</artifactId>
    <version>1.0</version>
</dependency>

// Python中加载模型（本质一样）
model = AutoModel.from_pretrained("bert-base-uncased")
```

### 2. AutoModel vs 依赖注入

AutoModel的自动推断和Spring的依赖注入思想相似：
```java
// Spring的自动装配
@Autowired
private Model model;  // 自动注入合适的实现

// HuggingFace的自动模型
model = AutoModel.from_pretrained("bert-base-uncased")  # 自动选择BertModel
```

好处一致：解耦具体实现，换模型/换实现只需要改配置。

### 3. Trainer vs Spring Boot AutoConfiguration

Trainer封装了训练的"样板代码"，就像Spring Boot的自动配置：

```java
// 没有Spring Boot：手动配置一切
DataSource ds = new HikariDataSource();
EntityManager em = EntityManagerFactory.createEntityManager();
TransactionManager tm = new TransactionManager(em);
// ... 大量样板代码

// 有Spring Boot：自动配置
@SpringBootApplication
public class App { ... }

// 没有Trainer：手动写训练循环
for epoch in range(epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()
    evaluate(...)

// 有Trainer：配置驱动
trainer = Trainer(model=model, args=training_args, ...)
trainer.train()
```

### 4. 微调 vs Fork+定制

微调预训练模型和从开源项目Fork后定制开发完全类似：
1. **选择基础项目**：选择合适的预训练模型（BERT/GPT/RoBERTa）
2. **Fork**：加载预训练权重
3. **定制修改**：替换分类头，微调参数
4. **测试**：在目标任务上评估
5. **发布**：推送到HuggingFace Hub

### 5. Tokenizer vs 序列化/反序列化

Tokenizer就是文本的序列化器：
```java
// Java序列化
String json = objectMapper.writeValueAsString(obj);   // 序列化
Object obj = objectMapper.readValue(json, MyClass.class); // 反序列化

// Tokenizer
ids = tokenizer.encode("Hello world")      // 文本 → 数字序列
text = tokenizer.decode(ids)               // 数字序列 → 文本
```

BPE/WordPiece算法的选择类似于JSON/Protobuf序列化格式的选择——不同场景选择不同方案。
