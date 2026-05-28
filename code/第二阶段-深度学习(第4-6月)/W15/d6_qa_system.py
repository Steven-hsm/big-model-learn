"""
W15-D6 问答系统 (Question Answering)
======================================
问答任务介绍, 构造 SQuAD 风格数据,
微调 BERT 进行抽取式问答, 自定义问题推理
"""

import numpy as np
import torch
from torch.utils.data import Dataset, random_split

print("=" * 60)
print("W15-D6 问答系统 (Question Answering)")
print("=" * 60)

# ============================================================
# 1. 问答任务介绍
# ============================================================
print("\n--- 1. 问答任务介绍 ---")
print("""
  抽取式问答 (Extractive QA):
    输入: context (上下文) + question (问题)
    输出: 从 context 中抽取的答案 span (start, end)

  示例:
    Context: "HuggingFace was founded in 2016 in New York."
    Question: "When was HuggingFace founded?"
    Answer: "2016" (start=28, end=32)

  模型输出两个位置:
    - start_logits: 每个 token 是答案起始位置的概率
    - end_logits:   每个 token 是答案结束位置的概率

  训练目标: 最小化 start + end 位置的交叉熵损失

  经典数据集: SQuAD 1.1, SQuAD 2.0, Natural Questions
""")

# ============================================================
# 2. 构造 SQuAD 风格数据
# ============================================================
print("\n--- 2. 构造 SQuAD 风格数据 ---")

# 合成 QA 数据: (context, question, answer_text, answer_start)
synthetic_qa_data = [
    (
        "Python was created by Guido van Rossum and first released in 1991.",
        "Who created Python?",
        "Guido van Rossum",
        22
    ),
    (
        "Python was created by Guido van Rossum and first released in 1991.",
        "When was Python first released?",
        "1991",
        63
    ),
    (
        "The Eiffel Tower is located in Paris, France and was built in 1889.",
        "Where is the Eiffel Tower located?",
        "Paris",
        30
    ),
    (
        "The Eiffel Tower is located in Paris, France and was built in 1889.",
        "When was the Eiffel Tower built?",
        "1889",
        64
    ),
    (
        "Albert Einstein developed the theory of relativity in 1915.",
        "Who developed the theory of relativity?",
        "Albert Einstein",
        0
    ),
    (
        "Albert Einstein developed the theory of relativity in 1915.",
        "In what year was the theory of relativity developed?",
        "1915",
        55
    ),
    (
        "The Great Wall of China is over 21000 kilometers long and was built over many centuries.",
        "How long is the Great Wall of China?",
        "over 21000 kilometers",
        26
    ),
    (
        "The Great Wall of China is over 21000 kilometers long and was built over many centuries.",
        "What was built over many centuries?",
        "The Great Wall of China",
        0
    ),
    (
        "Marie Curie won two Nobel Prizes, one in Physics in 1903 and one in Chemistry in 1911.",
        "How many Nobel Prizes did Marie Curie win?",
        "two",
        18
    ),
    (
        "Marie Curie won two Nobel Prizes, one in Physics in 1903 and one in Chemistry in 1911.",
        "In what year did she win the Chemistry Nobel Prize?",
        "1911",
        85
    ),
    (
        "The Pacific Ocean is the largest and deepest ocean on Earth covering about 63 million square miles.",
        "What is the largest ocean on Earth?",
        "The Pacific Ocean",
        0
    ),
    (
        "The Pacific Ocean is the largest and deepest ocean on Earth covering about 63 million square miles.",
        "How many square miles does the Pacific Ocean cover?",
        "about 63 million square miles",
        67
    ),
    (
        "Shakespeare wrote Romeo and Juliet around 1594 to 1596.",
        "What play did Shakespeare write around 1594?",
        "Romeo and Juliet",
        18
    ),
    (
        "Shakespeare wrote Romeo and Juliet around 1594 to 1596.",
        "Who wrote Romeo and Juliet?",
        "Shakespeare",
        0
    ),
    (
        "The speed of light in vacuum is approximately 299792458 meters per second.",
        "What is the speed of light in vacuum?",
        "approximately 299792458 meters per second",
        32
    ),
    (
        "The speed of light in vacuum is approximately 299792458 meters per second.",
        "In what unit is the speed of light measured?",
        "meters per second",
        60
    ),
    (
        "NASA sent the Apollo 11 mission to the Moon in 1969 with Neil Armstrong as the first human on the Moon.",
        "Who was the first human on the Moon?",
        "Neil Armstrong",
        61
    ),
    (
        "NASA sent the Apollo 11 mission to the Moon in 1969 with Neil Armstrong as the first human on the Moon.",
        "What year did humans first land on the Moon?",
        "1969",
        52
    ),
    (
        "The Amazon Rainforest produces about 20 percent of the world oxygen and spans nine countries.",
        "What percentage of the world oxygen does the Amazon produce?",
        "about 20 percent",
        35
    ),
    (
        "Leonardo da Vinci painted the Mona Lisa between 1503 and 1519 in Florence, Italy.",
        "What did Leonardo da Vinci paint?",
        "the Mona Lisa",
        26
    ),
]

print(f"  样本数量: {len(synthetic_qa_data)}")
for i in range(3):
    ctx, q, a, _ = synthetic_qa_data[i]
    print(f"\n  [{i}] Context: {ctx}")
    print(f"      Question: {q}")
    print(f"      Answer: {a}")

# ============================================================
# 3. 预处理 QA 数据
# ============================================================
print("\n--- 3. 预处理 QA 数据 ---")

from transformers import AutoTokenizer

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)


class QADataset(Dataset):
    """QA 数据集"""

    def __init__(self, qa_data, tokenizer, max_length=128):
        self.samples = []
        for context, question, answer, answer_start in qa_data:
            # tokenize question + context
            encoding = tokenizer(
                question,
                context,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )

            # 找到答案在 tokenized 序列中的位置
            inputs = tokenizer(
                question, context, truncation=True, max_length=max_length
            )
            input_ids = inputs["input_ids"]

            # 用 tokenizer 的 char_to_token 找到 start/end 位置
            # 先在 context 中定位
            ans_start_char = context.find(answer)
            if ans_start_char == -1:
                ans_start_char = answer_start

            ans_end_char = ans_start_char + len(answer)

            # 将字符位置转换为 token 位置
            # 需要考虑 question + [SEP] 的偏移
            encodings = tokenizer(
                question, context, truncation=True, max_length=max_length,
                return_offsets_mapping=True
            )
            offsets = encodings["offset_mapping"]

            # 找到 context 部分的起始 token index
            # question tokens + [SEP] 之后才是 context
            seq_ids = encodings.sequence_ids()

            start_token = None
            end_token = None

            for idx, (offset, seq_id) in enumerate(zip(offsets, seq_ids)):
                if seq_id != 1:  # 只看 context 部分 (seq_id=1)
                    continue
                if offset[0] <= ans_start_char and offset[1] > ans_start_char:
                    start_token = idx
                if offset[0] < ans_end_char and offset[1] >= ans_end_char:
                    end_token = idx

            # 如果找不到精确位置, 使用启发式
            if start_token is None or end_token is None:
                # 找到 context 的起始 token
                ctx_start = seq_ids.index(1) if 1 in seq_ids else len(seq_ids) - 1
                # 粗略估计
                ctx_text = context.lower()
                ans_text = answer.lower()
                char_pos = ctx_text.find(ans_text)
                if char_pos >= 0:
                    approx_tokens = len(tokenizer.tokenize(context[:char_pos]))
                    ans_tokens = len(tokenizer.tokenize(answer))
                    start_token = ctx_start + approx_tokens
                    end_token = start_token + ans_tokens - 1
                else:
                    start_token = ctx_start
                    end_token = ctx_start

            start_token = max(0, min(start_token, len(input_ids) - 1))
            end_token = max(0, min(end_token, len(input_ids) - 1))

            self.samples.append({
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "start_positions": torch.tensor(start_token, dtype=torch.long),
                "end_positions": torch.tensor(end_token, dtype=torch.long),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


full_dataset = QADataset(synthetic_qa_data, tokenizer, max_length=96)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"  总样本数: {len(full_dataset)}")
print(f"  训练集:   {len(train_dataset)}")
print(f"  验证集:   {len(val_dataset)}")
print(f"  样本特征: input_ids={train_dataset[0]['input_ids'].shape}")
print(f"  start_position={train_dataset[0]['start_positions'].item()}, "
      f"end_position={train_dataset[0]['end_positions'].item()}")

# ============================================================
# 4. 微调 DistilBERT 进行抽取式 QA
# ============================================================
print("\n--- 4. 微调 DistilBERT 进行 QA ---")

from transformers import (
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer
)

model = AutoModelForQuestionAnswering.from_pretrained(model_name)

total_params = sum(p.numel() for p in model.parameters())
print(f"  模型: {model_name}")
print(f"  参数量: {total_params:,}")
print(f"  QA 输出头: qa_outputs → {model.qa_outputs}")


def compute_qa_metrics(eval_pred):
    """计算 QA 评估指标"""
    start_logits, end_logits = eval_pred
    # 这里我们简化评估, 计算 start 和 end 的准确率
    # 实际 SQuAD 评估使用 Exact Match (EM) 和 F1
    return {"eval_loss": 0.0}  # Trainer 会自动计算 loss


training_args = TrainingArguments(
    output_dir="./results_qa",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    weight_decay=0.01,
    logging_steps=5,
    load_best_model_at_end=True,
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
)

print("  开始训练...")
trainer.train()

# ============================================================
# 5. 推理: 自定义问题
# ============================================================
print("\n--- 5. 推理演示 ---")


def answer_question(question, context, model, tokenizer, max_length=128):
    """从 context 中抽取问题答案"""
    inputs = tokenizer(
        question, context,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    )

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]

    # 获取最可能的 start 和 end 位置
    start_idx = torch.argmax(start_logits).item()
    end_idx = torch.argmax(end_logits).item()

    # 确保 end >= start
    if end_idx < start_idx:
        end_idx = start_idx

    # 解码答案
    input_ids = inputs["input_ids"][0]
    answer_ids = input_ids[start_idx:end_idx + 1]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True)

    # 置信度
    start_score = torch.softmax(start_logits, dim=0)[start_idx].item()
    end_score = torch.softmax(end_logits, dim=0)[end_idx].item()
    confidence = start_score * end_score

    return answer, confidence, start_idx, end_idx


test_qa = [
    (
        "What language did Guido van Rossum create?",
        "Python was created by Guido van Rossum and first released in 1991."
    ),
    (
        "Where is the Eiffel Tower?",
        "The Eiffel Tower is located in Paris, France and was built in 1889."
    ),
    (
        "Who won two Nobel Prizes?",
        "Marie Curie won two Nobel Prizes, one in Physics in 1903 and one in Chemistry in 1911."
    ),
    (
        "What is the speed of light?",
        "The speed of light in vacuum is approximately 299792458 meters per second."
    ),
]

for question, context in test_qa:
    answer, confidence, start, end = answer_question(
        question, context, model, tokenizer
    )
    print(f"  Context:  {context}")
    print(f"  Question: {question}")
    print(f"  Answer:   {answer}")
    print(f"  置信度:   {confidence:.4f} (位置: [{start}:{end}])")
    print()

# ============================================================
# 6. 使用 pipeline 快速推理
# ============================================================
print("\n--- 6. 使用 pipeline 快速推理 ---")

from transformers import pipeline

try:
    qa_pipeline = pipeline(
        "question-answering",
        model=model,
        tokenizer=tokenizer
    )

    result = qa_pipeline(
        question="What is the largest ocean?",
        context="The Pacific Ocean is the largest and deepest ocean on Earth."
    )
    print(f"  Pipeline 推理:")
    print(f"    答案: {result['answer']}")
    print(f"    置信度: {result['score']:.4f}")
    print(f"    位置: [{result['start']}:{result['end']}]")
except Exception as e:
    print(f"  Pipeline 推理异常: {e}")

# ============================================================
# 7. 总结
# ============================================================
print("\n--- 7. 总结 ---")
print("""
  本节学习了:
  1) 抽取式 QA 任务原理 (start + end position prediction)
  2) 构造 SQuAD 风格数据 (context, question, answer)
  3) 字符位置 → token 位置的对齐
  4) 微调 DistilBERT 进行 QA
  5) 推理和答案提取
  6) pipeline 快速推理

  关键概念:
    - QA 模型输出 start_logits 和 end_logits
    - 训练标签是答案在 tokenized 序列中的起止位置
    - SQuAD 评估使用 Exact Match (EM) 和 F1

  下一步: d7_transfer_project.py - 完整迁移学习项目
""")
