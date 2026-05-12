# W20 - 模型微调LoRA

> 第20周学习计划 | Java开发工程师转AI开发 | 工作日每晚2小时 + 周末6-8小时

---

## 一、本周目标

1. 理解微调策略全景（全量微调/Freeze/PEFT）及各自适用场景
2. 深入理解LoRA和QLoRA的数学原理和工程实现
3. 掌握PEFT库的使用，能独立完成LoRA微调实验
4. 掌握微调数据准备方法和数据质量控制
5. 完成领域模型微调实战项目（项目9）

---

## 二、时间安排

### 工作日（周一至周五，每晚2小时）

| 日期 | 主题 | 时间分配 |
|------|------|----------|
| Day 1 (周一) | 微调策略概览 | 理论学习90分钟 + 笔记30分钟 |
| Day 2 (周二) | LoRA原理 | 理论推导90分钟 + 参数计算30分钟 |
| Day 3 (周三) | QLoRA原理 | 理论60分钟 + 显存计算60分钟 |
| Day 4 (周四) | PEFT实战 | 学习60分钟 + 代码练习60分钟 |
| Day 5 (周五) | 微调数据准备 | 学习60分钟 + 数据处理60分钟 |

### 周末（周六至周日，每天6-8小时）

| 日期 | 主题 | 时间分配 |
|------|------|----------|
| Day 6 (周六) | 训练监控与调优 | 上午学习监控方法(3h) + 下午调优实验(3h) |
| Day 7 (周日) | 实战项目9 - 领域模型微调 | 全天实现项目(7-8h) |

---

## 三、详细学习内容

### Day 1: 微调策略概览

#### 3.1.1 全量微调 Full Fine-tuning

```
全量微调：更新模型的所有参数

假设模型参数量：7B (70亿)
需要更新的参数：7B
显存需求（FP16）：
  - 模型权重：7B × 2 bytes = 14 GB
  - 梯度：7B × 2 bytes = 14 GB
  - 优化器状态(AdamW)：7B × 8 bytes = 56 GB
  - 激活值：约 4-8 GB
  - 总计约：88-92 GB

需要：至少 4× A100(80GB) 或 8× A100(40GB)
```

**优点**：效果最好，模型完全适应新任务
**缺点**：资源需求极大，训练成本高，容易过拟合

#### 3.1.2 Freeze冻结策略

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# 冻结所有参数
for param in model.parameters():
    param.requires_grad = False

# 只解冻最后几层
for param in model.model.layers[-4:].parameters():
    param.requires_grad = True

# 只解冻LM Head
for param in model.lm_head.parameters():
    param.requires_grad = True

# 查看可训练参数数量
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"可训练参数: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
```

#### 3.1.3 PEFT参数高效微调

```
PEFT (Parameter-Efficient Fine-Tuning) 方法分类：

1. Adapter-based（适配器）
   - 在Transformer层之间插入小型全连接网络
   - 代表：Adapter、AdapterFusion
   - 推理时有额外延迟

2. Prefix-Tuning（前缀微调）
   - 在输入前添加可学习的虚拟token前缀
   - 代表：Prefix-Tuning、P-Tuning v2
   - 占用输入长度空间

3. LoRA（低秩适应）
   - 在权重矩阵旁添加低秩分解矩阵 W' = W + BA
   - 代表：LoRA、QLoRA
   - 推理时无额外延迟（可合并权重）
   - ⭐ 最主流的PEFT方法

4. Prompt-based
   - 在输入层优化连续的软提示
   - 代表：Prompt Tuning
   - 效果相对有限
```

```
| 方法           | 可训练参数比例 | 推理额外开销 | 效果     | 易用性 |
|---------------|--------------|-------------|---------|-------|
| Full Fine-tune | 100%         | 无          | 最好    | 高    |
| Freeze         | 5-20%        | 无          | 较好    | 高    |
| Adapter        | 1-5%         | 有          | 好      | 中    |
| Prefix-Tuning  | 0.1-1%       | 有          | 中      | 中    |
| LoRA           | 0.1-1%       | 无(可合并)   | 接近全量 | 高    |
| QLoRA          | 0.1-1%       | 无(可合并)   | 接近全量 | 高    |
```

---

### Day 2: LoRA原理

#### 3.2.1 核心思想

```
原始权重矩阵: W ∈ R^(d×d)  (例如 d=4096)
LoRA增量: ΔW = B × A

其中:
  A ∈ R^(r×d)  (降维矩阵)
  B ∈ R^(d×r)  (升维矩阵)
  r << d        (秩，远小于原始维度)

前向传播: h = Wx + ΔWx = Wx + BAx

参数量对比:
  原始: d × d = 4096 × 4096 = 16,777,216
  LoRA: d × r + r × d = 2 × d × r = 2 × 4096 × 16 = 131,072
  减少比例: 128倍！(r=16时)
```

#### 3.2.2 初始化策略

```
A: 使用正态分布初始化 A ~ N(0, σ²)，例如 Kaiming初始化
B: 初始化为全零矩阵 B = 0

初始状态: ΔW = B × A = 0 × A = 0

这意味着训练开始时 LoRA不改变原模型的行为，
训练过程中逐渐学习增量。
```

#### 3.2.3 关键参数

**rank (r)**：
```
r 的选择影响LoRA的表达能力：

r = 4:  最小参数量，适合简单任务（如风格调整）
r = 8:  平衡选择，适合大多数任务
r = 16: 通用推荐值，表达能力较强
r = 32: 复杂任务（如代码生成、多任务）
r = 64: 最大通常不建议，接近全量微调时可直接全量微调

注意：通常r=16就能达到接近全量微调的效果
```

**alpha (α)**：
```
缩放因子: 实际更新量 = (α/r) × BA

当 α = 2r 时: 缩放因子 = 2
当 α = r 时: 缩放因子 = 1（无缩放）

常用设置:
  r=8, alpha=16 → 缩放2倍
  r=16, alpha=32 → 缩放2倍
  r=32, alpha=64 → 缩放2倍

alpha/r 的比值控制LoRA的"学习强度"
```

**目标模块选择**：
```
Transformer中的线性层:
  q_proj: Query投影
  k_proj: Key投影
  v_proj: Value投影
  o_proj: Output投影
  gate_proj: Gate投影 (FFN)
  up_proj: Up投影 (FFN)
  down_proj: Down投影 (FFN)

推荐配置:
  基础: ["q_proj", "v_proj"]  → 最少参数，已有效果
  推荐: ["q_proj", "k_proj", "v_proj", "o_proj"]  → Attention全覆盖
  最强: 所有linear层  → 最大表达力
```

#### 3.2.4 参数量计算练习

```
以LLaMA-2-7B为例:
  hidden_size (d) = 4096
  num_attention_heads = 32
  num_hidden_layers = 32

单个Attention层的线性层:
  q_proj: d × d = 4096 × 4096 = 16,777,216
  k_proj: d × d = 4096 × 4096 = 16,777,216
  v_proj: d × d = 4096 × 4096 = 16,777,216
  o_proj: d × d = 4096 × 4096 = 16,777,216

LoRA参数量 (r=16, target=["q_proj","v_proj"]):
  每个目标层: 2 × d × r = 2 × 4096 × 16 = 131,072
  每个Transformer层: 131,072 × 2 = 262,144
  32层总计: 262,144 × 32 = 8,388,608

总参数: 7,000,000,000
LoRA参数: 8,388,608
比例: 8,388,608 / 7,000,000,000 = 0.12%

仅训练0.12%的参数就能达到接近全量微调的效果！
```

---

### Day 3: QLoRA原理

#### 3.3.1 核心创新

```
QLoRA = 量化(Quantization) + LoRA

核心思路：
1. 将基座模型量化为4-bit存储 → 大幅减少显存
2. 在4-bit模型上训练LoRA适配器（计算时反量化为BF16）
3. 只更新LoRA参数，基座模型权重不变

效果：在单张24GB显存的消费级GPU上微调7B模型！
```

#### 3.3.2 NF4 (NormalFloat4) 量化

```
传统量化(INT4)：均匀分布的量化级别
NF4量化：为正态分布权重优化的量化级别

原理：
  - 预训练模型的权重通常服从正态分布 N(0, σ)
  - NF4 在正态分布的高概率区域放置更多量化级别
  - 信息论最优：使量化误差最小

NF4的16个量化级别（4-bit = 2^4 = 16个级别）：
  [-1.0, -0.696, -0.476, -0.298, -0.150, -0.054, 0.0, 0.054,
   0.150, 0.298, 0.476, 0.696, 1.0]
  （对称分布，精度在0附近更高）
```

#### 3.3.3 双量化 Double Quantization

```
问题：量化需要存储量化常数（scale和zero_point），
     这些常数本身也占显存。

解决：对量化常数再进行一次量化！

第一层量化：FP16权重 → NF4 (存储量化常数为FP32)
第二层量化：FP32量化常数 → FP4 (再存储第二层的量化常数)

节省的显存：
  每64个参数节省：64 × 32bit - 64 × 4bit - 32bit - 4bit ≈ 1.8bit/参数
  对于7B模型：7B × 1.8bit ≈ 1.6 GB
```

#### 3.3.4 分页优化器 Paged Optimizer

```
问题：优化器状态（如Adam的m和v）占用大量显存，
     在接近显存上限时可能导致OOM。

解决：利用CPU内存作为"交换空间"
  - 使用NVIDIA统一内存管理
  - 优化器状态在GPU和CPU之间自动page in/out
  - 当GPU显存不足时，自动将不常用的优化器状态offload到CPU

效果：避免OOM，使得显存峰值更平滑
```

#### 3.3.5 QLoRA显存计算

```
以LLaMA-2-7B为例，QLoRA训练显存估算：

基座模型(4-bit)：7B × 0.5 bytes ≈ 3.5 GB
LoRA参数(FP16)：8.4M × 2 bytes ≈ 0.016 GB
梯度(FP16)：8.4M × 2 bytes ≈ 0.016 GB
优化器状态(FP32)：8.4M × 8 bytes ≈ 0.064 GB
激活值：≈ 2-4 GB
总计：≈ 6-8 GB

在RTX 3090/4090 (24GB)上绰绰有余！
甚至可以微调13B模型（约12-14GB显存）
```

```
| 模型大小 | 全量微调(FP16) | QLoRA(4-bit) | 所需GPU       |
|---------|---------------|-------------|---------------|
| 7B      | ~90 GB        | ~8 GB       | RTX 3090/4090 |
| 13B     | ~160 GB       | ~14 GB      | RTX 3090/4090 |
| 34B     | ~400 GB       | ~30 GB      | A100(40GB)    |
| 70B     | ~800 GB       | ~48 GB      | A100(80GB)    |
```

---

### Day 4: PEFT实战

#### 3.4.1 环境准备

```bash
pip install peft transformers accelerate bitsandbytes datasets trl
```

#### 3.4.2 完整LoRA微调流程

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer
from datasets import load_dataset

# ========== 1. 加载Tokenizer ==========
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ========== 2. 配置4-bit量化 ==========
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",              # NF4量化
    bnb_4bit_compute_dtype=torch.bfloat16,  # 计算时反量化为BF16
    bnb_4bit_use_double_quant=True,         # 双量化
)

# ========== 3. 加载模型 ==========
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",  # 自动分配GPU
    torch_dtype=torch.bfloat16,
)

# 准备模型用于k-bit训练
model = prepare_model_for_kbit_training(model)

# ========== 4. LoRA配置 ==========
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                        # 秩
    lora_alpha=32,               # 缩放因子
    lora_dropout=0.05,           # Dropout
    target_modules=[             # 目标模块
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    bias="none",
)

# 应用LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 输出: trainable params: 13,107,200 || all params: 6,738,415,616 || trainable%: 0.1945

# ========== 5. 准备数据 ==========
dataset = load_dataset("json", data_files="train_data.json")

def format_instruction(sample):
    """格式化为指令数据"""
    return f"""### Instruction:
{sample['instruction']}

### Input:
{sample['input']}

### Response:
{sample['output']}"""

# ========== 6. 训练参数 ==========
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,     # 等效batch_size=16
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
    gradient_checkpointing=True,       # 节省显存
    optim="paged_adamw_8bit",          # 分页优化器
    max_grad_norm=0.3,
)

# ========== 7. 开始训练 ==========
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    formatting_func=format_instruction,
    max_seq_length=512,
)

trainer.train()

# ========== 8. 保存模型 ==========
# 保存LoRA权重（只有几十MB）
trainer.model.save_pretrained("./lora_output")
tokenizer.save_pretrained("./lora_output")

# ========== 9. 合并权重 ==========
from peft import AutoPeftModelForCausalLM

# 加载并合并LoRA权重
merged_model = AutoPeftModelForCausalLM.from_pretrained(
    "./lora_output",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
merged_model = merged_model.merge_and_unload()
merged_model.save_pretrained("./merged_model")
```

#### 3.4.3 推理

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 方式1：加载LoRA权重（基座+LoRA）
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model = PeftModel.from_pretrained(base_model, "./lora_output")

# 方式2：加载已合并的模型
model = AutoModelForCausalLM.from_pretrained(
    "./merged_model",
    device_map="auto",
)

# 推理
tokenizer = AutoTokenizer.from_pretrained("./lora_output")
prompt = "### Instruction:\n解释什么是机器学习\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

---

### Day 5: 微调数据准备

#### 3.5.1 指令数据格式

```json
// Alpaca格式
{
    "instruction": "将以下英文翻译为中文",
    "input": "Machine learning is a subset of artificial intelligence.",
    "output": "机器学习是人工智能的一个子集。"
}

// ShareGPT格式
{
    "conversations": [
        {"from": "human", "value": "什么是深度学习？"},
        {"from": "gpt", "value": "深度学习是机器学习的一个分支..."}
    ]
}
```

#### 3.5.2 数据质量把控

```python
import json
from collections import Counter

class DataQualityChecker:
    """微调数据质量检查"""

    def __init__(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]

    def check_all(self):
        print(f"总数据量: {len(self.data)}")
        self.check_duplicates()
        self.check_length_distribution()
        self.check_diversity()
        self.check_empty_fields()

    def check_duplicates(self):
        """检查重复数据"""
        instructions = [d["instruction"] for d in self.data]
        dup_count = len(instructions) - len(set(instructions))
        print(f"重复指令数: {dup_count} ({100*dup_count/len(instructions):.1f}%)")

    def check_length_distribution(self):
        """检查长度分布"""
        lengths = {
            "instruction": [len(d["instruction"]) for d in self.data],
            "output": [len(d["output"]) for d in self.data],
        }
        for field, lens in lengths.items():
            print(f"\n{field}长度统计:")
            print(f"  最小: {min(lens)}, 最大: {max(lens)}")
            print(f"  平均: {sum(lens)/len(lens):.0f}, 中位数: {sorted(lens)[len(lens)//2]}")

            # 过短或过长的数据
            too_short = sum(1 for l in lens if l < 10)
            too_long = sum(1 for l in lens if l > 2000)
            print(f"  过短(<10字): {too_short}, 过长(>2000字): {too_long}")

    def check_diversity(self):
        """检查多样性"""
        # 指令类型多样性
        keywords = ["翻译", "总结", "分析", "编程", "解释", "生成", "修改", "评估", "对比", "提取"]
        keyword_counts = Counter()
        for d in self.data:
            for kw in keywords:
                if kw in d["instruction"]:
                    keyword_counts[kw] += 1
        print(f"\n指令类型分布:")
        for kw, count in keyword_counts.most_common():
            print(f"  {kw}: {count}")

    def check_empty_fields(self):
        """检查空字段"""
        for field in ["instruction", "output"]:
            empty = sum(1 for d in self.data if not d.get(field, "").strip())
            print(f"空{field}数: {empty}")

    def filter_quality(self, output_path):
        """过滤低质量数据"""
        filtered = []
        for d in self.data:
            # 过滤条件
            if len(d["instruction"]) < 5:
                continue
            if len(d["output"]) < 20:
                continue
            if len(d["output"]) > 4000:
                continue
            if not d["instruction"].strip() or not d["output"].strip():
                continue
            filtered.append(d)

        print(f"过滤: {len(self.data)} → {len(filtered)} (移除 {len(self.data)-len(filtered)})")
        with open(output_path, 'w', encoding='utf-8') as f:
            for d in filtered:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
```

#### 3.5.3 数据增强

```python
# Self-Instruct：用LLM生成训练数据
def generate_training_data(seed_tasks, llm_client, num_generated=100):
    """用Self-Instruct方法生成训练数据"""
    generated_data = []

    for i in range(num_generated):
        # 随机选几个种子任务作为示例
        import random
        examples = random.sample(seed_tasks, min(3, len(seed_tasks)))

        prompt = f"""基于以下示例，生成一个新的指令和回答。

示例：
{format_examples(examples)}

请生成一个新的指令和回答。以JSON格式输出：
{{"instruction": "...", "input": "...", "output": "..."}}"""

        response = llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
        )
        try:
            data = json.loads(response.choices[0].message.content)
            generated_data.append(data)
        except json.JSONDecodeError:
            continue

    return generated_data
```

#### 3.5.4 数据集划分

```python
def split_dataset(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """划分训练集/验证集/测试集"""
    import random
    random.seed(seed)
    random.shuffle(data)

    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return {
        "train": data[:train_end],
        "validation": data[train_end:val_end],
        "test": data[val_end:],
    }
```

---

### Day 6: 训练监控与调优

#### 3.6.1 Loss曲线分析

```
正常训练的Loss曲线：
Loss
  |
  |\
  | \
  |  \___________  ← 稳定下降后趋于平稳
  |
  +----------------→ Steps

过拟合的Loss曲线：
Train Loss             Val Loss
  |                      |
  |\                     |\
  | \                    | \
  |  \_______            |  \___/\_  ← 验证Loss开始上升
  |                       |
  +-----------→ Steps     +-----------→ Steps

欠拟合的Loss曲线：
Loss
  |
  |--------------------  ← Loss始终很高，下降缓慢
  |
  +----------------→ Steps

震荡的Loss曲线：
Loss
  |  /\  /\  /\
  | /  \/  \/  \  ← 学习率太大
  |
  +----------------→ Steps
```

#### 3.6.2 超参数调优经验

```
学习率 (learning_rate):
  推荐范围: 1e-4 ~ 5e-4 (LoRA常用 2e-4)
  太大: Loss震荡，不收敛
  太小: 收敛太慢，可能陷入局部最优
  调优: 从2e-4开始，如果震荡降低到1e-4，如果太慢提升到5e-4

Epoch数:
  推荐范围: 3-5
  太少: 欠拟合
  太多: 过拟合
  技巧: 使用early stopping，监控验证集Loss

Rank (r):
  推荐范围: 8-32
  太小: 表达能力不足
  太大: 过拟合风险增加
  经验: r=16是大多数任务的好起点

Batch Size:
  推荐范围: 4-16 (用梯度累积模拟大batch)
  实际batch=4, gradient_accumulation_steps=4 → 等效batch=16
  显存不够时降低batch_size，增大accumulation_steps
```

#### 3.6.3 生成样本定期评估

```python
def evaluate_generation(model, tokenizer, eval_prompts, device):
    """定期评估生成质量"""
    model.eval()
    results = []

    for prompt in eval_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append({"prompt": prompt, "response": response})

    model.train()
    return results

# 在训练过程中定期评估
eval_prompts = [
    "请解释什么是LoRA微调。",
    "用Python实现一个二分查找算法。",
    "分析以下代码的性能瓶颈：for i in range(len(list))",
]

# 每个epoch结束后评估
# trainer callback or manual loop
```

#### 3.6.4 Perplexity监控

```python
def compute_perplexity(model, tokenizer, texts, device):
    """计算困惑度"""
    model.eval()
    total_loss = 0
    total_tokens = 0

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    model.train()
    return perplexity
```

---

### Day 7: 实战项目9 - 领域模型微调

#### 项目选择

选择一个领域（以代码助手为例）：

```
领域：Java代码助手
目标：让模型能回答Java相关问题、生成Java代码、解释Java概念

数据来源：
  1. Java StackOverflow问答（爬取+清洗）
  2. Java官方文档的Q&A对
  3. 自己编写的Java教学Q&A

数据量目标：1000-3000条高质量指令数据
```

#### 完整流程

```python
"""
项目9：Java代码助手微调
使用QLoRA微调Qwen2-7B
"""

# Step 1: 数据收集和准备
# 假设已有数据在 java_qa_data.jsonl

# Step 2: 数据质量检查
checker = DataQualityChecker("java_qa_data.jsonl")
checker.check_all()
checker.filter_quality("java_qa_data_clean.jsonl")

# Step 3: 数据集划分
with open("java_qa_data_clean.jsonl", 'r') as f:
    data = [json.loads(line) for line in f]
splits = split_dataset(data)

# Step 4: QLoRA配置和训练（参考Day 4完整流程）
# 使用以下配置:
model_name = "Qwen/Qwen2-7B"  # 中文能力好
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
training_args = TrainingArguments(
    num_train_epochs=3,
    learning_rate=2e-4,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
)

# Step 5: 训练
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=splits["train"],
    formatting_func=format_java_instruction,
)
trainer.train()

# Step 6: 评估 - 对比微调前后
def compare_before_after(test_data, base_model, finetuned_model, tokenizer):
    """对比微调前后在领域问题上的表现"""
    results = []
    for sample in test_data:
        prompt = format_java_instruction(sample)

        # 基座模型回答
        base_response = generate(base_model, tokenizer, prompt)
        # 微调模型回答
        ft_response = generate(finetuned_model, tokenizer, prompt)

        results.append({
            "question": sample["instruction"],
            "reference": sample["output"],
            "base_response": base_response,
            "finetuned_response": ft_response,
        })
    return results
```

---

## 四、代码练习

### Day 2 练习：手动推导LoRA参数量

```
任务：
1. 计算LLaMA-2-7B在不同rank下的LoRA参数量
   - r=4: ?
   - r=8: ?
   - r=16: ?
   - r=32: ?
   - r=64: ?
2. target_modules=["q_proj","v_proj"] vs ["q_proj","k_proj","v_proj","o_proj"] vs 所有linear层
3. 计算每种配置的可训练参数比例
4. 输出：参数量计算表格
```

### Day 4 练习：用PEFT库微调一个7B模型

```
任务：
1. 选择一个开源模型（Qwen2-7B或LLaMA-2-7B）
2. 准备100条测试数据
3. 使用QLoRA配置进行微调
4. 监控训练Loss曲线
5. 输出：训练日志 + LoRA权重
```

### Day 5 练习：准备一个1000条的指令数据集

```
任务：
1. 选择一个领域（代码/医疗/法律/客服）
2. 使用Self-Instruct或手动编写方式准备数据
3. 运行数据质量检查
4. 划分训练/验证/测试集
5. 输出：train.jsonl + val.jsonl + test.jsonl
```

### Day 7 练习：完成项目9领域模型微调

```
任务：
1. 使用Day 5准备的数据集
2. 完成QLoRA微调全流程
3. 对比微调前后在领域问题上的表现（至少20个问题）
4. 保存LoRA权重和合并后的模型
5. 输出：完整项目代码 + 微调报告
```

---

## 五、本周产出

| 产出物 | 说明 | 完成标准 |
|--------|------|----------|
| lora_params_analysis.md | LoRA参数量分析 | 不同rank和target的参数量对比 |
| qlora_finetune.py | QLoRA微调脚本 | 完整训练流程 |
| instruction_dataset/ | 指令数据集 | 1000+条高质量数据 |
| lora_weights/ | LoRA权重文件 | 可加载推理 |
| finetune_report.md | 微调报告 | 包含Loss曲线+前后对比 |

---

## 六、自测题

### 题目

1. **LoRA为什么能达到接近全量微调的效果？**

<details>
<summary>参考答案</summary>

理论基础来自"Aghajanyan et al. 2020"的研究：预训练模型的微调过程具有"低内在维度"特性，即权重更新矩阵ΔW虽然维度很高，但有效秩很低，可以用低秩矩阵BA来近似。这意味着不需要更新所有参数，只需更新一个低秩的子空间就能捕获任务适应所需的大部分信息。此外，LoRA初始化B=0保证训练开始时不影响原模型，训练过程中只在小维度空间中优化，反而有助于减少过拟合。
</details>

2. **rank参数如何选择？**

<details>
<summary>参考答案</summary>

rank r的选择取决于任务复杂度和数据量：(1) 简单任务（风格调整、格式控制）：r=4-8即可；(2) 中等任务（领域问答、单语言任务）：r=8-16是好的起点；(3) 复杂任务（代码生成、多任务、推理）：r=16-32。r越大表达力越强但参数越多、过拟合风险越高。经验法则：从r=16开始实验，观察验证集Loss，如果欠拟合增大r，如果过拟合减小r。同时配合alpha=2r的设置。
</details>

3. **QLoRA如何实现4-bit训练？**

<details>
<summary>参考答案</summary>

QLoRA包含三个关键创新：(1) NF4量化：将基座模型权重从FP16/BF16量化为4-bit NormalFloat格式，利用权重正态分布特性使量化误差最小；(2) 双量化：对量化常数再做一次量化，进一步节省显存；(3) 分页优化器：利用CPU内存处理优化器状态，避免GPU OOM。训练时，前向传播将4-bit权重反量化为BF16进行计算，反向传播只更新LoRA参数（FP16），基座权重保持不变。这样在24GB显存上就能微调7B模型。
</details>

4. **微调数据的格式和质量如何影响效果？**

<details>
<summary>参考答案</summary>

格式影响：(1) 指令格式要与推理时的Prompt格式一致，否则模型学到的格式不匹配；(2) 输入输出的分隔要清晰明确。质量影响：(1) 数据准确性：错误的回答会误导模型；(2) 多样性：类型单一的指令导致模型泛化能力差；(3) 数量：太少（<100条）容易过拟合，太多低质量数据反而降低效果；(4) 长度分布：输出太短学不到丰富表达，太长增加训练难度。核心原则：质量 > 数量，1000条高质量数据 > 10000条低质量数据。
</details>

5. **训练loss正常下降但生成质量不好，可能的原因？**

<details>
<summary>参考答案</summary>

可能原因：(1) 过拟合：训练Loss下降但验证Loss上升，模型死记硬背训练数据，泛化差；(2) 格式不匹配：训练数据的格式和推理时的Prompt格式不一致；(3) 解码策略问题：temperature太高或太低、top_p设置不当；(4) 评估偏差：Loss衡量的是token级别的预测概率，但人感知的"质量"可能是连贯性、准确性等更高层次的特征；(5) 数据偏差：训练数据中某种回答模式过多导致模型偏向该模式。
</details>

---

## 七、Java开发者提示

### LoRA类比

```
LoRA ≈ Java中的装饰器模式 / AOP

// Java装饰器模式：不修改原类，添加新功能
interface DataService {
    String query(String input);
}

class BaseDataService implements DataService {
    public String query(String input) {
        return baseResult;  // 原始行为
    }
}

class LoRADecorator implements DataService {
    private DataService base;     // 原始服务（冻结）
    private LoRAAdapter adapter;  // LoRA适配器（可训练）

    public String query(String input) {
        return base.query(input) + adapter.adapt(input);
        // 相当于: Wx + BAx
    }
}

// LoRA: Wx + BAx
// W = 原始权重（冻结不更新）
// BA = LoRA增量（只训练这部分）
// 效果 ≈ 全量更新W，但只训练很少参数
```

### QLoRA显存类比

```
QLoRA ≈ Java中的内存优化策略

// Java内存优化
原始方案：所有对象都在堆内存（类比全量微调所有参数都在GPU显存）
优化方案：
  1. 压缩存储：用更紧凑的数据结构（类比NF4量化，FP16→4bit）
  2. 懒加载：需要时再加载到内存（类比分页优化器，GPU↔CPU交换）
  3. 只修改增量：不修改原始数据，只记录变更（类比LoRA只训练增量）
```

### 训练过程类比

```
模型训练 ≈ Java程序的性能调优

超参数调优：
  Java: JVM参数(-Xmx, -XX:+UseG1GC) → 吞吐量/延迟
  ML:   训练参数(lr, batch_size, epochs) → Loss/准确率

Loss曲线：
  Java: 压测时QPS曲线 → 判断是否稳定
  ML:   训练Loss曲线 → 判断是否收敛

过拟合：
  Java: 过度优化导致特定场景性能好但通用性差
  ML:   训练数据拟合太好但泛化能力差

Early Stopping：
  Java: 超时中断，避免无限运行
  ML:   验证Loss不再下降时停止训练
```

### 数据准备类比

```
微调数据准备 ≈ Java项目的数据清洗ETL

ETL流程：
  Extract：从多种来源采集数据（爬虫/API/手动编写）
  Transform：数据清洗、去重、格式转换
  Load：保存为标准格式

微调数据：
  Collect：收集领域问答数据
  Clean：去重、过滤低质量、长度过滤
  Format：转换为Alpaca/ShareGPT格式

都是"数据质量决定最终效果"的核心原则
```
