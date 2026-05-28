"""
W20-D5 微调Pipeline (Finetune Pipeline)
========================================
使用PEFT库微调: 数据准备, LoRA配置, 训练循环,
模型合并, 推理测试
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W20-D5 微调Pipeline (Finetune Pipeline)")
print("=" * 60)

# ============================================================
# 1. 微调Pipeline概述
# ============================================================
print("\n--- 1. 微调Pipeline概述 ---")
print("""
  完整微调流程:
    1) 准备数据: 收集/清洗/格式化训练数据
    2) 加载模型: 预训练模型 + LoRA配置
    3) 训练:     训练循环, 监控指标
    4) 评估:     在验证集上评估
    5) 合并:     将LoRA权重合并到基础模型
    6) 部署:     导出模型, 推理测试
""")

# ============================================================
# 2. 数据准备
# ============================================================
print("\n--- 2. 数据准备 ---")
print("""
  指令微调数据格式 (Alpaca格式):
    {
      "instruction": "将以下句子翻译为英文",
      "input": "今天天气很好",
      "output": "The weather is nice today."
    }

  对话格式 (ShareGPT格式):
    {
      "conversations": [
        {"role": "user", "content": "什么是机器学习？"},
        {"role": "assistant", "content": "机器学习是AI的子领域..."}
      ]
    }
""")


# 模拟数据集
class InstructionDataset:
    """指令数据集"""

    def __init__(self):
        self.data = [
            {"instruction": "解释以下概念", "input": "机器学习", "output": "机器学习是人工智能的一个子领域，通过算法从数据中学习模式，包括监督学习、无监督学习和强化学习。"},
            {"instruction": "解释以下概念", "input": "深度学习", "output": "深度学习是机器学习的子领域，使用多层神经网络进行特征学习，在图像和NLP领域取得突破。"},
            {"instruction": "解释以下概念", "input": "RAG", "output": "RAG是检索增强生成技术，通过检索外部知识库来增强大语言模型的回答质量和事实性。"},
            {"instruction": "比较两个概念", "input": "BERT vs GPT", "output": "BERT是encoder-only架构，适合理解任务；GPT是decoder-only架构，适合生成任务。"},
            {"instruction": "解释以下概念", "input": "LoRA", "output": "LoRA是低秩适应方法，通过低秩矩阵分解来高效微调大模型，只训练0.1-2%的参数。"},
            {"instruction": "列举要点", "input": "Transformer的关键组件", "output": "1.多头自注意力 2.位置编码 3.前馈网络 4.残差连接 5.层归一化"},
            {"instruction": "解释以下概念", "input": "向量数据库", "output": "向量数据库专门存储和检索高维向量，使用ANN算法实现毫秒级搜索，是RAG系统的核心。"},
            {"instruction": "给出建议", "input": "如何开始学习AI", "output": "建议路线: Python基础→数学基础→机器学习→深度学习→NLP→大模型→RAG/Agent。"},
        ]

    def format_prompt(self, item):
        """格式化为Prompt"""
        if item['input']:
            return f"### 指令:\n{item['instruction']}\n\n### 输入:\n{item['input']}\n\n### 回答:\n{item['output']}"
        return f"### 指令:\n{item['instruction']}\n\n### 回答:\n{item['output']}"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.format_prompt(self.data[idx])


dataset = InstructionDataset()
print(f"  数据集大小: {len(dataset)}")
print(f"\n  示例数据:")
print(f"  {dataset[0][:100]}...")

# ============================================================
# 3. LoRA 配置
# ============================================================
print("\n--- 3. LoRA 配置 ---")
print("""
  PEFT LoRA 配置参数:
    r:              LoRA秩, 常用 8, 16, 32, 64
    lora_alpha:     缩放因子, 通常 = 2*r
    target_modules: 应用LoRA的模块名
    lora_dropout:   Dropout率, 常用 0.05-0.1
    bias:           偏置处理, "none" 或 "all"
    task_type:      任务类型, CAUSAL_LM / SEQ_CLS 等

  推荐配置:
    小模型 (<7B):   r=8, alpha=16, target=["q_proj", "v_proj"]
    中模型 (7-13B): r=16, alpha=32, target=["q_proj", "k_proj", "v_proj", "o_proj"]
    大模型 (>30B):  r=64, alpha=128, target=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
""")


class LoRAConfig:
    """LoRA配置"""

    def __init__(self, r=8, lora_alpha=16, target_modules=None, lora_dropout=0.05):
        self.r = r
        self.lora_alpha = lora_alpha
        self.target_modules = target_modules or ["q_proj", "v_proj"]
        self.lora_dropout = lora_dropout

    def summary(self):
        print(f"  LoRA配置:")
        print(f"    r (秩):        {self.r}")
        print(f"    alpha:         {self.lora_alpha}")
        print(f"    alpha/r:       {self.lora_alpha / self.r:.1f}")
        print(f"    target:        {self.target_modules}")
        print(f"    dropout:       {self.lora_dropout}")

        # 估算参数量 (假设LLaMA-7B)
        d = 4096
        n_layers = 32
        n_targets = len(self.target_modules)
        lora_params = n_layers * n_targets * 2 * d * self.r
        total_params = 7e9
        print(f"    估算LoRA参数: {lora_params:,} ({lora_params/total_params*100:.3f}%)")


config = LoRAConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
config.summary()

# ============================================================
# 4. 模拟训练循环
# ============================================================
print("\n--- 4. 模拟训练循环 ---")


class SimulatedTrainingLoop:
    """模拟LoRA训练循环"""

    def __init__(self, config: LoRAConfig, n_epochs=10, lr=2e-4):
        self.config = config
        self.n_epochs = n_epochs
        self.lr = lr
        self.train_losses = []
        self.eval_losses = []
        self.learning_rates = []

    def _simulate_loss(self, epoch, total_epochs, is_eval=False):
        """模拟损失曲线"""
        # 指数衰减 + 噪声
        base_loss = 2.5 * np.exp(-0.3 * epoch) + 0.3
        noise = np.random.randn() * 0.05
        if is_eval:
            noise *= 1.5  # 验证集波动更大
            base_loss += 0.05  # 略高于训练
        return max(0.2, base_loss + noise)

    def _get_lr(self, epoch, step, total_steps):
        """余弦退火学习率"""
        progress = (epoch * total_steps + step) / (self.n_epochs * total_steps)
        warmup = 0.1
        if progress < warmup:
            return self.lr * progress / warmup
        return self.lr * 0.5 * (1 + np.cos(np.pi * (progress - warmup) / (1 - warmup)))

    def train(self, dataset):
        """模拟训练"""
        total_steps = len(dataset)
        print(f"  训练配置: epochs={self.n_epochs}, lr={self.lr}, steps/epoch={total_steps}")
        print(f"\n  {'Epoch':>5s} {'Train Loss':>10s} {'Eval Loss':>10s} {'LR':>12s}")
        print(f"  {'-'*5} {'-'*10} {'-'*10} {'-'*12}")

        for epoch in range(self.n_epochs):
            # 模拟训练
            train_loss = self._simulate_loss(epoch, self.n_epochs)
            self.train_losses.append(train_loss)

            # 模拟验证
            eval_loss = self._simulate_loss(epoch, self.n_epochs, is_eval=True)
            self.eval_losses.append(eval_loss)

            # 学习率
            lr = self._get_lr(epoch, 0, total_steps)
            self.learning_rates.append(lr)

            if epoch % 2 == 0 or epoch == self.n_epochs - 1:
                print(f"  {epoch:>5d} {train_loss:>10.4f} {eval_loss:>10.4f} {lr:>12.6f}")

        print(f"\n  训练完成! 最终train_loss={self.train_losses[-1]:.4f}, eval_loss={self.eval_losses[-1]:.4f}")
        return self.train_losses, self.eval_losses


# 运行训练
trainer = SimulatedTrainingLoop(config, n_epochs=20, lr=2e-4)
train_losses, eval_losses = trainer.train(dataset)

# ============================================================
# 5. 模型合并与推理
# ============================================================
print("\n--- 5. 模型合并与推理 ---")
print("""
  合并方式:
    1) 合并到基础模型:
       W_merged = W_base + (alpha/r) * B @ A
       优点: 推理无额外开销
       缺点: 无法切换不同任务

    2) 保存为单独的LoRA权重:
       只保存 B 和 A 矩阵
       推理时: 动态加载LoRA
       优点: 灵活切换任务, 节省存储

  PEFT合并代码:
    from peft import PeftModel
    model = PeftModel.from_pretrained(base_model, "lora_checkpoint")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained("merged_model")
""")


def simulate_merge():
    """模拟权重合并"""
    d = 32
    r = 4

    W_base = np.random.randn(d, d) * 0.02
    B = np.random.randn(d, r) * 0.1
    A = np.random.randn(r, d) * 0.01

    # 合并
    alpha_r = 32 / 4
    W_merged = W_base + alpha_r * B @ A

    # 验证
    x = np.random.randn(d)
    y_before = x @ W_base.T + alpha_r * (x @ A.T @ B.T)
    y_merged = x @ W_merged.T

    print(f"  合并前输出范数: {np.linalg.norm(y_before):.4f}")
    print(f"  合并后输出范数: {np.linalg.norm(y_merged):.4f}")
    print(f"  差异: {np.linalg.norm(y_before - y_merged):.10f}")
    print(f"  合并成功: {np.allclose(y_before, y_merged)}")


simulate_merge()

# ============================================================
# 6. 推理测试
# ============================================================
print("\n--- 6. 推理测试 ---")
print("""
  微调前后对比:

  微调前 (基础模型):
    Q: "什么是LoRA"
    A: "LoRA是一种无线通信技术..." (通用知识)

  微调后 (领域适应):
    Q: "什么是LoRA"
    A: "LoRA(低秩适应)是参数高效微调方法, 通过低秩矩阵分解
        ΔW=BA来减少可训练参数..." (专业知识)
""")


# 模拟推理
def simulate_inference(questions, is_finetuned=True):
    """模拟推理"""
    base_responses = {
        "什么是LoRA": "LoRA可以指无线通信技术或机器学习中的低秩适应方法。",
        "解释RAG": "RAG可能指不同的缩写，具体取决于上下文。",
        "Transformer是什么": "Transformer是一种深度学习架构，用于自然语言处理。",
    }

    finetuned_responses = {
        "什么是LoRA": "LoRA(Low-Rank Adaptation)是参数高效微调方法。通过将权重更新ΔW分解为B*A，只训练0.1-2%的参数即可达到接近全量微调的效果。秩r通常取8-64，alpha/r控制更新强度。",
        "解释RAG": "RAG(检索增强生成)通过检索外部知识库来增强LLM的回答。流程：文档切分→向量化→存储→检索→重排序→LLM生成。优势包括减少幻觉、实时知识更新和来源追溯。",
        "Transformer是什么": "Transformer是2017年Google提出的深度学习架构，完全基于自注意力机制。核心组件包括多头注意力、位置编码、前馈网络和残差连接。是BERT、GPT等现代大语言模型的基础。",
    }

    responses = finetuned_responses if is_finetuned else base_responses
    label = "微调后" if is_finetuned else "微调前"

    print(f"\n  === {label} ===")
    for q in questions:
        answer = responses.get(q, "[模拟回答]")
        print(f"\n  Q: {q}")
        print(f"  A: {answer[:80]}...")


questions = ["什么是LoRA", "解释RAG", "Transformer是什么"]
simulate_inference(questions, is_finetuned=False)
simulate_inference(questions, is_finetuned=True)

# ============================================================
# 7. 可视化
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 子图1: 训练曲线
ax1 = axes[0]
epochs = range(len(train_losses))
ax1.plot(epochs, train_losses, '-', linewidth=2, label='训练Loss', color='#3498db')
ax1.plot(epochs, eval_losses, '-', linewidth=2, label='验证Loss', color='#e74c3c')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('LoRA微调训练曲线', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 子图2: 学习率曲线
ax2 = axes[1]
ax2.plot(epochs, trainer.learning_rates, '-', linewidth=2, color='#2ecc71')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('学习率', fontsize=12)
ax2.set_title('余弦退火学习率', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 子图3: 不同rank的效果对比
ax3 = axes[2]
ranks = [1, 2, 4, 8, 16, 32, 64]
base_scores = [65, 72, 80, 87, 91, 93, 94]
finetuned_scores = [68, 76, 84, 91, 94, 95.5, 95.8]

ax3.plot(ranks, base_scores, 'o--', linewidth=2, label='基础模型', color='#3498db')
ax3.plot(ranks, finetuned_scores, 's-', linewidth=2, label='LoRA微调后', color='#e74c3c')
ax3.fill_between(ranks, base_scores, finetuned_scores, alpha=0.15, color='#2ecc71')
ax3.set_xlabel('LoRA秩 r', fontsize=12)
ax3.set_ylabel('任务分数', fontsize=12)
ax3.set_title('不同秩r的微调效果', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W20/finetune_pipeline.png', dpi=150, bbox_inches='tight')
print("\n  图表已保存: finetune_pipeline.png")
plt.close()

# ============================================================
# 8. 使用PEFT库的实际代码模板
# ============================================================
print("\n--- 8. PEFT库实际代码模板 ---")
print("""
  import torch
  from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
  from peft import LoraConfig, get_peft_model, TaskType
  from trl import SFTTrainer

  # 1. 加载模型
  model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B")
  tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

  # 2. LoRA配置
  lora_config = LoraConfig(
      r=16,
      lora_alpha=32,
      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
      lora_dropout=0.05,
      task_type=TaskType.CAUSAL_LM,
  )
  model = get_peft_model(model, lora_config)
  model.print_trainable_parameters()

  # 3. 训练配置
  training_args = TrainingArguments(
      output_dir="./output",
      num_train_epochs=3,
      per_device_train_batch_size=4,
      gradient_accumulation_steps=4,
      learning_rate=2e-4,
      fp16=True,
      logging_steps=10,
      save_steps=100,
  )

  # 4. 训练
  trainer = SFTTrainer(
      model=model,
      args=training_args,
      train_dataset=dataset,
      tokenizer=tokenizer,
  )
  trainer.train()

  # 5. 保存和合并
  model.save_pretrained("./lora_output")
  merged_model = model.merge_and_unload()
  merged_model.save_pretrained("./merged_model")
""")

print("\n" + "=" * 60)
print("W20-D5 完成! 本节实现了完整的LoRA微调Pipeline")
print("=" * 60)
