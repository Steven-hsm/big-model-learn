"""
W20-D7 微调项目 (Finetune Project)
====================================
微调项目: 选择模型, 准备数据, 配置LoRA,
训练, 评估, 合并保存, 对比微调前后效果
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W20-D7 微调项目 (Finetune Project)")
print("=" * 60)

# ============================================================
# 1. 项目概述
# ============================================================
print("\n--- 1. 项目概述 ---")
print("""
  项目目标: 微调一个小型语言模型用于AI知识问答
  步骤:
    1) 选择模型
    2) 准备数据
    3) 配置LoRA
    4) 训练
    5) 评估
    6) 合并保存
    7) 对比效果
""")

# ============================================================
# 2. 选择模型
# ============================================================
print("\n--- 2. 选择模型 ---")
print("""
  模型选择考虑因素:
    1) 任务类型: 生成 -> decoder-only (GPT/Qwen)
    2) 语言需求: 中文为主 -> Qwen2.5, ChatGLM, DeepSeek
    3) 硬件限制: 8GB GPU -> 1-3B模型或4-bit量化
    4) 许可证:   商用需注意license

  推荐选择:
    +------------------+--------+----------+------------------+
    |    模型          |  参数  | 中文能力 |     许可证       |
    +------------------+--------+----------+------------------+
    | Qwen2.5-7B       | 7B     | 优秀     | Apache 2.0       |
    | ChatGLM3-6B      | 6B     | 优秀     | Apache 2.0       |
    | LLaMA-3-8B       | 8B     | 良好     | LLaMA License    |
    | Mistral-7B       | 7B     | 一般     | Apache 2.0       |
    | GPT-2 small      | 124M   | 弱       | MIT              |
    +------------------+--------+----------+------------------+

  本项目选择: 模拟微调 (不依赖实际模型加载)
""")

# ============================================================
# 3. 准备训练数据
# ============================================================
print("\n--- 3. 准备训练数据 ---")

# AI知识问答数据集
train_data = [
    {"instruction": "什么是机器学习", "input": "", "output": "机器学习是人工智能的子领域，通过算法让计算机从数据中自动学习模式。主要类型包括监督学习(使用标注数据)、无监督学习(发现隐藏结构)和强化学习(通过奖励学习策略)。"},
    {"instruction": "解释深度学习", "input": "", "output": "深度学习是机器学习的分支，使用多层神经网络进行特征学习。它在图像识别(CNN)、自然语言处理(Transformer)和语音识别等领域取得了突破性进展。"},
    {"instruction": "什么是Transformer", "input": "", "output": "Transformer是2017年Google提出的深度学习架构，基于自注意力机制。它能并行处理序列数据，捕获长距离依赖。是BERT、GPT等现代语言模型的基础。"},
    {"instruction": "RAG是什么", "input": "", "output": "RAG(检索增强生成)通过检索外部知识库来增强LLM的回答质量。流程:文档切分→向量化→检索→重排序→LLM生成。优势:减少幻觉、实时更新、来源可追溯。"},
    {"instruction": "解释LoRA微调", "input": "", "output": "LoRA(低秩适应)通过低秩矩阵分解ΔW=BA来高效微调大模型。只训练0.1-2%的参数，内存需求大幅降低。训练后可合并回原权重，推理无额外开销。"},
    {"instruction": "什么是向量数据库", "input": "", "output": "向量数据库专门存储和检索高维向量，使用ANN算法(如HNSW、IVF)实现毫秒级搜索。常见选择:FAISS(高性能)、Milvus(分布式)、ChromaDB(轻量级)。"},
    {"instruction": "对比BERT和GPT", "input": "", "output": "BERT是encoder-only架构，双向注意力，适合理解任务(分类/NER/QA)。GPT是decoder-only架构，单向注意力，适合生成任务(对话/续写)。两者都基于Transformer。"},
    {"instruction": "什么是Prompt工程", "input": "", "output": "Prompt工程是设计和优化LLM输入提示的技术。主要方法:Zero-shot(直接提问)、Few-shot(提供示例)、CoT(链式思维推理)。好的Prompt能显著提升输出质量。"},
    {"instruction": "解释量化技术", "input": "", "output": "量化将模型参数从高精度(如fp32)降低到低精度(如int4)。NF4量化用于QLoRA微调，GPTQ用于推理部署。4-bit量化可减少75%内存，精度损失极小。"},
    {"instruction": "如何评估RAG系统", "input": "", "output": "RAG评估维度:检索质量(命中率/MRR/NDCG)和生成质量(忠实度/相关性)。推荐框架:RAGAS。优化方向:chunk_size调优、混合检索、重排序。"},
]

# 验证数据
eval_data = [
    {"instruction": "什么是自注意力机制", "input": "", "output": "自注意力机制让序列中每个位置关注所有其他位置。通过Query、Key、Value三个矩阵计算注意力权重，捕获序列内部的依赖关系。"},
    {"instruction": "LoRA的秩r如何选择", "input": "", "output": "LoRA秩r的推荐:小任务r=8,一般任务r=16,复杂任务r=32-64。r越大效果越好但参数越多。通常r=16是良好的起点。alpha通常设为2*r。"},
    {"instruction": "什么是BM25算法", "input": "", "output": "BM25是基于词频的文本检索算法，考虑词频(TF)、逆文档频率(IDF)和文档长度归一化。参数k1控制词频饱和(默认1.5)，b控制长度归一化(默认0.75)。"},
]

print(f"  训练数据: {len(train_data)} 条")
print(f"  验证数据: {len(eval_data)} 条")

# 统计
output_lens = [len(d['output']) for d in train_data]
print(f"  平均输出长度: {np.mean(output_lens):.0f}字")

# ============================================================
# 4. 配置LoRA
# ============================================================
print("\n--- 4. 配置LoRA ---")

# 模拟模型配置
model_config = {
    "name": "Qwen2.5-7B (模拟)",
    "hidden_size": 4096,
    "num_layers": 32,
    "num_heads": 32,
    "vocab_size": 152064,
    "total_params": 7e9,
}

lora_config = {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "lora_dropout": 0.05,
    "task_type": "CAUSAL_LM",
}

training_config = {
    "num_epochs": 3,
    "batch_size": 4,
    "gradient_accumulation": 4,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.1,
    "lr_scheduler": "cosine",
    "max_seq_length": 512,
    "fp16": True,
}

# 估算参数量
d = model_config['hidden_size']
n_layers = model_config['num_layers']
n_targets = len(lora_config['target_modules'])
r = lora_config['r']

lora_params = n_layers * n_targets * 2 * d * r
total_params = model_config['total_params']

print(f"  模型: {model_config['name']}")
print(f"  LoRA秩 r: {r}, alpha: {lora_config['lora_alpha']}")
print(f"  目标模块: {lora_config['target_modules']}")
print(f"  LoRA参数: {lora_params:,} ({lora_params/total_params*100:.3f}%)")
print(f"  训练轮数: {training_config['num_epochs']}")
print(f"  学习率: {training_config['learning_rate']}")
print(f"  有效batch_size: {training_config['batch_size'] * training_config['gradient_accumulation']}")

# ============================================================
# 5. 模拟训练
# ============================================================
print("\n--- 5. 模拟训练 ---")


def simulate_training(n_epochs, n_steps_per_epoch):
    """模拟训练过程"""
    np.random.seed(42)
    train_losses = []
    eval_losses = []
    learning_rates = []

    total_steps = n_epochs * n_steps_per_epoch
    warmup_steps = int(0.1 * total_steps)

    for epoch in range(n_epochs):
        for step in range(n_steps_per_epoch):
            global_step = epoch * n_steps_per_epoch + step

            # 学习率 (cosine with warmup)
            if global_step < warmup_steps:
                lr = 2e-4 * global_step / warmup_steps
            else:
                progress = (global_step - warmup_steps) / (total_steps - warmup_steps)
                lr = 2e-4 * 0.5 * (1 + np.cos(np.pi * progress))

            learning_rates.append(lr)

            # 训练loss (逐渐下降)
            base = 2.0 * np.exp(-0.15 * global_step / n_steps_per_epoch) + 0.3
            noise = np.random.randn() * 0.03 * max(0.1, 1 - global_step / total_steps)
            train_losses.append(max(0.2, base + noise))

        # 每epoch评估
        eval_base = 2.2 * np.exp(-0.12 * (epoch + 1)) + 0.35
        eval_losses.append(eval_base + np.random.randn() * 0.02)

    return train_losses, eval_losses, learning_rates


train_losses, eval_losses, learning_rates = simulate_training(
    n_epochs=training_config['num_epochs'],
    n_steps_per_epoch=len(train_data) // training_config['batch_size']
)

print(f"  训练步数: {len(train_losses)}")
print(f"  初始Loss: {train_losses[0]:.4f}")
print(f"  最终Loss: {train_losses[-1]:.4f}")
print(f"  最终Eval Loss: {eval_losses[-1]:.4f}")
print(f"  Loss下降: {(train_losses[0] - train_losses[-1]) / train_losses[0] * 100:.1f}%")

# ============================================================
# 6. 评估
# ============================================================
print("\n--- 6. 评估 ---")
print("""
  评估维度:
    1) 自动指标: Perplexity, BLEU, ROUGE
    2) LLM评估: 使用更强的模型评分
    3) 人工评估: 相关性、准确性、完整性
""")


def simulate_evaluation(base_scores, finetuned_scores):
    """模拟评估"""
    metrics = ['相关性', '准确性', '完整性', '流畅性', '专业度']

    print(f"\n  {'指标':<10s} {'微调前':>8s} {'微调后':>8s} {'提升':>8s}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
    for i, metric in enumerate(metrics):
        before = base_scores[i]
        after = finetuned_scores[i]
        improve = after - before
        print(f"  {metric:<10s} {before:>7.1f}% {after:>7.1f}% {improve:>+7.1f}%")

    avg_before = np.mean(base_scores)
    avg_after = np.mean(finetuned_scores)
    print(f"\n  平均: {avg_before:.1f}% -> {avg_after:.1f}% (提升 {avg_after - avg_before:+.1f}%)")


base_scores = [60, 55, 50, 85, 45]
finetuned_scores = [88, 85, 82, 90, 80]
simulate_evaluation(base_scores, finetuned_scores)

# ============================================================
# 7. 合并保存
# ============================================================
print("\n--- 7. 合并保存 ---")
print("""
  合并步骤:
    1) 加载LoRA权重
    2) 合并到基础模型: W = W + (alpha/r) * B @ A
    3) 保存合并后的模型
    4) (可选) 量化后保存

  实际代码:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B")

    # 加载LoRA
    model = PeftModel.from_pretrained(base_model, "./lora_checkpoint")

    # 合并
    merged_model = model.merge_and_unload()

    # 保存
    merged_model.save_pretrained("./merged_model")
    tokenizer.save_pretrained("./merged_model")

  存储对比:
    基础模型: ~14GB (fp16)
    LoRA权重: ~50MB (r=16)
    合并模型: ~14GB (与基础模型相同大小)
""")

# ============================================================
# 8. 微调前后效果对比
# ============================================================
print("\n--- 8. 微调前后效果对比 ---")

test_cases = [
    {
        "question": "什么是LoRA微调？",
        "before": "LoRA可以指低秩适应(Long Range)的一种技术，在不同领域有不同含义。",
        "after": "LoRA(Low-Rank Adaptation)是参数高效微调方法，通过低秩矩阵分解ΔW=BA来减少可训练参数。只训练0.1-2%的参数，内存需求大幅降低。秩r通常取8-64，alpha/r控制更新强度。训练后可将LoRA权重合并回原模型，推理时无额外开销。",
    },
    {
        "question": "RAG系统的核心组件有哪些？",
        "before": "RAG系统的组件可能包括数据处理和模型等部分。",
        "after": "RAG系统的核心组件: 1)文档处理:加载、清洗、切分 2)嵌入模型:文本向量化(bge-large-zh等) 3)向量数据库:存储和检索(FAISS/Milvus) 4)检索策略:BM25+向量混合检索 5)重排序:Cross-Encoder精排 6)LLM生成:基于上下文生成答案 7)评估系统:监控检索和生成质量。",
    },
    {
        "question": "如何选择LoRA的秩r？",
        "before": "秩的选择取决于具体的应用场景和数据量。",
        "after": "LoRA秩r的选择建议: r=8适合简单任务和小数据集(100-500条); r=16适合一般任务(500-2000条)，是推荐的默认值; r=32-64适合复杂任务和大数据集(>2000条)。alpha通常设为2r。注意r越大参数越多但收益递减，建议从r=16开始实验。",
    },
]

for case in test_cases:
    print(f"\n  Q: {case['question']}")
    print(f"  [微调前] {case['before'][:80]}...")
    print(f"  [微调后] {case['after'][:80]}...")

# ============================================================
# 9. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 子图1: 训练Loss曲线
ax1 = axes[0, 0]
ax1.plot(range(len(train_losses)), train_losses, linewidth=1.5, color='#3498db', alpha=0.8, label='训练Loss')
steps_per_epoch = len(train_losses) // training_config['num_epochs']
for i in range(training_config['num_epochs']):
    ax1.axvline(x=(i + 1) * steps_per_epoch, color='gray', linestyle=':', alpha=0.5)
    ax1.text((i + 1) * steps_per_epoch, max(train_losses) * 0.95, f'Epoch {i+1}', fontsize=8)
ax1.set_xlabel('Step', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('训练Loss曲线', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 子图2: 学习率曲线
ax2 = axes[0, 1]
ax2.plot(range(len(learning_rates)), learning_rates, linewidth=1.5, color='#e74c3c')
ax2.fill_between(range(len(learning_rates)), learning_rates, alpha=0.1, color='#e74c3c')
ax2.set_xlabel('Step', fontsize=12)
ax2.set_ylabel('学习率', fontsize=12)
ax2.set_title('余弦退火学习率', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 子图3: 微调前后评估对比
ax3 = axes[1, 0]
metrics = ['相关性', '准确性', '完整性', '流畅性', '专业度']
x = np.arange(len(metrics))
width = 0.35
ax3.bar(x - width/2, base_scores, width, label='微调前', color='#e74c3c', alpha=0.85, edgecolor='black')
ax3.bar(x + width/2, finetuned_scores, width, label='微调后', color='#2ecc71', alpha=0.85, edgecolor='black')
ax3.set_xticks(x)
ax3.set_xticklabels(metrics, fontsize=10)
ax3.set_ylabel('分数', fontsize=12)
ax3.set_title('微调前后评估对比', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.set_ylim(0, 105)
ax3.grid(True, alpha=0.3, axis='y')

# 子图4: 完整项目流程时间线
ax4 = axes[1, 1]
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.axis('off')
ax4.set_title('微调项目完整流程', fontsize=14, fontweight='bold')

steps = [
    (5, 9.0, "1.选择模型\nQwen2.5-7B", '#3498db'),
    (5, 7.8, "2.准备数据\n10条训练+3条验证", '#2ecc71'),
    (5, 6.6, "3.配置LoRA\nr=16, alpha=32", '#e67e22'),
    (5, 5.4, "4.训练\n3 epochs, lr=2e-4", '#e74c3c'),
    (5, 4.2, "5.评估\n多维度对比", '#9b59b6'),
    (5, 3.0, "6.合并保存\nLoRA -> 基础模型", '#1abc9c'),
    (5, 1.8, "7.部署推理\n量化/服务化", '#f39c12'),
]

for x_pos, y_pos, text, color in steps:
    ax4.add_patch(plt.Rectangle((x_pos - 2.5, y_pos - 0.4), 5.0, 0.8,
                                  facecolor=color, edgecolor='black', alpha=0.3, linewidth=2))
    ax4.text(x_pos, y_pos, text, ha='center', va='center', fontsize=9, fontweight='bold')

for i in range(len(steps) - 1):
    ax4.annotate('', xy=(5, steps[i + 1][1] + 0.4), xytext=(5, steps[i][1] - 0.4),
                 arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W20/finetune_project.png', dpi=150, bbox_inches='tight')
print("\n  图表已保存: finetune_project.png")
plt.close()

# ============================================================
# 10. 项目总结
# ============================================================
print("\n--- 10. 项目总结 ---")
print(f"""
  ================================================
  微调项目完成总结
  ================================================

  模型:     {model_config['name']}
  LoRA配置: r={lora_config['r']}, alpha={lora_config['lora_alpha']}
  训练数据: {len(train_data)} 条
  训练轮数: {training_config['num_epochs']}
  LoRA参数: {lora_params:,} ({lora_params/total_params*100:.3f}%)

  训练结果:
    初始Loss: {train_losses[0]:.4f}
    最终Loss: {train_losses[-1]:.4f}
    Loss下降: {(1 - train_losses[-1]/train_losses[0])*100:.1f}%

  评估提升:
    微调前平均: {np.mean(base_scores):.1f}%
    微调后平均: {np.mean(finetuned_scores):.1f}%
    平均提升:   {np.mean(finetuned_scores) - np.mean(base_scores):+.1f}%

  关键发现:
    1) LoRA仅训练{lora_params/total_params*100:.3f}%参数即可显著提升效果
    2) 微调后AI知识问答的准确性和专业度大幅提高
    3) 训练成本远低于全量微调
  ================================================
""")

print("\n" + "=" * 60)
print("W20-D7 完成! 本节完成了一个完整的LoRA微调项目")
print("恭喜完成第三阶段(大模型工程)全部课程!")
print("=" * 60)
