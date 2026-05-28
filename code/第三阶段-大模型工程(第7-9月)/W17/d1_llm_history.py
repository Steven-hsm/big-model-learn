"""
W17-D1 大语言模型发展历史 (LLM History)
========================================
GPT系列/BERT/LaMDA/LLaMA发展历程, Transformer架构回顾,
模型规模对比表, 参数量与性能关系图
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W17-D1 大语言模型发展历史 (LLM History)")
print("=" * 60)

# ============================================================
# 1. Transformer 架构回顾
# ============================================================
print("\n--- 1. Transformer 架构回顾 ---")
print("""
  Transformer (2017, "Attention is All You Need")
  ================================================
  核心创新: 完全基于注意力机制, 抛弃 RNN/CNN

  关键组件:
    1) 多头自注意力 (Multi-Head Self-Attention)
       - Query, Key, Value 线性变换
       - Scaled Dot-Product: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
       - 多头并行捕获不同子空间的信息

    2) 位置编码 (Positional Encoding)
       - 正弦/余弦位置编码: PE(pos, 2i) = sin(pos / 10000^(2i/d))
       - 让模型理解序列中 token 的位置关系

    3) 前馈网络 (Feed-Forward Network)
       - FFN(x) = max(0, xW1 + b1)W2 + b2
       - 两层线性变换 + ReLU 激活

    4) 残差连接 + 层归一化
       - LayerNorm(x + Sublayer(x))
       - 缓解深层网络梯度消失问题

    5) 编码器-解码器结构
       - Encoder: 6层, 自注意力 + FFN
       - Decoder: 6层, 掩码自注意力 + 交叉注意力 + FFN
""")

# ============================================================
# 2. 三大架构流派
# ============================================================
print("\n--- 2. 三大架构流派 ---")
print("""
  +------------------+------------------+------------------+
  |  Encoder-Only    |  Encoder-Decoder |  Decoder-Only    |
  +------------------+------------------+------------------+
  |  BERT系列        |  T5, BART        |  GPT系列         |
  |  双向注意力      |  编码器理解输入  |  单向(因果)注意力 |
  |  适合理解任务    |  解码器生成输出  |  适合生成任务     |
  |  NLU任务为主     |  翻译/摘要等     |  NLG任务为主      |
  |  分类/NER/QA     |  Seq2Seq任务     |  对话/续写        |
  +------------------+------------------+------------------+
""")

# ============================================================
# 3. 里程碑模型时间线
# ============================================================
print("\n--- 3. 里程碑模型时间线 ---")

timeline = [
    (2017, "Transformer",        "Google",         "~65M",    "提出自注意力架构, 奠定基础"),
    (2018, "GPT-1",              "OpenAI",         "117M",    "Decoder-only预训练+微调范式"),
    (2018, "BERT",               "Google",         "340M",    "双向预训练, 刷新11项NLU任务"),
    (2019, "GPT-2",              "OpenAI",         "1.5B",    "展示零样本学习能力, 因安全考虑延迟发布"),
    (2019, "XLNet",              "Google/CMU",     "340M",    "排列语言模型, 结合自回归和自编码优势"),
    (2019, "T5",                 "Google",         "11B",     "Text-to-Text统一框架"),
    (2020, "GPT-3",              "OpenAI",         "175B",    "少样本学习, In-Context Learning"),
    (2020, "Switch Transformer",  "Google",         "1.6T",    "稀疏专家模型, 万亿参数级别"),
    (2021, "LaMDA",              "Google",         "137B",    "对话AI, 强调安全性和事实性"),
    (2022, "PaLM",               "Google",         "540B",    "Pathways系统, 多语言推理能力强"),
    (2022, "ChatGPT",            "OpenAI",         "~175B",   "RLHF对齐, 引爆AI应用热潮"),
    (2023, "GPT-4",              "OpenAI",         "~1.8T*",  "多模态, 推理能力大幅提升(*估计值)"),
    (2023, "LLaMA",              "Meta",           "7B-65B",  "开源LLM, 推动开源社区发展"),
    (2023, "LLaMA 2",            "Meta",           "7B-70B",  "开源商用, 对话版本Chat模型"),
    (2023, "Mistral 7B",         "Mistral AI",     "7B",      "滑动窗口注意力, 性能超LLaMA2-13B"),
    (2023, "Mixtral 8x7B",       "Mistral AI",     "47B",     "MoE架构, 高效推理"),
    (2024, "Gemma",              "Google",         "2B-7B",   "轻量开源, 多语言支持"),
    (2024, "LLaMA 3",            "Meta",           "8B-70B",  "开源新标杆, 多语言多模态"),
]

for year, model, org, params, desc in timeline:
    print(f"  {year} | {model:<22s} | {org:<12s} | {params:<8s} | {desc}")

# ============================================================
# 4. 模型规模对比表
# ============================================================
print("\n--- 4. 模型规模对比表 ---")
print("""
  +------------------+---------+---------+-----------+-----------+
  |       模型       |  参数量 |  训练数据 |  层数(d)  |  隐藏维度  |
  +------------------+---------+---------+-----------+-----------+
  | GPT-1            |   117M  |  5GB    |    12     |    768    |
  | GPT-2            |   1.5B  |  40GB   |    48     |   1600    |
  | GPT-3            |   175B  |  570GB  |    96     |   12288   |
  | BERT-Large       |   340M  |  16GB   |    24     |   1024    |
  | T5-11B           |    11B  |  800GB  |    24     |   2048    |
  | LLaMA-7B         |     7B  |  1TB    |    32     |   4096    |
  | LLaMA-65B        |    65B  |  1.4TB  |    80     |   8192    |
  | LLaMA-3-70B      |    70B  |  15T tok|    80     |   8192    |
  +------------------+---------+---------+-----------+-----------+
""")

# ============================================================
# 5. 参数量与性能关系可视化
# ============================================================
print("\n--- 5. 参数量与性能关系可视化 ---")

# 模型参数量(对数)和基准测试分数(模拟数据)
models_plot = {
    'GPT-1\n(117M)':    (0.117,  38.5),
    'BERT-L\n(340M)':   (0.34,   42.0),
    'GPT-2\n(1.5B)':    (1.5,    48.2),
    'T5-11B':           (11,     55.0),
    'LLaMA-7B':         (7,      58.5),
    'LLaMA-13B':        (13,     62.0),
    'Mistral-7B':       (7,      61.0),
    'LLaMA-70B':        (70,     72.5),
    'GPT-3\n(175B)':    (175,    68.0),
    'LLaMA-3-70B':      (70,     75.0),
    'GPT-4\n(~1.8T)':   (1800,   86.4),
}

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# --- 子图1: 参数量 vs 性能 (散点图) ---
ax1 = axes[0]
params_log = [np.log10(p) for p, _ in models_plot.values()]
scores = [s for _, s in models_plot.values()]
sizes = [max(50, min(500, p * 2)) for p, _ in models_plot.values()]

scatter = ax1.scatter(params_log, scores, s=sizes, alpha=0.7, c=range(len(models_plot)), cmap='viridis', edgecolors='black')
for i, name in enumerate(models_plot.keys()):
    ax1.annotate(name, (params_log[i], scores[i]), fontsize=7, ha='center', va='bottom',
                 textcoords="offset points", xytext=(0, 10))

ax1.set_xlabel('参数量 (log10)', fontsize=12)
ax1.set_ylabel('基准测试分数 (模拟)', fontsize=12)
ax1.set_title('模型参数量与性能关系', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# --- 子图2: 训练数据量增长趋势 ---
ax2 = axes[1]

years = [2018, 2018, 2019, 2020, 2022, 2023, 2023, 2024]
data_sizes = [5, 16, 40, 570, 1000, 1400, 14000, 15000]  # GB
model_names_trend = ['GPT-1', 'BERT', 'GPT-2', 'GPT-3', 'LLaMA', 'LLaMA-2', 'LLaMA-3-8B', 'LLaMA-3-70B']

bars = ax2.bar(range(len(years)), data_sizes, color=plt.cm.Blues(np.linspace(0.3, 0.9, len(years))), edgecolor='black')
ax2.set_xticks(range(len(years)))
ax2.set_xticklabels([f"{m}\n({y})" for m, y in zip(model_names_trend, years)], fontsize=8, rotation=15)
ax2.set_ylabel('训练数据量 (GB/Tokens x10^9)', fontsize=12)
ax2.set_title('LLM训练数据量增长趋势', fontsize=14, fontweight='bold')

for bar, size in zip(bars, data_sizes):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
             f'{size}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W17/llm_history.png', dpi=150, bbox_inches='tight')
print("  图表已保存: llm_history.png")
plt.close()

# ============================================================
# 6. Transformer架构简图 (文字版)
# ============================================================
print("\n--- 6. Transformer 架构简图 ---")
print("""
  输入: "我 喜欢 机器 学习"
           |          |
      Token Embedding  +  Positional Encoding
           |          |
      +----+----------+----+
      |   Multi-Head Attn   |  <-- Encoder Block x N
      |   Add & Norm        |
      |   Feed Forward      |
      |   Add & Norm        |
      +----+----------+----+
                          |
                   +------+------+
                   |             |
              Encoder         Decoder
                   |   Masked Multi-Head Attn
                   |   Cross-Attention (K,V from Encoder)
                   |   Feed Forward
                   |   Linear + Softmax
                   +------> 输出概率分布
                           P(下一个token)
""")

# ============================================================
# 7. LLM关键技术演进
# ============================================================
print("\n--- 7. LLM 关键技术演进 ---")
print("""
  1) 预训练+微调范式 (2018-)
     GPT-1/BERT 开创, 在大规模语料上预训练, 下游任务微调

  2) In-Context Learning (2020-)
     GPT-3 展示, 无需微调, 通过提示词中的示例学习

  3) Instruction Tuning (2022-)
     FLAN, InstructGPT, 用指令数据微调, 提升指令跟随能力

  4) RLHF (2022-)
     ChatGPT 使用, 人类反馈强化学习, 对齐人类偏好

  5) 高效微调 (2022-)
     LoRA, QLoRA, Adapter等, 只训练少量参数即可适配新任务

  6) 多模态 (2023-)
     GPT-4V, LLaVA, 将视觉信息融入语言模型

  7) 长上下文 (2023-)
     128K+ context window, RoPE扩展, 注意力优化

  8) Agent (2024-)
     工具调用, 规划, 自主决策, Multi-Agent协作
""")

# ============================================================
# 8. 中国大模型概览
# ============================================================
print("\n--- 8. 中国大模型概览 ---")
chinese_models = [
    ("DeepSeek-V3",     "DeepSeek",   "671B(MoE)", "开源, 推理能力强"),
    ("Qwen2.5-72B",     "阿里云",     "72B",       "多语言, 代码能力强"),
    ("GLM-4",           "智谱AI",     "130B",      "中英双语, 工具调用"),
    ("Yi-Large",        "零一万物",   "~100B",     "李开复团队"),
    ("Baichuan2",       "百川智能",   "7B-53B",    "开源, 中文优化"),
    ("ChatGLM3-6B",     "智谱AI",     "6B",        "轻量开源, 部署友好"),
    ("InternLM2",       "上海AI Lab", "7B-20B",    "开源, 长上下文"),
]

for name, org, params, note in chinese_models:
    print(f"  {name:<18s} | {org:<10s} | {params:<12s} | {note}")

print("\n" + "=" * 60)
print("W17-D1 完成! 本节回顾了LLM发展历史和Transformer架构")
print("=" * 60)
