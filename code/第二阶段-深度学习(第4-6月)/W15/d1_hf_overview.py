"""
W15-D1 HuggingFace 生态系统概览
================================
介绍 HuggingFace 生态，安装/导入 transformers，
加载预训练模型和 tokenizer，使用 pipeline 进行推理，
打印模型架构摘要
"""

import sys

print("=" * 60)
print("W15-D1 HuggingFace 生态系统概览")
print("=" * 60)

# ============================================================
# 1. HuggingFace 生态简介
# ============================================================
print("\n--- 1. HuggingFace 生态简介 ---")
print("""
HuggingFace 是目前最大的 AI 开源社区和工具平台，
核心组件包括:

  - Transformers:   预训练模型库 (BERT, GPT, T5, LLaMA ...)
  - Datasets:       数据集加载与处理
  - Tokenizers:     高性能分词器
  - Accelerate:     分布式训练加速
  - PEFT:           参数高效微调 (LoRA, QLoRA ...)
  - Diffusers:      扩散模型 (Stable Diffusion ...)
  - Gradio / Spaces: 模型演示与部署

安装命令:
  pip install transformers datasets tokenizers accelerate
  pip install torch torchvision   # PyTorch 后端
""")

# ============================================================
# 2. 安装检查 & 导入
# ============================================================
print("\n--- 2. 安装检查 & 导入 ---")

try:
    import transformers
    print(f"  transformers  版本: {transformers.__version__}")
except ImportError:
    print("  [!] transformers 未安装, 请运行: pip install transformers")
    sys.exit(1)

try:
    import torch
    print(f"  torch         版本: {torch.__version__}")
    print(f"  CUDA 可用:    {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU 设备:     {torch.cuda.get_device_name(0)}")
except ImportError:
    print("  [!] torch 未安装, 请运行: pip install torch")
    sys.exit(1)

try:
    import datasets
    print(f"  datasets      版本: {datasets.__version__}")
except ImportError:
    print("  [!] datasets 未安装, 可选安装: pip install datasets")

try:
    import tokenizers
    print(f"  tokenizers    版本: {tokenizers.__version__}")
except ImportError:
    print("  [!] tokenizers 未安装, 可选安装: pip install tokenizers")

# ============================================================
# 3. 加载预训练模型和 Tokenizer
# ============================================================
print("\n--- 3. 加载预训练模型和 Tokenizer ---")

from transformers import AutoTokenizer, AutoModel

model_name = "distilbert-base-uncased"

print(f"  正在加载模型: {model_name} ...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

print(f"  Tokenizer 类型:   {type(tokenizer).__name__}")
print(f"  模型类型:         {type(model).__name__}")
print(f"  词表大小:         {tokenizer.vocab_size}")
print(f"  模型参数量:       {sum(p.numel() for p in model.parameters()):,}")
print(f"  最大位置编码:     {model.config.max_position_embeddings}")
print(f"  隐藏层维度:       {model.config.dim}")
print(f"  注意力头数:       {model.config.n_heads}")
print(f"  Transformer层数:  {model.config.n_layers}")

# ============================================================
# 4. 使用 Pipeline 进行基本推理
# ============================================================
print("\n--- 4. Pipeline 基本推理 ---")

from transformers import pipeline

# 情感分析 pipeline
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

texts = [
    "I love learning about HuggingFace transformers!",
    "This movie was terrible and a waste of time.",
    "The weather is okay today, nothing special."
]

print("  情感分析结果:")
for text in texts:
    result = classifier(text)[0]
    print(f"    文本: {text}")
    print(f"    标签: {result['label']}, 置信度: {result['score']:.4f}")
    print()

# 特征提取 pipeline (获取句向量)
print("  特征提取 (句子嵌入):")
feature_extractor = pipeline("feature-extraction", model=model_name)
text = "HuggingFace makes NLP easy."
features = feature_extractor(text)
print(f"    输入文本:    {text}")
print(f"    输出形状:    [batch, seq_len, hidden] = {torch.tensor(features[0]).shape}")
print(f"    [CLS] 向量:  {torch.tensor(features[0][0]).shape}")

# ============================================================
# 5. 模型架构摘要
# ============================================================
print("\n--- 5. 模型架构摘要 ---")
print(f"\n  模型名称: {model_name}")
print(f"  架构类型: {model.config.model_type}")
print()

# 逐层打印
print("  模型层级结构:")
for name, module in model.named_children():
    num_params = sum(p.numel() for p in module.parameters())
    print(f"    {name:30s} | 参数量: {num_params:>12,}")

print()
print("  详细模块 (embeddings + transformer layers):")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"    {name:55s} | 形状: {str(list(param.shape)):20s} | 参数: {param.numel():>10,}")

# ============================================================
# 6. 总结
# ============================================================
print("\n--- 6. 总结 ---")
print("""
  本节学习了:
  1) HuggingFace 生态的主要组件 (transformers, datasets, tokenizers, ...)
  2) 使用 AutoTokenizer / AutoModel 加载预训练模型
  3) 使用 pipeline API 进行零代码推理 (sentiment-analysis, feature-extraction)
  4) 查看模型架构和参数统计

  下一步: d2_tokenizer.py - 深入理解分词算法
""")
