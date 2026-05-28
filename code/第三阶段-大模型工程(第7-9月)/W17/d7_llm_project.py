"""
W17-D7 LLM综合项目 (LLM Project)
=================================
加载开源模型(GPT-2 small), 文本生成,
不同解码策略(贪心/beam search/top-k/top-p/temperature), 生成质量对比
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W17-D7 LLM综合项目 (LLM Project)")
print("=" * 60)


# ============================================================
# 1. 解码策略理论
# ============================================================
print("\n--- 1. 解码策略理论 ---")
print("""
  语言模型输出每个位置的概率分布 P(token | context)
  解码策略决定如何从这个分布中选择下一个token

  常见策略:
    1) 贪心搜索 (Greedy):     每步选概率最高的token
    2) Beam Search:            维护beam_size个候选序列
    3) Top-k 采样:             只从概率最高的k个token中采样
    4) Top-p (核采样):         从累积概率>=p的最小token集合中采样
    5) Temperature:            调整分布的尖锐程度
""")


def softmax(x, temperature=1.0):
    """带温度的softmax"""
    x = np.array(x) / temperature
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# ============================================================
# 2. 解码策略实现
# ============================================================
print("\n--- 2. 解码策略实现 ---")


class SimpleDecoder:
    """文本解码器 - 实现各种解码策略"""

    def __init__(self, vocab=None):
        self.vocab = vocab or [f"token_{i}" for i in range(50)]
        self.vocab_size = len(self.vocab)

    def greedy_search(self, model_fn, start_token, max_length=20):
        """贪心搜索: 每步选概率最高的"""
        tokens = [start_token]
        for _ in range(max_length):
            logits = model_fn(tokens)
            next_token = int(np.argmax(logits[-1]))
            tokens.append(next_token)
            if next_token == 2:  # EOS
                break
        return tokens

    def beam_search(self, model_fn, start_token, beam_size=3, max_length=20, length_penalty=1.0):
        """Beam Search"""
        # 每个beam: (token列表, 累积log概率)
        beams = [([start_token], 0.0)]

        for _ in range(max_length):
            all_candidates = []
            for tokens, score in beams:
                if tokens[-1] == 2:  # 已结束
                    all_candidates.append((tokens, score))
                    continue

                logits = model_fn(tokens)
                probs = softmax(logits[-1])
                log_probs = np.log(probs + 1e-10)

                top_k_idx = np.argsort(log_probs)[-beam_size:]
                for idx in top_k_idx:
                    new_tokens = tokens + [int(idx)]
                    new_score = score + log_probs[idx]
                    # 长度惩罚
                    lp = ((5 + len(new_tokens)) / 6) ** length_penalty
                    new_score = new_score / lp
                    all_candidates.append((new_tokens, score + log_probs[idx]))

            # 选择top beam_size个候选
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            beams = all_candidates[:beam_size]

            # 检查是否所有beam都结束
            if all(b[0][-1] == 2 for b in beams):
                break

        return max(beams, key=lambda x: x[1])[0]

    def top_k_sampling(self, model_fn, start_token, k=10, temperature=1.0, max_length=20):
        """Top-k 采样"""
        tokens = [start_token]
        for _ in range(max_length):
            logits = model_fn(tokens)
            probs = softmax(logits[-1], temperature)

            # 只保留top-k
            top_k_idx = np.argsort(probs)[-k:]
            top_k_probs = probs[top_k_idx]
            top_k_probs = top_k_probs / top_k_probs.sum()

            # 采样
            next_token = int(np.random.choice(top_k_idx, p=top_k_probs))
            tokens.append(next_token)
            if next_token == 2:
                break
        return tokens

    def top_p_sampling(self, model_fn, start_token, p=0.9, temperature=1.0, max_length=20):
        """Top-p (核采样)"""
        tokens = [start_token]
        for _ in range(max_length):
            logits = model_fn(tokens)
            probs = softmax(logits[-1], temperature)

            # 按概率降序排列
            sorted_idx = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_idx]
            cumulative_probs = np.cumsum(sorted_probs)

            # 找到累积概率 >= p 的最小集合
            cutoff = np.searchsorted(cumulative_probs, p) + 1
            selected_idx = sorted_idx[:cutoff]
            selected_probs = probs[selected_idx]
            selected_probs = selected_probs / selected_probs.sum()

            next_token = int(np.random.choice(selected_idx, p=selected_probs))
            tokens.append(next_token)
            if next_token == 2:
                break
        return tokens

    def generate_with_repetition_penalty(self, model_fn, start_token,
                                          temperature=0.8, penalty=1.2, max_length=20):
        """带重复惩罚的生成"""
        tokens = [start_token]
        for _ in range(max_length):
            logits = model_fn(tokens)
            probs = softmax(logits[-1], temperature)

            # 对已出现的token施加惩罚
            for t in set(tokens):
                probs[t] = probs[t] / (penalty ** (tokens.count(t)))

            probs = probs / probs.sum()
            next_token = int(np.random.choice(len(probs), p=probs))
            tokens.append(next_token)
            if next_token == 2:
                break
        return tokens


# ============================================================
# 3. 模拟模型
# ============================================================
print("\n--- 3. 模拟语言模型 ---")


class SimulatedLLM:
    """模拟语言模型, 用于演示解码策略"""

    def __init__(self, vocab_size=50, seed=42):
        np.random.seed(seed)
        self.vocab_size = vocab_size
        # 模拟每个token之间的关联
        self.transition = np.random.dirichlet(np.ones(vocab_size), size=vocab_size)
        self.base_logits = np.random.randn(vocab_size)

    def __call__(self, token_ids):
        """模拟前向传播, 返回每个位置的logits"""
        seq_len = len(token_ids)
        logits = np.zeros((seq_len, self.vocab_size))

        for i in range(seq_len):
            if i == 0:
                logits[i] = self.base_logits
            else:
                prev = token_ids[i - 1]
                logits[i] = self.base_logits + 3 * np.log(self.transition[prev] + 1e-10)
        return logits


# 创建模拟模型
model = SimulatedLLM(vocab_size=50)
decoder = SimpleDecoder(vocab=[f"w{i}" for i in range(50)])

# ============================================================
# 4. 对比不同解码策略
# ============================================================
print("\n--- 4. 对比不同解码策略 ---")

np.random.seed(42)
start = 5  # 起始token

# 贪心搜索
greedy_result = decoder.greedy_search(model, start, max_length=15)
print(f"  贪心搜索:     {greedy_result}")

# Beam Search
beam_result = decoder.beam_search(model, start, beam_size=5, max_length=15)
print(f"  Beam Search:  {beam_result}")

# Top-k
np.random.seed(42)
topk_result = decoder.top_k_sampling(model, start, k=10, temperature=1.0, max_length=15)
print(f"  Top-k (k=10): {topk_result}")

# Top-p
np.random.seed(42)
topp_result = decoder.top_p_sampling(model, start, p=0.9, temperature=1.0, max_length=15)
print(f"  Top-p (p=0.9): {topp_result}")

# 不同温度
for temp in [0.3, 0.7, 1.0, 1.5]:
    np.random.seed(42)
    result = decoder.top_k_sampling(model, start, k=10, temperature=temp, max_length=15)
    print(f"  Temp={temp}:  {result}")

# 重复惩罚
np.random.seed(42)
rep_result = decoder.generate_with_repetition_penalty(model, start, temperature=0.8, penalty=1.5, max_length=15)
print(f"  重复惩罚:     {rep_result}")

# ============================================================
# 5. 解码策略可视化
# ============================================================
print("\n--- 5. 解码策略可视化 ---")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 模拟概率分布
vocab_labels = [f"w{i}" for i in range(20)]
base_probs = softmax(np.random.randn(20))

# 子图1: 原始概率分布
ax1 = axes[0, 0]
colors = ['red' if p == max(base_probs) else 'steelblue' for p in base_probs]
ax1.bar(range(20), base_probs, color=colors, edgecolor='black', alpha=0.8)
ax1.set_xticks(range(20))
ax1.set_xticklabels(vocab_labels, fontsize=7, rotation=45)
ax1.set_title('原始概率分布', fontsize=14, fontweight='bold')
ax1.set_ylabel('概率')
ax1.grid(True, alpha=0.3, axis='y')

# 子图2: Top-k 过滤 (k=5)
ax2 = axes[0, 1]
top5_idx = np.argsort(base_probs)[-5:]
top5_probs = base_probs.copy()
mask = np.zeros(20)
mask[top5_idx] = base_probs[top5_idx]
filtered_probs = mask / mask.sum()

colors_topk = ['#e74c3c' if i in top5_idx else '#bdc3c7' for i in range(20)]
ax2.bar(range(20), filtered_probs, color=colors_topk, edgecolor='black', alpha=0.8)
ax2.set_xticks(range(20))
ax2.set_xticklabels(vocab_labels, fontsize=7, rotation=45)
ax2.set_title('Top-k 过滤 (k=5)', fontsize=14, fontweight='bold')
ax2.set_ylabel('概率')
ax2.grid(True, alpha=0.3, axis='y')

# 子图3: Top-p 过滤 (p=0.9)
ax3 = axes[0, 2]
sorted_idx = np.argsort(base_probs)[::-1]
cumsum = np.cumsum(base_probs[sorted_idx])
cutoff = np.searchsorted(cumsum, 0.9) + 1
selected = sorted_idx[:cutoff]
topp_mask = np.zeros(20)
topp_mask[selected] = base_probs[selected]
topp_filtered = topp_mask / topp_mask.sum()

colors_topp = ['#e74c3c' if i in selected else '#bdc3c7' for i in range(20)]
ax3.bar(range(20), topp_filtered, color=colors_topp, edgecolor='black', alpha=0.8)
ax3.set_xticks(range(20))
ax3.set_xticklabels(vocab_labels, fontsize=7, rotation=45)
ax3.set_title('Top-p 过滤 (p=0.9)', fontsize=14, fontweight='bold')
ax3.set_ylabel('概率')
ax3.grid(True, alpha=0.3, axis='y')

# 子图4-6: 不同Temperature
for idx, (temp, ax) in enumerate(zip([0.3, 1.0, 2.0], [axes[1, 0], axes[1, 1], axes[1, 2]])):
    temp_probs = softmax(np.random.randn(20), temperature=temp)
    entropy = -np.sum(temp_probs * np.log(temp_probs + 1e-10))

    colors_temp = ['red' if p == max(temp_probs) else 'steelblue' for p in temp_probs]
    ax.bar(range(20), temp_probs, color=colors_temp, edgecolor='black', alpha=0.8)
    ax.set_xticks(range(20))
    ax.set_xticklabels(vocab_labels, fontsize=7, rotation=45)
    ax.set_title(f'Temperature = {temp} (熵={entropy:.2f})', fontsize=14, fontweight='bold')
    ax.set_ylabel('概率')
    ax.set_ylim(0, max(temp_probs) * 1.2)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W17/decoding_strategies.png', dpi=150, bbox_inches='tight')
print("  图表已保存: decoding_strategies.png")
plt.close()

# ============================================================
# 6. 使用HuggingFace加载GPT-2并生成 (可选)
# ============================================================
print("\n--- 6. 使用 HuggingFace 加载 GPT-2 生成文本 (可选) ---")

try:
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    print("  正在加载 GPT-2 small ...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
    model_hf.eval()

    prompt = "The future of artificial intelligence is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    print(f"\n  输入: '{prompt}'")
    print(f"  Token数: {input_ids.shape[1]}")

    # 对比不同解码策略
    strategies = {
        "贪心 (do_sample=False)": {"do_sample": False, "max_new_tokens": 50},
        "Top-k (k=50, temp=0.7)": {"do_sample": True, "top_k": 50, "temperature": 0.7, "max_new_tokens": 50},
        "Top-p (p=0.9, temp=0.7)": {"do_sample": True, "top_p": 0.9, "temperature": 0.7, "max_new_tokens": 50},
        "高温 (temp=1.5)": {"do_sample": True, "temperature": 1.5, "max_new_tokens": 50},
        "低温 (temp=0.3)": {"do_sample": True, "temperature": 0.3, "max_new_tokens": 50},
    }

    for name, kwargs in strategies.items():
        kwargs["pad_token_id"] = tokenizer.eos_token_id
        with torch.no_grad():
            output = model_hf.generate(input_ids, **kwargs)
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"\n  [{name}]:")
        print(f"    {text[:200]}...")

except ImportError:
    print("  [!] transformers/torch 未安装, 跳过GPT-2演示")
    print("  安装命令: pip install transformers torch")
except Exception as e:
    print(f"  [!] 加载模型出错: {e}")

# ============================================================
# 7. 解码策略选择指南
# ============================================================
print("\n--- 7. 解码策略选择指南 ---")
print("""
  +--------------------+----------------------------+------------------+
  |      策略          |        适用场景            |     特点         |
  +--------------------+----------------------------+------------------+
  | 贪心搜索           | 翻译, 摘要                | 确定性, 可能重复  |
  | Beam Search        | 翻译, 摘要                | 更全局最优        |
  | Top-k + 中等温度   | 通用生成                  | 平衡多样性和质量  |
  | Top-p + 中等温度   | 对话, 创意写作            | 自适应多样性      |
  | 高温度(>1.0)       | 头脑风暴, 创意            | 非常多样         |
  | 低温度(<0.5)       | 代码生成, 事实问答        | 确定性高, 保守    |
  +--------------------+----------------------------+------------------+

  推荐默认参数:
    - 通用生成:   top_p=0.9, temperature=0.7
    - 代码生成:   top_p=0.95, temperature=0.2
    - 创意写作:   top_p=0.95, temperature=1.0
    - 事实问答:   temperature=0.0 (贪心)
""")

# ============================================================
# 8. 生成质量评估指标
# ============================================================
print("\n--- 8. 生成质量评估指标 ---")
print("""
  自动评估指标:
    - Perplexity (困惑度): 模型对文本的预测能力, 越低越好
    - BLEU: 与参考文本的n-gram重叠度 (翻译/摘要)
    - ROUGE: 与参考文本的召回率 (摘要)
    - Distinct-1/2: 生成文本中unique n-gram比例 (多样性)

  人工评估维度:
    - 流畅性 (Fluency)
    - 相关性 (Relevance)
    - 一致性 (Coherence)
    - 事实性 (Factuality)
    - 多样性 (Diversity)
""")


def compute_perplexity(probs_list):
    """计算困惑度"""
    log_probs = [np.log(p + 1e-10) for p in probs_list]
    avg_log_prob = np.mean(log_probs)
    return np.exp(-avg_log_prob)


def compute_distinct(tokens_list, n=1):
    """计算Distinct-n"""
    all_ngrams = []
    for tokens in tokens_list:
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        all_ngrams.extend(ngrams)
    if len(all_ngrams) == 0:
        return 0
    return len(set(all_ngrams)) / len(all_ngrams)


# 模拟不同策略的生成质量
print("\n  模拟生成质量对比:")
strategies_eval = {
    "贪心":        {"perplexity": 18.5, "distinct_1": 0.15, "distinct_2": 0.25},
    "Beam-5":      {"perplexity": 15.2, "distinct_1": 0.18, "distinct_2": 0.30},
    "Top-k=50":    {"perplexity": 22.3, "distinct_1": 0.45, "distinct_2": 0.65},
    "Top-p=0.9":   {"perplexity": 20.8, "distinct_1": 0.50, "distinct_2": 0.70},
    "Temp=0.3":    {"perplexity": 16.1, "distinct_1": 0.25, "distinct_2": 0.40},
    "Temp=1.5":    {"perplexity": 35.7, "distinct_1": 0.65, "distinct_2": 0.85},
}

print(f"  {'策略':<12s} {'困惑度':>8s} {'Distinct-1':>12s} {'Distinct-2':>12s}")
print(f"  {'-'*12} {'-'*8} {'-'*12} {'-'*12}")
for name, metrics in strategies_eval.items():
    print(f"  {name:<12s} {metrics['perplexity']:>8.1f} {metrics['distinct_1']:>12.2f} {metrics['distinct_2']:>12.2f}")

print("\n" + "=" * 60)
print("W17-D7 完成! 本节实现了多种解码策略并对比了生成质量")
print("=" * 60)
