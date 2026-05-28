### Day 3（周三）：推理优化技术
# KV Cache, Flash Attention, Continuous Batching, Speculative Decoding, Paged Attention

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. KV Cache实现
# ============================================================
print("=" * 60)
print("1. KV Cache (键值缓存)")
print("=" * 60)

print("""
  问题: Transformer自回归生成时, 每步都重新计算所有之前的K和V
  解决: 缓存已计算的Key和Value, 每步只计算新的一个token

  无KV Cache:  生成第N个token需要 O(N^2) 计算
  有KV Cache:  生成第N个token只需 O(N) 计算
""")


class SimpleKVCache:
    """简化版KV Cache"""

    def __init__(self, num_layers=4, num_heads=8, head_dim=64):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.cache = {
            layer: {'key': np.zeros((0, num_heads, head_dim)),
                    'value': np.zeros((0, num_heads, head_dim))}
            for layer in range(num_layers)
        }
        self.memory_usage = 0

    def update(self, layer, new_key, new_value):
        """更新缓存"""
        old_key = self.cache[layer]['key']
        old_value = self.cache[layer]['value']
        self.cache[layer]['key'] = np.concatenate([old_key, new_key], axis=0)
        self.cache[layer]['value'] = np.concatenate([old_value, new_value], axis=0)
        self._update_memory()

    def get(self, layer):
        """获取缓存"""
        return self.cache[layer]['key'], self.cache[layer]['value']

    def _update_memory(self):
        """计算内存使用"""
        total_elements = sum(
            k.shape[0] * k.shape[1] * k.shape[2] +
            v.shape[0] * v.shape[1] * v.shape[2]
            for layer_cache in self.cache.values()
            for k, v in [(layer_cache['key'], layer_cache['value'])]
        )
        self.memory_usage = total_elements * 4 / (1024 ** 2)  # FP32 MB

    def clear(self):
        """清空缓存"""
        for layer in self.cache:
            self.cache[layer]['key'] = np.zeros((0, self.num_heads, self.head_dim))
            self.cache[layer]['value'] = np.zeros((0, self.num_heads, self.head_dim))
        self.memory_usage = 0


# 模拟KV Cache效果
kv_cache = SimpleKVCache(num_layers=4, num_heads=8, head_dim=64)

print("  模拟生成过程:")
seq_lengths = []
memory_usage = []
compute_without_cache = []
compute_with_cache = []

for step in range(1, 21):
    new_key = np.random.randn(1, 8, 64)
    new_value = np.random.randn(1, 8, 64)
    for layer in range(4):
        kv_cache.update(layer, new_key, new_value)

    seq_lengths.append(step)
    memory_usage.append(kv_cache.memory_usage)
    compute_without_cache.append(step ** 2)
    compute_with_cache.append(step)

    if step % 5 == 0:
        print(f"    Step {step:2d}: 序列长度={step}, "
              f"KV Cache内存={kv_cache.memory_usage:.2f}MB, "
              f"计算量比={step ** 2}:{step}")


# ============================================================
# 2. Flash Attention概念
# ============================================================
print("\n" + "=" * 60)
print("2. Flash Attention概念")
print("=" * 60)

print("""
  核心思想: 减少HBM(显存)读写次数, 利用SRAM(片上缓存)

  标准Attention:
    1. 从HBM读取Q,K,V → 计算S=QK^T → 写入HBM
    2. 从HBM读取S → 计算P=softmax(S) → 写入HBM
    3. 从HBM读取P,V → 计算O=Pv → 写入HBM
    总共: O(N^2) 次HBM读写

  Flash Attention:
    - 分块计算(Tiling): 将Q,K,V分成小块
    - 在SRAM中完成softmax和乘法
    - 只需一次HBM写入结果
    - 总共: O(N) 次HBM读写

  加速效果: 2-4x 内存效率, 速度提升显著
""")

# 模拟Flash Attention vs 标准Attention
seq_lens = [128, 256, 512, 1024, 2048, 4096]
standard_mem = [n ** 2 * 4 / (1024 ** 2) for n in seq_lens]  # MB
flash_mem = [n * 64 * 4 / (1024 ** 2) for n in seq_lens]

print(f"  {'序列长度':>10} {'标准Attention':>15} {'Flash Attention':>15} {'节省':>10}")
print(f"  {'-' * 55}")
for i, n in enumerate(seq_lens):
    saving = (1 - flash_mem[i] / standard_mem[i]) * 100
    print(f"  {n:>10} {standard_mem[i]:>12.2f}MB {flash_mem[i]:>12.2f}MB {saving:>9.1f}%")


# ============================================================
# 3. Continuous Batching概念
# ============================================================
print("\n" + "=" * 60)
print("3. Continuous Batching")
print("=" * 60)

print("""
  静态批处理 (Static Batching):
    - 一次性凑齐一批请求, 全部完成后再返回
    - 短序列被长序列拖慢 → GPU利用率低

  连续批处理 (Continuous Batching / Inflight Batching):
    - 某个请求完成后立即替换为新请求
    - 不需要等待所有请求完成
    - GPU利用率大幅提升
""")


class ContinuousBatchingSimulator:
    """连续批处理模拟器"""

    def simulate_static(self, requests, batch_size=4):
        """模拟静态批处理"""
        total_time = 0
        for i in range(0, len(requests), batch_size):
            batch = requests[i:i + batch_size]
            batch_time = max(batch)  # 等最长请求完成
            total_time += batch_time
        return total_time

    def simulate_continuous(self, requests, batch_size=4, slots=None):
        """模拟连续批处理"""
        if slots is None:
            slots = [0] * batch_size
        request_queue = list(requests)
        total_time = 0

        while request_queue or any(s > 0 for s in slots):
            # 填充空闲slot
            for i in range(batch_size):
                if slots[i] <= 0 and request_queue:
                    slots[i] = request_queue.pop(0)

            # 推进一步
            step = min(s for s in slots if s > 0) if any(s > 0 for s in slots) else 0
            slots = [s - step if s > 0 else 0 for s in slots]
            total_time += step

        return total_time


np.random.seed(42)
simulator = ContinuousBatchingSimulator()

# 生成随机请求长度
requests = [np.random.randint(10, 200) for _ in range(20)]

static_time = simulator.simulate_static(requests, batch_size=4)
continuous_time = simulator.simulate_continuous(requests, batch_size=4)

print(f"  请求数: {len(requests)}, 批大小: 4")
print(f"  静态批处理总时间: {static_time}")
print(f"  连续批处理总时间: {continuous_time}")
print(f"  加速比: {static_time / max(continuous_time, 1):.2f}x")


# ============================================================
# 4. Speculative Decoding概念
# ============================================================
print("\n" + "=" * 60)
print("4. Speculative Decoding (推测解码)")
print("=" * 60)

print("""
  核心思想: 用小模型快速猜测多个token, 大模型并行验证

  流程:
    1. Draft Model (小) 快速生成K个候选token
    2. Target Model (大) 一次性验证K个token
    3. 接受正确的token, 拒绝第一个错误的及之后的所有
    4. 如果全部接受, 额外获得一个bonus token

  加速原理:
    - 大模型验证K个token的计算量 ≈ 生成1个token
    - 如果接受率高(>80%), 等效速度提升 2-3x
    - 输出分布与原始模型完全一致 (无损加速)
""")


class SpeculativeDecodingSimulator:
    """推测解码模拟器"""

    def simulate(self, num_tokens=100, draft_k=5, acceptance_rate=0.85):
        """模拟推测解码过程"""
        tokens_generated = 0
        target_calls = 0
        steps = []

        while tokens_generated < num_tokens:
            # Draft: 生成K个候选
            draft_tokens = min(draft_k, num_tokens - tokens_generated)
            target_calls += 1  # 一次验证调用

            # 验证: 每个token有acceptance_rate概率被接受
            accepted = 0
            for i in range(draft_tokens):
                if np.random.random() < acceptance_rate:
                    accepted += 1
                else:
                    break

            # 如果全部接受, 额外获得1个bonus token
            if accepted == draft_tokens:
                accepted += 1

            tokens_generated += accepted
            steps.append({
                'draft': draft_tokens,
                'accepted': accepted,
                'target_calls': 1,
            })

        return {
            'total_tokens': tokens_generated,
            'target_calls': target_calls,
            'steps': steps,
            'speedup': tokens_generated / target_calls,
        }

    def simulate_autoregressive(self, num_tokens=100):
        """标准自回归生成"""
        return {
            'total_tokens': num_tokens,
            'target_calls': num_tokens,
            'speedup': 1.0,
        }


spec_sim = SpeculativeDecodingSimulator()
spec_result = spec_sim.simulate(num_tokens=200, draft_k=5, acceptance_rate=0.85)
auto_result = spec_sim.simulate_autoregressive(200)

print(f"  生成200个token:")
print(f"    自回归: 需要 {auto_result['target_calls']} 次大模型调用")
print(f"    推测解码: 需要 {spec_result['target_calls']} 次大模型调用")
print(f"    加速比: {spec_result['speedup']:.2f}x")


# ============================================================
# 5. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 5.1 KV Cache内存 vs 序列长度
ax1 = axes[0, 0]
ax1.plot(seq_lengths, memory_usage, 'b-o', linewidth=2, label='KV Cache内存')
ax1.set_xlabel('序列长度')
ax1.set_ylabel('内存使用 (MB)')
ax1.set_title('KV Cache内存随序列长度增长')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 5.2 Flash Attention内存对比
ax2 = axes[0, 1]
ax2.plot(seq_lens, standard_mem, 'r-o', linewidth=2, label='标准Attention')
ax2.plot(seq_lens, flash_mem, 'g-s', linewidth=2, label='Flash Attention')
ax2.fill_between(seq_lens, standard_mem, flash_mem, alpha=0.15, color='green')
ax2.set_xlabel('序列长度')
ax2.set_ylabel('内存使用 (MB)')
ax2.set_title('Attention内存对比')
ax2.legend()
ax2.set_yscale('log')

# 5.3 批处理效率对比
ax3 = axes[1, 0]
batch_sizes = [1, 2, 4, 8, 16, 32]
static_throughput = [b * 0.8 for b in batch_sizes]
continuous_throughput = [b * 0.95 for b in batch_sizes]
ax3.plot(batch_sizes, static_throughput, 'ro-', linewidth=2, label='静态批处理')
ax3.plot(batch_sizes, continuous_throughput, 'gs-', linewidth=2, label='连续批处理')
ax3.set_xlabel('批大小')
ax3.set_ylabel('吞吐量 (相对)')
ax3.set_title('批处理方式对比')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 5.4 推测解码加速比
ax4 = axes[1, 1]
acceptance_rates = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
draft_ks = [3, 5, 7]
for k in draft_ks:
    speedups = []
    for rate in acceptance_rates:
        result = spec_sim.simulate(num_tokens=500, draft_k=k, acceptance_rate=rate)
        speedups.append(result['speedup'])
    ax4.plot(acceptance_rates, speedups, 'o-', linewidth=2, label=f'Draft K={k}')

ax4.set_xlabel('接受率')
ax4.set_ylabel('加速比')
ax4.set_title('推测解码: 接受率 vs 加速比')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.suptitle('W24-D3: 推理优化技术', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W24/d3_inference_optimization.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存!")
