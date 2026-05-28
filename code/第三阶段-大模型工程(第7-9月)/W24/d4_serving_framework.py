### Day 4（周四）：模型推理服务框架
# vLLM, TGI, Triton Inference Server, 性能对比

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 推理服务框架概览
# ============================================================
print("=" * 60)
print("1. LLM推理服务框架对比")
print("=" * 60)

frameworks = {
    'vLLM': {
        '开发方': 'UC Berkeley',
        '核心特性': 'PagedAttention, Continuous Batching',
        '优点': ['高吞吐量', '兼容OpenAI API', '支持多种模型'],
        '缺点': ['内存管理复杂', '某些模型不支持'],
        'GPU要求': 'A100/H100推荐',
    },
    'TGI': {
        '开发方': 'HuggingFace',
        '核心特性': 'Flash Attention, Watermark, 水印',
        '优点': ['易部署(Docker)', '丰富的HuggingFace模型支持'],
        '缺点': ['某些优化不如vLLM', '社区版功能有限'],
        'GPU要求': 'A10G/A100',
    },
    'Triton IS': {
        '开发方': 'NVIDIA',
        '核心特性': '多框架支持, 动态批处理',
        '优点': ['生产级稳定', '多模型编排', '支持ONNX/TensorRT'],
        '缺点': ['配置复杂', 'LLM优化不如vLLM'],
        'GPU要求': 'NVIDIA GPU',
    },
    'llama.cpp': {
        '开发方': '社区 (ggerganov)',
        '核心特性': 'GGUF量化, CPU/GPU混合推理',
        '优点': ['无需GPU', '极低内存', '跨平台'],
        '缺点': ['吞吐量低', '不支持分布式'],
        'GPU要求': '可选 (CPU即可)',
    },
}

for name, info in frameworks.items():
    print(f"\n  {name} ({info['开发方']}):")
    print(f"    核心: {info['核心特性']}")
    print(f"    优点: {', '.join(info['优点'])}")
    print(f"    缺点: {', '.join(info['缺点'])}")


# ============================================================
# 2. 模拟性能对比
# ============================================================
print("\n" + "=" * 60)
print("2. 框架性能对比 (模拟数据)")
print("=" * 60)

# 基于公开benchmark的模拟数据
benchmarks = {
    'Throughput (tokens/s)': {
        'vLLM': 180,
        'TGI': 140,
        'Triton': 120,
        'llama.cpp': 25,
    },
    'TTFT (ms)': {
        'vLLM': 45,
        'TGI': 60,
        'Triton': 55,
        'llama.cpp': 400,
    },
    'QPS': {
        'vLLM': 55,
        'TGI': 42,
        'Triton': 38,
        'llama.cpp': 3,
    },
    'GPU利用率 (%)': {
        'vLLM': 92,
        'TGI': 85,
        'Triton': 88,
        'llama.cpp': 5,
    },
}

print(f"\n  {'指标':<25}", end='')
for fw in frameworks:
    print(f"  {fw:>12}", end='')
print()
print(f"  {'-' * 75}")
for metric, values in benchmarks.items():
    print(f"  {metric:<25}", end='')
    for fw in frameworks:
        val = values.get(fw, 'N/A')
        if isinstance(val, (int, float)):
            print(f"  {val:>12}", end='')
        else:
            print(f"  {val:>12}", end='')
    print()


# ============================================================
# 3. vLLM PagedAttention概念
# ============================================================
print("\n" + "=" * 60)
print("3. vLLM PagedAttention原理")
print("=" * 60)

print("""
  问题: KV Cache内存碎片化, 浪费大量显存

  PagedAttention (灵感来自OS虚拟内存):
    1. 将KV Cache分成固定大小的"页" (blocks)
    2. 使用页表映射逻辑地址→物理地址
    3. 按需分配, 不需要连续内存

  效果:
    - 内存利用率: ~95% (vs 传统 ~60%)
    - 支持更长序列和更多并发请求
    - 类似操作系统管理虚拟内存的方式
""")


class PagedAttentionSimulator:
    """PagedAttention模拟器"""

    def __init__(self, total_pages=100, page_size=16):
        self.total_pages = total_pages
        self.page_size = page_size
        self.free_pages = total_pages
        self.page_tables = {}  # request_id → [page_ids]
        self.allocation_history = []

    def allocate(self, request_id, seq_length):
        """为请求分配KV Cache页"""
        pages_needed = (seq_length + self.page_size - 1) // self.page_size
        if pages_needed > self.free_pages:
            return False  # OOM

        self.free_pages -= pages_needed
        self.page_tables[request_id] = list(range(
            self.total_pages - self.free_pages - pages_needed,
            self.total_pages - self.free_pages,
        ))
        self.allocation_history.append({
            'request': request_id,
            'seq_len': seq_length,
            'pages': pages_needed,
            'free_after': self.free_pages,
        })
        return True

    def deallocate(self, request_id):
        """释放请求的KV Cache页"""
        if request_id in self.page_tables:
            pages = len(self.page_tables[request_id])
            self.free_pages += pages
            del self.page_tables[request_id]

    def get_utilization(self):
        return (self.total_pages - self.free_pages) / self.total_pages


# 模拟内存分配
pa = PagedAttentionSimulator(total_pages=100, page_size=16)
utilization_history = [pa.get_utilization()]

print("\n  模拟请求处理:")
for i in range(20):
    seq_len = np.random.randint(20, 200)
    success = pa.allocate(f"req_{i}", seq_len)
    util = pa.get_utilization()
    utilization_history.append(util)

    if not success:
        print(f"    请求{i}: seq_len={seq_len}, OOM! 释放最早请求...")
        # 模拟释放
        for rid in list(pa.page_tables.keys())[:3]:
            pa.deallocate(rid)
        pa.allocate(f"req_{i}", seq_len)
        utilization_history.append(pa.get_utilization())
    elif i % 5 == 0:
        print(f"    请求{i}: seq_len={seq_len}, 利用率={util:.1%}")


# ============================================================
# 4. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 5.1 吞吐量对比
ax1 = axes[0, 0]
fw_names = list(frameworks.keys())
throughput = [benchmarks['Throughput (tokens/s)'][f] for f in fw_names]
colors = ['#4CAF50', '#2196F3', '#76B900', '#FF9800']
ax1.bar(fw_names, throughput, color=colors, edgecolor='black')
ax1.set_ylabel('Tokens/s')
ax1.set_title('吞吐量对比')
for i, v in enumerate(throughput):
    ax1.text(i, v + 3, str(v), ha='center', fontsize=11)

# 5.2 TTFT对比
ax2 = axes[0, 1]
ttft = [benchmarks['TTFT (ms)'][f] for f in fw_names]
bars = ax2.bar(fw_names, ttft, color=colors, edgecolor='black')
ax2.set_ylabel('TTFT (ms)')
ax2.set_title('首Token延迟对比 (越低越好)')
for bar, v in zip(bars, ttft):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
             f'{v}ms', ha='center', fontsize=10)

# 5.3 综合性能雷达图
ax3 = axes[1, 0]
metric_names = ['吞吐量', '低延迟', 'QPS', 'GPU利用率', '易用性']
num_metrics = len(metric_names)
angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
angles += angles[:1]

for fw, color in zip(['vLLM', 'TGI', 'Triton IS'], ['#4CAF50', '#2196F3', '#76B900']):
    scores = [
        benchmarks['Throughput (tokens/s)'][fw.replace(' IS', '')] / 200 * 5,
        5 - benchmarks['TTFT (ms)'][fw.replace(' IS', '')] / 100,
        benchmarks['QPS'][fw.replace(' IS', '')] / 60 * 5,
        benchmarks['GPU利用率 (%)'][fw.replace(' IS', '')] / 20,
        4 if fw == 'vLLM' else 3,
    ]
    scores += scores[:1]
    ax3.plot(angles, scores, 'o-', linewidth=2, label=fw, color=color)
    ax3.fill(angles, scores, alpha=0.05)

ax3.set_xticks(angles[:-1])
ax3.set_xticklabels(metric_names)
ax3.set_ylim(0, 5.5)
ax3.set_title('框架综合性能')
ax3.legend()

# 5.4 PagedAttention内存利用率
ax4 = axes[1, 1]
ax4.plot(range(len(utilization_history)), utilization_history, 'g-', linewidth=2)
ax4.axhline(y=0.6, color='r', linestyle='--', alpha=0.5, label='传统方法利用率')
ax4.axhline(y=0.95, color='b', linestyle='--', alpha=0.5, label='PagedAttention目标')
ax4.fill_between(range(len(utilization_history)), utilization_history, alpha=0.15, color='green')
ax4.set_xlabel('请求数')
ax4.set_ylabel('内存利用率')
ax4.set_title('PagedAttention内存利用率变化')
ax4.legend()
ax4.set_ylim(0, 1.05)
ax4.grid(True, alpha=0.3)

plt.suptitle('W24-D4: 推理服务框架', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W24/d4_serving_framework.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存!")
