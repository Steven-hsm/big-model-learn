### Day 6（周六）：边缘部署
# ONNX Runtime, GGUF/llama.cpp, 边缘设备部署, 模型格式��换

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 边缘部署概述
# ============================================================
print("=" * 60)
print("1. 边缘部署概述")
print("=" * 60)

edge_platforms = {
    '手机端': {
        '硬件': 'Snapdragon/A16, 8-16GB RAM',
        '模型': '1-3B参数量化模型 (INT4)',
        '工具': 'llama.cpp, MLX, ONNX Runtime Mobile',
        '速度': '5-20 tokens/s',
    },
    '笔记本': {
        '硬件': 'M1/M2/M3, 16-64GB RAM',
        '模型': '7-13B参数量化模型 (Q4/Q5)',
        '工具': 'Ollama, llama.cpp, LM Studio',
        '速度': '15-50 tokens/s',
    },
    '树莓派/Jetson': {
        '硬件': 'ARM CPU / Jetson Nano, 4-8GB RAM',
        '模型': '0.5-3B参数 (Q4)',
        '工具': 'llama.cpp, ONNX Runtime',
        '速度': '2-10 tokens/s',
    },
    '浏览器': {
        '硬件': 'WebAssembly + WebGPU',
        '模型': '0.5-3B参数量化模型',
        '工具': 'WebLLM, Transformers.js',
        '速度': '3-15 tokens/s',
    },
}

for platform, info in edge_platforms.items():
    print(f"\n  {platform}:")
    print(f"    硬件: {info['硬件']}")
    print(f"    模型: {info['模型']}")
    print(f"    工具: {info['工具']}")
    print(f"    速度: {info['速度']}")


# ============================================================
# 2. ONNX Runtime推理模拟
# ============================================================
print("\n" + "=" * 60)
print("2. ONNX Runtime推理")
print("=" * 60)

print("""
  ONNX (Open Neural Network Exchange):
    - 统一的模型格式, 支持跨框架部署
    - PyTorch → ONNX → ONNX Runtime

  转换流程:
    torch.onnx.export(model, dummy_input, "model.onnx")
    # 优化
    session = ort.InferenceSession("model.onnx")
    # 推理
    outputs = session.run(None, {"input": input_data})
""")


class ONNXRuntimeSimulator:
    """ONNX Runtime推理模拟器"""

    def __init__(self, model_size_mb=100):
        self.model_size = model_size_mb
        self.optimization_level = 'none'

    def set_optimization(self, level):
        """设置优化级别"""
        self.optimization_level = level

    def simulate_inference(self, input_size, num_runs=100):
        """模拟推理性能"""
        np.random.seed(42)

        speedups = {
            'none': 1.0,
            'basic': 1.3,
            'extended': 1.8,
            'all': 2.5,
        }

        base_latency = input_size * 0.01 + self.model_size * 0.001
        speedup = speedups.get(self.optimization_level, 1.0)

        latencies = np.random.normal(base_latency / speedup, 0.01, num_runs)
        latencies = np.maximum(latencies, 0.001)

        return {
            'avg_latency_ms': np.mean(latencies) * 1000,
            'p99_latency_ms': np.percentile(latencies, 99) * 1000,
            'throughput': 1 / np.mean(latencies),
        }

    def convert_from_pytorch(self):
        """模拟PyTorch→ONNX转换"""
        print("  [ONNX] 转换 PyTorch → ONNX...")
        print(f"  [ONNX] 模型大小: {self.model_size}MB")
        print("  [ONNX] 转换完成!")
        return True


onnx_sim = ONNXRuntimeSimulator(model_size_mb=500)

# 不同优化级别对比
opt_levels = ['none', 'basic', 'extended', 'all']
onnx_results = {}

for level in opt_levels:
    onnx_sim.set_optimization(level)
    result = onnx_sim.simulate_inference(input_size=128)
    onnx_results[level] = result
    print(f"\n  优化级别: {level}")
    print(f"    平均延迟: {result['avg_latency_ms']:.2f}ms")
    print(f"    P99延迟: {result['p99_latency_ms']:.2f}ms")
    print(f"    吞吐量: {result['throughput']:.1f} req/s")


# ============================================================
# 3. GGUF格式与llama.cpp
# ============================================================
print("\n" + "=" * 60)
print("3. GGUF格式与llama.cpp")
print("=" * 60)

print("""
  GGUF (GPT-Generated Unified Format):
    - llama.cpp使用的模型格式
    - 支持多种量化级别: Q2_K, Q3_K, Q4_K_M, Q5_K, Q8_0
    - 单文件分发, 包含模型+元数据

  llama.cpp核心特性:
    - 纯C/C++实现, 无Python依赖
    - 支持CPU/AVX2/ARM NEON/GPU(CUDA/Metal)
    - 极低的内存占用
    - 支持GGUF量化格式

  量化等级对比:
    Q4_K_M:  4bit量化, ~60%原始大小, 轻微质量损失
    Q5_K_M:  5bit量化, ~70%原始大小, 较小质量损失
    Q8_0:    8bit量化, ~85%原始大小, 几乎无损
""")


class GGUFQuantizationSimulator:
    """GGUF量化模拟器"""

    def __init__(self, model_params_b=7):
        self.model_params = model_params_b  # 十亿参数

    def estimate_size(self, quant_type):
        """估算模型大小"""
        bytes_per_param = {
            'FP16': 2.0,
            'Q8_0': 1.0,
            'Q5_K_M': 0.70,
            'Q4_K_M': 0.56,
            'Q4_K_S': 0.50,
            'Q3_K_M': 0.42,
            'Q2_K': 0.35,
        }
        bpb = bytes_per_param.get(quant_type, 2.0)
        size_gb = self.model_params * bpb
        return size_gb

    def estimate_performance(self, quant_type):
        """估算性能"""
        # 模拟不同量化级别对推理速度和精度的影响
        configs = {
            'FP16': {'speed': 15, 'perplexity_increase': 0},
            'Q8_0': {'speed': 25, 'perplexity_increase': 0.01},
            'Q5_K_M': {'speed': 35, 'perplexity_increase': 0.05},
            'Q4_K_M': {'speed': 42, 'perplexity_increase': 0.12},
            'Q4_K_S': {'speed': 45, 'perplexity_increase': 0.18},
            'Q3_K_M': {'speed': 50, 'perplexity_increase': 0.35},
            'Q2_K': {'speed': 55, 'perplexity_increase': 0.80},
        }
        return configs.get(quant_type, configs['FP16'])


gguf = GGUFQuantizationSimulator(model_params_b=7)

print(f"\n  7B模型不同量化级别对比:")
print(f"  {'量化':>8} {'大小(GB)':>10} {'速度(tok/s)':>12} {'困惑度增加':>12}")
print(f"  {'-' * 45}")

gguf_results = {}
for qtype in ['FP16', 'Q8_0', 'Q5_K_M', 'Q4_K_M', 'Q3_K_M', 'Q2_K']:
    size = gguf.estimate_size(qtype)
    perf = gguf.estimate_performance(qtype)
    gguf_results[qtype] = {'size': size, **perf}
    print(f"  {qtype:>8} {size:>10.2f} {perf['speed']:>12} {perf['perplexity_increase']:>12.2f}")


# ============================================================
# 4. 模型格式转换流程
# ============================================================
print("\n" + "=" * 60)
print("4. 模型格式转换流程")
print("=" * 60)

print("""
  常见转换路径:

  PyTorch (.bin/.safetensors)
    ├──→ ONNX (.onnx)          → ONNX Runtime
    ├──→ TensorRT (.engine)     → Triton/NVIDIA
    ├──→ GGUF (.gguf)           → llama.cpp
    ├──→ GGML (.bin)            → llama.cpp (旧格式)
    └──→ SafeTensors (.safetensors) → HuggingFace Transformers

  转换命令示例 (概念):
    # PyTorch → ONNX
    python -m onnxruntime.tools.convert model.onnx

    # HF → GGUF
    python convert-hf-to-gguf.py /path/to/model --outtype f16

    # GGUF量化
    ./llama-quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M
""")


# ============================================================
# 5. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 5.1 ONNX优化级别效果
ax1 = axes[0, 0]
opt_names = list(onnx_results.keys())
latencies = [onnx_results[o]['avg_latency_ms'] for o in opt_names]
throughputs = [onnx_results[o]['throughput'] for o in opt_names]

ax1_twin = ax1.twinx()
l1 = ax1.bar(opt_names, latencies, color='#FF9800', alpha=0.7, edgecolor='black', label='延迟(ms)')
l2 = ax1_twin.plot(opt_names, throughputs, 'g-o', linewidth=2, label='吞吐量(req/s)')
ax1.set_ylabel('延迟 (ms)', color='#FF9800')
ax1_twin.set_ylabel('吞吐量 (req/s)', color='green')
ax1.set_title('ONNX Runtime优化级别效果')
lines = [l1] + l2
ax1.legend(lines, [l.get_label() for l in lines])

# 5.2 GGUF量化: 大小 vs 速度
ax2 = axes[0, 1]
qtypes = list(gguf_results.keys())
sizes = [gguf_results[q]['size'] for q in qtypes]
speeds = [gguf_results[q]['speed'] for q in qtypes]

scatter_colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(qtypes)))
for i, (q, sz, sp) in enumerate(zip(qtypes, sizes, speeds)):
    ax2.scatter(sz, sp, s=200, color=scatter_colors[i], edgecolor='black', zorder=5)
    ax2.annotate(q, (sz, sp), textcoords="offset points", xytext=(8, 5), fontsize=9)
ax2.set_xlabel('模型大小 (GB)')
ax2.set_ylabel('推理速度 (tokens/s)')
ax2.set_title('GGUF量化: 大小 vs 速度 (7B模型)')
ax2.grid(True, alpha=0.3)

# 5.3 不同边缘设备性能
ax3 = axes[1, 0]
devices = ['Jetson Nano', 'RPi 5', 'M2 MacBook', 'iPhone 15', 'RTX 4090']
edge_speeds = [3, 5, 35, 12, 150]
edge_memory = [4, 8, 16, 8, 24]
colors_dev = ['#F44336', '#FF9800', '#4CAF50', '#2196F3', '#9C27B0']

ax3_twin = ax3.twinx()
l1 = ax3.bar(devices, edge_speeds, color=colors_dev, alpha=0.7, edgecolor='black', label='速度(tok/s)')
l2 = ax3_twin.plot(devices, edge_memory, 'ro-', linewidth=2, markersize=10, label='内存(GB)')
ax3.set_ylabel('速度 (tokens/s)', color='blue')
ax3_twin.set_ylabel('内存 (GB)', color='red')
ax3.set_title('边缘设备推理性能')
ax3.set_xticklabels(devices, fontsize=9)
lines = [l1] + l2
ax3.legend(lines, [l.get_label() for l in lines])

# 5.4 量化精度损失 vs 压缩比
ax4 = axes[1, 1]
quant_bits = [2, 3, 4, 5, 8, 16]
compression_ratio = [32 / b for b in quant_bits]
quality_retention = [0.75, 0.85, 0.92, 0.95, 0.99, 1.0]

ax4.plot(compression_ratio, quality_retention, 'bo-', linewidth=2, markersize=10)
ax4.fill_between(compression_ratio, quality_retention, alpha=0.15, color='blue')
ax4.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='可接受阈值')
ax4.set_xlabel('压缩比')
ax4.set_ylabel('精度保留率')
ax4.set_title('量化: 压缩比 vs 精度')
ax4.legend()
ax4.grid(True, alpha=0.3)

for i, bits in enumerate(quant_bits):
    ax4.annotate(f'{bits}bit', (compression_ratio[i], quality_retention[i]),
                 textcoords="offset points", xytext=(8, -5), fontsize=9)

plt.suptitle('W24-D6: 边缘部署', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W24/d6_edge_deployment.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存!")
