### Day 1（周一）：模型部署全景
# 云端/边缘/本地部署, 部署架构设计, 性能指标, 部署方案对比

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 模型部署全景
# ============================================================
print("=" * 60)
print("1. LLM部署全景图")
print("=" * 60)

deployment_scenarios = {
    '云端部署': {
        '描述': '在云服务器(GPU实例)上部署模型, 通过API提供服务',
        '优点': '弹性扩展, 高可用, 易维护',
        '缺点': '成本高, 网络延迟, 数据隐私风险',
        '工具': 'vLLM, TGI, Triton, AWS SageMaker',
        '适用': '生产环境, 大规模服务',
    },
    '边缘部署': {
        '描述': '在本地设备(手机/嵌入式)上运行量化后的小模型',
        '优点': '低延迟, 隐私保护, 离线可用',
        '缺点': '模型能力受限, 硬件限制',
        '工具': 'llama.cpp, ONNX Runtime, GGML',
        '适用': '移动端, IoT设备, 隐私敏感场景',
    },
    '本地部署': {
        '描述': '在本地服务器或工作站上部署模型',
        '优点': '数据安全, 可定制, 无网络依赖',
        '缺点': '硬件成本高, 扩展困难',
        '工具': 'Ollama, llama.cpp, text-generation-webui',
        '适用': '企业内部, 开发测试',
    },
    '混合部署': {
        '描述': '大模型在云端, 小模型在边缘, 协同工作',
        '优点': '兼顾性能和成本',
        '缺点': '架构复杂, 同步问题',
        '工具': 'LangChain + 自定义路由',
        '适用': '复杂应用场景',
    },
}

for scenario, info in deployment_scenarios.items():
    print(f"\n  {scenario}:")
    print(f"    描述: {info['描述']}")
    print(f"    优点: {info['优点']}")
    print(f"    缺点: {info['缺点']}")
    print(f"    工具: {info['工具']}")


# ============================================================
# 2. 性能指标
# ============================================================
print("\n" + "=" * 60)
print("2. 关键性能指标")
print("=" * 60)

class PerformanceMetrics:
    """LLM服务性能指标"""

    @staticmethod
    def latency_breakdown(total_ms, ttft_ms, tps):
        """延迟分解"""
        return {
            'TTFT (首token延迟)': ttft_ms,        # Time To First Token
            'TPS (每秒token数)': tps,               # Tokens Per Second
            '总延迟': total_ms,
            '生成token数': int(total_ms / 1000 * tps),
        }

    @staticmethod
    def throughput(qps, avg_latency_ms, concurrency):
        """吞吐量计算"""
        return {
            'QPS (每秒请求数)': qps,
            '平均延迟': f'{avg_latency_ms}ms',
            '并发数': concurrency,
            '理论最大QPS': 1000 / avg_latency_ms * concurrency,
        }

# 模拟不同部署方案的指标
np.random.seed(42)
deployments = {
    'vLLM (A100)': {'ttft': 50, 'tps': 120, 'qps': 50, 'latency': 200},
    'TGI (A10G)': {'ttft': 80, 'tps': 80, 'qps': 30, 'latency': 350},
    'llama.cpp (CPU)': {'ttft': 500, 'tps': 15, 'qps': 2, 'latency': 2000},
    'ONNX (GPU)': {'ttft': 100, 'tps': 90, 'qps': 35, 'latency': 300},
    'Ollama (M2)': {'ttft': 120, 'tps': 40, 'qps': 5, 'latency': 800},
}

print(f"\n  {'方案':<20} {'TTFT(ms)':>10} {'TPS':>8} {'QPS':>8} {'延迟(ms)':>10}")
print(f"  {'-' * 58}")
for name, metrics in deployments.items():
    print(f"  {name:<20} {metrics['ttft']:>10} {metrics['tps']:>8} "
          f"{metrics['qps']:>8} {metrics['latency']:>10}")


# ============================================================
# 3. 部署架构设计
# ============================================================
print("\n" + "=" * 60)
print("3. 典型部署架构")
print("=" * 60)

print("""
  生产环境LLM服务架构:

  ┌──────────┐     ┌──────────┐     ┌──────────────┐     ┌──────────┐
  │  客户端   │ ──→ │ 负载均衡  │ ──→ │ API Gateway  │ ──→ │ 模型服务  │
  │ (Client) │     │(Nginx/LB)│     │  (FastAPI)   │     │  (vLLM)  │
  └──────────┘     └──────────┘     └──────────────┘     └──────────┘
                                          │                    │
                                    ┌─────┴─────┐        ┌────┴────┐
                                    │  请求队列  │        │  GPU池   │
                                    │  (Redis)  │        │(A100/H100)│
                                    └───────────┘        └─────────┘
""")

# 成本估算
class CostEstimator:
    """部署成本估算"""

    def __init__(self):
        self.gpu_prices = {
            'A100 80GB': 2.50,   # $/hour
            'H100 80GB': 4.00,
            'A10G 24GB': 1.50,
            'T4 16GB': 0.70,
        }

    def estimate_monthly(self, gpu_type, num_gpus, hours_per_day=24):
        """月度成本估算"""
        hourly = self.gpu_prices[gpu_type] * num_gpus
        daily = hourly * hours_per_day
        monthly = daily * 30
        return {
            'gpu_type': gpu_type,
            'num_gpus': num_gpus,
            'hourly_cost': hourly,
            'daily_cost': daily,
            'monthly_cost': monthly,
        }

estimator = CostEstimator()
print("  部署成本估算 (按月):")
print(f"  {'GPU类型':<15} {'数量':>4} {'时费($)':>8} {'月费($)':>10}")
print(f"  {'-' * 40}")
for gpu in estimator.gpu_prices:
    for num in [1, 4, 8]:
        est = estimator.estimate_monthly(gpu, num)
        print(f"  {gpu:<15} {num:>4} {est['hourly_cost']:>8.2f} {est['monthly_cost']:>10.0f}")


# ============================================================
# 4. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 5.1 部署方案性能对比
ax1 = axes[0, 0]
names = list(deployments.keys())
ttfts = [deployments[n]['ttft'] for n in names]
colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#E91E63']
ax1.barh(names, ttfts, color=colors, edgecolor='black')
ax1.set_xlabel('首Token延迟 TTFT (ms)')
ax1.set_title('首Token延迟对比 (越低越好)')
for i, v in enumerate(ttfts):
    ax1.text(v + 10, i, f'{v}ms', va='center')

# 5.2 TPS (吞吐量) 对比
ax2 = axes[0, 1]
tps_values = [deployments[n]['tps'] for n in names]
bars = ax2.bar(range(len(names)), tps_values, color=colors, edgecolor='black')
ax2.set_xticks(range(len(names)))
ax2.set_xticklabels([n.split('(')[0].strip() for n in names], fontsize=9)
ax2.set_ylabel('Tokens/秒')
ax2.set_title('生成速度对比 (越高越好)')
for bar, v in zip(bars, tps_values):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
             str(v), ha='center', fontsize=10)

# 5.3 QPS vs 延迟 散点图
ax3 = axes[1, 0]
qps_values = [deployments[n]['qps'] for n in names]
latencies = [deployments[n]['latency'] for n in names]
for i, name in enumerate(names):
    ax3.scatter(latencies[i], qps_values[i], s=200, color=colors[i],
                edgecolor='black', zorder=5)
    ax3.annotate(name.split('(')[0].strip(), (latencies[i], qps_values[i]),
                 textcoords="offset points", xytext=(10, 5), fontsize=9)
ax3.set_xlabel('平均延迟 (ms)')
ax3.set_ylabel('QPS')
ax3.set_title('QPS vs 延迟 (右上角最优)')
ax3.grid(True, alpha=0.3)

# 5.4 月度成本对比
ax4 = axes[1, 1]
gpu_names = list(estimator.gpu_prices.keys())
costs_1gpu = [estimator.gpu_prices[g] * 1 * 24 * 30 for g in gpu_names]
costs_4gpu = [estimator.gpu_prices[g] * 4 * 24 * 30 for g in gpu_names]
x = np.arange(len(gpu_names))
width = 0.35
ax4.bar(x - width / 2, costs_1gpu, width, label='1 GPU', color='#4CAF50', edgecolor='black')
ax4.bar(x + width / 2, costs_4gpu, width, label='4 GPU', color='#2196F3', edgecolor='black')
ax4.set_xticks(x)
ax4.set_xticklabels(gpu_names, fontsize=9)
ax4.set_ylabel('月费 ($)')
ax4.set_title('GPU租用月度成本估算')
ax4.legend()

plt.suptitle('W24-D1: 模型部署全景', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W24/d1_deployment_overview.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存!")
