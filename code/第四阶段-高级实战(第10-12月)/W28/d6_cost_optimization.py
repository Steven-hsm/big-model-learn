"""
W28-D6 成本优化
=================
训练成本优化(Spot实例/梯度检查点), 推理成本优化(批处理/缓存), 资源利用率分析
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W28-D6 成本优化")
print("=" * 60)

# ============================================================
# 1. ML成本概述
# ============================================================
print("\n--- 1. ML成本概述 ---")
print("""
  ML成本构成:
  ┌──────────────┬──────────────────────────────────────────┐
  │ 阶段         │ 成本因素                                │
  ├──────────────┼──────────────────────────────────────────┤
  │ 数据准备     │ 存储 + 计算 + 标注人力                  │
  │ 模型训练     │ GPU时 + 存储 + 实验                     │
  │ 模型推理     │ 服务器 + GPU + 网络                     │
  │ 运维监控     │ 人力 + 工具 + 存储                      │
  └──────────────┴──────────────────────────────────────────┘

  成本优化原则:
    1. 只为需要的资源付费
    2. 充分利用已有资源
    3. 选择合适的实例类型
    4. 自动缩容空闲资源
""")

# ============================================================
# 2. 训练成本优化
# ============================================================
print("\n--- 2. 训练成本优化 ---")

# 模拟训练成本数据
training_scenarios = {
    '标准训练 (按需GPU)': {
        'gpu_type': 'A100',
        'gpu_hours': 100,
        'cost_per_hour': 3.5,
        'success_rate': 1.0,
        'total_cost': 350,
    },
    'Spot实例': {
        'gpu_type': 'A100 (Spot)',
        'gpu_hours': 110,  # 多10%因为可能被中断
        'cost_per_hour': 1.0,
        'success_rate': 0.9,
        'total_cost': 110,
    },
    '梯度检查点': {
        'gpu_type': 'A100',
        'gpu_hours': 130,  # 多30%计算时间
        'cost_per_hour': 3.5,
        'success_rate': 1.0,
        'total_cost': 455,
        'note': '但可以用更小的GPU',
    },
    '混合精度 (FP16)': {
        'gpu_type': 'A100',
        'gpu_hours': 55,   # 节省45%时间
        'cost_per_hour': 3.5,
        'success_rate': 1.0,
        'total_cost': 192.5,
    },
    'Spot+混合精度': {
        'gpu_type': 'A100 (Spot)',
        'gpu_hours': 60,
        'cost_per_hour': 1.0,
        'success_rate': 0.9,
        'total_cost': 60,
    },
}

print("\n  训练成本对比:")
print(f"  {'方案':<25} {'GPU时':<8} {'$/小时':<8} {'成功率':<8} {'总成本($)'}")
print("  " + "-" * 65)
for name, s in training_scenarios.items():
    print(f"  {name:<25} {s['gpu_hours']:<8} {s['cost_per_hour']:<8} "
          f"{s['success_rate']:<8.0%} {s['total_cost']:<8.1f}")

print(f"\n  节省最多: Spot+混合精度, 节省 {(1 - 60/350)*100:.0f}%")

print("""
  训练优化策略:
    1. Spot/Preemptible实例: 节省60-70%, 需要断点续训
    2. 混合精度训练: FP16/BF16, 节省40-50%时间
    3. 梯度累积: 小GPU模拟大batch
    4. 梯度检查点: 用时间换空间, 降低内存需求
    5. 数据预取: 避免GPU等待数据
    6. 早停: 不浪费算力在收敛后的epoch
    7. 渐进式训练: 先小模型, 再大模型
""")

# ============================================================
# 3. 推理成本优化
# ============================================================
print("\n--- 3. 推理成本优化 ---")

# 模拟推理成本
inference_scenarios = {
    '实时推理 (GPU)': {
        'instances': 5,
        'cost_per_hour': 3.5,
        'qps': 500,
        'latency_p99': 20,
        'monthly_cost': 5 * 3.5 * 730,
    },
    '实时推理 (CPU优化)': {
        'instances': 10,
        'cost_per_hour': 0.5,
        'qps': 500,
        'latency_p99': 50,
        'monthly_cost': 10 * 0.5 * 730,
    },
    '批量推理': {
        'instances': 2,
        'cost_per_hour': 3.5,
        'qps': 'N/A (批量)',
        'latency_p99': 'N/A (批量)',
        'monthly_cost': 2 * 3.5 * 200,  # 只用部分时间
    },
    'GPU + 缓存': {
        'instances': 3,
        'cost_per_hour': 3.5,
        'qps': 800,  # 缓存命中增加吞吐
        'latency_p99': 15,
        'monthly_cost': 3 * 3.5 * 730,
        'note': '80%缓存命中率',
    },
    '模型量化 (INT8)': {
        'instances': 3,
        'cost_per_hour': 3.5,
        'qps': 700,  # 量化加速
        'latency_p99': 12,
        'monthly_cost': 3 * 3.5 * 730,
    },
}

print("\n  推理成本对比 (月成本):")
print(f"  {'方案':<25} {'实例数':<8} {'QPS':<12} {'月成本($)'}")
print("  " + "-" * 55)
for name, s in inference_scenarios.items():
    qps_str = str(s['qps'])
    print(f"  {name:<25} {s['instances']:<8} {qps_str:<12} {s['monthly_cost']:.0f}")

print("""
  推理优化策略:
    1. 批量推理: 非实时场景, 定时批量处理
    2. 模型缓存: 相同输入复用结果 (Redis)
    3. 模型量化: FP32 -> FP16 -> INT8, 速度快4x
    4. 模型蒸馏: 大模型 -> 小模型, 体积小10x
    5. ONNX Runtime: 通用加速
    6. 自动缩放: 低峰期减少实例
    7. 多模型共享: 同一GPU部署多模型
""")

# ============================================================
# 4. 资源利用率分析
# ============================================================
print("\n--- 4. 资源利用率分析 ---")


class ResourceAnalyzer:
    """资源利用率分析器"""

    def __init__(self):
        self.samples = []

    def record(self, timestamp, gpu_util, cpu_util, memory_util, active_models, total_requests):
        self.samples.append({
            'timestamp': timestamp,
            'gpu_util': gpu_util,
            'cpu_util': cpu_util,
            'memory_util': memory_util,
            'active_models': active_models,
            'total_requests': total_requests,
        })

    def analyze(self):
        """分析利用率"""
        df = pd.DataFrame(self.samples)

        print(f"\n  资源利用率分析 ({len(df)} 个采样点):")
        for col in ['gpu_util', 'cpu_util', 'memory_util']:
            values = df[col]
            print(f"    {col}: mean={values.mean():.1%}, "
                  f"min={values.min():.1%}, max={values.max():.1%}, "
                  f"利用率<50%的比例={((values < 0.5).sum() / len(values)):.1%}")

        # 浪费分析
        avg_gpu = df['gpu_util'].mean()
        if avg_gpu < 0.5:
            potential_saving = (1 - avg_gpu) * 0.5  # 可节省约50%
            print(f"\n    GPU利用率低 ({avg_gpu:.1%}), 优化建议:")
            print(f"    - 减少GPU实例数量, 预计节省 {potential_saving:.0%} GPU成本")
            print(f"    - 考虑合并模型到同一GPU")
            print(f"    - 使用自动缩放")

        return df


# 生成模拟利用率数据
np.random.seed(42)
analyzer = ResourceAnalyzer()

hours = np.arange(24 * 30)  # 30天, 每小时
for h in hours:
    hour_of_day = h % 24
    # 模拟日间高峰
    if 8 <= hour_of_day <= 22:
        gpu_util = np.clip(np.random.normal(0.6, 0.15), 0.05, 0.95)
        cpu_util = np.clip(np.random.normal(0.5, 0.1), 0.05, 0.95)
    else:
        gpu_util = np.clip(np.random.normal(0.2, 0.1), 0.02, 0.5)
        cpu_util = np.clip(np.random.normal(0.15, 0.08), 0.02, 0.5)

    memory_util = np.clip(np.random.normal(0.65, 0.1), 0.3, 0.95)
    active_models = max(1, int(gpu_util * 10))
    total_requests = int(np.random.poisson(100 * (1 + 0.5 * np.sin(hour_of_day * np.pi / 12))))

    analyzer.record(h, gpu_util, cpu_util, memory_util, active_models, total_requests)

df_usage = analyzer.analyze()

# ============================================================
# 5. 成本计算器
# ============================================================
print("\n--- 5. 成本计算器 ---")


class CostCalculator:
    """ML成本计算器"""

    PRICING = {
        'gpu_a100_per_hour': 3.5,
        'gpu_v100_per_hour': 2.5,
        'cpu_per_core_hour': 0.05,
        'storage_per_gb_month': 0.1,
        'spot_discount': 0.3,  # Spot是按需价格的30%
    }

    def calculate_monthly(self, gpu_hours, cpu_core_hours, storage_gb,
                          use_spot=False, auto_scale_saving=0):
        """计算月成本"""
        gpu_price = self.PRICING['gpu_a100_per_hour']
        if use_spot:
            gpu_price *= self.PRICING['spot_discount']

        gpu_cost = gpu_hours * gpu_price
        cpu_cost = cpu_core_hours * self.PRICING['cpu_per_core_hour']
        storage_cost = storage_gb * self.PRICING['storage_per_gb_month']

        total = gpu_cost + cpu_cost + storage_cost
        total *= (1 - auto_scale_saving)

        return {
            'gpu_cost': gpu_cost,
            'cpu_cost': cpu_cost,
            'storage_cost': storage_cost,
            'total': total,
        }

    def compare_scenarios(self):
        """对比场景"""
        scenarios = [
            ('基线(全按需)', False, 0),
            ('Spot实例', True, 0),
            ('Spot+自动缩放', True, 0.3),
            ('优化模型(减半GPU)', False, 0.5),
            ('最优组合', True, 0.4),
        ]

        print(f"\n  月成本对比 (基于1000 GPU时, 5000 CPU核时, 500GB存储):")
        print(f"  {'方案':<25} {'GPU($)':<10} {'CPU($)':<10} {'存储($)':<10} {'总计($)'}")
        print("  " + "-" * 65)

        results = []
        for name, spot, saving in scenarios:
            cost = self.calculate_monthly(1000, 5000, 500, spot, saving)
            results.append((name, cost))
            print(f"  {name:<25} {cost['gpu_cost']:<10.0f} {cost['cpu_cost']:<10.0f} "
                  f"{cost['storage_cost']:<10.0f} {cost['total']:<10.0f}")

        return results


calc = CostCalculator()
results = calc.compare_scenarios()

baseline = results[0][1]['total']
best = results[-1][1]['total']
print(f"\n  最优方案可节省: {(1 - best/baseline)*100:.0f}%")


# ============================================================
# 6. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 左上: 训练成本对比
ax = axes[0, 0]
train_names = list(training_scenarios.keys())
train_costs = [s['total_cost'] for s in training_scenarios.values()]
colors = ['#F44336', '#4CAF50', '#FF9800', '#2196F3', '#9C27B0']
bars = ax.barh(train_names, train_costs, color=colors[:len(train_names)])
ax.set_xlabel('成本 ($)')
ax.set_title('训练成本对比 (单次训练)')
ax.axvline(train_costs[0], color='red', linestyle='--', alpha=0.3)
for bar, cost in zip(bars, train_costs):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
            f'${cost:.0f}', va='center')

# 右上: GPU利用率分布
ax = axes[0, 1]
gpu_utils = [s['gpu_util'] for s in analyzer.samples]
ax.hist(gpu_utils, bins=50, color='#2196F3', alpha=0.7, edgecolor='white')
ax.axvline(0.5, color='red', linestyle='--', label='50%利用率')
ax.axvline(np.mean(gpu_utils), color='green', linestyle='--',
           label=f'均值={np.mean(gpu_utils):.1%}')
ax.set_xlabel('GPU利用率')
ax.set_ylabel('频次')
ax.set_title('GPU利用率分布 (30天)')
ax.legend()

# 左下: 月成本对比
ax = axes[1, 0]
scenario_names = [r[0] for r in results]
total_costs = [r[1]['total'] for r in results]
gpu_costs = [r[1]['gpu_cost'] for r in results]
cpu_costs = [r[1]['cpu_cost'] for r in results]

x = np.arange(len(scenario_names))
width = 0.5
ax.bar(x, gpu_costs, width, label='GPU', color='#2196F3')
ax.bar(x, cpu_costs, width, bottom=gpu_costs, label='CPU', color='#4CAF50')
storage = [r[1]['storage_cost'] for r in results]
ax.bar(x, storage, width, bottom=[g+c for g, c in zip(gpu_costs, cpu_costs)],
       label='存储', color='#FF9800')
ax.set_xticks(x)
ax.set_xticklabels(scenario_names, rotation=30, ha='right', fontsize=8)
ax.set_ylabel('月成本 ($)')
ax.set_title('月成本对比')
ax.legend()

# 右下: 24小时利用率热力图
ax = axes[1, 1]
# 按小时聚合
hourly_gpu = []
for h in range(24):
    mask = [i % 24 == h for i in range(len(analyzer.samples))]
    utils = [analyzer.samples[i]['gpu_util'] for i in range(len(mask)) if mask[i]]
    hourly_gpu.append(np.mean(utils))

# 按天聚合
days = 30
heatmap_data = np.array(hourly_gpu * days).reshape(days, 24)
im = ax.imshow(heatmap_data, cmap='YlGn', aspect='auto', vmin=0, vmax=1)
ax.set_xlabel('小时')
ax.set_ylabel('天数')
ax.set_title('GPU利用率热力图 (30天)')
plt.colorbar(im, ax=ax, label='利用率')

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W28/d6_cost_optimization.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存: d6_cost_optimization.png")

print("\n完成! 成本优化要点:")
print("  1. 训练: Spot实例+混合精度, 可节省70%以上")
print("  2. 推理: 量化+缓存+自动缩放, 降低单位成本")
print("  3. 利用率: 监控并优化低利用率时段")
print("  4. 选择: 合适的实例类型, 不过度配置")
print("  5. 持续: 成本是持续优化的过程, 需要定期审视")
