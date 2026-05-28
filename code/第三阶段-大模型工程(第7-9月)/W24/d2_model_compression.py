### Day 2（周二）：模型压缩方法
# 剪枝(Pruning), 量化(Quantization), 知识蒸馏, 组合方法

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 模型压缩方法概览
# ============================================================
print("=" * 60)
print("1. 模型压缩方法概览")
print("=" * 60)

methods = {
    '剪枝 (Pruning)': {
        '原理': '移除不重要的权重或神经元',
        '类型': ['非结构化剪枝(个别权重)', '结构化剪枝(整个通道/层)'],
        '压缩率': '2x-10x',
        '精度损失': '低-中',
    },
    '量化 (Quantization)': {
        '原理': '降低权重和激活值的精度 (FP32→INT8/INT4)',
        '类型': ['PTQ (训练后量化)', 'QAT (量化感知训练)'],
        '压缩率': '4x-8x',
        '精度损失': '低',
    },
    '知识蒸馏 (Distillation)': {
        '原理': '大模型(Teacher)指导小模型(Student)训练',
        '类型': ['响应蒸馏', '特征蒸馏', '关系蒸馏'],
        '压缩率': '2x-10x',
        '精度损失': '中',
    },
}

for method, info in methods.items():
    print(f"\n  {method}:")
    print(f"    原理: {info['原理']}")
    print(f"    类型: {', '.join(info['类型'])}")
    print(f"    压缩率: {info['压缩率']}, 精度损失: {info['精度损失']}")


# ============================================================
# 2. 剪枝 (Pruning) 实现
# ============================================================
print("\n" + "=" * 60)
print("2. 剪枝 (Pruning) 模拟")
print("=" * 60)

class PruningSimulator:
    """剪枝模拟器"""

    def __init__(self, weight_matrix):
        self.original_weights = weight_matrix.copy()
        self.weights = weight_matrix.copy()
        self.pruning_history = []

    def magnitude_pruning(self, sparsity):
        """幅度剪枝: 移除绝对值最小的权重"""
        flat = np.abs(self.weights.flatten())
        threshold = np.percentile(flat, sparsity * 100)
        mask = np.abs(self.weights) >= threshold
        self.weights = self.weights * mask
        actual_sparsity = 1 - np.count_nonzero(self.weights) / self.weights.size
        self.pruning_history.append({
            'target_sparsity': sparsity,
            'actual_sparsity': actual_sparsity,
            'nonzero': np.count_nonzero(self.weights),
        })
        return actual_sparsity

    def compute_accuracy_proxy(self):
        """用权重差异代理精度损失"""
        diff = np.linalg.norm(self.weights - self.original_weights)
        original_norm = np.linalg.norm(self.original_weights)
        return max(1 - diff / original_norm, 0)

    def get_compression_ratio(self):
        nonzero = np.count_nonzero(self.weights)
        total = self.weights.size
        return total / max(nonzero, 1)


np.random.seed(42)
weights = np.random.randn(100, 100)
pruner = PruningSimulator(weights)

sparsities = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
pruning_results = []

print(f"\n  {'目标稀疏度':>10} {'实际稀疏度':>10} {'非零权重':>10} {'精度代理':>10}")
print(f"  {'-' * 45}")
for sparsity in sparsities:
    pruner = PruningSimulator(np.random.randn(100, 100))
    actual = pruner.magnitude_pruning(sparsity)
    accuracy = pruner.compute_accuracy_proxy()
    ratio = pruner.get_compression_ratio()
    pruning_results.append({
        'sparsity': sparsity,
        'accuracy': accuracy,
        'ratio': ratio,
    })
    print(f"  {sparsity:>10.0%} {actual:>10.2%} "
          f"{pruner.pruning_history[-1]['nonzero']:>10} {accuracy:>10.4f}")


# ============================================================
# 3. 量化 (Quantization) 实现
# ============================================================
print("\n" + "=" * 60)
print("3. 量化 (Quantization) 模拟")
print("=" * 60)

class QuantizationSimulator:
    """量化模拟器"""

    def __init__(self, weights_fp32):
        self.fp32 = weights_fp32.copy()

    def quantize_ptq(self, bits=8):
        """训练后量化 (Post-Training Quantization)"""
        w_min = self.fp32.min()
        w_max = self.fp32.max()
        scale = (w_max - w_min) / (2 ** bits - 1)
        zero_point = -w_min / scale
        quantized = np.round(self.fp32 / scale + zero_point).astype(np.int32)
        quantized = np.clip(quantized, 0, 2 ** bits - 1)
        # 反量化
        dequantized = (quantized - zero_point) * scale
        return dequantized, scale, zero_point

    def compute_error(self, quantized):
        """计算量化误差"""
        mse = np.mean((self.fp32 - quantized) ** 2)
        max_error = np.max(np.abs(self.fp32 - quantized))
        return {'mse': mse, 'max_error': max_error}

    def compute_size_ratio(self, original_bits, quantized_bits):
        """计算压缩比"""
        return original_bits / quantized_bits


# 测试不同精度量化
np.random.seed(42)
test_weights = np.random.randn(500).astype(np.float32) * 2

quantizer = QuantizationSimulator(test_weights)
quant_configs = [
    ('FP32', 32), ('FP16', 16), ('INT8', 8), ('INT4', 4), ('INT2', 2),
]

print(f"\n  {'精度':>6} {'MSE':>12} {'最大误差':>12} {'压缩比':>8}")
print(f"  {'-' * 42}")

quant_results = []
for name, bits in quant_configs:
    if name == 'FP32':
        deq = test_weights
        error = {'mse': 0, 'max_error': 0}
        ratio = 1.0
    else:
        deq, scale, zp = quantizer.quantize_ptq(bits)
        error = quantizer.compute_error(deq)
        ratio = quantizer.compute_size_ratio(32, bits)

    quant_results.append({
        'name': name, 'bits': bits,
        'mse': error['mse'], 'max_error': error['max_error'],
        'ratio': ratio,
    })
    print(f"  {name:>6} {error['mse']:>12.6f} {error['max_error']:>12.6f} {ratio:>8.1f}x")


# ============================================================
# 4. 知识蒸馏模拟
# ============================================================
print("\n" + "=" * 60)
print("4. 知识蒸馏 (Knowledge Distillation)")
print("=" * 60)

class DistillationSimulator:
    """知识蒸馏模拟器"""

    def __init__(self, teacher_size=768, student_size=256):
        self.teacher_size = teacher_size
        self.student_size = student_size
        self.teacher_logits = None
        self.student_logits = None

    def teacher_predict(self, inputs, num_classes=10):
        """Teacher模型预测 (大模型)"""
        np.random.seed(42)
        logits = np.random.randn(len(inputs), num_classes)
        # Teacher的预测更准确 (softmax后更尖锐)
        probs = self._softmax(logits, temperature=1.0)
        self.teacher_logits = logits
        return probs

    def student_predict(self, inputs, num_classes=10):
        """Student模型预测 (小模型)"""
        np.random.seed(43)
        logits = np.random.randn(len(inputs), num_classes)
        probs = self._softmax(logits, temperature=1.0)
        self.student_logits = logits
        return probs

    def distillation_loss(self, teacher_probs, student_probs, temperature=3.0, alpha=0.7):
        """蒸馏损失 = alpha * KL(T_soft || S_soft) + (1-alpha) * CE(label, S)"""
        # Soft targets
        teacher_soft = self._softmax(self.teacher_logits, temperature)
        student_soft = self._softmax(self.student_logits, temperature)

        # KL散度
        kl_div = np.sum(teacher_soft * (np.log(teacher_soft + 1e-8) - np.log(student_soft + 1e-8)))

        # 模拟CE损失
        ce_loss = np.random.uniform(0.5, 1.5)

        total_loss = alpha * kl_div + (1 - alpha) * ce_loss
        return total_loss

    @staticmethod
    def _softmax(x, temperature=1.0):
        exp_x = np.exp((x - np.max(x, axis=-1, keepdims=True)) / temperature)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


distiller = DistillationSimulator(teacher_size=768, student_size=256)
inputs = np.random.randn(32, 100)

teacher_probs = distiller.teacher_predict(inputs)
student_probs = distiller.student_predict(inputs)

# 模拟蒸馏训练过程
print(f"\n  模型大小对比: Teacher={distiller.teacher_size}d, Student={distiller.student_size}d")
print(f"  压缩比: {distiller.teacher_size / distiller.student_size:.1f}x")
print(f"\n  蒸馏训练模拟 (温度=3.0):")
for epoch in range(1, 11):
    loss = distiller.distillation_loss(teacher_probs, student_probs, temperature=3.0)
    loss *= (1 - epoch * 0.05)  # 模拟损失下降
    acc = min(0.5 + epoch * 0.04, 0.92)
    print(f"    Epoch {epoch:2d}: Loss={loss:.4f}, Acc={acc:.1%}")


# ============================================================
# 5. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 5.1 剪枝: 稀疏度 vs 精度
ax1 = axes[0, 0]
spars = [r['sparsity'] for r in pruning_results]
accs = [r['accuracy'] for r in pruning_results]
ax1.plot(spars, accs, 'ro-', linewidth=2, markersize=8)
ax1.set_xlabel('稀疏度')
ax1.set_ylabel('精度代理')
ax1.set_title('剪枝: 稀疏度 vs 精度')
ax1.set_ylim(0, 1.1)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='可接受阈值')
ax1.legend()

# 5.2 量化: 不同精度的误差
ax2 = axes[0, 1]
quant_names = [r['name'] for r in quant_results]
mses = [r['mse'] for r in quant_results]
colors_q = ['#4CAF50', '#8BC34A', '#FF9800', '#F44336', '#9C27B0']
ax2.bar(quant_names, mses, color=colors_q, edgecolor='black')
ax2.set_ylabel('MSE')
ax2.set_title('量化精度 vs 误差')

# 5.3 压缩方法综合对比
ax3 = axes[1, 0]
comp_methods = ['原始FP32', '剪枝50%', 'INT8量化', 'INT4量化', '蒸馏1/3', '蒸馏+量化']
compression_ratios = [1, 2, 4, 8, 3, 12]
accuracy_retention = [1.0, 0.95, 0.97, 0.90, 0.88, 0.85]

ax3_twin = ax3.twinx()
l1 = ax3.bar(range(len(comp_methods)), compression_ratios, color='#2196F3',
             alpha=0.6, edgecolor='black', label='压缩比')
l2 = ax3_twin.plot(range(len(comp_methods)), accuracy_retention, 'ro-', linewidth=2,
                   markersize=10, label='精度保留率')
ax3.set_xticks(range(len(comp_methods)))
ax3.set_xticklabels(comp_methods, fontsize=8, rotation=15)
ax3.set_ylabel('压缩比', color='blue')
ax3_twin.set_ylabel('精度保留率', color='red')
ax3_twin.set_ylim(0.7, 1.05)
ax3.set_title('压缩方法综合对比')
lines = [l1] + l2
ax3.legend(lines, [l.get_label() for l in lines])

# 5.4 模型大小 vs 性能
ax4 = axes[1, 1]
model_sizes = [0.5, 1, 3, 7, 13, 30, 70, 175]  # 十亿参数
perf_scores = [0.45, 0.55, 0.65, 0.75, 0.82, 0.88, 0.92, 0.96]
# 量化后
quant_sizes = [s * 0.25 for s in model_sizes]
quant_perf = [p * 0.95 for p in perf_scores]

ax4.plot(model_sizes, perf_scores, 'bo-', linewidth=2, label='FP16原始模型')
ax4.plot(quant_sizes, quant_perf, 'gs-', linewidth=2, label='INT4量化模型')
ax4.set_xlabel('模型大小 (B参数)')
ax4.set_ylabel('性能评分')
ax4.set_title('模型大小 vs 性能 (Scaling Law)')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_xscale('log')

plt.suptitle('W24-D2: 模型压缩方法', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W24/d2_model_compression.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存!")
