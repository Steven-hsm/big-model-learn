"""
d5_activation_functions.py - 激活函数
======================================
- 实现 Sigmoid, Tanh, ReLU, LeakyReLU, GELU, Swish
- 2x3 子图绘制 6 种激活函数
- 2x3 子图绘制 6 种导数
- 对比梯度流（展示 Sigmoid/Tanh 梯度消失）
- 打印属性对比表
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 激活函数实现
# ============================================================

def sigmoid(x):
    """Sigmoid: sigma(x) = 1 / (1 + e^{-x})"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_deriv(x):
    """Sigmoid 导数: sigma(x) * (1 - sigma(x))"""
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x):
    """Tanh: tanh(x)"""
    return np.tanh(x)


def tanh_deriv(x):
    """Tanh 导数: 1 - tanh^2(x)"""
    t = np.tanh(x)
    return 1 - t ** 2


def relu(x):
    """ReLU: max(0, x)"""
    return np.maximum(0, x)


def relu_deriv(x):
    """ReLU 导数: 1 if x > 0 else 0"""
    return (x > 0).astype(float)


def leaky_relu(x, alpha=0.01):
    """LeakyReLU: x if x > 0 else alpha * x"""
    return np.where(x > 0, x, alpha * x)


def leaky_relu_deriv(x, alpha=0.01):
    """LeakyReLU 导数"""
    return np.where(x > 0, 1.0, alpha)


def gelu(x):
    """
    GELU: x * Phi(x)
    近似公式: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def gelu_deriv(x):
    """GELU 导数（近似）"""
    sqrt_2_pi = np.sqrt(2 / np.pi)
    inner = sqrt_2_pi * (x + 0.044715 * x ** 3)
    tanh_inner = np.tanh(inner)
    sech2 = 1 - tanh_inner ** 2
    d_inner = sqrt_2_pi * (1 + 3 * 0.044715 * x ** 2)
    return 0.5 * (1 + tanh_inner) + 0.5 * x * sech2 * d_inner


def swish(x, beta=1.0):
    """Swish: x * sigmoid(beta * x)"""
    return x * sigmoid(beta * x)


def swish_deriv(x, beta=1.0):
    """Swish 导数: sigmoid(beta*x) + beta*x*sigmoid(beta*x)*(1-sigmoid(beta*x))"""
    s = sigmoid(beta * x)
    return s + beta * x * s * (1 - s)


# ============================================================
# 2. 打印激活函数属性对比表
# ============================================================
print("=" * 80)
print("激活函数属性对比表")
print("=" * 80)
print(f"{'函数':<12} | {'值域':<18} | {'梯度范围':<20} | {'零中心':<8} | {'主要特点'}")
print("-" * 95)
print(f"{'Sigmoid':<12} | {'(0, 1)':<18} | {'(0, 0.25)':<20} | {'否':<8} | {'梯度消失，用于二分类输出'}")
print(f"{'Tanh':<12} | {'(-1, 1)':<18} | {'(0, 1)':<20} | {'是':<8} | {'零中心化，但仍有梯度消失'}")
print(f"{'ReLU':<12} | {'[0, +inf)':<18} | {'{0, 1}':<20} | {'否':<8} | {'简单高效，但存在死亡神经元'}")
print(f"{'LeakyReLU':<12} | {'(-inf, +inf)':<18} | {'{a, 1}':<20} | {'近似':<8} | {'解决死亡神经元问题'}")
print(f"{'GELU':<12} | {'(-0.17, +inf)':<18} | {'(0, ~1.1)':<20} | {'近似':<8} | {'平滑，Transformer 常用'}")
print(f"{'Swish':<12} | {'(-0.28, +inf)':<18} | {'(0, ~1.1)':<20} | {'近似':<8} | {'自门控，Google 提出'}")


# ============================================================
# 3. 绘制激活函数 (2x3 子图)
# ============================================================
x = np.linspace(-6, 6, 500)

activations = [
    ("Sigmoid", sigmoid, sigmoid_deriv),
    ("Tanh", tanh, tanh_deriv),
    ("ReLU", relu, relu_deriv),
    ("LeakyReLU", leaky_relu, leaky_relu_deriv),
    ("GELU", gelu, gelu_deriv),
    ("Swish", swish, swish_deriv),
]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for idx, (name, func, _) in enumerate(activations):
    row, col = idx // 3, idx % 3
    ax = axes[row, col]

    ax.plot(x, func(x), 'b-', linewidth=2, label=name)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_title(name, fontsize=13, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.legend(fontsize=11)

plt.suptitle("六种常见激活函数", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig("d5_activation_functions.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n图1保存完成: d5_activation_functions.png")


# ============================================================
# 4. 绘制激活函数的导数 (2x3 子图)
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for idx, (name, _, deriv) in enumerate(activations):
    row, col = idx // 3, idx % 3
    ax = axes[row, col]

    ax.plot(x, deriv(x), 'r-', linewidth=2, label=f"{name}'")
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{name} 的导数", fontsize=13, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel("f'(x)")
    ax.legend(fontsize=11)

    # 标注关键梯度值
    if name == "Sigmoid":
        ax.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5)
        ax.text(4, 0.26, '最大梯度=0.25', fontsize=9, color='gray')
    elif name == "Tanh":
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.text(4, 1.02, '最大梯度=1.0', fontsize=9, color='gray')
    elif name == "ReLU":
        ax.text(2, 0.9, 'x>0时梯度恒为1', fontsize=9, color='gray')

plt.suptitle("六种激活函数的导数", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig("d5_activation_derivatives.png", dpi=150, bbox_inches='tight')
plt.show()
print("图2保存完成: d5_activation_derivatives.png")


# ============================================================
# 5. 梯度消失对比
# ============================================================
print("\n" + "=" * 60)
print("梯度消失对比：多层网络中的梯度衰减")
print("=" * 60)

# 模拟：在一个 10 层网络中，观察每层的梯度大小
n_layers = 10
x_init = 1.0  # 初始输入

activations_for_test = {
    "Sigmoid": (sigmoid, sigmoid_deriv),
    "Tanh": (tanh, tanh_deriv),
    "ReLU": (relu, relu_deriv),
    "LeakyReLU": (leaky_relu, leaky_relu_deriv),
    "GELU": (gelu, gelu_deriv),
    "Swish": (swish, swish_deriv),
}

print(f"\n假设输入 x={x_init}，经过 {n_layers} 层网络后各激活函数的梯度倍数:")
print("-" * 60)

gradient_products = {}
for name, (_, deriv) in activations_for_test.items():
    grad = deriv(np.array([x_init]))[0]
    # 梯度经过 n_layers 层后的乘积
    total_grad = grad ** n_layers
    gradient_products[name] = total_grad
    print(f"  {name:<12}: 单层梯度={grad:.6f}, "
          f"{n_layers}层后梯度={total_grad:.10f}")

# 用不同输入值进行更全面的测试
print(f"\n不同输入值下的梯度衰减 (10层后):")
print(f"{'x值':>6} | ", end="")
for name in activations_for_test:
    print(f"{name:>10} | ", end="")
print()
print("-" * 80)

for x_val in [0.5, 1.0, 2.0, 5.0]:
    print(f"{x_val:>6.1f} | ", end="")
    for name, (_, deriv) in activations_for_test.items():
        grad = deriv(np.array([x_val]))[0]
        total = grad ** n_layers
        print(f"{total:>10.2e} | ", end="")
    print()


# ============================================================
# 6. 可视化梯度消失
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 左图：梯度随层数的变化
layers = np.arange(1, 21)
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

for idx, (name, (_, deriv)) in enumerate(activations_for_test.items()):
    grad_at_1 = deriv(np.array([1.0]))[0]
    grads_per_layer = grad_at_1 ** layers
    axes[0].plot(layers, grads_per_layer, color=colors[idx],
                 linewidth=2, label=name, marker='o', markersize=3)

axes[0].set_xlabel('网络层数')
axes[0].set_ylabel('梯度大小 (对数尺度)')
axes[0].set_yscale('log')
axes[0].set_title('梯度随层数的衰减 (输入 x=1)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=1e-5, color='gray', linestyle='--', alpha=0.5, label='梯度消失阈值')

# 右图：各激活函数在 x 范围内的梯度值
for idx, (name, (_, deriv)) in enumerate(activations_for_test.items()):
    axes[1].plot(x, deriv(x), color=colors[idx], linewidth=2, label=name)

axes[1].set_xlabel('x')
axes[1].set_ylabel("f'(x)")
axes[1].set_title('各激活函数的导数分布')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(-0.1, 1.5)
axes[1].axhline(y=0, color='k', linewidth=0.5)

plt.suptitle("激活函数的梯度消失问题对比", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("d5_gradient_vanishing.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n图3保存完成: d5_gradient_vanishing.png")

print("\n" + "=" * 60)
print("总结:")
print("  - Sigmoid: 最大梯度仅 0.25，深层网络梯度消失严重")
print("  - Tanh: 最大梯度 1.0，比 Sigmoid 好，但远离原点时仍有问题")
print("  - ReLU: 正区间梯度恒为 1，有效缓解梯度消失，但存在死亡神经元")
print("  - LeakyReLU: 在负区间也有梯度，解决死亡神经元问题")
print("  - GELU: 平滑的类 ReLU 函数，Transformer 模型常用")
print("  - Swish: 自门控激活函数，Google 提出，理论上限更高")
print("=" * 60)
