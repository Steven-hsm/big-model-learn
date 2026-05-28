"""
d2_forward_propagation.py - 前向传播
====================================
- 用 NumPy 搭建 4 层网络 [2, 3, 4, 1]
- 前向传播：ReLU 隐藏层 + Sigmoid 输出层
- 每步打印形状，追踪维度变化
- 与 PyTorch nn.Sequential 对比结果
- 演示非线性激活函数的必要性
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 激活函数定义
# ============================================================
def relu(x):
    """ReLU 激活函数: max(0, x)"""
    return np.maximum(0, x)


def sigmoid(x):
    """Sigmoid 激活函数"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


# ============================================================
# 2. NumPy 前向传播实现
# ============================================================
print("=" * 60)
print("NumPy 前向传播: 4层网络 [2 -> 3 -> 4 -> 1]")
print("=" * 60)

# 网络结构
layer_dims = [2, 3, 4, 1]  # 输入层2, 隐藏层3, 隐藏层4, 输出层1

# 设置随机种子（与 PyTorch 对比时使用相同权重）
np.random.seed(42)

# 初始化权重和偏置（He 初始化）
params = {}
for i in range(1, len(layer_dims)):
    params[f'W{i}'] = np.random.randn(layer_dims[i-1], layer_dims[i]) * np.sqrt(2.0 / layer_dims[i-1])
    params[f'b{i}'] = np.zeros((1, layer_dims[i]))

print("\n网络参数初始化:")
for key, val in params.items():
    print(f"  {key}: shape={val.shape}")

# 输入数据（2 个样本，每个 2 维）
X_np = np.array([[0.5, 0.3],
                  [0.8, -0.2]])
print(f"\n输入数据 X: shape={X_np.shape}")
print(X_np)

# 前向传播 —— 逐层计算
print("\n" + "-" * 50)
print("前向传播逐层追踪:")
print("-" * 50)

cache = {'A0': X_np}  # 保存中间结果

# 第1层: Linear -> ReLU
Z1 = X_np @ params['W1'] + params['b1']
A1 = relu(Z1)
cache['Z1'], cache['A1'] = Z1, A1
print(f"\n第1层 (Linear + ReLU):")
print(f"  Z1 = X @ W1 + b1: shape={Z1.shape}")
print(f"  A1 = ReLU(Z1):    shape={A1.shape}")
print(f"  A1 值:\n{A1}")

# 第2层: Linear -> ReLU
Z2 = A1 @ params['W2'] + params['b2']
A2 = relu(Z2)
cache['Z2'], cache['A2'] = Z2, A2
print(f"\n第2层 (Linear + ReLU):")
print(f"  Z2 = A1 @ W2 + b2: shape={Z2.shape}")
print(f"  A2 = ReLU(Z2):     shape={A2.shape}")
print(f"  A2 值:\n{A2}")

# 第3层: Linear -> Sigmoid
Z3 = A2 @ params['W3'] + params['b3']
A3 = sigmoid(Z3)
cache['Z3'], cache['A3'] = Z3, A3
print(f"\n第3层 (Linear + Sigmoid):")
print(f"  Z3 = A2 @ W3 + b3:  shape={Z3.shape}")
print(f"  A3 = Sigmoid(Z3):   shape={A3.shape}")
print(f"  A3 值（最终输出）:\n{A3}")

# 维度变化总结
print("\n" + "=" * 60)
print("维度变化总结:")
print("=" * 60)
print(f"  输入 X:    {X_np.shape}")
print(f"  Z1, A1:    {Z1.shape}  (2样本, 3神经元)")
print(f"  Z2, A2:    {Z2.shape}  (2样本, 4神经元)")
print(f"  Z3, A3:    {Z3.shape}  (2样本, 1神经元)")


# ============================================================
# 3. PyTorch 等价实现
# ============================================================
print("\n" + "=" * 60)
print("PyTorch nn.Sequential 等价实现")
print("=" * 60)

# 构建 PyTorch 模型
model = nn.Sequential(
    nn.Linear(2, 3),
    nn.ReLU(),
    nn.Linear(3, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)

# 将 NumPy 的权重复制到 PyTorch 模型
with torch.no_grad():
    model[0].weight.data = torch.tensor(params['W1'].T.copy())  # PyTorch: [out, in]
    model[0].bias.data = torch.tensor(params['b1'].flatten().copy())
    model[2].weight.data = torch.tensor(params['W2'].T.copy())
    model[2].bias.data = torch.tensor(params['b2'].flatten().copy())
    model[4].weight.data = torch.tensor(params['W3'].T.copy())
    model[4].bias.data = torch.tensor(params['b3'].flatten().copy())

# 前向传播
X_torch = torch.tensor(X_np, dtype=torch.float32)
output_torch = model(X_torch).detach().numpy()

print(f"\nPyTorch 输出:\n{output_torch}")
print(f"\nNumPy 输出:\n{A3}")
print(f"\n两者差异（应接近 0）:")
print(f"  最大绝对差异: {np.max(np.abs(A3 - output_torch)):.2e}")
print(f"  平均绝对差异: {np.mean(np.abs(A3 - output_torch)):.2e}")


# ============================================================
# 4. 线性堆叠 vs 非线性激活
# ============================================================
print("\n" + "=" * 60)
print("演示：为什么需要非线性激活函数")
print("=" * 60)

# 设定：两个线性层的输出 = 一个线性层
np.random.seed(42)
W1_lin = np.random.randn(2, 3)
b1_lin = np.random.randn(1, 3)
W2_lin = np.random.randn(3, 2)
b2_lin = np.random.randn(1, 2)

X_test = np.array([[1.0, 2.0]])

# 情况1: 两个线性层（不加激活函数）—— 等价于一个线性层
Z1_lin = X_test @ W1_lin + b1_lin          # (1, 3)
Z2_lin = Z1_lin @ W2_lin + b2_lin          # (1, 2)

# 等价的单层变换
W_combined = W1_lin @ W2_lin
b_combined = (b1_lin @ W2_lin) + b2_lin
Z_combined = X_test @ W_combined + b_combined

print(f"\n情况1 - 不使用激活函数（纯线性堆叠）:")
print(f"  两层线性变换结果: {Z2_lin}")
print(f"  单层等价变换结果: {Z_combined}")
print(f"  差异: {np.max(np.abs(Z2_lin - Z_combined)):.2e}")
print(f"  结论: 两个线性层等价于一个线性层！没有增加表达能力")

# 情况2: 加入 ReLU 非线性
Z1_relu = X_test @ W1_lin + b1_lin
A1_relu = relu(Z1_relu)
Z2_relu = A1_relu @ W2_lin + b2_lin

print(f"\n情况2 - 使用 ReLU 激活函数:")
print(f"  带ReLU的两层结果: {Z2_relu}")
print(f"  单层线性结果:     {Z_combined}")
print(f"  差异: {np.max(np.abs(Z2_relu - Z_combined)):.2e}")
print(f"  结论: 非线性激活使网络获得了更强的表达能力！")


# ============================================================
# 5. 可视化：线性 vs 非线性变换
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 生成一些2D数据
np.random.seed(0)
X_vis = np.random.randn(100, 2)

# 线性变换
W_vis = np.array([[1.5, -1.0], [0.5, 2.0]])
X_linear = X_vis @ W_vis

axes[0].scatter(X_vis[:, 0], X_vis[:, 1], c='blue', alpha=0.5, label='原始数据')
axes[0].scatter(X_linear[:, 0], X_linear[:, 1], c='red', alpha=0.5, label='线性变换后')
axes[0].set_title('线性变换\n(数据的相对关系不变)')
axes[0].legend()
axes[0].set_xlabel('$x_1$')
axes[0].set_ylabel('$x_2$')

# 两层线性变换（等价于单层）
X_two_linear = (X_vis @ W_vis) @ np.array([[0.8, 0.3], [-0.5, 1.2]])
axes[1].scatter(X_vis[:, 0], X_vis[:, 1], c='blue', alpha=0.5, label='原始数据')
axes[1].scatter(X_two_linear[:, 0], X_two_linear[:, 1], c='green', alpha=0.5, label='两层线性变换')
axes[1].set_title('两层线性变换\n(仍然只是线性变换)')
axes[1].legend()
axes[1].set_xlabel('$x_1$')
axes[1].set_ylabel('$x_2$')

# 两层非线性变换（加入 ReLU）
X_relu = relu(X_vis @ W_vis) @ np.array([[0.8, 0.3], [-0.5, 1.2]])
axes[2].scatter(X_vis[:, 0], X_vis[:, 1], c='blue', alpha=0.5, label='原始数据')
axes[2].scatter(X_relu[:, 0], X_relu[:, 1], c='orange', alpha=0.5, label='带ReLU变换')
axes[2].set_title('非线性变换 (ReLU)\n(数据结构发生根本变化)')
axes[2].legend()
axes[2].set_xlabel('$x_1$')
axes[2].set_ylabel('$x_2$')

plt.suptitle("前向传播可视化：为什么需要非线性激活函数", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("d2_forward_propagation.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n图保存完成: d2_forward_propagation.png")

print("\n" + "=" * 60)
print("总结:")
print("  - 前向传播 = 逐层计算 Z = A_prev @ W + b, 然后 A = activation(Z)")
print("  - 维度变化: 每层变换样本的维度 (n_samples, d_prev) -> (n_samples, d_curr)")
print("  - NumPy 和 PyTorch 的计算结果完全一致")
print("  - 没有非线性激活函数，多层线性网络等价于单层线性网络")
print("  - 非线性激活函数赋予网络拟合复杂函数的能力")
print("=" * 60)
