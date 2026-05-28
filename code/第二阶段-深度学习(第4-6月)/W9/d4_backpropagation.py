"""
d4_backpropagation.py - 反向传播
=================================
- 2 层网络 [2, 3, 1] 的前向+反向传播
- 手动用链式法则计算所有梯度
- 数值梯度验证解析梯度
- 训练循环：更新权重，展示损失下降
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 激活函数及其导数
# ============================================================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_derivative(a):
    """Sigmoid 的导数，输入为 sigmoid 的输出 a"""
    return a * (1 - a)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(a):
    """ReLU 的导数，输入为 relu 的输出 a"""
    return (a > 0).astype(float)


# ============================================================
# 2. 损失函数及其导数
# ============================================================
def bce_loss(y_true, y_pred, eps=1e-7):
    """二元交叉熵损失"""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def bce_loss_gradient(y_true, y_pred, eps=1e-7):
    """BCE 损失对预测值（输出层激活值）的梯度"""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return (-y_true / y_pred + (1 - y_true) / (1 - y_pred)) / y_true.shape[0]


# ============================================================
# 3. 网络定义与初始化
# ============================================================
print("=" * 60)
print("2 层神经网络 [2 -> 3 -> 1] 反向传播")
print("=" * 60)

np.random.seed(42)

# 网络参数
input_dim, hidden_dim, output_dim = 2, 3, 1

W1 = np.random.randn(input_dim, hidden_dim) * 0.5
b1 = np.zeros((1, hidden_dim))
W2 = np.random.randn(hidden_dim, output_dim) * 0.5
b2 = np.zeros((1, output_dim))

print(f"\n初始化参数:")
print(f"  W1: shape={W1.shape}\n{W1}")
print(f"  b1: shape={b1.shape}\n{b1}")
print(f"  W2: shape={W2.shape}\n{W2}")
print(f"  b2: shape={b2.shape}\n{b2}")


# ============================================================
# 4. 前向传播
# ============================================================
# 训练数据：简单的二分类问题
X = np.array([[0.0, 0.0],
              [0.0, 1.0],
              [1.0, 0.0],
              [1.0, 1.0]])
y = np.array([[0.0], [1.0], [1.0], [0.0]])  # XOR

print(f"\n输入数据 X: shape={X.shape}")
print(f"标签数据 y: shape={y.shape}")

print("\n--- 前向传播 ---")

# 第1层
Z1 = X @ W1 + b1
A1 = sigmoid(Z1)
print(f"Z1 = X @ W1 + b1: shape={Z1.shape}")
print(f"A1 = sigmoid(Z1): shape={A1.shape}")

# 第2层
Z2 = A1 @ W2 + b2
A2 = sigmoid(Z2)
print(f"Z2 = A1 @ W2 + b2: shape={Z2.shape}")
print(f"A2 = sigmoid(Z2): shape={A2.shape} (输出预测)")

# 计算损失
loss = bce_loss(y, A2)
print(f"\n初始损失 (BCE): {loss:.6f}")


# ============================================================
# 5. 反向传播（手动链式法则）
# ============================================================
print("\n--- 反向传播 (链式法则) ---")

# dL/dA2: 损失对输出层激活的梯度
dL_dA2 = bce_loss_gradient(y, A2)
print(f"dL/dA2 (损失对输出的梯度): shape={dL_dA2.shape}")
print(f"  {dL_dA2.flatten()}")

# dL/dZ2 = dL/dA2 * dA2/dZ2 = dL/dA2 * sigmoid'(Z2) = dL/dA2 * A2*(1-A2)
dL_dZ2 = dL_dA2 * sigmoid_derivative(A2)
print(f"\ndL/dZ2 (损失对输出层线性输出的梯度): shape={dL_dZ2.shape}")
print(f"  {dL_dZ2.flatten()}")

# dL/dW2 = A1.T @ dL/dZ2
dL_dW2 = A1.T @ dL_dZ2
print(f"\ndL/dW2 (损失对W2的梯度): shape={dL_dW2.shape}")
print(f"  {dL_dW2.flatten()}")

# dL/db2 = sum(dL/dZ2, axis=0)
dL_db2 = np.sum(dL_dZ2, axis=0, keepdims=True)
print(f"\ndL/db2 (损失对b2的梯度): shape={dL_db2.shape}")
print(f"  {dL_db2.flatten()}")

# 传播到隐藏层
# dL/dA1 = dL/dZ2 @ W2.T
dL_dA1 = dL_dZ2 @ W2.T
print(f"\ndL/dA1 (损失对隐藏层激活的梯度): shape={dL_dA1.shape}")
print(f"  {dL_dA1}")

# dL/dZ1 = dL/dA1 * sigmoid'(A1)
dL_dZ1 = dL_dA1 * sigmoid_derivative(A1)
print(f"\ndL/dZ1 (损失对隐藏层线性输出的梯度): shape={dL_dZ1.shape}")
print(f"  {dL_dZ1}")

# dL/dW1 = X.T @ dL/dZ1
dL_dW1 = X.T @ dL_dZ1
print(f"\ndL/dW1 (损失对W1的梯度): shape={dL_dW1.shape}")
print(f"  {dL_dW1}")

# dL/db1 = sum(dL/dZ1, axis=0)
dL_db1 = np.sum(dL_dZ1, axis=0, keepdims=True)
print(f"\ndL/db1 (损失对b1的梯度): shape={dL_db1.shape}")
print(f"  {dL_db1}")

# 保存解析梯度
analytical_grads = {
    'W1': dL_dW1.copy(),
    'b1': dL_db1.copy(),
    'W2': dL_dW2.copy(),
    'b2': dL_db2.copy(),
}


# ============================================================
# 6. 数值梯度验证
# ============================================================
def compute_loss(params_flat, X, y, shapes, eps=1e-7):
    """根据扁平化的参数计算损失"""
    # 还原参数
    idx = 0
    params = {}
    for name, shape in shapes:
        size = np.prod(shape)
        params[name] = params_flat[idx:idx + size].reshape(shape)
        idx += size

    W1_ = params['W1']; b1_ = params['b1']
    W2_ = params['W2']; b2_ = params['b2']

    Z1_ = X @ W1_ + b1_
    A1_ = sigmoid(Z1_)
    Z2_ = A1_ @ W2_ + b2_
    A2_ = sigmoid(Z2_)
    return bce_loss(y, A2_)


def numerical_gradient(params_flat, X, y, shapes, eps=1e-5):
    """用中心差分法计算数值梯度"""
    grad = np.zeros_like(params_flat)
    for i in range(len(params_flat)):
        params_plus = params_flat.copy()
        params_minus = params_flat.copy()
        params_plus[i] += eps
        params_minus[i] -= eps
        loss_plus = compute_loss(params_plus, X, y, shapes)
        loss_minus = compute_loss(params_minus, X, y, shapes)
        grad[i] = (loss_plus - loss_minus) / (2 * eps)
    return grad


print("\n" + "=" * 60)
print("数值梯度验证")
print("=" * 60)

# 将所有参数扁平化
shapes = [('W1', W1.shape), ('b1', b1.shape), ('W2', W2.shape), ('b2', b2.shape)]
params_flat = np.concatenate([W1.flatten(), b1.flatten(), W2.flatten(), b2.flatten()])

# 计算数值梯度
num_grad = numerical_gradient(params_flat, X, y, shapes, eps=1e-5)

# 还原数值梯度的形状
num_grads = {}
idx = 0
for name, shape in shapes:
    size = np.prod(shape)
    num_grads[name] = num_grad[idx:idx + size].reshape(shape)
    idx += size

# 对比解析梯度和数值梯度
print("\n梯度对比 (解析梯度 vs 数值梯度):")
print("-" * 60)
for name in ['W1', 'b1', 'W2', 'b2']:
    ana = analytical_grads[name]
    num = num_grads[name]
    diff = np.max(np.abs(ana - num))
    rel_diff = diff / (np.max(np.abs(ana)) + 1e-8)
    print(f"\n  {name}:")
    print(f"    解析梯度: {ana.flatten()}")
    print(f"    数值梯度: {num.flatten()}")
    print(f"    最大绝对差异: {diff:.2e}")
    print(f"    相对差异:     {rel_diff:.2e}")
    if rel_diff < 1e-5:
        print(f"    --> 验证通过! (diff < 1e-5)")
    else:
        print(f"    --> 需要检查!")

# 总体差异
ana_flat = np.concatenate([analytical_grads[n].flatten() for n in ['W1', 'b1', 'W2', 'b2']])
total_diff = np.max(np.abs(ana_flat - num_grad))
print(f"\n总体最大绝对差异: {total_diff:.2e}")


# ============================================================
# 7. 训练循环
# ============================================================
print("\n" + "=" * 60)
print("训练循环")
print("=" * 60)

# 重新初始化
np.random.seed(42)
W1 = np.random.randn(input_dim, hidden_dim) * 0.5
b1 = np.zeros((1, hidden_dim))
W2 = np.random.randn(hidden_dim, output_dim) * 0.5
b2 = np.zeros((1, output_dim))

lr = 1.0        # 学习率
epochs = 5000   # 训练轮数
losses = []

for epoch in range(epochs):
    # ---- 前向传播 ----
    Z1 = X @ W1 + b1
    A1 = sigmoid(Z1)
    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2)

    # ---- 计算损失 ----
    loss = bce_loss(y, A2)
    losses.append(loss)

    # ---- 反向传播 ----
    dL_dZ2 = bce_loss_gradient(y, A2) * sigmoid_derivative(A2)
    dL_dW2 = A1.T @ dL_dZ2
    dL_db2 = np.sum(dL_dZ2, axis=0, keepdims=True)

    dL_dA1 = dL_dZ2 @ W2.T
    dL_dZ1 = dL_dA1 * sigmoid_derivative(A1)
    dL_dW1 = X.T @ dL_dZ1
    dL_db1 = np.sum(dL_dZ1, axis=0, keepdims=True)

    # ---- 更新参数 ----
    W1 -= lr * dL_dW1
    b1 -= lr * dL_db1
    W2 -= lr * dL_dW2
    b2 -= lr * dL_db2

    if (epoch + 1) % 500 == 0:
        predictions = (A2 >= 0.5).astype(int)
        accuracy = np.mean(predictions == y)
        print(f"  Epoch {epoch+1:5d}: loss={loss:.6f}, accuracy={accuracy:.2f}")

# 最终预测
Z1_final = X @ W1 + b1
A1_final = sigmoid(Z1_final)
Z2_final = A1_final @ W2 + b2
A2_final = sigmoid(Z2_final)
final_pred = (A2_final >= 0.5).astype(int)

print(f"\n最终预测:")
for i in range(len(X)):
    print(f"  输入: {X[i]} -> 预测概率: {A2_final[i,0]:.4f} -> "
          f"预测: {final_pred[i,0]} (真实: {y[i,0]:.0f})")
print(f"最终损失: {losses[-1]:.6f}")
print(f"最终准确率: {np.mean(final_pred == y):.2f}")


# ============================================================
# 8. 可视化训练过程
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 损失曲线
axes[0].plot(losses, 'b-', linewidth=1)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('BCE Loss')
axes[0].set_title('训练损失曲线')
axes[0].grid(True, alpha=0.3)
axes[0].annotate(f'初始损失: {losses[0]:.4f}', xy=(0, losses[0]),
                 xytext=(500, losses[0] + 0.2), fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='red'))
axes[0].annotate(f'最终损失: {losses[-1]:.4f}', xy=(len(losses)-1, losses[-1]),
                 xytext=(len(losses)-1500, losses[-1] + 0.3), fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='green'))

# 决策边界
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100),
                      np.linspace(-0.5, 1.5, 100))
grid = np.c_[xx.ravel(), yy.ravel()]
Z1_g = grid @ W1 + b1
A1_g = sigmoid(Z1_g)
Z2_g = A1_g @ W2 + b2
A2_g = sigmoid(Z2_g).reshape(xx.shape)

axes[1].contourf(xx, yy, A2_g, levels=20, cmap='RdYlBu', alpha=0.7)
axes[1].contour(xx, yy, A2_g, levels=[0.5], colors='black', linewidths=2)

for cls, marker, color in [(0, 'o', 'red'), (1, 's', 'blue')]:
    mask = (y.flatten() == cls)
    axes[1].scatter(X[mask, 0], X[mask, 1], c=color, marker=marker,
                    s=200, edgecolors='black', linewidth=2, label=f'类别 {cls}', zorder=5)

axes[1].set_xlabel('$x_1$')
axes[1].set_ylabel('$x_2$')
axes[1].set_title('训练后的决策边界')
axes[1].legend()

plt.suptitle("反向传播训练结果", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("d4_backpropagation.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n图保存完成: d4_backpropagation.png")

print("\n" + "=" * 60)
print("总结:")
print("  - 反向传播通过链式法则逐层计算梯度")
print("  - 数值梯度验证确保解析梯度正确 (diff < 1e-5)")
print("  - 训练循环: 前向传播 -> 计算损失 -> 反向传播 -> 更新参数")
print("  - 损失随训练轮数逐步下降，模型学会拟合数据")
print("=" * 60)
