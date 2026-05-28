"""
W04-D6 链式法则与反向传播
==========================
内容：
1. 计算图示例：f(x,y,z) = (x+y)*z，手动前向+反向传播
2. 两层神经网络（sigmoid激活，MSE损失）
3. 前向传播
4. 反向传播（链式法则计算所有梯度）
5. 梯度下降更新并验证损失下降
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 计算图示例：f(x,y,z) = (x+y)*z
# ============================================================
print("=" * 60)
print("1. 计算图示例：f(x,y,z) = (x+y)*z")
print("=" * 60)

# 前向传播
x, y, z = -2, 5, -4

# Step 1: q = x + y
q = x + y

# Step 2: f = q * z
f = q * z

print(f"\n  输入: x={x}, y={y}, z={z}")
print(f"\n  === 前向传播 ===")
print(f"    Step 1: q = x + y = {x} + {y} = {q}")
print(f"    Step 2: f = q * z = {q} * {z} = {f}")
print(f"    输出: f = {f}")

# 反向传播
print(f"\n  === 反向传播 ===")

# df/df = 1
grad_f = 1.0
print(f"    初始梯度: ∂f/∂f = {grad_f}")

# df/dq = z, df/dz = q  (乘法节点的反向传播)
grad_q = z * grad_f    # ∂f/∂q = ∂f/∂f * z
grad_z = q * grad_f    # ∂f/∂z = ∂f/∂f * q
print(f"    乘法节点 (f = q*z):")
print(f"      ∂f/∂q = z * ∂f/∂f = {z} * {grad_f} = {grad_q}")
print(f"      ∂f/∂z = q * ∂f/∂f = {q} * {grad_f} = {grad_z}")

# df/dx = 1 * dq/dx = 1, df/dy = 1 * dq/dy = 1  (加法节点的反向传播)
grad_x = 1.0 * grad_q  # ∂f/∂x = ∂f/∂q * ∂q/∂x
grad_y = 1.0 * grad_q  # ∂f/∂y = ∂f/∂q * ∂q/∂y
print(f"    加法节点 (q = x+y):")
print(f"      ∂f/∂x = ∂f/∂q * ∂q/∂x = {grad_q} * 1 = {grad_x}")
print(f"      ∂f/∂y = ∂f/∂q * ∂q/∂y = {grad_q} * 1 = {grad_y}")

# 数值梯度验证
print(f"\n  === 数值梯度验证 ===")
eps = 1e-5

def f_xyz(x, y, z):
    return (x + y) * z

num_grad_x = (f_xyz(x + eps, y, z) - f_xyz(x - eps, y, z)) / (2 * eps)
num_grad_y = (f_xyz(x, y + eps, z) - f_xyz(x, y - eps, z)) / (2 * eps)
num_grad_z = (f_xyz(x, y, z + eps) - f_xyz(x, y, z - eps)) / (2 * eps)

print(f"    ∂f/∂x: 解析={grad_x:.6f}, 数值={num_grad_x:.6f}, "
      f"误差={abs(grad_x - num_grad_x):.2e}")
print(f"    ∂f/∂y: 解析={grad_y:.6f}, 数值={num_grad_y:.6f}, "
      f"误差={abs(grad_y - num_grad_y):.2e}")
print(f"    ∂f/∂z: 解析={grad_z:.6f}, 数值={num_grad_z:.6f}, "
      f"误差={abs(grad_z - num_grad_z):.2e}")

# 绘制计算图
fig, ax = plt.subplots(figsize=(12, 5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 5)

# 节点
nodes = {
    'x': (1, 4), 'y': (1, 2), 'z': (1, 0),
    '+': (4, 3), '*': (7, 2), 'f': (10, 2)
}

for name, (nx, ny) in nodes.items():
    if name in ['+', '*']:
        ax.add_patch(plt.Circle((nx, ny), 0.4, fill=True, color='lightblue',
                                 ec='black', linewidth=2))
    else:
        ax.add_patch(plt.Circle((nx, ny), 0.4, fill=True, color='lightyellow',
                                 ec='black', linewidth=2))
    ax.text(nx, ny, name, ha='center', va='center', fontsize=12, fontweight='bold')

# 前向传播箭头（蓝色）
ax.annotate('', xy=(3.6, 3.3), xytext=(1.4, 3.9),
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
ax.annotate('', xy=(3.6, 2.7), xytext=(1.4, 2.1),
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
ax.annotate('', xy=(6.6, 2.3), xytext=(4.4, 2.8),
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
ax.annotate('', xy=(6.6, 1.7), xytext=(1.4, 0.3),
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
ax.annotate('', xy=(9.6, 2), xytext=(7.4, 2),
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))

# 标注前向值和反向梯度
ax.text(2.5, 3.9, f'x={x}', fontsize=10, color='blue', ha='center')
ax.text(2.5, 2.1, f'y={y}', fontsize=10, color='blue', ha='center')
ax.text(4, 1.2, f'z={z}', fontsize=10, color='blue', ha='center')
ax.text(5.5, 3.0, f'q={q}', fontsize=10, color='blue', ha='center')
ax.text(8.5, 2.5, f'f={f}', fontsize=10, color='blue', ha='center')

ax.text(2.0, 3.3, f'∂f/∂x={grad_x}', fontsize=8, color='red')
ax.text(2.0, 1.5, f'∂f/∂y={grad_y}', fontsize=8, color='red')
ax.text(2.5, 0.6, f'∂f/∂z={grad_z}', fontsize=8, color='red')
ax.text(5.0, 2.3, f'∂f/∂q={grad_q}', fontsize=8, color='red')

ax.set_title('计算图: f(x,y,z) = (x+y)*z\n蓝色=前向传播, 红色=反向传播梯度')
ax.axis('off')
plt.tight_layout()
plt.savefig('d6_computation_graph.png', dpi=150)
plt.close()
print("\n  [图] 计算图已保存为 d6_computation_graph.png")


# ============================================================
# 2. 两层神经网络
# ============================================================
print("\n" + "=" * 60)
print("2. 两层神经网络（手动前向+反向传播）")
print("=" * 60)


def sigmoid(x):
    """Sigmoid激活函数"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_derivative(a):
    """Sigmoid的导数（给定输出a时）"""
    return a * (1.0 - a)


# 网络参数
np.random.seed(42)

# 输入层维度: 2, 隐藏层维度: 3, 输出层维度: 1
n_input = 2
n_hidden = 3
n_output = 1

# 初始化权重和偏置
W1 = np.random.randn(n_hidden, n_input) * 0.5   # (3, 2)
b1 = np.zeros((n_hidden, 1))                      # (3, 1)
W2 = np.random.randn(n_output, n_hidden) * 0.5   # (1, 3)
b2 = np.zeros((n_output, 1))                      # (1, 1)

print(f"\n  网络结构: {n_input} -> {n_hidden} -> {n_output}")
print(f"  激活函数: sigmoid")
print(f"  损失函数: MSE (均方误差)")

# 训练数据: 简单的AND逻辑门
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T  # (2, 4)
y_train = np.array([[0, 0, 0, 1]])                        # (1, 4)

print(f"\n  训练数据 (AND门):")
print(f"    输入: {X_train.T.tolist()}")
print(f"    标签: {y_train.flatten().tolist()}")


# ---- 训练循环 ----
learning_rate = 5.0
n_epochs = 5000
losses = []

print(f"\n  开始训练 (lr={learning_rate}, epochs={n_epochs})...")

for epoch in range(n_epochs):
    # ===== 前向传播 =====

    # Layer 1: z1 = W1 @ x + b1
    z1 = W1 @ X_train + b1                    # (3, 4)

    # Activation 1: a1 = sigmoid(z1)
    a1 = sigmoid(z1)                           # (3, 4)

    # Layer 2: z2 = W2 @ a1 + b2
    z2 = W2 @ a1 + b2                          # (1, 4)

    # Activation 2: a2 = sigmoid(z2) (输出)
    a2 = sigmoid(z2)                           # (1, 4)

    # Loss: MSE = (1/n) * Σ (a2 - y)²
    m = X_train.shape[1]  # 样本数
    loss = np.mean((a2 - y_train) ** 2)
    losses.append(loss)

    # ===== 反向传播 =====

    # 输出层误差
    # dL/da2 = 2(a2 - y) / m
    da2 = 2 * (a2 - y_train) / m               # (1, 4)

    # 通过 sigmoid 反向传播
    # dL/dz2 = dL/da2 * sigmoid'(z2) = dL/da2 * a2*(1-a2)
    dz2 = da2 * sigmoid_derivative(a2)         # (1, 4)

    # dL/dW2 = dz2 @ a1.T
    dW2 = dz2 @ a1.T                           # (1, 3)

    # dL/db2 = sum(dz2, axis=1, keepdims=True)
    db2 = np.sum(dz2, axis=1, keepdims=True)   # (1, 1)

    # 隐��层误差
    # dL/da1 = W2.T @ dz2
    da1 = W2.T @ dz2                           # (3, 4)

    # 通过 sigmoid 反向传播
    # dL/dz1 = dL/da1 * sigmoid'(z1) = dL/da1 * a1*(1-a1)
    dz1 = da1 * sigmoid_derivative(a1)         # (3, 4)

    # dL/dW1 = dz1 @ X.T
    dW1 = dz1 @ X_train.T                      # (3, 2)

    # dL/db1 = sum(dz1, axis=1, keepdims=True)
    db1_grad = np.sum(dz1, axis=1, keepdims=True)  # (3, 1)

    # ===== 梯度下降更新 =====
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1_grad

    # 打印进度
    if epoch % 1000 == 0 or epoch == n_epochs - 1:
        predictions = (a2 > 0.5).astype(int)
        accuracy = np.mean(predictions == y_train)
        print(f"    Epoch {epoch:>5d}: loss={loss:.6f}, accuracy={accuracy:.2%}")

# 最终结果
print(f"\n  === 训练完成 ===")
print(f"  最终损失: {losses[-1]:.8f}")
print(f"  初始损失: {losses[0]:.8f}")
print(f"  损失下降: {(losses[0] - losses[-1]) / losses[0]:.2%}")

# 验证预测
z1 = W1 @ X_train + b1
a1 = sigmoid(z1)
z2 = W2 @ a1 + b2
a2 = sigmoid(z2)
predictions = (a2 > 0.5).astype(int)

print(f"\n  预测结果:")
print(f"  {'输入':<15s} {'预测值':<12s} {'预测类别':<10s} {'真实类别':<10s} {'正确'}")
print("  " + "-" * 60)
for i in range(X_train.shape[1]):
    inp = X_train[:, i].flatten().tolist()
    pred_val = a2.flatten()[i]
    pred_class = predictions.flatten()[i]
    true_class = y_train.flatten()[i]
    correct = "V" if pred_class == true_class else "X"
    print(f"  {str(inp):<15s} {pred_val:<12.6f} {pred_class:<10d} "
          f"{true_class:<10d} {correct}")

# 绘制损失曲线
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 损失曲线
axes[0].plot(losses, 'b-', linewidth=1)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')
axes[0].set_title('训练损失曲线')
axes[0].grid(True, alpha=0.3)
axes[0].set_yscale('log')

# 损失前100轮（展示初期快速下降）
axes[1].plot(range(min(200, len(losses))), losses[:200], 'b-', linewidth=1.5)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MSE Loss')
axes[1].set_title('训练损失曲线（前200轮）')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('d6_training_loss.png', dpi=150)
plt.close()
print("\n  [图] 训练损失曲线已保存为 d6_training_loss.png")


# ============================================================
# 3. 梯度验证（数值梯度 vs 解析梯度）
# ============================================================
print("\n" + "=" * 60)
print("3. 梯度验证（数值梯度 vs 解析梯度）")
print("=" * 60)


def compute_loss(W1_v, b1_v, W2_v, b2_v, X, y):
    """计算给定参数下的MSE损失"""
    z1_v = W1_v @ X + b1_v
    a1_v = sigmoid(z1_v)
    z2_v = W2_v @ a1_v + b2_v
    a2_v = sigmoid(z2_v)
    return np.mean((a2_v - y) ** 2)


def compute_analytical_gradients(W1_v, b1_v, W2_v, b2_v, X, y):
    """计算解析梯度"""
    # 前向传播
    z1_v = W1_v @ X + b1_v
    a1_v = sigmoid(z1_v)
    z2_v = W2_v @ a1_v + b2_v
    a2_v = sigmoid(z2_v)
    m = X.shape[1]

    # 反向传播
    da2 = 2 * (a2_v - y) / m
    dz2 = da2 * sigmoid_derivative(a2_v)
    dW2_v = dz2 @ a1_v.T
    db2_v = np.sum(dz2, axis=1, keepdims=True)
    da1_v = W2_v.T @ dz2
    dz1_v = da1_v * sigmoid_derivative(a1_v)
    dW1_v = dz1_v @ X.T
    db1_v = np.sum(dz1_v, axis=1, keepdims=True)

    return dW1_v, db1_v, dW2_v, db2_v


eps = 1e-5
ana_dW1, ana_db1, ana_dW2, ana_db2 = compute_analytical_gradients(
    W1, b1, W2, b2, X_train, y_train
)

# 对 W1 的第一个元素做数值梯度验证
print(f"\n  W1[0,0] 的梯度验证:")
W1_plus = W1.copy()
W1_plus[0, 0] += eps
W1_minus = W1.copy()
W1_minus[0, 0] -= eps
num_grad = (compute_loss(W1_plus, b1, W2, b2, X_train, y_train) -
            compute_loss(W1_minus, b1, W2, b2, X_train, y_train)) / (2 * eps)
ana_grad = ana_dW1[0, 0]
print(f"    数值梯度: {num_grad:.8f}")
print(f"    解析梯度: {ana_grad:.8f}")
print(f"    相对误差: {abs(num_grad - ana_grad) / (abs(num_grad) + abs(ana_grad) + 1e-10):.2e}")

# 对 W2 的 [0,1] 元素做验证
print(f"\n  W2[0,1] 的梯度验证:")
W2_plus = W2.copy()
W2_plus[0, 1] += eps
W2_minus = W2.copy()
W2_minus[0, 1] -= eps
num_grad2 = (compute_loss(W1, b1, W2_plus, b2, X_train, y_train) -
             compute_loss(W1, b1, W2_minus, b2, X_train, y_train)) / (2 * eps)
ana_grad2 = ana_dW2[0, 1]
print(f"    数值梯度: {num_grad2:.8f}")
print(f"    解析梯度: {ana_grad2:.8f}")
print(f"    相对误差: {abs(num_grad2 - ana_grad2) / (abs(num_grad2) + abs(ana_grad2) + 1e-10):.2e}")

print(f"\n  梯度验证通过！解析梯度和数值梯度一致。")

print("\n" + "=" * 60)
print("D6 完成！")
print("=" * 60)
