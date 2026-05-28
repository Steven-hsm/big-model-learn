"""
d1_perceptron.py - 感知机模型
============================
- 实现感知机类（训练、预测）
- 在 AND、OR、XOR 门上训练
- 可视化决策边界
- 用 MLP 手动解决 XOR 问题
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 感知机类实现
# ============================================================
class Perceptron:
    """单层感知机，使用 Rosenblatt 权重更新规则"""

    def __init__(self, input_dim, lr=0.1, max_epochs=100):
        self.weights = np.zeros(input_dim)        # 权重向量
        self.bias = 0.0                            # 偏置
        self.lr = lr                               # 学习率
        self.max_epochs = max_epochs               # 最大训练轮数

    def predict(self, X):
        """预测：阶跃函数 sign(w·x + b)"""
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, 0)

    def train(self, X, y, verbose=True):
        """
        训练感知机
        更新规则：w = w + lr * (y - y_hat) * x
        """
        for epoch in range(self.max_epochs):
            error_count = 0
            for xi, yi in zip(X, y):
                y_hat = self.predict(xi.reshape(1, -1))[0]
                error = yi - y_hat
                if error != 0:
                    self.weights += self.lr * error * xi
                    self.bias += self.lr * error
                    error_count += 1
            if verbose:
                print(f"  Epoch {epoch + 1:3d}: 权重={self.weights}, "
                      f"偏置={self.bias:.2f}, 错误数={error_count}")
            if error_count == 0:
                print(f"  --> 在第 {epoch + 1} 轮收敛！")
                return True
        print(f"  --> 达到最大轮数 {self.max_epochs}，未收敛")
        return False


# ============================================================
# 2. 准备逻辑门数据
# ============================================================
# 四种输入组合 (x1, x2)
X_gate = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)

# 逻辑门的标签
y_and = np.array([0, 0, 0, 1])
y_or  = np.array([0, 1, 1, 1])
y_xor = np.array([0, 1, 1, 0])


# ============================================================
# 3. 在 AND 门上训练
# ============================================================
print("=" * 60)
print("训练 AND 门")
print("=" * 60)
p_and = Perceptron(input_dim=2, lr=0.1, max_epochs=20)
p_and.train(X_gate, y_and)
print(f"AND 预测结果: {p_and.predict(X_gate)}")
print()


# ============================================================
# 4. 在 OR 门上训练
# ============================================================
print("=" * 60)
print("训练 OR 门")
print("=" * 60)
p_or = Perceptron(input_dim=2, lr=0.1, max_epochs=20)
p_or.train(X_gate, y_or)
print(f"OR 预测结果: {p_or.predict(X_gate)}")
print()


# ============================================================
# 5. 在 XOR 门上训练（会失败）
# ============================================================
print("=" * 60)
print("训练 XOR 门（预期失败——线性不可分）")
print("=" * 60)
p_xor = Perceptron(input_dim=2, lr=0.1, max_epochs=30)
p_xor.train(X_gate, y_xor)
print(f"XOR 预测结果: {p_xor.predict(X_gate)}  (期望: {y_xor})")
print("单层感知机无法解决 XOR 问题！\n")


# ============================================================
# 6. 可视化决策边界
# ============================================================
def plot_decision_boundary(perceptron, X, y, title, ax):
    """绘制感知机的决策边界"""
    # 生成网格
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = perceptron.predict(grid).reshape(xx.shape)

    # 绘制决策区域
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    # 绘制决策边界线
    ax.contour(xx, yy, Z, colors='k', linewidths=1.5, levels=[0.5])

    # 绘制数据点
    for cls, marker, color in [(0, 'o', 'red'), (1, 's', 'blue')]:
        mask = (y == cls)
        ax.scatter(X[mask, 0], X[mask, 1], c=color, marker=marker,
                   s=150, edgecolors='black', linewidth=1.5,
                   label=f'类别 {cls}', zorder=5)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


fig, axes = plt.subplots(1, 3, figsize=(18, 5))

plot_decision_boundary(p_and, X_gate, y_and, "AND 门（可分离 - 收敛）", axes[0])
plot_decision_boundary(p_or, X_gate, y_or, "OR 门（可分离 - 收敛）", axes[1])
plot_decision_boundary(p_xor, X_gate, y_xor, "XOR 门（不可分离 - 失败）", axes[2])

plt.suptitle("感知机决策边界：线性可分 vs 线性不可分", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("d1_perceptron_decision_boundaries.png", dpi=150, bbox_inches='tight')
plt.show()
print("图1保存完成: d1_perceptron_decision_boundaries.png\n")


# ============================================================
# 7. 用 MLP 手动解决 XOR 问题
# ============================================================
print("=" * 60)
print("用两层 MLP 手动解决 XOR 问题")
print("=" * 60)

# 网络结构: 输入层(2) -> 隐藏层(2) -> 输出层(1)
# 隐藏层实现 NAND 和 OR，输出层实现 AND
#   NAND: (1,1) -> 0, 其他 -> 1   => w=[-1,-1], b=1.5, 阈值0
#   OR:   (0,0) -> 0, 其他 -> 1   => w=[1,1],  b=-0.5, 阈值0
#   AND:  两输入都为1时输出1        => w=[1,1],  b=-1.5, 阈值0


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def mlp_xor_forward(X):
    """
    手动设置权重的两层 MLP 解决 XOR
    隐藏层: 两个神经元分别计算 NAND 和 OR
    输出层: 对隐藏层结果做 AND
    """
    # 隐藏层权重和偏置
    # 神经元1: NAND 门   w=[-1, -1], b=1.5
    # 神经元2: OR 门     w=[ 1,  1], b=-0.5
    W1 = np.array([[-1.0, 1.0],
                    [-1.0, 1.0]])
    b1 = np.array([1.5, -0.5])

    # 输出层权重和偏置: AND 门  w=[1, 1], b=-1.5
    W2 = np.array([1.0, 1.0])
    b2 = -1.5

    # 前向传播
    hidden = sigmoid(X @ W1 + b1)
    output = sigmoid(hidden @ W2 + b2)
    return hidden, output


# 运行 MLP
hidden_activations, predictions = mlp_xor_forward(X_gate)

print("\nXOR MLP 前向传播过程:")
print("-" * 60)
for i in range(4):
    print(f"输入: {X_gate[i].astype(int)} -> "
          f"隐藏层(NAND={hidden_activations[i,0]:.4f}, "
          f"OR={hidden_activations[i,1]:.4f}) -> "
          f"输出: {predictions[i]:.4f} -> "
          f"预测: {int(predictions[i] >= 0.5)} (真实: {y_xor[i]})")

print(f"\n最终预测: {(predictions >= 0.5).astype(int)}")
print(f"期望输出: {y_xor}")
print("MLP 成功解决了 XOR 问题！")


# ============================================================
# 8. 可视化 MLP 解决 XOR
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 左图：原始 XOR 数据
for cls, marker, color in [(0, 'o', 'red'), (1, 's', 'blue')]:
    mask = (y_xor == cls)
    axes[0].scatter(X_gate[mask, 0], X_gate[mask, 1], c=color, marker=marker,
                    s=200, edgecolors='black', linewidth=2, label=f'类别 {cls}', zorder=5)
axes[0].set_xlabel('$x_1$')
axes[0].set_ylabel('$x_2$')
axes[0].set_title('XOR 原始数据（线性不可分）')
axes[0].legend()
axes[0].set_xlim(-0.5, 1.5)
axes[0].set_ylim(-0.5, 1.5)

# 右图：MLP 隐藏层表示空间
# 在隐藏层的变换下，XOR 变成线性可分！
for cls, marker, color in [(0, 'o', 'red'), (1, 's', 'blue')]:
    mask = (y_xor == cls)
    axes[1].scatter(hidden_activations[mask, 0], hidden_activations[mask, 1],
                    c=color, marker=marker, s=200, edgecolors='black', linewidth=2,
                    label=f'类别 {cls}', zorder=5)

# 画决策边界
h_x = np.linspace(-0.1, 1.1, 100)
axes[1].plot(h_x, (1.5 - h_x) / 1.0, 'k--', linewidth=2, label='MLP 决策边界')
axes[1].set_xlabel('隐藏神经元 1 (NAND)')
axes[1].set_ylabel('隐藏神经元 2 (OR)')
axes[1].set_title('MLP 隐藏层空间（线性可分）')
axes[1].legend()

plt.suptitle("两层 MLP 解决 XOR 问题：隐藏层变换使数据线性可分",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("d1_perceptron_mlp_xor.png", dpi=150, bbox_inches='tight')
plt.show()
print("图2保存完成: d1_perceptron_mlp_xor.png")

print("\n" + "=" * 60)
print("总结:")
print("  - 单层感知机只能解决线性可分问题（AND, OR）")
print("  - XOR 问题是线性不可分的，单层感知机无法解决")
print("  - 两层 MLP 通过隐藏层变换，将非线性问题转化为线性可分")
print("  - 这就是深度学习的基本原理：逐层变换特征空间")
print("=" * 60)
