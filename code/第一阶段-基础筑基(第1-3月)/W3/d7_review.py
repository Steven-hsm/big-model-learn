# W03 Day 7 - 综合复习：线性代数与AI映射关系
# 反向传播手动计算 + 思维导图总结

import numpy as np
import matplotlib.pyplot as plt

# ========================
# 1. 手动计算2层网络的梯度（验证反向传播）
# ========================

# 前向传播
X = np.array([[0.5, 0.3]])       # 输入 (1, 2)
W1 = np.array([[0.1, 0.2],       # 第一层权重 (2, 2)
               [0.3, 0.4]])
b1 = np.array([[0.1, 0.1]])      # 第一层偏置
W2 = np.array([[0.5], [0.6]])    # 第二层权重 (2, 1)
b2 = np.array([[0.2]])           # 第二层偏置
y_true = np.array([[1.0]])       # 真实标签


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Layer 1
z1 = X @ W1 + b1
a1 = sigmoid(z1)

# Layer 2
z2 = a1 @ W2 + b2
a2 = sigmoid(z2)

# 损失 (MSE)
loss = 0.5 * (y_true - a2) ** 2
print(f"预测值: {a2[0, 0]:.6f}")
print(f"损失: {loss[0, 0]:.6f}")

# 反向传播（手动计算梯度）
dL_da2 = -(y_true - a2)
da2_dz2 = a2 * (1 - a2)
dL_dz2 = dL_da2 * da2_dz2

dL_dW2 = a1.T @ dL_dz2
dL_db2 = dL_dz2

dL_da1 = dL_dz2 @ W2.T
da1_dz1 = a1 * (1 - a1)
dL_dz1 = dL_da1 * da1_dz1

dL_dW1 = X.T @ dL_dz1
dL_db1 = dL_dz1

print(f"\n梯度 dL/dW1:\n{dL_dW1}")
print(f"梯度 dL/dW2:\n{dL_dW2}")

# 梯度下降更新
lr = 0.1
W1_new = W1 - lr * dL_dW1
b1_new = b1 - lr * dL_db1
W2_new = W2 - lr * dL_dW2
b2_new = b2 - lr * dL_db2

# 验证损失下降
z1_new = X @ W1_new + b1_new
a1_new = sigmoid(z1_new)
z2_new = a1_new @ W2_new + b2_new
a2_new = sigmoid(z2_new)
loss_new = 0.5 * (y_true - a2_new) ** 2
print(f"\n更新后预测值: {a2_new[0, 0]:.6f}")
print(f"更新后损失: {loss_new[0, 0]:.6f} (应该减小)")

# ========================
# 2. 多步训练验证
# ========================

W1_t = W1.copy()
b1_t = b1.copy()
W2_t = W2.copy()
b2_t = b2.copy()

losses = []
for epoch in range(100):
    # Forward
    z1_t = X @ W1_t + b1_t
    a1_t = sigmoid(z1_t)
    z2_t = a1_t @ W2_t + b2_t
    a2_t = sigmoid(z2_t)
    l = 0.5 * (y_true - a2_t) ** 2
    losses.append(l[0, 0])

    # Backward
    dL_da2 = -(y_true - a2_t)
    dL_dz2 = dL_da2 * a2_t * (1 - a2_t)
    dL_dW2 = a1_t.T @ dL_dz2
    dL_db2 = dL_dz2
    dL_da1 = dL_dz2 @ W2_t.T
    dL_dz1 = dL_da1 * a1_t * (1 - a1_t)
    dL_dW1 = X.T @ dL_dz1
    dL_db1 = dL_dz1

    # Update
    W1_t -= lr * dL_dW1
    b1_t -= lr * dL_db1
    W2_t -= lr * dL_dW2
    b2_t -= lr * dL_db2

plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('2-Layer Network Training Loss (Manual Backprop)')
plt.grid(True)
plt.savefig('backprop_training.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n训练100个epoch后:")
print(f"  预测值: {a2_t[0, 0]:.6f} (目标: {y_true[0, 0]})")
print(f"  损失: {losses[-1]:.6f}")

# ========================
# 3. 线性代数在AI中的映射总结
# ========================

print("\n" + "=" * 60)
print("线性代数与AI映射关系总结")
print("=" * 60)

mapping = {
    "向量": "数据样本（特征向量）、权重向量",
    "矩阵乘法": "神经网络前向传播 y = Wx + b",
    "矩阵分解": "推荐系统、模型压缩",
    "特征值/特征向量": "PCA降维、模型稳定性分析",
    "SVD": "模型压缩、低秩近似、推荐系统",
    "向量范数": "正则化（L1/L2）、梯度裁剪",
    "余弦相似度": "文本相似度、推荐系统",
    "正交矩阵": "特征独立性、数值稳定性",
    "点积": "注意力机制（Attention）、相似度计算",
}

for concept, application in mapping.items():
    print(f"  {concept:12s} -> {application}")

print("\n核心洞察：神经网络的本质 = 矩阵乘法 + 非线性激活函数")
print("权重矩阵W就是线性变换，将输入空间映射到输出空间")
