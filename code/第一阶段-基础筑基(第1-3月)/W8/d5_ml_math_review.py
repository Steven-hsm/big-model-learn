"""
W08 Day 5: ML数学回顾 + 深度学习预习
回顾所有ML算法的数学基础，实现感知机，展示激活函数
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Matplotlib 中文显示设置
# ============================================================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. ML 算法数学基础回顾
# ============================================================
print("=" * 60)
print("ML 数学基础回顾")
print("=" * 60)

math_review = """
1. 线性回归 = 最小二乘法 / 最大似然估计(MLE)
   - 损失函数: L(w) = sum((y_i - w^T x_i)^2)  (最小二乘)
   - MLE 视角: 假设 y = w^T x + epsilon, epsilon ~ N(0, sigma^2)
   - 解析解: w* = (X^T X)^(-1) X^T y

2. 逻辑回归 = MLE -> 交叉熵损失
   - sigmoid: p = 1 / (1 + exp(-w^T x))
   - 损失函数: L = -sum(y*log(p) + (1-y)*log(1-p))  (交叉熵)
   - 无解析解, 使用梯度下降迭代求解

3. SVM = 凸优化 + KKT 条件
   - 目标: min ||w||^2/2  s.t. y_i(w^T x_i + b) >= 1
   - 拉格朗日函数 + KKT 条件 -> 对偶问题
   - 核函数技巧: K(x,z) = phi(x)^T phi(z)

4. 决策树 = 信息论
   - 信息熵: H(D) = -sum(p_k * log2(p_k))
   - 信息增益: Gain(D, a) = H(D) - sum(|D_v|/|D| * H(D_v))
   - 基尼系数: Gini(D) = 1 - sum(p_k^2)

5. K-Means = EM 算法的特例
   - E步: 分配样本到最近聚类中心
   - M步: 更新聚类中心为组内均值
   - 目标: min sum(||x_i - mu_{c_i}||^2)

6. PCA = 特征分解 / SVD
   - 协方差矩阵: C = X^T X / (n-1)
   - 特征分解: C = V Lambda V^T
   - 降维: X' = X V_k (取前k个主成分)
"""

print(math_review)

# ============================================================
# 2. 从零实现感知机 (Perceptron)
# ============================================================
print("\n--- 2. 从零实现感知机 ---")


class Perceptron:
    """单层感知机实现"""

    def __init__(self, learning_rate=0.01, n_epochs=100):
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None

    def step_function(self, x):
        return np.where(x >= 0, 1, 0)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for epoch in range(self.n_epochs):
            errors = 0
            for i in range(n_samples):
                linear_output = np.dot(X[i], self.weights) + self.bias
                prediction = self.step_function(linear_output)

                # 更新规则
                error = y[i] - prediction
                if error != 0:
                    self.weights += self.lr * error * X[i]
                    self.bias += self.lr * error
                    errors += 1

            if errors == 0:
                print(f"  在第 {epoch+1} 轮收敛")
                break

        return self

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.step_function(linear_output)


# AND 门测试
print("\nAND 门测试:")
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

perceptron_and = Perceptron(learning_rate=0.1, n_epochs=100)
perceptron_and.fit(X_and, y_and)
print(f"  预测: {perceptron_and.predict(X_and)}")
print(f"  期望: {y_and}")
print(f"  权重: {perceptron_and.weights}, 偏置: {perceptron_and.bias}")

# OR 门测试
print("\nOR 门测试:")
y_or = np.array([0, 1, 1, 1])
perceptron_or = Perceptron(learning_rate=0.1, n_epochs=100)
perceptron_or.fit(X_and, y_or)
print(f"  预测: {perceptron_or.predict(X_and)}")
print(f"  期望: {y_or}")

# ============================================================
# 3. XOR 问题 - 单层感知机无法解决
# ============================================================
print("\n--- 3. XOR 问题 (单层无法解决) ---")

y_xor = np.array([0, 1, 1, 0])

perceptron_xor = Perceptron(learning_rate=0.1, n_epochs=1000)
perceptron_xor.fit(X_and, y_xor)
xor_pred = perceptron_xor.predict(X_and)

print(f"  XOR 预测: {xor_pred}")
print(f"  XOR 期望: {y_xor}")
print(f"  单层感知机无法解决 XOR! (线性不可分)")

# XOR 不可分可视化
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# AND
colors_and = ['red' if v == 0 else 'blue' for v in y_and]
axes[0].scatter(X_and[:, 0], X_and[:, 1], c=colors_and, s=200, edgecolors='black', zorder=5)
axes[0].set_title('AND 门 (线性可分)', fontsize=13)
axes[0].set_xlabel('x1')
axes[0].set_ylabel('x2')
# 画决策边界
w = perceptron_and.weights
b = perceptron_and.bias
x_line = np.linspace(-0.5, 1.5, 100)
y_line = -(w[0] * x_line + b) / w[1] if w[1] != 0 else np.zeros_like(x_line)
axes[0].plot(x_line, y_line, 'g--', linewidth=2, label='决策边界')
axes[0].legend()
axes[0].set_xlim(-0.5, 1.5)
axes[0].set_ylim(-0.5, 1.5)
axes[0].grid(True, alpha=0.3)

# OR
colors_or = ['red' if v == 0 else 'blue' for v in y_or]
axes[1].scatter(X_and[:, 0], X_and[:, 1], c=colors_or, s=200, edgecolors='black', zorder=5)
axes[1].set_title('OR 门 (线性可分)', fontsize=13)
axes[1].set_xlabel('x1')
axes[1].set_ylabel('x2')
w = perceptron_or.weights
b = perceptron_or.bias
y_line = -(w[0] * x_line + b) / w[1] if w[1] != 0 else np.zeros_like(x_line)
axes[1].plot(x_line, y_line, 'g--', linewidth=2, label='决策边界')
axes[1].legend()
axes[1].set_xlim(-0.5, 1.5)
axes[1].set_ylim(-0.5, 1.5)
axes[1].grid(True, alpha=0.3)

# XOR
colors_xor = ['red' if v == 0 else 'blue' for v in y_xor]
axes[2].scatter(X_and[:, 0], X_and[:, 1], c=colors_xor, s=200, edgecolors='black', zorder=5)
axes[2].set_title('XOR 门 (线性不可分!)', fontsize=13)
axes[2].set_xlabel('x1')
axes[2].set_ylabel('x2')
# 尝试画感知机的失败决策边界
w = perceptron_xor.weights
b = perceptron_xor.bias
if w[1] != 0:
    y_line = -(w[0] * x_line + b) / w[1]
    axes[2].plot(x_line, y_line, 'g--', linewidth=2, label='尝试的边界')
axes[2].legend()
axes[2].set_xlim(-0.5, 1.5)
axes[2].set_ylim(-0.5, 1.5)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('d5_xor_problem.png', dpi=150, bbox_inches='tight')
plt.close()
print("[图表已保存] d5_xor_problem.png")

# ============================================================
# 4. 多层感知机解决 XOR
# ============================================================
print("\n--- 4. 多层感知机解决 XOR ---")


class SimpleMLP:
    """简单的两层神经网络 (手动实现)"""

    def __init__(self, input_size, hidden_size, output_size, lr=0.5):
        self.lr = lr
        # 随机初始化权重
        np.random.seed(42)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y):
        m = X.shape[0]

        # 输出层梯度
        dz2 = self.a2 - y
        dW2 = self.a1.T @ dz2 / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # 隐藏层梯度
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.sigmoid_derivative(self.z1)
        dW1 = X.T @ dz1 / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # 更新权重
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, y, epochs=5000):
        losses = []
        for epoch in range(epochs):
            output = self.forward(X)
            loss = np.mean((y - output) ** 2)
            losses.append(loss)
            self.backward(X, y)

            if (epoch + 1) % 1000 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")
        return losses

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)


# 训练 MLP 解决 XOR
print("\n训练 2 层 MLP 解决 XOR:")
mlp = SimpleMLP(input_size=2, hidden_size=2, output_size=1, lr=2.0)
y_xor_2d = y_xor.reshape(-1, 1).astype(float)
losses = mlp.train(X_and.astype(float), y_xor_2d, epochs=5000)

xor_mlp_pred = mlp.predict(X_and.astype(float))
print(f"\n  MLP XOR 预测: {xor_mlp_pred.flatten()}")
print(f"  XOR 期望:     {y_xor}")
print(f"  MLP 成功解决 XOR 问题!")

# ============================================================
# 5. 激活函数对比
# ============================================================
print("\n--- 5. 激活函数对比 ---")

x = np.linspace(-6, 6, 200)

# 各激活函数
def step(x):
    return np.where(x >= 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

activations = {
    'Step (阶跃)': step,
    'Sigmoid (S型)': sigmoid,
    'Tanh (双曲正切)': tanh,
    'ReLU': relu,
    'Leaky ReLU': leaky_relu,
    'ELU': elu,
}

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for idx, (name, func) in enumerate(activations.items()):
    y = func(x)
    axes[idx].plot(x, y, linewidth=2.5, color='steelblue')
    axes[idx].set_title(name, fontsize=13)
    axes[idx].set_xlabel('x')
    axes[idx].set_ylabel('f(x)')
    axes[idx].grid(True, alpha=0.3)
    axes[idx].axhline(0, color='black', linewidth=0.5)
    axes[idx].axvline(0, color='black', linewidth=0.5)

    # 标注关键特征
    if 'ReLU' in name:
        axes[idx].annotate('稀疏激活', xy=(3, 3), fontsize=9, color='red')
    elif 'Sigmoid' in name:
        axes[idx].annotate('输出范围(0,1)', xy=(2, 0.8), fontsize=9, color='red')
    elif 'Tanh' in name:
        axes[idx].annotate('输出范围(-1,1)', xy=(2, 0.8), fontsize=9, color='red')

plt.suptitle('常用激活函数对比', fontsize=16, y=1.01)
plt.tight_layout()
plt.savefig('d5_activation_functions.png', dpi=150, bbox_inches='tight')
plt.close()
print("[图表已保存] d5_activation_functions.png")

# ============================================================
# 6. XOR 决策边界可视化 + 训练曲线
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 训练损失曲线
axes[0].plot(losses, linewidth=1, color='steelblue')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss (MSE)', fontsize=12)
axes[0].set_title('MLP 训练损失曲线 (XOR)', fontsize=13)
axes[0].grid(True, alpha=0.3)

# 决策边界
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100),
                      np.linspace(-0.5, 1.5, 100))
grid = np.c_[xx.ravel(), yy.ravel()].astype(float)
Z = mlp.forward(grid)
Z = Z.reshape(xx.shape)

axes[1].contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.6)
axes[1].contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
colors_xor = ['red' if v == 0 else 'blue' for v in y_xor]
axes[1].scatter(X_and[:, 0], X_and[:, 1], c=colors_xor,
                s=200, edgecolors='black', zorder=5)
axes[1].set_title('MLP XOR 决策边界', fontsize=13)
axes[1].set_xlabel('x1')
axes[1].set_ylabel('x2')

plt.tight_layout()
plt.savefig('d5_xor_decision_boundary.png', dpi=150, bbox_inches='tight')
plt.close()
print("[图表已保存] d5_xor_decision_boundary.png")

# ============================================================
# 7. 激活函数特性总结
# ============================================================
print("\n--- 7. 激活函数特性总结 ---")
print(f"""
{'函数':<15} {'范围':<15} {'优点':<25} {'缺点':<25}
{'='*80}
{'Step':<15} {'{0, 1}':<15} {'简单直观':<25} {'不可导, 无法梯度下降':<25}
{'Sigmoid':<15} {'(0, 1)':<15} {'输出概率, 平滑':<25} {'梯度消失, 非零中心':<25}
{'Tanh':<15} {'(-1, 1)':<15} {'零中心化':<25} {'梯度消失':<25}
{'ReLU':<15} {'[0, +inf)':<15} {'计算快, 稀疏激活':<25} {'Dead ReLU':<25}
{'LeakyReLU':<15} {'(-inf, +inf)':<15} {'缓解Dead ReLU':<25} {'需调alpha参数':<25}
{'ELU':<15} {'(-alpha, +inf)':<15} {'输出接近零均值':<25} {'计算指数较慢':<25}
""")

print("=" * 60)
print("ML数学回顾 + 深度学习预习 总结")
print("=" * 60)
print("""
1. 每个ML算法背后都有坚实的数学基础
2. 单层感知机只能解决线性可分问题
3. 多层网络通过非线性变换解决复杂问题(XOR)
4. 激活函数是神经网络的核心组件
5. ReLU 是目前最常用的激活函数
""")
