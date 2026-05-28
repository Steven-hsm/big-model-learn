### Day 2：线性回归原理 + NumPy实现
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as SklearnLR
from sklearn.metrics import mean_squared_error, r2_score

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 从零实现线性回归（梯度下降）
# ============================================================

class LinearRegression:
    """使用梯度下降实现的线性回归"""

    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.losses = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for i in range(self.n_iters):
            # 前向传播：y_pred = Xw + b
            y_pred = np.dot(X, self.weights) + self.bias

            # 计算损失 (MSE)
            loss = np.mean((y_pred - y) ** 2)
            self.losses.append(loss)

            # 计算梯度
            dw = (2 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (2 / n_samples) * np.sum(y_pred - y)

            # 参数更新
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


# ============================================================
# 2. 生成合成数据 y = 3x + 2 + noise
# ============================================================
np.random.seed(42)
n_samples = 200

X = np.random.uniform(-3, 3, size=(n_samples, 1))
y = 3 * X.squeeze() + 2 + np.random.randn(n_samples) * 1.5  # 加入噪声

print("=" * 50)
print("线性回归 - 从零实现 vs sklearn")
print("=" * 50)
print(f"真实参数: weight=3, bias=2")
print(f"数据量: {n_samples} 样本")
print(f"特征范围: [{X.min():.2f}, {X.max():.2f}]")

# ============================================================
# 3. 训练自实现模型
# ============================================================
model_scratch = LinearRegression(lr=0.01, n_iters=1000)
model_scratch.fit(X, y)
y_pred_scratch = model_scratch.predict(X)

print(f"\n【自实现模型】")
print(f"  weight = {model_scratch.weights[0]:.4f}  (真实值=3)")
print(f"  bias   = {model_scratch.bias:.4f}  (真实值=2)")
print(f"  MSE    = {mean_squared_error(y, y_pred_scratch):.4f}")
print(f"  R^2    = {r2_score(y, y_pred_scratch):.4f}")

# ============================================================
# 4. 训练sklearn模型
# ============================================================
model_sklearn = SklearnLR()
model_sklearn.fit(X, y)
y_pred_sklearn = model_sklearn.predict(X)

print(f"\n【sklearn模型】")
print(f"  weight = {model_sklearn.coef_[0]:.4f}  (真实值=3)")
print(f"  bias   = {model_sklearn.intercept_:.4f}  (真实值=2)")
print(f"  MSE    = {mean_squared_error(y, y_pred_sklearn):.4f}")
print(f"  R^2    = {r2_score(y, y_pred_sklearn):.4f}")

# ============================================================
# 5. 可视化
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 图1：数据点 + 回归线
ax1 = axes[0]
ax1.scatter(X, y, alpha=0.4, s=20, label='数据点')
x_line = np.linspace(-3, 3, 100).reshape(-1, 1)
ax1.plot(x_line, model_scratch.predict(x_line), 'r-', linewidth=2,
         label=f'自实现: y={model_scratch.weights[0]:.2f}x+{model_scratch.bias:.2f}')
ax1.plot(x_line, model_sklearn.predict(x_line), 'g--', linewidth=2,
         label=f'sklearn: y={model_sklearn.coef_[0]:.2f}x+{model_sklearn.intercept_:.2f}')
ax1.plot(x_line, 3 * x_line + 2, 'k:', linewidth=1.5, label='真实: y=3x+2')
ax1.set_xlabel('X')
ax1.set_ylabel('y')
ax1.set_title('线性回归拟合结果')
ax1.legend()

# 图2：损失曲线
ax2 = axes[1]
ax2.plot(model_scratch.losses, 'b-', linewidth=1.5)
ax2.set_xlabel('迭代次数')
ax2.set_ylabel('MSE Loss')
ax2.set_title('梯度下降损失曲线')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
