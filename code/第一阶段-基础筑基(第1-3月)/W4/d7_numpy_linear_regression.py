"""
W04-D7 综合实战：NumPy线性回归
================================
内容：
1. 生成合成数据 y = 3.5x + 1.2 + noise
2. 实现 predict(), mse_loss(), compute_gradients(), train()
3. 梯度下降训练 (200 epochs, lr=0.05)
4. 损失曲线 + 回归线 (2面板图)
5. 正规方程解: w = (X^T X)^{-1} X^T y
6. 多变量线性回归 (3特征) + 正规方程
7. California Housing 数据集实战
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 生成合成数据
# ============================================================
print("=" * 60)
print("1. 生成合成数据：y = 3.5x + 1.2 + noise")
print("=" * 60)

np.random.seed(42)

n_samples = 200
true_w = 3.5
true_b = 1.2
noise_std = 0.8

# 生成数据
X_raw = np.random.uniform(0, 5, size=n_samples)
noise = np.random.normal(0, noise_std, size=n_samples)
y_raw = true_w * X_raw + true_b + noise

print(f"  样本量: {n_samples}")
print(f"  真实参数: w={true_w}, b={true_b}")
print(f"  噪声标准差: {noise_std}")
print(f"  X 范围: [{X_raw.min():.2f}, {X_raw.max():.2f}]")
print(f"  y 范围: [{y_raw.min():.2f}, {y_raw.max():.2f}]")


# ============================================================
# 2. 实现线性回归函数
# ============================================================
print("\n" + "=" * 60)
print("2. 实现线性回归函数")
print("=" * 60)


def predict(X, w, b):
    """
    线性回归预测: y = wX + b

    Parameters:
        X: 输入特征 (n_features, m) 或 (m,)
        w: 权重
        b: 偏置

    Returns:
        预测值
    """
    return w * X + b


def mse_loss(y_pred, y_true):
    """
    均方误差损失: L = (1/m) * Σ(y_pred - y_true)²

    Parameters:
        y_pred: 预测值
        y_true: 真实值

    Returns:
        MSE损失值
    """
    m = len(y_true)
    return np.mean((y_pred - y_true) ** 2)


def compute_gradients(X, y, w, b):
    """
    计算MSE损失对w和b的梯度

    ∂L/∂w = (2/m) * Σ(y_pred - y_true) * x
    ∂L/∂b = (2/m) * Σ(y_pred - y_true)

    Parameters:
        X: 输入特征
        y: 真实标签
        w: 当前权重
        b: 当前偏置

    Returns:
        (dw, db) 梯度元组
    """
    m = len(y)
    y_pred = predict(X, w, b)
    error = y_pred - y

    dw = (2 / m) * np.dot(error, X)
    db = (2 / m) * np.sum(error)

    return dw, db


def train(X, y, lr=0.05, n_epochs=200, verbose=True):
    """
    使用梯度下降训练线性回归

    Parameters:
        X: 输入特征
        y: 真实标签
        lr: 学习率
        n_epochs: 训练轮数
        verbose: 是否打印训练信息

    Returns:
        w: 训练后的权重
        b: 训练后的偏置
        loss_history: 损失历史
    """
    # 初始化参数
    w = 0.0
    b = 0.0
    loss_history = []

    for epoch in range(n_epochs):
        # 计算梯度
        dw, db = compute_gradients(X, y, w, b)

        # 梯度下降更新
        w -= lr * dw
        b -= lr * db

        # 记录损失
        loss = mse_loss(predict(X, w, b), y)
        loss_history.append(loss)

        if verbose and (epoch % 50 == 0 or epoch == n_epochs - 1):
            print(f"    Epoch {epoch:>4d}: loss={loss:.6f}, "
                  f"w={w:.6f}, b={b:.6f}")

    return w, b, loss_history


print("  函数定义完成: predict(), mse_loss(), compute_gradients(), train()")


# ============================================================
# 3. 梯度下降训练
# ============================================================
print("\n" + "=" * 60)
print("3. 梯度下降训练 (200 epochs, lr=0.05)")
print("=" * 60)

w_trained, b_trained, loss_history = train(X_raw, y_raw, lr=0.05, n_epochs=200)

print(f"\n  训练结果:")
print(f"    训练参数: w={w_trained:.6f}, b={b_trained:.6f}")
print(f"    真实参数: w={true_w}, b={true_b}")
print(f"    误差: Δw={abs(w_trained - true_w):.6f}, Δb={abs(b_trained - true_b):.6f}")
print(f"    最终损失: {loss_history[-1]:.6f}")


# ============================================================
# 4. 损失曲线 + 回归线 (2面板图)
# ============================================================
print("\n" + "=" * 60)
print("4. 可视化：损失曲线 + 回归线")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：损失曲线
axes[0].plot(loss_history, 'b-', linewidth=1.5)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')
axes[0].set_title('训练损失曲线')
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=noise_std**2, color='r', linestyle='--', alpha=0.5,
                label=f'理论最低损失 ≈ σ² = {noise_std**2:.2f}')
axes[0].legend()

# 右图：回归线
axes[1].scatter(X_raw, y_raw, alpha=0.4, s=15, label='数据点')
x_line = np.linspace(0, 5, 100)
y_pred_line = predict(x_line, w_trained, b_trained)
y_true_line = true_w * x_line + true_b
axes[1].plot(x_line, y_pred_line, 'r-', linewidth=2,
             label=f'拟合: y={w_trained:.2f}x+{b_trained:.2f}')
axes[1].plot(x_line, y_true_line, 'g--', linewidth=2, alpha=0.7,
             label=f'真实: y={true_w}x+{true_b}')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_title('线性回归拟合结果')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('d7_linear_regression.png', dpi=150)
plt.close()
print("  [图] 线性回归图已保存为 d7_linear_regression.png")


# ============================================================
# 5. 正规方程解
# ============================================================
print("\n" + "=" * 60)
print("5. 正规方程解: w = (X^T X)^{-1} X^T y")
print("=" * 60)

# 构建增广矩阵 X_aug = [X, 1]（加入偏置列）
X_aug = np.column_stack([X_raw, np.ones(n_samples)])  # (m, 2)

# 正规方程: θ = (X^T X)^{-1} X^T y
theta_normal = np.linalg.inv(X_aug.T @ X_aug) @ X_aug.T @ y_raw

w_normal = theta_normal[0]
b_normal = theta_normal[1]

print(f"\n  正规方程结果:")
print(f"    w = {w_normal:.6f}")
print(f"    b = {b_normal:.6f}")
print(f"    MSE = {mse_loss(predict(X_raw, w_normal, b_normal), y_raw):.6f}")

print(f"\n  对比:")
print(f"  {'方法':<20s} {'w':<12s} {'b':<12s} {'MSE':<12s}")
print("  " + "-" * 56)
print(f"  {'真实值':<20s} {true_w:<12.4f} {true_b:<12.4f} {'-':<12s}")
print(f"  {'梯度下降':<20s} {w_trained:<12.6f} {b_trained:<12.6f} "
      f"{mse_loss(predict(X_raw, w_trained, b_trained), y_raw):<12.6f}")
print(f"  {'正规方程':<20s} {w_normal:<12.6f} {b_normal:<12.6f} "
      f"{mse_loss(predict(X_raw, w_normal, b_normal), y_raw):<12.6f}")


# ============================================================
# 6. 多变量线性回归（3个特征）
# ============================================================
print("\n" + "=" * 60)
print("6. 多变量线性回归（3个特征）")
print("=" * 60)

np.random.seed(123)

n_multi = 300
n_features = 3

# 真实参数
true_w_multi = np.array([2.0, -1.5, 0.8])
true_b_multi = 0.5

# 生成数据
X_multi = np.random.randn(n_multi, n_features)
noise_multi = np.random.normal(0, 0.5, n_multi)
y_multi = X_multi @ true_w_multi + true_b_multi + noise_multi

print(f"  样本量: {n_multi}, 特征数: {n_features}")
print(f"  真实权重: {true_w_multi}")
print(f"  真实偏置: {true_b_multi}")

# 使用正规方程
X_multi_aug = np.column_stack([X_multi, np.ones(n_multi)])
theta_multi = np.linalg.inv(X_multi_aug.T @ X_multi_aug) @ X_multi_aug.T @ y_multi

w_multi = theta_multi[:n_features]
b_multi = theta_multi[n_features]

print(f"\n  正规方程估计:")
print(f"    权重: {w_multi}")
print(f"    偏置: {b_multi:.6f}")
print(f"    权重误差: {np.abs(w_multi - true_w_multi)}")

y_pred_multi = X_multi @ w_multi + b_multi
mse_multi = np.mean((y_pred_multi - y_multi) ** 2)
print(f"    MSE: {mse_multi:.6f}")


# ============================================================
# 7. California Housing 数据集实战
# ============================================================
print("\n" + "=" * 60)
print("7. California Housing 数据集实战")
print("=" * 60)

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 加载数据集
print("\n  加载 California Housing 数据集...")
housing = fetch_california_housing()
X_house = housing.data    # (20640, 8)
y_house = housing.target  # (20640,)

print(f"  数据形状: X={X_house.shape}, y={y_house.shape}")
print(f"  特征名称: {housing.feature_names}")
print(f"  目标变量: 房价中位数 (单位: 十万美元)")
print(f"  目标范围: [{y_house.min():.2f}, {y_house.max():.2f}]")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_house, y_house, test_size=0.2, random_state=42
)
print(f"\n  训练集大小: {X_train.shape[0]}")
print(f"  测试集大小: {X_test.shape[0]}")

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"  标准化后训练集均值范围: [{X_train_scaled.mean(axis=0).min():.6f}, "
      f"{X_train_scaled.mean(axis=0).max():.6f}]")
print(f"  标准化后训练集标准差范围: [{X_train_scaled.std(axis=0).min():.6f}, "
      f"{X_train_scaled.std(axis=0).max():.6f}]")


# 正规方程求解
def normal_equation(X, y):
    """
    使用正规方程求解线性回归

    θ = (X^T X)^{-1} X^T y
    """
    X_aug = np.column_stack([X, np.ones(X.shape[0])])
    theta = np.linalg.inv(X_aug.T @ X_aug) @ X_aug.T @ y
    return theta


print("\n  使用正规方程训练...")
theta_house = normal_equation(X_train_scaled, y_train)

w_house = theta_house[:-1]
b_house = theta_house[-1]

print(f"\n  模型参数:")
print(f"  {'特征':<25s} {'权重':<12s}")
print("  " + "-" * 37)
for name, weight in zip(housing.feature_names, w_house):
    print(f"  {name:<25s} {weight:<12.6f}")
print(f"  {'偏置(bias)':<25s} {b_house:<12.6f}")


# 评估函数
def evaluate(X, y, w, b):
    """评估模型性能"""
    y_pred = X @ w + b
    mse = np.mean((y_pred - y) ** 2)
    rmse = np.sqrt(mse)

    # R² = 1 - SS_res / SS_tot
    ss_res = np.sum((y_pred - y) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot

    return mse, rmse, r2


# 训练集评估
train_mse, train_rmse, train_r2 = evaluate(X_train_scaled, y_train, w_house, b_house)
# 测试集评估
test_mse, test_rmse, test_r2 = evaluate(X_test_scaled, y_test, w_house, b_house)

print(f"\n  模型评估:")
print(f"  {'指标':<10s} {'训练集':<15s} {'测试集':<15s}")
print("  " + "-" * 40)
print(f"  {'MSE':<10s} {train_mse:<15.6f} {test_mse:<15.6f}")
print(f"  {'RMSE':<10s} {train_rmse:<15.6f} {test_rmse:<15.6f}")
print(f"  {'R²':<10s} {train_r2:<15.6f} {test_r2:<15.6f}")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：特征权重条形图
colors = ['#3498db' if w > 0 else '#e74c3c' for w in w_house]
bars = axes[0].barh(housing.feature_names, w_house, color=colors, alpha=0.8)
axes[0].set_xlabel('权重')
axes[0].set_title('各特征权重（标准化后）')
axes[0].axvline(x=0, color='black', linewidth=0.5)
axes[0].grid(True, alpha=0.3, axis='x')

# 右图：预测值 vs 真实值散点图
y_test_pred = X_test_scaled @ w_house + b_house
axes[1].scatter(y_test, y_test_pred, alpha=0.2, s=10, color='steelblue')
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
             label='完美预测线')
axes[1].set_xlabel('真实房价')
axes[1].set_ylabel('预测房价')
axes[1].set_title(f'预测 vs 真实 (R²={test_r2:.4f})')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_aspect('equal')

plt.tight_layout()
plt.savefig('d7_california_housing.png', dpi=150)
plt.close()
print("\n  [图] California Housing 结果图已保存为 d7_california_housing.png")

print("\n" + "=" * 60)
print("D7 完成！W04 概率统计与微积分 全部完成！")
print("=" * 60)
