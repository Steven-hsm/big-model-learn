### Day 3：线性回归进阶 + 正则化
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 生成非线性数据
# ============================================================
np.random.seed(42)
n_samples = 80

X = np.sort(np.random.uniform(-3, 3, n_samples)).reshape(-1, 1)
y_true = 0.5 * X.squeeze() ** 2 - 1.5 * X.squeeze() + 1
y = y_true + np.random.randn(n_samples) * 1.5

print("=" * 60)
print("线性回归进阶：多项式回归 + 正则化")
print("=" * 60)

# ============================================================
# 2. 多项式回归 — 展示过拟合
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
degrees = [1, 3, 15]

for ax, deg in zip(axes, degrees):
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=deg, include_bias=False)),
        ('linear', LinearRegression())
    ])
    model.fit(X, y)
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    n_coefs = len(model.named_steps['linear'].coef_)

    ax.scatter(X, y, alpha=0.4, s=20, label='数据')
    x_plot = np.linspace(-3, 3, 200).reshape(-1, 1)
    ax.plot(x_plot, model.predict(x_plot), 'r-', linewidth=2, label=f'degree={deg}')
    ax.set_title(f'多项式 degree={deg}\nMSE={mse:.3f}, 参数数={n_coefs}')
    ax.set_ylim(-8, 12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    print(f"  degree={deg:>2d}: MSE={mse:.4f}, 参数数={n_coefs}")

plt.suptitle('多项式回归：从欠拟合到过拟合', fontsize=14)
plt.tight_layout()
plt.show()

# ============================================================
# 3. Ridge回归 (L2) — 不同alpha
# ============================================================
print("\n" + "=" * 60)
print("Ridge回归 (L2正则化)")
print("=" * 60)

alphas = [0.001, 0.01, 0.1, 1, 10, 100]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for ax, alpha in zip(axes, alphas):
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=15, include_bias=False)),
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=alpha))
    ])
    model.fit(X, y)
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)

    ax.scatter(X, y, alpha=0.4, s=20, label='数据')
    x_plot = np.linspace(-3, 3, 200).reshape(-1, 1)
    ax.plot(x_plot, model.predict(x_plot), 'r-', linewidth=2)
    ax.set_title(f'Ridge alpha={alpha}\nMSE={mse:.3f}')
    ax.set_ylim(-8, 12)
    ax.grid(True, alpha=0.3)

    coefs = model.named_steps['ridge'].coef_
    print(f"  alpha={alpha:>6.3f}: MSE={mse:.4f}, 系数范数={np.linalg.norm(coefs):.4f}")

plt.suptitle('Ridge回归：不同alpha的影响 (degree=15)', fontsize=14)
plt.tight_layout()
plt.show()

# ============================================================
# 4. Lasso回归 (L1) — 系数稀疏化
# ============================================================
print("\n" + "=" * 60)
print("Lasso回归 (L1正则化) — 系数稀疏化")
print("=" * 60)

poly = PolynomialFeatures(degree=10, include_bias=False)
X_poly = poly.fit_transform(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

lasso_alphas = np.logspace(-4, 1, 50)
coefs_lasso = []
n_nonzero = []

for alpha in lasso_alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_scaled, y)
    coefs_lasso.append(lasso.coef_)
    n_nonzero.append(np.sum(np.abs(lasso.coef_) > 1e-5))

coefs_lasso = np.array(coefs_lasso)

# 图：Lasso系数路径
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
for i in range(coefs_lasso.shape[1]):
    ax1.plot(lasso_alphas, coefs_lasso[:, i], linewidth=1.2)
ax1.set_xscale('log')
ax1.set_xlabel('Alpha (log scale)')
ax1.set_ylabel('系数值')
ax1.set_title('Lasso系数路径')
ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(lasso_alphas, n_nonzero, 'b-o', markersize=4)
ax2.set_xscale('log')
ax2.set_xlabel('Alpha (log scale)')
ax2.set_ylabel('非零系数个数')
ax2.set_title('Lasso：非零系数数量 vs Alpha')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# 5. Ridge vs Lasso vs ElasticNet 对比
# ============================================================
print("\n" + "=" * 60)
print("Ridge vs Lasso vs ElasticNet 对比")
print("=" * 60)

ridge_alphas = np.logspace(-3, 3, 100)
coefs_ridge = []
coefs_lasso2 = []
coefs_enet = []

for alpha in ridge_alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_scaled, y)
    coefs_ridge.append(ridge.coef_)

    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_scaled, y)
    coefs_lasso2.append(lasso.coef_)

    enet = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=10000)
    enet.fit(X_scaled, y)
    coefs_enet.append(enet.coef_)

coefs_ridge = np.array(coefs_ridge)
coefs_lasso2 = np.array(coefs_lasso2)
coefs_enet = np.array(coefs_enet)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, coefs, name in zip(axes,
                            [coefs_ridge, coefs_lasso2, coefs_enet],
                            ['Ridge (L2)', 'Lasso (L1)', 'ElasticNet (L1+L2)']):
    for i in range(coefs.shape[1]):
        ax.plot(ridge_alphas, coefs[:, i], linewidth=1.2)
    ax.set_xscale('log')
    ax.set_xlabel('Alpha (log scale)')
    ax.set_ylabel('系数值')
    ax.set_title(f'{name} 系数路径')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 对比最终模型效果
print("\n模型对比 (alpha=1.0, degree=10):")
for name, Model in [('Ridge', Ridge), ('Lasso', Lasso), ('ElasticNet', ElasticNet)]:
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=10, include_bias=False)),
        ('scaler', StandardScaler()),
        ('model', Model(alpha=1.0, max_iter=10000))
    ])
    model.fit(X, y)
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    coefs = model.named_steps['model'].coef_
    nonzero = np.sum(np.abs(coefs) > 1e-5)
    print(f"  {name:>12s}: MSE={mse:.4f}, 非零系数={nonzero}/{len(coefs)}")
