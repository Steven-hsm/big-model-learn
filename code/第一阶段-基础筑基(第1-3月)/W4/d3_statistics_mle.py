"""
W04-D3 统计基础与MLE（最大似然估计）
=====================================
内容：
1. 从正态分布中采样，计算样本均值和方差
2. 协方差和相关系数
3. MLE：正态分布参数估计
4. 手动负对数似然函数 + scipy.optimize.minimize
5. 解析解 vs 数值优化对比
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 样本均值与方差
# ============================================================
print("=" * 60)
print("1. 样本均值与方差")
print("=" * 60)

np.random.seed(42)

# 从正态分布 N(5, 3²) 中采样
true_mu = 5.0
true_sigma = 3.0
sample_sizes = [50, 500, 5000, 50000]

print(f"\n  真实参数: μ={true_mu}, σ={true_sigma}")
print(f"\n  {'样本量':<10s} {'样本均值':<12s} {'样本方差':<12s} {'样本标准差':<12s}")
print("  " + "-" * 46)

for n in sample_sizes:
    sample = np.random.normal(true_mu, true_sigma, size=n)
    sample_mean = np.mean(sample)
    sample_var = np.var(sample, ddof=1)  # 无偏估计（除以 n-1）
    sample_std = np.std(sample, ddof=1)
    print(f"  {n:<10d} {sample_mean:<12.4f} {sample_var:<12.4f} {sample_std:<12.4f}")

print(f"\n  注意：样本量越大，统计量越接近真实参数（大数定律）")


# ============================================================
# 2. 协方差和相关系数
# ============================================================
print("\n" + "=" * 60)
print("2. 协方差和相关系数")
print("=" * 60)

n = 1000

# 情况1：强正相关（X与Y=2X+noise）
X1 = np.random.normal(0, 1, n)
noise1 = np.random.normal(0, 0.5, n)
Y1 = 2 * X1 + noise1 + 1

# 情况2：无相关（独立变量）
X2 = np.random.normal(0, 1, n)
Y2 = np.random.normal(0, 1, n)

# 情况3：强负相关
X3 = np.random.normal(0, 1, n)
noise3 = np.random.normal(0, 0.5, n)
Y3 = -3 * X3 + noise3 + 2

cases = [
    ("强正相关 (Y ≈ 2X + 1)", X1, Y1),
    ("无相关 (独立变量)", X2, Y2),
    ("强负相关 (Y ≈ -3X + 2)", X3, Y3),
]

print()
for name, X, Y in cases:
    cov_matrix = np.cov(X, Y)
    cov_xy = cov_matrix[0, 1]
    corr_xy = np.corrcoef(X, Y)[0, 1]
    print(f"  {name}:")
    print(f"    协方差 cov(X,Y) = {cov_xy:.4f}")
    print(f"    相关系数 r(X,Y) = {corr_xy:.4f}")
    print()

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for ax, (name, X, Y) in zip(axes, cases):
    corr = np.corrcoef(X, Y)[0, 1]
    ax.scatter(X, Y, alpha=0.3, s=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'{name}\nr = {corr:.4f}')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('d3_correlation.png', dpi=150)
plt.close()
print("  [图] 相关系数散点图已保存为 d3_correlation.png")


# ============================================================
# 3. MLE：正态分布参数估计
# ============================================================
print("\n" + "=" * 60)
print("3. MLE：正态分布参数估计")
print("=" * 60)

# 生成样本
np.random.seed(123)
true_mu_mle = 3.0
true_sigma_mle = 1.5
n_samples = 200
data = np.random.normal(true_mu_mle, true_sigma_mle, size=n_samples)

# 解析解（MLE公式）
mu_mle = np.mean(data)       # μ_MLE = (1/n) Σ x_i
sigma_mle = np.std(data, ddof=0)  # σ_MLE = sqrt((1/n) Σ (x_i - μ)²)

print(f"\n  真实参数: μ={true_mu_mle}, σ={true_sigma_mle}")
print(f"  样本量: {n_samples}")
print(f"\n  解析 MLE 估计:")
print(f"    μ_MLE  = (1/n) Σ x_i      = {mu_mle:.6f}")
print(f"    σ_MLE  = sqrt((1/n) Σ(x-μ)²) = {sigma_mle:.6f}")


# ============================================================
# 4. 数值优化：手动负对数似然 + scipy.optimize
# ============================================================
print("\n" + "=" * 60)
print("4. 数值优化：负对数似然最小化")
print("=" * 60)


def neg_log_likelihood(params, data):
    """
    正态分布的负对数似然函数

    NLL(μ, σ) = (n/2)ln(2π) + (n/2)ln(σ²) + (1/2σ²)Σ(x_i - μ)²

    Parameters:
        params: [mu, sigma] 待估参数
        data: 观测数据
    """
    mu, sigma = params
    n = len(data)
    if sigma <= 0:
        return 1e10  # sigma必须为正

    # 负对数似然
    nll = (n / 2) * np.log(2 * np.pi) + \
          (n / 2) * np.log(sigma ** 2) + \
          (1 / (2 * sigma ** 2)) * np.sum((data - mu) ** 2)
    return nll


# 初始猜测
initial_params = [0.0, 1.0]

# 使用 scipy.optimize.minimize 进行优化
result = minimize(
    neg_log_likelihood,
    initial_params,
    args=(data,),
    method='L-BFGS-B',
    bounds=[(None, None), (1e-6, None)]  # sigma > 0
)

mu_numerical = result.x[0]
sigma_numerical = result.x[1]

print(f"\n  数值优化结果:")
print(f"    优化器: L-BFGS-B")
print(f"    收敛: {'是' if result.success else '否'}")
print(f"    μ_num  = {mu_numerical:.6f}")
print(f"    σ_num  = {sigma_numerical:.6f}")
print(f"    NLL值  = {result.fun:.4f}")


# ============================================================
# 5. 解析解 vs 数值优化 对比
# ============================================================
print("\n" + "=" * 60)
print("5. 解析解 vs 数值优化 对比")
print("=" * 60)

print(f"\n  {'方法':<20s} {'μ估计':<15s} {'σ估计':<15s} {'误差μ':<15s} {'误差σ':<15s}")
print("  " + "-" * 80)

methods = [
    ("真实值", true_mu_mle, true_sigma_mle),
    ("解析MLE", mu_mle, sigma_mle),
    ("数值优化", mu_numerical, sigma_numerical),
]

for name, mu_val, sigma_val in methods:
    err_mu = abs(mu_val - true_mu_mle)
    err_sigma = abs(sigma_val - true_sigma_mle)
    print(f"  {name:<20s} {mu_val:<15.6f} {sigma_val:<15.6f} {err_mu:<15.6f} {err_sigma:<15.6f}")

# 可视化：似然函数在参数空间中的等高线
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# 左图：似然函数等高线
mu_range = np.linspace(mu_mle - 1, mu_mle + 1, 100)
sigma_range = np.linspace(max(0.5, sigma_mle - 1), sigma_mle + 1, 100)
MU, SIGMA = np.meshgrid(mu_range, sigma_range)
NLL = np.zeros_like(MU)

for i in range(MU.shape[0]):
    for j in range(MU.shape[1]):
        NLL[i, j] = neg_log_likelihood([MU[i, j], SIGMA[i, j]], data)

# 减去最小值以便展示
NLL_relative = NLL - NLL.min()
cs = axes[0].contourf(MU, SIGMA, NLL_relative, levels=30, cmap='viridis', alpha=0.8)
axes[0].contour(MU, SIGMA, NLL_relative, levels=15, colors='white', linewidths=0.5, alpha=0.5)
plt.colorbar(cs, ax=axes[0], label='相对NLL')
axes[0].plot(true_mu_mle, true_sigma_mle, 'r*', markersize=15, label='真实值', zorder=5)
axes[0].plot(mu_mle, sigma_mle, 'w^', markersize=10, label='解析MLE', zorder=5)
axes[0].plot(mu_numerical, sigma_numerical, 'cs', markersize=10, label='数值优化', zorder=5)
axes[0].set_xlabel('μ')
axes[0].set_ylabel('σ')
axes[0].set_title('负对数似然函数等高线')
axes[0].legend()

# 右图：拟合效果
axes[1].hist(data, bins=30, density=True, alpha=0.6, color='steelblue',
             edgecolor='white', label='样本数据')
x_fit = np.linspace(data.min() - 1, data.max() + 1, 200)
# 真实分布
axes[1].plot(x_fit, stats.norm.pdf(x_fit, true_mu_mle, true_sigma_mle),
             'r-', linewidth=2, label=f'真实 N({true_mu_mle}, {true_sigma_mle}²)')
# MLE拟合
axes[1].plot(x_fit, stats.norm.pdf(x_fit, mu_mle, sigma_mle),
             'g--', linewidth=2, label=f'MLE N({mu_mle:.2f}, {sigma_mle:.2f}²)')
axes[1].set_xlabel('x')
axes[1].set_ylabel('概率密度')
axes[1].set_title('MLE 拟合效果')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('d3_mle.png', dpi=150)
plt.close()
print("\n  [图] MLE估计图已保存为 d3_mle.png")

print("\n" + "=" * 60)
print("D3 完成！")
print("=" * 60)
