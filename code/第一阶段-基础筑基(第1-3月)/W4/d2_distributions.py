"""
W04-D2 常见概率分布
====================
内容：
1. 伯努利分布与二项分布（PMF柱状图）
2. 泊松分布
3. 正态分布：PDF和CDF图，68-95-99.7法则
4. 不同参数正态分布对比
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 伯努利分布与二项分布
# ============================================================
print("=" * 60)
print("1. 伯努利分布与二项分布")
print("=" * 60)

# 伯努利分布：抛硬币
p = 0.5  # 正面概率
bernoulli_dist = stats.bernoulli(p)
print(f"  伯努利分布 (p={p})")
print(f"    P(X=0) = {bernoulli_dist.pmf(0):.4f}")
print(f"    P(X=1) = {bernoulli_dist.pmf(1):.4f}")
print(f"    均值 = {bernoulli_dist.mean():.4f}")
print(f"    方差 = {bernoulli_dist.var():.4f}")

# 二项分布：n次独立伯努利试验
n_trials = 10
p_success = 0.3
binom_dist = stats.binom(n_trials, p_success)

print(f"\n  二项分布 (n={n_trials}, p={p_success})")
print(f"    均值 = np = {binom_dist.mean():.2f}")
print(f"    方差 = np(1-p) = {binom_dist.var():.2f}")
print(f"    标准差 = {binom_dist.std():.4f}")

# 模拟验证
np.random.seed(42)
simulations = binom_dist.rvs(size=100000)
print(f"    模拟均值 = {np.mean(simulations):.4f}")
print(f"    模拟方差 = {np.var(simulations):.4f}")

# 绘制二项分布PMF柱状图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 不同 p 值的二项分布 PMF
x_binom = np.arange(0, n_trials + 1)
for p_val in [0.2, 0.5, 0.8]:
    pmf = stats.binom.pmf(x_binom, n_trials, p_val)
    axes[0].bar(x_binom + (p_val - 0.5) * 0.25, pmf, width=0.25,
                label=f'p={p_val}', alpha=0.8)
axes[0].set_xlabel('成功次数 k')
axes[0].set_ylabel('P(X=k)')
axes[0].set_title(f'二项分布 PMF (n={n_trials})')
axes[0].legend()
axes[0].set_xticks(x_binom)

# 不同 n 值的二项分布 PMF
for n_val in [10, 20, 50]:
    x = np.arange(0, n_val + 1)
    pmf = stats.binom.pmf(x, n_val, 0.3)
    axes[1].bar(x, pmf, width=0.8, label=f'n={n_val}', alpha=0.7)
axes[1].set_xlabel('成功次数 k')
axes[1].set_ylabel('P(X=k)')
axes[1].set_title('二项分布 PMF (p=0.3, 不同n)')
axes[1].legend()

plt.tight_layout()
plt.savefig('d2_binomial.png', dpi=150)
plt.close()
print("\n  [图] 二项分布图已保存为 d2_binomial.png")


# ============================================================
# 2. 泊松分布
# ============================================================
print("\n" + "=" * 60)
print("2. 泊松分布")
print("=" * 60)

# 泊松分布：单位时间/空间内随机事件发生次数
# P(X=k) = (λ^k * e^(-λ)) / k!
lambdas = [1, 3, 5, 10]

print("\n  泊松分布性质：")
for lam in lambdas:
    poisson_dist = stats.poisson(lam)
    print(f"    λ={lam:2d}: 均值={poisson_dist.mean():.2f}, "
          f"方差={poisson_dist.var():.2f}, "
          f"标准差={poisson_dist.std():.4f}")

# 可视化泊松分布
fig, ax = plt.subplots(figsize=(10, 5))

for lam in lambdas:
    x = np.arange(0, 25)
    pmf = stats.poisson.pmf(x, lam)
    ax.bar(x + (lam - 5.5) * 0.2, pmf, width=0.2,
           label=f'λ={lam}', alpha=0.85)

ax.set_xlabel('事件发生次数 k')
ax.set_ylabel('P(X=k)')
ax.set_title('泊松分布 PMF（不同 λ 值）')
ax.legend()
ax.set_xticks(range(0, 25, 2))

plt.tight_layout()
plt.savefig('d2_poisson.png', dpi=150)
plt.close()
print("  [图] 泊松分布图已保存为 d2_poisson.png")

# 模拟：每小时接到的电话数（λ=5）
np.random.seed(42)
calls = stats.poisson.rvs(mu=5, size=1000)
print(f"\n  模拟每小时电话数 (λ=5, 1000小时):")
print(f"    模拟均值 = {np.mean(calls):.2f}")
print(f"    模拟方差 = {np.var(calls):.2f}")


# ============================================================
# 3. 正态分布：PDF和CDF，68-95-99.7法则
# ============================================================
print("\n" + "=" * 60)
print("3. 正态分布：PDF、CDF与68-95-99.7法则")
print("=" * 60)

mu, sigma = 0, 1  # 标准正态分布
norm_dist = stats.norm(mu, sigma)

# 68-95-99.7 法则
print(f"\n  标准正态分布 N({mu}, {sigma}²):")
for k in [1, 2, 3]:
    prob = norm_dist.cdf(mu + k * sigma) - norm_dist.cdf(mu - k * sigma)
    print(f"    P(μ-{k}σ < X < μ+{k}σ) = P({-k} < X < {k}) = {prob:.4f} = {prob:.2%}")

# 绘制PDF和CDF
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# PDF (概率密度函数)
x = np.linspace(-4, 4, 200)
pdf = norm_dist.pdf(x)

# 绘制不同区间的阴影
axes[0].plot(x, pdf, 'b-', linewidth=2, label='PDF')
axes[0].fill_between(x, pdf, where=(x >= -1) & (x <= 1),
                      alpha=0.3, color='red', label='68.27%')
axes[0].fill_between(x, pdf, where=((x >= -2) & (x < -1)) | ((x > 1) & (x <= 2)),
                      alpha=0.3, color='orange', label='95.45%')
axes[0].fill_between(x, pdf, where=((x >= -3) & (x < -2)) | ((x > 2) & (x <= 3)),
                      alpha=0.3, color='green', label='99.73%')
axes[0].set_xlabel('x')
axes[0].set_ylabel('概率密度 f(x)')
axes[0].set_title('标准正态分布 PDF 与 68-95-99.7 法则')
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)

# CDF (累积分布函数)
cdf = norm_dist.cdf(x)
axes[1].plot(x, cdf, 'b-', linewidth=2, label='CDF')
axes[1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='P=0.5')
axes[1].axhline(y=0.8413, color='orange', linestyle='--', alpha=0.5)
axes[1].axhline(y=0.9772, color='green', linestyle='--', alpha=0.5)
axes[1].axhline(y=0.9987, color='purple', linestyle='--', alpha=0.5)

# 标注关键点
for k, color in [(1, 'orange'), (2, 'green'), (3, 'purple')]:
    axes[1].plot(k, norm_dist.cdf(k), 'o', color=color, markersize=6)
    axes[1].plot(-k, norm_dist.cdf(-k), 'o', color=color, markersize=6)
    axes[1].annotate(f'{norm_dist.cdf(k):.4f}', xy=(k, norm_dist.cdf(k)),
                     xytext=(k + 0.3, norm_dist.cdf(k) - 0.05), fontsize=8)

axes[1].set_xlabel('x')
axes[1].set_ylabel('累积概率 F(x)')
axes[1].set_title('标准正态分布 CDF')
axes[1].legend(loc='upper left')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('d2_normal_pdf_cdf.png', dpi=150)
plt.close()
print("\n  [图] 正态分布PDF/CDF图已保存为 d2_normal_pdf_cdf.png")


# ============================================================
# 4. 不同参数正态分布对比
# ============================================================
print("\n" + "=" * 60)
print("4. 不同参数正态分布对比")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 不同均值，相同方差
x = np.linspace(-6, 10, 300)
for mu_val in [-2, 0, 2, 4]:
    pdf = stats.norm.pdf(x, mu_val, 1)
    axes[0].plot(x, pdf, linewidth=2, label=f'μ={mu_val}, σ=1')
axes[0].set_xlabel('x')
axes[0].set_ylabel('f(x)')
axes[0].set_title('不同均值的正态分布 (σ=1)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 相同均值，不同方差
for sigma_val in [0.5, 1.0, 2.0, 3.0]:
    pdf = stats.norm.pdf(x, 0, sigma_val)
    axes[1].plot(x, pdf, linewidth=2, label=f'μ=0, σ={sigma_val}')
axes[1].set_xlabel('x')
axes[1].set_ylabel('f(x)')
axes[1].set_title('不同标准差的正态分布 (μ=0)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('d2_normal_comparison.png', dpi=150)
plt.close()
print("  [图] 正态分布对比图已保存为 d2_normal_comparison.png")

print("\n" + "=" * 60)
print("D2 完成！")
print("=" * 60)
