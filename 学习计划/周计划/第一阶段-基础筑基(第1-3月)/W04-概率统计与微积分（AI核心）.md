# W04 - 概率统计与微积分（AI核心）

> 第4周 | Java开发工程师转AI开发学习计划
> 工作日每晚2小时 | 周末6-8小时

---

## 一、本周目标

1. 掌握概率基础：条件概率、贝叶斯定理及其在AI中的应用
2. 理解常见概率分布及其参数含义
3. 掌握期望、方差、最大似然估计（MLE）
4. 理解导数、梯度和梯度下降的数学原理
5. 能从零用NumPy实现线性回归

---

## 二、时间安排

| 日期 | 类型 | 时长 | 主题 |
|------|------|------|------|
| Day 1 | 工作日晚 | 2h | 概率基础与贝叶斯 |
| Day 2 | 工作日晚 | 2h | 常见概率分布 |
| Day 3 | 工作日晚 | 2h | 统计基础与MLE |
| Day 4 | 工作日晚 | 2h | 导数与偏导数 |
| Day 5 | 工作日晚 | 2h | 梯度下降 |
| Day 6 | 周末 | 3-4h | 链式法则与反向传播 |
| Day 7 | 周末 | 3-4h | 综合实战：NumPy线性回归 |

---

## 三、详细学习内容

### Day 1 - 概率基础与贝叶斯（2h）

**目标**：理解概率论基础，掌握贝叶斯定理。

**1. 概率基础**

```python
import numpy as np

# 样本空间Ω：所有可能结果的集合
# 事件：样本空间的子集
# 概率P(A)：事件A发生的可能性，0 ≤ P(A) ≤ 1

# 古典概率（等可能事件）
# P(A) = A的有利结果数 / 总结果数
# 例：掷骰子得到偶数的概率 = 3/6 = 0.5

# 用NumPy模拟
dice_rolls = np.random.randint(1, 7, size=100000)
prob_even = np.mean(dice_rolls % 2 == 0)
print(f"掷骰子得偶数的概率（模拟）: {prob_even:.4f}")  # ≈ 0.5

# 加法规则：P(A∪B) = P(A) + P(B) - P(A∩B)
# 互斥事件：P(A∪B) = P(A) + P(B)

# 乘法规则：P(A∩B) = P(A) * P(B|A) = P(B) * P(A|B)
```

**2. 条件概率**

```python
# 条件概率：在B已发生的条件下A发生的概率
# P(A|B) = P(A∩B) / P(B)

# 例：在一副牌中，已知抽到的是红色牌，求是A的概率
# P(A|红) = P(A∩红) / P(红) = (2/52) / (26/52) = 2/26 = 1/13

# 全概率公式
# P(A) = Σ P(A|B_i) * P(B_i)
# 把事件A的概率分解为在各种条件下的加权平均

# 独立事件：P(A|B) = P(A)，即B的发生不影响A
# 等价于 P(A∩B) = P(A) * P(B)
```

**3. 贝叶斯定理（重点！）**

```python
# 贝叶斯定理：P(H|D) = P(D|H) * P(H) / P(D)
#
# H = Hypothesis（假设）
# D = Data（观测数据）
# P(H)     = 先验概率（Prior）：看到数据前对假设的相信程度
# P(D|H)   = 似然（Likelihood）：假设为真时观测到数据的概率
# P(H|D)   = 后验概率（Posterior）：看到数据后对假设的相信程度
# P(D)     = 证据（Evidence）：观测到数据的总概率（归一化常数）
#
# 核心思想：用观测数据更新对假设的信念
#
# 贝叶斯定理在AI中的应用：
# - 朴素贝叶斯分类器
# - 贝叶斯优化（超参数调优）
# - 贝叶斯神经网络（不确定性估计）

# 例：医疗检测
# 某疾病发病率 1%（先验）
# 检测灵敏度 99%（有病检测为阳性）
# 检测假阳性率 5%（没病检测为阳性）
# 问题：检测为阳性时，真正有病的概率？

prior_sick = 0.01
likelihood_pos_given_sick = 0.99
likelihood_pos_given_healthy = 0.05
prior_healthy = 1 - prior_sick

# P(阳性) = P(阳性|有病)*P(有病) + P(阳性|没病)*P(没病)
evidence = (likelihood_pos_given_sick * prior_sick +
            likelihood_pos_given_healthy * prior_healthy)

# P(有病|阳性) = P(阳性|有病) * P(有病) / P(阳性)
posterior = likelihood_pos_given_sick * prior_sick / evidence
print(f"阳性时真正有病的概率: {posterior:.2%}")  # ≈ 16.6%
# 结果出人意料地低！因为先验概率太低（发病率只有1%）
```

**4. 贝叶斯垃圾邮件分类器实现**

```python
class NaiveBayesSpamFilter:
    """朴素贝叶斯垃圾邮件分类器"""

    def __init__(self):
        self.spam_word_counts = {}   # 垃圾邮件中各词的出现次数
        self.ham_word_counts = {}    # 正常邮件中各词的出现次数
        self.spam_count = 0          # 垃圾邮件总数
        self.ham_count = 0           # 正常邮件总数
        self.vocab = set()           # 词汇表

    def train(self, emails, labels):
        """训练
        emails: 邮件文本列表
        labels: 标签列表（1=垃圾邮件, 0=正常邮件）
        """
        for email, label in zip(emails, labels):
            words = set(email.lower().split())
            self.vocab.update(words)

            if label == 1:  # 垃圾邮件
                self.spam_count += 1
                for word in words:
                    self.spam_word_counts[word] = self.spam_word_counts.get(word, 0) + 1
            else:  # 正常邮件
                self.ham_count += 1
                for word in words:
                    self.ham_word_counts[word] = self.ham_word_counts.get(word, 0) + 1

    def predict(self, email):
        """预测邮件是否为垃圾邮件"""
        words = set(email.lower().split())
        vocab_size = len(self.vocab)

        # 先验概率（拉普拉斯平滑）
        total = self.spam_count + self.ham_count
        log_p_spam = np.log(self.spam_count / total)
        log_p_ham = np.log(self.ham_count / total)

        # 似然（对每个词累加log概率，避免下溢）
        for word in words:
            # P(word|spam) 拉普拉斯平滑
            p_word_spam = (self.spam_word_counts.get(word, 0) + 1) / (self.spam_count + vocab_size)
            p_word_ham = (self.ham_word_counts.get(word, 0) + 1) / (self.ham_count + vocab_size)

            log_p_spam += np.log(p_word_spam)
            log_p_ham += np.log(p_word_ham)

        return 1 if log_p_spam > log_p_ham else 0

# 测试
train_emails = [
    "buy cheap viagra online",      # 垃圾
    "hello how are you",             # 正常
    "free money click here",         # 垃圾
    "meeting at 3pm today",          # 正常
    "win prize lottery now",         # 垃圾
    "project deadline reminder",     # 正常
    "cheap discount sale buy now",   # 垃圾
    "lunch tomorrow at noon",        # 正常
]
train_labels = [1, 0, 1, 0, 1, 0, 1, 0]

filter = NaiveBayesSpamFilter()
filter.train(train_emails, train_labels)

test_emails = [
    "buy now cheap offer",
    "meeting reminder for tomorrow",
    "free prize win click",
    "how are you today"
]
for email in test_emails:
    result = "垃圾邮件" if filter.predict(email) == 1 else "正常邮件"
    print(f"[{result}] {email}")
```

---

### Day 2 - 常见概率分布（2h）

**目标**：理解常见概率分布及其参数含义，能用scipy可视化。

**1. 离散分布**

```python
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# 伯努利分布 Bernoulli(p)：单次实验，成功(1)或失败(0)
# 例：抛一次硬币，P(正面)=0.5
p = 0.5
bernoulli = stats.bernoulli(p)
print(f"P(X=1) = {bernoulli.pmf(1)}")  # 0.5
print(f"均值 = {bernoulli.mean()}")     # 0.5
print(f"方差 = {bernoulli.var()}")      # 0.25

# 二项分布 Binomial(n, p)：n次独立实验中成功的次数
# 例：抛10次硬币，正面朝上的次数
n, p = 10, 0.5
binom = stats.binom(n, p)
x = np.arange(0, 11)
plt.figure(figsize=(8, 5))
plt.bar(x, binom.pmf(x))
plt.title(f'Binomial(n={n}, p={p})')
plt.xlabel('Number of successes')
plt.ylabel('Probability')
plt.show()

# 泊松分布 Poisson(λ)：单位时间内稀有事件发生的次数
# 例：每小时收到邮件的数量（平均5封）
lam = 5
poisson = stats.poisson(lam)
x = np.arange(0, 20)
plt.bar(x, poisson.pmf(x))
plt.title(f'Poisson(λ={lam})')
plt.show()

# 多项式分布：n次实验中各类别出现的次数
# 例：掷骰子60次，每个面出现的次数
# AI中用于文本分类（多项式朴素贝叶斯）
```

**2. 连续分布**

```python
# 正态分布 Normal(μ, σ²)：最重要的分布！
# 68-95-99.7法则：
# 68%的数据在μ±σ内
# 95%的数据在μ±2σ内
# 99.7%的数据在μ±3σ内

mu, sigma = 0, 1  # 标准正态分布
normal = stats.norm(mu, sigma)

x = np.linspace(-4, 4, 100)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x, normal.pdf(x))  # 概率密度函数 PDF
plt.title('Standard Normal PDF')
plt.axvline(mu, color='r', linestyle='--', label='μ')
plt.axvline(mu+sigma, color='g', linestyle='--', label='μ+σ')
plt.axvline(mu-sigma, color='g', linestyle='--', label='μ-σ')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x, normal.cdf(x))  # 累积分布函数 CDF
plt.title('Standard Normal CDF')

plt.tight_layout()
plt.show()

# 正态分布在AI中无处不在：
# - 权重初始化常用正态分布
# - 假设误差服从正态分布 → MSE损失
# - VAE（变分自编码器）用正态分布建模潜在空间

# 不同参数的正态分布对比
plt.figure(figsize=(10, 6))
for mu, sigma in [(0, 1), (0, 2), (2, 1)]:
    x = np.linspace(-6, 8, 200)
    plt.plot(x, stats.norm(mu, sigma).pdf(x),
             label=f'N({mu}, {sigma}²)')
plt.legend()
plt.title('Normal Distributions with Different Parameters')
plt.show()
```

---

### Day 3 - 统计基础与MLE（2h）

**目标**：掌握期望、方差、协方差，理解最大似然估计。

**1. 期望与方差**

```python
# 期望 E[X]：随机变量的加权平均值（"长期平均"）
# 离散：E[X] = Σ x_i * P(x_i)
# 连续：E[X] = ∫ x * f(x) dx

import numpy as np

# 用样本均值估计期望
data = np.random.normal(loc=5.0, scale=2.0, size=10000)
sample_mean = np.mean(data)        # ≈ 5.0
print(f"样本均值（E[X]的估计）: {sample_mean:.2f}")

# 方差 Var(X) = E[(X - μ)²]：衡量数据的离散程度
sample_var = np.var(data, ddof=1)  # ddof=1用n-1（无偏估计）
sample_std = np.std(data, ddof=1)  # 标准差
print(f"样本方差: {sample_var:.2f}")  # ≈ 4.0
print(f"样本标准差: {sample_std:.2f}")  # ≈ 2.0

# 期望的性质
# E[aX + b] = aE[X] + b
# E[X + Y] = E[X] + E[Y]（无条件成立）
```

**2. 协方差与相关系数**

```python
# 协方差 Cov(X, Y) = E[(X-μ_x)(Y-μ_y)]
# 衡量两个变量的线性关系
# Cov > 0：正相关，< 0：负相关，= 0：无线性关系

x = np.random.normal(0, 1, 1000)
y = 0.8 * x + np.random.normal(0, 0.5, 1000)  # 正相关
z = np.random.normal(0, 1, 1000)                # 无关

print(f"Cov(x, y) = {np.cov(x, y)[0, 1]:.3f}")  # 正值
print(f"Cov(x, z) = {np.cov(x, z)[0, 1]:.3f}")  # ≈ 0

# 相关系数 Corr(X, Y) = Cov(X, Y) / (σ_x * σ_y)
# 值域 [-1, 1]，1=完全正相关，-1=完全负相关，0=无线性关系
print(f"Corr(x, y) = {np.corrcoef(x, y)[0, 1]:.3f}")  # ≈ 0.8
print(f"Corr(x, z) = {np.corrcoef(x, z)[0, 1]:.3f}")  # ≈ 0

# AI中：相关系数矩阵用于特征选择（删除高相关特征）
```

**3. 最大似然估计（MLE）**

```python
# MLE核心思想：找到使观测数据出现概率最大的参数
# θ_MLE = argmax_θ P(D|θ) = argmax_θ ∏ P(x_i|θ)
# 取对数：θ_MLE = argmax_θ Σ log P(x_i|θ)

# 例1：正态分布的MLE
# 假设数据X ~ N(μ, σ²)，求μ和σ的MLE
np.random.seed(42)
true_mu, true_sigma = 3.0, 1.5
data = np.random.normal(true_mu, true_sigma, 1000)

# MLE推导结果：
# μ_MLE = (1/n) Σ x_i = 样本均值
# σ²_MLE = (1/n) Σ (x_i - μ)² = 样本方差（有偏版本）
mu_mle = np.mean(data)
sigma2_mle = np.mean((data - mu_mle)**2)

print(f"真实μ={true_mu}, MLE估计μ={mu_mle:.3f}")
print(f"真实σ²={true_sigma**2}, MLE估计σ²={sigma2_mle:.3f}")

# 手动实现MLE（以正态分布为例）
def neg_log_likelihood(mu, sigma, data):
    """负对数似然（最小化 = 最大化似然）"""
    n = len(data)
    return n/2 * np.log(2*np.pi*sigma**2) + np.sum((data - mu)**2) / (2*sigma**2)

from scipy.optimize import minimize
result = minimize(
    lambda params: neg_log_likelihood(params[0], params[1], data),
    x0=[0, 1],  # 初始值
    bounds=[(None, None), (0.01, None)]  # sigma > 0
)
print(f"优化结果: μ={result.x[0]:.3f}, σ={result.x[1]:.3f}")
```

---

### Day 4 - 导数与偏导数（2h）

**目标**：理解导数的概念和计算方法。

**1. 导数定义**

```python
# 导数定义：f'(x) = lim(h→0) [f(x+h) - f(x)] / h
# 几何意义：函数在某点的切线斜率
# 物理意义：瞬时变化率

import numpy as np
import matplotlib.pyplot as plt

def numerical_derivative(f, x, h=1e-5):
    """数值求导（用定义近似）"""
    return (f(x + h) - f(x - h)) / (2 * h)

# 例：f(x) = x², f'(x) = 2x
f = lambda x: x**2
f_prime = lambda x: 2*x

x0 = 3.0
print(f"解析导数: f'({x0}) = {f_prime(x0)}")
print(f"数值导数: f'({x0}) = {numerical_derivative(f, x0):.6f}")

# 可视化
x = np.linspace(-3, 5, 100)
y = f(x)
tangent_slope = f_prime(x0)
tangent_y = f(x0) + tangent_slope * (x - x0)

plt.figure(figsize=(8, 6))
plt.plot(x, y, 'b-', label='f(x) = x²')
plt.plot(x, tangent_y, 'r--', label=f'tangent at x={x0}')
plt.plot(x0, f(x0), 'ro')
plt.legend()
plt.grid(True)
plt.title('Derivative as Tangent Line')
plt.show()
```

**2. 常见函数导数**

```python
# 常用导数公式表：
# f(x) = c       → f'(x) = 0
# f(x) = x^n     → f'(x) = n*x^(n-1)
# f(x) = e^x     → f'(x) = e^x
# f(x) = ln(x)   → f'(x) = 1/x
# f(x) = sin(x)  → f'(x) = cos(x)
# f(x) = cos(x)  → f'(x) = -sin(x)

# 求导法则：
# 1. 和法则：(f+g)' = f' + g'
# 2. 积法则：(fg)' = f'g + fg'
# 3. 商法则：(f/g)' = (f'g - fg') / g²
# 4. 链式法则：(f(g(x)))' = f'(g(x)) * g'(x)  ← 最重要的法则！
```

**3. 偏导数与梯度**

```python
# 偏导数：多元函数对一个变量求导（其他变量视为常数）
# 例：f(x, y) = x² + xy + y²
# ∂f/∂x = 2x + y
# ∂f/∂y = x + 2y

def f(x, y):
    return x**2 + x*y + y**2

def partial_f_x(x, y, h=1e-5):
    """∂f/∂x 的数值计算"""
    return (f(x+h, y) - f(x-h, y)) / (2*h)

def partial_f_y(x, y, h=1e-5):
    """∂f/∂y 的数值计算"""
    return (f(x, y+h) - f(x, y-h)) / (2*h)

x0, y0 = 1.0, 2.0
print(f"∂f/∂x = {partial_f_x(x0, y0):.4f}")  # ≈ 2*1+2 = 4
print(f"∂f/∂y = {partial_f_y(x0, y0):.4f}")  # ≈ 1+2*2 = 5

# 梯度：所有偏导数组成的向量
# ∇f = [∂f/∂x, ∂f/∂y]
gradient = np.array([partial_f_x(x0, y0), partial_f_y(x0, y0)])
print(f"梯度: {gradient}")

# 梯度的意义：函数在当前点最陡的上升方向
# AI中：梯度指向损失增长最快的方向，所以我们要沿负梯度方向走（梯度下降）

# 雅可比矩阵（多输出函数）
# 如果 f: R^n → R^m，则雅可比矩阵 J[i][j] = ∂f_i/∂x_j
# AI中：神经网络每一层的梯度就是一个雅可比矩阵
```

---

### Day 5 - 梯度下降（2h）

**目标**：理解梯度下降的原理，实现并可视化。

**1. 梯度下降原理**

```python
import numpy as np
import matplotlib.pyplot as plt

# 梯度下降算法：
# x_new = x_old - α * ∇f(x_old)
# α = 学习率（步长）
# 沿负梯度方向走 → 最速下降方向

# 例1：求 f(x) = x² + 2x + 1 = (x+1)² 的最小值
# 解析解：x = -1, f(-1) = 0
def f(x):
    return x**2 + 2*x + 1

def grad_f(x):
    return 2*x + 2

def gradient_descent(grad, x0, lr=0.1, n_iters=50):
    """梯度下降"""
    x = x0
    history = [x]
    for _ in range(n_iters):
        x = x - lr * grad(x)
        history.append(x)
    return x, history

x_min, history = gradient_descent(grad_f, x0=3.0, lr=0.1)
print(f"最小值点: x = {x_min:.6f}, f(x) = {f(x_min):.6f}")
```

**2. 学习率的影响**

```python
# 不同学习率的对比
learning_rates = [0.01, 0.1, 0.9, 1.05]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
x_plot = np.linspace(-4, 4, 100)

for ax, lr in zip(axes.flatten(), learning_rates):
    x_min, history = gradient_descent(grad_f, x0=3.0, lr=lr, n_iters=50)
    f_history = [f(x) for x in history]

    ax.plot(x_plot, f(x_plot), 'b-', label='f(x)')
    ax.plot(history, f_history, 'ro-', markersize=3, label=f'lr={lr}')
    ax.set_title(f'lr={lr}, converged={"Yes" if abs(x_min+1)<0.1 else "No"}')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()

# 总结：
# lr太小（0.01）→ 收敛很慢
# lr适中（0.1）→ 快速收敛
# lr较大（0.9）→ 震荡但最终收敛
# lr太大（1.05）→ 发散！
```

**3. 多变量梯度下降**

```python
# 例：f(x, y) = x² + 4y²（椭圆抛物面）
def f_2d(x, y):
    return x**2 + 4*y**2

def grad_f_2d(x, y):
    return np.array([2*x, 8*y])

def gradient_descent_2d(grad, x0, lr=0.1, n_iters=50):
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    for _ in range(n_iters):
        x = x - lr * grad(x[0], x[1])
        history.append(x.copy())
    return x, np.array(history)

x_min, history = gradient_descent_2d(grad_f_2d, [3.0, 2.0], lr=0.05)

# 可视化：等高线 + 路径
plt.figure(figsize=(10, 8))
x_grid = np.linspace(-4, 4, 100)
y_grid = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x_grid, y_grid)
Z = f_2d(X, Y)

plt.contour(X, Y, Z, levels=20)
plt.plot(history[:, 0], history[:, 1], 'ro-', markersize=3)
plt.plot(0, 0, 'g*', markersize=15, label='Minimum')
plt.title('Gradient Descent on f(x,y) = x² + 4y²')
plt.legend()
plt.grid(True)
plt.show()
```

---

### Day 6 - 链式法则与反向传播（3-4h）

**目标**：理解链式法则如何驱动反向传播。

**1. 计算图**

```python
# 计算图：将计算过程画成有向无环图（DAG）
# 例：f(x, y, z) = (x + y) * z
# 前向传播：
#   a = x + y
#   f = a * z
# 反向传播：
#   ∂f/∂a = z          （乘法的导数）
#   ∂f/∂z = a
#   ∂f/∂x = ∂f/∂a * ∂a/∂x = z * 1 = z  （链式法则！）
#   ∂f/∂y = ∂f/∂a * ∂a/∂y = z * 1 = z

# 用代码验证
x, y, z = 2, 3, 4
a = x + y
f = a * z
print(f"f = {f}")  # 20

# 反向传播
df_df = 1
df_dz = a        # ∂f/∂z = a = 5
df_da = z        # ∂f/∂a = z = 4
da_dx = 1        # ∂a/∂x = 1
da_dy = 1        # ∂a/∂y = 1

df_dx = df_da * da_dx  # ∂f/∂x = 4 * 1 = 4
df_dy = df_da * da_dy  # ∂f/∂y = 4 * 1 = 4

print(f"∂f/∂x = {df_dx}")  # 4
print(f"∂f/∂y = {df_dy}")  # 4
print(f"∂f/∂z = {df_dz}")  # 5
```

**2. 反向传播（以2层网络为例）**

```python
import numpy as np

# 2层神经网络：
# h = sigmoid(W1 @ x + b1)    隐藏层
# y = sigmoid(W2 @ h + b2)    输出层
# L = 0.5 * (y - t)²          MSE损失

# 前向传播
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(a):
    """sigmoid的导数，a=sigmoid(z)"""
    return a * (1 - a)

# 初始化
np.random.seed(42)
x = np.array([[0.5], [0.3]])     # 输入 (2,1)
t = np.array([[1.0]])            # 目标 (1,1)
W1 = np.random.randn(3, 2) * 0.5 # (3,2)
b1 = np.zeros((3, 1))
W2 = np.random.randn(1, 3) * 0.5 # (1,3)
b2 = np.zeros((1, 1))

# Forward pass
z1 = W1 @ x + b1           # (3,1)
a1 = sigmoid(z1)            # (3,1)  隐藏层激活
z2 = W2 @ a1 + b2           # (1,1)
a2 = sigmoid(z2)            # (1,1)  输出
loss = 0.5 * np.sum((a2 - t)**2)

print(f"预测值: {a2[0,0]:.4f}")
print(f"损失: {loss:.4f}")

# Backward pass（链式法则逐层求导）
# ∂L/∂a2 = a2 - t
dL_da2 = a2 - t                          # (1,1)
# ∂L/∂z2 = ∂L/∂a2 * sigmoid'(z2)
dL_dz2 = dL_da2 * sigmoid_deriv(a2)      # (1,1)
# ∂L/∂W2 = ∂L/∂z2 * a1.T
dL_dW2 = dL_dz2 @ a1.T                   # (1,3)
# ∂L/∂b2 = ∂L/∂z2
dL_db2 = dL_dz2                           # (1,1)

# ∂L/∂a1 = W2.T @ ∂L/∂z2
dL_da1 = W2.T @ dL_dz2                   # (3,1)
# ∂L/∂z1 = ∂L/∂a1 * sigmoid'(z1)
dL_dz1 = dL_da1 * sigmoid_deriv(a1)      # (3,1)
# ∂L/∂W1 = ∂L/∂z1 @ x.T
dL_dW1 = dL_dz1 @ x.T                    # (3,2)
# ∂L/∂b1 = ∂L/∂z1
dL_db1 = dL_dz1                           # (3,1)

# 梯度下降更新
lr = 0.1
W1 -= lr * dL_dW1
b1 -= lr * dL_db1
W2 -= lr * dL_dW2
b2 -= lr * dL_db2

print(f"\n更新后:")
z1_new = W1 @ x + b1
a1_new = sigmoid(z1_new)
z2_new = W2 @ a1_new + b2
a2_new = sigmoid(z2_new)
loss_new = 0.5 * np.sum((a2_new - t)**2)
print(f"预测值: {a2_new[0,0]:.4f}")
print(f"损失: {loss_new:.4f}（应该减小）")
```

---

### Day 7 - 综合实战：NumPy线性回归（3-4h）

**目标**：从零用NumPy实现完整的线性回归。

```python
import numpy as np
import matplotlib.pyplot as plt

# ============ 1. 生成数据 ============
np.random.seed(42)
n_samples = 200
X = np.random.randn(n_samples, 1) * 2
true_w, true_b = 3.5, 1.2
y = true_w * X.flatten() + true_b + np.random.randn(n_samples) * 0.5

print(f"真实参数: w={true_w}, b={true_b}")

# ============ 2. 假设函数 ============
# h(x) = w*x + b
def predict(X, w, b):
    return X @ w + b

# ============ 3. 损失函数 ============
# MSE = (1/n) * Σ(h(x) - y)²
def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true)**2)

# ============ 4. 梯度计算 ============
# ∂L/∂w = (2/n) * Σ(h(x) - y) * x
# ∂L/∂b = (2/n) * Σ(h(x) - y)
def compute_gradients(X, y, w, b):
    n = len(y)
    y_pred = predict(X, w, b)
    error = y_pred - y
    dw = (2/n) * (X.flatten() @ error)
    db = (2/n) * np.sum(error)
    return dw, db

# ============ 5. 梯度下降训练 ============
def train(X, y, lr=0.01, n_epochs=100):
    w = np.random.randn()
    b = 0.0

    losses = []
    for epoch in range(n_epochs):
        y_pred = predict(X, w, b)
        loss = mse_loss(y_pred, y)
        losses.append(loss)

        dw, db = compute_gradients(X, y, w, b)
        w -= lr * dw
        b -= lr * db

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}: loss={loss:.4f}, w={w:.4f}, b={b:.4f}")

    return w, b, losses

# 训练
w_trained, b_trained, losses = train(X, y, lr=0.05, n_epochs=200)
print(f"\n训练结果: w={w_trained:.4f}, b={b_trained:.4f}")
print(f"真实参数: w={true_w}, b={true_b}")

# ============ 6. 可视化 ============
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 损失曲线
axes[0].plot(losses)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')
axes[0].set_title('Training Loss Curve')
axes[0].grid(True)

# 回归线
x_line = np.linspace(X.min(), X.max(), 100)
axes[1].scatter(X.flatten(), y, alpha=0.5, s=10, label='Data')
axes[1].plot(x_line, w_trained * x_line + b_trained, 'r-', linewidth=2,
             label=f'y={w_trained:.2f}x+{b_trained:.2f}')
axes[1].set_xlabel('X')
axes[1].set_ylabel('y')
axes[1].set_title('Linear Regression Fit')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('linear_regression.png', dpi=150)
plt.show()

# ============ 7. 解析解对比 ============
# 正规方程：w = (X.T @ X)^(-1) @ X.T @ y
X_b = np.column_stack([np.ones(n_samples), X.flatten()])  # 添加偏置列
w_normal = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
print(f"\n正规方程解: w={w_normal[1]:.4f}, b={w_normal[0]:.4f}")
print(f"梯度下降解: w={w_trained:.4f}, b={b_trained:.4f}")
print(f"真实参数:   w={true_w}, b={true_b}")

# ============ 8. 多元线性回归 ============
# 多特征版本
np.random.seed(42)
n_samples = 200
n_features = 3
X_multi = np.random.randn(n_samples, n_features)
true_W = np.array([2.0, -1.0, 0.5])
true_b_multi = 1.0
y_multi = X_multi @ true_W + true_b_multi + np.random.randn(n_samples) * 0.3

# 解析解
X_b_multi = np.column_stack([np.ones(n_samples), X_multi])
theta = np.linalg.inv(X_b_multi.T @ X_b_multi) @ X_b_multi.T @ y_multi
print(f"\n多元回归解析解:")
print(f"偏置 b = {theta[0]:.4f}（真实 {true_b_multi}）")
print(f"权重 W = {theta[1:]}（真实 {true_W}）")
```

---

## 四、代码练习

### Day 1 练习：贝叶斯垃圾邮件分类器

（见Day 1详细内容中的完整实现）

### Day 5 练习：不同学习率的梯度下降

```python
# gradient_descent_comparison.py
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return (x - 2)**2 + 1

def grad_f(x):
    return 2 * (x - 2)

learning_rates = [0.01, 0.1, 1.0, 10.0]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, lr in zip(axes.flatten(), learning_rates):
    x = 10.0  # 起点
    history = [x]
    for _ in range(50):
        x = x - lr * grad_f(x)
        history.append(x)

    ax.plot(history, 'bo-', markersize=3)
    ax.axhline(y=2, color='r', linestyle='--', label='x*=2')
    ax.set_title(f'lr={lr}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('x')
    ax.legend()
    ax.grid(True)

plt.suptitle('Gradient Descent with Different Learning Rates')
plt.tight_layout()
plt.savefig('learning_rates.png', dpi=150)
plt.show()
```

### Day 7 练习：波士顿房价线性回归

```python
# 使用sklearn的加州房价数据集（波士顿房价已弃用）
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# 加载数据
housing = fetch_california_housing()
X, y = housing.data, housing.target

# 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 用NumPy实现正规方程
X_b = np.column_stack([np.ones(len(X_train_scaled)), X_train_scaled])
theta = np.linalg.lstsq(X_b, y_train, rcond=None)[0]

# 预测
X_test_b = np.column_stack([np.ones(len(X_test_scaled)), X_test_scaled])
y_pred = X_test_b @ theta

# 评估
mse = np.mean((y_pred - y_test)**2)
rmse = np.sqrt(mse)
r2 = 1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - y_test.mean())**2)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
```

---

## 五、本周产出

| 产出物 | 说明 |
|--------|------|
| 贝叶斯垃圾邮件分类器 | 朴素贝叶斯完整实现 |
| 概率分布可视化 | 6种分布的PDF/PMF图 |
| 梯度下降对比图 | 4种学习率的收敛过程 |
| 反向传播手动计算 | 2层网络的梯度推导 |
| NumPy线性回归 | 从零实现 + 可视化 |
| 梯度下降动态可视化 | 损失曲线 + 回归线 |

---

## 六、自测题

1. **贝叶斯定理中先验、似然、后验分别是什么？**

   <details>
   <summary>参考答案</summary>

   - 先验 P(H)：在观测数据之前对假设的信念（如疾病的发病率）
   - 似然 P(D|H)：假设为真时观测到数据的概率（如患病时检测为阳性的概率）
   - 后验 P(H|D)：观测到数据后对假设的更新信念（如检测为阳性时真正患病的概率）
   - 核心公式：后验 ∝ 似然 × 先验
   </details>

2. **为什么深度学习常用交叉熵而不是MSE做分类损失？**

   <details>
   <summary>参考答案</summary>

   - MSE用于分类时，配合sigmoid/softmax会导致梯度在预测接近0或1时非常小（学习速度极慢）
   - 交叉熵的梯度与误差成正比，不受饱和区影响，学习速度更稳定
   - MSE假设误差服从正态分布（适合回归），交叉熵来自最大似然（适合分类）
   </details>

3. **链式法则在反向传播中的作用？**

   <details>
   <summary>参考答案</summary>

   反向传播需要计算损失函数对每个参数的梯度。神经网络是多层复合函数，链式法则允许我们将复合函数的导数分解为各层导数的乘积：∂L/∂w = ∂L/∂a_n * ∂a_n/∂a_{n-1} * ... * ∂a_1/∂w。反向传播就是从输出层到输入层逐层应用链式法则。
   </details>

4. **学习率太大或太小分别会怎样？**

   <details>
   <summary>参考答案</summary>

   - 太大：步长过大，跳过最优点，甚至发散（损失增加）
   - 太小：收敛极慢，可能陷入局部最优
   - 合适：损失平稳下降，在合理迭代次数内收敛
   - 实践中常用学习率调度：初始较大快速接近，后期逐渐减小精细调整
   </details>

5. **MLE的核心思想是什么？**

   <details>
   <summary>参考答案</summary>

   MLE（最大似然估计）的核心思想是：在所有可能的参数中，选择使已观测数据出现概率最大的那个参数。即 θ_MLE = argmax P(D|θ)。具体步骤：(1)写出似然函数 ∏ P(x_i|θ)；(2)取对数得到对数似然（连乘变连加，方便计算）；(3)对参数求导令其为0，解方程得MLE估计。线性回归的MSE损失就是正态分布假设下MLE的结果。
   </details>

---

## 七、Java开发者提示

| Java/工程概念 | 数学概念对应 | 说明 |
|--------------|-------------|------|
| 单元测试断言 | 概率 | 概率就是对"某个结果出现"的量化预期 |
| 日志权重 | 贝叶斯更新 | 先验=已有经验，似然=新证据，后验=更新后的认知 |
| HashMap冲突概率 | 正态分布 | 大部分数据集中在均值附近，极端值很少 |
| 性能优化的profiling | 梯度 | 梯度告诉你在哪个方向努力效果最大 |
| 迭代优化 | 梯度下降 | 一小步一小步地改进，逐步逼近最优 |
| 责任链模式 | 链式法则 | 每层只关心自己的梯度，乘以上一层传来的梯度 |
| 递归 | 反向传播 | 反向传播本质上是反向递归地应用链式法则 |

**核心洞察**：机器学习 = 概率论（建模不确定性）+ 微积分（优化参数）+ 线性代数（高效计算）。数学不是障碍，而是理解AI模型行为的工具。
