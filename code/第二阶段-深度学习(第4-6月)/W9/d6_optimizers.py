"""
d6_optimizers.py - 优化算法
============================
- 从零实现 SGD, Momentum, RMSprop, Adam
- 在 Rosenbrock 函数上测试
- 绘制等高线图上的优化轨迹
- 对比收敛速度
- 展示学习率调度策略
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. Rosenbrock 函数定义
# ============================================================
def rosenbrock(x, y):
    """Rosenbrock 函数: f(x,y) = (1-x)^2 + 100*(y-x^2)^2"""
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def rosenbrock_grad(x, y):
    """Rosenbrock 函数的梯度"""
    df_dx = -2 * (1 - x) - 400 * x * (y - x ** 2)
    df_dy = 200 * (y - x ** 2)
    return np.array([df_dx, df_dy])


# ============================================================
# 2. 优化器实现
# ============================================================

class SGD:
    """随机梯度下降"""
    def __init__(self, lr=0.001):
        self.lr = lr

    def step(self, params, grad):
        return params - self.lr * grad


class Momentum:
    """带动量的 SGD"""
    def __init__(self, lr=0.001, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.v = None

    def step(self, params, grad):
        if self.v is None:
            self.v = np.zeros_like(params)
        self.v = self.beta * self.v + (1 - self.beta) * grad
        return params - self.lr * self.v


class RMSprop:
    """RMSprop 优化器"""
    def __init__(self, lr=0.01, beta=0.99, eps=1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.s = None

    def step(self, params, grad):
        if self.s is None:
            self.s = np.zeros_like(params)
        self.s = self.beta * self.s + (1 - self.beta) * grad ** 2
        return params - self.lr * grad / (np.sqrt(self.s) + self.eps)


class Adam:
    """Adam 优化器"""
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grad):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2
        # 偏差修正
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ============================================================
# 3. 运行优化并记录轨迹
# ============================================================
def optimize(optimizer, start, n_iters=5000):
    """使用给定优化器优化 Rosenbrock 函数"""
    params = start.copy()
    trajectory = [params.copy()]
    losses = [rosenbrock(params[0], params[1])]

    for _ in range(n_iters):
        grad = rosenbrock_grad(params[0], params[1])
        params = optimizer.step(params, grad)
        trajectory.append(params.copy())
        losses.append(rosenbrock(params[0], params[1]))

    return np.array(trajectory), np.array(losses)


# 起始点和参数
start_point = np.array([-1.0, 1.0])  # Rosenbrock 最小值在 (1, 1)
n_iters = 3000

print("=" * 60)
print("优化 Rosenbrock 函数: f(x,y) = (1-x)^2 + 100*(y-x^2)^2")
print(f"起始点: ({start_point[0]}, {start_point[1]})")
print(f"最小值点: (1, 1), 最小值: 0")
print(f"迭代次数: {n_iters}")
print("=" * 60)

# 各优化器配置（学习率经过调优以确保收敛）
optimizers = {
    'SGD (lr=0.001)': SGD(lr=0.001),
    'Momentum (lr=0.001)': Momentum(lr=0.001, beta=0.9),
    'RMSprop (lr=0.005)': RMSprop(lr=0.005, beta=0.99),
    'Adam (lr=0.01)': Adam(lr=0.01),
}

results = {}
for name, opt in optimizers.items():
    # 每个优化器需要独立的实例
    opt_class = type(opt)
    opt_new = opt_class(**{k: v for k, v in opt.__dict__.items()
                           if not k.startswith('_')})
    trajectory, losses = optimize(opt_new, start_point, n_iters)
    results[name] = (trajectory, losses)
    final_x, final_y = trajectory[-1]
    print(f"\n{name}:")
    print(f"  终点: ({final_x:.4f}, {final_y:.4f})")
    print(f"  最终损失: {losses[-1]:.6f}")
    # 收敛到损失 < 0.01 的迭代次数
    converged = np.where(losses < 0.01)[0]
    if len(converged) > 0:
        print(f"  收敛到 <0.01 的迭代次数: {converged[0]}")
    else:
        print(f"  在 {n_iters} 次迭代内未收敛到 <0.01")


# ============================================================
# 4. 可视化优化轨迹
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 绘制等高线
x_range = np.linspace(-1.5, 2.0, 400)
y_range = np.linspace(-0.5, 2.0, 400)
X, Y = np.meshgrid(x_range, y_range)
Z = rosenbrock(X, Y)

colors_opt = ['blue', 'green', 'orange', 'red']

for idx, (name, (trajectory, losses)) in enumerate(results.items()):
    ax = axes[idx // 2][idx % 2]

    # 等高线（对数尺度，使低值区域更清晰）
    levels = np.logspace(-3, 3, 50)
    ax.contour(X, Y, Z, levels=levels, cmap='gray_r', alpha=0.5, linewidths=0.5)

    # 优化轨迹
    # 降采样以便可视化（每隔 step 取一个点）
    step = max(1, len(trajectory) // 200)
    traj = trajectory[::step]

    ax.plot(traj[:, 0], traj[:, 1], color=colors_opt[idx], linewidth=1.5,
            alpha=0.8, label=name)
    ax.scatter(traj[0, 0], traj[0, 1], color='blue', s=100, zorder=5,
               marker='o', label='起点', edgecolors='black')
    ax.scatter(traj[-1, 0], traj[-1, 1], color='red', s=100, zorder=5,
               marker='*', label='终点', edgecolors='black')
    ax.scatter(1, 1, color='gold', s=150, zorder=5, marker='D',
               edgecolors='black', label='最优点 (1,1)')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(name)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_xlim(-1.5, 2.0)
    ax.set_ylim(-0.5, 2.0)

plt.suptitle("四种优化算法在 Rosenbrock 函数上的轨迹", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("d6_optimizers_trajectories.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n图1保存完成: d6_optimizers_trajectories.png")


# ============================================================
# 5. 收敛速度对比
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 损失曲线（对数尺度）
for idx, (name, (_, losses)) in enumerate(results.items()):
    axes[0].plot(losses, color=colors_opt[idx], linewidth=1.5, label=name)

axes[0].set_xlabel('迭代次数')
axes[0].set_ylabel('损失值 (对数尺度)')
axes[0].set_yscale('log')
axes[0].set_title('收敛速度对比')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 到最优点的距离
for idx, (name, (trajectory, _)) in enumerate(results.items()):
    optimal = np.array([1.0, 1.0])
    distances = np.linalg.norm(trajectory - optimal, axis=1)
    axes[1].plot(distances, color=colors_opt[idx], linewidth=1.5, label=name)

axes[1].set_xlabel('迭代次数')
axes[1].set_ylabel('到最优点的距离 (对数尺度)')
axes[1].set_yscale('log')
axes[1].set_title('到最优点 (1,1) 的距离')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle("优化算法收敛速度对比", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("d6_optimizers_convergence.png", dpi=150, bbox_inches='tight')
plt.show()
print("图2保存完成: d6_optimizers_convergence.png")


# ============================================================
# 6. 学习率调度策略
# ============================================================
print("\n" + "=" * 60)
print("学习率调度策略")
print("=" * 60)

n_epochs = 100

# StepLR: 每 step_size 个 epoch 乘以 gamma
def step_lr(initial_lr, epoch, step_size=30, gamma=0.1):
    return initial_lr * (gamma ** (epoch // step_size))

# CosineAnnealing: 余弦退火
def cosine_annealing_lr(initial_lr, epoch, T_max=n_epochs):
    return initial_lr * (1 + np.cos(np.pi * epoch / T_max)) / 2

# ExponentialLR: 指数衰减
def exponential_lr(initial_lr, epoch, gamma=0.95):
    return initial_lr * (gamma ** epoch)

# WarmupCosine: 先线性升温再余弦退火
def warmup_cosine_lr(initial_lr, epoch, warmup_epochs=10, T_max=n_epochs):
    if epoch < warmup_epochs:
        return initial_lr * epoch / warmup_epochs
    return initial_lr * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (T_max - warmup_epochs))) / 2

initial_lr = 0.1
epochs_range = np.arange(n_epochs)
lrs_step = [step_lr(initial_lr, e) for e in epochs_range]
lrs_cosine = [cosine_annealing_lr(initial_lr, e) for e in epochs_range]
lrs_exp = [exponential_lr(initial_lr, e) for e in epochs_range]
lrs_warmup = [warmup_cosine_lr(initial_lr, e) for e in epochs_range]

# 打印一些关键节点
print(f"\n初始学习率: {initial_lr}")
print(f"\n{'Epoch':>6} | {'StepLR':>10} | {'CosineAnnealing':>16} | {'Exponential':>12} | {'WarmupCosine':>13}")
print("-" * 70)
for e in [0, 10, 20, 30, 50, 70, 90, 99]:
    print(f"{e:>6} | {step_lr(initial_lr, e):>10.6f} | "
          f"{cosine_annealing_lr(initial_lr, e):>16.6f} | "
          f"{exponential_lr(initial_lr, e):>12.6f} | "
          f"{warmup_cosine_lr(initial_lr, e):>13.6f}")

fig, ax = plt.subplots(1, 1, figsize=(12, 5))

ax.plot(epochs_range, lrs_step, 'b-', linewidth=2, label='StepLR (step=30, gamma=0.1)')
ax.plot(epochs_range, lrs_cosine, 'r-', linewidth=2, label='CosineAnnealing')
ax.plot(epochs_range, lrs_exp, 'g-', linewidth=2, label='Exponential (gamma=0.95)')
ax.plot(epochs_range, lrs_warmup, 'purple', linewidth=2, label='WarmupCosine (warmup=10)')

ax.set_xlabel('Epoch')
ax.set_ylabel('学习率')
ax.set_title('学习率调度策略对比')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("d6_lr_schedules.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n图3保存完成: d6_lr_schedules.png")

print("\n" + "=" * 60)
print("总结:")
print("  - SGD: 最简单的优化器，在 Rosenbrock 上收敛最慢")
print("  - Momentum: 利用历史梯度加速，在狭长山谷中效果显著")
print("  - RMSprop: 自适应学习率，对不同参数使用不同更新幅度")
print("  - Adam: 结合 Momentum + RMSprop，通常是最好的默认选择")
print("  - 学习率调度：StepLR 阶梯下降，CosineAnnealing 平滑衰减")
print("=" * 60)
