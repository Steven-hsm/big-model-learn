"""
W04-D5 梯度下降
================
内容：
1. 实现 gradient_descent() 函数
2. 求 f(x) = (x+1)² 的最小值
3. 不同学习率对比（2x2子图）
4. 2D梯度下降 + 等高线图 + 路径
5. 收敛行为分析
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 梯度下降函数实现
# ============================================================
print("=" * 60)
print("1. 梯度下降函数实现")
print("=" * 60)


def gradient_descent(grad, x0, lr=0.01, n_iters=100, tol=1e-8):
    """
    通用梯度下降算法

    Parameters:
        grad: 梯度函数，输入当前参数，返回梯度
        x0: 初始参数（标量或数组）
        lr: 学习率 (learning rate)
        n_iters: 最大迭代次数
        tol: 收敛阈值（梯度范数小于此值时停止）

    Returns:
        path: 参数更新的路径列表
        losses: 每次迭代的损失值（如果返回梯度时附带函数值）
    """
    x = np.array(x0, dtype=float)
    path = [x.copy()]

    for i in range(n_iters):
        g = np.array(grad(x))

        # 检查收敛
        if np.linalg.norm(g) < tol:
            print(f"    在第 {i+1} 次迭代收敛，梯度范数={np.linalg.norm(g):.2e}")
            break

        # 梯度下降更新：x = x - lr * grad
        x = x - lr * g
        path.append(x.copy())

    return path


print("  gradient_descent(grad, x0, lr, n_iters) 函数已定义")
print("  更新规则: x_{t+1} = x_t - lr * ∇f(x_t)")


# ============================================================
# 2. 求 f(x) = (x+1)² 的最小值
# ============================================================
print("\n" + "=" * 60)
print("2. 求 f(x) = (x+1)² 的最小值")
print("=" * 60)


def f1(x):
    """f(x) = (x+1)²"""
    return (x + 1) ** 2


def grad_f1(x):
    """f'(x) = 2(x+1)"""
    return 2 * (x + 1)


# 从 x=3 开始梯度下降
x0 = 3.0
print(f"\n  函数: f(x) = (x+1)²")
print(f"  解析最小值: x = -1, f(-1) = 0")
print(f"  初始点: x₀ = {x0}")

path = gradient_descent(grad_f1, x0, lr=0.1, n_iters=50)

print(f"\n  迭代过程:")
print(f"  {'迭代':<8s} {'x值':<15s} {'f(x)':<15s} {'梯度':<15s}")
print("  " + "-" * 53)

for i, x_val in enumerate(path):
    if i <= 10 or i == len(path) - 1 or i % 5 == 0:
        print(f"  {i:<8d} {x_val:<15.8f} {f1(x_val):<15.8f} {grad_f1(x_val):<15.8f}")
    elif i == 11:
        print("  ...")

final_x = path[-1]
print(f"\n  最终结果: x = {final_x:.8f}, f(x) = {f1(final_x):.2e}")
print(f"  与解析解的误差: {abs(final_x - (-1)):.2e}")


# ============================================================
# 3. 不同学习率对比
# ============================================================
print("\n" + "=" * 60)
print("3. 不同学习率对比")
print("=" * 60)

learning_rates = [0.01, 0.1, 0.9, 1.05]
n_iters_test = 50

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

# 绘制函数曲线
x_plot = np.linspace(-3, 4, 300)
y_plot = f1(x_plot)

for idx, lr in enumerate(learning_rates):
    ax = axes[idx]

    # 运行梯度下降
    path = gradient_descent(grad_f1, x0=3.0, lr=lr, n_iters=n_iters_test)
    path_arr = np.array(path)
    loss_arr = np.array([f1(x) for x in path])

    # 绘制函数
    ax.plot(x_plot, y_plot, 'b-', linewidth=1.5, alpha=0.5)

    # 绘制梯度下降路径
    for i in range(len(path) - 1):
        ax.annotate('', xy=(path[i+1], f1(path[i+1])),
                     xytext=(path[i], f1(path[i])),
                     arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    ax.plot(path_arr, loss_arr, 'ro-', markersize=4, linewidth=1, alpha=0.7)
    ax.plot(path_arr[0], loss_arr[0], 'gs', markersize=10, label=f'起点 x₀={path_arr[0]:.1f}')
    ax.plot(path_arr[-1], loss_arr[-1], 'b*', markersize=12, label=f'终点 x={path_arr[-1]:.4f}')

    # 分析收敛行为
    if lr < 0.5:
        behavior = "稳定收敛"
    elif lr < 1.0:
        behavior = "快速收敛（有振荡）"
    elif lr == 1.0:
        behavior = "临界（不收敛）"
    else:
        behavior = "发散！"

    ax.set_title(f'lr = {lr}  ({behavior})\n'
                 f'最终 x={path_arr[-1]:.6f}, iter={len(path)-1}')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 设置合理的y轴范围
    if lr > 1.0:
        ax.set_ylim(-5, max(loss_arr.max(), 20))
    else:
        ax.set_ylim(-0.5, f1(3.0) + 1)

    print(f"  lr={lr:<6s} 迭代次数={len(path)-1:<5d} "
          f"最终x={path_arr[-1]:.8f} 最终f(x)={f1(path_arr[-1]):.2e}  {behavior}")

plt.suptitle('梯度下降：不同学习率对 f(x) = (x+1)² 的影响', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('d5_learning_rates.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n  [图] 不同学习率对比图已保存为 d5_learning_rates.png")


# ============================================================
# 4. 2D梯度下降：f(x,y) = x² + 4y²
# ============================================================
print("\n" + "=" * 60)
print("4. 2D梯度下降：f(x,y) = x² + 4y²")
print("=" * 60)


def f2(xy):
    """f(x,y) = x² + 4y²"""
    return xy[0]**2 + 4 * xy[1]**2


def grad_f2(xy):
    """∇f = [2x, 8y]"""
    return np.array([2 * xy[0], 8 * xy[1]])


# 解析最小值在 (0, 0)
print(f"\n  函数: f(x,y) = x² + 4y²")
print(f"  梯度: ∇f = [2x, 8y]")
print(f"  解析最小值: (0, 0), f(0,0) = 0")

# 从不同起始点进行梯度下降
start_points = [
    (np.array([2.5, 2.5]), 0.05, "起点(2.5,2.5), lr=0.05"),
    (np.array([2.5, 2.5]), 0.1,  "起点(2.5,2.5), lr=0.1"),
    (np.array([-2.0, 1.5]), 0.08, "起点(-2.0,1.5), lr=0.08"),
]

fig, ax = plt.subplots(figsize=(9, 8))

# 绘制等高线
x_grid = np.linspace(-3, 3, 200)
y_grid = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(x_grid, y_grid)
Z = X**2 + 4*Y**2

# 等高线
levels = [0.1, 0.5, 1, 2, 4, 8, 16, 32]
cs = ax.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.7)
ax.clabel(cs, inline=True, fontsize=8)

colors = ['red', 'blue', 'orange']
for idx, (start, lr, desc) in enumerate(start_points):
    path = gradient_descent(grad_f2, start, lr=lr, n_iters=100)
    path_arr = np.array(path)

    # 绘制路径
    ax.plot(path_arr[:, 0], path_arr[:, 1], '-o', color=colors[idx],
            markersize=3, linewidth=1.5, label=desc, alpha=0.8)
    ax.plot(path_arr[0, 0], path_arr[0, 1], 's', color=colors[idx],
            markersize=10, zorder=5)
    ax.plot(path_arr[-1, 0], path_arr[-1, 1], '*', color=colors[idx],
            markersize=12, zorder=5)

    print(f"\n  {desc}:")
    print(f"    迭代次数: {len(path)-1}")
    print(f"    终点: ({path_arr[-1, 0]:.6f}, {path_arr[-1, 1]:.6f})")
    print(f"    f(终点): {f2(path_arr[-1]):.2e}")

# 标记最优点
ax.plot(0, 0, 'k*', markersize=15, label='最优点 (0,0)', zorder=10)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('2D梯度下降: f(x,y) = x² + 4y² 等高线与下降路径')
ax.legend(loc='upper right')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('d5_gradient_descent_2d.png', dpi=150)
plt.close()
print("\n  [图] 2D梯度下降图已保存为 d5_gradient_descent_2d.png")


# ============================================================
# 5. 收敛行为分析
# ============================================================
print("\n" + "=" * 60)
print("5. 收敛行为分析")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：损失曲线对比
for lr in [0.01, 0.05, 0.1, 0.2, 0.5]:
    path = gradient_descent(grad_f2, np.array([2.5, 2.5]), lr=lr, n_iters=100)
    losses = [f2(p) for p in path]
    axes[0].plot(range(len(losses)), losses, '-o', markersize=2,
                 label=f'lr={lr}', linewidth=1.5)

axes[0].set_xlabel('迭代次数')
axes[0].set_ylabel('f(x,y)')
axes[0].set_title('不同学习率的收敛曲线')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_yscale('log')

# 右图：lr vs 最终损失
lr_range = np.linspace(0.01, 0.3, 30)
final_losses = []
n_iters_all = []

for lr in lr_range:
    path = gradient_descent(grad_f2, np.array([2.5, 2.5]), lr=lr, n_iters=100)
    final_losses.append(f2(path[-1]))
    n_iters_all.append(len(path) - 1)

axes[1].plot(lr_range, final_losses, 'b-', linewidth=2)
axes[1].set_xlabel('学习率')
axes[1].set_ylabel('最终损失 f(x,y)')
axes[1].set_title('学习率 vs 最终损失 (100次迭代)')
axes[1].grid(True, alpha=0.3)

# 标注最佳学习率
best_idx = np.argmin(final_losses)
axes[1].plot(lr_range[best_idx], final_losses[best_idx], 'r*', markersize=12)
axes[1].annotate(f'最佳lr={lr_range[best_idx]:.3f}\n损失={final_losses[best_idx]:.2e}',
                 xy=(lr_range[best_idx], final_losses[best_idx]),
                 xytext=(lr_range[best_idx]+0.03, final_losses[best_idx]*5),
                 arrowprops=dict(arrowstyle='->', color='red'))

plt.tight_layout()
plt.savefig('d5_convergence_analysis.png', dpi=150)
plt.close()
print("  [图] 收敛分析图已保存为 d5_convergence_analysis.png")

print("\n" + "=" * 60)
print("D5 完成！")
print("=" * 60)
