"""
W04-D4 导数与偏导数
====================
内容：
1. 数值导数（中心差分法）
2. 验证 f(x)=x² 的导数
3. 函数 + 切线可视化
4. 偏导数 f(x,y) = x² + xy + y²
5. 梯度向量计算
6. 常见导数公式参考
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 数值导数函数（中心差分法）
# ============================================================
print("=" * 60)
print("1. 数值导数函数（中心差分法）")
print("=" * 60)


def numerical_derivative(f, x, h=1e-7):
    """
    使用中心差分法计算数值导数

    f'(x) ≈ [f(x+h) - f(x-h)] / (2h)

    Parameters:
        f: 目标函数
        x: 求导点
        h: 步长（默认1e-7）

    Returns:
        f 在 x 处的数值导数
    """
    return (f(x + h) - f(x - h)) / (2 * h)


print("\n  中心差分公式: f'(x) ≈ [f(x+h) - f(x-h)] / (2h)")
print("  优点：精度为 O(h²)，比前向/后向差分更精确")


# ============================================================
# 2. 验证 f(x) = x² 的导数
# ============================================================
print("\n" + "=" * 60)
print("2. 验证 f(x) = x² 的导数")
print("=" * 60)


def f_square(x):
    """f(x) = x²"""
    return x ** 2


# 解析导数：f'(x) = 2x
# 在 x=3 处：f'(3) = 6
x_test = 3.0
analytical_deriv = 2 * x_test  # = 6
numerical_deriv = numerical_derivative(f_square, x_test)

print(f"\n  函数: f(x) = x²")
print(f"  解析导数: f'(x) = 2x")
print(f"  在 x={x_test} 处:")
print(f"    解析解 f'({x_test}) = {analytical_deriv}")
print(f"    数值解 f'({x_test}) = {numerical_deriv:.10f}")
print(f"    绝对误差 = {abs(analytical_deriv - numerical_deriv):.2e}")

# 测试更多函数
print(f"\n  验证其他函数的导数:")
test_cases = [
    ("sin(x)",  np.sin,  np.cos,    np.pi/4),
    ("exp(x)",  np.exp,  np.exp,    1.0),
    ("ln(x)",   np.log,  lambda x: 1/x, 2.0),
    ("x³",      lambda x: x**3, lambda x: 3*x**2, 2.0),
    ("1/x",     lambda x: 1/x,  lambda x: -1/x**2, 2.0),
]

for name, f, df, x_val in test_cases:
    num_d = numerical_derivative(f, x_val)
    ana_d = df(x_val)
    err = abs(num_d - ana_d)
    print(f"    {name:>8s} 在 x={x_val}: 数值={num_d:.8f}, "
          f"解析={ana_d:.8f}, 误差={err:.2e}")


# ============================================================
# 3. 函数 + 切线可视化
# ============================================================
print("\n" + "=" * 60)
print("3. 函数 + 切线可视化")
print("=" * 60)


def plot_function_with_tangent(f, df_analytical, x_range, tangent_points, title):
    """
    绘制函数及其在指定点的切线

    Parameters:
        f: 函数
        df_analytical: 解析导数函数
        x_range: (x_min, x_max) 绘图范围
        tangent_points: 切线点的x坐标列表
        title: 图标题
    """
    x = np.linspace(x_range[0], x_range[1], 300)
    y = f(x)

    fig, ax = plt.subplots(figsize=(9, 6))

    # 绘制函数
    ax.plot(x, y, 'b-', linewidth=2, label='f(x)')
    colors = ['red', 'green', 'orange', 'purple']

    for i, x0 in enumerate(tangent_points):
        color = colors[i % len(colors)]
        f_at_x0 = f(x0)
        slope = numerical_derivative(f, x0)

        # 切线方程: y = f(x0) + f'(x0) * (x - x0)
        tangent_y = f_at_x0 + slope * (x - x0)

        ax.plot(x, tangent_y, '--', color=color, linewidth=1.5, alpha=0.8,
                label=f"切线 x={x0}, 斜率={slope:.2f}")
        ax.plot(x0, f_at_x0, 'o', color=color, markersize=8, zorder=5)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


# f(x) = x² 的切线
fig1 = plot_function_with_tangent(
    f_square, lambda x: 2*x,
    x_range=(-1, 4),
    tangent_points=[-0.5, 1, 2, 3],
    title='f(x) = x² 及其切线'
)
plt.tight_layout()
plt.savefig('d4_tangent_x2.png', dpi=150)
plt.close()
print("  [图] f(x)=x²切线图已保存为 d4_tangent_x2.png")

# f(x) = sin(x) 的切线
fig2 = plot_function_with_tangent(
    np.sin, np.cos,
    x_range=(-2*np.pi, 2*np.pi),
    tangent_points=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
    title='f(x) = sin(x) 及其切线'
)
plt.tight_layout()
plt.savefig('d4_tangent_sin.png', dpi=150)
plt.close()
print("  [图] f(x)=sin(x)切线图已保存为 d4_tangent_sin.png")

# f(x) = exp(-x²) 的切线（高斯形状）
fig3 = plot_function_with_tangent(
    lambda x: np.exp(-x**2),
    lambda x: -2*x*np.exp(-x**2),
    x_range=(-3, 3),
    tangent_points=[-1.5, -0.5, 0, 0.5, 1.5],
    title='f(x) = exp(-x²) 及其切线'
)
plt.tight_layout()
plt.savefig('d4_tangent_gauss.png', dpi=150)
plt.close()
print("  [图] f(x)=exp(-x²)切线图已保存为 d4_tangent_gauss.png")


# ============================================================
# 4. 偏导数
# ============================================================
print("\n" + "=" * 60)
print("4. 偏导数：f(x,y) = x² + xy + y²")
print("=" * 60)


def f_xy(x, y):
    """f(x,y) = x² + xy + y²"""
    return x**2 + x*y + y**2


def partial_derivative(f, point, var_index, h=1e-7):
    """
    计算多变量函数在指定点的偏导数（数值方法）

    Parameters:
        f: 多变量函数 f(x1, x2, ..., xn)
        point: 求偏导的点，如 [x, y]
        var_index: 对第几个变量求偏导（0-indexed）
        h: 步长

    Returns:
        偏导数的数值近似
    """
    point_plus = np.array(point, dtype=float)
    point_minus = np.array(point, dtype=float)
    point_plus[var_index] += h
    point_minus[var_index] -= h
    return (f(*point_plus) - f(*point_minus)) / (2 * h)


# 解析偏导数
# ∂f/∂x = 2x + y
# ∂f/∂y = x + 2y

print(f"\n  函数: f(x,y) = x² + xy + y²")
print(f"  解析偏导数: ∂f/∂x = 2x + y,  ∂f/∂y = x + 2y")

test_points = [(1, 2), (0, 0), (3, -1), (-2, 1)]

print(f"\n  {'点(x,y)':<15s} {'∂f/∂x(数值)':<15s} {'∂f/∂x(解析)':<15s} "
      f"{'∂f/∂y(数值)':<15s} {'∂f/∂y(解析)':<15s}")
print("  " + "-" * 75)

for (x_val, y_val) in test_points:
    # 数值偏导数
    num_px = partial_derivative(f_xy, [x_val, y_val], 0)
    num_py = partial_derivative(f_xy, [x_val, y_val], 1)

    # 解析偏导数
    ana_px = 2 * x_val + y_val
    ana_py = x_val + 2 * y_val

    print(f"  ({x_val},{y_val}){'':<7s} {num_px:<15.8f} {ana_px:<15.8f} "
          f"{num_py:<15.8f} {ana_py:<15.8f}")


# ============================================================
# 5. 梯度向量
# ============================================================
print("\n" + "=" * 60)
print("5. 梯度向量")
print("=" * 60)


def gradient(f, point, h=1e-7):
    """
    计算多变量函数在指定点的梯度向量

    ∇f(x1,...,xn) = [∂f/∂x1, ∂f/∂x2, ..., ∂f/∂xn]

    Parameters:
        f: 多变量函数
        point: 求梯度的点
        h: 步长

    Returns:
        梯度向量（numpy数组）
    """
    point = np.array(point, dtype=float)
    grad = np.zeros_like(point)
    for i in range(len(point)):
        grad[i] = partial_derivative(f, point, i, h)
    return grad


print(f"\n  梯度 = [∂f/∂x, ∂f/∂y] 指向函数值增长最快的方向")

for (x_val, y_val) in test_points:
    grad = gradient(f_xy, [x_val, y_val])
    ana_grad = np.array([2*x_val + y_val, x_val + 2*y_val])
    print(f"\n  点 ({x_val}, {y_val}):")
    print(f"    数值梯度 ∇f = [{grad[0]:.8f}, {grad[1]:.8f}]")
    print(f"    解析梯度 ∇f = [{ana_grad[0]:.8f}, {ana_grad[1]:.8f}]")
    print(f"    梯度模 |∇f| = {np.linalg.norm(grad):.8f}")
    print(f"    梯度方向    = {np.degrees(np.arctan2(grad[1], grad[0])):.2f}°")

# 可视化：梯度场
fig, ax = plt.subplots(figsize=(8, 7))

x_grid = np.linspace(-3, 3, 20)
y_grid = np.linspace(-3, 3, 20)
X, Y = np.meshgrid(x_grid, y_grid)

# 计算每个网格点的梯度
U = 2 * X + Y  # ∂f/∂x
V = X + 2 * Y  # ∂f/∂y

# 绘制等高线（函数值）
Z = f_xy(X, Y)
contour = ax.contour(X, Y, Z, levels=20, cmap='coolwarm', alpha=0.6)
ax.clabel(contour, inline=True, fontsize=8)

# 绘制梯度向量场
ax.quiver(X, Y, U, V, Z, cmap='viridis', alpha=0.8)

# 标记一些特殊点
special_points = [(1, 2), (0, 0), (-1, -2)]
for (px, py) in special_points:
    grad_at_p = gradient(f_xy, [px, py])
    ax.plot(px, py, 'ro', markersize=8)
    ax.annotate(f'({px},{py})\n∇f=[{grad_at_p[0]:.1f},{grad_at_p[1]:.1f}]',
                xy=(px, py), xytext=(px+0.3, py+0.3), fontsize=9)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('f(x,y) = x² + xy + y² 的等高线与梯度场')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('d4_gradient_field.png', dpi=150)
plt.close()
print("\n  [图] 梯度场图已保存为 d4_gradient_field.png")


# ============================================================
# 6. 常见导数公式参考
# ============================================================
print("\n" + "=" * 60)
print("6. 常见导数公式参考")
print("=" * 60)

formulas = [
    ("常数",       "f(x) = c",         "f'(x) = 0"),
    ("幂函数",     "f(x) = xⁿ",        "f'(x) = nxⁿ⁻¹"),
    ("指数函数",   "f(x) = eˣ",        "f'(x) = eˣ"),
    ("指数函数",   "f(x) = aˣ",        "f'(x) = aˣ ln(a)"),
    ("自然对数",   "f(x) = ln(x)",     "f'(x) = 1/x"),
    ("正弦函数",   "f(x) = sin(x)",    "f'(x) = cos(x)"),
    ("余弦函数",   "f(x) = cos(x)",    "f'(x) = -sin(x)"),
    ("正切函数",   "f(x) = tan(x)",    "f'(x) = sec²(x)"),
    ("复合函数",   "f(g(x))",          "f'(x) = f'(g(x))·g'(x)  [链式法则]"),
    ("乘法法则",   "f(x) = u·v",       "f'(x) = u'v + uv'"),
    ("除法法则",   "f(x) = u/v",       "f'(x) = (u'v - uv') / v²"),
]

print(f"\n  {'类型':<12s} {'函数':<20s} {'导数':<40s}")
print("  " + "-" * 72)
for ftype, func, deriv in formulas:
    print(f"  {ftype:<12s} {func:<20s} {deriv:<40s}")

print("\n" + "=" * 60)
print("D4 完成！")
print("=" * 60)
