####### 理解矩阵运算的本质是线性变换
#矩阵乘法详解
A = np.array([[1, 2], [3, 4]])  # 2x2矩阵
B = np.array([[5, 6], [7, 8]])  # 2x2矩阵

C = A @ B
# C[0,0] = 1*5 + 2*7 = 19
# C[0,1] = 1*6 + 2*8 = 22
# C[1,0] = 3*5 + 4*7 = 43
# C[1,1] = 3*6 + 4*8 = 50
print(C)  # [[19, 22], [43, 50]]

# 矩阵 × 向量 = 线性变换（这是AI中最核心的操作！）
x = np.array([1, 0])
A @ x  # [1, 3]  → A将(1,0)变换为(1,3)

# 理解：矩阵就是对向量做变换的工具
# 旋转、缩放、剪切、投影都是线性变换
# 神经网络的权重矩阵W就是线性变换

# 矩阵乘法的性质
# (A @ B) @ C = A @ (B @ C)  结合律
# A @ (B + C) = A @ B + A @ C  分配律
# A @ B ≠ B @ A  不满足交换律！

#特殊矩阵
# 单位矩阵（乘法单位元，类似数字1）
I = np.eye(3)
A = np.random.randn(3, 3)
assert np.allclose(A @ I, A)  # A * I = A

# 转置
A = np.array([[1, 2], [3, 4]])
A.T  # [[1, 3], [2, 4]]
# 性质：(A @ B).T = B.T @ A.T

# 对称矩阵（A = A.T）
S = np.array([[1, 2], [2, 3]])
assert (S == S.T).all()

# 对角矩阵
D = np.diag([1, 2, 3])

# 逆矩阵
A = np.array([[1, 2], [3, 4]], dtype=float)

# 逆矩阵：A @ A_inv = I
A_inv = np.linalg.inv(A)
assert np.allclose(A @ A_inv, np.eye(2))

# 条件数：衡量矩阵接近奇异的程度
cond = np.linalg.cond(A)
print(f"条件数: {cond:.2f}")
# 条件数大 → 矩阵接近奇异 → 数值计算不稳定
# 条件数小 → 矩阵"健康" → 计算稳定

# 什么时候矩阵不可逆？
# 1. 行列式为0（奇异矩阵）
# 2. 不是方阵
# 3. 行或列线性相关

#行列式
A = np.array([[1, 2], [3, 4]], dtype=float)
det = np.linalg.det(A)
print(f"行列式: {det:.2f}")  # -2.0

# 行列式的几何意义：矩阵变换对面积的缩放比例
# det > 0：保持方向，det < 0：反转方向，det = 0：降维（面积压缩为0）

# 2x2行列式公式：det = a11*a22 - a12*a21

#矩阵的秩
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
rank = np.linalg.matrix_rank(A)
print(f"秩: {rank}")  # 2（第3行 = 2*第2行 - 第1行，线性相关）

# 秩的意义：
# 秩 = 矩阵中线性无关的行/列的最大数量
# 秩 < min(m,n) → 矩阵不满秩 → 不可逆

#线性变换的几何直观
import matplotlib.pyplot as plt

# 可视化矩阵变换
def plot_transform(A, title):
    """可视化线性变换"""
    # 原始单位正方形的顶点
    square = np.array([[0,0], [1,0], [1,1], [0,1], [0,0]]).T

    # 变换后的顶点
    transformed = A @ square

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(square[0], square[1], 'b-o')
    ax1.set_title('Original')
    ax1.set_xlim(-3, 3); ax1.set_ylim(-3, 3)
    ax1.grid(True); ax1.set_aspect('equal')

    ax2.plot(transformed[0], transformed[1], 'r-o')
    ax2.set_title(title)
    ax2.set_xlim(-3, 3); ax2.set_ylim(-3, 3)
    ax2.grid(True); ax2.set_aspect('equal')

    plt.show()

# 旋转45度
theta = np.pi / 4
R = np.array([[np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)]])
plot_transform(R, f'Rotation {np.degrees(theta):.0f}°')

# 缩放
S = np.array([[2, 0], [0, 0.5]])
plot_transform(S, 'Scaling (2x, 0.5y)')

# 剪切
H = np.array([[1, 1], [0, 1]])
plot_transform(H, 'Shear')