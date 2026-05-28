#############掌握广播机制、矩阵运算和线性代数操作
#广播机制（Broadcasting）
# 广播：不同形状的数组进行运算时，自动"扩展"小数组
# 3条规则：
# 1. 如果维度数不同，在较小数组的形状前面补1
# 2. 如果某个维度为1，则沿该维度复制扩展
# 3. 其他维度必须相等或其中一个为1

# 例1：标量 + 数组
a = np.array([1, 2, 3])
a + 10  # [11, 12, 13]  标量10被"广播"为[10,10,10]

# 例2：(3,1) + (1,4) → (3,4)
a = np.array([[1], [2], [3]])   # shape (3,1)
b = np.array([10, 20, 30, 40])  # shape (4,)
a + b  # shape (3,4)
# [[11, 21, 31, 41],
#  [12, 22, 32, 42],
#  [13, 23, 33, 43]]

# 例3：常见用法 - 数据标准化
data = np.random.randn(100, 5)  # 100个样本，5个特征
mean = data.mean(axis=0)         # 每个特征的均值 shape(5,)
std = data.std(axis=0)           # 每个特征的标准差 shape(5,)
normalized = (data - mean) / std  # 广播！data(100,5) - mean(5,) → (100,5)

###矩阵运算
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 逐元素运算（注意！不是矩阵乘法）
A * B    # [[5,12],[21,32]]  对应元素相乘
A + B    # [[6,8],[10,12]]
A ** 2   # [[1,4],[9,16]]

# 矩阵乘法（三种写法，推荐@）
np.dot(A, B)     # [[19,22],[43,50]]
np.matmul(A, B)  # 同上
A @ B            # 同上（推荐，最简洁）

# 转置
A.T              # [[1,3],[2,4]]

# 逆矩阵
A_inv = np.linalg.inv(A)
A @ A_inv        # ≈ 单位矩阵（浮点误差）

# 行列式
np.linalg.det(A)  # -2.0

# 矩阵的秩
np.linalg.matrix_rank(A)  # 2

# 解线性方程组 Ax = b
b = np.array([1, 2])
x = np.linalg.solve(A, b)  # 比 A_inv @ b 更稳定
np.allclose(A @ x, b)       # True，验证解的正确性

###线性代数
# 特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)

# SVD奇异值分解
U, S, Vt = np.linalg.svd(A)

# QR分解
Q, R = np.linalg.qr(A)

# Cholesky分解（要求矩阵正定）
L = np.linalg.cholesky(A @ A.T)  # 需要正定矩阵

# 范数
np.linalg.norm(A)       # Frobenius范数
np.linalg.norm(A, ord=1)  # L1范数
np.linalg.norm(A, ord=2)  # L2范数（最大奇异值）

# 条件数（衡量矩阵是否接近奇异，越大越不稳定）
np.linalg.cond(A)


##### 统计函数
arr = np.random.randn(1000)

arr.mean()       # 均值
arr.std()        # 标准差（注意：默认除以n，ddof=1则除以n-1即样本标准差）
arr.var()        # 方差
arr.sum()        # 求和
arr.cumsum()     # 累积和
arr.min()        # 最小值
arr.max()        # 最大值
arr.argmin()     # 最小值索引
arr.argmax()     # 最大值索引

# 沿轴计算（二维数组）
mat = np.random.randn(3, 4)
mat.mean(axis=0)  # 每列的均值 → shape(4,)
mat.mean(axis=1)  # 每行的均值 → shape(3,)
mat.sum(axis=0)   # 每列求和
mat.cumsum(axis=1) # 沿列累积

# 分位数
np.percentile(arr, [25, 50, 75])  # Q1, 中位数, Q3

# 相关系数矩阵
np.corrcoef(mat)  # 行之间的相关系数

