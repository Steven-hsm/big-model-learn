###### 理解各种矩阵分解方法及其应用场景
from scipy.linalg import lu

#LU分解
# A = L @ U （或 A = P @ L @ U，P是置换矩阵）
A = np.array([[2, 1, 1], [4, 3, 3], [8, 7, 9]], dtype=float)

P, L, U = lu(A)
print("P (置换矩阵):\n", P)
print("L (下三角):\n", L)
print("U (上三角):\n", U)
print("验证 P@L@U ≈ A:", np.allclose(P @ L @ U, A))

# 应用：解线性方程组 Ax = b
# LU分解后：Ly = Pb，Ux = y
# 优点：分解一次，可以解多个不同的b

# QR分解
from scipy.linalg import qr

# A = Q @ R
# Q是正交矩阵（Q.T @ Q = I），R是上三角矩阵
A = np.array([[1, 1], [1, 2], [1, 3]], dtype=float)

Q, R = qr(A, mode='economic')
print("Q (正交):\n", Q)
print("R (上三角):\n", R)
print("验证 Q@R ≈ A:", np.allclose(Q @ R, A))

# 应用：最小二乘法（线性回归）
# min ||Ax - b||² → R x = Q.T @ b
b = np.array([1, 2, 2], dtype=float)
x_qr = np.linalg.solve(R, Q.T @ b)
x_lstsq = np.linalg.lstsq(A, b, rcond=None)[0]
print("QR求解:", x_qr)
print("最小二乘:", x_lstsq)

# Cholesky分解
# A = L @ L.T （要求A是对称正定矩阵）
A = np.array([[4, 2], [2, 3]], dtype=float)  # 对称正定

L = np.linalg.cholesky(A)
print("L (下三角):\n", L)
print("验证 L@L.T ≈ A:", np.allclose(L @ L.T, A))

# 应用：比通用LU分解快2倍，用于求解正定方程组
# 正定矩阵判断：所有特征值 > 0
eigenvalues = np.linalg.eigvalsh(A)
print("特征值:", eigenvalues)  # 全正 → 正定

#PCA中的特征分解（预览）
# PCA的数学本质：对协方差矩阵做特征分解
data = np.random.randn(100, 3)  # 100个样本，3个特征

# 1. 中心化
data_centered = data - data.mean(axis=0)

# 2. 计算协方差矩阵
cov_matrix = np.cov(data_centered, rowvar=False)

# 3. 特征分解
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# 4. 按特征值从大到小排序
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("特征值（方差解释量）:", eigenvalues)
print("方差解释比:", eigenvalues / eigenvalues.sum())

