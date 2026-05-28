###### 理解特征值/特征向量的几何意义和计算方法
# 特征值和特征向量的定义：
# A @ v = λ * v
# A是方阵，v是特征向量（方向不变），λ是特征值（缩放倍数）
A = np.array([[4, 2], [1, 3]], dtype=float)
eigenvalues, eigenvectors = np.linalg.eig(A)

print("特征值:", eigenvalues)
print("特征向量:\n", eigenvectors)

# 验证 A @ v = λ * v
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lambda_i = eigenvalues[i]
    assert np.allclose(A @ v, lambda_i * v)
    print(f"λ={lambda_i:.2f}, v={v}, A@v={A@v}, λ*v={lambda_i*v}")

# 几何意义：
# 特征向量 v 是矩阵 A 变换后方向不变的向量
# 特征值 λ 是该方向上的缩放倍数
# λ > 0：同方向缩放
# λ < 0：反方向
# |λ| > 1：拉伸，|λ| < 1：压缩

#对称矩阵的特征值
# 对称矩阵的特征值都是实数，特征向量互相正交
S = np.array([[2, 1], [1, 3]], dtype=float)
eigenvalues, eigenvectors = np.linalg.eigh(S)  # 注意用eigh（对称专用）

print("特征值（实数）:", eigenvalues)
print("特征向量正交验证:", np.allclose(eigenvectors[:, 0] @ eigenvectors[:, 1], 0))

# 谱分解：对称矩阵 A = Q Λ Q.T
# Q是特征向量矩阵（正交），Λ是对角特征值矩阵
Q = eigenvectors
Lambda = np.diag(eigenvalues)
reconstructed = Q @ Lambda @ Q.T
assert np.allclose(reconstructed, S)
print("谱分解验证通过: A = Q Λ Q.T")

## 正定/半正定矩阵判断
def check_positive_definite(A, tol=1e-10):
    """检查矩阵是否正定"""
    eigenvalues = np.linalg.eigvalsh(A)
    print(f"特征值: {eigenvalues}")
    if np.all(eigenvalues > tol):
        print("正定矩阵（所有特征值 > 0）")
    elif np.all(eigenvalues >= -tol):
        print("半正定矩阵（所有特征值 >= 0）")
    else:
        print("不定矩阵（有正有负）")

# 正定：f(x) = x.T A x > 0 对所有 x ≠ 0 成立
# AI中意义：损失函数的海森矩阵正定 → 局部极小值存在
# 协方差矩阵总是半正定的

A_pd = np.array([[2, 0], [0, 3]])    # 正定（对角元都正）
A_psd = np.array([[1, 1], [1, 1]])   # 半正定
check_positive_definite(A_pd)
check_positive_definite(A_psd)
