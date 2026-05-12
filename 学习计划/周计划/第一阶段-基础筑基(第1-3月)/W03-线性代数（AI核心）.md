# W03 - 线性代数（AI核心）

> 第3周 | Java开发工程师转AI开发学习计划
> 工作日每晚2小时 | 周末6-8小时

---

## 一、本周目标

1. 掌握向量运算：点积、范数、余弦相似度
2. 理解矩阵运算的本质：线性变换
3. 掌握矩阵分解：LU、QR、特征分解、SVD
4. 理解PCA的数学原理并实现降维
5. 建立线性代数与AI模型之间的映射关系

---

## 二、时间安排

| 日期 | 类型 | 时长 | 主题 |
|------|------|------|------|
| Day 1 | 工作日晚 | 2h | 向量运算 |
| Day 2 | 工作日晚 | 2h | 矩阵运算 |
| Day 3 | 工作日晚 | 2h | 矩阵分解 |
| Day 4 | 工作日晚 | 2h | 特征值与特征向量 |
| Day 5 | 工作日晚 | 2h | SVD奇异值分解 |
| Day 6 | 周末 | 3-4h | PCA主成分分析 |
| Day 7 | 周末 | 3-4h | 综合复习与思维导图 |

---

## 三、详细学习内容

### Day 1 - 向量运算（2h）

**目标**：理解向量的各种运算及其几何意义。

**1. 向量表示与基本运算**

```python
import numpy as np

# 向量表示（NumPy中向量就是一维数组）
row_vec = np.array([1, 2, 3])       # 行向量
col_vec = np.array([[1], [2], [3]])  # 列向量 (3,1)

# 向量加法（几何：平行四边形法则）
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
a + b  # [5, 7, 9]

# 标量乘法（几何：缩放向量长度）
2 * a  # [2, 4, 6]

# 向量减法
a - b  # [-3, -3, -3]
```

**2. 点积（Dot Product）—— 最重要**

```python
# 点积的计算
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 方法1：np.dot
np.dot(a, b)  # 1*4 + 2*5 + 3*6 = 32

# 方法2：@运算符
a @ b  # 32

# 方法3：手动计算
sum(a * b)  # 32

# 点积的几何意义
# a . b = |a| * |b| * cos(theta)
# 点积 = 一个向量在另一个向量方向上的投影 × 另一个向量的长度

# 点积 = 0 → 两向量正交（垂直）
a = np.array([1, 0])
b = np.array([0, 1])
np.dot(a, b)  # 0，正交

# AI中的意义：神经网络的每一层都在做点积运算
# y = W @ x + b，其中W是权重矩阵，x是输入向量
```

**3. 叉积（Cross Product）**

```python
# 叉积只在3D中有定义
a = np.array([1, 0, 0])
b = np.array([0, 1, 0])
np.cross(a, b)  # [0, 0, 1]  垂直于a和b的向量

# 几何意义：结果向量的长度 = 两个向量构成的平行四边形面积
```

**4. 向量范数（Norm）**

```python
a = np.array([3, 4])

# L1范数：所有元素绝对值之和（曼哈顿距离）
l1 = np.sum(np.abs(a))  # 7
# 或
l1 = np.linalg.norm(a, ord=1)  # 7.0

# L2范数：欧几里得长度（最常用）
l2 = np.linalg.norm(a)  # 5.0（3²+4²=25，√25=5）

# L-inf范数：最大元素的绝对值
linf = np.linalg.norm(a, ord=np.inf)  # 4.0

# AI中范数的用途：
# L1正则化（Lasso）→ 稀疏解，特征选择
# L2正则化（Ridge）→ 防止过拟合
# 梯度裁剪 → 防止梯度爆炸
```

**5. 余弦相似度**

```python
def cosine_similarity(a, b):
    """计算两个向量的余弦相似度
    cos(a,b) = (a.b) / (|a| * |b|)
    值域 [-1, 1]，1表示方向相同，-1表示方向相反，0表示正交
    """
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot / (norm_a * norm_b)

# 例：比较两段文本的相似度
doc1 = np.array([3, 0, 2, 1, 0])  # "AI machine learning deep"
doc2 = np.array([1, 2, 1, 0, 3])  # "machine data model train"
print(f"余弦相似度: {cosine_similarity(doc1, doc2):.4f}")

# AI应用：
# - 文本相似度（TF-IDF向量 → 余弦相似度）
# - 推荐系统（用户向量 → 余弦相似度找相似用户）
# - 词向量（Word2Vec → 余弦相似度衡量语义相似度）
```

**6. 向量正交**

```python
# 两个向量正交：点积为0
a = np.array([1, 0, 0])
b = np.array([0, 1, 0])
np.dot(a, b)  # 0 → 正交

# 正交向量组：任意两个向量都正交
# 标准正交基：正交 + 每个向量长度为1
# AI中的意义：特征之间正交 = 没有冗余信息 = 理想特征
```

---

### Day 2 - 矩阵运算（2h）

**目标**：理解矩阵运算的本质是线性变换。

**1. 矩阵乘法详解**

```python
A = np.array([[1, 2], [3, 4]])  # 2x2矩阵
B = np.array([[5, 6], [7, 8]])  # 2x2矩阵

# 矩阵乘法：A的行 × B的列
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
```

**2. 特殊矩阵**

```python
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
```

**3. 逆矩阵**

```python
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
```

**4. 行列式**

```python
A = np.array([[1, 2], [3, 4]], dtype=float)
det = np.linalg.det(A)
print(f"行列式: {det:.2f}")  # -2.0

# 行列式的几何意义：矩阵变换对面积的缩放比例
# det > 0：保持方向，det < 0：反转方向，det = 0：降维（面积压缩为0）

# 2x2行列式公式：det = a11*a22 - a12*a21
```

**5. 矩阵的秩**

```python
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
rank = np.linalg.matrix_rank(A)
print(f"秩: {rank}")  # 2（第3行 = 2*第2行 - 第1行，线性相关）

# 秩的意义：
# 秩 = 矩阵中线性无关的行/列的最大数量
# 秩 < min(m,n) → 矩阵不满秩 → 不可逆
```

**6. 线性变换的几何直观**

```python
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
```

---

### Day 3 - 矩阵分解（2h）

**目标**：理解各种矩阵分解方法及其应用场景。

**1. LU分解**

```python
from scipy.linalg import lu

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
```

**2. QR分解**

```python
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
```

**3. Cholesky分解**

```python
# A = L @ L.T （要求A是对称正定矩阵）
A = np.array([[4, 2], [2, 3]], dtype=float)  # 对称正定

L = np.linalg.cholesky(A)
print("L (下三角):\n", L)
print("验证 L@L.T ≈ A:", np.allclose(L @ L.T, A))

# 应用：比通用LU分解快2倍，用于求解正定方程组
# 正定矩阵判断：所有特征值 > 0
eigenvalues = np.linalg.eigvalsh(A)
print("特征值:", eigenvalues)  # 全正 → 正定
```

**4. PCA中的特征分解（预览）**

```python
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
```

---

### Day 4 - 特征值与特征向量（2h）

**目标**：理解特征值/特征向量的几何意义和计算方法。

**1. 定义与几何意义**

```python
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
```

**2. 对称矩阵的特征值**

```python
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
```

**3. 正定/半正定矩阵判断**

```python
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
```

---

### Day 5 - SVD奇异值分解（2h）

**目标**：理解SVD的原理和应用。

**1. SVD原理**

```python
# SVD：任何矩阵都可以分解为 A = U Σ V.T
# U: 左奇异向量（m×m正交矩阵）
# Σ: 奇异值对角矩阵（m×n，对角线上是σ₁≥σ₂≥...≥σ_r>0）
# V.T: 右奇异向量的转置（n×n正交矩阵）

A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)  # 3×2矩阵

U, S, Vt = np.linalg.svd(A, full_matrices=False)
# full_matrices=False: U是3×2, Σ是2×2, Vt是2×2

print("U shape:", U.shape)   # (3, 2)
print("S:", S)               # 奇异值（降序排列）
print("Vt shape:", Vt.shape) # (2, 2)

# 验证 A = U @ diag(S) @ Vt
Sigma = np.diag(S)
reconstructed = U @ Sigma @ Vt
assert np.allclose(reconstructed, A)
print("SVD验证通过: A = U Σ V.T")
```

**2. SVD与特征分解的关系**

```python
# A = U Σ V.T
# A.T @ A = V Σ² V.T  → A.T @ A 的特征向量就是V，特征值是σ²
# A @ A.T = U Σ² U.T  → A @ A.T 的特征向量就是U

A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)

# 验证
eigenvalues_ATA, V = np.linalg.eigh(A.T @ A)
# 特征值按升序排列，取降序
eigenvalues_ATA = eigenvalues_ATA[::-1]
V = V[:, ::-1]

U, S, Vt = np.linalg.svd(A, full_matrices=False)

print("A.T@A 的特征值:", eigenvalues_ATA)
print("奇异值的平方 S²:", S**2)
print("近似相等:", np.allclose(eigenvalues_ATA, S**2))
```

**3. 低秩近似（截断SVD）**

```python
from scipy.linalg import svd

# 保留前k个奇异值 → 低秩近似
# 信息保留比例 = 保留的奇异值平方和 / 全部奇异值平方和

def truncated_svd(A, k):
    """截断SVD，保留前k个奇异值"""
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    # 只保留前k个
    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]

    # 信息保留比
    info_retained = np.sum(S_k**2) / np.sum(S**2)
    print(f"保留{k}个奇异值，信息保留: {info_retained:.2%}")

    # 重建
    A_k = U_k @ np.diag(S_k) @ Vt_k
    # 重建误差
    error = np.linalg.norm(A - A_k, 'fro') / np.linalg.norm(A, 'fro')
    print(f"重建误差: {error:.2%}")

    return A_k

A = np.random.randn(50, 20)
for k in [1, 5, 10, 15, 20]:
    print(f"\n--- k={k} ---")
    truncated_svd(A, k)
```

**4. 应用：图像压缩**

```python
from PIL import Image
import matplotlib.pyplot as plt

# 加载灰度图
img = Image.open('sample.jpg').convert('L')
img_array = np.array(img, dtype=float)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes[0, 0].imshow(img_array, cmap='gray')
axes[0, 0].set_title(f'Original ({img_array.shape})')

U, S, Vt = np.linalg.svd(img_array, full_matrices=False)

for i, k in enumerate([5, 10, 20, 50, 100]):
    # 重建
    compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    compressed = np.clip(compressed, 0, 255)

    ax = axes[(i+1)//3, (i+1)%3]
    ax.imshow(compressed, cmap='gray')
    info = np.sum(S[:k]**2) / np.sum(S**2)
    ax.set_title(f'k={k}, info={info:.1%}')

plt.tight_layout()
plt.show()
```

**5. 应用：推荐系统（矩阵分解）**

```python
# 用户-物品评分矩阵（稀疏，有很多缺失值）
# SVD可以将评分矩阵分解为用户特征矩阵和物品特征矩阵
# 用于预测缺失的评分 → 推荐未看过的电影

# 简化示例
R = np.array([
    [5, 3, 0, 1],  # 用户1对4部电影的评分（0=未看）
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [0, 0, 0, 4],
    [0, 1, 5, 4],
], dtype=float)

# 截断SVD → 潜在因子模型
k = 2
U, S, Vt = np.linalg.svd(R, full_matrices=False)
R_approx = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
print("预测评分矩阵:\n", np.round(R_approx, 1))
# 值接近0的位置表示预测评分 → 可以推荐高分项
```

---

### Day 6 - PCA主成分分析（3-4h）

**目标**：理解PCA的完整数学原理并动手实现。

**1. PCA的数学原理**

```python
# PCA目标：找到数据方差最大的方向（主成分）
# 步骤：
# 1. 数据中心化：X_centered = X - mean(X)
# 2. 计算协方差矩阵：C = (1/n) X.T @ X
# 3. 对C做特征分解（或对X做SVD）
# 4. 取前k个最大特征值对应的特征向量作为主成分
# 5. 投影：X_pca = X_centered @ W_k

import numpy as np
import matplotlib.pyplot as plt

def pca_manual(X, n_components=2):
    """手动实现PCA"""
    n_samples, n_features = X.shape

    # 1. 中心化
    mean = X.mean(axis=0)
    X_centered = X - mean

    # 2. 协方差矩阵
    cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)

    # 3. 特征分解
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # 4. 按特征值从大到小排序
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 5. 选择前k个主成分
    components = eigenvectors[:, :n_components]

    # 6. 投影
    X_pca = X_centered @ components

    # 7. 方差解释比
    explained_variance_ratio = eigenvalues / eigenvalues.sum()

    return X_pca, components, eigenvalues, explained_variance_ratio, mean

# 生成测试数据
np.random.seed(42)
mean = [0, 0]
cov = [[3, 2], [2, 2]]  # 有相关性的数据
X = np.random.multivariate_normal(mean, cov, 200)

# 应用PCA
X_pca, components, eigenvalues, explained_ratio, mean = pca_manual(X, n_components=2)

print("特征值:", eigenvalues)
print("方差解释比:", explained_ratio)
print("累积方差:", np.cumsum(explained_ratio))

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 原始数据 + 主成分方向
axes[0].scatter(X[:, 0], X[:, 1], alpha=0.5)
for i, (ev, vec) in enumerate(zip(eigenvalues, components.T)):
    axes[0].arrow(0, 0, vec[0]*ev**0.5*2, vec[1]*ev**0.5*2,
                  head_width=0.1, color=['red', 'blue'][i], linewidth=2)
axes[0].set_title('Original Data + Principal Components')
axes[0].set_aspect('equal')
axes[0].grid(True)

# PCA投影后
axes[1].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
axes[1].set_title('PCA Transformed Data')
axes[1].set_xlabel(f'PC1 ({explained_ratio[0]:.1%} variance)')
axes[1].set_ylabel(f'PC2 ({explained_ratio[1]:.1%} variance)')
axes[1].grid(True)

plt.tight_layout()
plt.show()
```

**2. 选择主成分数量**

```python
# 方法1：累积方差解释比 > 90% 或 95%
cumsum = np.cumsum(explained_ratio)
n_90 = np.argmax(cumsum >= 0.90) + 1
n_95 = np.argmax(cumsum >= 0.95) + 1
print(f"90%方差需要 {n_90} 个主成分")
print(f"95%方差需要 {n_95} 个主成分")

# 方法2：碎石图（Scree Plot）
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_ratio)+1), explained_ratio, 'bo-')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.axhline(y=0.05, color='r', linestyle='--', label='5% threshold')
plt.legend()
plt.grid(True)
plt.show()

# 方法3：累积方差图
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumsum)+1), cumsum, 'ro-')
plt.axhline(y=0.95, color='g', linestyle='--', label='95% threshold')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance')
plt.legend()
plt.grid(True)
plt.show()
```

**3. 用PCA对Iris数据集降维可视化**

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# 加载数据
iris = load_iris()
X = iris.data    # 4个特征
y = iris.target  # 3类花
target_names = iris.target_names

# PCA前必须标准化！
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 用sklearn的PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"原始维度: {X.shape}")
print(f"降维后: {X_pca.shape}")
print(f"方差解释比: {pca.explained_variance_ratio_}")
print(f"累积方差: {pca.explained_variance_ratio_.sum():.2%}")

# 可视化（标注3类花）
plt.figure(figsize=(10, 8))
colors = ['red', 'blue', 'green']
for color, target_name in zip(colors, target_names):
    plt.scatter(X_pca[y == list(target_names).index(target_name), 0],
                X_pca[y == list(target_names).index(target_name), 1],
                c=color, label=target_name, alpha=0.7, edgecolors='k')

# 修正：用数字索引
plt.figure(figsize=(10, 8))
for i, (color, target_name) in enumerate(zip(colors, target_names)):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1],
                c=color, label=target_name, alpha=0.7, edgecolors='k')

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.title('PCA: Iris Dataset (4D → 2D)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

**4. PCA vs LDA**

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# LDA：有监督降维（利用类别信息）
# PCA：无监督降维（只用特征，不用标签）

lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# PCA
for i, (color, name) in enumerate(zip(colors, target_names)):
    axes[0].scatter(X_pca[y == i, 0], X_pca[y == i, 1],
                    c=color, label=name, alpha=0.7)
axes[0].set_title('PCA (Unsupervised)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# LDA
for i, (color, name) in enumerate(zip(colors, target_names)):
    axes[1].scatter(X_lda[y == i, 0], X_lda[y == i, 1],
                    c=color, label=name, alpha=0.7)
axes[1].set_title('LDA (Supervised)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

### Day 7 - 综合复习（3-4h）

**目标**：建立线性代数与AI的完整映射关系。

**1. 线性代数在AI中的映射**

```
线性代数概念          →  AI中的应用
─────────────────────────────────────────
向量                 →  数据样本（特征向量）、权重向量
矩阵乘法             →  神经网络前向传播 y = Wx + b
矩阵分解             →  推荐系统、模型压缩
特征值/特征向量       →  PCA降维、模型稳定性分析
SVD                  →  模型压缩、低秩近似、推荐系统
向量范数             →  正则化（L1/L2）、梯度裁剪
余弦相似度           →  文本相似度、推荐系统
正交矩阵             →  特征独立性、数值稳定性
点积                 →  注意力机制（Attention）、相似度计算
```

**2. 手动计算示例**

```python
# 手动计算一个2层网络的梯度（验证反向传播）
import numpy as np

# 前向传播
X = np.array([[0.5, 0.3]])   # 输入 (1, 2)
W1 = np.array([[0.1, 0.2],   # 第一层权重 (2, 2)
               [0.3, 0.4]])
b1 = np.array([[0.1, 0.1]])  # 第一层偏置
W2 = np.array([[0.5], [0.6]])# 第二层权重 (2, 1)
b2 = np.array([[0.2]])       # 第二层偏置
y_true = np.array([[1.0]])   # 真实标签

# Layer 1
z1 = X @ W1 + b1             # (1, 2)
a1 = 1 / (1 + np.exp(-z1))   # sigmoid激活

# Layer 2
z2 = a1 @ W2 + b2            # (1, 1)
a2 = 1 / (1 + np.exp(-z2))   # 输出

# 损失（MSE）
loss = 0.5 * (y_true - a2)**2

# 反向传播（手动计算梯度）
dL_da2 = -(y_true - a2)                    # ∂L/∂a2
da2_dz2 = a2 * (1 - a2)                    # sigmoid导数
dL_dz2 = dL_da2 * da2_dz2                  # ∂L/∂z2

dL_dW2 = a1.T @ dL_dz2                     # ∂L/∂W2
dL_db2 = dL_dz2                             # ∂L/∂b2

dL_da1 = dL_dz2 @ W2.T                     # ∂L/∂a1
da1_dz1 = a1 * (1 - a1)                    # sigmoid导数
dL_dz1 = dL_da1 * da1_dz1                  # ∂L/∂z1

dL_dW1 = X.T @ dL_dz1                      # ∂L/∂W1
dL_db1 = dL_dz1                             # ∂L/∂b1

print("Loss:", loss)
print("dL/dW1:\n", dL_dW1)
print("dL/dW2:\n", dL_dW2)
```

**3. 思维导图框架**

```
线性代数（AI核心）
├── 向量
│   ├── 运算：加减、点积、叉积
│   ├── 范数：L1、L2、L-inf
│   ├── 余弦相似度 → 文本/推荐
│   └── 正交 → 特征独立性
├── 矩阵
│   ├── 乘法 → 线性变换 → Wx+b
│   ├── 逆矩阵 → 解方程组
│   ├── 行列式 → 面积缩放
│   └── 秩 → 线性相关性
├── 矩阵分解
│   ├── LU → 解方程
│   ├── QR → 最小二乘
│   ├── Cholesky → 正定方程
│   ├── 特征分解 → PCA
│   └── SVD → 压缩/推荐/降维
├── 特征值/特征向量
│   ├── 几何意义：方向不变的向量
│   ├── 对称矩阵：实特征值、正交特征向量
│   └── 正定判断：特征值全正
└── PCA
    ├── 协方差矩阵特征分解
    ├── 方差解释比
    ├── 选择主成分数量
    └── 数据标准化（必须！）
```

---

## 四、代码练习

### 练习1：余弦相似度函数

```python
# cosine_similarity.py
import numpy as np
from collections import Counter
import math

def tfidf_vector(doc, all_docs, vocab):
    """计算TF-IDF向量"""
    tf = Counter(doc.split())
    doc_len = len(doc.split())
    n_docs = len(all_docs)

    vector = np.zeros(len(vocab))
    for i, word in enumerate(vocab):
        # TF: 词频
        tf_val = tf.get(word, 0) / doc_len if doc_len > 0 else 0
        # IDF: 逆文档频率
        df = sum(1 for d in all_docs if word in d.split())
        idf_val = math.log(n_docs / (1 + df)) if df > 0 else 0
        vector[i] = tf_val * idf_val
    return vector

def cosine_similarity(a, b):
    """余弦相似度"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 测试
docs = [
    "AI machine learning deep learning",
    "machine learning data science",
    "web development javascript python"
]
vocab = sorted(set(word for doc in docs for word in doc.split()))

vectors = [tfidf_vector(doc, docs, vocab) for doc in docs]
print(f"文档1和文档2的相似度: {cosine_similarity(vectors[0], vectors[1]):.4f}")
print(f"文档1和文档3的相似度: {cosine_similarity(vectors[0], vectors[2]):.4f}")
print(f"文档2和文档3的相似度: {cosine_similarity(vectors[1], vectors[2]):.4f}")
```

### 练习2：SVD图像压缩

```python
# svd_image_compression.py
import numpy as np
import matplotlib.pyplot as plt

# 创建一个渐变图像（如果没有真实图片）
img = np.zeros((100, 100))
for i in range(100):
    for j in range(100):
        img[i, j] = (i + j) / 200

U, S, Vt = np.linalg.svd(img, full_matrices=False)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('Original')

ks = [5, 10, 20, 30, 50]
for i, k in enumerate(ks):
    compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    ax = axes[(i+1)//3, (i+1)%3]
    ax.imshow(compressed, cmap='gray')
    info = np.sum(S[:k]**2) / np.sum(S**2)
    ax.set_title(f'k={k}, info={info:.1%}')

plt.tight_layout()
plt.savefig('svd_compression.png', dpi=150)
plt.show()
```

### 练习3：PCA对Iris降维

```python
# pca_iris.py
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

iris = load_iris()
X, y = iris.data, iris.target

# 标准化
X_scaled = StandardScaler().fit_transform(X)

# PCA降维到2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 可视化
plt.figure(figsize=(10, 8))
for i, name in enumerate(iris.target_names):
    plt.scatter(X_pca[y==i, 0], X_pca[y==i, 1], label=name, alpha=0.7, edgecolors='k')

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.title('PCA: Iris 4D → 2D')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('pca_iris.png', dpi=150)
plt.show()

print(f"方差解释比: {pca.explained_variance_ratio_}")
print(f"累积方差: {pca.explained_variance_ratio_.sum():.2%}")
```

---

## 五、本周产出

| 产出物 | 说明 |
|--------|------|
| 向量运算笔记 | 点积/范数/余弦相似度的数学与代码 |
| 矩阵运算可视化 | 线性变换的几何直观 |
| SVD图像压缩 | 不同k值的压缩效果对比 |
| PCA Iris降维 | 4D→2D可视化，标注3类花 |
| 线性代数思维导图 | 与AI概念的完整映射 |
| cosine_similarity.py | 基于TF-IDF的文本相似度 |

---

## 六、自测题

1. **点积的几何意义是什么？**

   <details>
   <summary>参考答案</summary>

   点积 a.b = |a||b|cos(theta)。几何意义是一个向量在另一个向量方向上的投影与另一个向量长度的乘积。点积为0表示两向量正交（垂直）。在AI中，点积用于衡量两个向量的"相似程度"，是注意力机制（Attention）的核心计算。
   </details>

2. **矩阵不可逆的条件是什么？条件数大意味着什么？**

   <details>
   <summary>参考答案</summary>

   不可逆条件：行列式为0（奇异矩阵）、不是方阵、行或列线性相关（秩不满）。
   条件数大意味着矩阵接近奇异，数值计算会很不稳定（小扰动导致大变化）。在AI中，条件数大的矩阵会导致梯度不稳定、训练困难。
   </details>

3. **特征值和特征向量的几何意义？**

   <details>
   <summary>参考答案</summary>

   特征向量v是矩阵A变换后方向不变的向量（A v = lambda v），特征值lambda是该方向上的缩放倍数。|lambda|>1表示拉伸，<1表示压缩，=1不变，<0反转方��。在PCA中，最大特征值对应的方向就是数据方差最大的方向（主成分）。
   </details>

4. **SVD和特征分解的区别与联系？**

   <details>
   <summary>参考答案</summary>

   区别：特征分解只能用于方阵，SVD可以用于任何矩阵。SVD: A = U Sigma V.T，特征分解: A = Q Lambda Q.T。
   联系：A.T A 的特征向量就是V，特征值是奇异值的平方；A A.T 的特征向量就是U。SVD本质上是将特征分解推广到非方阵。
   </details>

5. **PCA为什么能降维？选择主成分数量的标准？**

   <details>
   <summary>参考答案</summary>

   PCA能降维因为：数据往往存在相关性，高维数据实际信息集中在少数几个方向上。PCA找到方差最大的方向（主成分），丢弃方差小的方向（噪声），从而在尽量少丢失信息的前提下降低维度。
   选择标准：累积方差解释比 > 90%-95%；或看碎石图（Scree Plot）在"肘部"截断。
   </details>

---

## 七、Java开发者提示

| Java/工程概念 | 线性代数对应 | 说明 |
|--------------|-------------|------|
| 二维数组 `double[][]` | 矩阵 | 但NumPy矩阵支持向量化运算，远比Java数组高效 |
| 坐标变换 | 线性变换 | 旋转、缩放、剪切本质上就是矩阵乘法 |
| JSON序列化/反序列化 | 矩阵分解与重建 | A = U Sigma V.T，类似将数据"序列化"为因子 |
| HashMap的key | 特征向量 | 特征向量是矩阵的"固有属性"，不随变换改变方向 |
| 数据库索引 | 秩 | 秩决定了矩阵的"有效信息量" |
| 图片压缩 | SVD低秩近似 | 保留主要奇异值就能近似重建 |
| 缓存命中率 | 方差解释比 | 保留多少主成分 = 命中多少有效信息 |

**核心洞察**：神经网络的本质就是一系列矩阵乘法 + 非线性激活函数。理解矩阵运算就是理解AI模型的基础。权重矩阵W就是一个线性变换，将输入空间映射到输出空间。
