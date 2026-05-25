#### 理解向量的各种运算及其几何意义。
#向量表示与基本运算
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

###点积（Dot Product）—— 最重要
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

####叉积
# 叉积只在3D中有定义
a = np.array([1, 0, 0])
b = np.array([0, 1, 0])
np.cross(a, b)  # [0, 0, 1]  垂直于a和b的向量

# 几何意义：结果向量的长度 = 两个向量构成的平行四边形面积

### 向量范数
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

### 余弦相似度
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

### 向量正交
# 两个向量正交：点积为0
a = np.array([1, 0, 0])
b = np.array([0, 1, 0])
np.dot(a, b)  # 0 → 正交

# 正交向量组：任意两个向量都正交
# 标准正交基：正交 + 每个向量长度为1
# AI中的意义：特征之间正交 = 没有冗余信息 = 理想特征

