##################ndarray创建
import numpy as np
a = np.array([1, 2, 3, 4, 5])            # 一维数组
b = np.array([[1, 2], [3, 4]])            # 二维数组（矩阵）

# 特殊数组
zeros = np.zeros((3, 4))                   # 3x4全零矩阵
ones = np.ones((2, 3))                     # 2x3全一矩阵
eye = np.eye(3)                            # 3x3单位矩阵
full = np.full((2, 3), 7)                  # 2x3全填充7
empty = np.empty((2, 2))                   # 未初始化（值随机）

# 序列生成
range_arr = np.arange(0, 10, 2)            # [0, 2, 4, 6, 8]，类似Python range
linspace = np.linspace(0, 1, 5)            # [0, 0.25, 0.5, 0.75, 1.0]，等间距
logspace = np.logspace(0, 2, 5)            # [  1. 3.16227766  10. 31.6227766  100.]，对数间距

# 随机数
np.random.seed(42)                         # 设置随机种子（确保可复现）
rand = np.random.rand(3, 4)                # [0,1)均匀分布
randn = np.random.randn(3, 4)              # 标准正态分布
randint = np.random.randint(0, 10, (3, 4)) # [0,10)随机整数


##################数据类型与属性
a = np.array([1, 2, 3], dtype=np.float64)
print(a)
# 重要属性
a.ndim        # 维度数（1表示一维，2表示二维）
a.shape       # 各维度大小 (3,)
a.size        # 元素总数 3
a.dtype       # 数据类型 float64
a.itemsize    # 每个元素字节数 8
a.nbytes      # 总字节数 = size * itemsize

# 类型转换
a.astype(np.int32)        # float → int，截断小数
a.astype(np.float32)      # float64 → float32（AI中常用，节省显存）

######################索引与切片
arr = np.arange(12).reshape(3, 4)

# 基础索引
arr[0]          # [0, 1, 2, 3]     第一行
arr[0, 1]       # 1                第一行第二列
arr[-1]         # [8, 9, 10, 11]   最后一行

# 切片
arr[0:2, 1:3]   # [[1,2],[5,6]]    前两行的第2-3列
arr[:, 0]       # [0, 4, 8]        所有行的第一列
arr[1, :]       # [4, 5, 6, 7]     第二行的所有列

# 布尔索引（非常强大！Java中没有的）
arr[arr > 5]    # [6, 7, 8, 9, 10, 11]  所有大于5的元素
arr[arr % 2 == 0] = -1  # 将偶数替换为-1

# 花式索引（Fancy Indexing）
arr[[0, 2]]     # 第0行和第2行
arr[[0, 1], [1, 3]]  # (0,1)和(1,3)位置的元素 → [1, 7]
# 注意：切片是视图（修改会影响原数组），花式索引是副本

##################形状操作
a = np.arange(12)

# reshape：改变形状（不改变数据）
a.reshape(3, 4)     # 3行4列
a.reshape(2, 6)     # 2行6列
a.reshape(3, -1)    # -1表示自动计算：3行4列

# flatten vs ravel
a.reshape(3,4).flatten()   # 返回副本（1D）
a.reshape(3,4).ravel()     # 返回视图（1D，修改会影响原数组）

# 转置
a.reshape(3,4).T           # 4行3列
a.reshape(3,4).transpose() # 同上

# 拼接
a_34 = a.reshape(3, 4)
b_34 = a_34 * 10  # 用于演示的第二个数组

# vstack：垂直拼接（按行堆叠），行数增加
# (3,4) + (3,4) → (6,4)
np.vstack([a_34, b_34])

# hstack：水平拼接（按列堆叠），列数增加
# (3,4) + (3,4) → (3,8)
np.hstack([a_34, b_34])

# concatenate：按指定轴拼接
# axis=0 同 vstack（沿行方向），axis=1 同 hstack（沿列方向）
np.concatenate([a_34, b_34], axis=0)  # → (6,4)
np.concatenate([a_34, b_34], axis=1)  # → (3,8)

# stack：沿新维度堆叠，会增加一个维度
# (3,4) + (3,4) → (2,3,4)
np.stack([a_34, b_34], axis=0)