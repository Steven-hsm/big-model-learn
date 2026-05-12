# W02 - NumPy与Pandas数据处理

> 第2周 | Java开发工程师转AI开发学习计划
> 工作日每晚2小时 | 周末6-8小时

---

## 一、本周目标

1. 掌握NumPy的核心数据结构ndarray及其向量化运算
2. 掌握Pandas的Series和DataFrame，能完成常见数据清洗和处理
3. 学会用Matplotlib和Seaborn做数据可视化
4. 完成Titanic数据集的EDA（探索性数据分析）
5. 建立数据处理到可视化的完整工作流

---

## 二、时间安排

| 日期 | 类型 | 时长 | 主题 |
|------|------|------|------|
| Day 1 | 工作日晚 | 2h | NumPy基础 |
| Day 2 | 工作日晚 | 2h | NumPy进阶 |
| Day 3 | 工作日晚 | 2h | Pandas基础 |
| Day 4 | 工作日晚 | 2h | Pandas数据处理 |
| Day 5 | 工作日晚 | 2h | Pandas进阶 |
| Day 6 | 周末 | 3-4h | Matplotlib可视化 |
| Day 7 | 周末 | 3-4h | Seaborn进阶与Titanic EDA |

---

## 三、详细学习内容

### Day 1 - NumPy基础（2h）

**目标**：掌握ndarray的创建、索引和基本操作。

**1. ndarray创建**

```python
import numpy as np

# 从列表创建
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
logspace = np.logspace(0, 2, 5)            # [1, 10, 100, ..., 100]，对数间距

# 随机数
np.random.seed(42)                         # 设置随机种子（确保可复现）
rand = np.random.rand(3, 4)                # [0,1)均匀分布
randn = np.random.randn(3, 4)              # 标准正态分布
randint = np.random.randint(0, 10, (3, 4)) # [0,10)随机整数
```

**2. 数据类型与属性**

```python
a = np.array([1, 2, 3], dtype=np.float64)

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
```

**3. 索引与切片**

```python
arr = np.arange(12).reshape(3, 4)
# [[ 0,  1,  2,  3],
#  [ 4,  5,  6,  7],
#  [ 8,  9, 10, 11]]

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
```

**4. 形状操作**

```python
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
np.vstack([a.reshape(3,4), a.reshape(3,4)])  # 垂直拼接 → (6,4)
np.hstack([a.reshape(3,4), a.reshape(3,4)])  # 水平拼接 → (3,8)
np.concatenate([a.reshape(3,4)]*2, axis=0)   # 按轴拼接
```

---

### Day 2 - NumPy进阶（2h）

**目标**：掌握广播机制、矩阵运算和线性代数操作。

**1. 广播机制（Broadcasting）**

```python
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
```

**2. 矩阵运算**

```python
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
```

**3. 线性代数**

```python
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
```

**4. 统计函数**

```python
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
```

---

### Day 3 - Pandas基础（2h）

**目标**：掌握Series和DataFrame的创建、查看和基本选择操作。

**1. Series（一维数据结构）**

```python
import pandas as pd

# 创建Series
s = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])
# a    10
# b    20
# c    30
# d    40

# 从字典创建
s2 = pd.Series({'Alice': 85, 'Bob': 92, 'Charlie': 78})

# 基本操作
s.values        # array([10, 20, 30, 40])  获取值数组
s.index         # Index(['a','b','c','d'])  获取索引
s.dtype         # dtype('int64')
s.shape         # (4,)

# 索引和切片
s['a']          # 10
s[0]            # 10（位置索引）
s[['a', 'c']]   # a=10, c=30
s[1:3]          # b=20, c=30

# 运算（自动对齐索引）
s * 2           # 所有值乘以2
s + pd.Series([1, 2], index=['a', 'b'])  # 不匹配的索引为NaN
```

**2. DataFrame创建**

```python
# 从字典创建（最常用）
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'age': [25, 30, 35, 28],
    'city': ['Beijing', 'Shanghai', 'Guangzhou', 'Shenzhen'],
    'salary': [15000, 20000, 18000, 16000]
})

# 从CSV文件读取（最常见的数据来源）
df = pd.read_csv('data.csv')
df = pd.read_csv('data.csv', encoding='utf-8', index_col=0, nrows=1000)

# 从Excel读取（需要openpyxl库）
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# 从JSON读取
df = pd.read_json('data.json')

# 从SQL读取
import sqlite3
conn = sqlite3.connect('database.db')
df = pd.read_sql('SELECT * FROM users', conn)
```

**3. 数据预览**

```python
df = pd.read_csv('titanic.csv')

df.head(5)       # 前5行
df.tail(3)       # 后3行
df.info()        # 数据概览：列名、类型、非空数量、内存占用
df.describe()    # 数值列的统计摘要（count/mean/std/min/25%/50%/75%/max）
df.dtypes        # 每列的数据类型
df.shape         # (行数, 列数)
df.columns       # 列名列表
df.index         # 行索引
df.isnull().sum()# 每列缺失值数量
df.nunique()     # 每列唯一值数量
```

**4. 数据选择**

```python
# 选择列
df['name']           # 选一列，返回Series
df[['name', 'age']]  # 选多列，返回DataFrame

# loc：基于标签的索引（左闭右闭）
df.loc[0]                    # 第一行（标签为0）
df.loc[0:2]                  # 前3行（0,1,2都包含！）
df.loc[0:2, ['name','age']]  # 前3行的name和age列
df.loc[df['age'] > 30]       # 条件选择（布尔索引）

# iloc：基于位置的索引（左闭右开）
df.iloc[0]           # 第一行
df.iloc[0:3]         # 前3行（0,1,2）
df.iloc[0:3, 0:2]   # 前3行的前2列

# 条件过滤
df[df['age'] > 30]                          # 单条件
df[(df['age'] > 30) & (df['city'] == 'Beijing')]  # 多条件（注意括号！）
df[df['city'].isin(['Beijing', 'Shanghai'])]       # isin过滤
df.query('age > 30 and city == "Beijing"')          # query语法（更可读）
```

---

### Day 4 - Pandas数据处理（2h）

**目标**：掌握数据清洗、过滤、分组和合并操作。

**1. 缺失值处理**

```python
df = pd.DataFrame({
    'A': [1, np.nan, 3, np.nan, 5],
    'B': [10, 20, np.nan, 40, 50],
    'C': ['x', 'y', 'z', np.nan, 'w']
})

# 检测缺失值
df.isnull()          # 返回布尔DataFrame
df.isnull().sum()    # 每列缺失值数量
df.isnull().any()    # 哪些列有缺失值

# 删除缺失值
df.dropna()                  # 删除任何含缺失值的行
df.dropna(subset=['A'])      # 只看A列
df.dropna(how='all')         # 只删除全为NaN的行
df.dropna(axis=1)            # 删除含缺失值的列

# 填充缺失值
df.fillna(0)                         # 用0填充
df.fillna(df.mean(numeric_only=True))# 用均值填充（数值列）
df.fillna(method='ffill')            # 前向填充
df.fillna(method='bfill')            # 后向填充
df['C'].fillna('unknown')            # 指定列填充

# 插值
df.interpolate(method='linear')      # 线性插值
```

**2. 数据过滤与排序**

```python
# 排序
df.sort_values('age', ascending=False)           # 按列排序
df.sort_values(['city', 'age'], ascending=[True, False])  # 多列排序
df.sort_index()                                   # 按索引排序

# 去重
df.drop_duplicates()                  # 删除完全重复的行
df.drop_duplicates(subset=['name'])   # 按列去重
df.duplicated()                       # 返回布尔Series标记重复行

# 值替换
df['city'].replace('Beijing', 'BJ')   # 替换值
df.replace({np.nan: 0})               # 批量替换
```

**3. 分组聚合（groupby）**

```python
# 基本分组
df.groupby('city')['salary'].mean()   # 每个城市的平均工资

# 多种聚合
df.groupby('city')['salary'].agg(['mean', 'median', 'std', 'count'])

# 自定义聚合函数
df.groupby('city')['salary'].agg(
    avg_salary='mean',
    max_salary='max',
    range_salary=lambda x: x.max() - x.min()
)

# 多列分组
df.groupby(['city', 'age'])['salary'].mean()

# agg vs transform vs apply
# agg：聚合后每组一行
df.groupby('city')['salary'].agg('mean')

# transform：聚合后保持原形状（广播回每行）
df['city_avg'] = df.groupby('city')['salary'].transform('mean')
df['salary_diff'] = df['salary'] - df['city_avg']

# apply：对每组应用任意函数
df.groupby('city').apply(lambda g: g.nlargest(2, 'salary'))
```

**4. 合并操作**

```python
# merge：类似SQL的JOIN
pd.merge(df1, df2, on='id')                        # 内连接（默认）
pd.merge(df1, df2, on='id', how='left')             # 左连接
pd.merge(df1, df2, on='id', how='right')            # 右连接
pd.merge(df1, df2, on='id', how='outer')            # 外连接（全连接）
pd.merge(df1, df2, left_on='id1', right_on='id2')   # 列名不同时

# concat：拼接
pd.concat([df1, df2], axis=0)    # 纵向拼接（增加行）
pd.concat([df1, df2], axis=1)    # 横向拼接（增加列）

# join：基于索引连接
df1.join(df2, how='left')
```

---

### Day 5 - Pandas进阶（2h）

**目标**：掌握行/列级操作、透视表、时间序列处理。

**1. apply / map / applymap**

```python
# map：Series的元素级操作
df['city_code'] = df['city'].map({'Beijing': 1, 'Shanghai': 2})

# apply：Series或DataFrame的行/列级操作
df['salary_k'] = df['salary'].apply(lambda x: x / 1000)  # Series.apply
df[['age', 'salary']].apply(np.sqrt)                       # 对每列应用
df.apply(lambda row: row['age'] * 1000, axis=1)           # 对每行应用

# applymap：DataFrame的元素级操作
df[['age', 'salary']].applymap(lambda x: f"{x:.2f}")

# pipe：链式操作（类似Java Stream）
result = (df
    .pipe(lambda df: df[df['age'] > 25])
    .pipe(lambda df: df.assign(salary_k=df['salary'] / 1000))
    .groupby('city')['salary_k'].mean()
)
```

**2. 透视表与交叉表**

```python
# pivot_table：类似Excel的数据透视表
df.pivot_table(values='salary', index='city', columns='gender', aggfunc='mean')

# 多个聚合函数
df.pivot_table(values='salary', index='city', aggfunc=['mean', 'sum', 'count'])

# margins：添加行列合计
df.pivot_table(values='salary', index='city', aggfunc='mean', margins=True)

# crosstab：交叉表（计算频数）
pd.crosstab(df['city'], df['gender'])
pd.crosstab(df['city'], df['gender'], normalize='index')  # 行百分比
```

**3. 时间序列**

```python
# 创建时间序列
dates = pd.date_range('2024-01-01', periods=100, freq='D')
ts = pd.Series(np.random.randn(100), index=dates)

# 转换字符串为日期
df['date'] = pd.to_datetime(df['date_str'])

# 提取时间特征
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['weekday'] = df['date'].dt.dayofweek
df['is_weekend'] = df['date'].dt.dayofweek >= 5

# 重采样（resample）
ts.resample('W').mean()    # 按周平均
ts.resample('M').sum()     # 按月求和

# 滚动窗口（rolling）
ts.rolling(window=7).mean()   # 7日移动平均
ts.rolling(window=30).std()   # 30日滚动标准差

# 时间差
df['days_since'] = (pd.Timestamp.now() - df['date']).dt.days
```

**4. 字符串操作（.str访问器）**

```python
s = pd.Series(['Hello World', 'Python AI', 'data science'])

s.str.lower()          # 全部小写
s.str.upper()          # 全部大写
s.str.strip()          # 去除首尾空格
s.str.split(' ')       # 按空格分割
s.str.contains('AI')   # 是否包含
s.str.replace(' ', '_')# 替换
s.str.len()            # 字符串长度
s.str[0:5]             # 切片
```

---

### Day 6 - Matplotlib可视化（3-4h）

**目标**：掌握Matplotlib绑定各种常见图表。

```python
import matplotlib.pyplot as plt

# 设置中文字体（Windows）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============ 折线图 ============
x = np.linspace(0, 2*np.pi, 100)
plt.figure(figsize=(10, 6))
plt.plot(x, np.sin(x), label='sin(x)', color='blue', linewidth=2)
plt.plot(x, np.cos(x), label='cos(x)', color='red', linewidth=2, linestyle='--')
plt.title('三角函数')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('trig.png', dpi=150, bbox_inches='tight')
plt.show()

# ============ 柱状图 ============
categories = ['A', 'B', 'C', 'D']
values = [23, 45, 56, 78]
plt.figure(figsize=(8, 5))
plt.bar(categories, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
plt.title('各类别数量')
plt.ylabel('数量')
for i, v in enumerate(values):
    plt.text(i, v + 1, str(v), ha='center')
plt.show()

# ============ 散点图 ============
plt.figure(figsize=(8, 6))
plt.scatter(df['age'], df['salary'], c=df['score'], cmap='viridis', s=50, alpha=0.6)
plt.colorbar(label='Score')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.title('Age vs Salary (colored by Score)')
plt.show()

# ============ 直方图 ============
plt.figure(figsize=(8, 5))
plt.hist(df['salary'], bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Salary')
plt.ylabel('Frequency')
plt.title('Salary Distribution')
plt.axvline(df['salary'].mean(), color='red', linestyle='--', label='Mean')
plt.legend()
plt.show()

# ============ 饼图 ============
plt.figure(figsize=(8, 8))
plt.pie(df['city'].value_counts(), labels=df['city'].value_counts().index,
        autopct='%1.1f%%', startangle=90)
plt.title('City Distribution')
plt.show()

# ============ 子图 ============
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].plot(x, np.sin(x))
axes[0, 0].set_title('Sin')
axes[0, 1].plot(x, np.cos(x))
axes[0, 1].set_title('Cos')
axes[1, 0].hist(np.random.randn(1000), bins=30)
axes[1, 0].set_title('Histogram')
axes[1, 1].scatter(np.random.rand(50), np.random.rand(50))
axes[1, 1].set_title('Scatter')
plt.tight_layout()
plt.show()
```

---

### Day 7 - Seaborn进阶与Titanic EDA（3-4h）

**目标**：掌握Seaborn高级图表，完成Titanic数据集的完整EDA。

**1. Seaborn基础**

```python
import seaborn as sns

# 设置主题
sns.set_theme(style='whitegrid')

# 加载示例数据集
titanic = sns.load_dataset('titanic')
tips = sns.load_dataset('tips')

# 热力图（相关系数矩阵）
plt.figure(figsize=(10, 8))
numeric_titanic = titanic.select_dtypes(include=[np.number])
sns.heatmap(numeric_titanic.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# 箱线图
plt.figure(figsize=(10, 6))
sns.boxplot(data=titanic, x='pclass', y='age', hue='survived')
plt.title('Age Distribution by Class and Survival')
plt.show()

# 小提琴图
plt.figure(figsize=(10, 6))
sns.violinplot(data=titanic, x='pclass', y='age', hue='survived', split=True)
plt.title('Age Distribution (Violin Plot)')
plt.show()

# 分布图
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(titanic['age'].dropna(), kde=True, ax=axes[0])
axes[0].set_title('Age Distribution')
sns.kdeplot(data=titanic, x='age', hue='survived', fill=True, ax=axes[1])
axes[1].set_title('Age Distribution by Survival')
plt.show()

# pairplot（多变量关系图）
sns.pairplot(titanic[['survived', 'age', 'fare', 'pclass']].dropna(),
             hue='survived', diag_kind='kde')
plt.show()

# catplot（分类图）
sns.catplot(data=titanic, x='pclass', y='survived', hue='sex', kind='bar')
plt.show()

# relplot（关系图）
sns.relplot(data=tips, x='total_bill', y='tip', hue='time', size='size',
            sizes=(50, 200), alpha=0.7)
plt.show()
```

**2. Titanic EDA完整流程**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 加载数据
titanic = sns.load_dataset('titanic')

# 2. 数据概览
print(titanic.shape)         # (891, 15)
print(titanic.info())
print(titanic.describe())
print(titanic.isnull().sum())

# 3. 目标变量分布
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
titanic['survived'].value_counts().plot.pie(autopct='%1.1f%%', ax=axes[0])
axes[0].set_title('Survival Rate')
sns.countplot(data=titanic, x='survived', ax=axes[1])
axes[1].set_title('Survival Count')
plt.show()

# 4. 单变量分析
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sns.histplot(titanic['age'].dropna(), kde=True, ax=axes[0,0])
axes[0,0].set_title('Age Distribution')
sns.countplot(data=titanic, x='sex', ax=axes[0,1])
axes[0,1].set_title('Gender Distribution')
sns.countplot(data=titanic, x='pclass', ax=axes[1,0])
axes[1,0].set_title('Class Distribution')
sns.histplot(titanic['fare'], kde=True, ax=axes[1,1])
axes[1,1].set_title('Fare Distribution')
plt.tight_layout()
plt.show()

# 5. 生存率分析
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.barplot(data=titanic, x='sex', y='survived', ax=axes[0])
axes[0].set_title('Survival by Gender')
sns.barplot(data=titanic, x='pclass', y='survived', ax=axes[1])
axes[1].set_title('Survival by Class')
sns.barplot(data=titanic, x='embarked', y='survived', ax=axes[2])
axes[2].set_title('Survival by Embarked')
plt.show()

# 6. 相关性分析
plt.figure(figsize=(10, 8))
numeric_cols = titanic.select_dtypes(include=[np.number])
sns.heatmap(numeric_cols.corr(), annot=True, cmap='RdBu_r', center=0)
plt.title('Feature Correlation Heatmap')
plt.show()

# 7. 多变量交互分析
sns.catplot(data=titanic, x='pclass', y='survived', hue='sex', kind='point')
plt.title('Survival: Class x Gender')
plt.show()

# 8. 年龄与生存
plt.figure(figsize=(10, 6))
sns.kdeplot(data=titanic, x='age', hue='survived', fill=True, common_norm=False)
plt.title('Age Distribution by Survival')
plt.show()
```

---

## 四、代码练习

### Day 2 练习：矩阵乘法与线性方程组

```python
# matrix_operations.py
import numpy as np

# 1. 手动实现矩阵乘法（不用@运算符）
def matrix_multiply(A, B):
    """手动实现矩阵乘法"""
    assert A.shape[1] == B.shape[0], "矩阵维度不匹配"
    result = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                result[i, j] += A[i, k] * B[k, j]
    return result

# 验证
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
assert np.allclose(matrix_multiply(A, B), A @ B)

# 2. 解线性方程组 Ax = b
# 3x + 2y = 8
# x + 4y = 6
A = np.array([[3, 2], [1, 4]], dtype=float)
b = np.array([8, 6], dtype=float)

# 方法1：直接求解
x = np.linalg.solve(A, b)
print(f"x={x[0]:.2f}, y={x[1]:.2f}")

# 方法2：逆矩阵
x2 = np.linalg.inv(A) @ b
print(f"验证: {np.allclose(x, x2)}")

# 方法3：最小二乘法（过定方程组）
x3, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
```

### Day 4 练习：Titanic数据处理

```python
# titanic_data_processing.py
import pandas as pd
import seaborn as sns

titanic = sns.load_dataset('titanic')

# 任务1：缺失值处理
print("缺失值统计：")
print(titanic.isnull().sum())
# age用中位数填充，embarked用众数填充，deck删除（缺失太多）
titanic['age'].fillna(titanic['age'].median(), inplace=True)
titanic['embarked'].fillna(titanic['embarked'].mode()[0], inplace=True)
titanic.drop(columns=['deck'], inplace=True)

# 任务2：特征编码
titanic['sex_encoded'] = (titanic['sex'] == 'male').astype(int)
titanic = pd.get_dummies(titanic, columns=['embarked'], prefix='embarked')

# 任务3：统计分析
print("\n按舱位和性别的生存率：")
print(titanic.groupby(['pclass', 'sex'])['survived'].agg(['mean', 'count']))

print("\n年龄分布统计：")
print(titanic.groupby('survived')['age'].describe())
```

### Day 7 练习：Titanic完整EDA可视化报告

对Titanic数据集创建至少8种不同图表的完整EDA报告，包含：
1. 生存率饼图
2. 年龄分布直方图
3. 票价分布箱线图
4. 各舱位生存率柱状图
5. 性别与生存率
6. 相关系数热力图
7. 年龄与生存KDE图
8. 舱位×性别交互分析图

---

## 五、本周产出

| 产出物 | 说明 |
|--------|------|
| NumPy基础笔记 | ndarray创建/索引/运算/线性代数 |
| Pandas数据清洗脚本 | 缺失值处理/过滤/分组/合并完整示例 |
| Titanic数据处理脚本 | 完整的数据清洗和特征编码 |
| Titanic EDA可视化报告 | 至少8种图表的完整分析 |
| matrix_operations.py | 矩阵乘法和解方程组实现 |

---

## 六、自测题

1. **NumPy广播机制的3条规则是什么？**

   <details>
   <summary>参考答案</summary>

   1. 如果两个数组维度数不同，在较小数组的形状左侧补1（如(3,) → (1,3)）
   2. 如果某个维度大小为1，则沿该维度复制扩展到与另一个数组相同
   3. 其他维度必须大小相等，否则报错
   </details>

2. **loc和iloc的区别？**

   <details>
   <summary>参考答案</summary>

   - `loc`：基于标签（label）索引，切片是**左闭右闭**的（`df.loc[0:2]` 包含0,1,2）
   - `iloc`：基于位置（integer position）索引，切片是**左闭右开**的（`df.iloc[0:2]` 包含0,1）
   </details>

3. **groupby的agg/transform/apply三者的区别？**

   <details>
   <summary>参考答案</summary>

   - `agg`：聚合后每组一行，返回缩减后的结果
   - `transform`：聚合但保持原形状，将聚合值广播回每一行（如计算组内均值并添加为新列）
   - `apply`：最灵活，对每个分组应用任意函数，返回的结果形状取决于函数
   </details>

4. **merge的inner/outer/left/right分别是什么效果？**

   <details>
   <summary>参考答案</summary>

   - `inner`：只保留两个表都有的key（交集）
   - `outer`：保留所有key，缺失的用NaN填充（并集）
   - `left`：保留左表所有key，右表没有的用NaN填充
   - `right`：保留右表所有key，左表没有的用NaN填充
   </details>

5. **如何处理DataFrame中的缺失值？有哪些策略？**

   <details>
   <summary>参考答案</summary>

   - 检测：`isnull()`、`isna()`、`info()`
   - 删除：`dropna()`（删除含缺失值的行/列）
   - 填充固定值：`fillna(0)` 或 `fillna('unknown')`
   - 填充统计值：`fillna(df.mean())`（均值/中位数/众数）
   - 前向/后向填充：`fillna(method='ffill'/'bfill')`
   - 插值：`interpolate(method='linear')`
   - 策略选择：缺失<5%可删除，5-20%用填充，>20%考虑删除列或用模型预测
   </details>

---

## 七、Java开发者提示

| Java概念 | Python/Pandas对应 | 说明 |
|----------|-------------------|------|
| 多维数组 `int[][]` | NumPy `ndarray` | ndarray支持向量化运算，不需要双层for循环 |
| `for` 循环遍历数组 | 向量化运��� | NumPy中应避免用for循环，用向量化运算快100倍 |
| `ResultSet` | Pandas `DataFrame` | DataFrame更灵活，支持列名直接访问 |
| Spark DataFrame | Pandas DataFrame | 概念相似，但Pandas是单机的，Spark是分布式的 |
| Stream API | Pandas链式操作 | `.pipe()` + 方法链 ≈ Java Stream |
| SQL JOIN | `pd.merge()` | 完全对应的left/right/inner/outer |
| SQL GROUP BY | `df.groupby().agg()` | 完全对应 |
| Jackson/Gson（JSON解析） | `pd.read_json()` | Pandas原生支持 |
| Apache POI（Excel） | `pd.read_excel()` | 一行代码搞定 |
| JDBC | `pd.read_sql()` | 直接读SQL结果为DataFrame |

**重要思维转换**：
- NumPy/Pandas的核心思想是**向量化运算**，避免for循环。用向量化运算可以让代码快10-100倍。
- Pandas的DataFrame就是一个"内存中的数据库表"，几乎所有SQL操作都有对应的方法。
- 在AI开发中，数据处理占80%的时间，模型训练只占20%。掌握Pandas非常重要。
