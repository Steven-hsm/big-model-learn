####################掌握Series和DataFrame的创建、查看和基本选择操作
import pandas as pd

# 创建Series
s = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])
# a    10
# b    20
# c    30
# d    40
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

# 从字典创建
s2 = pd.Series({'Alice': 85, 'Bob': 92, 'Charlie': 78})

# 运算（自动对齐索引）
s * 2           # 所有值乘以2
s + pd.Series([1, 2], index=['a', 'b'])  # 不匹配的索引为NaN

######### DataFrame创建
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

###数据预览
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

###数据选择
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

