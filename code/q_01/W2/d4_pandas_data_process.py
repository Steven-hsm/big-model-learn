###########################掌握数据清洗、过滤、分组和合并操作
##### 缺失值处理
import pandas as pd
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

########数据过滤与排序
# 排序
df.sort_values('age', ascending=False)           # 按列排序
df.sort_values(['city', 'age'], ascending=[True, False])  # 多列排序
df.sort_index()

# 去重
df.drop_duplicates()                  # 删除完全重复的行
df.drop_duplicates(subset=['name'])   # 按列去重
df.duplicated()                       # 返回布尔Series标记重复行

# 值替换
df['city'].replace('Beijing', 'BJ')   # 替换值
df.replace({np.nan: 0})               # 批量替换

#### 分组聚合（groupby）
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

#多列分组
df.groupby(['city', 'age'])['salary'].mean()

# agg vs transform vs apply
# agg：聚合后每组一行
df.groupby('city')['salary'].agg('mean')

# transform：聚合后保持原形状（广播回每行）
df['city_avg'] = df.groupby('city')['salary'].transform('mean')
df['salary_diff'] = df['salary'] - df['city_avg']

# apply：对每组应用任意函数
df.groupby('city').apply(lambda g: g.nlargest(2, 'salary'))

### 合并操作
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