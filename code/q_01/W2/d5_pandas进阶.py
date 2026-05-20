############################# 掌握行/列级操作、透视表、时间序列处理。
#### apply/map/applymap
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'age': [25, 30, 35, 28],
    'city': ['Beijing', 'Shanghai', 'Guangzhou', 'Shenzhen'],
    'salary': [15000, 20000, 18000, 16000]
})

# map：Series的元素级操作
df['city_code'] = df['city'].map({"Beijing":1,'Shanghai':2})

# apply：series或DataFrame的行/列级操作
df['salary_k'] = df['salary'].apply(lambda x: x / 1000) 

#Series.apply
df[['age', 'salary']].apply(np.sqrt)

# 对每列应用
df.apply(lambda row: row['age'] * 1000, axis=1)
# 对每行应用
df[['age', 'salary']].map(lambda x: f"{x:.2f}")

# pipe：链式操作（类似Java Stream）
print(result)

###透视表与交叉表
# pivot_table：类似Excel的数据透视表
df.pivot_table(values='salary', index='city', aggfunc='mean')

# 多个聚合函数
df.pivot_table(values='salary', index='city', aggfunc=['mean', 'sum', 'count'])

# margins：添加行列合计
df.pivot_table(values='salary', index='city', aggfunc='mean', margins=True)

# crosstab：交叉表（计算频数）
pd.crosstab(df['city'], df['gender'])
pd.crosstab(df['city'], df['gender'], normalize='index')  # 行百分比

### 时间序列
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

#### 字符串操作（.str访问器）
s = pd.Series(['Hello World', 'Python AI', 'data science'])

s.str.lower()          # 全部小写
s.str.upper()          # 全部大写
s.str.strip()          # 去除首尾空格
s.str.split(' ')       # 按空格分割
s.str.contains('AI')   # 是否包含
s.str.replace(' ', '_')# 替换
s.str.len()            # 字符串长度
s.str[0:5]             # 切片
