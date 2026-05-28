##############掌握Matplotlib绑定各种常见图表
import matplotlib.pyplot as plt  # 导入matplotlib的绘图模块，plt是约定别名
import numpy as np               # 导入numpy，np是约定别名
import pandas as pd

# 设置中文字体（Windows）
plt.rcParams['font.sans-serif'] = ['SimHei']   # 设置默认字体为黑体，否则中文会显示为方块
plt.rcParams['axes.unicode_minus'] = False      # 正常显示负号，否则负号会显示为方块

# ============ 折线图 ============
x = np.linspace(0, 2*np.pi, 100)  # 生成0到2π之间100个等间距的点，用于画平滑曲线
plt.figure(figsize=(10, 6))        # 创建画布，宽10英寸高6英寸
plt.plot(x, np.sin(x), label='sin(x)', color='blue', linewidth=2)       # 绘制sin曲线，蓝色实线，线宽2，标签为sin(x)
plt.plot(x, np.cos(x), label='cos(x)', color='red', linewidth=2, linestyle='--')  # 绘制cos曲线，红色虚线(--)，线宽2
plt.title('三角函数')               # 设置图表标题
plt.xlabel('x')                     # 设置x轴标签
plt.ylabel('y')                     # 设置y轴标签
plt.legend()                        # 显示图例（即label标注的线条名称）
plt.grid(True, alpha=0.3)           # 显示网格线，透明度0.3（浅灰色辅助线）
plt.savefig('trig.png', dpi=150, bbox_inches='tight')  # 保存图片，dpi=150高清，tight裁掉白边
plt.show()                          # 弹窗显示图表

# ============ 柱状图 ============
categories = ['A', 'B', 'C', 'D']                    # 四个类别名称
values = [23, 45, 56, 78]                            # 对应每个类别的数值
plt.figure(figsize=(8, 5))                           # 创建画布，宽8英寸高5英寸
plt.bar(categories, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])  # 绘制柱状图，每根柱子指定不同颜色
plt.title('各类别数量')                               # 设置标题
plt.ylabel('数量')                                    # 设置y轴标签
for i, v in enumerate(values):                       # enumerate同时获取索引i和值v
    plt.text(i, v + 1, str(v), ha='center')          # 在每根柱子顶部标注数值，ha='center'水平居中
plt.show()                                            # 弹窗显示图表

# ============ 散点图 ============
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'age': [25, 30, 35, 28],
    'city': ['Beijing', 'Shanghai', 'Guangzhou', 'Shenzhen'],
    'salary': [15000, 20000, 18000, 16000]
})

plt.figure(figsize=(8, 6))                                           # 创建画布
plt.scatter(df['age'], df['salary'], c=df['salary'], cmap='viridis', s=50, alpha=0.6)  # 绘制散点图：x=年龄，y=薪水，颜色按分数映射(c)，viridis色表，点大小50，透明度0.6
plt.colorbar(label='Score')                                          # 显示颜色条，标注为Score
plt.xlabel('Age')                                                    # x轴标签
plt.ylabel('Salary')                                                 # y轴标签
plt.title('Age vs Salary (colored by Score)')                        # 标题
plt.show()                                                           # 弹窗显示图表

# ============ 直方图 ============
plt.figure(figsize=(8, 5))                                           # 创建画布
plt.hist(df['salary'], bins=30, edgecolor='black', alpha=0.7)        # 绘制直方图：数据为salary，分30个区间(bins)，柱子边框黑色，透明度0.7
plt.xlabel('Salary')                                                 # x轴标签
plt.ylabel('Frequency')                                              # y轴标签（频率/频次）
plt.title('Salary Distribution')                                     # 标题
plt.axvline(df['salary'].mean(), color='red', linestyle='--', label='Mean')  # 画一条红色虚线标记均值位置，axvline=垂直线
plt.legend()                                                         # 显示图例
plt.show()                                                           # 弹窗显示图表

# ============ 饼图 ============
plt.figure(figsize=(8, 8))                                           # 创建画布，饼图通常用正方形画布
plt.pie(df['city'].value_counts(), labels=df['city'].value_counts().index,  # 数据：各城市出现次数，标签：城市名称
        autopct='%1.1f%%', startangle=90)                             # autopct显示百分比（保留1位小数），startangle=90从90度开始画（即从顶部开始）
plt.title('City Distribution')                                       # 标题
plt.show()                                                           # 弹窗显示图表

# ============ 子图 ============
fig, axes = plt.subplots(2, 2, figsize=(12, 10))                    # 创建2×2共4个子图的画布，fig是整个画布，axes是子图数组
axes[0, 0].plot(x, np.sin(x))                                       # 第1个子图（左上）：sin曲线
axes[0, 0].set_title('Sin')                                         # 第1个子图标题
axes[0, 1].plot(x, np.cos(x))                                       # 第2个子图（右上）：cos曲线
axes[0, 1].set_title('Cos')                                         # 第2个子图标题
axes[1, 0].hist(np.random.randn(1000), bins=30)                     # 第3个子图（左下）：1000个正态分布随机数的直方图
axes[1, 0].set_title('Histogram')                                   # 第3个子图标题
axes[1, 1].scatter(np.random.rand(50), np.random.rand(50))          # 第4个子图（右下）：50个随机点的散点图
axes[1, 1].set_title('Scatter')                                     # 第4个子图标题
plt.tight_layout()                                                   # 自动调整子图间距，防止标题和标签重叠
plt.show()                                                           # 弹窗显示图表