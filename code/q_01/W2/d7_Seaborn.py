########掌握Seaborn高级图表，完成Titanic数据集的完整EDA
#### Seaborn基础
import seaborn as sns                   # 导入seaborn库，sns是约定别名，基于matplotlib的高级绘图库
# 设置主题
sns.set_theme(style='whitegrid')        # 设置全局主题为白底网格风格，后续所有图表都会继承此风格

# 加载示例数据集
titanic = sns.load_dataset('titanic')   # 加载Seaborn内置的泰坦尼克号数据集（891条乘客记录）
tips = sns.load_dataset('tips')         # 加载Seaborn内置的餐厅小费数据集（用餐金额、小费、时间等）

# 热力图（相关系数矩阵）
plt.figure(figsize=(10, 8))             # 创建画布，宽10英寸高8英寸
numeric_titanic = titanic.select_dtypes(include=[np.number])  # 只选取数值类型的列（排除字符串等），因为corr()只能计算数值相关性
sns.heatmap(numeric_titanic.corr(), annot=True, fmt='.2f', cmap='coolwarm')  # 绘制热力图：corr()计算相关系数矩阵，annot=True在每个格子显示数值，fmt='.2f'保留两位小数，cmap='coolwarm'蓝红双色（负相关蓝色，正相关红色）
plt.title('Correlation Heatmap')        # 设置标题：相关性热力图
plt.show()                              # 弹窗显示图表

# 箱线图
plt.figure(figsize=(10, 6))             # 创建画布
sns.boxplot(data=titanic, x='pclass', y='age', hue='survived')  # 绘制箱线图：x轴为舱位等级(1/2/3)，y轴为年龄，hue='survived'按生存状态分色（蓝色=未生存，橙色=生存）；箱线图显示中位数、四分位数和异常值
plt.title('Age Distribution by Class and Survival')  # 标题：按舱位和生存状态的年龄分布
plt.show()                              # 弹窗显示图表

# 小提琴图
plt.figure(figsize=(10, 6))             # 创建画布
sns.violinplot(data=titanic, x='pclass', y='age', hue='survived', split=True)  # 绘制小提琴图：split=True将同一组两种颜色左右分开显示，比箱线图更能看出分布形状（宽=密度高，窄=密度低）
plt.title('Age Distribution (Violin Plot)')  # 标题：年龄分布（小提琴图）
plt.show()                              # 弹窗显示图表

# 分布图
fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 创建1行2列的子图布局，总宽12英寸
sns.histplot(titanic['age'].dropna(), kde=True, ax=axes[0])  # 第1个子图：绘制年龄直方图，dropna()去掉缺失值，kde=True叠加核密度估计曲线（平滑的概率密度线）
axes[0].set_title('Age Distribution')   # 第1个子图标题：年龄分布
sns.kdeplot(data=titanic, x='age', hue='survived', fill=True, ax=axes[1])  # 第2个子图：只画核密度曲线，hue='survived'按生存状态分色，fill=True填充颜色区域，更直观对比两组分布差异
axes[1].set_title('Age Distribution by Survival')  # 第2个子图标题：按生存状态的年龄分布
plt.show()                              # 弹窗显示图表

# pairplot（多变量关系图）
sns.pairplot(titanic[['survived', 'age', 'fare', 'pclass']].dropna(),  # 选取4个关键列，dropna()去掉缺失行；pairplot自动画出所有变量两两之间的散点图
    hue='survived', diag_kind='kde')    # hue='survived'按生存状态分色，diag_kind='kde'对角线上画密度曲线（默认是直方图）
plt.show()                              # 弹窗显示图表

# catplot（分类图）
sns.catplot(data=titanic, x='pclass', y='survived', hue='sex', kind='bar')  # catplot是分类图的统一接口，kind='bar'画柱状图；x=舱位等级，y=生存率（自动计算均值），hue='sex'按性别分色
plt.show()                              # 弹窗显示图表

# relplot（关系图）
sns.relplot(data=tips, x='total_bill', y='tip', hue='time', size='size',  # relplot是关系图的统一接口；x=账单金额，y=小费，hue='time'按用餐时间(Lunch/Dinner)分色
            sizes=(50, 200), alpha=0.7)  # size='size'按用餐人数映射点大小，sizes=(50,200)指定最小和最大点的像素尺寸，alpha=0.7设置透明度避免点重叠遮挡
plt.show()                              # 弹窗显示图表

### Titanic EDA完整流程
import pandas as pd                     # 导入pandas，用于数据操作
import numpy as np                      # 导入numpy，用于数值计算
import matplotlib.pyplot as plt         # 导入matplotlib绘图模块
import seaborn as sns                   # 导入seaborn高级绘图库

# 1. 加载数据
titanic = sns.load_dataset('titanic')   # 加载泰坦尼克号数据集

# 2. 数据概览
print(titanic.shape)                    # 打印数据维度：(891行, 15列)
print(titanic.info())                   # 打印每列的类型、非空数量、内存占用
print(titanic.describe())               # 打印数值列的统计摘要（均值、标准差、最小值、最大值等）
print(titanic.isnull().sum())           # 打印每列的缺失值数量，发现age有177个缺失、deck有688个缺失等

# 3. 目标变量分布
fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 创建1行2列子图
titanic['survived'].value_counts().plot.pie(autopct='%1.1f%%', ax=axes[0])  # 第1个子图：饼图显示生存率百分比，autopct='%1.1f%%'保留1位小数
axes[0].set_title('Survival Rate')      # 标题：生存率
sns.countplot(data=titanic, x='survived', ax=axes[1])  # 第2个子图：柱状图显示生存/未生存的人数，更直观看绝对数量
axes[1].set_title('Survival Count')     # 标题：生存人数
plt.show()                              # 弹窗显示图表

# 4. 单变量分析
fig, axes = plt.subplots(2, 2, figsize=(14, 10))  # 创建2×2共4个子图，用于同时查看4个变量的分布
sns.histplot(titanic['age'].dropna(), kde=True, ax=axes[0,0])  # 左上：年龄分布直方图+密度曲线
axes[0,0].set_title('Age Distribution')  # 标题
sns.countplot(data=titanic, x='sex', ax=axes[0,1])  # 右上：性别分布柱状图（男女数量对比）
axes[0,1].set_title('Gender Distribution')  # 标题
sns.countplot(data=titanic, x='pclass', ax=axes[1,0])  # 左下：舱位等级分布柱状图（1/2/3等舱人数）
axes[1,0].set_title('Class Distribution')  # 标题
sns.histplot(titanic['fare'], kde=True, ax=axes[1,1])  # 右下：票价分布直方图+密度曲线（右偏分布，大部分票价较低）
axes[1,1].set_title('Fare Distribution')  # 标题
plt.tight_layout()                      # 自动调整子图间距，防止标签重叠
plt.show()                              # 弹窗显示图表

# 5. 生存率分析
fig, axes = plt.subplots(1, 3, figsize=(18, 5))  # 创建1行3列子图，分别从性别、舱位、登船港口分析生存率
sns.barplot(data=titanic, x='sex', y='survived', ax=axes[0])  # 左：按性别看生存率，女性生存率远高于男性（约74% vs 19%）
axes[0].set_title('Survival by Gender')  # 标题
sns.barplot(data=titanic, x='pclass', y='survived', ax=axes[1])  # 中：按舱位看生存率，1等舱最高，3等舱最低
axes[1].set_title('Survival by Class')  # 标题
sns.barplot(data=titanic, x='embarked', y='survived', ax=axes[2])  # 右：按登船港口(C/Q/S)看生存率
axes[2].set_title('Survival by Embarked')  # 标题
plt.show()                              # 弹窗显示图表

# 6. 相关性分析
plt.figure(figsize=(10, 8))             # 创建画布
numeric_cols = titanic.select_dtypes(include=[np.number])  # 选取数值列，排除非数值列
sns.heatmap(numeric_cols.corr(), annot=True, cmap='RdBu_r', center=0)  # 绘制热力图：center=0以0为中心对称着色（红=正相关，蓝=负相关，白=无相关）
plt.title('Feature Correlation Heatmap')  # 标题：特征相关性热力图
plt.show()                              # 弹窗显示图表

# 7. 多变量交互分析
sns.catplot(data=titanic, x='pclass', y='survived', hue='sex', kind='point')  # catplot的kind='point'画点图：同时看舱位和性别对生存率的影响，每个点表示该组的生存率均值，竖线表示置信区间
plt.title('Survival: Class x Gender')   # 标题
plt.show()                              # 弹窗显示图表

# 8. 年龄与生存
plt.figure(figsize=(10, 6))             # 创建画布
sns.kdeplot(data=titanic, x='age', hue='survived', fill=True, common_norm=False)  # 核密度曲线对比：common_norm=False分别计算两组的密度（否则会按总人数归一化，导致两组形状不对等），fill=True填充颜色便于观察重叠区域
plt.title('Age Distribution by Survival')  # 标题
plt.show()                              # 弹窗显示图表