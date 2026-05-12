# W08 - ML 综合实战与阶段总结

> 学习时间：第 8 周（第 2 月第 4 周）
> 本周是第一阶段最后一周，完成 Kaggle 实战、模型部署入门、数学回顾和深度学习预习

---

## 本周目标

1. 能独立完成端到端的 Kaggle 竞赛项目
2. 掌握模型持久化和 API 化部署
3. 系统回顾 ML 算法背后的数学原理
4. 了解深度学习的基本概念，为第二阶段做准备
5. 完善第一阶段所有项目并整理 GitHub

---

## 时间安排

| 时间 | 内容 |
|------|------|
| 周一（2h） | Kaggle 实战：EDA + 数据探索 |
| 周二（2h） | Kaggle 实战：特征工程 + 基线模型 |
| 周三（2h） | Kaggle 实战：模型调优 + 集成 + 提交 |
| 周四（2h） | 模型保存 + FastAPI 推理服务 |
| 周五（2h） | ML 数学回顾 + 深度学习预习 |
| 周六（4h） | 项目代码整理 + GitHub 完善 |
| 周日（4h） | 写学习总结 + 制定第二阶段计划 |

---

## 详细学习内容

### Day 1（周一）：Kaggle 实战 - EDA

#### 选择竞赛

推荐：House Prices - Advanced Regression Techniques（房价预测进阶版）
地址：https://www.kaggle.com/c/house-prices-advanced-regression-techniques

#### EDA 流程

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 1. 数据概览
print(f"训练集: {train.shape}, 测试集: {test.shape}")
print(train.info())
print(train.describe())

# 2. 目标变量分析
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
train['SalePrice'].hist(bins=50, ax=axes[0])
axes[0].set_title('SalePrice Distribution')
np.log1p(train['SalePrice']).hist(bins=50, ax=axes[1])
axes[1].set_title('Log(SalePrice) Distribution')
plt.show()
# 右偏 → 取对数使其接近正态分布

# 3. 缺失值分析
missing = train.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
missing_pct = missing / len(train) * 100
print("缺失值比例：")
print(missing_pct.head(20))
# 缺失 > 80% 的列考虑删除

# 4. 数值特征相关性
corr = train.select_dtypes(include=[np.number]).corr()
top_corr = corr['SalePrice'].abs().sort_values(ascending=False).head(15)
print("与目标最相关的特征：")
print(top_corr)

# 相关性热力图（Top 10）
top_features = top_corr.index[:10]
sns.heatmap(train[top_features].corr(), annot=True, cmap='RdBu_r', fmt='.2f')
plt.show()

# 5. 类别特征分析
cat_cols = train.select_dtypes(include=['object']).columns
for col in cat_cols[:5]:  # 看前5个
    plt.figure(figsize=(8, 4))
    train.boxplot(column='SalePrice', by=col)
    plt.title(f'SalePrice by {col}')
    plt.show()
```

---

### Day 2（周二）：Kaggle 实战 - 特征工程 + 基线模型

```python
# ===== 特征工程 =====

# 合并训练和测试集一起处理（方便统一编码）
all_data = pd.concat([train.drop('SalePrice', axis=1), test], axis=0)

# 1. 缺失值处理
# 有意义的缺失（如 PoolQC 缺失 = 没有泳池）
none_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
             'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
             'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
for col in none_cols:
    all_data[col].fillna('None', inplace=True)

# 数值缺失用 0 或中位数
zero_cols = ['GarageYrBlt', 'GarageArea', 'GarageCars',
             'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'MasVnrArea']
for col in zero_cols:
    all_data[col].fillna(0, inplace=True)

# LotFrontage 用同 Neighborhood 的中位数填充
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median()))

# 2. 特征变换
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['TotalBath'] = all_data['FullBath'] + 0.5 * all_data['HalfBath']
all_data['HasPool'] = (all_data['PoolArea'] > 0).astype(int)
all_data['IsRemodeled'] = (all_data['YearBuilt'] != all_data['YearRemodAdd']).astype(int)

# 3. 类别编码
# 有序类别用 Label Encoding
from sklearn.preprocessing import LabelEncoder
quality_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
quality_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
                'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond']
for col in quality_cols:
    all_data[col] = all_data[col].map(quality_map)

# 无序类别用 One-Hot
all_data = pd.get_dummies(all_data)

# 4. 分割回训练/测试集
X_train = all_data[:len(train)].values
X_test = all_data[len(train):].values
y_train = np.log1p(train['SalePrice'])

# ===== 基线模型对比 =====
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb

models = {
    'Ridge': Ridge(alpha=10),
    'Lasso': Lasso(alpha=0.001),
    'ElasticNet': ElasticNet(alpha=0.001),
    'RF': RandomForestRegressor(n_estimators=200, random_state=42),
    'GBR': GradientBoostingRegressor(n_estimators=500, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, random_state=42),
    'LightGBM': lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42)
}

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5,
                            scoring='neg_root_mean_squared_error')
    print(f"{name}: RMSE={-scores.mean():.5f} ± {scores.std():.5f}")
```

---

### Day 3（周三）：Kaggle 实战 - 模型调优 + 集成

```python
# ===== 超参数调优 =====
from sklearn.model_selection import RandomizedSearchCV

# XGBoost 调参
xgb_params = {
    'n_estimators': [500, 1000, 2000],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5],
    'reg_alpha': [0, 0.01, 0.1],
    'reg_lambda': [1, 1.5, 2]
}
xgb_search = RandomizedSearchCV(
    xgb.XGBRegressor(random_state=42), xgb_params,
    n_iter=50, cv=5, scoring='neg_root_mean_squared_error',
    random_state=42, n_jobs=-1
)
xgb_search.fit(X_train, y_train)

# LightGBM 调参
lgb_params = {
    'n_estimators': [500, 1000, 2000],
    'max_depth': [3, 5, 7, -1],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'num_leaves': [15, 31, 63, 127],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# ===== Stacking 集成 =====
from sklearn.ensemble import StackingRegressor

estimators = [
    ('ridge', Ridge(alpha=10)),
    ('xgb', xgb_search.best_estimator_),
    ('lgb', lgb_search.best_estimator_)
]
stacking = StackingRegressor(estimators=estimators, final_estimator=Ridge())
stacking.fit(X_train, y_train)

# ===== 生成提交文件 =====
y_pred = np.expm1(stacking.predict(X_test))
submission = pd.DataFrame({'Id': test['Id'], 'SalePrice': y_pred})
submission.to_csv('submission.csv', index=False)
```

---

### Day 4（周四）：模型保存 + FastAPI 部署

```python
# ===== 模型保存 =====
import joblib

# 保存模型和预处理器
joblib.dump(best_model, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(label_encoders, 'encoders.joblib')

# 加载模型
model = joblib.load('model.joblib')

# ===== FastAPI 推理服务 =====
# 文件: app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="House Price Prediction API")
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

class HouseFeatures(BaseModel):
    gr_liv_area: float
    total_bsmt_sf: float
    overall_qual: int
    year_built: int
    garage_cars: int
    # ... 其他特征

@app.post("/predict")
def predict(features: HouseFeatures):
    data = np.array([[features.gr_liv_area, features.total_bsmt_sf,
                      features.overall_qual, features.year_built,
                      features.garage_cars]])
    data_scaled = scaler.transform(data)
    prediction = np.expm1(model.predict(data_scaled))[0]
    return {"predicted_price": round(prediction, 2)}

# 运行: uvicorn app:app --reload --port 8000
# 测试: curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"gr_liv_area":1500,"total_bsmt_sf":1000,"overall_qual":7,"year_built":2000,"garage_cars":2}'
```

---

### Day 5（周五）：ML 数学回顾 + 深度学习预习

#### ML 数学总结

| 算法 | 核心数学 |
|------|----------|
| 线性回归 | 最小二乘法 / MLE → 解析解或梯度下降 |
| 逻辑回归 | 最大似然估计 → 交叉熵损失 → 梯度下降 |
| SVM | 凸优化 + KKT 条件 + 对偶问题 + 核技巧 |
| 决策树 | 信息论（熵/信息增益/基尼系数） |
| K-Means | EM 算法（E步分配/M步更新中心） |
| PCA | 协方差矩阵特征分解 / SVD |
| 正则化 | L1 (拉普拉斯先验) / L2 (高斯先验) |

#### 深度学习预习

```
神经网络发展时间线：
1943 McCulloch-Pitts 神经元模型（数学模型）
1958 感知机 Perceptron（单层，只能处理线性可分）
1969 Minsky 指出感知机不能解决 XOR 问题 → AI 寒冬
1986 反向传播算法 Backpropagation → MLP 可以训练了
1998 LeNet-5（第一个成功的 CNN，手写数字识别）
2012 AlexNet（深度学习爆发，GPU 训练 + ReLU）
2014 GAN（生成对抗网络）
2017 Transformer（Attention Is All You Need）
2020 GPT-3（大语言模型时代）

感知机：f(x) = σ(w·x + b)
  σ 是激活函数（Step/Sigmoid/ReLU）
  只有一层 → 只能处理线性可分问题

为什么需要激活函数？
  没有激活函数：f(x) = W3(W2(W1x)) = W'x（还是线性变换！）
  有激活函数：引入非线性，使网络能拟合任意复杂函数（万能逼近定理）
```

---

### Day 6-7（周末）：项目整理 + GitHub + 总结

#### 项目代码整理

```
big-model-learn/
├── project-01-house-price/
│   ├── README.md
│   ├── notebooks/
│   │   └── exploration.ipynb
│   ├── src/
│   │   ├── train.py
│   │   └── predict.py
│   ├── models/
│   └── requirements.txt
├── project-02-customer-churn/
│   └── ...
├── project-03-user-segmentation/
│   └── ...
└── 学习计划/
    └── ...
```

#### README 模板

```markdown
# 房价预测项目

## 项目描述
使用 Ames 房价数据集，通过特征工程和模型集成预测房屋价格。

## 技术栈
- Python 3.11, pandas, scikit-learn, XGBoost, LightGBM

## 模型对比
| 模型 | RMSE | R² |
|------|------|-----|
| Ridge | 0.1234 | 0.89 |
| XGBoost | 0.1120 | 0.91 |
| Stacking | 0.1085 | 0.93 |

## 如何运行
\`\`\`bash
pip install -r requirements.txt
python src/train.py
\`\`\`
```

---

## 本周产出

- [ ] Kaggle 竞赛提交（至少 1 次）
- [ ] FastAPI 模型推理服务
- [ ] ML 数学知识总结文档
- [ ] GitHub 上 3 个完整 ML 项目（README + 代码 + 结果）

---

## 自测题

1. **从数据到部署的完整 ML 流程是什么？**
   <details><summary>参考答案</summary>
   数据收集 → EDA → 数据清洗 → 特征工程 → 模型选择 → 训练 → 超参数调优 → 评估 → 模型保存 → API 部署 → 监控。关键是在每个步骤都做好记录和版本管理，确保可复现。
   </details>

2. **如何判断模型是过拟合还是欠拟合？分别怎么处理？**
   <details><summary>参考答案</summary>
   过拟合：train_loss << val_loss，训练集表现远好于测试集。处理：增加数据、正则化、降低模型复杂度、Dropout、Early Stopping、数据增强。
   欠拟合：train_loss 和 val_loss 都高。处理：增加特征、使用更复杂模型、减少正则化、增加训练时间。
   </details>

3. **集成学习为什么通常比单模型效果好？**
   <details><summary>参考答案</summary>
   Bagging：多个独立模型的错误不相关，投票后错误率降低（数学：k个独立分类器，每个错误率ε<0.5，多数投票错误率随k增加而降低）。
   Boosting：每个新模型专注于纠正之前的错误，逐步降低偏差。
   Stacking：让元模型学习如何最优地组合基模型的预测。关键前提：基模型要有差异性（不同算法/不同数据/不同超参数）。
   </details>

4. **模型部署时需要注意什么？**
   <details><summary>参考答案</summary>
   (1) 训练-推理一致性：预处理步骤必须完全一致（用 Pipeline 保存）。
   (2) 输入验证：API 要验证输入格式和范围。
   (3) 版本管理：模型和代码要关联版本。
   (4) 性能：推理延迟、并发能力。
   (5) 监控：数据漂移、模型效果退化告警。
   </details>

5. **监督学习和无监督学习各适合什么场景？**
   <details><summary>参考答案</summary>
   监督学习：有明确预测目标（分类/回归），如有标签数据。场景：垃圾邮件分类、房价预测、疾病诊断。
   无监督学习：没有标签，发现数据内在结构。场景：用户分群、异常检测、降维可视化、推荐系统的协同过滤。
   实际工作中，常先无监督探索数据，再有监督建模。
   </details>

---

## Java 开发者提示

| 概念 | Java 类比 |
|------|-----------|
| Kaggle 竞赛 | 类似 Hackathon 编程比赛，用真实数据比拼 |
| FastAPI 部署模型 | 类似 Spring Boot 暴露 REST API |
| joblib 保存模型 | 类似 Java 序列化对象到文件 |
| 模型版本管理 | 类似 Git 管理代码版本 |
| Stacking 集成 | 类似责任链模式 + 组合模式 |
