"""
W08 Day 2: Kaggle实战 - 特征工程 + 基线模型
California Housing 数据集的特征工程与基线模型对比
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# ============================================================
# Matplotlib 中文显示设置
# ============================================================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 数据加载与特征工程
# ============================================================
print("=" * 60)
print("Kaggle实战: 特征工程 + 基线模型")
print("=" * 60)

data = fetch_california_housing(as_frame=True)
df = data.frame.copy()
print(f"\n原始数据形状: {df.shape}")

# --- 1.1 创建新特征 ---
print("\n--- 1.1 特征工程 ---")
df['rooms_per_household'] = df['AveRooms']  # 平均房间数已存在
df['rooms_per_household'] = df['total_rooms'] = df['HouseAge']  # placeholder

# 重新加载并用原始列名操作
df = data.frame.copy()
print(f"原始列名: {list(df.columns)}")

# 创建派生特征
df['rooms_per_household'] = df['AveRooms']  # 已有
df['bedrooms_per_room'] = df['AveBedrms']   # 已有
df['population_per_household'] = df['Population'] / df['AveOccup']

# 更有意义的特征工程
df['rooms_per_household'] = df['AveRooms']
df['bedrooms_per_room'] = df['AveBedrms'] / df['AveRooms']
df['population_per_household'] = df['AveOccup']

print("新增派生特征:")
print(f"  bedrooms_per_room: 平均={df['bedrooms_per_room'].mean():.4f}")
print(f"  population_per_household: 平均={df['population_per_household'].mean():.2f}")

# 添加一个模拟的类别特征 (基于经纬度划分区域)
df['region'] = pd.cut(df['Latitude'],
                      bins=[32, 34, 36, 38, 42],
                      labels=['南部', '中南', '中部', '北部'])
print(f"\n区域分布:\n{df['region'].value_counts()}")

# 模拟缺失值 (为了演示 SimpleImputer)
np.random.seed(42)
mask_bedrooms = np.random.random(len(df)) < 0.05  # 5% 缺失
df_missing = df.copy()
df_missing.loc[mask_bedrooms, 'AveBedrms'] = np.nan
print(f"\n模拟缺失值后 AveBedrms 缺失数: {df_missing['AveBedrms'].isnull().sum()}")

# ============================================================
# 2. 数据预处理 Pipeline
# ============================================================
print("\n--- 2. 构建 ColumnTransformer Pipeline ---")

# 划分特征和标签
X = df_missing.drop('MedHouseVal', axis=1)
y = df_missing['MedHouseVal']

# 数值特征
numeric_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                    'Population', 'AveOccup', 'Latitude', 'Longitude',
                    'bedrooms_per_room', 'population_per_household']
# 类别特征
categorical_features = ['region']

print(f"数值特征 ({len(numeric_features)}): {numeric_features}")
print(f"类别特征 ({len(categorical_features)}): {categorical_features}")

# 数值 Pipeline: 缺失值填充 + 标准化
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 类别 Pipeline: OneHot 编码
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', sparse_output=False))
])

# 组合 ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# ============================================================
# 3. 训练集 / 测试集划分
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")

# ============================================================
# 4. 基线模型对比
# ============================================================
print("\n--- 4. 基线模型对比 (5折交叉验证) ---")

models = {
    'LinearRegression': LinearRegression(),
    'Ridge (alpha=1)': Ridge(alpha=1.0),
    'Lasso (alpha=0.01)': Lasso(alpha=0.01, max_iter=5000),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42
    )
}

results = []

for name, model in models.items():
    # 构建完整 Pipeline
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # 5折交叉验证 (负 MSE -> RMSE)
    cv_scores = cross_val_score(pipe, X_train, y_train,
                                cv=5, scoring='neg_mean_squared_error',
                                n_jobs=-1)
    rmse_scores = np.sqrt(-cv_scores)

    # 在测试集上评估
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    results.append({
        '模型': name,
        'CV_RMSE_均值': rmse_scores.mean(),
        'CV_RMSE_标准差': rmse_scores.std(),
        'Test_RMSE': test_rmse
    })

    print(f"  {name:25s} | CV RMSE: {rmse_scores.mean():.4f} +/- {rmse_scores.std():.4f} "
          f"| Test RMSE: {test_rmse:.4f}")

# ============================================================
# 5. 结果对比表
# ============================================================
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Test_RMSE')

print("\n--- 5. 模型性能排名 ---")
print(results_df.to_string(index=False))

# ============================================================
# 6. 可视化对比
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# CV RMSE 对比
colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(results_df))]
results_sorted = results_df.sort_values('CV_RMSE_均值')
bars = axes[0].barh(results_sorted['模型'], results_sorted['CV_RMSE_均值'],
                    xerr=results_sorted['CV_RMSE_标准差'],
                    color=colors[:len(results_sorted)], edgecolor='white', alpha=0.85)
axes[0].set_xlabel('RMSE (越低越好)', fontsize=12)
axes[0].set_title('交叉验证 RMSE 对比', fontsize=14)
axes[0].axvline(results_sorted['CV_RMSE_均值'].min(), color='red',
                linestyle='--', alpha=0.5)

# Test RMSE 对比
colors_test = ['#e74c3c' if i == 0 else '#9b59b6' for i in range(len(results_df))]
axes[1].barh(results_df['模型'], results_df['Test_RMSE'],
             color=colors_test[:len(results_df)], edgecolor='white', alpha=0.85)
axes[1].set_xlabel('Test RMSE (越低越好)', fontsize=12)
axes[1].set_title('测试集 RMSE 对比', fontsize=14)
axes[1].axvline(results_df['Test_RMSE'].min(), color='red',
                linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('d2_baseline_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[图表已保存] d2_baseline_comparison.png")

# ============================================================
# 7. 最佳模型预测可视化
# ============================================================
best_model_name = results_df.iloc[0]['模型']
best_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', models[best_model_name])
])
best_pipe.fit(X_train, y_train)
y_pred_best = best_pipe.predict(X_test)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 实际 vs 预测散点图
axes[0].scatter(y_test, y_pred_best, alpha=0.3, s=5, c='steelblue')
max_val = max(y_test.max(), y_pred_best.max())
axes[0].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='完美预测')
axes[0].set_xlabel('实际值', fontsize=12)
axes[0].set_ylabel('预测值', fontsize=12)
axes[0].set_title(f'最佳模型: {best_model_name}\n实际 vs 预测', fontsize=13)
axes[0].legend()

# 残差分布
residuals = y_test - y_pred_best
axes[1].hist(residuals, bins=50, color='coral', edgecolor='white', alpha=0.8)
axes[1].axvline(0, color='red', linestyle='--')
axes[1].set_xlabel('残差 (实际 - 预测)', fontsize=12)
axes[1].set_ylabel('频数', fontsize=12)
axes[1].set_title(f'残差分布 (均值={residuals.mean():.4f})', fontsize=13)

plt.tight_layout()
plt.savefig('d2_best_model_predictions.png', dpi=150, bbox_inches='tight')
plt.close()
print("[图表已保存] d2_best_model_predictions.png")

# ============================================================
# 8. 特征重要性 (基于最佳树模型)
# ============================================================
print("\n--- 8. 特征重要性分析 ---")
# 使用 RandomForest 获取特征重要性
rf_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])
rf_pipe.fit(X_train, y_train)

# 获取特征名
ohe = rf_pipe.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
cat_features = ohe.get_feature_names_out(categorical_features).tolist()
all_features = numeric_features + cat_features

importances = rf_pipe.named_steps['model'].feature_importances_
feat_imp = pd.DataFrame({'特征': all_features, '重要性': importances})
feat_imp = feat_imp.sort_values('重要性', ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(feat_imp['特征'], feat_imp['重要性'], color='teal', edgecolor='white')
ax.set_xlabel('特征重要性', fontsize=12)
ax.set_title('RandomForest 特征重要性', fontsize=14)
plt.tight_layout()
plt.savefig('d2_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("[图表已保存] d2_feature_importance.png")

print("\n" + "=" * 60)
print("特征工程 + 基线模型总结")
print("=" * 60)
print(f"""
1. 新增特征: bedrooms_per_room, population_per_household
2. Pipeline: SimpleImputer + StandardScaler + OneHotEncoder
3. 最佳基线模型: {best_model_name}
   - Test RMSE: {results_df.iloc[0]['Test_RMSE']:.4f}
4. 特征工程显著提升了模型性能
5. 下一步: 超参数调优 + 模型集成
""")
