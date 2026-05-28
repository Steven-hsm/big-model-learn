"""
W08 Day 3: Kaggle实战 - 模型调优 + 集成
使用 RandomizedSearchCV 调优 + Stacking 集成
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     RandomizedSearchCV)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import (RandomForestRegressor,
                              GradientBoostingRegressor,
                              StackingRegressor)
from sklearn.metrics import mean_squared_error
from scipy.stats import uniform, randint

# ============================================================
# Matplotlib 中文显示设置
# ============================================================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 数据准备
# ============================================================
print("=" * 60)
print("Kaggle实战: 模型调优 + 集成")
print("=" * 60)

data = fetch_california_housing(as_frame=True)
df = data.frame.copy()

# 特征工程
df['bedrooms_per_room'] = df['AveBedrms'] / df['AveRooms']
df['population_per_household'] = df['AveOccup']
df['rooms_per_household'] = df['AveRooms']

# 区域特征
df['region'] = pd.cut(df['Latitude'],
                      bins=[32, 34, 36, 38, 42],
                      labels=['south', 'mid_south', 'mid', 'north'])

X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

numeric_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                    'Population', 'AveOccup', 'Latitude', 'Longitude',
                    'bedrooms_per_room', 'population_per_household',
                    'rooms_per_household']
categorical_features = ['region']

# 预处理 Pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

# ============================================================
# 2. RandomForest 超参数调优
# ============================================================
print("\n--- 2. RandomForest RandomizedSearchCV ---")

rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42, n_jobs=-1))
])

rf_param_dist = {
    'model__n_estimators': randint(100, 500),
    'model__max_depth': randint(5, 30),
    'model__min_samples_split': randint(2, 20),
    'model__min_samples_leaf': randint(1, 10),
    'model__max_features': ['sqrt', 'log2', 0.5, 0.8],
}

rf_search = RandomizedSearchCV(
    rf_pipeline, rf_param_dist,
    n_iter=30, cv=3,
    scoring='neg_mean_squared_error',
    random_state=42, n_jobs=-1, verbose=0
)
rf_search.fit(X_train, y_train)

rf_best_rmse = np.sqrt(-rf_search.best_score_)
print(f"  最佳参数: {rf_search.best_params_}")
print(f"  最佳 CV RMSE: {rf_best_rmse:.4f}")

# ============================================================
# 3. GradientBoosting 超参数调优
# ============================================================
print("\n--- 3. GradientBoosting RandomizedSearchCV ---")

gb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', GradientBoostingRegressor(random_state=42))
])

gb_param_dist = {
    'model__n_estimators': randint(100, 500),
    'model__max_depth': randint(3, 10),
    'model__learning_rate': uniform(0.01, 0.2),
    'model__min_samples_split': randint(2, 20),
    'model__min_samples_leaf': randint(1, 10),
    'model__subsample': uniform(0.6, 0.4),
}

gb_search = RandomizedSearchCV(
    gb_pipeline, gb_param_dist,
    n_iter=30, cv=3,
    scoring='neg_mean_squared_error',
    random_state=42, n_jobs=-1, verbose=0
)
gb_search.fit(X_train, y_train)

gb_best_rmse = np.sqrt(-gb_search.best_score_)
print(f"  最佳参数: {gb_search.best_params_}")
print(f"  最佳 CV RMSE: {gb_best_rmse:.4f}")

# ============================================================
# 4. 调优结果对比
# ============================================================
print("\n--- 4. 调优模型测试集对比 ---")

rf_test_pred = rf_search.best_estimator_.predict(X_test)
rf_test_rmse = np.sqrt(mean_squared_error(y_test, rf_test_pred))

gb_test_pred = gb_search.best_estimator_.predict(X_test)
gb_test_rmse = np.sqrt(mean_squared_error(y_test, gb_test_pred))

print(f"  RandomForest (调优后):  Test RMSE = {rf_test_rmse:.4f}")
print(f"  GradientBoosting (调优后): Test RMSE = {gb_test_rmse:.4f}")

# ============================================================
# 5. Stacking 集成
# ============================================================
print("\n--- 5. Stacking 集成 ---")

# 先预处理数据 (StackingRegressor 需要直接数值输入)
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# 提取最佳参数训练基学习器
rf_best = RandomForestRegressor(
    **{k.replace('model__', ''): v for k, v in rf_search.best_params_.items()},
    random_state=42, n_jobs=-1
)

gb_best = GradientBoostingRegressor(
    **{k.replace('model__', ''): v for k, v in gb_search.best_params_.items()},
    random_state=42
)

# Stacking: Ridge 作为元学习器
stacking_model = StackingRegressor(
    estimators=[
        ('ridge', Ridge(alpha=1.0)),
        ('rf', rf_best),
        ('gb', gb_best)
    ],
    final_estimator=Ridge(alpha=1.0),
    cv=3,
    n_jobs=-1
)

stacking_model.fit(X_train_processed, y_train)

# 交叉验证评估
stack_cv_scores = cross_val_score(
    stacking_model, X_train_processed, y_train,
    cv=5, scoring='neg_mean_squared_error', n_jobs=-1
)
stack_cv_rmse = np.sqrt(-stack_cv_scores)

stack_test_pred = stacking_model.predict(X_test_processed)
stack_test_rmse = np.sqrt(mean_squared_error(y_test, stack_test_pred))

print(f"  Stacking CV RMSE: {stack_cv_rmse.mean():.4f} +/- {stack_cv_rmse.std():.4f}")
print(f"  Stacking Test RMSE: {stack_test_rmse:.4f}")

# ============================================================
# 6. 最终对比汇总
# ============================================================
print("\n--- 6. 最终模型对比 ---")
comparison = pd.DataFrame({
    '模型': ['RandomForest (调优)', 'GradientBoosting (调优)', 'Stacking 集成'],
    'CV_RMSE': [rf_best_rmse, gb_best_rmse, stack_cv_rmse.mean()],
    'Test_RMSE': [rf_test_rmse, gb_test_rmse, stack_test_rmse]
}).sort_values('Test_RMSE')

print(comparison.to_string(index=False))

# ============================================================
# 7. 可视化: 模型对比 + 特征重要性
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 7.1 RMSE 对比条形图
x_pos = np.arange(len(comparison))
width = 0.35
bars1 = axes[0].bar(x_pos - width/2, comparison['CV_RMSE'], width,
                    label='CV RMSE', color='steelblue', edgecolor='white')
bars2 = axes[0].bar(x_pos + width/2, comparison['Test_RMSE'], width,
                    label='Test RMSE', color='coral', edgecolor='white')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(comparison['模型'], rotation=15, ha='right')
axes[0].set_ylabel('RMSE (越低越好)', fontsize=12)
axes[0].set_title('模型性能对比', fontsize=14)
axes[0].legend()
axes[0].set_ylim(bottom=max(0, comparison['Test_RMSE'].min() - 0.05))

# 在柱子上标注数值
for bar in bars1:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{bar.get_height():.4f}', ha='center', fontsize=9)
for bar in bars2:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{bar.get_height():.4f}', ha='center', fontsize=9)

# 7.2 特征重要性 (来自调优后的 GBM)
ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
cat_features = ohe.get_feature_names_out(categorical_features).tolist()
all_features = numeric_features + cat_features

gb_best.fit(X_train_processed, y_train)
importances = gb_best.feature_importances_
feat_imp = pd.DataFrame({'特征': all_features, '重要性': importances})
feat_imp = feat_imp.sort_values('重要性', ascending=True).tail(10)

axes[1].barh(feat_imp['特征'], feat_imp['重要性'], color='teal', edgecolor='white')
axes[1].set_xlabel('特征重要性', fontsize=12)
axes[1].set_title('Top 10 特征重要性 (GradientBoosting)', fontsize=14)

plt.tight_layout()
plt.savefig('d3_tuning_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[图表已保存] d3_tuning_results.png")

# ============================================================
# 8. Stacking 预测可视化
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

predictions = {
    'RandomForest': rf_test_pred,
    'GradientBoosting': gb_test_pred,
    'Stacking': stack_test_pred
}

for idx, (name, pred) in enumerate(predictions.items()):
    axes[idx].scatter(y_test, pred, alpha=0.2, s=5)
    max_val = 5.5
    axes[idx].plot([0, max_val], [0, max_val], 'r--', linewidth=1.5)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    axes[idx].set_xlabel('实际值', fontsize=11)
    axes[idx].set_ylabel('预测值', fontsize=11)
    axes[idx].set_title(f'{name}\nRMSE={rmse:.4f}', fontsize=12)
    axes[idx].set_xlim(0, max_val)
    axes[idx].set_ylim(0, max_val)

plt.tight_layout()
plt.savefig('d3_predictions_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("[图表已保存] d3_predictions_comparison.png")

print("\n" + "=" * 60)
print("模型调优 + 集成总结")
print("=" * 60)
print(f"""
1. RandomForest 调优后 RMSE: {rf_test_rmse:.4f}
2. GradientBoosting 调优后 RMSE: {gb_test_rmse:.4f}
3. Stacking 集成 RMSE: {stack_test_rmse:.4f}
4. 集成方法通常能进一步提升性能
5. Top 特征: MedInc (收入中位数) 是最重要的预测因子
""")
