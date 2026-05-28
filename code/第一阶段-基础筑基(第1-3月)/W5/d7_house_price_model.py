### Day 7：房价预测（模型训练 + 调参 + 评估）
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 加载和准备数据
# ============================================================
print("=" * 60)
print("California Housing 房价预测 — 模型训练与评估")
print("=" * 60)

# 尝试加载d6保存的数据，若不存在则重新处理
save_dir = os.path.dirname(os.path.abspath(__file__))
npz_path = os.path.join(save_dir, 'california_housing_processed.npz')

if os.path.exists(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    feature_names = list(data['feature_names'])
    print("已加载d6保存的预处理数据")
else:
    print("未找到d6数据，重新加载处理...")
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    feature_names = list(housing.feature_names)

    df_raw = fetch_california_housing(as_frame=True).frame
    df_raw['RoomsPerHousehold'] = df_raw['AveRooms'] / df_raw['AveOccup']
    df_raw['BedroomsPerRoom'] = df_raw['AveBedrms'] / df_raw['AveRooms']
    df_raw['PopulationPerHousehold'] = df_raw['Population'] / df_raw['AveOccup']

    feature_cols = feature_names + ['RoomsPerHousehold', 'BedroomsPerRoom', 'PopulationPerHousehold']
    X = df_raw[feature_cols].values
    y = df_raw['MedHouseVal'].values
    feature_names = feature_cols

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

print(f"训练集: {X_train.shape}")
print(f"测试集: {X_test.shape}")
print(f"特征数: {len(feature_names)}")
print(f"特征: {feature_names}")

# ============================================================
# 2. 多模型对比 — 交叉验证
# ============================================================
print("\n" + "-" * 60)
print("2. 多模型交叉验证对比")
print("-" * 60)

models = {
    'LinearRegression': LinearRegression(),
    'Ridge(alpha=1)': Ridge(alpha=1),
    'Lasso(alpha=0.01)': Lasso(alpha=0.01, max_iter=10000),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
}

cv_results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5,
                             scoring='neg_mean_squared_error', n_jobs=-1)
    rmse_scores = np.sqrt(-scores)
    cv_results[name] = rmse_scores
    print(f"  {name:>20s}: RMSE = {rmse_scores.mean():.4f} (+/- {rmse_scores.std():.4f})")

# 可视化交叉验证结果
fig, ax = plt.subplots(figsize=(10, 5))
names = list(cv_results.keys())
means = [cv_results[n].mean() for n in names]
stds = [cv_results[n].std() for n in names]

bars = ax.bar(range(len(names)), means, yerr=stds, capsize=5,
              color=['steelblue', 'coral', 'green', 'purple'], alpha=0.8)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=15)
ax.set_ylabel('RMSE (交叉验证)')
ax.set_title('多模型交叉验证对比')
ax.grid(True, alpha=0.3, axis='y')
for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(i, m + s + 0.01, f'{m:.3f}', ha='center', fontsize=10)
plt.tight_layout()
plt.show()

# ============================================================
# 3. GridSearchCV — Ridge调参
# ============================================================
print("\n" + "-" * 60)
print("3. GridSearchCV Ridge alpha调参")
print("-" * 60)

ridge_params = {'alpha': [0.001, 0.01, 0.1, 1, 10, 50, 100, 500, 1000]}
ridge_grid = GridSearchCV(Ridge(max_iter=10000), ridge_params, cv=5,
                          scoring='neg_mean_squared_error', n_jobs=-1)
ridge_grid.fit(X_train, y_train)

print(f"最佳alpha: {ridge_grid.best_params_['alpha']}")
print(f"最佳CV RMSE: {np.sqrt(-ridge_grid.best_score_):.4f}")

# Ridge系数分析
best_ridge = ridge_grid.best_estimator_
print("\nRidge回归系数:")
for name, coef in sorted(zip(feature_names, best_ridge.coef_),
                          key=lambda x: abs(x[1]), reverse=True):
    print(f"  {name:>25s}: {coef:>8.4f}")

fig, ax = plt.subplots(figsize=(10, 5))
coefs_sorted = sorted(zip(feature_names, best_ridge.coef_), key=lambda x: x[1])
names_sorted, coefs_sorted_vals = zip(*coefs_sorted)
colors = ['coral' if c < 0 else 'steelblue' for c in coefs_sorted_vals]
ax.barh(range(len(names_sorted)), coefs_sorted_vals, color=colors)
ax.set_yticks(range(len(names_sorted)))
ax.set_yticklabels(names_sorted)
ax.set_xlabel('系数值')
ax.set_title(f'Ridge回归系数 (alpha={ridge_grid.best_params_["alpha"]})')
ax.axvline(0, color='k', linewidth=0.5)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

# ============================================================
# 4. 最终模型训练 + 评估
# ============================================================
print("\n" + "-" * 60)
print("4. 最终模型评估 (在测试集上)")
print("-" * 60)

final_models = {
    'LinearRegression': LinearRegression(),
    'Ridge(best)': ridge_grid.best_estimator_,
    'Lasso(0.01)': Lasso(alpha=0.01, max_iter=10000),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
}

test_results = {}
for name, model in final_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    test_results[name] = {'y_pred': y_pred, 'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}
    print(f"  {name:>18s}: RMSE={rmse:.4f}, MAE={mae:.4f}, R^2={r2:.4f}")

# ============================================================
# 5. 可视化: 实际 vs 预测
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

for ax, (name, res) in zip(axes.flatten(), test_results.items()):
    y_pred = res['y_pred']
    ax.scatter(y_test, y_pred, alpha=0.3, s=10)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
            'r--', linewidth=2)
    ax.set_xlabel('实际值')
    ax.set_ylabel('预测值')
    ax.set_title(f'{name}\nRMSE={res["rmse"]:.3f}, R^2={res["r2"]:.3f}')
    ax.grid(True, alpha=0.3)

plt.suptitle('实际值 vs 预测值', fontsize=14)
plt.tight_layout()
plt.show()

# ============================================================
# 6. 残差分布
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, (name, res) in zip(axes.flatten(), test_results.items()):
    residuals = y_test - res['y_pred']
    ax.hist(residuals, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(0, color='red', linestyle='--')
    ax.set_xlabel('残差 (实际 - 预测)')
    ax.set_ylabel('频数')
    ax.set_title(f'{name} 残差分布\n均值={residuals.mean():.4f}, 标准差={residuals.std():.4f}')

plt.suptitle('残差分布对比', fontsize=14)
plt.tight_layout()
plt.show()

# ============================================================
# 7. 特征重要性 (Random Forest)
# ============================================================
print("\n" + "-" * 60)
print("5. RandomForest特征重要性")
print("-" * 60)

rf = final_models['RandomForest']
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

print("特征重要性排序:")
for i, idx in enumerate(indices):
    print(f"  {i+1:>2d}. {feature_names[idx]:>25s}: {importances[idx]:.4f}")

fig, ax = plt.subplots(figsize=(10, 6))
sorted_idx = np.argsort(importances)
ax.barh(range(len(feature_names)), importances[sorted_idx], color='steelblue')
ax.set_yticks(range(len(feature_names)))
ax.set_yticklabels([feature_names[i] for i in sorted_idx])
ax.set_xlabel('特征重要性')
ax.set_title('Random Forest 特征重要性')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

# ============================================================
# 8. 最佳模型详细分析
# ============================================================
best_model_name = min(test_results, key=lambda k: test_results[k]['rmse'])
print("\n" + "=" * 60)
print(f"最佳模型: {best_model_name}")
print(f"  RMSE: {test_results[best_model_name]['rmse']:.4f}")
print(f"  MAE:  {test_results[best_model_name]['mae']:.4f}")
print(f"  R^2:  {test_results[best_model_name]['r2']:.4f}")
print("=" * 60)
