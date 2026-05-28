### Day 4（周四）：t-SNE / UMAP可视化 + sklearn Pipeline
# 降维可视化对比、Pipeline构建、自定义Transformer、GridSearchCV

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. t-SNE可视化 (digits数据集)
# ============================================================
print("=== t-SNE 可视化 ===\n")

digits = load_digits()
X_digits = digits.data
y_digits = digits.target

print(f"数据形状: {X_digits.shape}")
print(f"类别数: {len(np.unique(y_digits))}")

# 先用PCA降维到30维（t-SNE的常见预处理）
pca_pre = PCA(n_components=30)
X_pca_pre = pca_pre.fit_transform(X_digits)

# t-SNE降维到2D
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_pca_pre)

# PCA降维到2D作为对比
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_digits)

print(f"t-SNE结果形状: {X_tsne.shape}")
print(f"PCA 2D结果形状: {X_pca_2d.shape}")


# ============================================================
# 2. PCA vs t-SNE 可视化对比
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# PCA可视化
scatter1 = axes[0].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y_digits,
                           cmap='tab10', s=8, alpha=0.6)
axes[0].set_title('PCA 2D可视化')
axes[0].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
axes[0].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
plt.colorbar(scatter1, ax=axes[0], label='数字类别')

# t-SNE可视化
scatter2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_digits,
                           cmap='tab10', s=8, alpha=0.6)
axes[1].set_title('t-SNE 2D可视化')
axes[1].set_xlabel('t-SNE维度1')
axes[1].set_ylabel('t-SNE维度2')
plt.colorbar(scatter2, ax=axes[1], label='数字类别')

plt.suptitle('PCA vs t-SNE: 手写数字可视化', fontsize=14)
plt.tight_layout()
plt.show()


# ============================================================
# 3. 不同perplexity值对比
# ============================================================
print("\n=== 不同 perplexity 值对比 ===\n")

perplexities = [5, 15, 30, 50]

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for idx, perp in enumerate(perplexities):
    tsne_p = TSNE(n_components=2, random_state=42, perplexity=perp)
    X_t = tsne_p.fit_transform(X_pca_pre)
    axes[idx].scatter(X_t[:, 0], X_t[:, 1], c=y_digits, cmap='tab10', s=8, alpha=0.6)
    axes[idx].set_title(f'perplexity={perp}')
    axes[idx].set_xticks([])
    axes[idx].set_yticks([])

plt.suptitle('t-SNE: 不同perplexity值对比', fontsize=14)
plt.tight_layout()
plt.show()


# ============================================================
# 4. UMAP (try/except, fallback to PCA)
# ============================================================
print("\n=== UMAP 可视化 ===\n")

try:
    from umap import UMAP
    umap_model = UMAP(n_components=2, random_state=42)
    X_umap = umap_model.fit_transform(X_digits)
    umap_available = True
    print("UMAP可用，已完成降维")
except ImportError:
    print("UMAP未安装，使用PCA作为替代")
    umap_available = False
    X_umap = X_pca_2d

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

axes[0].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y_digits, cmap='tab10', s=8, alpha=0.6)
axes[0].set_title('PCA')

axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_digits, cmap='tab10', s=8, alpha=0.6)
axes[1].set_title('t-SNE')

axes[2].scatter(X_umap[:, 0], X_umap[:, 1], c=y_digits, cmap='tab10', s=8, alpha=0.6)
axes[2].set_title('UMAP' if umap_available else 'PCA (UMAP替代)')

plt.suptitle('降维方法对比: PCA / t-SNE / UMAP', fontsize=14)
plt.tight_layout()
plt.show()


# ============================================================
# 5. sklearn Pipeline: ColumnTransformer + Pipeline
# ============================================================
print("\n=== sklearn Pipeline 构建 ===\n")

# 构造混合类型数据（模拟真实场景）
np.random.seed(42)
n_samples = 500
numeric_features = np.random.randn(n_samples, 3)
categorical_features = np.random.choice(['A', 'B', 'C'], size=(n_samples, 2))
y_class = (numeric_features[:, 0] + numeric_features[:, 1] > 0).astype(int)

print(f"数值特征形状: {numeric_features.shape}")
print(f"类别特征形状: {categorical_features.shape}")
print(f"目标类别分布: {np.bincount(y_class)}")

# 定义ColumnTransformer
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, [0, 1, 2]),           # 数值列
        ('cat', categorical_transformer, [3, 4])           # 类别列
    ]
)

# 合并特征
X_combined = np.hstack([numeric_features, categorical_features])

# 完整Pipeline
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

# 训练
full_pipeline.fit(X_combined, y_class)
score = full_pipeline.score(X_combined, y_class)
print(f"\nPipeline训练准确率: {score:.4f}")


# ============================================================
# 6. 自定义Transformer
# ============================================================
print("\n=== 自定义Transformer ===\n")


class LogTransformer(BaseEstimator, TransformerMixin):
    """对数变换Transformer: log(1 + X)"""

    def __init__(self, base='e'):
        self.base = base

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.array(X, dtype=float)
        X_transformed = np.log1p(np.abs(X))
        if self.base == '10':
            X_transformed = np.log10(1 + np.abs(X))
        elif self.base == '2':
            X_transformed = np.log2(1 + np.abs(X))
        return X_transformed

    def get_feature_names_out(self, input_features=None):
        return input_features


# 测试自定义Transformer
log_trans = LogTransformer(base='e')
test_data = np.array([[1, 10], [100, 1000], [0, -5]])
transformed = log_trans.transform(test_data)
print("原始数据:")
print(test_data)
print("\nLog变换后:")
print(transformed)


# ============================================================
# 7. Pipeline + GridSearchCV
# ============================================================
print("\n=== Pipeline + GridSearchCV ===\n")

# 构建可调参的Pipeline
pipeline_cv = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('clf', LogisticRegression(random_state=42, max_iter=1000))
])

# 参数网格
param_grid = {
    'pca__n_components': [2, 3],
    'clf__C': [0.01, 0.1, 1, 10],
    'clf__penalty': ['l2']
}

# GridSearchCV
grid_search = GridSearchCV(
    pipeline_cv,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)

grid_search.fit(numeric_features, y_class)

print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳CV准确率: {grid_search.best_score_:.4f}")
print("\n所有结果:")
means = grid_search.cv_results_['mean_test_score']
params = grid_search.cv_results_['params']
for mean, param in zip(means, params):
    print(f"  {param} → 准确率={mean:.4f}")

print("\n=== Day 4 完成: t-SNE / UMAP可视化 + sklearn Pipeline ===")
