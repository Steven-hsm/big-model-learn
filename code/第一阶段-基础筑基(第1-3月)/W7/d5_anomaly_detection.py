### Day 5（周五）：异常检测
# Z-Score、IQR、IsolationForest、LOF 方法对比

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 生成带异常值的数据
# ============================================================
print("=== 生成带异常值的数据 ===\n")

np.random.seed(42)

# 正常数据: 两个簇
X_normal = np.vstack([
    np.random.multivariate_normal([2, 2], [[1, 0.3], [0.3, 1]], 300),
    np.random.multivariate_normal([-2, -2], [[1.5, 0.5], [0.5, 1.5]], 200),
])

# 异常数据: 随机散布
n_outliers = 30
X_outliers = np.random.uniform(-7, 7, size=(n_outliers, 2))

X_all = np.vstack([X_normal, X_outliers])
y_true = np.hstack([np.zeros(len(X_normal)), np.ones(n_outliers)])  # 0=正常, 1=异常

print(f"正常样本: {len(X_normal)}")
print(f"异常样本: {n_outliers}")
print(f"总样本数: {len(X_all)}")


# ============================================================
# 2. Z-Score 方法
# ============================================================
print("\n=== Z-Score 异常检测 ===\n")


def zscore_detect(X, threshold=3.0):
    """基于Z-Score的异常检测"""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    z_scores = np.abs((X - mean) / std)
    # 任一特征的Z-Score超过阈值即为异常
    outliers = np.any(z_scores > threshold, axis=1)
    return outliers.astype(int), z_scores


zscore_labels, z_scores = zscore_detect(X_all, threshold=3.0)
n_detected = zscore_labels.sum()
print(f"Z-Score检测到的异常数: {n_detected}")

# 真阳性/假阳性
tp = np.sum((zscore_labels == 1) & (y_true == 1))
fp = np.sum((zscore_labels == 1) & (y_true == 0))
fn = np.sum((zscore_labels == 0) & (y_true == 1))
tn = np.sum((zscore_labels == 0) & (y_true == 0))
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
print(f"  精确率: {precision:.3f}, 召回率: {recall:.3f}")


# ============================================================
# 3. IQR 方法
# ============================================================
print("\n=== IQR 异常检测 ===\n")


def iqr_detect(X, k=1.5):
    """基于IQR的异常检测"""
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    outliers = np.any((X < lower) | (X > upper), axis=1)
    return outliers.astype(int), (lower, upper)


iqr_labels, (iqr_lower, iqr_upper) = iqr_detect(X_all, k=1.5)
n_detected_iqr = iqr_labels.sum()
print(f"IQR检测到的异常数: {n_detected_iqr}")

tp_iqr = np.sum((iqr_labels == 1) & (y_true == 1))
fp_iqr = np.sum((iqr_labels == 1) & (y_true == 0))
fn_iqr = np.sum((iqr_labels == 0) & (y_true == 1))
tn_iqr = np.sum((iqr_labels == 0) & (y_true == 0))
precision_iqr = tp_iqr / (tp_iqr + fp_iqr) if (tp_iqr + fp_iqr) > 0 else 0
recall_iqr = tp_iqr / (tp_iqr + fn_iqr) if (tp_iqr + fn_iqr) > 0 else 0
print(f"  精确率: {precision_iqr:.3f}, 召回率: {recall_iqr:.3f}")


# ============================================================
# 4. IsolationForest
# ============================================================
print("\n=== IsolationForest 异常检测 ===\n")

iso_forest = IsolationForest(
    n_estimators=100,
    contamination=float(n_outliers) / len(X_all),
    random_state=42
)
iso_labels = iso_forest.fit_predict(X_all)
# IsolationForest: 1=正常, -1=异常 → 转换为 0=正常, 1=异常
iso_labels = (iso_labels == -1).astype(int)

n_detected_iso = iso_labels.sum()
print(f"IsolationForest检测到的异常数: {n_detected_iso}")

tp_iso = np.sum((iso_labels == 1) & (y_true == 1))
fp_iso = np.sum((iso_labels == 1) & (y_true == 0))
fn_iso = np.sum((iso_labels == 0) & (y_true == 1))
tn_iso = np.sum((iso_labels == 0) & (y_true == 0))
precision_iso = tp_iso / (tp_iso + fp_iso) if (tp_iso + fp_iso) > 0 else 0
recall_iso = tp_iso / (tp_iso + fn_iso) if (tp_iso + fn_iso) > 0 else 0
print(f"  精确率: {precision_iso:.3f}, 召回率: {recall_iso:.3f}")


# ============================================================
# 5. LocalOutlierFactor (LOF)
# ============================================================
print("\n=== LocalOutlierFactor 异常检测 ===\n")

lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=float(n_outliers) / len(X_all)
)
lof_labels = lof.fit_predict(X_all)
lof_labels = (lof_labels == -1).astype(int)

n_detected_lof = lof_labels.sum()
print(f"LOF检测到的异常数: {n_detected_lof}")

tp_lof = np.sum((lof_labels == 1) & (y_true == 1))
fp_lof = np.sum((lof_labels == 1) & (y_true == 0))
fn_lof = np.sum((lof_labels == 0) & (y_true == 1))
tn_lof = np.sum((lof_labels == 0) & (y_true == 0))
precision_lof = tp_lof / (tp_lof + fp_lof) if (tp_lof + fp_lof) > 0 else 0
recall_lof = tp_lof / (tp_lof + fn_lof) if (tp_lof + fn_lof) > 0 else 0
print(f"  精确率: {precision_lof:.3f}, 召回率: {recall_lof:.3f}")


# ============================================================
# 6. 可视化对比
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

methods = [
    ('真实标签', y_true),
    ('Z-Score', zscore_labels),
    ('IQR', iqr_labels),
    ('IsolationForest', iso_labels),
    ('LOF', lof_labels),
]

for idx, (name, labels) in enumerate(methods):
    row, col = idx // 3, idx % 3
    ax = axes[row, col]

    # 正常点
    normal_mask = labels == 0
    ax.scatter(X_all[normal_mask, 0], X_all[normal_mask, 1],
               c='steelblue', s=15, alpha=0.5, label='正常')

    # 异常点
    outlier_mask = labels == 1
    ax.scatter(X_all[outlier_mask, 0], X_all[outlier_mask, 1],
               c='red', s=40, alpha=0.8, marker='x', label='异常')

    n_det = labels.sum()
    ax.set_title(f'{name} (检测={n_det})')
    ax.legend(fontsize=8)

# 隐藏多余的子图
axes[1, 2].axis('off')

plt.suptitle('异常检测方法对比', fontsize=14)
plt.tight_layout()
plt.show()


# ============================================================
# 7. 方法对比表格
# ============================================================
print("\n" + "=" * 65)
print("异常检测方法对比汇总表")
print("=" * 65)
print(f"{'方法':<20} {'检测数':>6} {'精确率':>8} {'召回率':>8} {'F1':>8}")
print("-" * 65)

methods_stats = [
    ('Z-Score', n_detected, precision, recall),
    ('IQR', n_detected_iqr, precision_iqr, recall_iqr),
    ('IsolationForest', n_detected_iso, precision_iso, recall_iso),
    ('LOF', n_detected_lof, precision_lof, recall_lof),
]

for name, n_det, prec, rec in methods_stats:
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    print(f"{name:<20} {n_det:>6} {prec:>8.3f} {rec:>8.3f} {f1:>8.3f}")

print("-" * 65)
print(f"{'真实异常数':<20} {n_outliers:>6}")

print("\n=== Day 5 完成: 异常检测 ===")
