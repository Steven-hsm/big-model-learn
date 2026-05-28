### Day 4：逻辑回归 + 分类评估
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc, precision_recall_curve,
                             average_precision_score)
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 从零实现逻辑回归
# ============================================================

class LogisticRegression:
    """使用梯度下降实现的逻辑回归"""

    def __init__(self, lr=0.1, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.losses = []

    @staticmethod
    def _sigmoid(z):
        # 防止溢出
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _cross_entropy(self, y, y_pred):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for i in range(self.n_iters):
            z = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(z)

            loss = self._cross_entropy(y, y_pred)
            self.losses.append(loss)

            # 梯度
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# ============================================================
# 2. 生成二分类数据
# ============================================================
np.random.seed(42)
X, y = make_classification(n_samples=500, n_features=2, n_redundant=0,
                           n_informative=2, n_clusters_per_class=1,
                           class_sep=1.5, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print("=" * 60)
print("逻辑回归：从零实现 + 分类评估")
print("=" * 60)
print(f"训练集: {X_train.shape[0]} 样本")
print(f"测试集: {X_test.shape[0]} 样本")
print(f"类别分布 (训练): 0={sum(y_train==0)}, 1={sum(y_train==1)}")

# ============================================================
# 3. 训练模型
# ============================================================
model_scratch = LogisticRegression(lr=0.1, n_iters=1000)
model_scratch.fit(X_train_s, y_train)

y_pred_scratch = model_scratch.predict(X_test_s)
y_prob_scratch = model_scratch.predict_proba(X_test_s)

print(f"\n【自实现逻辑回归】")
print(f"  weights = {model_scratch.weights}")
print(f"  bias    = {model_scratch.bias:.4f}")
print(f"  准确率  = {np.mean(y_pred_scratch == y_test):.4f}")

# sklearn对比
model_sklearn = SklearnLR()
model_sklearn.fit(X_train_s, y_train)
y_pred_sklearn = model_sklearn.predict(X_test_s)
y_prob_sklearn = model_sklearn.predict_proba(X_test_s)[:, 1]
print(f"\n【sklearn LogisticRegression】")
print(f"  准确率  = {np.mean(y_pred_sklearn == y_test):.4f}")

# ============================================================
# 4. 可视化决策边界
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, model, title, is_scratch in [
    (axes[0], model_scratch, '自实现逻辑回归', True),
    (axes[1], model_sklearn, 'sklearn逻辑回归', False)
]:
    h = 0.05
    x_min, x_max = X_train_s[:, 0].min() - 1, X_train_s[:, 0].max() + 1
    y_min, y_max = X_train_s[:, 1].min() - 1, X_train_s[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]

    if is_scratch:
        Z = model.predict_proba(grid).reshape(xx.shape)
    else:
        Z = model.predict_proba(grid)[:, 1].reshape(xx.shape)

    ax.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.3)
    ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    ax.scatter(X_train_s[y_train == 0, 0], X_train_s[y_train == 0, 1],
               c='blue', alpha=0.5, s=20, label='类别 0')
    ax.scatter(X_train_s[y_train == 1, 0], X_train_s[y_train == 1, 1],
               c='red', alpha=0.5, s=20, label='类别 1')
    ax.set_xlabel('特征 1 (标准化)')
    ax.set_ylabel('特征 2 (标准化)')
    ax.set_title(title)
    ax.legend()

plt.tight_layout()
plt.show()

# ============================================================
# 5. 混淆矩阵 + 分类报告
# ============================================================
cm = confusion_matrix(y_test, y_pred_scratch)
print(f"\n混淆矩阵:")
print(f"  {cm}")
print(f"\n分类报告:")
print(classification_report(y_test, y_pred_scratch, digits=4))

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap='Blues')
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                fontsize=20, color='white' if cm[i, j] > cm.max()/2 else 'black')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['预测 0', '预测 1'])
ax.set_yticklabels(['实际 0', '实际 1'])
ax.set_title('混淆矩阵')
plt.colorbar(im)
plt.tight_layout()
plt.show()

# ============================================================
# 6. ROC曲线 + AUC
# ============================================================
fpr, tpr, thresholds_roc = roc_curve(y_test, y_prob_scratch)
roc_auc = auc(fpr, tpr)

# sklearn ROC
fpr_sk, tpr_sk, _ = roc_curve(y_test, y_prob_sklearn)
roc_auc_sk = auc(fpr_sk, tpr_sk)

print(f"\nAUC:")
print(f"  自实现: {roc_auc:.4f}")
print(f"  sklearn: {roc_auc_sk:.4f}")

# ============================================================
# 7. Precision-Recall曲线
# ============================================================
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_prob_scratch)
ap = average_precision_score(y_test, y_prob_scratch)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC曲线
ax1 = axes[0]
ax1.plot(fpr, tpr, 'b-', linewidth=2, label=f'自实现 (AUC={roc_auc:.3f})')
ax1.plot(fpr_sk, tpr_sk, 'g--', linewidth=2, label=f'sklearn (AUC={roc_auc_sk:.3f})')
ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC曲线')
ax1.legend()
ax1.grid(True, alpha=0.3)

# PR曲线
ax2 = axes[1]
ax2.plot(recall, precision, 'r-', linewidth=2, label=f'AP={ap:.3f}')
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('Precision-Recall曲线')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
