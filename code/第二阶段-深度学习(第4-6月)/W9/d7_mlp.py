"""
d7_mlp.py - 完整3层MLP实现
============================
- 完整 MLP 类: forward, backward, train
- 在 make_moons 数据集上训练
- Dropout (Inverted Dropout)
- BatchNorm 和 LayerNorm 实现
- 可视化：决策边界、训练损失曲线
- 早停实现
- 正则化效果对比（有/无 Dropout）
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 激活函数
# ============================================================
def relu(x):
    return np.maximum(0, x)


def relu_deriv(x):
    return (x > 0).astype(float)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)


# ============================================================
# 2. 损失函数
# ============================================================
def bce_loss(y_true, y_pred, eps=1e-7):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def bce_loss_grad(y_true, y_pred, eps=1e-7):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return (-y_true / y_pred + (1 - y_true) / (1 - y_pred)) / y_true.shape[0]


# ============================================================
# 3. BatchNorm 和 LayerNorm
# ============================================================
class BatchNorm:
    """批归一化 (Batch Normalization)"""

    def __init__(self, n_features, momentum=0.9, eps=1e-5):
        self.gamma = np.ones((1, n_features))    # 缩放参数
        self.beta = np.zeros((1, n_features))    # 偏移参数
        self.running_mean = np.zeros((1, n_features))
        self.running_var = np.ones((1, n_features))
        self.momentum = momentum
        self.eps = eps
        # 缓存用于反向传播
        self.cache = None
        # 梯度
        self.dgamma = None
        self.dbeta = None

    def forward(self, Z, training=True):
        if training:
            mean = np.mean(Z, axis=0, keepdims=True)
            var = np.var(Z, axis=0, keepdims=True)
            Z_norm = (Z - mean) / np.sqrt(var + self.eps)
            out = self.gamma * Z_norm + self.beta

            # 更新运行统计量
            self.running_mean = (self.momentum * self.running_mean +
                                 (1 - self.momentum) * mean)
            self.running_var = (self.momentum * self.running_var +
                                (1 - self.momentum) * var)

            # 缓存
            self.cache = (Z, Z_norm, mean, var)
            return out
        else:
            Z_norm = (Z - self.running_mean) / np.sqrt(self.running_var + self.eps)
            return self.gamma * Z_norm + self.beta

    def backward(self, dout):
        Z, Z_norm, mean, var = self.cache
        N = Z.shape[0]

        self.dgamma = np.sum(dout * Z_norm, axis=0, keepdims=True)
        self.dbeta = np.sum(dout, axis=0, keepdims=True)

        dZ_norm = dout * self.gamma
        dvar = np.sum(dZ_norm * (Z - mean) * -0.5 * (var + self.eps) ** (-1.5),
                      axis=0, keepdims=True)
        dmean = np.sum(dZ_norm * -1 / np.sqrt(var + self.eps), axis=0, keepdims=True) + \
                dvar * np.mean(-2 * (Z - mean), axis=0, keepdims=True)
        dZ = dZ_norm / np.sqrt(var + self.eps) + dvar * 2 * (Z - mean) / N + dmean / N
        return dZ


class LayerNorm:
    """层归一化 (Layer Normalization)"""

    def __init__(self, n_features, eps=1e-5):
        self.gamma = np.ones((1, n_features))
        self.beta = np.zeros((1, n_features))
        self.eps = eps
        self.cache = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, Z):
        mean = np.mean(Z, axis=1, keepdims=True)
        var = np.var(Z, axis=1, keepdims=True)
        Z_norm = (Z - mean) / np.sqrt(var + self.eps)
        out = self.gamma * Z_norm + self.beta
        self.cache = (Z, Z_norm, mean, var)
        return out

    def backward(self, dout):
        Z, Z_norm, mean, var = self.cache
        d = Z.shape[1]

        self.dgamma = np.sum(dout * Z_norm, axis=0, keepdims=True)
        self.dbeta = np.sum(dout, axis=0, keepdims=True)

        dZ_norm = dout * self.gamma
        dvar = np.sum(dZ_norm * (Z - mean) * -0.5 * (var + self.eps) ** (-1.5),
                      axis=1, keepdims=True)
        dmean = np.sum(dZ_norm * -1 / np.sqrt(var + self.eps), axis=1, keepdims=True) + \
                dvar * np.mean(-2 * (Z - mean), axis=1, keepdims=True)
        dZ = dZ_norm / np.sqrt(var + self.eps) + dvar * 2 * (Z - mean) / d + dmean / d
        return dZ


# ============================================================
# 4. 完整 MLP 类
# ============================================================
class MLP:
    """
    多层感知机
    支持: Dropout, BatchNorm/LayerNorm, 早停
    """

    def __init__(self, layer_dims, activation='relu', dropout_rate=0.0,
                 use_batchnorm=False, use_layernorm=False):
        """
        参数:
          layer_dims: 各层维度列表, 如 [2, 16, 16, 1]
          activation: 隐藏层激活函数 ('relu' 或 'sigmoid')
          dropout_rate: dropout 比率
          use_batchnorm: 是否使用批归一化
          use_layernorm: 是否使用层归一化
        """
        self.layer_dims = layer_dims
        self.n_layers = len(layer_dims) - 1
        self.dropout_rate = dropout_rate
        self.activation = activation

        # 初始化权重 (He 初始化)
        self.weights = {}
        for i in range(1, self.n_layers + 1):
            self.weights[f'W{i}'] = (np.random.randn(layer_dims[i-1], layer_dims[i])
                                      * np.sqrt(2.0 / layer_dims[i-1]))
            self.weights[f'b{i}'] = np.zeros((1, layer_dims[i]))

        # 初始化 BatchNorm / LayerNorm
        self.use_batchnorm = use_batchnorm
        self.use_layernorm = use_layernorm
        self.norms = {}
        if use_batchnorm or use_layernorm:
            for i in range(1, self.n_layers):  # 不对输出层做归一化
                if use_batchnorm:
                    self.norms[i] = BatchNorm(layer_dims[i])
                elif use_layernorm:
                    self.norms[i] = LayerNorm(layer_dims[i])

        # 训练状态
        self.cache = {}
        self.training = True

    def _activate(self, Z, layer_idx):
        """激活函数"""
        if layer_idx < self.n_layers:
            if self.activation == 'relu':
                return relu(Z)
            else:
                return sigmoid(Z)
        else:
            # 输出层用 sigmoid
            return sigmoid(Z)

    def _activate_deriv(self, Z, layer_idx):
        """激活函数导数"""
        if layer_idx < self.n_layers:
            if self.activation == 'relu':
                return relu_deriv(Z)
            else:
                return sigmoid_deriv(Z)
        else:
            return sigmoid_deriv(Z)

    def forward(self, X):
        """前向传播"""
        A = X
        self.cache['A0'] = A

        for i in range(1, self.n_layers + 1):
            # 线性变换
            Z = A @ self.weights[f'W{i}'] + self.weights[f'b{i}']
            self.cache[f'Z{i}'] = Z

            # 归一化 (隐藏层)
            if i < self.n_layers and i in self.norms:
                if self.use_batchnorm:
                    Z = self.norms[i].forward(Z, training=self.training)
                else:
                    Z = self.norms[i].forward(Z)
                self.cache[f'Z_norm{i}'] = Z

            # 激活函数
            A = self._activate(Z, i)
            self.cache[f'A{i}'] = A

            # Dropout (隐藏层)
            if i < self.n_layers and self.dropout_rate > 0 and self.training:
                mask = (np.random.rand(*A.shape) > self.dropout_rate).astype(float)
                A = A * mask / (1 - self.dropout_rate)  # Inverted Dropout
                self.cache[f'drop_mask{i}'] = mask

        return A

    def backward(self, y_true, y_pred, lr=0.01, l2_lambda=0.0):
        """反向传播"""
        m = y_true.shape[0]
        grads = {}

        # 输出层梯度
        dA = bce_loss_grad(y_true, y_pred)
        dZ = dA * self._activate_deriv(self.cache[f'Z{self.n_layers}'], self.n_layers)

        # 逐层反向传播
        for i in range(self.n_layers, 0, -1):
            A_prev = self.cache[f'A{i-1}']

            # 计算权重梯度
            grads[f'dW{i}'] = A_prev.T @ dZ / m
            grads[f'db{i}'] = np.sum(dZ, axis=0, keepdims=True) / m

            # L2 正则化
            if l2_lambda > 0:
                grads[f'dW{i}'] += l2_lambda * self.weights[f'W{i}'] / m

            if i > 1:
                # 传播梯度到前一层
                dA = dZ @ self.weights[f'W{i}'].T

                # Dropout 反向传播
                if i - 1 < self.n_layers and self.dropout_rate > 0:
                    if f'drop_mask{i-1}' in self.cache:
                        dA = dA * self.cache[f'drop_mask{i-1}'] / (1 - self.dropout_rate)

                # BatchNorm / LayerNorm 反向传播
                if (i - 1) in self.norms:
                    dA = self.norms[i-1].backward(dA)

                # 激活函数反向传播
                dZ = dA * self._activate_deriv(self.cache[f'Z{i-1}'], i - 1)

        # 更新参数
        for i in range(1, self.n_layers + 1):
            self.weights[f'W{i}'] -= lr * grads[f'dW{i}']
            self.weights[f'b{i}'] -= lr * grads[f'db{i}']

        # 更新 BatchNorm/LayerNorm 参数
        for i, norm in self.norms.items():
            if hasattr(norm, 'dgamma') and norm.dgamma is not None:
                norm.gamma -= lr * norm.dgamma
                norm.beta -= lr * norm.dbeta

    def predict(self, X):
        """预测（关闭训练模式）"""
        self.training = False
        pred = self.forward(X)
        self.training = True
        return pred

    def train(self, X_train, y_train, X_val=None, y_val=None,
              lr=0.01, epochs=1000, batch_size=32, l2_lambda=0.0,
              patience=20, verbose=True):
        """
        训练 MLP
        支持小批量训练和早停
        """
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_weights = None
        patience_counter = 0

        for epoch in range(epochs):
            # 小批量训练
            indices = np.random.permutation(X_train.shape[0])
            epoch_loss = 0
            n_batches = 0

            for start in range(0, X_train.shape[0], batch_size):
                end = min(start + batch_size, X_train.shape[0])
                batch_idx = indices[start:end]
                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]

                # 前向传播
                self.training = True
                y_pred = self.forward(X_batch)
                batch_loss = bce_loss(y_batch, y_pred)

                # 反向传播 + 更新参数
                self.backward(y_batch, y_pred, lr=lr, l2_lambda=l2_lambda)

                epoch_loss += batch_loss
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            train_losses.append(avg_loss)

            # 验证集评估
            if X_val is not None:
                val_pred = self.predict(X_val)
                v_loss = bce_loss(y_val, val_pred)
                val_losses.append(v_loss)

                # 早停检查
                if v_loss < best_val_loss:
                    best_val_loss = v_loss
                    best_weights = {k: v.copy() for k, v in self.weights.items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    if verbose:
                        print(f"  早停! 在第 {epoch+1} 轮停止 "
                              f"(patience={patience})")
                    # 恢复最佳权重
                    self.weights = best_weights
                    break

            if verbose and (epoch + 1) % 100 == 0:
                train_acc = self._accuracy(X_train, y_train)
                msg = f"  Epoch {epoch+1:4d}: train_loss={avg_loss:.4f}"
                if X_val is not None:
                    val_acc = self._accuracy(X_val, y_val)
                    msg += f", val_loss={v_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}"
                else:
                    msg += f", train_acc={train_acc:.4f}"
                print(msg)

        return train_losses, val_losses

    def _accuracy(self, X, y):
        """计算准确率"""
        pred = self.predict(X)
        pred_labels = (pred >= 0.5).astype(int)
        return np.mean(pred_labels == y)


# ============================================================
# 5. 生成数据
# ============================================================
print("=" * 60)
print("生成 make_moons 数据集")
print("=" * 60)

X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 调整形状
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

print(f"训练集: {X_train.shape[0]} 样本")
print(f"测试集: {X_test.shape[0]} 样本")
print(f"特征维度: {X_train.shape[1]}")


# ============================================================
# 6. 训练基础 MLP
# ============================================================
print("\n" + "=" * 60)
print("训练基础 MLP (无 Dropout)")
print("=" * 60)

np.random.seed(42)
mlp_basic = MLP([2, 32, 16, 1], activation='relu', dropout_rate=0.0)
train_loss_basic, val_loss_basic = mlp_basic.train(
    X_train, y_train, X_test, y_test,
    lr=0.01, epochs=1000, batch_size=32,
    patience=30, verbose=True
)

basic_acc = mlp_basic._accuracy(X_test, y_test)
print(f"\n基础 MLP 测试准确率: {basic_acc:.4f}")


# ============================================================
# 7. 训练带 Dropout 的 MLP
# ============================================================
print("\n" + "=" * 60)
print("训练带 Dropout 的 MLP (dropout_rate=0.3)")
print("=" * 60)

np.random.seed(42)
mlp_dropout = MLP([2, 32, 16, 1], activation='relu', dropout_rate=0.3)
train_loss_dropout, val_loss_dropout = mlp_dropout.train(
    X_train, y_train, X_test, y_test,
    lr=0.01, epochs=1000, batch_size=32,
    patience=30, verbose=True
)

dropout_acc = mlp_dropout._accuracy(X_test, y_test)
print(f"\nDropout MLP 测试准确率: {dropout_acc:.4f}")


# ============================================================
# 8. 可视化决策边界
# ============================================================
def plot_decision_boundary(mlp, X, y, title, ax):
    """绘制决策边界"""
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = mlp.predict(grid).reshape(xx.shape)

    ax.contourf(xx, yy, probs, levels=20, cmap='RdYlBu', alpha=0.7)
    ax.contour(xx, yy, probs, levels=[0.5], colors='black', linewidths=2)

    for cls, marker, color in [(0, 'o', 'red'), (1, 's', 'blue')]:
        mask = (y.flatten() == cls)
        ax.scatter(X[mask, 0], X[mask, 1], c=color, marker=marker,
                   s=30, edgecolors='white', linewidth=0.5, label=f'类别 {cls}',
                   zorder=5, alpha=0.6)

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=9)


fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# 左: 训练数据
for cls, marker, color in [(0, 'o', 'red'), (1, 's', 'blue')]:
    mask = (y == cls)
    axes[0].scatter(X[mask, 0], X[mask, 1], c=color, marker=marker,
                    s=30, edgecolors='white', linewidth=0.5, alpha=0.6, label=f'类别 {cls}')
axes[0].set_xlabel('$x_1$')
axes[0].set_ylabel('$x_2$')
axes[0].set_title('make_moons 原始数据')
axes[0].legend()

# 中: 基础 MLP
plot_decision_boundary(mlp_basic, X, y, f'基础 MLP (acc={basic_acc:.3f})', axes[1])

# 右: Dropout MLP
plot_decision_boundary(mlp_dropout, X, y, f'Dropout MLP (acc={dropout_acc:.3f})', axes[2])

plt.suptitle("MLP 决策边界对比", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("d7_mlp_decision_boundary.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n图1保存完成: d7_mlp_decision_boundary.png")


# ============================================================
# 9. 训练损失曲线
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 训练损失
axes[0].plot(train_loss_basic, 'b-', linewidth=1.5, label='基础 MLP')
axes[0].plot(train_loss_dropout, 'r-', linewidth=1.5, label='Dropout MLP')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('训练损失 (BCE)')
axes[0].set_title('训练损失曲线')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 验证损失
if val_loss_basic:
    axes[1].plot(val_loss_basic, 'b-', linewidth=1.5, label='基础 MLP')
if val_loss_dropout:
    axes[1].plot(val_loss_dropout, 'r-', linewidth=1.5, label='Dropout MLP')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('验证损失 (BCE)')
axes[1].set_title('验证损失曲线')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle("训练过程对比（基础 vs Dropout）", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("d7_mlp_loss_curves.png", dpi=150, bbox_inches='tight')
plt.show()
print("图2保存完成: d7_mlp_loss_curves.png")


# ============================================================
# 10. BatchNorm 和 LayerNorm 演示
# ============================================================
print("\n" + "=" * 60)
print("BatchNorm 和 LayerNorm 演示")
print("=" * 60)

np.random.seed(42)
X_demo = np.random.randn(4, 3) * 2 + 1  # 4个样本，3个特征
print(f"输入数据 (4样本, 3特征):\n{X_demo}")

# BatchNorm
bn = BatchNorm(3)
bn_out = bn.forward(X_demo, training=True)
print(f"\nBatchNorm 输出 (按特征归一化):\n{bn_out}")
print(f"  每列均值: {bn_out.mean(axis=0).round(6)}")
print(f"  每列方差: {bn_out.var(axis=0).round(6)}")

# LayerNorm
ln = LayerNorm(3)
ln_out = ln.forward(X_demo)
print(f"\nLayerNorm 输出 (按样本归一化):\n{ln_out}")
print(f"  每行均值: {ln_out.mean(axis=1).round(6)}")
print(f"  每行方差: {ln_out.var(axis=1).round(6)}")


# ============================================================
# 11. 正则化效果对比总结
# ============================================================
print("\n" + "=" * 60)
print("正则化效果对比")
print("=" * 60)

configs = [
    ("无正则化", {"dropout_rate": 0.0}),
    ("Dropout=0.2", {"dropout_rate": 0.2}),
    ("Dropout=0.5", {"dropout_rate": 0.5}),
]

results_table = []
for name, kwargs in configs:
    np.random.seed(42)
    mlp = MLP([2, 64, 32, 1], activation='relu', **kwargs)
    t_loss, v_loss = mlp.train(
        X_train, y_train, X_test, y_test,
        lr=0.01, epochs=500, batch_size=32,
        patience=20, verbose=False
    )
    acc = mlp._accuracy(X_test, y_test)
    final_v = v_loss[-1] if v_loss else 0
    results_table.append((name, acc, t_loss[-1], final_v))
    print(f"  {name:<15}: 测试准确率={acc:.4f}, "
          f"训练损失={t_loss[-1]:.4f}, 验证损失={final_v:.4f}")


# ============================================================
# 12. Dropout 效果可视化
# ============================================================
fig, axes = plt.subplots(1, len(configs), figsize=(6 * len(configs), 5))

for idx, (name, kwargs) in enumerate(configs):
    np.random.seed(42)
    mlp = MLP([2, 64, 32, 1], activation='relu', **kwargs)
    mlp.train(X_train, y_train, X_test, y_test,
              lr=0.01, epochs=500, batch_size=32,
              patience=20, verbose=False)
    plot_decision_boundary(mlp, X, y, name, axes[idx])

plt.suptitle("不同 Dropout 比率的决策边界", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("d7_mlp_dropout_comparison.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n图3保存完成: d7_mlp_dropout_comparison.png")

print("\n" + "=" * 60)
print("总结:")
print("  - 完整 MLP 包含: 前向传播、反向传播、小批量训练")
print("  - Dropout: 训练时随机丢弃神经元，防止过拟合 (Inverted Dropout)")
print("  - BatchNorm: 按特征归一化，加速训练，每列均值0方差1")
print("  - LayerNorm: 按样本归一化，适用于 Transformer 等序列模型")
print("  - 早停: 监控验证损失，在过拟合前停止训练")
print("  - 适当的 Dropout 可以提高泛化能力")
print("=" * 60)
