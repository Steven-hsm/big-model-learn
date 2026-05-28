"""
W16-D3 高级正则化技术 (Advanced Regularization)
================================================
Label Smoothing, Dropout 变体 (DropPath, SpatialDropout),
Stochastic Depth, Weight Decay 对比, 展示对过拟合的影响
"""

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W16-D3 高级正则化技术 (Advanced Regularization)")
print("=" * 60)

# ============================================================
# 1. 正则化概述
# ============================================================
print("\n--- 1. 正则化概述 ---")
print("""
  过拟合的表现: 训练 loss 低, 验证 loss 高

  正则化方法总结:
    基础:
      - L1/L2 正则化 (Weight Decay)
      - Dropout
      - Early Stopping
      - Batch Normalization

    高级:
      - Label Smoothing (标签平滑)
      - DropPath / Stochastic Depth (随机深度)
      - SpatialDropout (空间 Dropout)
      - Mixup / CutMix (数据层面)
      - Dropout variants (DropBlock, Zoneout)
""")

# ============================================================
# 2. Label Smoothing (标签平滑)
# ============================================================
print("\n--- 2. Label Smoothing ---")
print("""
  标签平滑:
    将 one-hot 标签的 1 变为 1-ε+ε/K, 0 变为 ε/K
    防止模型过于自信, 提高泛化能力

    例 (K=3, ε=0.1):
      原始: [1, 0, 0]
      平滑: [0.933, 0.033, 0.033]
            = [(1-0.1+0.1/3), 0.1/3, 0.1/3]
""")


def label_smoothing(labels, num_classes, smoothing=0.1):
    """
    Label Smoothing 实现

    参数:
        labels: 原始标签 (one-hot 或整数)
        num_classes: 类别数
        smoothing: 平滑系数 ε

    返回:
        平滑后的标签
    """
    if isinstance(labels, np.ndarray) and labels.ndim == 1:
        # 整数标签 → one-hot
        one_hot = np.zeros((len(labels), num_classes))
        one_hot[np.arange(len(labels)), labels] = 1.0
    else:
        one_hot = labels.copy()

    # 平滑: new = (1 - ε) * original + ε / K
    smoothed = one_hot * (1 - smoothing) + smoothing / num_classes
    return smoothed


# 演示
num_classes = 5
original_label = np.array([2])  # 类别 2

for eps in [0.0, 0.05, 0.1, 0.2]:
    smoothed = label_smoothing(original_label, num_classes, smoothing=eps)
    print(f"  ε={eps:.2f}: {smoothed[0]}")


class LabelSmoothingLoss(nn.Module):
    """PyTorch Label Smoothing Loss"""

    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        """
        pred: (batch, num_classes) logits
        target: (batch,) 整数标签
        """
        log_probs = F.log_softmax(pred, dim=-1)

        # 平滑标签
        with torch.no_grad():
            smooth_labels = torch.zeros_like(log_probs)
            smooth_labels.fill_(self.smoothing / self.num_classes)
            smooth_labels.scatter_(1, target.unsqueeze(1), self.confidence)

        loss = (-smooth_labels * log_probs).sum(dim=-1).mean()
        return loss


# 验证 Label Smoothing Loss
criterion = LabelSmoothingLoss(num_classes=5, smoothing=0.1)
pred = torch.randn(3, 5)
target = torch.tensor([0, 2, 4])
loss = criterion(pred, target)
print(f"\n  Label Smoothing Loss 验证: loss={loss.item():.4f}")

# ============================================================
# 3. Dropout 变体
# ============================================================
print("\n--- 3. Dropout 变体 ---")

# --- 标准 Dropout ---
print("\n  3.1 标准 Dropout:")
print("    随机将部分神经元置零, 训练时使用, 推理时关闭")

dropout = nn.Dropout(p=0.5)
x = torch.ones(10)
print(f"    输入:   {x.tolist()}")
print(f"    Dropout: {dropout(x).tolist()}  (约 50% 被置零, 其余 ×2)")


# --- SpatialDropout ---
print("\n  3.2 SpatialDropout:")
print("    对整个通道进行 Dropout (而非单个元素)")
print("    适用于 CNN, 避免特征图中的信息泄露")


class SpatialDropout(nn.Module):
    """Spatial Dropout: 随机丢弃整个通道"""

    def __init__(self, drop_prob=0.2):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        # x shape: (batch, channels, height, width)
        if not self.training or self.drop_prob == 0:
            return x

        # 只在 channel 维度采样
        mask = torch.bernoulli(
            torch.ones(x.size(0), x.size(1), 1, 1, device=x.device) * (1 - self.drop_prob)
        )
        return x * mask / (1 - self.drop_prob)


spatial_dropout = SpatialDropout(drop_prob=0.3)
x_cnn = torch.ones(2, 4, 3, 3)  # batch=2, channels=4, H=3, W=3
out = spatial_dropout(x_cnn)
print(f"    输入 shape: {x_cnn.shape}")
print(f"    通道掩码示例: {out[0, :, 0, 0].tolist()}")


# --- DropPath (Stochastic Depth) ---
print("\n  3.3 DropPath (Stochastic Depth):")
print("    随机跳过整个残差块")
print("    用于 ResNet, ViT 等深层网络")


class DropPath(nn.Module):
    """Drop Path / Stochastic Depth"""

    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        keep_prob = 1 - self.drop_prob
        # 生成 (batch, 1, 1, ..., 1) 形状的掩码
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.ones(shape, device=x.device) * keep_prob)
        return x * mask / keep_prob


drop_path = DropPath(drop_prob=0.2)
x_test = torch.ones(4, 64)
out = drop_path(x_test)
survived = (out != 0).all(dim=1).sum().item()
print(f"    输入: batch={x_test.shape[0]}, dim={x_test.shape[1]}")
print(f"    DropPath (p=0.2): {survived}/{x_test.shape[0]} 样本保留 (整行)")


# --- DropBlock ---
print("\n  3.4 DropBlock:")
print("    在特征图上丢弃连续区域 (比 Dropout 更强)")


class DropBlock2D(nn.Module):
    """DropBlock: 在特征图上丢弃连续方块区域"""

    def __init__(self, block_size=3, drop_prob=0.1):
        super().__init__()
        self.block_size = block_size
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x

        N, C, H, W = x.shape
        gamma = self.drop_prob / (self.block_size ** 2)

        # 随机生成中心点掩码
        mask = torch.bernoulli(
            torch.ones(N, C, H - self.block_size + 1, W - self.block_size + 1,
                       device=x.device) * gamma
        )

        # 扩展为 block_size x block_size 的块
        mask = F.pad(mask, [self.block_size // 2] * 4, value=0)
        mask = F.conv2d(
            mask,
            weight=torch.ones(C, 1, self.block_size, self.block_size, device=x.device),
            groups=C,
            padding=0
        )
        mask = (mask >= 1).float()

        # 截取到原始大小
        mask = mask[:, :, :H, :W]

        return x * (1 - mask) / (1 - mask.mean() + 1e-7)


print("    DropBlock2D 已实现 (block_size=3, drop_prob=0.1)")

# ============================================================
# 4. Weight Decay 对比
# ============================================================
print("\n--- 4. Weight Decay 对比 ---")


def train_with_weight_decay(wd, epochs=200, lr=0.01):
    """用不同 weight decay 训练简单模型"""
    # 生成数据 (带噪声)
    np.random.seed(42)
    n = 50
    X = np.random.randn(n, 10)
    true_w = np.array([1, 0.5, 0, 0, 0, 0, 0, 0, 0, 0])  # 稀疏权重
    y = X @ true_w + np.random.randn(n) * 0.5

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    model = nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)

    train_losses = []
    for epoch in range(epochs):
        pred = model(X_t).squeeze()
        loss = F.mse_loss(pred, y_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    # 计算权重
    weights = model.weight.detach().numpy().squeeze()
    return train_losses, weights


weight_decays = [0.0, 0.001, 0.01, 0.1]
results = {}
for wd in weight_decays:
    losses, weights = train_with_weight_decay(wd)
    results[wd] = (losses, weights)

print("  Weight Decay 对比:")
print(f"  {'Weight Decay':>15s}", end="")
for i in range(10):
    print(f"  {'w'+str(i):>8s}", end="")
print()

for wd in weight_decays:
    _, weights = results[wd]
    print(f"  {wd:>15.3f}", end="")
    for w in weights:
        print(f"  {w:>8.3f}", end="")
    print()

# ============================================================
# 5. 演示正则化对过拟合的影响
# ============================================================
print("\n--- 5. 正则化对过拟合的影响 ---")


def demonstrate_overfitting(use_dropout=False, use_weight_decay=False,
                            use_label_smoothing=False, epochs=300):
    """演示正则化对过拟合的影响"""
    np.random.seed(42)
    torch.manual_seed(42)

    # 生成数据: 训练集少, 测试集多 (容易过拟合)
    n_train, n_test = 30, 200
    X_train = np.random.randn(n_train, 5)
    X_test = np.random.randn(n_test, 5)
    true_w = np.array([2, -1, 0.5, 0, 0])
    y_train = X_train @ true_w + np.random.randn(n_train) * 0.8
    y_test = X_test @ true_w + np.random.randn(n_test) * 0.8

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    # 过大的模型 (容易过拟合)
    model = nn.Sequential(
        nn.Linear(5, 64),
        nn.ReLU(),
        nn.Dropout(0.3) if use_dropout else nn.Identity(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(0.3) if use_dropout else nn.Identity(),
        nn.Linear(32, 1)
    )

    wd = 0.01 if use_weight_decay else 0.0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=wd)
    criterion = nn.MSELoss()

    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        model.train()
        pred = model(X_train_t).squeeze()
        loss = criterion(pred, y_train_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            test_pred = model(X_test_t).squeeze()
            test_loss = criterion(test_pred, y_test_t)

        train_losses.append(loss.item())
        test_losses.append(test_loss.item())

    return train_losses, test_losses


# 运行对比实验
configs = [
    ("无正则化 (过拟合)", False, False, False),
    ("Dropout",           True,  False, False),
    ("Weight Decay",      False, True,  False),
    ("Dropout + WD",      True,  True,  False),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("正则化对过拟合的影响", fontsize=14, fontweight="bold")

all_results = {}
for idx, (name, dropout, wd, ls) in enumerate(configs):
    train_l, test_l = demonstrate_overfitting(
        use_dropout=dropout, use_weight_decay=wd,
        use_label_smoothing=ls, epochs=300
    )
    all_results[name] = (train_l, test_l)

    ax = axes[idx // 2][idx % 2]
    ax.plot(train_l, label="Train Loss", linewidth=2)
    ax.plot(test_l, label="Test Loss", linewidth=2)
    ax.set_title(name, fontsize=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    gap = test_l[-1] - train_l[-1]
    print(f"  {name:25s}: Train={train_l[-1]:.4f}, Test={test_l[-1]:.4f}, Gap={gap:.4f}")

plt.tight_layout()
plt.savefig("D:/code/big-model-learn/code/q_01/W16/regularization_effect.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  图表已保存: W16/regularization_effect.png")

# ============================================================
# 6. 各正则化方法使用建议
# ============================================================
print("\n--- 6. 正则化使用建议 ---")
print("""
  方法                  适用场景              典型参数
  ─────────────────────────────────────────────────────
  Weight Decay (L2)     几乎所有模型          1e-4 ~ 1e-2
  Dropout               全连接层, Transformer p=0.1 ~ 0.5
  SpatialDropout        CNN 特征图            p=0.1 ~ 0.2
  DropPath              深层 ResNet, ViT      p=0.1 ~ 0.3 (线性增长)
  Label Smoothing       分类任务              ε=0.05 ~ 0.1
  DropBlock             CNN 空间特征          block=5~7, p=0.1
  Early Stopping        所有训练              patience=5~10

  组合推荐:
    - Transformer 微调:  Weight Decay + Dropout + Label Smoothing
    - CNN 从头训练:      Weight Decay + DropBlock + Mixup
    - ViT 训练:          Weight Decay + DropPath + Label Smoothing
""")

# ============================================================
# 7. 总结
# ============================================================
print("\n--- 7. 总结 ---")
print("""
  本节学习了:
  1) Label Smoothing: 防止模型过于自信
  2) Dropout 变体: SpatialDropout, DropPath, DropBlock
  3) Weight Decay 对比: 不同强度的正则化效果
  4) 过拟合实验: 展示正则化如何缩小 train-val gap

  核心思想: 正则化通过增加训练难度来提高泛化能力
    "训练时更难 → 推理时更强"

  下一步: d4_loss_functions.py - 高级损失函数
""")
