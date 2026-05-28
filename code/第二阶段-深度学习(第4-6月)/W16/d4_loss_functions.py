"""
W16-D4 高级损失函数 (Advanced Loss Functions)
===============================================
Focal Loss, Contrastive Loss (SimCLR), Triplet Loss,
Label Smoothing Loss, 在不平衡分类任务上对比
"""

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W16-D4 高级损失函数 (Advanced Loss Functions)")
print("=" * 60)

# ============================================================
# 1. 损失函数概述
# ============================================================
print("\n--- 1. 损失函数概述 ---")
print("""
  基础损失函数:
    - CrossEntropyLoss (交叉熵): 多分类标准选择
    - MSELoss (均方误差): 回归任务
    - BCELoss (二元交叉熵): 二分类

  高级损失函数:
    - Focal Loss:        解决类别不平衡
    - Contrastive Loss:  对比学习 (SimCLR)
    - Triplet Loss:      度量学习 (人脸识别)
    - Label Smoothing:   防止过自信
    - Dice Loss:         图像分割
    - Lovász Loss:       图像分割 (替代 IoU)
""")

# ============================================================
# 2. Focal Loss
# ============================================================
print("\n--- 2. Focal Loss ---")
print("""
  Focal Loss (Lin et al., 2017):
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    p_t: 正确类别的预测概率
    α_t: 类别权重 (处理类别不平衡)
    γ:   聚焦参数 (降低易分样本的损失)

    γ=0 退化为标准交叉熵
    γ>0 使模型更关注难分样本 (hard examples)

    例: 如果 p_t=0.9 (容易分), γ=2:
        FL = (1-0.9)^2 * CE = 0.01 * CE => 损失被压制到 1%
""")


class FocalLoss(nn.Module):
    """Focal Loss 实现"""

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # 类别权重
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: (batch, num_classes)
        targets: (batch,) 整数标签
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # p_t = exp(-CE)

        # Focal 调制
        focal_weight = (1 - pt) ** self.gamma

        # 类别权重
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# Focal Loss 调制因子可视化
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

pt = np.linspace(0.01, 1.0, 200)
gammas = [0, 0.5, 1, 2, 5]

ax1.set_title("Focal Loss 调制因子 (1-p_t)^γ")
for g in gammas:
    weight = (1 - pt) ** g
    ax1.plot(pt, weight, linewidth=2, label=f"γ={g}")
ax1.set_xlabel("p_t (正确类别概率)")
ax1.set_ylabel("调制因子")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Focal Loss 值
ax2.set_title("Focal Loss 值")
for g in gammas:
    ce = -np.log(pt)
    focal = (1 - pt) ** g * ce
    ax2.plot(pt, focal, linewidth=2, label=f"γ={g}")
ax2.set_xlabel("p_t")
ax2.set_ylabel("Loss")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("D:/code/big-model-learn/code/q_01/W16/focal_loss_visualization.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Focal Loss 可视化已保存: W16/focal_loss_visualization.png")

# ============================================================
# 3. Contrastive Loss (SimCLR 风格)
# ============================================================
print("\n--- 3. Contrastive Loss (SimCLR 风格) ---")
print("""
  对比学习 (Contrastive Learning):
    - 同一数据增强的两个视图互为正例
    - 不同数据的增强互为负例
    - 拉近正例, 推远负例

  NT-Xent Loss (SimCLR):
    L = -log[ exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ) ]

    sim: 余弦相似度
    τ:   温度参数
    z_i, z_j: 同一样本的两个增强视图
""")


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss (SimCLR)"""

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        """
        z1, z2: (batch, dim) 同一 batch 的两个增强视图
        """
        batch_size = z1.shape[0]

        # L2 归一化
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        # 拼接: (2N, dim)
        z = torch.cat([z1, z2], dim=0)

        # 计算相似度矩阵: (2N, 2N)
        sim = torch.mm(z, z.t()) / self.temperature

        # 掩码: 排除自身
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, -1e9)

        # 正例对: (i, i+N) 和 (i+N, i)
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(0, batch_size)
        ]).to(z.device)

        loss = F.cross_entropy(sim, labels)
        return loss


# 演示
ntxent = NTXentLoss(temperature=0.5)
z1 = torch.randn(8, 64)
z2 = z1 + torch.randn(8, 64) * 0.1  # 相似的视图
loss = ntxent(z1, z2)
print(f"  NT-Xent Loss 验证: loss={loss.item():.4f}")

# ============================================================
# 4. Triplet Loss
# ============================================================
print("\n--- 4. Triplet Loss ---")
print("""
  Triplet Loss (Schroff et al., 2015):
    L = max(0, d(anchor, positive) - d(anchor, negative) + margin)

    anchor:   基准样本
    positive: 与 anchor 同类的样本
    negative: 与 anchor 不同类的样本
    margin:   安全距离

    目标: 使 d(a,p) + margin < d(a,n)
    即: 正例距离 < 负例距离 - margin
""")


class TripletLoss(nn.Module):
    """Triplet Loss 实现"""

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """
        anchor, positive, negative: (batch, dim)
        """
        d_ap = F.pairwise_distance(anchor, positive, p=2)
        d_an = F.pairwise_distance(anchor, negative, p=2)

        loss = F.relu(d_ap - d_an + self.margin)
        return loss.mean()


# 演示
triplet_loss = TripletLoss(margin=1.0)
anchor = torch.randn(4, 64)
positive = anchor + torch.randn(4, 64) * 0.3  # 接近 anchor
negative = torch.randn(4, 64) * 2              # 远离 anchor

loss = triplet_loss(anchor, positive, negative)
d_ap = F.pairwise_distance(anchor, positive).mean()
d_an = F.pairwise_distance(anchor, negative).mean()
print(f"  Triplet Loss 验证: loss={loss.item():.4f}")
print(f"    d(anchor, positive) = {d_ap.item():.4f}")
print(f"    d(anchor, negative) = {d_an.item():.4f}")

# ============================================================
# 5. Label Smoothing Loss
# ============================================================
print("\n--- 5. Label Smoothing Loss ---")


class LabelSmoothingLoss(nn.Module):
    """Label Smoothing Cross Entropy Loss"""

    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            smooth = torch.full_like(log_probs, self.smoothing / self.num_classes)
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing + self.smoothing / self.num_classes)

        loss = (-smooth * log_probs).sum(dim=-1).mean()
        return loss


# ============================================================
# 6. 不平衡分类任务对比实验
# ============================================================
print("\n--- 6. 不平衡分类任务对比实验 ---")

# 生成不平衡数据
np.random.seed(42)
torch.manual_seed(42)

n_classes = 3
class_sizes = [500, 100, 50]  # 不平衡: 类别0多, 类别2少

X_list, y_list = [], []
for cls, size in enumerate(class_sizes):
    center = np.random.randn(10) * 3
    X_list.append(center + np.random.randn(size, 10))
    y_list.append(np.full(size, cls))

X = np.vstack(X_list)
y = np.concatenate(y_list)

# 打乱顺序
indices = np.random.permutation(len(y))
X, y = X[indices], y[indices]

X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.long)

print(f"  数据集: {len(y)} 样本, {n_classes} 类")
for cls in range(n_classes):
    print(f"    类别 {cls}: {(y == cls).sum()} 样本")


# 训练函数
def train_model(criterion_fn, epochs=100, lr=0.01):
    """训练简单分类模型"""
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, n_classes)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        logits = model(X_t)
        loss = criterion_fn(logits, y_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 评估
    with torch.no_grad():
        logits = model(X_t)
        preds = torch.argmax(logits, dim=-1)

    # 各类别准确率
    acc_per_class = []
    for cls in range(n_classes):
        mask = (y_t == cls)
        if mask.sum() > 0:
            acc = (preds[mask] == cls).float().mean().item()
            acc_per_class.append(acc)
        else:
            acc_per_class.append(0)

    overall_acc = (preds == y_t).float().mean().item()
    return overall_acc, acc_per_class


# 定义不同损失函数
ce_loss = nn.CrossEntropyLoss()

# 加权 CrossEntropy (按类别频率加权)
class_weights = torch.tensor([1.0 / s for s in class_sizes])
class_weights = class_weights / class_weights.sum() * n_classes
weighted_ce = nn.CrossEntropyLoss(weight=class_weights)

# Focal Loss
focal = FocalLoss(gamma=2.0)

# Focal Loss + 类别权重
focal_weighted = FocalLoss(alpha=class_weights, gamma=2.0)

# Label Smoothing
label_smooth = LabelSmoothingLoss(n_classes, smoothing=0.1)

losses = {
    "CrossEntropy":        ce_loss,
    "Weighted CE":         weighted_ce,
    "Focal Loss":          focal,
    "Focal+Weighted":      focal_weighted,
    "Label Smoothing":     label_smooth,
}

print("\n  各损失函数对比 (100 epochs):")
print(f"  {'方法':20s} {'总体准确率':>10s}  {'类别0':>8s}  {'类别1':>8s}  {'类别2':>8s}")
print("  " + "-" * 65)

results = {}
for name, criterion in losses.items():
    overall, per_class = train_model(criterion)
    results[name] = (overall, per_class)
    per_class_str = "  ".join(f"{a:.3f}" for a in per_class)
    print(f"  {name:20s} {overall:>10.3f}  {per_class_str}")

# ============================================================
# 7. 可视化对比
# ============================================================
print("\n--- 7. 可视化对比 ---")

fig, ax = plt.subplots(figsize=(12, 6))

names = list(results.keys())
x = np.arange(len(names))
width = 0.2

class_accs = np.array([results[n][1] for n in names])

for cls in range(n_classes):
    bars = ax.bar(x + cls * width, class_accs[:, cls], width,
                  label=f"类别 {cls} ({class_sizes[cls]} 样本)")

ax.set_xlabel("损失函数")
ax.set_ylabel("准确率")
ax.set_title("不同损失函数在不平衡数据上的各类别准确率", fontsize=13)
ax.set_xticks(x + width)
ax.set_xticklabels(names, rotation=15, ha="right")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
ax.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig("D:/code/big-model-learn/code/q_01/W16/loss_functions_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  对比图已保存: W16/loss_functions_comparison.png")

# ============================================================
# 8. 损失函数选择指南
# ============================================================
print("\n--- 8. 损失函数选择指南 ---")
print("""
  场景                    推荐损失函数
  ──────────────────────────────────────────────────────
  标准多分类              CrossEntropyLoss
  不平衡分类              Focal Loss 或 Weighted CE
  二分类                  BCEWithLogitsLoss
  回归                    MSELoss / SmoothL1Loss
  度量学习 (人脸/检索)     Triplet Loss
  对比学习 (自监督)        NT-Xent Loss (SimCLR)
  防止过自信              Label Smoothing CE
  图像分割                Dice Loss + CE
  目标检测                Focal Loss (RetinaNet)
  知识蒸馏                KL Divergence (软目标)

  下一步: d5_knowledge_distillation.py - 知识蒸馏
""")

# ============================================================
# 9. 总结
# ============================================================
print("\n--- 9. 总结 ---")
print("""
  本节学习了:
  1) Focal Loss: 降低易分样本权重, 聚焦难分样本
  2) Contrastive Loss (NT-Xent): 对比学习, 拉近正例推远负例
  3) Triplet Loss: 度量学习, anchor-positive-negative
  4) Label Smoothing Loss: 防止模型过于自信
  5) 在不平衡数据上对比了 5 种损失函数

  关键要点:
    - 不同任务需要不同的损失函数
    - 类别不平衡: 用 Focal Loss 或加权 CE
    - 表示学习: 用 Contrastive/Triplet Loss
    - 实践中经常组合使用多种损失

  下一步: d5_knowledge_distillation.py - 知识蒸馏
""")
