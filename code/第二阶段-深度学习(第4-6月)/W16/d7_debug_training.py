"""
W16-D7 训练调试技巧 (Debugging Training)
==========================================
常见训练问题: NaN loss, loss 不下降, 过拟合, 欠拟合,
学习率查找器, 梯度流可视化
"""

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W16-D7 训练调试技巧 (Debugging Training)")
print("=" * 60)

# ============================================================
# 1. 常见训练问题清单
# ============================================================
print("\n--- 1. 常见训练问题 ---")
print("""
  ┌────────────────┬──────────────────────────┬────────────────────────┐
  │  症状          │  可能原因                │  排查方向              │
  ├────────────────┼──────────────────────────┼────────────────────────┤
  │  Loss = NaN    │ 梯度爆炸, LR过大,       │ 降低LR, 梯度裁剪,     │
  │                │ 数据有NaN/Inf            │ 检查数据              │
  ├────────────────┼──────────────────────────┼────────────────────────┤
  │  Loss不下降    │ LR太小/太大, 数据错误,  │ 学习率查找器,         │
  │                │ 梯度消失, 架构错误       │ 检查数据pipeline      │
  ├────────────────┼──────────────────────────┼────────────────────────┤
  │  过拟合        │ 模型太大, 数据太少,     │ 增加正则化, 更多数据, │
  │  (train↓ val↑) │ 正则化不足              │ Dropout, Weight Decay │
  ├────────────────┼──────────────────────────┼────────────────────────┤
  │  欠拟合        │ 模型太小, LR不对,       │ 增大模型, 调整LR,     │
  │  (train不降)   │ 训练不够, 特征不足      │ 增加训练轮数          │
  └────────────────┴──────────────────────────┴────────────────────────┘
""")

# ============================================================
# 2. 问题1: NaN Loss 诊断与修复
# ============================================================
print("\n--- 2. NaN Loss 诊断与修复 ---")
print("""
  NaN Loss 的常见原因:
    1) 学习率过大 → 梯度爆炸 → 权重 NaN → Loss NaN
    2) 输入数据包含 NaN 或 Inf
    3) 数值不稳定 (log(0), 除以0)
    4) 损失函数计算中的溢出
""")

# 复现 NaN
print("  复现 NaN Loss:")

# 极大学习率导致 NaN
model_nan = nn.Sequential(nn.Linear(5, 3), nn.ReLU(), nn.Linear(3, 2))
optimizer_nan = torch.optim.SGD(model_nan.parameters(), lr=100.0)  # 极大 LR

x_nan = torch.randn(4, 5)
y_nan = torch.tensor([0, 1, 0, 1])

nan_detected = False
for step in range(50):
    logits = model_nan(x_nan)
    loss = F.cross_entropy(logits, y_nan)
    optimizer_nan.zero_grad()
    loss.backward()
    optimizer_nan.step()

    if torch.isnan(loss) or torch.isinf(loss):
        print(f"    Step {step}: Loss = NaN/Inf! (LR=100.0 导致梯度爆炸)")
        nan_detected = True
        break

if not nan_detected:
    print(f"    50步内未出现NaN, 但loss={loss.item():.2f} (可能不稳定)")

# 修复: 梯度裁剪
print("\n  修复方案 1: 梯度裁剪 (Gradient Clipping)")
model_fix = nn.Sequential(nn.Linear(5, 3), nn.ReLU(), nn.Linear(3, 2))
optimizer_fix = torch.optim.SGD(model_fix.parameters(), lr=100.0)

for step in range(50):
    logits = model_fix(x_nan)
    loss = F.cross_entropy(logits, y_nan)
    optimizer_fix.zero_grad()
    loss.backward()
    # 梯度裁剪: 限制梯度范数
    torch.nn.utils.clip_grad_norm_(model_fix.parameters(), max_norm=1.0)
    optimizer_fix.step()

    if torch.isnan(loss):
        print(f"    Step {step}: 仍然 NaN")
        break
else:
    print(f"    梯度裁剪后 50 步 loss={loss.item():.4f} (稳定!)")

# 修复: 降低学习率
print("\n  修复方案 2: 降低学习率")
model_lr = nn.Sequential(nn.Linear(5, 3), nn.ReLU(), nn.Linear(3, 2))
optimizer_lr = torch.optim.SGD(model_lr.parameters(), lr=0.01)

for step in range(50):
    logits = model_lr(x_nan)
    loss = F.cross_entropy(logits, y_nan)
    optimizer_lr.zero_grad()
    loss.backward()
    optimizer_lr.step()

print(f"    LR=0.01 后 50 步 loss={loss.item():.4f}")

# 检查数据中的 NaN
print("\n  检查数据中的异常值:")
clean_data = torch.randn(100, 5)
nan_data = clean_data.clone()
nan_data[10, 2] = float('nan')
nan_data[50, 0] = float('inf')

print(f"    NaN 数量: {torch.isnan(nan_data).sum().item()}")
print(f"    Inf 数量: {torch.isinf(nan_data).sum().item()}")
print(f"    修复: nan_data[torch.isnan(nan_data)] = 0")
nan_data[torch.isnan(nan_data)] = 0
nan_data[torch.isinf(nan_data)] = 0
print(f"    修复后 NaN: {torch.isnan(nan_data).sum().item()}, Inf: {torch.isinf(nan_data).sum().item()}")

# ============================================================
# 3. 问题2: Loss 不下降
# ============================================================
print("\n--- 3. Loss 不下降诊断 ---")

# 准备数据
np.random.seed(42)
torch.manual_seed(42)
X = torch.randn(200, 5)
y = (X[:, 0] > 0).long()
dataset = TensorDataset(X, y)
train_set, val_set = random_split(dataset, [160, 40])
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

# 演示不同学习率
print("  不同学习率对 Loss 下降的影响:")

lr_losses = {}
for lr, desc in [(0.0001, "太小"), (0.01, "合适"), (1.0, "太大")]:
    model = nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []

    for epoch in range(50):
        for X_b, y_b in train_loader:
            logits = model(X_b)
            loss = F.cross_entropy(logits, y_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(loss.item())

    lr_losses[lr] = losses
    print(f"    LR={lr:>8.4f} ({desc:>4s}): 初始={losses[0]:.4f} → 最终={losses[-1]:.4f}")

# ============================================================
# 4. 问题3: 过拟合诊断 (Train vs Val Gap)
# ============================================================
print("\n--- 4. 过拟合诊断 ---")

# 故意制造过拟合: 大模型 + 少数据
torch.manual_seed(42)
X_small = torch.randn(30, 5)
y_small = (X_small[:, 0] > 0).long()

overfit_model = nn.Sequential(
    nn.Linear(5, 256), nn.ReLU(),
    nn.Linear(256, 128), nn.ReLU(),
    nn.Linear(128, 64), nn.ReLU(),
    nn.Linear(64, 2)
)

optimizer_of = torch.optim.Adam(overfit_model.parameters(), lr=0.01)

# 分 train/val
train_X, val_X = X_small[:20], X_small[20:]
train_y, val_y = y_small[:20], y_small[20:]

train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(100):
    overfit_model.train()
    logits = overfit_model(train_X)
    loss = F.cross_entropy(logits, train_y)
    optimizer_of.zero_grad()
    loss.backward()
    optimizer_of.step()

    train_losses.append(loss.item())
    with torch.no_grad():
        val_logits = overfit_model(val_X)
        val_loss = F.cross_entropy(val_logits, val_y)
    val_losses.append(val_loss.item())

    train_pred = torch.argmax(logits, dim=-1)
    val_pred = torch.argmax(val_logits, dim=-1)
    train_accs.append((train_pred == train_y).float().mean().item())
    val_accs.append((val_pred == val_y).float().mean().item())

print(f"  过拟合实验 (大模型 + 少数据):")
print(f"    训练 Loss: {train_losses[0]:.4f} → {train_losses[-1]:.4f}")
print(f"    验证 Loss: {val_losses[0]:.4f} → {val_losses[-1]:.4f}")
print(f"    训练 Acc:  {train_accs[0]:.4f} → {train_accs[-1]:.4f}")
print(f"    验证 Acc:  {val_accs[0]:.4f} → {val_accs[-1]:.4f}")
print(f"    Train-Val Gap: {abs(train_losses[-1] - val_losses[-1]):.4f}")
print(f"    诊断: {'过拟合!' if train_losses[-1] < val_losses[-1] * 0.5 else '正常'}")

# ============================================================
# 5. 问题4: 欠拟合诊断
# ============================================================
print("\n--- 5. 欠拟合诊断 ---")

# 极小模型
underfit_model = nn.Linear(5, 2)  # 无隐藏层
optimizer_uf = torch.optim.SGD(underfit_model.parameters(), lr=0.001)

uf_losses = []
for epoch in range(50):
    logits = underfit_model(train_X)
    loss = F.cross_entropy(logits, train_y)
    optimizer_uf.zero_grad()
    loss.backward()
    optimizer_uf.step()
    uf_losses.append(loss.item())

print(f"  欠拟合实验 (线性模型 + 低LR):")
print(f"    训练 Loss: {uf_losses[0]:.4f} → {uf_losses[-1]:.4f}")
print(f"    诊断: {'欠拟合 - Loss 仍然很高' if uf_losses[-1] > 0.3 else '正常'}")
print(f"    解决: 增大模型, 增大LR, 增加特征")

# ============================================================
# 6. 学习率查找器 (LR Finder)
# ============================================================
print("\n--- 6. 学习率查找器 (LR Finder) ---")
print("""
  LR Finder (Smith, 2017):
    1) 从极小 LR 开始 (如 1e-7)
    2) 每个 batch 指数增大 LR
    3) 记录每个 LR 对应的 loss
    4) 找到 loss 下降最快的位置 → 最佳 LR

  规则: 选择 loss 下降最快处的 LR (而非最小 loss 处)
""")


def lr_finder(model, train_loader, lr_start=1e-7, lr_end=10, num_iter=100):
    """学习率查找器"""
    model_copy = nn.Sequential(*[nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 2)])

    # 指数增长的 LR 序列
    lrs = np.logspace(np.log10(lr_start), np.log10(lr_end), num_iter)
    losses = []

    optimizer = torch.optim.SGD(model_copy.parameters(), lr=lr_start)
    criterion = nn.CrossEntropyLoss()

    data_iter = iter(train_loader)
    for i, lr in enumerate(lrs):
        # 更新 LR
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 获取一个 batch
        try:
            X_b, y_b = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            X_b, y_b = next(data_iter)

        model_copy.train()
        logits = model_copy(X_b)
        loss = criterion(logits, y_b)

        if torch.isnan(loss) or torch.isinf(loss):
            break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return lrs[:len(losses)], losses


# 运行 LR Finder
torch.manual_seed(42)
finder_model = nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 2))
lrs, losses = lr_finder(finder_model, train_loader)

# 找到最佳 LR (loss 下降最快处)
if len(losses) > 5:
    # 平滑 loss
    smoothed = np.convolve(losses, np.ones(5) / 5, mode='valid')
    # 计算梯度 (下降速率)
    grads = np.gradient(smoothed)
    # 最小梯度 = 下降最快
    best_idx = np.argmin(grads) + 2  # 偏移补偿
    best_lr = lrs[min(best_idx, len(lrs) - 1)]
    print(f"  LR Finder 结果:")
    print(f"    最佳学习率: {best_lr:.6f}")
    print(f"    搜索范围: {lrs[0]:.2e} ~ {lrs[-1]:.2e}")
    print(f"    最小 Loss: {min(losses):.4f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(lrs, losses, 'b-', linewidth=1.5, alpha=0.8)
ax.set_xscale('log')
ax.set_xlabel('Learning Rate')
ax.set_ylabel('Loss')
ax.set_title('学习率查找器 (LR Finder)', fontsize=13)
if len(losses) > 5:
    ax.axvline(x=best_lr, color='r', linestyle='--', linewidth=2, label=f'Best LR={best_lr:.6f}')
    ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("D:/code/big-model-learn/code/q_01/W16/lr_finder.png", dpi=150, bbox_inches="tight")
plt.close()
print("  LR Finder 图已保存: W16/lr_finder.png")

# ============================================================
# 7. 梯度流可视化
# ============================================================
print("\n--- 7. 梯度流可视化 ---")
print("""
  梯度流诊断:
    - 检查各层梯度范数, 判断梯度消失/爆炸
    - 如果某层梯度接近 0 → 梯度消失
    - 如果某层梯度极大 → 梯度爆炸
    - 理想情况: 各层梯度范数相近
""")


def analyze_gradients(model, train_loader, num_batches=5):
    """分析模型各层梯度"""
    criterion = nn.CrossEntropyLoss()
    layer_grads = {name: [] for name, _ in model.named_parameters()}

    model.train()
    for i, (X_b, y_b) in enumerate(train_loader):
        if i >= num_batches:
            break

        logits = model(X_b)
        loss = criterion(logits, y_b)
        model.zero_grad()
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                layer_grads[name].append(param.grad.norm().item())

    return layer_grads


# 构建不同深度的模型来演示梯度问题
print("  梯度流分析 (不同深度):")

for n_layers in [3, 10, 20]:
    layers = []
    for _ in range(n_layers):
        layers.extend([nn.Linear(16, 16), nn.Sigmoid()])  # Sigmoid 容易梯度消失
    layers.append(nn.Linear(16, 2))
    deep_model = nn.Sequential(*layers)

    # 前向传播 + 反向传播
    x = torch.randn(8, 16)
    y = torch.randint(0, 2, (8,))
    logits = deep_model(x)
    loss = F.cross_entropy(logits, y)
    deep_model.zero_grad()
    loss.backward()

    # 收集各层梯度
    grad_norms = []
    for name, param in deep_model.named_parameters():
        if 'weight' in name and param.grad is not None:
            grad_norms.append(param.grad.norm().item())

    first_grad = grad_norms[0] if grad_norms else 0
    last_grad = grad_norms[-1] if grad_norms else 0
    ratio = last_grad / (first_grad + 1e-10)

    print(f"    {n_layers} 层 (Sigmoid): 首层梯度={first_grad:.6f}, "
          f"末层梯度={last_grad:.6f}, 比值={ratio:.6f}")

# ReLU 的梯度对比
print("\n  ReLU vs Sigmoid (10层):")
for act_name, act_fn in [("Sigmoid", nn.Sigmoid), ("ReLU", nn.ReLU), ("LeakyReLU", nn.LeakyReLU)]:
    layers = []
    for _ in range(10):
        layers.extend([nn.Linear(16, 16), act_fn()])
    layers.append(nn.Linear(16, 2))
    model_act = nn.Sequential(*layers)

    x = torch.randn(8, 16)
    y = torch.randint(0, 2, (8,))
    logits = model_act(x)
    loss = F.cross_entropy(logits, y)
    model_act.zero_grad()
    loss.backward()

    grad_norms = []
    for name, param in model_act.named_parameters():
        if 'weight' in name and param.grad is not None:
            grad_norms.append(param.grad.norm().item())

    first_grad = grad_norms[0] if grad_norms else 0
    last_grad = grad_norms[-1] if grad_norms else 0
    print(f"    {act_name:12s}: 首层={first_grad:.6f}, 末层={last_grad:.6f}, "
          f"比值={last_grad / (first_grad + 1e-10):.6f}")

# 梯度流可视化
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 训练 Loss vs 验证 Loss (过拟合)
ax1.plot(train_losses, 'b-', linewidth=2, label="训练 Loss")
ax1.plot(val_losses, 'r-', linewidth=2, label="验证 Loss")
ax1.fill_between(range(len(train_losses)), train_losses, val_losses,
                 alpha=0.2, color='orange', label="过拟合 Gap")
ax1.set_title("过拟合诊断: Train vs Val Loss", fontsize=13)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(True, alpha=0.3)

# 学习率对 loss 的影响
for lr, losses in lr_losses.items():
    ax2.plot(losses, linewidth=2, label=f"LR={lr}")
ax2.set_title("不同学习率的 Loss 变化", fontsize=13)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("D:/code/big-model-learn/code/q_01/W16/training_diagnostics.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  诊断图已保存: W16/training_diagnostics.png")

# ============================================================
# 8. 调试检查清单
# ============================================================
print("\n--- 8. 调试检查清单 ---")
print("""
  Step 1: 数据检查
    □ 数据是否正确加载? (打印几个样本)
    □ 标签是否正确? (类别分布)
    □ 是否有 NaN/Inf? (torch.isnan().sum())
    □ 数据归一化了吗? (mean≈0, std≈1)

  Step 2: 模型检查
    □ 输出形状正确? (model(x).shape)
    □ 参数量合理? (sum(p.numel() for p in model.parameters()))
    □ 过一次 batch 能跑通? (单步前向+反向)

  Step 3: 初始检查
    □ 初始 Loss 是否合理? (≈-log(1/K) for K classes)
    □ 初始准确率是否≈随机? (≈1/K)

  Step 4: 训练检查
    □ Loss 是否下降? (前 10 步)
    □ 梯度是否正常? (无 NaN, 无极端值)
    □ 学习率是否合适? (用 LR Finder)

  Step 5: 评估检查
    □ Train vs Val Loss 趋势? (过拟合/欠拟合)
    □ 验证集准确率是否合理?
    □ 各类别指标是否平衡?

  常用调试代码:
    # 检查梯度
    for name, param in model.named_parameters():
        print(f"{name}: grad_norm={param.grad.norm():.6f}")

    # 检查数据
    print(f"NaN: {torch.isnan(data).sum()}, Inf: {torch.isinf(data).sum()}")

    # 检查模型输出
    print(f"Output range: [{logits.min():.2f}, {logits.max():.2f}]")
""")

# ============================================================
# 9. 总结
# ============================================================
print("\n--- 9. 总结 ---")
print("""
  本节学习了:
  1) NaN Loss: 梯度裁剪, 降低 LR, 检查数据
  2) Loss 不下降: LR Finder, 检查数据 pipeline
  3) 过拟合诊断: Train vs Val Loss gap
  4) 欠拟合诊断: Train Loss 不下降
  5) 学习率查找器: 从小到大指数增长 LR
  6) 梯度流可视化: 各层梯度范数分析
  7) 完整的调试检查清单

  调试黄金法则:
    - 从简单开始 (小数据, 小模型)
    - 每步验证 (不要一次改太多)
    - 先让模型过拟合训练集 (证明模型有学习能力)
    - 再解决过拟合 (正则化, 数据增强)

  W16 (高级训练技术) 全部完成!
  回顾: LR调度, 数据增强, 正则化, 损失函数,
        知识蒸馏, 超参数优化, 训练调试
""")
