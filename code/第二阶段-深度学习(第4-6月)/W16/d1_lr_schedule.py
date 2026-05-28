"""
W16-D1 学习率调度 (Learning Rate Schedule)
============================================
Cosine Annealing, Warmup, OneCycleLR, ReduceLROnPlateau 实现,
绘制所有调度曲线, 展示对训练的影响
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W16-D1 学习率调度 (Learning Rate Schedule)")
print("=" * 60)

# ============================================================
# 1. 学习率调度的重要性
# ============================================================
print("\n--- 1. 学习率调度的重要性 ---")
print("""
  学习率 (Learning Rate) 是训练中最重要的超参数之一:
    - 太大: 损失震荡, 甚至发散
    - 太小: 收敛太慢, 陷入局部最优
    - 合理的调度:
        训练初期: 大 LR → 快速收敛
        训练后期: 小 LR → 精细调优

  常见调度策略:
    1) StepLR:         固定步长衰减
    2) CosineAnnealing: 余弦退火
    3) Warmup + Decay:  先热身再衰减
    4) OneCycleLR:      单周期策略
    5) ReduceLROnPlateau: 自适应衰减
    6) ExponentialLR:   指数衰减
""")

# ============================================================
# 2. 手动实现各种调度器
# ============================================================
print("\n--- 2. 手动实现调度器 ---")

num_epochs = 50
steps_per_epoch = 100
total_steps = num_epochs * steps_per_epoch
base_lr = 0.1


# --- StepLR ---
def step_lr(step, initial_lr, drop_rate=0.5, drop_every=500):
    """固定步长衰减"""
    drops = step // drop_every
    return initial_lr * (drop_rate ** drops)


# --- ExponentialLR ---
def exponential_lr(step, initial_lr, gamma=0.999):
    """指数衰减: lr = lr * gamma^step"""
    return initial_lr * (gamma ** step)


# --- Cosine Annealing ---
def cosine_annealing_lr(step, initial_lr, min_lr=1e-6, total_steps=total_steps):
    """余弦退火"""
    if step >= total_steps:
        return min_lr
    return min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(np.pi * step / total_steps))


# --- Warmup + Cosine Decay ---
def warmup_cosine_lr(step, initial_lr, min_lr=1e-6, warmup_steps=500, total_steps=total_steps):
    """线性热身 + 余弦衰减"""
    if step < warmup_steps:
        # 线性热身
        return initial_lr * step / warmup_steps
    else:
        # 余弦衰减
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(np.pi * progress))


# --- OneCycleLR ---
def one_cycle_lr(step, max_lr=0.1, min_lr=0.001, total_steps=total_steps):
    """单周期策略: 热身 → 最大 → 退火"""
    pct_start = 0.3  # 30% 时间用于热身
    if step <= total_steps * pct_start:
        # 热身阶段: 线性增长
        return min_lr + (max_lr - min_lr) * step / (total_steps * pct_start)
    else:
        # 退火阶段: 余弦衰减到 min_lr
        progress = (step - total_steps * pct_start) / (total_steps * (1 - pct_start))
        final_lr = min_lr / 10  # 最终衰减到 min_lr/10
        return final_lr + (max_lr - final_lr) * 0.5 * (1 + np.cos(np.pi * progress))


# --- ReduceLROnPlateau (模拟) ---
def simulate_reduce_on_plateau(initial_lr=0.1, patience=100, factor=0.5):
    """模拟 ReduceLROnPlateau 行为"""
    # 模拟训练过程中 loss 不再下降时降低学习率
    lrs = []
    lr = initial_lr
    for step in range(total_steps):
        # 模拟: 在特定步骤 loss 停滞, 触发衰减
        if step == 800:
            lr *= factor
        elif step == 1800:
            lr *= factor
        elif step == 3200:
            lr *= factor
        lrs.append(lr)
    return lrs


print("  6 种调度器已定义")

# ============================================================
# 3. 计算所有调度器的学习率
# ============================================================
print("\n--- 3. 计算学习率序列 ---")

steps = np.arange(total_steps)

lrs_step = [step_lr(s, base_lr) for s in steps]
lrs_exp = [exponential_lr(s, base_lr) for s in steps]
lrs_cosine = [cosine_annealing_lr(s, base_lr) for s in steps]
lrs_warmup = [warmup_cosine_lr(s, base_lr) for s in steps]
lrs_onecycle = [one_cycle_lr(s) for s in steps]
lrs_plateau = simulate_reduce_on_plateau(base_lr)

print(f"  总步数: {total_steps}")
print(f"  初始 LR: {base_lr}")

# ============================================================
# 4. 绘制所有调度曲线
# ============================================================
print("\n--- 4. 绘制调度曲线 ---")

fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle("学习率调度策略对比", fontsize=16, fontweight="bold")

schedules = [
    ("StepLR (固定步长衰减)", lrs_step),
    ("ExponentialLR (指数衰减)", lrs_exp),
    ("Cosine Annealing (余弦退火)", lrs_cosine),
    ("Warmup + Cosine Decay (热身+余弦)", lrs_warmup),
    ("OneCycleLR (单周期策略)", lrs_onecycle),
    ("ReduceLROnPlateau (自适应衰减)", lrs_plateau),
]

for idx, (title, lrs) in enumerate(schedules):
    ax = axes[idx // 2][idx % 2]
    ax.plot(steps, lrs, linewidth=2)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Learning Rate")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, total_steps)

plt.tight_layout()
plt.savefig("D:/code/big-model-learn/code/q_01/W16/lr_schedules.png", dpi=150, bbox_inches="tight")
plt.close()
print("  图表已保存: W16/lr_schedules.png")

# ============================================================
# 5. 对比图 (所有调度在一张图)
# ============================================================
print("\n--- 5. 绘制对比图 ---")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 全量视图
for title, lrs in schedules:
    ax1.plot(steps, lrs, linewidth=2, label=title.split("(")[0].strip())
ax1.set_title("所有调度策略对比", fontsize=14)
ax1.set_xlabel("Training Step")
ax1.set_ylabel("Learning Rate")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# 对数视图
for title, lrs in schedules:
    ax2.semilogy(steps, lrs, linewidth=2, label=title.split("(")[0].strip())
ax2.set_title("学习率对比 (对数刻度)", fontsize=14)
ax2.set_xlabel("Training Step")
ax2.set_ylabel("Learning Rate (log scale)")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("D:/code/big-model-learn/code/q_01/W16/lr_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  对比图已保存: W16/lr_comparison.png")

# ============================================================
# 6. 展示对训练的影响 (模拟)
# ============================================================
print("\n--- 6. 学习率对训练的影响 (模拟) ---")


def simulate_training(lr_schedule_fn, label, num_steps=2000):
    """模拟训练过程, 返回 loss 曲线"""
    losses = []
    loss = 2.0  # 初始 loss
    for step in range(num_steps):
        lr = lr_schedule_fn(step) if callable(lr_schedule_fn) else lr_schedule_fn[step]
        # 模拟: loss 下降 + 噪声
        noise = np.random.normal(0, 0.02)
        loss = loss * (1 - lr * 0.5) + noise
        loss = max(loss, 0.05)
        losses.append(loss)
    return losses


np.random.seed(42)

# 模拟固定 LR vs 调度 LR
fixed_lrs = [0.01] * total_steps
loss_fixed = simulate_training(lambda s: 0.01, "固定 LR")
loss_cosine = simulate_training(
    lambda s: cosine_annealing_lr(s, 0.05), "Cosine"
)
loss_warmup = simulate_training(
    lambda s: warmup_cosine_lr(s, 0.05), "Warmup+Cosine"
)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(loss_fixed, alpha=0.7, linewidth=1.5, label="固定 LR=0.01")
ax.plot(loss_cosine, alpha=0.7, linewidth=1.5, label="Cosine (初始=0.05)")
ax.plot(loss_warmup, alpha=0.7, linewidth=1.5, label="Warmup+Cosine (初始=0.05)")
ax.set_title("不同学习率策略下的训练 Loss (模拟)", fontsize=14)
ax.set_xlabel("Training Step")
ax.set_ylabel("Loss")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("D:/code/big-model-learn/code/q_01/W16/lr_training_effect.png", dpi=150, bbox_inches="tight")
plt.close()
print("  训练效果图已保存: W16/lr_training_effect.png")

# 打印模拟结果
print(f"\n  模拟训练最终 Loss (2000 steps):")
print(f"    固定 LR=0.01:       {loss_fixed[-1]:.4f}")
print(f"    Cosine:             {loss_cosine[-1]:.4f}")
print(f"    Warmup + Cosine:    {loss_warmup[-1]:.4f}")

# ============================================================
# 7. PyTorch 实现
# ============================================================
print("\n--- 7. PyTorch 实现 ---")
print("""
  在 PyTorch 中使用学习率调度:

  import torch
  from torch.optim import AdamW
  from torch.optim.lr_scheduler import (
      StepLR, CosineAnnealingLR, OneCycleLR,
      ReduceLROnPlateau, CosineAnnealingWarmRestarts
  )

  optimizer = AdamW(model.parameters(), lr=2e-5)

  # Cosine Annealing
  scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

  # OneCycleLR
  scheduler = OneCycleLR(
      optimizer, max_lr=2e-5,
      total_steps=num_training_steps
  )

  # ReduceLROnPlateau
  scheduler = ReduceLROnPlateau(
      optimizer, mode='min', patience=3, factor=0.5
  )

  # Warmup + Decay (HuggingFace)
  from transformers import get_scheduler
  scheduler = get_scheduler(
      "cosine",
      optimizer,
      num_warmup_steps=100,
      num_training_steps=1000
  )

  # 训练循环中使用:
  for epoch in range(num_epochs):
      for batch in dataloader:
          loss = model(batch)
          loss.backward()
          optimizer.step()
          scheduler.step()  # 更新 LR
""")

# 验证 PyTorch 调度器 (如果 torch 可用)
try:
    import torch
    from torch.optim import SGD
    from torch.optim.lr_scheduler import (
        StepLR, CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau
    )

    dummy_model = torch.nn.Linear(10, 2)
    optimizer = SGD(dummy_model.parameters(), lr=0.1)

    # CosineAnnealing
    scheduler = CosineAnnealingLR(optimizer, T_max=100)
    lrs_torch = []
    for _ in range(100):
        lrs_torch.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    print(f"\n  PyTorch CosineAnnealingLR 验证:")
    print(f"    初始 LR: {lrs_torch[0]:.6f}")
    print(f"    最终 LR: {lrs_torch[-1]:.6f}")
    print(f"    最小 LR: {min(lrs_torch):.6f}")
    print("  验证通过!")

except ImportError:
    print("  [!] PyTorch 未安装, 跳过验证")

# ============================================================
# 8. 总结
# ============================================================
print("\n--- 8. 总结 ---")
print("""
  本节学习了 6 种学习率调度策略:
  1) StepLR:              固定步长衰减 (简单但不灵活)
  2) ExponentialLR:       指数衰减 (平滑递减)
  3) Cosine Annealing:    余弦退火 (平滑, 常用)
  4) Warmup + Cosine:     热身 + 衰减 (Transformer 标配)
  5) OneCycleLR:          单周期策略 (快速收敛)
  6) ReduceLROnPlateau:   自适应衰减 (按需调整)

  实践建议:
    - Transformer 微调: Warmup + Linear/Cosine Decay
    - 从头训练 CNN:     OneCycleLR
    - 不确定时:         ReduceLROnPlateau 作为安全选项

  下一步: d2_data_augmentation.py - 数据增强技术
""")
