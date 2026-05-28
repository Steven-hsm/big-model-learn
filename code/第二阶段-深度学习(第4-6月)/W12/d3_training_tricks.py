"""
W12-D3: Learning rate warmup, gradient accumulation, mixed precision (AMP), gradient clipping
=============================================================================================
演示 PyTorch 训练中的常用技巧。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import math


# ============================================================
# 1. Learning Rate Warmup Scheduler
# ============================================================
def get_warmup_scheduler(optimizer, warmup_steps, total_steps):
    """
    创建带 warmup 的学习率调度器
    先线性 warmup，然后 cosine decay
    """
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # 线性 warmup: lr 从 0 线性增长到 base_lr
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine decay
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def demo_lr_warmup():
    print("=" * 60)
    print("1. Learning Rate Warmup 调度器")
    print("=" * 60)

    model = nn.Linear(10, 2)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    warmup_steps = 100
    total_steps = 1000
    scheduler = get_warmup_scheduler(optimizer, warmup_steps, total_steps)

    # 记录学习率变化
    steps_to_check = [0, 10, 50, 99, 100, 200, 500, 999]
    lrs = {}

    for step in range(total_steps):
        if step in steps_to_check:
            lrs[step] = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()

    print(f"{'Step':>6} | {'LR':>12} | 阶段")
    print("-" * 40)
    for step in steps_to_check:
        phase = "warmup" if step < warmup_steps else "decay"
        print(f"{step:>6} | {lrs[step]:>12.6f} | {phase}")

    print(f"\nWarmup 阶段: 0 -> {warmup_steps} 步，LR 从 0 线性增长")
    print(f"Decay 阶段: {warmup_steps} -> {total_steps} 步，余弦衰减到 0")
    print()


# ============================================================
# 2. Gradient Accumulation
# ============================================================
def demo_gradient_accumulation():
    print("=" * 60)
    print("2. 梯度累积")
    print("=" * 60)

    model = nn.Linear(10, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 模拟数据
    data = [(torch.randn(4, 10), torch.randint(0, 2, (4,))) for _ in range(8)]

    accumulation_steps = 4  # 每 accumulation_steps 个 batch 更新一次

    # --- 不使用梯度累积（等效 batch_size=32）---
    # 重置模型和优化器
    torch.manual_seed(42)
    model_ref = nn.Linear(10, 2)
    model_ref.load_state_dict(model.state_dict())
    optimizer_ref = optim.SGD(model_ref.parameters(), lr=0.01)

    # 拼接所有数据为一个大批次
    all_x = torch.cat([d[0] for d in data], dim=0)  # [32, 10]
    all_y = torch.cat([d[1] for d in data], dim=0)  # [32]

    optimizer_ref.zero_grad()
    loss_ref = criterion(model_ref(all_x), all_y)
    loss_ref.backward()
    optimizer_ref.step()

    # --- 使用梯度累积（等效效果）---
    optimizer.zero_grad()
    for i, (x, y) in enumerate(data):
        loss = criterion(model(x), y) / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    # 比较权重更新
    ref_weight = model_ref.weight.data
    acc_weight = model.weight.data
    diff = (ref_weight - acc_weight).abs().max().item()
    print(f"不使用累积的 loss: {loss_ref.item():.6f}")
    print(f"梯度累积后权重与大批次最大差异: {diff:.8f}")
    print(f"说明: 差异很小是因为计算顺序不同导致的浮点误差")
    print()


# ============================================================
# 3. Mixed Precision Training (AMP)
# ============================================================
def demo_mixed_precision():
    print("=" * 60)
    print("3. 混合精度训练 (AMP)")
    print("=" * 60)

    device = torch.device("cpu")  # 演示用 CPU
    model = nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 创建 GradScaler（用于 FP16 训练时缩放梯度防止下溢）
    scaler = torch.amp.GradScaler(device)

    # 模拟数据
    x = torch.randn(32, 784, device=device)
    y = torch.randint(0, 10, (32,), device=device)

    # --- AMP 训练一个 step ---
    model.train()
    optimizer.zero_grad()

    # 使用 autocast 自动混合精度
    with torch.amp.autocast(device):
        output = model(x)
        loss = criterion(output, y)

    # 使用 scaler 缩放 loss 并反向传播
    scaler.scale(loss).backward()

    # 查看梯度
    scaler.unscale_(optimizer)

    # 检查梯度是否包含 inf/nan
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()

    # 使用 scaler 更新参数
    scaler.step(optimizer)
    scaler.update()

    print(f"Loss: {loss.item():.4f}")
    print(f"梯度范数:")
    for name, norm in grad_norms.items():
        print(f"  {name}: {norm:.4f}")

    print(f"\nAMP 关键步骤:")
    print(f"  1. autocast: 自动将合适的操作转为 FP16")
    print(f"  2. scaler.scale(loss): 缩放 loss 防止 FP16 梯度下溢")
    print(f"  3. scaler.step(): 自动 unscale 并更新参数")
    print(f"  4. scaler.update(): 更新缩放因子")
    print()


# ============================================================
# 4. Gradient Clipping
# ============================================================
def demo_gradient_clipping():
    print("=" * 60)
    print("4. 梯度裁剪")
    print("=" * 60)

    model = nn.Sequential(
        nn.Linear(100, 200),
        nn.ReLU(),
        nn.Linear(200, 10),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 生成一个会产生大梯度的输入
    x = torch.randn(16, 100) * 10  # 放大输入
    y = torch.randint(0, 10, (16,))

    # 计算梯度
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()

    # 裁剪前的梯度范数
    total_norm_before = torch.sqrt(
        sum(p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None)
    )
    print(f"裁剪前梯度总范数: {total_norm_before.item():.4f}")

    # 方式一: clip_grad_norm_ (按范数裁剪) — 最常用
    max_norm = 1.0
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    print(f"clip_grad_norm_ 后梯度范数: {grad_norm:.4f} (限制: {max_norm})")

    # 重新计算梯度
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()

    # 方式二: clip_grad_value_ (按值裁剪)
    clip_value = 0.5
    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)

    # 检查梯度最大值
    max_grad = max(
        p.grad.abs().max().item() for p in model.parameters() if p.grad is not None
    )
    print(f"clip_grad_value_ 后梯度最大绝对值: {max_grad:.4f} (限制: {clip_value})")

    print(f"\n常用选择:")
    print(f"  clip_grad_norm_:  推荐用于 Transformer（限制梯度总范数）")
    print(f"  clip_grad_value_: 推荐用于 RNN（限制每个梯度元素）")
    print()


# ============================================================
# 5. 综合训练循环模板
# ============================================================
def demo_training_loop_template():
    print("=" * 60)
    print("5. 综合训练循环模板（包含所有技巧）")
    print("=" * 60)

    # 配置
    config = {
        "lr": 5e-4,
        "warmup_steps": 5,
        "total_steps": 20,
        "accumulation_steps": 2,
        "max_grad_norm": 1.0,
    }

    model = nn.Linear(10, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
    scheduler = get_warmup_scheduler(
        optimizer, config["warmup_steps"], config["total_steps"]
    )
    scaler = torch.amp.GradScaler("cpu")

    print(f"{'Step':>4} | {'Loss':>8} | {'LR':>10} | {'GradNorm':>10}")
    print("-" * 45)

    for step in range(config["total_steps"]):
        # 模拟数据
        x = torch.randn(8, 10)
        y = torch.randint(0, 2, (8,))

        # AMP autocast
        with torch.amp.autocast("cpu"):
            loss = criterion(model(x), y) / config["accumulation_steps"]

        # Scaler backward
        scaler.scale(loss).backward()

        # 梯度累积完成时更新
        if (step + 1) % config["accumulation_steps"] == 0:
            # Unscale 以进行梯度裁剪
            scaler.unscale_(optimizer)

            # 梯度裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["max_grad_norm"]
            )

            # 更新参数
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"{step:>4} | {loss.item() * config['accumulation_steps']:>8.4f} | "
            f"{current_lr:>10.6f} | {grad_norm if (step + 1) % config['accumulation_steps'] == 0 else 0:>10.4f}"
        )

    print()


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)

    demo_lr_warmup()
    demo_gradient_accumulation()
    demo_mixed_precision()
    demo_gradient_clipping()
    demo_training_loop_template()

    print("所有演示完成！")
