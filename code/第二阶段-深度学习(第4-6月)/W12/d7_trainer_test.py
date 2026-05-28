"""
W12-D7: Test Trainer on MNIST, demonstrate all features, TensorBoard logging
============================================================================
使用 Trainer 类训练 MNIST，展示所有功能。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os
import tempfile
import time

# 导入 Trainer 和 TrainerConfig
# 这里直接内联关键代码以避免导入问题
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d6_trainer_class import Trainer, TrainerConfig


# ============================================================
# 定义模型
# ============================================================
class MNISTModel(nn.Module):
    """简单的 MNIST 分类模型"""

    def __init__(self, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 10),
        )

    def forward(self, x):
        return self.net(self.flatten(x))


# ============================================================
# 配置模板
# ============================================================
def get_default_config():
    """返回默认训练配置"""
    return {
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 5,
        "batch_size": 64,
        "accumulation_steps": 2,      # 等效 batch_size=128
        "max_grad_norm": 1.0,
        "early_stopping_patience": 3,
        "use_amp": False,              # CPU 上设 False
        "log_interval": 50,
        "device": "auto",
    }


# ============================================================
# 主测试流程
# ============================================================
def main():
    torch.manual_seed(42)

    print("=" * 70)
    print("Trainer 类完整测试 — MNIST 数据集")
    print("=" * 70)

    # --- 1. 准备数据 ---
    print("\n[1] 准备 MNIST 数据...")

    data_dir = os.path.join(tempfile.gettempdir(), "mnist_data")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    # 下载并加载 MNIST
    try:
        full_train = datasets.MNIST(
            data_dir, train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            data_dir, train=False, download=True, transform=transform
        )
    except Exception as e:
        print(f"MNIST 下载失败 ({e})，使用随机数据演示...")
        # 使用随机数据代替
        from torch.utils.data import TensorDataset

        full_train = TensorDataset(
            torch.randn(1000, 1, 28, 28),
            torch.randint(0, 10, (1000,)),
        )
        test_dataset = TensorDataset(
            torch.randn(200, 1, 28, 28),
            torch.randint(0, 10, (200,)),
        )

    # 划分训练集和验证集
    val_size = int(len(full_train) * 0.1)
    train_size = len(full_train) - val_size
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size])

    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    print(f"测试集: {len(test_dataset)} 样本")

    # --- 2. 创建配置 ---
    print("\n[2] 创建训练配置...")
    config = TrainerConfig(get_default_config())
    print(config)

    # --- 3. 创建数据加载器 ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    # --- 4. 创建模型和训练器 ---
    print("\n[3] 创建模型...")
    model = MNISTModel(hidden_dim=256, dropout=0.2)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params:,} (可训练: {trainable_params:,})")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    trainer = Trainer(
        config=config,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
    )
    trainer.set_dataloaders(train_loader, val_loader)

    # --- 5. 设置学习率调度器 ---
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-5
    )
    trainer.set_scheduler(scheduler)

    # --- 6. 设置 TensorBoard ---
    try:
        from torch.utils.tensorboard import SummaryWriter

        log_dir = os.path.join(tempfile.gettempdir(), "mnist_trainer_demo")
        writer = SummaryWriter(log_dir=log_dir)
        trainer.set_writer(writer)
        print(f"\n[4] TensorBoard 日志目录: {log_dir}")
    except ImportError:
        print("\n[4] TensorBoard 未安装，跳过日志记录")

    # --- 7. 训练 ---
    print("\n[5] 开始训练...")
    start_time = time.time()
    history = trainer.fit()
    train_time = time.time() - start_time
    print(f"训练耗时: {train_time:.1f}s")

    # --- 8. 在测试集上评估 ---
    print("\n[6] 测试集评估...")
    test_loss, test_acc = trainer.evaluate(test_loader)
    print(f"测试集 Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}")

    # --- 9. 保存和加载模型 ---
    print("\n[7] 保存和加载检查点...")
    ckpt_dir = os.path.join(tempfile.gettempdir(), "mnist_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
    trainer.save_checkpoint(ckpt_path)

    # 创建新训练器并加载
    new_model = MNISTModel(hidden_dim=256, dropout=0.2)
    new_trainer = Trainer(
        config=config,
        model=new_model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optim.AdamW(new_model.parameters(), lr=config.lr),
    )
    new_trainer.load_checkpoint(ckpt_path)
    test_loss2, test_acc2 = new_trainer.evaluate(test_loader)
    print(f"加载后测试结果: Loss={test_loss2:.4f} Acc={test_acc2:.4f}")
    print(f"结果一致: {abs(test_loss - test_loss2) < 1e-6 and abs(test_acc - test_acc2) < 1e-6}")

    # --- 10. 展示训练历史 ---
    print("\n[8] 训练历史:")
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | {'Val Acc':>10}")
    print("-" * 45)
    for i in range(len(history["train_loss"])):
        t_loss = history["train_loss"][i]
        v_loss = history["val_loss"][i] if i < len(history["val_loss"]) else float("nan")
        v_acc = history["val_accuracy"][i] if i < len(history["val_accuracy"]) else float("nan")
        print(f"{i + 1:>5} | {t_loss:>10.4f} | {v_loss:>10.4f} | {v_acc:>10.4f}")

    # 清理
    if trainer.writer:
        trainer.writer.close()

    print("\n" + "=" * 70)
    print("Trainer 类所有功能测试完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
