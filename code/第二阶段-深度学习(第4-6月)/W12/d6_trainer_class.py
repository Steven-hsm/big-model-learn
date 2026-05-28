"""
W12-D6: Generic Trainer Class
==============================
一个通用的训练器类，支持: AMP、梯度累积、学习率调度、早停、检查点保存/加载。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import copy
import time


# ============================================================
# Trainer 配置类
# ============================================================
class TrainerConfig:
    """训练配置"""

    def __init__(self, config_dict=None, **kwargs):
        # 默认配置
        defaults = {
            "lr": 1e-3,
            "weight_decay": 0.0,
            "epochs": 10,
            "batch_size": 32,
            "accumulation_steps": 1,
            "max_grad_norm": 1.0,
            "warmup_steps": 0,
            "early_stopping_patience": 0,  # 0 = 不启用
            "use_amp": False,
            "device": "auto",
            "checkpoint_dir": "checkpoints",
            "log_interval": 10,
        }

        # 从字典更新
        if config_dict:
            defaults.update(config_dict)
        defaults.update(kwargs)

        for key, value in defaults.items():
            setattr(self, key, value)

    def __repr__(self):
        lines = ["TrainerConfig("]
        for k, v in sorted(self.__dict__.items()):
            lines.append(f"  {k} = {v!r},")
        lines.append(")")
        return "\n".join(lines)


# ============================================================
# Trainer 类
# ============================================================
class Trainer:
    """
    通用训练器

    用法:
        trainer = Trainer(config, model, criterion, optimizer)
        trainer.set_dataloaders(train_loader, val_loader)
        trainer.fit()
    """

    def __init__(self, config, model=None, criterion=None, optimizer=None):
        self.config = config if isinstance(config, TrainerConfig) else TrainerConfig(config)

        # 设置设备
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        # 模型
        self.model = None
        self.criterion = criterion
        self.optimizer = optimizer

        if model is not None:
            self.set_model(model)

        # 调度器
        self.scheduler = None

        # AMP scaler
        self.scaler = torch.amp.GradScaler(self.device.type) if self.config.use_amp else None

        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.best_model_state = None
        self.history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

        # TensorBoard writer（可选）
        self.writer = None

    def set_model(self, model):
        """设置模型"""
        self.model = model.to(self.device)
        # 同步优化器（如果模型改变了）
        if self.optimizer is not None:
            self.optimizer = type(self.optimizer)(
                self.model.parameters(),
                lr=self.optimizer.defaults.get("lr", self.config.lr),
                **{k: v for k, v in self.optimizer.defaults.items() if k != "lr"},
            )

    def set_criterion(self, criterion):
        """设置损失函数"""
        self.criterion = criterion

    def set_optimizer(self, optimizer):
        """设置优化器"""
        self.optimizer = optimizer

    def set_scheduler(self, scheduler):
        """设置学习率调度器"""
        self.scheduler = scheduler

    def set_dataloaders(self, train_loader, val_loader=None):
        """设置训练和验证数据加载器"""
        self.train_loader = train_loader
        self.val_loader = val_loader

    def set_writer(self, writer):
        """设置 TensorBoard writer"""
        self.writer = writer

    # ----------------------------------------------------------
    # 训练一个 epoch
    # ----------------------------------------------------------
    def train_epoch(self):
        """训练一个 epoch，返回平均 loss"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        self.optimizer.zero_grad()

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # 前向传播（可选 AMP）
            if self.config.use_amp:
                with torch.amp.autocast(self.device.type):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets) / self.config.accumulation_steps
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets) / self.config.accumulation_steps
                loss.backward()

            # 梯度累积
            if (batch_idx + 1) % self.config.accumulation_steps == 0:
                if self.config.use_amp:
                    self.scaler.unscale_(self.optimizer)

                # 梯度裁剪
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )

                if self.config.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.global_step += 1

            total_loss += loss.item() * self.config.accumulation_steps
            num_batches += 1

            # 日志记录
            if (batch_idx + 1) % self.config.log_interval == 0:
                avg = total_loss / num_batches
                if self.writer:
                    self.writer.add_scalar("Loss/train_batch", avg, self.global_step)

        # 处理最后不完整的累积步
        if num_batches % self.config.accumulation_steps != 0:
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
            if self.config.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()

        return total_loss / num_batches

    # ----------------------------------------------------------
    # 评估
    # ----------------------------------------------------------
    def evaluate(self, loader=None):
        """评估模型，返回 (loss, accuracy)"""
        if loader is None:
            loader = self.val_loader

        if loader is None:
            return None, None

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                if self.config.use_amp:
                    with torch.amp.autocast(self.device.type):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                preds = outputs.argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += len(targets)

        avg_loss = total_loss / len(loader)
        accuracy = correct / total if total > 0 else 0.0

        return avg_loss, accuracy

    # ----------------------------------------------------------
    # 完整训练
    # ----------------------------------------------------------
    def fit(self, train_loader=None, val_loader=None):
        """
        完整训练流程

        参数:
            train_loader: 训练数据（可选，如果已 set_dataloaders）
            val_loader: 验证数据（可选）
        返回:
            history: 训练历史记录
        """
        if train_loader is not None:
            self.train_loader = train_loader
        if val_loader is not None:
            self.val_loader = val_loader

        patience_counter = 0

        print(f"开始训练 | 设备: {self.device} | Epochs: {self.config.epochs}")
        print("-" * 70)

        for epoch in range(self.current_epoch, self.config.epochs):
            start_time = time.time()

            # 训练
            train_loss = self.train_epoch()
            self.history["train_loss"].append(train_loss)

            # 验证
            val_loss, val_acc = self.evaluate()
            if val_loss is not None:
                self.history["val_loss"].append(val_loss)
                self.history["val_accuracy"].append(val_acc)

            # 学习率调度
            if self.scheduler is not None:
                self.scheduler.step()

            elapsed = time.time() - start_time
            current_lr = self.optimizer.param_groups[0]["lr"]

            # 输出
            val_str = f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}" if val_loss else "No validation"
            print(
                f"Epoch {epoch + 1:3d}/{self.config.epochs} | "
                f"Train Loss: {train_loss:.4f} | {val_str} | "
                f"LR: {current_lr:.2e} | Time: {elapsed:.1f}s"
            )

            # TensorBoard
            if self.writer:
                self.writer.add_scalar("Loss/train_epoch", train_loss, epoch)
                if val_loss is not None:
                    self.writer.add_scalar("Loss/val_epoch", val_loss, epoch)
                    self.writer.add_scalar("Accuracy/val", val_acc, epoch)
                self.writer.add_scalar("LR", current_lr, epoch)

            # 早停和最佳模型
            if val_loss is not None:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_model_state = copy.deepcopy(self.model.state_dict())
                    patience_counter = 0
                    print(f"  -> 新最佳模型! Val Loss: {val_loss:.4f}")
                else:
                    patience_counter += 1

                if self.config.early_stopping_patience > 0:
                    if patience_counter >= self.config.early_stopping_patience:
                        print(f"\n早停! 连续 {patience_counter} 个 epoch 验证 loss 未改善")
                        break

            self.current_epoch = epoch + 1

        # 恢复最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"\n已恢复最佳模型 (Val Loss: {self.best_val_loss:.4f})")

        print("-" * 70)
        print("训练完成!")
        return self.history

    # ----------------------------------------------------------
    # 保存和加载检查点
    # ----------------------------------------------------------
    def save_checkpoint(self, filepath):
        """保存完整检查点"""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)

        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "history": self.history,
            "config": self.config.__dict__,
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, filepath)
        print(f"检查点已保存: {filepath}")

    def load_checkpoint(self, filepath):
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.history = checkpoint["history"]

        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        print(f"检查点已加载: {filepath} (Epoch {self.current_epoch})")

    def save_model(self, filepath):
        """只保存模型权重"""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        torch.save(self.model.state_dict(), filepath)
        print(f"模型权重已保存: {filepath}")

    def load_model(self, filepath):
        """只加载模型权重"""
        self.model.load_state_dict(torch.load(filepath, map_location=self.device, weights_only=False))
        print(f"模型权重已加载: {filepath}")


# ============================================================
# 简单测试
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)

    # 创建简单数据集
    class SimpleDataset(Dataset):
        def __init__(self, n=500):
            self.x = torch.randn(n, 784)
            self.y = (self.x[:, 0] + self.x[:, 1] > 0).long()

        def __len__(self):
            return len(self.x)

        def __getitem__(self, idx):
            return self.x[idx], self.y[idx]

    # 配置
    config = TrainerConfig({
        "lr": 1e-3,
        "epochs": 5,
        "batch_size": 32,
        "accumulation_steps": 2,
        "max_grad_norm": 1.0,
        "early_stopping_patience": 3,
        "use_amp": False,
        "log_interval": 5,
    })
    print(config)
    print()

    # 模型
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 2),
    )

    # 数据
    train_dataset = SimpleDataset(500)
    val_dataset = SimpleDataset(100)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    # 训练器
    trainer = Trainer(
        config=config,
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optim.Adam(model.parameters(), lr=config.lr),
    )
    trainer.set_dataloaders(train_loader, val_loader)

    # 训练
    history = trainer.fit()

    # 保存检查点
    trainer.save_checkpoint("temp_checkpoint.pt")

    # 加载检查点并继续训练
    trainer.load_checkpoint("temp_checkpoint.pt")
    print("\n从检查点继续训练 2 个 epoch...")
    trainer.config.epochs = trainer.current_epoch + 2
    trainer.fit()

    # 清理
    os.remove("temp_checkpoint.pt")

    print("\nTrainer 类测试完成!")
