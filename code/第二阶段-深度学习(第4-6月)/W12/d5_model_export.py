"""
W12-D5: TorchScript, ONNX export, TensorBoard, wandb integration
=================================================================
演示模型导出和训练监控工具。
"""

import torch
import torch.nn as nn
import io
import tempfile
import os


# ============================================================
# 1. TorchScript: trace 和 script
# ============================================================
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class ModelWithControlFlow(nn.Module):
    """包含控制流的模型 — 需要 script 而非 trace"""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x, use_relu=True):
        x = self.fc(x)
        if use_relu:     # 控制流 — trace 无法捕获
            x = torch.relu(x)
        return x


def demo_torchscript():
    print("=" * 60)
    print("1. TorchScript (trace + script)")
    print("=" * 60)

    # --- Trace: 记录一次前向传播的操作 ---
    print("--- Trace 方式 ---")
    model = SimpleModel()
    model.eval()

    example_input = torch.randn(1, 784)
    traced_model = torch.jit.trace(model, example_input)

    # 验证输出一致
    with torch.no_grad():
        original_out = model(example_input)
        traced_out = traced_model(example_input)
    print(f"Trace 输出一致: {torch.allclose(original_out, traced_out)}")

    # 保存和加载
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        traced_model.save(f.name)
        loaded = torch.jit.load(f.name)
        print(f"保存后加载输出一致: {torch.allclose(original_out, loaded(example_input))}")
        os.unlink(f.name)

    # 查看 traced 代码
    print(f"Traced 代码:\n{traced_model.code}")

    # --- Script: 分析 Python 源码 ---
    print("\n--- Script 方式 ---")
    ctrl_model = ModelWithControlFlow()
    scripted_model = torch.jit.script(ctrl_model)

    x = torch.randn(1, 10)
    out_relu = scripted_model(x, use_relu=True)
    out_no_relu = scripted_model(x, use_relu=False)
    print(f"use_relu=True 输出非负: {(out_relu >= 0).all().item()}")
    print(f"use_relu=False 输出有负: {(out_no_relu < 0).any().item()}")
    print(f"Scripted 代码:\n{scripted_model.code}")
    print()


# ============================================================
# 2. ONNX 导出
# ============================================================
def demo_onnx_export():
    print("=" * 60)
    print("2. ONNX 导出")
    print("=" * 60)

    model = SimpleModel()
    model.eval()

    dummy_input = torch.randn(1, 784)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,        # 包含模型参数
        opset_version=14,          # ONNX opset 版本
        do_constant_folding=True,  # 优化常量折叠
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={             # 支持动态 batch size
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    print(f"ONNX 模型已导出到: {onnx_path}")
    print(f"文件大小: {os.path.getsize(onnx_path)} bytes")

    # 验证 ONNX 模型
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX 模型验证通过!")

        # 打印模型信息
        print(f"输入: {[inp.name for inp in onnx_model.graph.input]}")
        print(f"输出: {[out.name for out in onnx_model.graph.output]}")
    except ImportError:
        print("onnx 库未安装，跳过验证 (pip install onnx)")

    os.unlink(onnx_path)
    print()


# ============================================================
# 3. TensorBoard SummaryWriter
# ============================================================
def demo_tensorboard():
    print("=" * 60)
    print("3. TensorBoard SummaryWriter")
    print("=" * 60)

    from torch.utils.tensorboard import SummaryWriter

    # 创建临时目录
    log_dir = os.path.join(tempfile.gettempdir(), "tensorboard_demo")
    writer = SummaryWriter(log_dir=log_dir)

    # 模拟训练过程
    model = SimpleModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(20):
        x = torch.randn(32, 784)
        y = torch.randint(0, 10, (32,))

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        # 记录 scalar
        writer.add_scalar("Loss/train", loss.item(), step)

        # 记录学习率
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], step)

        # 记录参数直方图
        for name, param in model.named_parameters():
            writer.add_histogram(f"Params/{name}", param, step)

    # 记录模型结构
    sample_input = torch.randn(1, 784)
    writer.add_graph(model, sample_input)

    # 记录文本
    writer.add_text("config", "SimpleModel: 784->256->10, Adam lr=1e-3")

    writer.close()
    print(f"TensorBoard 日志已写入: {log_dir}")
    print(f"查看命令: tensorboard --logdir={log_dir}")
    print()


# ============================================================
# 4. wandb (Weights & Biases) 集成示例
# ============================================================
def demo_wandb():
    print("=" * 60)
    print("4. wandb 集成示例")
    print("=" * 60)

    print("wandb 是一个强大的实验跟踪工具，代码示例:")
    print("""
    import wandb

    # 1. 初始化 wandb
    wandb.init(
        project="my-project",
        config={
            "learning_rate": 1e-3,
            "architecture": "SimpleMLP",
            "dataset": "MNIST",
            "epochs": 10,
        },
    )

    # 2. 训练循环中记录指标
    for epoch in range(config.epochs):
        train_loss = train_one_epoch()
        val_loss, val_acc = evaluate()

        # 记录指标到 wandb
        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "epoch": epoch,
        })

    # 3. 记录模型
    artifact = wandb.Artifact("model", type="model")
    artifact.add_file("model.pt")
    wandb.log_artifact(artifact)

    # 4. 结束
    wandb.finish()
    """)

    print("wandb 特点:")
    print("  - 自动记录和可视化训练曲线")
    print("  - 超参数比较和搜索")
    print("  - 团队协作和实验共享")
    print("  - 模型版本管理")
    print("  - 安装: pip install wandb")
    print()


# ============================================================
# 5. 完整训练监控模板
# ============================================================
def demo_full_monitoring_template():
    print("=" * 60)
    print("5. 完整训练监控模板（TensorBoard + 文件日志）")
    print("=" * 60)

    from torch.utils.tensorboard import SummaryWriter

    log_dir = os.path.join(tempfile.gettempdir(), "full_monitoring_demo")
    writer = SummaryWriter(log_dir=log_dir)

    model = SimpleModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_loss = float("inf")

    for epoch in range(10):
        # 模拟训练
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for _ in range(5):
            x = torch.randn(16, 784)
            y = torch.randint(0, 10, (16,))

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            correct += (output.argmax(1) == y).sum().item()
            total += len(y)

        train_loss /= 5
        train_acc = correct / total

        # 模拟验证
        model.eval()
        with torch.no_grad():
            x_val = torch.randn(32, 784)
            y_val = torch.randint(0, 10, (32,))
            val_output = model(x_val)
            val_loss = criterion(val_output, y_val).item()
            val_acc = (val_output.argmax(1) == y_val).float().mean().item()

        scheduler.step()

        # TensorBoard 记录
        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc}, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        # 控制台输出
        print(
            f"Epoch {epoch:2d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # 检查点保存
        if val_loss < best_loss:
            best_loss = val_loss
            print(f"  -> 新最佳模型! Val Loss: {val_loss:.4f}")

    writer.close()
    print(f"\n日志目录: {log_dir}")
    print()


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)

    demo_torchscript()
    demo_onnx_export()
    demo_tensorboard()
    demo_wandb()
    demo_full_monitoring_template()

    print("所有演示完成！")
