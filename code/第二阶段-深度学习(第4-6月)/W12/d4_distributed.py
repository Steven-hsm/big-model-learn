"""
W12-D4: DataParallel vs DistributedDataParallel, Accelerate, DDP config template
==================================================================================
演示 PyTorch 分布式训练的概念和配置方式。
注意：本文件以代码示例和说明为主，部分代码需要多 GPU 环境才能运行。
"""

import torch
import torch.nn as nn


# ============================================================
# 1. DataParallel (DP) — 简单但效率较低
# ============================================================
def demo_dataparallel():
    """
    DataParallel 使用说明（需要多 GPU 才能运行）
    """
    print("=" * 60)
    print("1. DataParallel (DP)")
    print("=" * 60)

    print("DataParallel 工作原理:")
    print("  1. 在 GPU 0 上复制模型到所有 GPU")
    print("  2. 将 batch 分割发送到各 GPU")
    print("  3. 各 GPU 并行前向传播")
    print("  4. 在 GPU 0 上收集并计算 loss")
    print("  5. 在 GPU 0 上反向传播，分发梯度到各 GPU")
    print()

    print("代码示例:")
    print("""
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )

    # 单行代码启用多 GPU
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])

    # 使用方式和单 GPU 完全一致
    output = model(input_data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    """)

    print("DataParallel 的缺点:")
    print("  - GPU 0 负载不均衡（收集/分发开销）")
    print("  - GIL 限制，无法真正异步")
    print("  - 仅支持单机多卡")
    print("  - 推荐: 仅用于快速验证，生产环境用 DDP")
    print()


# ============================================================
# 2. DistributedDataParallel (DDP) — 推荐方式
# ============================================================
def demo_ddp_concept():
    """
    DDP 概念和配置模板
    """
    print("=" * 60)
    print("2. DistributedDataParallel (DDP)")
    print("=" * 60)

    print("DDP 工作原理:")
    print("  1. 每个 GPU 有独立的进程（避免 GIL）")
    print("  2. 每个 GPU 持有模型副本")
    print("  3. 各进程独立前向/反向传播")
    print("  4. 通过 Ring-AllReduce 同步梯度")
    print("  5. 各进程独立更新参数（参数保持一致）")
    print()

    print("DDP 配置模板:")
    print("""
    import os
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, DistributedSampler

    def setup(rank, world_size):
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    def cleanup():
        dist.destroy_process_group()

    def train(rank, world_size):
        setup(rank, world_size)

        # 创建模型并包装为 DDP
        model = nn.Linear(784, 10).to(rank)
        ddp_model = DDP(model, device_ids=[rank])

        # 使用 DistributedSampler 确保数据不重叠
        dataset = MyDataset()
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        loader = DataLoader(dataset, batch_size=32, sampler=sampler)

        optimizer = torch.optim.Adam(ddp_model.parameters(), lr=1e-3)

        for epoch in range(num_epochs):
            sampler.set_epoch(epoch)  # 每个 epoch 设置以保证 shuffle
            for batch in loader:
                optimizer.zero_grad()
                output = ddp_model(batch['x'].to(rank))
                loss = criterion(output, batch['y'].to(rank))
                loss.backward()
                optimizer.step()

        cleanup()

    # 启动方式: torchrun --nproc_per_node=4 train.py
    """)

    print("DDP 启动命令:")
    print("  单机多卡: torchrun --nproc_per_node=4 train.py")
    print("  多机多卡: torchrun --nnodes=2 --nproc_per_node=4 --master_addr=xxx train.py")
    print()


# ============================================================
# 3. DDP 完整配置模板（可直接使用）
# ============================================================
DDP_TEMPLATE = '''
"""
DDP 训练模板 - 可以直接复制使用
启动: torchrun --nproc_per_node=NUM_GPUS ddp_template.py
"""
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.net(x)


def main():
    # 从环境变量获取 rank 和 world_size（torchrun 自动设置）
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

    # 创建模型
    model = SimpleModel().to(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank])

    # 创建数据
    dataset = TensorDataset(torch.randn(1000, 784), torch.randint(0, 10, (1000,)))
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=32, sampler=sampler)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=1e-3)

    # 训练循环
    for epoch in range(10):
        sampler.set_epoch(epoch)
        ddp_model.train()
        total_loss = 0.0

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(local_rank)
            batch_y = batch_y.to(local_rank)

            optimizer.zero_grad()
            output = ddp_model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if rank == 0:
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch}: loss = {avg_loss:.4f}")

    # 保存检查点（只在 rank 0 保存）
    if rank == 0:
        torch.save(ddp_model.module.state_dict(), "checkpoint.pt")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
'''


def demo_ddp_template():
    print("=" * 60)
    print("3. DDP 完整训练模板")
    print("=" * 60)
    print("以下是一个可以直接使用的 DDP 训练模板:")
    print("-" * 40)
    for line in DDP_TEMPLATE.strip().split("\n"):
        print(line)
    print("-" * 40)
    print()


# ============================================================
# 4. Accelerate 库 — 简化分布式训练
# ============================================================
def demo_accelerate():
    """
    Accelerate 库使用示例
    """
    print("=" * 60)
    print("4. Accelerate 库")
    print("=" * 60)

    print("Accelerate 优��:")
    print("  - 一行代码适配 CPU / 单GPU / 多GPU / TPU")
    print("  - 自动处理混合精度、梯度累积、分布式采样")
    print("  - 无需手动设置 rank、world_size 等")
    print()

    print("Accelerate 代码示例:")
    print("""
    from accelerate import Accelerator

    # 初始化 Accelerator（自动检测环境）
    accelerator = Accelerator(
        mixed_precision="fp16",        # 混合精度
        gradient_accumulation_steps=4, # 梯度累积
    )

    # 准备模型、优化器、数据
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 一行包装！自动处理分布式
    model, optimizer, train_loader = accelerator.prepare(
        model, optimizer, train_loader
    )

    # 训练循环
    for epoch in range(num_epochs):
        for batch_x, batch_y in train_loader:
            output = model(batch_x)
            loss = criterion(output, batch_y)

            # 使用 accelerator.backward() 替代 loss.backward()
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

    # 保存模型（自动处理分布式）
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        torch.save(unwrapped.state_dict(), "model.pt")
    """)

    print("启动命令:")
    print("  accelerate launch train.py")
    print("  accelerate config  # 交互式配置")
    print()


# ============================================================
# 5. DataParallel vs DDP 对比
# ============================================================
def demo_comparison():
    print("=" * 60)
    print("5. DataParallel vs DDP 对比")
    print("=" * 60)

    comparison = {
        "特性":           ["DP",              "DDP"],
        "进程模型":       ["单进程多线程",    "多进程"],
        "GIL 影响":       ["受 GIL 限制",     "无 GIL 限制"],
        "负载均衡":       ["不均衡(GPU0瓶颈)", "均衡"],
        "多机支持":       ["不支持",          "支持"],
        "通信方式":       ["线程间拷贝",      "Ring-AllReduce"],
        "启动方式":       ["直接 python",     "torchrun"],
        "代码改动":       ["极少",            "中等"],
        "推荐场景":       ["快速验证",        "生产训练"],
        "性能":           ["~60-70%线性",     "~90%线性"],
    }

    # 格式化输出
    headers = list(comparison.keys())
    col_widths = [max(len(str(v)) for v in [h] + comparison[h]) for h in headers]

    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))

    for i in range(len(comparison["特性"])):
        row = " | ".join(
            str(comparison[h][i]).ljust(w) for h, w in zip(headers, col_widths)
        )
        print(row)

    print()
    print("选择建议:")
    print("  快速原型/调试:  DataParallel (一行代码)")
    print("  正式训练:       DDP (性能更好)")
    print("  想省事:         Accelerate (推荐!)")
    print()


# ============================================================
# 6. GPU 信息查看
# ============================================================
def demo_gpu_info():
    print("=" * 60)
    print("6. GPU 信息查看")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"CUDA 可用: True")
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            print(f"    显存: {props.total_mem / 1024**3:.1f} GB")
            print(f"    多处理器数: {props.multi_processor_count}")
    else:
        print("CUDA 不可用（当前环境无 GPU）")

    print()


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    demo_dataparallel()
    demo_ddp_concept()
    demo_ddp_template()
    demo_accelerate()
    demo_comparison()
    demo_gpu_info()

    print("所有演示完成！")
