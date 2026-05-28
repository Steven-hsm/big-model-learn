"""
W12-D1: Custom Dataset with collate_fn, DataLoader optimization, WeightedRandomSampler
=====================================================================================
演示 PyTorch 中自定义数据集、DataLoader 优化技巧和加权采样器的用法。
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from collections import Counter
import random


# ============================================================
# 1. 自定义 Dataset
# ============================================================
class CustomTextDataset(Dataset):
    """模拟一个文本分类数据集，支持变长序列"""

    def __init__(self, num_samples=1000, max_len=50, vocab_size=5000, num_classes=3):
        super().__init__()
        self.num_samples = num_samples
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.num_classes = num_classes

        # 生成变长序列（模拟真实文本数据）
        self.data = []
        self.labels = []
        self.lengths = []

        for _ in range(num_samples):
            seq_len = random.randint(5, max_len)
            tokens = torch.randint(1, vocab_size, (seq_len,))
            label = random.randint(0, num_classes - 1)
            self.data.append(tokens)
            self.labels.append(label)
            self.lengths.append(seq_len)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "tokens": self.data[idx],
            "label": self.labels[idx],
            "length": self.lengths[idx],
        }


# ============================================================
# 2. collate_fn: 处理变长序列的 batch 整理函数
# ============================================================
def pad_collate_fn(batch):
    """
    自定义 collate_fn：将变长序列 pad 到同一长度
    batch 是一个 list，每个元素是 __getitem__ 返回的 dict
    """
    tokens_list = [item["tokens"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch])
    lengths = torch.tensor([item["length"] for item in batch])

    # Pad sequences to the same length within the batch
    padded_tokens = torch.nn.utils.rnn.pad_sequence(
        tokens_list, batch_first=True, padding_value=0
    )

    # 创建 attention mask（1=有效，0=padding）
    mask = torch.zeros_like(padded_tokens, dtype=torch.bool)
    for i, length in enumerate(lengths):
        mask[i, :length] = True

    return {
        "tokens": padded_tokens,
        "labels": labels,
        "lengths": lengths,
        "mask": mask,
    }


# ============================================================
# 3. DataLoader 优化参数演示
# ============================================================
def demo_dataloader_optimization():
    """演示 DataLoader 的各种优化参数"""
    print("=" * 60)
    print("DataLoader 优化参数演示")
    print("=" * 60)

    dataset = CustomTextDataset(num_samples=100)

    # --- 基本用法 ---
    loader_basic = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
    )

    # --- 优化用法 ---
    # num_workers: 多进程加载数据（Windows 上建议设为 0 避免问题）
    # pin_memory:  将数据锁定在内存中，加速 GPU 传输
    # prefetch_factor: 预取的 batch 数量
    # persistent_workers: 保持 worker 进程存活，避免重复创建
    loader_optimized = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,          # Windows 设 0；Linux 可设 2-8
        pin_memory=True,        # GPU 训练时设 True
        drop_last=True,         # 丢弃最后不完整的 batch
        collate_fn=pad_collate_fn,
    )

    batch = next(iter(loader_optimized))
    print(f"Tokens shape:  {batch['tokens'].shape}")   # [8, max_seq_len_in_batch]
    print(f"Labels shape:  {batch['labels'].shape}")    # [8]
    print(f"Lengths shape: {batch['lengths'].shape}")   # [8]
    print(f"Mask shape:    {batch['mask'].shape}")      # [8, max_seq_len_in_batch]
    print(f"Mask sum per sample: {batch['mask'].sum(dim=1).tolist()}")
    print(f"Actual lengths:       {batch['lengths'].tolist()}")
    print()


# ============================================================
# 4. WeightedRandomSampler: 处理类别不平衡
# ============================================================
class ImbalancedDataset(Dataset):
    """模拟不平衡数据集"""

    def __init__(self, num_samples=1000):
        super().__init__()
        # 类别 0: 700, 类别 1: 200, 类别 2: 100
        self.labels = (
            [0] * 700 + [1] * 200 + [2] * 100
        )
        random.shuffle(self.labels)
        self.features = torch.randn(num_samples, 10)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def demo_weighted_sampler():
    """演示 WeightedRandomSampler 处理类别不平衡"""
    print("=" * 60)
    print("WeightedRandomSampler 演示")
    print("=" * 60)

    dataset = ImbalancedDataset(num_samples=1000)

    # 统计原始分布
    label_counter = Counter(dataset.labels)
    print(f"原始分布: {dict(label_counter)}")

    # 计算每个样本的权重
    # 权重 = 1 / (类别样本数)
    class_counts = np.array([label_counter[i] for i in range(3)])
    class_weights = 1.0 / class_counts  # [1/700, 1/200, 1/100]
    print(f"类别权重: {class_weights}")

    # 为每个样本分配权重
    sample_weights = [class_weights[label] for label in dataset.labels]
    sample_weights = torch.tensor(sample_weights, dtype=torch.float)

    # 创建 WeightedRandomSampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),   # 每个 epoch 采样的总数
        replacement=True,           # 允许重复采样
    )

    # 使用 sampler 的 DataLoader（注意：不能同时设置 shuffle=True）
    loader = DataLoader(
        dataset,
        batch_size=64,
        sampler=sampler,       # 使用 sampler 替代 shuffle
    )

    # 验证采样后的分布
    sampled_labels = []
    for _, labels in loader:
        sampled_labels.extend(labels.tolist())

    sampled_counter = Counter(sampled_labels)
    print(f"采样后分布: {dict(sampled_counter)}")
    total = sum(sampled_counter.values())
    for k in sorted(sampled_counter):
        print(f"  类别 {k}: {sampled_counter[k]/total:.2%}")
    print()


# ============================================================
# 5. Subset 和随机划分
# ============================================================
def demo_train_val_split():
    """演示手动划分训练集和验证集"""
    print("=" * 60)
    print("训练集/验证集划分")
    print("=" * 60)

    dataset = CustomTextDataset(num_samples=100)
    val_ratio = 0.2
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size

    # 随机划分
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    print(f"总样本数: {len(dataset)}")
    print(f"训练集:   {len(train_dataset)}")
    print(f"验证集:   {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=pad_collate_fn,
    )

    batch = next(iter(train_loader))
    print(f"训练 batch tokens shape: {batch['tokens'].shape}")
    print()


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    demo_dataloader_optimization()
    demo_weighted_sampler()
    demo_train_val_split()

    print("所有演示完成！")
