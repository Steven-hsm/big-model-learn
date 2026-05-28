"""
W10-D2 池化层与数据流 (Pooling Layers & Data Flow)
===================================================
从零实现池化层，构建SimpleCNN并追踪形状变化
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 从零实现池化层
# ============================================================
def max_pool2d(input_matrix, pool_size=2, stride=None):
    """
    手动实现最大池化

    参数:
        input_matrix: (H, W) 输入矩阵
        pool_size: 池化窗口大小
        stride: 步幅（默认等于pool_size）
    """
    if stride is None:
        stride = pool_size

    h_in, w_in = input_matrix.shape
    h_out = (h_in - pool_size) // stride + 1
    w_out = (w_in - pool_size) // stride + 1

    output = np.zeros((h_out, w_out))
    for i in range(h_out):
        for j in range(w_out):
            h_s = i * stride
            w_s = j * stride
            window = input_matrix[h_s:h_s + pool_size, w_s:w_s + pool_size]
            output[i, j] = np.max(window)

    return output


def avg_pool2d(input_matrix, pool_size=2, stride=None):
    """
    手动实现平均池化

    参数:
        input_matrix: (H, W) 输入矩阵
        pool_size: 池化窗口大小
        stride: 步幅（默认等于pool_size）
    """
    if stride is None:
        stride = pool_size

    h_in, w_in = input_matrix.shape
    h_out = (h_in - pool_size) // stride + 1
    w_out = (w_in - pool_size) // stride + 1

    output = np.zeros((h_out, w_out))
    for i in range(h_out):
        for j in range(w_out):
            h_s = i * stride
            w_s = j * stride
            window = input_matrix[h_s:h_s + pool_size, w_s:w_s + pool_size]
            output[i, j] = np.mean(window)

    return output


def global_avg_pool2d(input_matrix):
    """
    全局平均池化 (Global Average Pooling, GAP)
    将整个空间维度压缩为1x1

    参数:
        input_matrix: (H, W) 或 (C, H, W)
    """
    if input_matrix.ndim == 2:
        return np.mean(input_matrix)
    elif input_matrix.ndim == 3:
        C, H, W = input_matrix.shape
        output = np.zeros(C)
        for c in range(C):
            output[c] = np.mean(input_matrix[c])
        return output


# 验证池化实现
print("=" * 60)
print("池化层实现验证")
print("=" * 60)

test_matrix = np.array([
    [1, 3, 2, 4, 5, 6],
    [6, 2, 1, 3, 7, 8],
    [5, 4, 3, 2, 1, 0],
    [8, 7, 6, 5, 4, 3],
    [2, 1, 0, 9, 8, 7],
    [3, 4, 5, 6, 2, 1]
], dtype=np.float32)

print(f"\n输入矩阵 ({test_matrix.shape}):\n{test_matrix}")

max_out = max_pool2d(test_matrix, pool_size=2, stride=2)
print(f"\nMaxPool2d(2x2) 输出 ({max_out.shape}):\n{max_out}")

avg_out = avg_pool2d(test_matrix, pool_size=2, stride=2)
print(f"\nAvgPool2d(2x2) 输出 ({avg_out.shape}):\n{avg_out}")

gap_out = global_avg_pool2d(test_matrix)
print(f"\nGlobal Average Pooling 输出: {gap_out:.4f}")

# 多通道 GAP
multi_channel = np.random.randn(3, 4, 4)
gap_multi = global_avg_pool2d(multi_channel)
print(f"\n多通道 GAP: 输入 {multi_channel.shape} -> 输出 {gap_multi.shape}")
print(f"  通道0 GAP: {gap_multi[0]:.4f}")
print(f"  通道1 GAP: {gap_multi[1]:.4f}")
print(f"  通道2 GAP: {gap_multi[2]:.4f}")

# 可视化池化效果
np.random.seed(42)
feature_map = np.random.randn(8, 8) * 2 + 5
max_pooled = max_pool2d(feature_map, pool_size=2, stride=2)
avg_pooled = avg_pool2d(feature_map, pool_size=2, stride=2)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
im0 = axes[0].imshow(feature_map, cmap='viridis')
axes[0].set_title(f'原始特征图 ({feature_map.shape})')
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(max_pooled, cmap='viridis')
axes[1].set_title(f'最大池化 ({max_pooled.shape})')
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(avg_pooled, cmap='viridis')
axes[2].set_title(f'平均池化 ({avg_pooled.shape})')
plt.colorbar(im2, ax=axes[2])

plt.suptitle('池化操作对比', fontsize=14)
plt.tight_layout()
plt.savefig('d2_pooling_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图像已保存: d2_pooling_comparison.png")


# ============================================================
# 2. 使用 PyTorch 构建 SimpleCNN
# ============================================================
print("\n" + "=" * 60)
print("SimpleCNN 形状追踪")
print("=" * 60)


class SimpleCNN(nn.Module):
    """
    一个简单的CNN网络，用于CIFAR-10 (32x32 RGB图像)
    """

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # 卷积块1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 卷积块2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 卷积块3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接层
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        return self._forward_with_trace(x, trace=False)

    def _forward_with_trace(self, x, trace=False):
        if trace:
            print(f"\n{'层':<30} {'输出形状':<20} {'参数量':>10}")
            print("-" * 65)
            self._print_trace("输入", x, 0)

        x = self.conv1(x)
        if trace:
            self._print_trace("Conv2d(3->32, 3x3, pad=1)", x,
                              self.conv1.weight.numel() + self.conv1.bias.numel())

        x = self.relu1(x)
        if trace:
            self._print_trace("ReLU", x, 0)

        x = self.pool1(x)
        if trace:
            self._print_trace("MaxPool2d(2x2)", x, 0)

        x = self.conv2(x)
        if trace:
            self._print_trace("Conv2d(32->64, 3x3, pad=1)", x,
                              self.conv2.weight.numel() + self.conv2.bias.numel())

        x = self.relu2(x)
        if trace:
            self._print_trace("ReLU", x, 0)

        x = self.pool2(x)
        if trace:
            self._print_trace("MaxPool2d(2x2)", x, 0)

        x = self.conv3(x)
        if trace:
            self._print_trace("Conv2d(64->128, 3x3, pad=1)", x,
                              self.conv3.weight.numel() + self.conv3.bias.numel())

        x = self.relu3(x)
        if trace:
            self._print_trace("ReLU", x, 0)

        x = self.pool3(x)
        if trace:
            self._print_trace("MaxPool2d(2x2)", x, 0)

        x = self.flatten(x)
        if trace:
            self._print_trace("Flatten", x, 0)

        x = self.fc1(x)
        if trace:
            self._print_trace("Linear(128*4*4 -> 256)", x,
                              self.fc1.weight.numel() + self.fc1.bias.numel())

        x = self.relu4(x)
        if trace:
            self._print_trace("ReLU", x, 0)

        x = self.dropout(x)
        if trace:
            self._print_trace("Dropout(0.5)", x, 0)

        x = self.fc2(x)
        if trace:
            self._print_trace("Linear(256 -> 10)", x,
                              self.fc2.weight.numel() + self.fc2.bias.numel())

        return x

    def _print_trace(self, layer_name, tensor, params):
        shape_str = str(list(tensor.shape))
        print(f"  {layer_name:<30} {shape_str:<20} {params:>10,}")


# 创建模型并追踪形状
model = SimpleCNN(num_classes=10)
dummy_input = torch.randn(1, 3, 32, 32)

print("\n追踪 SimpleCNN 形状流:")
print(f"输入: 一张 32x32 RGB 图像 (batch=1)")
output = model._forward_with_trace(dummy_input, trace=True)

print(f"\n最终输出: {output.shape}")

# ============================================================
# 3. 统计模型参数
# ============================================================
print("\n" + "=" * 60)
print("模型参数统计")
print("=" * 60)

total_params = 0
trainable_params = 0
print(f"\n{'层':<35} {'参数量':>10}")
print("-" * 50)
for name, param in model.named_parameters():
    num_params = param.numel()
    total_params += num_params
    if param.requires_grad:
        trainable_params += num_params
    print(f"  {name:<35} {num_params:>10,}")

print("-" * 50)
print(f"  {'总参数量':<35} {total_params:>10,}")
print(f"  {'可训练参数量':<35} {trainable_params:>10,}")


# ============================================================
# 4. 不同输入尺寸的形状流
# ============================================================
print("\n" + "=" * 60)
print("不同输入尺寸的形状变化")
print("=" * 60)

# 模拟不同输入尺寸
input_sizes = [
    (1, 3, 32, 32, "CIFAR-10"),
    (1, 1, 28, 28, "MNIST"),
    (1, 3, 64, 64, "ImageNet (small)"),
]

# 为MNIST修改模型
class SimpleCNNFlexible(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = None  # 延迟初始化
        self.fc2 = None
        self.num_classes = num_classes
        self._initialized = False

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))

        if not self._initialized:
            flat_size = x.shape[1] * x.shape[2] * x.shape[3]
            self.fc1 = nn.Linear(flat_size, 256)
            self.fc2 = nn.Linear(256, self.num_classes)
            self._initialized = True

        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


print(f"\n{'数据集':<20} {'输入':<18} {'Conv1后':<18} {'Pool1后':<18} {'Conv2后':<18} {'Pool2后':<18} {'Flatten':<14}")
print("-" * 120)

for batch, channels, h, w, name in input_sizes:
    dummy = torch.randn(batch, channels, h, w)
    # 手动追踪
    c1 = torch.relu(nn.Conv2d(channels, 32, 3, padding=1)(dummy))
    p1 = nn.MaxPool2d(2, 2)(c1)
    c2 = torch.relu(nn.Conv2d(32, 64, 3, padding=1)(p1))
    p2 = nn.MaxPool2d(2, 2)(c2)
    c3 = torch.relu(nn.Conv2d(64, 128, 3, padding=1)(p2))
    p3 = nn.MaxPool2d(2, 2)(c3)
    flat = p3.view(p3.size(0), -1)

    print(f"{name:<20} {str(list(dummy.shape)):<18} {str(list(c1.shape)):<18} "
          f"{str(list(p1.shape)):<18} {str(list(c2.shape)):<18} "
          f"{str(list(p2.shape)):<18} {str(list(flat.shape)):<14}")


# ============================================================
# 5. GAP (Global Average Pooling) vs 展平对比
# ============================================================
print("\n" + "=" * 60)
print("GAP vs Flatten 对比")
print("=" * 60)

print("\n传统方法: Flatten + FC")
print(f"  Conv输出: (batch, 128, 4, 4) -> Flatten: (batch, 2048) -> FC: (batch, 256)")
print(f"  FC参数量: 2048 * 256 + 256 = {2048 * 256 + 256:,}")

print("\nGAP方法: Global Average Pooling + FC")
print(f"  Conv输出: (batch, 128, 4, 4) -> GAP: (batch, 128) -> FC: (batch, 256)")
print(f"  FC参数量: 128 * 256 + 256 = {128 * 256 + 256:,}")

print("\nGAP优势:")
print("  1. 大幅减少参数量，降低过拟合风险")
print("  2. 不依赖固定的空间尺寸")
print("  3. 强制特征图与类别对应，更具可解释性")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("D2 池化层与数据流 完成！")
    print("=" * 60)
