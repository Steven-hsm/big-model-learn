"""
W10-D3 经典架构 (Classic CNN Architectures)
============================================
实现 LeNet-5, AlexNet, VGG，对比参数量与创新点
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def count_parameters(model):
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_model_summary(model, model_name):
    """打印模型摘要"""
    total, trainable = count_parameters(model)
    print(f"\n{'=' * 60}")
    print(f"{model_name} 模型摘要")
    print(f"{'=' * 60}")
    print(f"\n{'层':<40} {'形状':<25} {'参数量':>10}")
    print("-" * 80)
    for name, param in model.named_parameters():
        print(f"  {name:<40} {str(list(param.shape)):<25} {param.numel():>10,}")
    print("-" * 80)
    print(f"  总参数量: {total:,}")
    print(f"  可训练参数量: {trainable:,}")
    return total


# ============================================================
# 1. LeNet-5 (1998)
# ============================================================
class LeNet5(nn.Module):
    """
    LeNet-5: Yann LeCun, 1998
    用于手写数字识别 (MNIST: 1x28x28)
    创新点: 首个成功的CNN架构，证明了卷积网络的有效性
    """

    def __init__(self, num_classes=10, in_channels=1):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(
            # C1: 卷积层
            nn.Conv2d(in_channels, 6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            # S2: 平均池化
            nn.AvgPool2d(kernel_size=2, stride=2),
            # C3: 卷积层
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Sigmoid(),
            # S4: 平均池化
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================
# 2. AlexNet (2012) - 简化版
# ============================================================
class AlexNetSimple(nn.Module):
    """
    AlexNet 简化版: Alex Krizhevsky, 2012
    原版用于 ImageNet (224x224)，这里简化适配小尺寸输入
    创新点:
      - 使用ReLU激活函数（取代Sigmoid）
      - 使用Dropout防止过拟合
      - 数据增强
      - GPU训练
    """

    def __init__(self, num_classes=10, in_channels=3):
        super(AlexNetSimple, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================
# 3. VGG-style 网络
# ============================================================
class VGGBlock(nn.Module):
    """VGG基本块: 多个3x3卷积 + ReLU + 最大池化"""

    def __init__(self, in_channels, out_channels, num_convs):
        super(VGGBlock, self).__init__()
        layers = []
        for i in range(num_convs):
            layers.append(nn.Conv2d(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size=3, padding=1
            ))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class VGGSimple(nn.Module):
    """
    VGG 简化版: Simonyan & Zisserman, 2014
    创新点:
      - 使用小卷积核(3x3)替代大卷积核
      - 网络更深更规则
      - 证明了"更深的网络 = 更好的性能"
    配置: 类似VGG11 (简化版)
    """

    def __init__(self, num_classes=10, in_channels=3):
        super(VGGSimple, self).__init__()
        self.features = nn.Sequential(
            VGGBlock(in_channels, 64, num_convs=1),     # Block 1
            VGGBlock(64, 128, num_convs=1),              # Block 2
            VGGBlock(128, 256, num_convs=2),             # Block 3
            VGGBlock(256, 512, num_convs=2),             # Block 4
            VGGBlock(512, 512, num_convs=2),             # Block 5
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 1 * 1, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================
# 4. 形状追踪
# ============================================================
def trace_shape(model_name, model, input_tensor):
    """追踪模型中每一层的形状变化"""
    print(f"\n{'=' * 60}")
    print(f"{model_name} 形状追踪")
    print(f"{'=' * 60}")
    print(f"  输入: {list(input_tensor.shape)}")

    x = input_tensor
    layer_idx = 0

    for module in model.features:
        x = module(x)
        if isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
            layer_idx += 1
            print(f"  Layer {layer_idx} ({module.__class__.__name__}): "
                  f"{list(x.shape)}")

    for module in model.classifier:
        if isinstance(module, nn.Flatten):
            pre_flat_size = x.shape
        x = module(x)
        if isinstance(module, (nn.Linear, nn.Flatten)):
            layer_idx += 1
            print(f"  Layer {layer_idx} ({module.__class__.__name__}): "
                  f"{list(x.shape)}")

    return x


# ============================================================
# 5. 实例化模型并对比
# ============================================================
print("=" * 60)
print("经典CNN架构对比")
print("=" * 60)

# LeNet-5 (MNIST: 1x28x28)
lenet = LeNet5(num_classes=10, in_channels=1)
lenet_input = torch.randn(1, 1, 28, 28)
lenet_params = print_model_summary(lenet, "LeNet-5")
trace_shape("LeNet-5", lenet, lenet_input)

# AlexNet (CIFAR-10: 3x32x32)
alexnet = AlexNetSimple(num_classes=10, in_channels=3)
alexnet_input = torch.randn(1, 3, 32, 32)
alexnet_params = print_model_summary(alexnet, "AlexNet (简化版)")
trace_shape("AlexNet (简化版)", alexnet, alexnet_input)

# VGG (CIFAR-10: 3x32x32)
vgg = VGGSimple(num_classes=10, in_channels=3)
vgg_input = torch.randn(1, 3, 32, 32)
vgg_params = print_model_summary(vgg, "VGG (简化版)")
trace_shape("VGG (简化版)", vgg, vgg_input)


# ============================================================
# 6. 综合对比表格
# ============================================================
print("\n" + "=" * 60)
print("经典CNN架构综合对比")
print("=" * 60)

print(f"""
┌──────────────┬────────────┬────────┬──────────────────────────────────────┐
│ 架构         │ 参数量     │ 深度   │ 主要创新                             │
├──────────────┼────────────┼────────┼──────────────────────────────────────┤
│ LeNet-5      │ {lenet_params:>10,} │ ~5层  │ 首个成功的CNN；卷积+池化结构         │
│ (1998)       │            │        │ 证明了梯度下降训练卷积网络的有效性    │
├──────────────┼────────────┼────────┼���─────────────────────────────────────┤
│ AlexNet      │ {alexnet_params:>10,} │ ~8层  │ ReLU激活函数；Dropout正则化          │
│ (2012)       │            │        │ GPU加速训练；数据增强                 │
├──────────────┼────────────┼────────┼──────────────────────────────────────┤
│ VGG          │ {vgg_params:>10,} │ ~11层 │ 3x3小卷积核堆叠替代大卷积核          │
│ (2014)       │            │        │ 证明更深的网络性能更好                │
└──────────────┴────────────┴────────┴──────────────────────────────────────┘

关键演进趋势:
  1. 网络越来越深: 5层 -> 8层 -> 11层 -> 152层(ResNet) -> 更深
  2. 卷积核越来越小: 5x5/11x11 -> 3x3
  3. 激活函数: Sigmoid -> ReLU
  4. 参数量: 先增后减 (VGG很大, 后来通过GAP等技巧减少)
""")


# ============================================================
# 7. 可视化参数量对比
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 参数量对比
names = ['LeNet-5\n(1998)', 'AlexNet\n(2012, 简化)', 'VGG\n(2014, 简化)']
params = [lenet_params, alexnet_params, vgg_params]
colors = ['#2ecc71', '#3498db', '#e74c3c']

bars = ax1.bar(names, params, color=colors, alpha=0.8, edgecolor='black')
for bar, p in zip(bars, params):
    ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + max(params) * 0.02,
             f'{p:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax1.set_ylabel('参数量', fontsize=12)
ax1.set_title('经典CNN参数量对比', fontsize=14)
ax1.grid(axis='y', alpha=0.3)

# 层数对比
depths = [5, 8, 11]
bars2 = ax2.bar(names, depths, color=colors, alpha=0.8, edgecolor='black')
for bar, d in zip(bars2, depths):
    ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.2,
             f'{d}层', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax2.set_ylabel('网络深度 (层数)', fontsize=12)
ax2.set_title('经典CNN网络深度对比', fontsize=14)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('d3_classic_architectures.png', dpi=150, bbox_inches='tight')
plt.show()
print("图像已保存: d3_classic_architectures.png")


# ============================================================
# 8. VGG中两个3x3卷积替代一个5x5卷积
# ============================================================
print("\n" + "=" * 60)
print("VGG核心思想: 两个3x3卷积 = 一个5x5卷积的感受野")
print("=" * 60)

print("""
感受野计算:
  1层3x3卷积: 感受野 = 3x3
  2层3x3卷积: 感受野 = 5x5
  3层3x3卷积: 感受野 = 7x7

参数对比 (输入通道=C, 输出通道=C):
  一个5x5卷积参数量: C * C * 5 * 5 = 25C²
  两个3x3卷积参数量: 2 * C * C * 3 * 3 = 18C²

优势:
  1. 参数更少: 18C² < 25C² (节省28%)
  2. 更多的非线性变换: 2个ReLU vs 1个ReLU
  3. 更容易学习复杂特征
""")

# 实验验证
C = 64
demo_input = torch.randn(1, C, 32, 32)

# 一个 5x5 卷积
conv5x5 = nn.Conv2d(C, C, kernel_size=5, padding=2)
out_5x5 = conv5x5(demo_input)

# 两个 3x3 卷积
conv3x3_1 = nn.Conv2d(C, C, kernel_size=3, padding=1)
conv3x3_2 = nn.Conv2d(C, C, kernel_size=3, padding=1)
out_3x3 = conv3x3_2(torch.relu(conv3x3_1(demo_input)))

print(f"一个5x5卷积:")
print(f"  输入: {list(demo_input.shape)} -> 输出: {list(out_5x5.shape)}")
print(f"  参数量: {sum(p.numel() for p in conv5x5.parameters()):,}")

print(f"\n两个3x3卷积:")
print(f"  输入: {list(demo_input.shape)} -> 输出: {list(out_3x3.shape)}")
print(f"  参数量: {sum(p.numel() for p in conv3x3_1.parameters()) + sum(p.numel() for p in conv3x3_2.parameters()):,}")
print(f"  输出尺寸相同，感受野相同，但参数更少且有更多非线性！")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("D3 经典架构 完成！")
    print("=" * 60)
