"""
W10-D4 ResNet实现 (Residual Networks)
======================================
实现 BasicBlock 和 ResNet18，对比有无残差连接的效果
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import time

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. BasicBlock 实现
# ============================================================
class BasicBlock(nn.Module):
    """
    ResNet基本块

    结构:
      x -> Conv -> BN -> ReLU -> Conv -> BN -> [+] -> ReLU -> out
      |                                        ^
      |________________________________________|
                     shortcut (残差连接)
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample  # 当维度不匹配时使用

    def forward(self, x):
        identity = x

        # 主路径
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 残差连接 (shortcut)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # 核心: F(x) + x
        out = self.relu(out)

        return out


class BasicBlockNoShortCut(nn.Module):
    """没有残差连接的BasicBlock，用于对比"""

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlockNoShortCut, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 没有残差连接！
        out = self.relu(out)

        return out


# ============================================================
# 2. ResNet18 实现
# ============================================================
class ResNet18(nn.Module):
    """
    ResNet-18

    结构:
      conv1 -> layer1 (2 blocks) -> layer2 (2 blocks)
      -> layer3 (2 blocks) -> layer4 (2 blocks) -> avgpool -> fc

    总深度: 1 + 2*2*4 = 17 层卷积 + 1 FC = 18 层
    """

    def __init__(self, num_classes=10, block_class=BasicBlock):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.block_class = block_class

        # 初始卷积层 (适配小尺寸输入如CIFAR-10)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # 不使用原版的大kernel和maxpool，因为CIFAR-10只有32x32

        # 残差层
        self.layer1 = self._make_layer(64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(512, num_blocks=2, stride=2)

        # 全局平均池化 + 全连接
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(self.block_class(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(self.block_class(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# ============================================================
# 3. 形状流追踪
# ============================================================
print("=" * 60)
print("ResNet-18 形状流追踪")
print("=" * 60)

resnet = ResNet18(num_classes=10)
dummy = torch.randn(1, 3, 32, 32)

print(f"\n输入: {list(dummy.shape)}")

x = resnet.conv1(dummy)
print(f"  conv1: {list(x.shape)}")
x = resnet.bn1(x)
x = resnet.relu(x)
print(f"  bn1 + relu: {list(x.shape)}")

for i, layer in enumerate([resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4]):
    x = layer(x)
    print(f"  layer{i + 1}: {list(x.shape)}")

x = resnet.avgpool(x)
print(f"  avgpool: {list(x.shape)}")
x = torch.flatten(x, 1)
print(f"  flatten: {list(x.shape)}")
x = resnet.fc(x)
print(f"  fc: {list(x.shape)}")

# 参数统计
total_params = sum(p.numel() for p in resnet.parameters())
print(f"\nResNet-18 总参数量: {total_params:,}")


# ============================================================
# 4. 对比: 有残差 vs 无残差
# ============================================================
print("\n" + "=" * 60)
print("对比: 有残差连接 vs 无残差连接")
print("=" * 60)

# 创建两个网络
resnet_with_shortcut = ResNet18(num_classes=10, block_class=BasicBlock)
resnet_no_shortcut = ResNet18(num_classes=10, block_class=BasicBlockNoShortCut)

params_with = sum(p.numel() for p in resnet_with_shortcut.parameters())
params_without = sum(p.numel() for p in resnet_no_shortcut.parameters())
print(f"\n有残差连接 ResNet-18 参数量: {params_with:,}")
print(f"无残差连接 ResNet-18 参数量: {params_without:,}")

# ============================================================
# 5. 训练对比实验
# ============================================================
print("\n" + "=" * 60)
print("训练对比实验 (简单合成数据)")
print("=" * 60)

# 生成简单的合成数据
np.random.seed(42)
torch.manual_seed(42)

num_samples = 500
# 创建 8x8 的简单图像，2个类别
X_data = torch.randn(num_samples, 3, 8, 8)
# 类别0: 左上角有较高值；类别1: 右下角有较高值
for i in range(num_samples):
    label = i % 2
    if label == 0:
        X_data[i, :, :4, :4] += 1.0
    else:
        X_data[i, :, 4:, 4:] += 1.0
y_data = torch.tensor([i % 2 for i in range(num_samples)], dtype=torch.long)


class SmallResNet(nn.Module):
    """小型ResNet用于快速实验"""
    def __init__(self, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(32, 64, 1, stride=2, bias=False)  # shortcut projection
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(64)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))

        # Block 1 (残差连接)
        identity = x
        out = self.relu(self.bn2(self.conv2(x)))
        if self.use_residual:
            out = out + identity  # 残差连接
        else:
            pass  # 不加残差

        # Block 2 (带下采样的残差连接)
        identity2 = self.bn4(self.conv4(out))
        out2 = self.relu(self.bn3(self.conv3(out)))
        if self.use_residual:
            out2 = out2 + identity2
        else:
            pass

        # Block 3
        identity3 = out2
        out3 = self.relu(self.bn5(self.conv5(out2)))
        if self.use_residual:
            out3 = out3 + identity3
        else:
            pass

        out3 = self.avgpool(out3)
        out3 = torch.flatten(out3, 1)
        out3 = self.fc(out3)
        return out3


def train_model(model, X, y, epochs=50, lr=0.01):
    """训练模型并记录loss和梯度"""
    optimizer = torch.optim.Adam(model, lr=lr)
    criterion = nn.CrossEntropyLoss()
    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()

        # 记录梯度范数
        total_grad_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.data.norm(2).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5

        optimizer.step()
        losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                pred = model(X).argmax(dim=1)
                acc = (pred == y).float().mean().item()
            print(f"  Epoch {epoch + 1:3d}/{epochs}: "
                  f"Loss={loss.item():.4f}, Acc={acc:.4f}, "
                  f"GradNorm={total_grad_norm:.4f}")

    return losses


# 重新创建模型
model_residual = SmallResNet(use_residual=True)
model_no_residual = SmallResNet(use_residual=False)

print("\n--- 有残差连接 ---")
losses_residual = train_model(model_residual, X_data, y_data, epochs=50)

print("\n--- 无残差连接 ---")
losses_no_residual = train_model(model_no_residual, X_data, y_data, epochs=50)


# ============================================================
# 6. 梯度流可视化
# ============================================================
def measure_gradient_flow(model, X, y):
    """测量每层梯度范数"""
    criterion = nn.CrossEntropyLoss()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()

    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.data.norm(2).item()

    model.zero_grad()
    return grad_norms


grad_with = measure_gradient_flow(SmallResNet(use_residual=True), X_data, y_data)
grad_without = measure_gradient_flow(SmallResNet(use_residual=False), X_data, y_data)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 训练曲线
axes[0].plot(losses_residual, label='有残差连接', color='#2ecc71', linewidth=2)
axes[0].plot(losses_no_residual, label='无残差连接', color='#e74c3c', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('训练Loss对比')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 梯度流
layers = list(grad_with.keys())
g_with = [grad_with[l] for l in layers]
g_without = [grad_without[l] for l in layers]

x_pos = np.arange(len(layers))
width = 0.35
axes[1].bar(x_pos - width / 2, g_with, width, label='有残差连接', color='#2ecc71', alpha=0.8)
axes[1].bar(x_pos + width / 2, g_without, width, label='无残差连接', color='#e74c3c', alpha=0.8)
axes[1].set_xlabel('层')
axes[1].set_ylabel('梯度范数')
axes[1].set_title('各层梯度范数对比')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels([l[:10] for l in layers], rotation=45, fontsize=8)
axes[1].legend()
axes[1].grid(alpha=0.3)

# ResNet架构示意图
axes[2].set_xlim(0, 10)
axes[2].set_ylim(0, 10)
axes[2].axis('off')
axes[2].set_title('ResNet残差连接示意图', fontsize=14)

# 绘制主路径
axes[2].annotate('', xy=(5, 9), xytext=(5, 1),
                 arrowprops=dict(arrowstyle='->', color='black', lw=2))
axes[2].text(5.3, 5, '主路径\nF(x)', fontsize=11, ha='left', va='center')

# 绘制残差连接
axes[2].annotate('', xy=(7.5, 9), xytext=(7.5, 1),
                 arrowprops=dict(arrowstyle='->', color='red', lw=2,
                                 connectionstyle='arc3,rad=0.3'))
axes[2].text(8.5, 5, '残差连接\nx', fontsize=11, ha='left', va='center', color='red')

# 标注加法
axes[2].plot(6.25, 9, 'o', markersize=15, color='orange', zorder=5)
axes[2].text(6.25, 9, '+', fontsize=14, ha='center', va='center', fontweight='bold')
axes[2].text(6.25, 9.5, 'F(x) + x', fontsize=11, ha='center', va='bottom')

# 输入输出
axes[2].text(5, 0.5, '输入 x', fontsize=12, ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='lightblue'))
axes[2].text(6.25, 9.8, '输出', fontsize=12, ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='lightgreen'))

plt.tight_layout()
plt.savefig('d4_resnet_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("图像已保存: d4_resnet_comparison.png")

print("""
残差连接的核心思想:
  普通网络: H(x) = 学习到的映射
  残差网络: H(x) = F(x) + x, 网络只需学习 F(x) = H(x) - x (残差)

优势:
  1. 缓解梯度消失: 梯度可以通过shortcut直接传播
  2. 身份映射容易学习: 如果最优解是恒等映射，只需F(x)=0
  3. 允许训练更深的网络: ResNet-152, 甚至1000+层
""")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("D4 ResNet实现 完成！")
    print("=" * 60)
