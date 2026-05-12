# 第10周 - 卷积神经网络CNN

> 本周学习计算机视觉的核心架构CNN。理解卷积操作的数学原理、经典网络架构的演进思路，以及迁移学习的实践方法。

---

## 一、本周目标

1. 深入理解卷积操作的数学原理，掌握输出尺寸计算公式
2. 了解CNN经典架构的演进脉络（LeNet → AlexNet → VGG → ResNet）
3. 理解ResNet残差连接的本质——解决网络退化问题
4. 掌握数据增强和迁移学习的实践方法
5. 完成CIFAR-10图像分类项目，对比3种方案的效果

---

## 二、时间安排

### 工作日（每晚2小时）

| 日期 | 主题 | 时间分配 |
|------|------|----------|
| Day 1 (周一) | 卷积操作 | 理论50min + NumPy实现70min |
| Day 2 (周二) | 池化层与数据流 | 理论40min + 代码练习80min |
| Day 3 (周三) | 经典架构 | 论文阅读60min + 代码分析60min |
| Day 4 (周四) | 深层网络(ResNet等) | 理论50min + PyTorch实现70min |
| Day 5 (周五) | 数据增强与迁移学习 | 理论40min + 代码实践80min |

### 周末（6-8小时）

| 日期 | 主题 | 时间分配 |
|------|------|----------|
| Day 6 (周六) | CIFAR-10项目(上) | 数据准备60min + 模型构建120min + 训练120min |
| Day 7 (周日) | CIFAR-10项目(下) | 训练调优120min + 结果分析60min + 总结60min |

---

## 三、详细学习内容

### Day 1: 卷积操作

#### 1. 一维卷积（信号处理背景）

一维卷积是信号处理的基础概念：
```
(f * g)[n] = Σ f[m] · g[n-m]
```

在深度学习中，实际使用的是**互相关(cross-correlation)**（不翻转核）：
```
output[n] = Σ input[m] · kernel[n-m]  (互相关，不翻转)
```

PyTorch中的 `nn.Conv1d` 实际计算的是互相关，但业界习惯称为"卷积"。

#### 2. 二维卷积（图像处理）

二维卷积的数学定义：
```
Output(i, j) = Σ_m Σ_n Input(i+m, j+n) · Kernel(m, n) + bias
```

直观理解：卷积核（滤波器）在输入图像上滑动，每个位置计算逐元素乘积之和。

**卷积核的直觉**：
- 水平边缘检测核：`[[-1,-1,-1],[0,0,0],[1,1,1]]`
- 垂直边缘检测核：`[[-1,0,1],[-1,0,1],[-1,0,1]]`
- 锐化核：`[[0,-1,0],[-1,5,-1],[0,-1,0]]`
- 高斯模糊核：每个位置的值是高斯分布的采样

#### 3. 关键超参数

**卷积核大小 (kernel_size)**：通常使用3x3或5x5，奇数尺寸保证有中心点。

**步长 (stride)**：卷积核每次移动的像素数。stride=2相当于下采样。

**填充 (padding)**：
- `valid`：不填充，输出尺寸减小
- `same`：填充使输出尺寸与输入相同（stride=1时）

#### 4. 输出尺寸公式

```
output_size = (input_size - kernel_size + 2 * padding) / stride + 1
```

例题：输入32x32，kernel 5x5，stride=1，padding=0：
```
output = (32 - 5 + 0) / 1 + 1 = 28
```

输入28x28，kernel 3x3，stride=1，padding=1：
```
output = (28 - 3 + 2) / 1 + 1 = 28  （same padding）
```

#### 5. 多通道卷积

对于RGB图像（3通道），每个卷积核也是一个3D张量（3 x kernel_h x kernel_w）：
```
output(i,j) = Σ_c Σ_m Σ_n Input(c, i+m, j+n) · Kernel(c, m, n) + bias
```

一个卷积核产生一个输出通道。要产生N个输出通道，就需要N个这样的卷积核。

参数量计算：
```
params = (C_in × K × K + 1) × C_out
```

例如：3通道输入，64个3x3卷积核：
```
params = (3 × 3 × 3 + 1) × 64 = 1792
```

#### 6. NumPy实现2D卷积

```python
import numpy as np

def conv2d(input_matrix, kernel, stride=1, padding=0):
    """
    input_matrix: shape (H, W)
    kernel: shape (kH, kW)
    """
    # Padding
    if padding > 0:
        input_matrix = np.pad(input_matrix, padding, mode='constant')

    H, W = input_matrix.shape
    kH, kW = kernel.shape
    out_H = (H - kH) // stride + 1
    out_W = (W - kW) // stride + 1

    output = np.zeros((out_H, out_W))
    for i in range(out_H):
        for j in range(out_W):
            region = input_matrix[i*stride:i*stride+kH, j*stride:j*stride+kW]
            output[i, j] = np.sum(region * kernel)

    return output

# 验证
input_5x5 = np.array([
    [1, 2, 0, 1, 3],
    [0, 1, 2, 3, 1],
    [1, 2, 3, 0, 2],
    [2, 3, 1, 0, 1],
    [0, 1, 2, 3, 0]
])
kernel_3x3 = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])
result = conv2d(input_5x5, kernel_3x3, stride=1, padding=0)
print(f"输出 shape: {result.shape}")
print(result)
```

---

### Day 2: 池化层

#### 1. 最大池化 MaxPool2d

取每个区域的最大值：
```
对2x2区域 [[1, 3], [2, 4]] → max = 4
```

作用：保留最显著的特征，减少计算量，提供一定的平移不变性。

```python
import torch.nn as nn
pool = nn.MaxPool2d(kernel_size=2, stride=2)
# 输入 (N, C, H, W) → 输出 (N, C, H/2, W/2)
```

#### 2. 平均池化 AvgPool2d

取每个区域的平均值：
```
对2x2区域 [[1, 3], [2, 4]] → mean = 2.5
```

比最大池化更平滑，常用于最后一层。

#### 3. 全局平均池化 GAP (Global Average Pooling)

对整个特征图取平均，将 `(N, C, H, W)` 变为 `(N, C)`。

```python
gap = nn.AdaptiveAvgPool2d(1)  # 输出size=1x1
```

**为什么用GAP替代全连接层**：
- 大幅减少参数量
- 强制特征图与类别对应
- 防止过拟合
- Network In Network (NiN) 论文首次提出

#### 4. Flatten操作

将多维特征展平为一维向量：
```python
# (N, C, H, W) → (N, C*H*W)
x = x.view(x.size(0), -1)  # 或 torch.flatten(x, 1)
```

#### 5. 完整数据流

一个典型CNN的数据流：
```
输入: (N, 3, 32, 32)
  → Conv1(3→32, 3x3): (N, 32, 32, 32)   # same padding
  → ReLU: (N, 32, 32, 32)
  → MaxPool(2x2): (N, 32, 16, 16)
  → Conv2(32→64, 3x3): (N, 64, 16, 16)
  → ReLU: (N, 64, 16, 16)
  → MaxPool(2x2): (N, 64, 8, 8)
  → Flatten: (N, 64*8*8) = (N, 4096)
  → FC(4096→128): (N, 128)
  → ReLU: (N, 128)
  → FC(128→10): (N, 10)
```

---

### Day 3: 经典架构

#### 1. LeNet-5 (1998, Yann LeCun)

```
Input(32x32) → Conv(6, 5x5) → AvgPool(2x2) → Conv(16, 5x5)
→ AvgPool(2x2) → FC(120) → FC(84) → FC(10)
```

历史意义：第一个成功的CNN架构，用于手写数字识别（MNIST）。

特点：
- 使用平均池化（当时还没有MaxPool）
- 使用Sigmoid/Tanh激活函数
- 参数量约60K
- 准确率：MNIST 99%+

```python
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),  # MNIST是单通道
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
```

#### 2. AlexNet (2012, Alex Krizhevsky)

突破性成果：ImageNet竞赛将错误率从26%降到16%。

关键创新：
- **ReLU激活函数**：替代Sigmoid，解决梯度消失
- **Dropout正则化**：防止过拟合
- **数据增强**：随机裁剪、水平翻转、颜色变换
- **GPU训练**：使用两块GTX 580并行训练
- **LRN（局部响应归一化）**：后来证明作用不大

架构：
```
Input(224x224x3) → Conv(96,11x11,s=4) → MaxPool(3x3,s=2)
→ Conv(256,5x5) → MaxPool(3x3,s=2)
→ Conv(384,3x3) → Conv(384,3x3) → Conv(256,3x3) → MaxPool(3x3,s=2)
→ FC(4096) → Dropout → FC(4096) → Dropout → FC(1000)
```

参数量：约6000万。

#### 3. VGGNet (2014, Oxford)

核心思想：**使用多个小卷积核(3x3)堆叠代替大卷积核**。

为什么2个3x3等效于1个5x5？
- 2个3x3的感受野 = 5x5
- 参数更少：2×(3×3×C) = 18C vs 1×(5×5×C) = 25C
- 更多非线性（2次ReLU vs 1次ReLU）

VGG16架构：
```
[Conv3-64] ×2 → Pool
[Conv3-128] ×2 → Pool
[Conv3-256] ×3 → Pool
[Conv3-512] ×3 → Pool
[Conv3-512] ×3 → Pool
FC-4096 → FC-4096 → FC-1000
```

参数量：约1.38亿（大部分在第一个FC层）。

#### 4. 架构演进思路

```
LeNet (1998): 开山之作，验证CNN可行性
    ↓
AlexNet (2012): ReLU + Dropout + 数据增强 + GPU
    ↓ （更深）
VGGNet (2014): 小卷积核堆叠，网络更深更规则
    ↓ （更高效）
GoogLeNet/Inception (2014): 多尺度并行卷积，1x1降维
    ↓ （更深不退化）
ResNet (2015): 残差连接，可以训练152层
    ↓
DenseNet (2017): 密集连接，特征复用
    ↓
EfficientNet (2019): 复合缩放，精度和效率平衡
```

演进核心思路：**更深（更多层）、更高效（更少参数和计算）、更好的梯度流动**。

---

### Day 4: 深层网络

#### 1. ResNet 残差连接 (2015, Kaiming He)

**问题**：网络加深后，训练误差反而上升（退化问题，不是梯度消失）。

**解决方案**：残差学习
```
常规: H(x) = 学习直接映射
残差: F(x) = H(x) - x → H(x) = F(x) + x
```

网络只需要学习残差 `F(x)`，比学习完整映射 `H(x)` 更容易。当残差为0时，网络层等效于恒等映射。

```python
class BasicBlock(nn.Module):
    """ResNet的基础残差块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # shortcut连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)  # shortcut分支
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # 残差连接：F(x) + x
        out = F.relu(out)
        return out
```

**为什么ResNet有效**：
1. 残差连接提供了梯度"高速公路"，梯度可以直接回传
2. 每层至少可以学到恒等映射（不比浅层差）
3. 残差比完整映射更容易优化

#### 2. DenseNet (2017)

每层与所有前面的层相连：
```
Layer l 的输入 = [x_0, x_1, ..., x_{l-1}]（通道维度拼接）
```

优点：特征复用，参数更少，梯度流动更好。
缺点：内存消耗大（特征图不断拼接）。

#### 3. Inception/GoogLeNet (2014)

核心思想：多尺度并行处理
```
输入 → [1x1 Conv] → Concat
    → [1x1 Conv → 3x3 Conv] → Concat
    → [1x1 Conv → 5x5 Conv] → Concat
    → [3x3 MaxPool → 1x1 Conv] → Concat
```

1x1卷积的作用：降维减少通道数，减少计算量。

---

### Day 5: 数据增强与迁移学习

#### 1. 数据增强 torchvision.transforms

```python
from torchvision import transforms

# 训练时增强
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),           # 随机裁剪并缩放
    transforms.RandomHorizontalFlip(p=0.5),      # 随机水平翻转
    transforms.ColorJitter(                       # 颜色抖动
        brightness=0.2, contrast=0.2,
        saturation=0.2, hue=0.1
    ),
    transforms.RandomRotation(degrees=15),        # 随机旋转
    transforms.ToTensor(),
    transforms.Normalize(                         # 标准化
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# 验证/测试时不增强
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
```

#### 2. Normalize的mean和std

ImageNet的标准化参数：`mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`

为什么要标准化：
- 加速收敛（各维度数值范围一致）
- 与预训练模型一致（迁移学习时必须使用相同的归一化参数）

#### 3. Mixup数据增强

```python
def mixup_data(x, y, alpha=0.2):
    """Mixup: 两张图片线性插值"""
    lam = np.random.beta(alpha, alpha)  # λ ~ Beta(α, α)
    batch_size = x.size(0)
    index = torch.randperm(batch_size)  # 随机打乱索引

    mixed_x = lam * x + (1 - lam) * x[index]  # 混合图片
    y_a, y_b = y, y[index]                      # 混合标签
    return mixed_x, y_a, y_b, lam

# Mixup损失函数
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
```

#### 4. CutMix

```python
def cutmix_data(x, y, beta=1.0):
    """CutMix: 从一张图裁剪区域粘贴到另一张图"""
    lam = np.random.beta(beta, beta)
    rand_index = torch.randperm(x.size(0))
    y_a, y_b = y, y[rand_index]

    # 生成随机裁剪区域
    W, H = x.size(2), x.size(3)
    cut_ratio = np.sqrt(1.0 - lam)
    rw, rh = int(W * cut_ratio), int(H * cut_ratio)
    cx, cy = np.random.randint(W), np.random.randint(H)

    x1, x2 = np.clip(cx - rw//2, 0, W), np.clip(cx + rw//2, 0, W)
    y1, y2 = np.clip(cy - rh//2, 0, H), np.clip(cy + rh//2, 0, H)

    x_clone = x.clone()
    x_clone[:, :, x1:x2, y1:y2] = x[rand_index, :, x1:x2, y1:y2]

    # 根据实际裁剪面积调整lambda
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    return x_clone, y_a, y_b, lam
```

#### 5. 迁移学习

迁移学习是利用在大数据集（如ImageNet）上预训练的模型，迁移到自己的小数据集上。

```python
import torchvision.models as models

# 加载预训练ResNet18
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# 方法1：冻结特征提取层，只训练分类头
for param in model.parameters():
    param.requires_grad = False  # 冻结所有参数

# 替换最后的全连接层
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)  # 10类
# 新的fc层默认 requires_grad=True

# 方法2：逐步解冻（Fine-tuning）
# 先训练分类头几个epoch
# 然后解冻最后几个block继续训练
for param in model.layer4.parameters():
    param.requires_grad = True
```

**迁移学习有效的条件**：
- 源域和目标域有一定相似性
- 源域数据量远大于目标域
- 预训练模型学到的底层特征（边缘、纹理等）是通用的

**不适合的情况**：
- 目标域与ImageNet差异极大（如医学影像、卫星图像）
- 目标域数据量充足（可以直接从头训练）

---

## 四、代码练习

### Day 1：手动计算卷积

```python
"""���务：手动计算5x5输入经过3x3卷积核的输出"""
import numpy as np

# 输入5x5
input_5x5 = np.array([
    [1, 0, 1, 2, 0],
    [0, 1, 0, 1, 1],
    [1, 0, 1, 0, 2],
    [0, 2, 0, 1, 0],
    [1, 0, 2, 1, 1]
])

# 卷积核3x3（垂直边缘检测）
kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

# 手动计算每个位置的输出
# 位置(0,0): 1*1 + 0*0 + 1*(-1) + 0*1 + 1*0 + 0*(-1) + 1*1 + 0*0 + 1*(-1) = 0
# 位置(0,1): ...
# 完成全部3x3=9个位置的计算

# 用conv2d函数验证
def conv2d(input_matrix, kernel, stride=1, padding=0):
    if padding > 0:
        input_matrix = np.pad(input_matrix, padding, mode='constant')
    H, W = input_matrix.shape
    kH, kW = kernel.shape
    out_H = (H - kH) // stride + 1
    out_W = (W - kW) // stride + 1
    output = np.zeros((out_H, out_W))
    for i in range(out_H):
        for j in range(out_W):
            region = input_matrix[i*stride:i*stride+kH, j*stride:j*stride+kW]
            output[i, j] = np.sum(region * kernel)
    return output

print("手动计算 vs 函数计算 对比验证")
```

### Day 2：CNN完整数据流

```python
"""任务：用PyTorch实现一个简单的CNN，追踪每层的shape"""
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),   # (N,3,32,32) → (N,32,32,32)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                # (N,32,16,16)
            nn.Conv2d(32, 64, 3, padding=1),   # (N,64,16,16)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                # (N,64,8,8)
            nn.Conv2d(64, 128, 3, padding=1),  # (N,128,8,8)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                # (N,128,4,4)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        print(f"After features: {x.shape}")
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 测试
model = SimpleCNN()
dummy = torch.randn(4, 3, 32, 32)
output = model(dummy)
print(f"Output shape: {output.shape}")  # (4, 10)

# 统计参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
```

### Day 4：ResNet BasicBlock实现

```python
"""任务：实现ResNet的BasicBlock和完整ResNet18"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 残差连接
        out = F.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
```

### Day 6-7：CIFAR-10项目

```python
"""项目4：CIFAR-10图像分类 - 3模型对比"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import time

# ==================== 数据准备 ====================
transform_base = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

transform_aug = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

train_dataset_base = datasets.CIFAR10(root='./data', train=True,
                                       download=True, transform=transform_base)
train_dataset_aug = datasets.CIFAR10(root='./data', train=True,
                                      download=True, transform=transform_aug)
test_dataset = datasets.CIFAR10(root='./data', train=False,
                                 download=True, transform=transform_base)

# ==================== 训练函数 ====================
def train_and_evaluate(model, train_loader, test_loader, name, epochs=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    print(f"\n===== {name} =====")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total

        # 验证
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_acc = 100. * correct / total
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: Train Acc={train_acc:.1f}%, Test Acc={test_acc:.1f}%")

    elapsed = time.time() - start_time
    print(f"训练时间: {elapsed:.1f}s, 最终测试准确率: {test_acc:.1f}%")
    return test_acc

# ==================== 模型1: 自定义CNN ====================
class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout(0.25),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout(0.25),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ==================== 模型2: ResNet18预训练微调 ====================
def get_pretrained_resnet18():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

# ==================== 运行对比 ====================
batch_size = 128
train_loader_base = DataLoader(train_dataset_base, batch_size=batch_size, shuffle=True, num_workers=2)
train_loader_aug = DataLoader(train_dataset_aug, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

results = {}
results['CustomCNN'] = train_and_evaluate(
    CustomCNN(), train_loader_base, test_loader, "自定义CNN", epochs=20)
results['ResNet18-预训练'] = train_and_evaluate(
    get_pretrained_resnet18(), train_loader_base, test_loader, "ResNet18预训练", epochs=20)
results['ResNet18-预训练+增强'] = train_and_evaluate(
    get_pretrained_resnet18(), train_loader_aug, test_loader, "ResNet18预训练+数据增强", epochs=20)

# 结果汇总
print("\n===== 结果汇总 =====")
for name, acc in results.items():
    print(f"{name}: {acc:.1f}%")
```

---

## 五、本周产出

### 周末交付物

1. **CIFAR-10对比项目**：3个模型的完整训练代码和对比结果，文件 `cifar10_comparison.py`
2. **ResNet实现**：自己实现的ResNet18代码 `resnet18.py`
3. **卷积计算验证**：手动卷积计算与NumPy/PyTorch对比 `conv_calculation.py`
4. **模型对比报告**：包含各模型准确率、训练时间、参数量的对比表格

---

## 六、自测题

### 题1：卷积的3个重要超参数如何影响输出尺寸？

<details>
<summary>参考答案</summary>

输出尺寸公式：`output = (input - kernel + 2*padding) / stride + 1`

- **kernel_size**：越大，输出尺寸越小。5x5卷积比3x3卷积使尺寸多减小2
- **stride**：越大，输出尺寸越小。stride=2使输出尺寸大约减半
- **padding**：越大，输出尺寸越大。padding=1配合3x3卷积和stride=1可以保持尺寸不变（same padding）

实际设计中，常用组合：
- 保持尺寸：kernel=3, stride=1, padding=1
- 下采样：kernel=3, stride=2, padding=1（尺寸减半）
</details>

### 题2：为什么用多个小卷积核(3x3)代替大卷积核(7x7)？

<details>
<summary>参考答案</summary>

3个3x3卷积核堆叠 vs 1个7x7卷积核：
- **感受野相同**：3个3x3的感受野 = 7x7
- **参数更少**：3 × (3×3×C×C) = 27C² vs 1 × (7×7×C×C) = 49C²，减少约45%
- **更多非线性**：3次ReLU vs 1次ReLU，表达能力更强
- **更容易优化**：每层更简单，梯度流动更好

这就是VGGNet的核心设计理念。
</details>

### 题3：ResNet的残差连接为什么有效？

<details>
<summary>参考答案</summary>

1. **解决退化问题**：网络加深后训练误差反而上升（不是梯度消失）。残差连接让网络至少可以学到恒等映射——将残差F(x)学为0，等效于跳过该层，确保不比浅层差。

2. **梯度高速公路**：反向传播时，梯度可以通过shortcut直接传回前层：`∂L/∂x = ∂L/∂y · (∂F/∂x + 1)`，即使F的梯度很小，+1也保证了梯度不会消失。

3. **学习残差更容易**：学习目标值与输入的差值（残差）比学习完整映射更容易。比如目标是恒等映射时，学习F(x)=0比学习H(x)=x更简单。
</details>

### 题4：迁移学习为什么有效？什么情况下不适合？

<details>
<summary>参考答案</summary>

有效的原理：
- 卷积网络的浅层学到的是通用特征（边缘、纹理、颜色等），这些特征在大多数视觉任务中都有用
- 深层学到的是任务特定的高级语义特征
- 在大数据集上预训练相当于给模型一个很好的初始化，比随机初始化好得多
- 小数据集上从头训练容易过拟合，迁移学习可以利用预训练知识

不适合的情况：
1. 目标域与源域差异很大（如ImageNet预训练用于医学影像、卫星图像）
2. 目标域有大量数据，足以从头训练
3. 目标域的输入格式不同（如深度图、点云数据）
</details>

### 题5：数据增强的本质作用是什么？

<details>
<summary>参考答案</summary>

数据增强的本质是**扩大训练数据的分布**，具体来说：

1. **增加数据多样性**：通过对现有数据做变换，生成"新"样本，扩大模型见过的数据分布范围
2. **防止过拟合**：每个epoch看到的图片都有随机变化，模型无法记住训练集
3. **引入不变性**：模型学到对某些变换不敏感（如旋转不变性、翻转不变性）
4. **模拟真实场景**：实际应用中图片可能有不同光照、角度、遮挡，增强让模型提前适应

本质上是在数据层面做正则化。Mixup和CutMix更进一步，直接在数据空间做插值，让决策边界更平滑。
</details>

---

## 七、Java开发者提示

### 1. 卷积 vs 过滤器模式

| CNN概念 | Java中的类比 |
|---------|-------------|
| 卷积核(Filter/Kernel) | Servlet Filter / Interceptor |
| 卷积操作(滑动窗口) | 滑动窗口处理数据流 |
| 步长(Stride) | for循环的步进值 `i += stride` |
| 填充(Padding) | Buffer的预分配空间 |
| 通道(Channel) | 数据的不同维度（如日志的不同字段） |

```java
// 卷积操作类似一个滑动窗口处理器
// 就像用一个固定大小的"检查窗口"遍历数据
for (int i = 0; i < data.length - windowSize; i += stride) {
    float sum = 0;
    for (int j = 0; j < windowSize; j++) {
        sum += data[i + j] * kernel[j];
    }
    output[i / stride] = sum;
}
```

### 2. 池化 vs 数据聚合

池化类似于Java Stream中的聚合操作：
```java
// 最大池化 = 取窗口内最大值
Optional<BigDecimal> max = data.stream().max();

// 平均池化 = 取窗口内平均值
double avg = data.stream().mapToInt(Integer::intValue).average().orElse(0);

// 全局平均池化 = Collectors.averagingDouble
```

### 3. ResNet残差连接 vs 装饰器模式

残差连接类似Java中的装饰器模式——在原有操作基础上添加额外功能：
```java
// 普通层：直接学习 H(x)
// 残差块：学习 F(x)，输出 F(x) + x

// 类似装饰器模式
interface Layer {
    Tensor forward(Tensor x);
}

class ResidualBlock implements Layer {
    private Layer inner;
    public Tensor forward(Tensor x) {
        return inner.forward(x).add(x);  // F(x) + x
    }
}
```

### 4. 迁移学习 vs 代码复用

迁移学习的本质和软件开发中的代码复用一样：
- **预训练模型** = 成熟的第三方库（如Spring Boot）
- **冻结层** = 直接使用库的功能，不修改源码
- **微调** = 通过配置和扩展点定制功能
- **全量微调** = Fork仓库并修改

就像你不会从零写一个Web框架，而是基于Spring Boot开发一样；CNN也不需要从零学特征，而是基于ImageNet预训练模型微调。

### 5. 数据增强 vs 单元测试的Mock数据

数据增强类似于为单元测试生成多样化的Mock数据：
- 目标一致：覆盖更多场景，提高鲁棒性
- 方法类似：在真实数据基础上做变换
- 注意点：变换不能改变数据的本质含义（标签不变）
