"""
W10-D6 CIFAR-10 项目（���型构建与训练）
========================================
加载CIFAR-10，构建CustomCNN，训练10个epoch
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


# ============================================================
# 1. CIFAR-10 数据加载
# ============================================================
print("=" * 60)
print("CIFAR-10 数据加载")
print("=" * 60)

# 训练集数据增强 + 标准化
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),          # 随机裁剪 (先填充再裁剪)
    transforms.RandomHorizontalFlip(),              # 随机水平翻转
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 颜色抖动
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],   # CIFAR-10均值
                         std=[0.2023, 0.1994, 0.2010])     # CIFAR-10标准差
])

# 测试集只做标准化
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010])
])

# 下载数据集
data_root = os.path.join(os.path.dirname(__file__), 'data')

train_dataset = torchvision.datasets.CIFAR10(
    root=data_root, train=True, download=True, transform=transform_train
)
test_dataset = torchvision.datasets.CIFAR10(
    root=data_root, train=False, download=True, transform=transform_test
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)

# CIFAR-10 类别
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")
print(f"类别: {classes}")
print(f"图像尺寸: 3x32x32")

# 可视化部分训练图像
def imshow(img, title=''):
    img = img * torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1) \
          + torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    npimg = img.numpy().transpose(1, 2, 0)
    npimg = np.clip(npimg, 0, 1)
    return npimg


fig, axes = plt.subplots(2, 8, figsize=(16, 4))
dataiter = iter(train_loader)
images, labels = next(dataiter)
for i in range(16):
    ax = axes[i // 8, i % 8]
    ax.imshow(imshow(images[i]))
    ax.set_title(classes[labels[i]], fontsize=9)
    ax.axis('off')
plt.suptitle('CIFAR-10 训练样本 (增强后)', fontsize=14)
plt.tight_layout()
plt.savefig('d6_cifar10_samples.png', dpi=150, bbox_inches='tight')
plt.show()
print("图像已保存: d6_cifar10_samples.png")


# ============================================================
# 2. CustomCNN 模型定义
# ============================================================
print("\n" + "=" * 60)
print("CustomCNN 模型 (with BatchNorm & Dropout)")
print("=" * 60)


class CustomCNN(nn.Module):
    """
    自定义CNN模型
    特点: BatchNorm + Dropout + 全局平均池化
    """

    def __init__(self, num_classes=10):
        super(CustomCNN, self).__init__()

        # 卷积块1: Conv -> BN -> ReLU -> Pool
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )

        # 卷积块2
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )

        # 卷积块3
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )

        # 全局平均池化 + 分类器
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


# 实例化模型
model = CustomCNN(num_classes=10).to(device)

# 打印模型结构
print(f"\n{'层':<45} {'参数量':>10}")
print("-" * 60)
total_params = 0
for name, param in model.named_parameters():
    num_params = param.numel()
    total_params += num_params
    print(f"  {name:<45} {num_params:>10,}")
print("-" * 60)
print(f"  总参数量: {total_params:,}")

# 形状追踪
dummy_input = torch.randn(1, 3, 32, 32).to(device)
print(f"\n形状流追踪:")
print(f"  输入: {list(dummy_input.shape)}")
with torch.no_grad():
    x = dummy_input
    for name, module in model.named_children():
        x = module(x)
        print(f"  {name}: {list(x.shape)}")


# ============================================================
# 3. 训练函数
# ============================================================
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 100 == 0:
            print(f"    Batch {batch_idx + 1}/{len(train_loader)}, "
                  f"Loss: {loss.item():.4f}, "
                  f"Acc: {100. * correct / total:.2f}%")

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, test_loader, criterion, device):
    """评估模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


# ============================================================
# 4. 训练循环
# ============================================================
print("\n" + "=" * 60)
print("开始训练 CustomCNN")
print("=" * 60)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

num_epochs = 10
train_losses, train_accs = [], []
test_losses, test_accs = [], []
best_acc = 0.0

total_start = time.time()

for epoch in range(num_epochs):
    epoch_start = time.time()

    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    scheduler.step()

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    epoch_time = time.time() - epoch_start

    print(f"\n  Epoch {epoch + 1}/{num_epochs} ({epoch_time:.1f}s):")
    print(f"    Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"    Test  Loss: {test_loss:.4f}, Test  Acc: {test_acc:.2f}%")
    print(f"    LR: {optimizer.param_groups[0]['lr']:.6f}")

    # 保存最佳模型
    if test_acc > best_acc:
        best_acc = test_acc
        checkpoint_path = os.path.join(os.path.dirname(__file__), 'best_custom_cnn.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
        }, checkpoint_path)
        print(f"    *** 保存最佳模型 (Acc: {best_acc:.2f}%) ***")

total_time = time.time() - total_start
print(f"\n训练完成! 总时间: {total_time:.1f}s, 最佳测试准确率: {best_acc:.2f}%")


# ============================================================
# 5. 可视化训练曲线
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

epochs_range = range(1, num_epochs + 1)

# Loss曲线
ax1.plot(epochs_range, train_losses, 'b-o', label='训练Loss', markersize=4)
ax1.plot(epochs_range, test_losses, 'r-o', label='测试Loss', markersize=4)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('训练和测试Loss曲线')
ax1.legend()
ax1.grid(alpha=0.3)

# Accuracy曲线
ax2.plot(epochs_range, train_accs, 'b-o', label='训练准确率', markersize=4)
ax2.plot(epochs_range, test_accs, 'r-o', label='测试准确率', markersize=4)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('准确率 (%)')
ax2.set_title('训练和测试准确率曲线')
ax2.legend()
ax2.grid(alpha=0.3)

plt.suptitle(f'CustomCNN 训练曲线 (最佳: {best_acc:.2f}%)', fontsize=14)
plt.tight_layout()
plt.savefig('d6_training_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("图像已保存: d6_training_curves.png")


# ============================================================
# 6. 测试结果分析
# ============================================================
print("\n" + "=" * 60)
print("测试结果分析")
print("=" * 60)

# 加载最佳模型
checkpoint = torch.load(checkpoint_path, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"已加载最佳模型 (Epoch {checkpoint['epoch']}, Acc: {checkpoint['best_acc']:.2f}%)")

# 每个类别的准确率
class_correct = [0] * 10
class_total = [0] * 10

model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        correct = predicted.eq(labels)
        for i in range(labels.size(0)):
            label = labels[i].item()
            class_total[label] += 1
            class_correct[label] += correct[i].item()

print(f"\n各类别准确率:")
print(f"  {'类别':<10} {'正确':>6} {'总数':>6} {'准确率':>8}")
print("  " + "-" * 35)
for i in range(10):
    acc = 100. * class_correct[i] / class_total[i]
    print(f"  {classes[i]:<10} {class_correct[i]:>6} {class_total[i]:>6} {acc:>7.2f}%")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("D6 CIFAR-10项目（模型构建与训练）完成！")
    print("=" * 60)
