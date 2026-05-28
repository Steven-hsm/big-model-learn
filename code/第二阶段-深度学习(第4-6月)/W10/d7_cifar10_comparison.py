"""
W10-D7 CIFAR-10 项目（模型对比与评估）
========================================
对比 CustomCNN, 预训练ResNet18(冻结), 预训练ResNet18(微调)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from sklearn.metrics import confusion_matrix

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


# ============================================================
# 1. 数据加载
# ============================================================
print("=" * 60)
print("CIFAR-10 数据加载")
print("=" * 60)

data_root = os.path.join(os.path.dirname(__file__), 'data')

# 为预训练模型准备的变换 (需要Resize到224)
transform_train_pretrained = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomCrop(224, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_test_pretrained = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 为CustomCNN准备的变换 (32x32)
transform_train_custom = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010])
])

transform_test_custom = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010])
])

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# CustomCNN 数据
train_dataset_custom = torchvision.datasets.CIFAR10(
    root=data_root, train=True, download=True, transform=transform_train_custom)
test_dataset_custom = torchvision.datasets.CIFAR10(
    root=data_root, train=False, download=True, transform=transform_test_custom)

train_loader_custom = DataLoader(train_dataset_custom, batch_size=128, shuffle=True, num_workers=0)
test_loader_custom = DataLoader(test_dataset_custom, batch_size=100, shuffle=False, num_workers=0)

# 预训练模型数据 (使用小批次以节省内存)
train_dataset_pretrained = torchvision.datasets.CIFAR10(
    root=data_root, train=True, download=True, transform=transform_train_pretrained)
test_dataset_pretrained = torchvision.datasets.CIFAR10(
    root=data_root, train=False, download=True, transform=transform_test_pretrained)

train_loader_pretrained = DataLoader(train_dataset_pretrained, batch_size=64, shuffle=True, num_workers=0)
test_loader_pretrained = DataLoader(test_dataset_pretrained, batch_size=50, shuffle=False, num_workers=0)

print(f"训练集大小: {len(train_dataset_custom)}")
print(f"测试集大小: {len(test_dataset_custom)}")


# ============================================================
# 2. 模型定义
# ============================================================

# CustomCNN (与d6相同)
class CustomCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomCNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.MaxPool2d(2, 2), nn.Dropout(0.25),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2), nn.Dropout(0.25),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2), nn.Dropout(0.25),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


def create_frozen_resnet18(num_classes=10):
    """预训练ResNet18, 冻结骨干"""
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(512, num_classes)
    return model


def create_finetuned_resnet18(num_classes=10):
    """预训练ResNet18, 微调layer4 + fc"""
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    # 解冻 layer4
    for param in model.layer4.parameters():
        param.requires_grad = True
    model.fc = nn.Linear(512, num_classes)
    return model


# ============================================================
# 3. 通用训练和评估函数
# ============================================================
def train_and_evaluate(model, train_loader, test_loader, model_name,
                       num_epochs=5, lr=0.001):
    """训练并评估模型"""
    print(f"\n{'=' * 60}")
    print(f"训练 {model_name}")
    print(f"{'=' * 60}")

    model = model.to(device)

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss()

    # 根据模型类型设置优化器
    if trainable_params < total_params:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=lr)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    start_time = time.time()

    for epoch in range(num_epochs):
        # 训练
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
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

        train_loss = running_loss / total
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 评估
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

        test_loss = running_loss / total
        test_acc = 100. * correct / total
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        scheduler.step()

        print(f"  Epoch {epoch + 1}/{num_epochs}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
              f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}%")

    training_time = time.time() - start_time
    print(f"\n  训练完成! 时间: {training_time:.1f}s, 最终测试准确率: {test_acc:.2f}%")

    return {
        'model': model,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'training_time': training_time,
        'final_test_acc': test_acc,
    }


# ============================================================
# 4. 训练所有模型
# ============================================================
print("=" * 60)
print("模型对比实验")
print("=" * 60)

NUM_EPOCHS = 5  # 为节省时间使用5个epoch

# 模型1: CustomCNN (从头训练)
results_custom = train_and_evaluate(
    CustomCNN(num_classes=10),
    train_loader_custom, test_loader_custom,
    "CustomCNN (从头训练)",
    num_epochs=NUM_EPOCHS, lr=0.001
)

# 模型2: 预训练ResNet18 (冻结骨干)
results_frozen = train_and_evaluate(
    create_frozen_resnet18(num_classes=10),
    train_loader_pretrained, test_loader_pretrained,
    "ResNet18 预训练(冻结骨干)",
    num_epochs=NUM_EPOCHS, lr=0.001
)

# 模型3: 预训练ResNet18 (微调)
results_finetuned = train_and_evaluate(
    create_finetuned_resnet18(num_classes=10),
    train_loader_pretrained, test_loader_pretrained,
    "ResNet18 预训练(微调layer4+fc)",
    num_epochs=NUM_EPOCHS, lr=0.0001
)


# ============================================================
# 5. 训练曲线对比
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

all_results = [
    ('CustomCNN (从头训练)', results_custom, '#e74c3c'),
    ('ResNet18 (冻结)', results_frozen, '#3498db'),
    ('ResNet18 (微调)', results_finetuned, '#2ecc71'),
]

epochs_range = range(1, NUM_EPOCHS + 1)

for name, result, color in all_results:
    axes[0, 0].plot(epochs_range, result['train_losses'], '-o', color=color,
                    label=name, markersize=4)
    axes[0, 1].plot(epochs_range, result['test_losses'], '-o', color=color,
                    label=name, markersize=4)
    axes[1, 0].plot(epochs_range, result['train_accs'], '-o', color=color,
                    label=name, markersize=4)
    axes[1, 1].plot(epochs_range, result['test_accs'], '-o', color=color,
                    label=name, markersize=4)

axes[0, 0].set_title('训练Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

axes[0, 1].set_title('测试Loss')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

axes[1, 0].set_title('训练准确率')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('准确率 (%)')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

axes[1, 1].set_title('测试准确率')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('准确率 (%)')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.suptitle('CIFAR-10 模型对比', fontsize=16)
plt.tight_layout()
plt.savefig('d7_model_comparison_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("图像已保存: d7_model_comparison_curves.png")


# ============================================================
# 6. 混淆矩阵 (最佳模型)
# ============================================================
print("\n" + "=" * 60)
print("混淆矩阵")
print("=" * 60)

# 选择测试准确率最高的模型
best_name = max(all_results, key=lambda x: x[1]['final_test_acc'])[0]
best_result = max(all_results, key=lambda x: x[1]['final_test_acc'])[1]
best_model = best_result['model']

# 确定使用哪个loader
if 'CustomCNN' in best_name:
    eval_loader = test_loader_custom
else:
    eval_loader = test_loader_pretrained

print(f"最佳模型: {best_name} (Acc: {best_result['final_test_acc']:.2f}%)")

# 生成混淆矩阵
best_model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in eval_loader:
        inputs = inputs.to(device)
        outputs = best_model(inputs)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=classes, yticklabels=classes,
       ylabel='真实标签', xlabel='预测标签',
       title=f'混淆矩阵 - {best_name}')
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# 在每个格子上显示数字
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.savefig('d7_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("图像已保存: d7_confusion_matrix.png")


# ============================================================
# 7. 最终对比表
# ============================================================
print("\n" + "=" * 60)
print("最终对比表")
print("=" * 60)

print(f"\n{'模型':<30} {'测试准确率':>10} {'总参数':>14} {'可训练参数':>14} {'训练时间':>10}")
print("-" * 85)
for name, result, _ in all_results:
    print(f"{name:<30} {result['final_test_acc']:>9.2f}% "
          f"{result['total_params']:>14,} "
          f"{result['trainable_params']:>14,} "
          f"{result['training_time']:>9.1f}s")


# ============================================================
# 8. 可视化对比表
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

model_names = [name for name, _, _ in all_results]
colors_list = [color for _, _, color in all_results]

# 准确率对比
accs = [r['final_test_acc'] for _, r, _ in all_results]
bars = axes[0].bar(model_names, accs, color=colors_list, alpha=0.8, edgecolor='black')
for bar, acc in zip(bars, accs):
    axes[0].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                 f'{acc:.1f}%', ha='center', fontsize=10, fontweight='bold')
axes[0].set_ylabel('测试准确率 (%)')
axes[0].set_title('测试准确率对比')
axes[0].tick_params(axis='x', rotation=15)
axes[0].grid(axis='y', alpha=0.3)

# 参数量对比
totals = [r['total_params'] for _, r, _ in all_results]
trainables = [r['trainable_params'] for _, r, _ in all_results]
x_pos = np.arange(len(model_names))
width = 0.35
bars1 = axes[1].bar(x_pos - width / 2, totals, width, label='总参数', color='#bdc3c7', alpha=0.8)
bars2 = axes[1].bar(x_pos + width / 2, trainables, width, label='可训练参数', color='#2ecc71', alpha=0.8)
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(model_names, rotation=15)
axes[1].set_ylabel('参数量')
axes[1].set_title('参数量对比')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

# 训练时间对比
times = [r['training_time'] for _, r, _ in all_results]
bars3 = axes[2].bar(model_names, times, color=colors_list, alpha=0.8, edgecolor='black')
for bar, t in zip(bars3, times):
    axes[2].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 2,
                 f'{t:.1f}s', ha='center', fontsize=10, fontweight='bold')
axes[2].set_ylabel('训练时间 (秒)')
axes[2].set_title('训练时间对比')
axes[2].tick_params(axis='x', rotation=15)
axes[2].grid(axis='y', alpha=0.3)

plt.suptitle('CIFAR-10 模型综合对比', fontsize=16)
plt.tight_layout()
plt.savefig('d7_final_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("图像已保存: d7_final_comparison.png")


# ============================================================
# 9. 特征可视化 (激活图)
# ============================================================
print("\n" + "=" * 60)
print("特征可视化 (激活图)")
print("=" * 60)

# 使用CustomCNN进行特征可视化
custom_model = results_custom['model']
custom_model.eval()

# 获取一张测试图像
test_img, test_label = test_dataset_custom[0]
input_tensor = test_img.unsqueeze(0).to(device)

print(f"输入图像: 类别 '{classes[test_label]}'")

# 提取各层特征
activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# 注册hook
custom_model.block1.register_forward_hook(get_activation('block1'))
custom_model.block2.register_forward_hook(get_activation('block2'))
custom_model.block3.register_forward_hook(get_activation('block3'))

# 前向传播
with torch.no_grad():
    output = custom_model(input_tensor)
    pred = output.argmax(dim=1).item()

print(f"预测: '{classes[pred]}'")

# 可视化激活图
fig, axes = plt.subplots(3, 8, figsize=(16, 6))

for block_idx, block_name in enumerate(['block1', 'block2', 'block3']):
    act = activations[block_name][0].cpu()
    num_channels = act.shape[0]
    # 选择前8个通道
    for ch in range(min(8, num_channels)):
        ax = axes[block_idx, ch]
        ax.imshow(act[ch].numpy(), cmap='viridis')
        ax.axis('off')
        if ch == 0:
            ax.set_ylabel(f'{block_name}\n({num_channels}通道)', fontsize=10)

plt.suptitle(f'CustomCNN 特征激活图 (输入: {classes[test_label]}, 预测: {classes[pred]})',
             fontsize=14)
plt.tight_layout()
plt.savefig('d7_activation_maps.png', dpi=150, bbox_inches='tight')
plt.show()
print("图像已保存: d7_activation_maps.png")

print("\n激活图解读:")
print("  - Block1 (浅层): 检测边缘、颜色等低级特征")
print("  - Block2 (中层): 检测纹理、形状等中级特征")
print("  - Block3 (深层): 检测更抽象的高级特征（与类别相关）")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("D7 CIFAR-10项目（模型对比与评估）完成！")
    print("=" * 60)
