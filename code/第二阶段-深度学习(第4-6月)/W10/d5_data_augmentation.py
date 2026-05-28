"""
W10-D5 数据增强与迁移学习 (Data Augmentation & Transfer Learning)
================================================================
演示 torchvision transforms、Mixup、迁移学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 数据增强演示
# ============================================================
print("=" * 60)
print("数据增强演示")
print("=" * 60)

# 定义各种增强变换
augmentation_transforms = {
    '原始图像': transforms.Compose([
        transforms.ToTensor(),
    ]),
    '随机裁剪': transforms.Compose([
        transforms.RandomCrop(24),
        transforms.ToTensor(),
    ]),
    '水平翻转': transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
    ]),
    '颜色抖动': transforms.Compose([
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        transforms.ToTensor(),
    ]),
    '随机旋转': transforms.Compose([
        transforms.RandomRotation(degrees=30),
        transforms.ToTensor(),
    ]),
    '随机仿射': transforms.Compose([
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
    ]),
    '高斯模糊': transforms.Compose([
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
    ]),
    '组合增强': transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ]),
}

# 使用 CIFAR-10 或创建示例图像
# 创建一个示例图像 (PIL)
from PIL import Image

sample_array = np.zeros((32, 32, 3), dtype=np.uint8)
# 画一些图案
sample_array[:16, :16, 0] = 200  # 红色方块 (左上)
sample_array[16:, 16:, 1] = 200  # 绿色方块 (右下)
sample_array[:16, 16:, 2] = 200  # 蓝色方块 (右上)
sample_array[16:, :16, :] = 128  # 灰色方块 (左下)
sample_img = Image.fromarray(sample_array)

# 可视化增强效果
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for idx, (name, transform) in enumerate(augmentation_transforms.items()):
    torch.manual_seed(42)
    augmented = transform(sample_img)
    # 转换为显示格式
    img_np = augmented.numpy().transpose(1, 2, 0)
    img_np = np.clip(img_np, 0, 1)
    axes[idx].imshow(img_np)
    axes[idx].set_title(name, fontsize=12)
    axes[idx].axis('off')

plt.suptitle('数据增强效果演示', fontsize=16)
plt.tight_layout()
plt.savefig('d5_augmentation_demo.png', dpi=150, bbox_inches='tight')
plt.show()
print("图像已保存: d5_augmentation_demo.png")


# ============================================================
# 2. 多次增强同一图像
# ============================================================
print("\n--- 同一图像多次随机增强 ---")

transform_multi = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
])

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
# 第一行显示原图
for i in range(5):
    img_np = np.array(sample_img) / 255.0
    axes[0, i].imshow(img_np)
    axes[0, i].set_title(f'原始' if i == 0 else '', fontsize=10)
    axes[0, i].axis('off')

# 第二行显示增强后
for i in range(5):
    torch.manual_seed(i * 10)
    augmented = transform_multi(sample_img)
    img_np = augmented.numpy().transpose(1, 2, 0)
    img_np = np.clip(img_np, 0, 1)
    axes[1, i].imshow(img_np)
    axes[1, i].set_title(f'增强 #{i + 1}', fontsize=10)
    axes[1, i].axis('off')

plt.suptitle('同一图像的多种增强结果', fontsize=14)
plt.tight_layout()
plt.savefig('d5_multiple_augmentations.png', dpi=150, bbox_inches='tight')
plt.show()
print("图像已保存: d5_multiple_augmentations.png")


# ============================================================
# 3. Mixup 数据增强
# ============================================================
print("\n" + "=" * 60)
print("Mixup 数据增强")
print("=" * 60)


def mixup_data(x, y, alpha=0.4):
    """
    Mixup 数据增强: 线性插值混合两个样本

    x_mix = lambda * x_i + (1 - lambda) * x_j
    y_mix = lambda * y_i + (1 - lambda) * y_j

    参数:
        x: 输入数据 (batch, ...)
        y: 标签 (batch,)
        alpha: Beta分布参数

    返回:
        mixed_x, y_a, y_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup损失函数"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# 演示 Mixup
print("\nMixup 示例:")
torch.manual_seed(42)

# 创建两个简单的"图像"
img_a = torch.zeros(1, 3, 8, 8)
img_a[0, 0] = 1.0  # 红色图像

img_b = torch.zeros(1, 3, 8, 8)
img_b[0, 1] = 1.0  # 绿色图像

batch = torch.cat([img_a, img_b], dim=0)
labels = torch.tensor([0, 1])

mixed_batch, y_a, y_b, lam = mixup_data(batch, labels, alpha=0.4)

print(f"  lambda值: {lam:.4f}")
print(f"  标签A: {y_a.tolist()}, 标签B: {y_b.tolist()}")
print(f"  混合后: {lam:.2f} * 样本A + {1 - lam:.2f} * 样本B")

# 可视化 Mixup
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

img_a_np = img_a[0].numpy().transpose(1, 2, 0)
axes[0].imshow(img_a_np)
axes[0].set_title('图像A (红色)')
axes[0].axis('off')

img_b_np = img_b[0].numpy().transpose(1, 2, 0)
axes[1].imshow(img_b_np)
axes[1].set_title('图像B (绿色)')
axes[1].axis('off')

mixed_np = mixed_batch[0].numpy().transpose(1, 2, 0)
mixed_np = np.clip(mixed_np, 0, 1)
axes[2].imshow(mixed_np)
axes[2].set_title(f'Mixup (lambda={lam:.2f})')
axes[2].axis('off')

plt.suptitle('Mixup 数据增强', fontsize=14)
plt.tight_layout()
plt.savefig('d5_mixup_demo.png', dpi=150, bbox_inches='tight')
plt.show()
print("图像已保存: d5_mixup_demo.png")

print("""
Mixup 数据增强原理:
  1. 从 Beta 分布采样 lambda
  2. 混合两个样本: x_mix = lambda * x_i + (1-lambda) * x_j
  3. 混合标签:   y_mix = lambda * y_i + (1-lambda) * y_j
  4. 用混合后的数据和标签训练

优势:
  - 减少过拟合
  - 提高模型泛化能力
  - 使决策边界更平滑
  - 对对抗样本更鲁棒
""")


# ============================================================
# 4. 迁移学习
# ============================================================
print("\n" + "=" * 60)
print("迁移学习 (Transfer Learning)")
print("=" * 60)

# 加载预训练 ResNet-18
print("\n加载预训练 ResNet-18...")
weights = ResNet18_Weights.DEFAULT
pretrained_resnet = resnet18(weights=weights)

# 查看原始分类器
print(f"\n原始全连接层: {pretrained_resnet.fc}")
total_pretrained = sum(p.numel() for p in pretrained_resnet.parameters())
print(f"预训练模型总参数量: {total_pretrained:,}")

# ============================================================
# 策略1: 冻结骨干网络，只训练分类头
# ============================================================
print("\n--- 策略1: 冻结骨干网络 ---")

model_frozen = resnet18(weights=weights)

# 冻结所有参数
for param in model_frozen.parameters():
    param.requires_grad = False

# 替换最后的全连接层 (ImageNet 1000类 -> 自定义 N 类)
num_classes = 10
model_frozen.fc = nn.Linear(512, num_classes)

# 统计参数
frozen_total = sum(p.numel() for p in model_frozen.parameters())
frozen_trainable = sum(p.numel() for p in model_frozen.parameters() if p.requires_grad)
frozen_frozen_count = frozen_total - frozen_trainable

print(f"  总参数量: {frozen_total:,}")
print(f"  可训练参数: {frozen_trainable:,} ({frozen_trainable / frozen_total * 100:.2f}%)")
print(f"  冻结参数: {frozen_frozen_count:,} ({frozen_frozen_count / frozen_total * 100:.2f}%)")

# 详细查看每层是否冻结
print("\n  冻结状态详情:")
print(f"  {'层':<40} {'形状':<20} {'可训练':>8}")
print("  " + "-" * 72)
for name, param in model_frozen.named_parameters():
    status = "Yes" if param.requires_grad else "No"
    print(f"  {name:<40} {str(list(param.shape)):<20} {status:>8}")

# ============================================================
# 策略2: 部分解冻（微调最后几层）
# ============================================================
print("\n--- 策略2: 部分解冻 (微调layer4 + fc) ---")

model_finetune = resnet18(weights=weights)

# 先冻结所有
for param in model_finetune.parameters():
    param.requires_grad = False

# 解冻 layer4
for param in model_finetune.layer4.parameters():
    param.requires_grad = True

# 替换 fc
model_finetune.fc = nn.Linear(512, num_classes)
# fc 层默认 requires_grad=True

finetune_total = sum(p.numel() for p in model_finetune.parameters())
finetune_trainable = sum(p.numel() for p in model_finetune.parameters() if p.requires_grad)

print(f"  总参数量: {finetune_total:,}")
print(f"  可训练参数: {finetune_trainable:,} ({finetune_trainable / finetune_total * 100:.2f}%)")

print("\n  各层可训练状态:")
layer_status = {}
for name, param in model_finetune.named_parameters():
    # 获取层名
    parts = name.split('.')
    if len(parts) >= 2:
        layer_name = '.'.join(parts[:2]) if parts[0] != 'fc' else 'fc'
    else:
        layer_name = name
    if layer_name not in layer_status:
        layer_status[layer_name] = param.requires_grad

for ln, trainable in layer_status.items():
    status = "可训练" if trainable else "冻结"
    print(f"    {ln:<25} -> {status}")


# ============================================================
# 5. 迁移学习参数对比可视化
# ============================================================
print("\n--- 迁移学习策略对比 ---")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 参数量对比
strategies = ['从头训练\n(全部参数)', '冻结骨干\n(只训练FC)', '部分解冻\n(layer4+FC)']
all_params = [total_pretrained, frozen_total, finetune_total]
trainable_params = [total_pretrained, frozen_trainable, finetune_trainable]
frozen_params = [0, frozen_total - frozen_trainable, finetune_total - finetune_trainable]

x_pos = np.arange(len(strategies))
width = 0.5

bars_frozen = axes[0].bar(x_pos, frozen_params, width, label='冻结参数',
                          color='#bdc3c7', alpha=0.8)
bars_train = axes[0].bar(x_pos, trainable_params, width, bottom=frozen_params,
                         label='可训练参数', color='#2ecc71', alpha=0.8)

for i, (tp, fp) in enumerate(zip(trainable_params, frozen_params)):
    axes[0].text(i, fp + tp + total_pretrained * 0.02,
                 f'训练: {tp:,}', ha='center', fontsize=9)

axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(strategies)
axes[0].set_ylabel('参数量')
axes[0].set_title('迁移学习策略参数对比')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# 迁移学习工作流程图
axes[1].set_xlim(0, 10)
axes[1].set_ylim(0, 10)
axes[1].axis('off')
axes[1].set_title('迁移学习工作流程', fontsize=14)

# 步骤
steps = [
    (5, 9, '1. 加载预训练模型\n(ResNet-18, ImageNet)', '#3498db'),
    (5, 7, '2. 冻结骨干网络\n(param.requires_grad=False)', '#e74c3c'),
    (5, 5, '3. 替换分类头\n(nn.Linear(512, num_classes))', '#2ecc71'),
    (5, 3, '4. 训练分类头\n(只更新FC层参数)', '#f39c12'),
    (5, 1, '5. (可选) 微调\n(解冻部分层, 小学习率)', '#9b59b6'),
]

for x, y, text, color in steps:
    rect = plt.Rectangle((x - 2.5, y - 0.7), 5, 1.4,
                          facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
    axes[1].add_patch(rect)
    axes[1].text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

# 箭头
for i in range(len(steps) - 1):
    axes[1].annotate('', xy=(5, steps[i + 1][1] + 0.7),
                     xytext=(5, steps[i][1] - 0.7),
                     arrowprops=dict(arrowstyle='->', color='black', lw=2))

plt.tight_layout()
plt.savefig('d5_transfer_learning.png', dpi=150, bbox_inches='tight')
plt.show()
print("图像已保存: d5_transfer_learning.png")


# ============================================================
# 6. 实际使用示例
# ============================================================
print("\n" + "=" * 60)
print("迁移学习代码模板")
print("=" * 60)

print("""
# === 迁移学习标准代码模板 ===

# 1. 数据预处理 (使用预训练模型的标准化参数)
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])

# 2. 加载预训练模型
model = resnet18(weights=ResNet18_Weights.DEFAULT)

# 3. 冻结骨干
for param in model.parameters():
    param.requires_grad = False

# 4. 替换分类头
model.fc = nn.Linear(512, num_classes)

# 5. 只优化分类头参数
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

# 6. 训练
for epoch in range(num_epochs):
    for images, labels in train_loader:
        output = model(images)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 7. 微调 (可选)
for param in model.layer4.parameters():
    param.requires_grad = True
optimizer = torch.optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-5},
    {'params': model.fc.parameters(), 'lr': 1e-3},
])
""")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("D5 数据增强与迁移学习 完成！")
    print("=" * 60)
