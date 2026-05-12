# 第12周 - PyTorch进阶

> 本周学习PyTorch的高级特性，掌握高效的数据加载、混合精度训练、分布式训练概念、模型导出、实验管理，以及封装可复用的训练框架。

---

## 一、本周目标

1. 掌握自定义Dataset、DataLoader的高级用法（collate_fn、sampler、多进程加载）
2. 理解并实践混合精度训练(AMP)、梯度累积等训练加速技巧
3. 了解分布式训练的基本概念（DP vs DDP）
4. 掌握模型导出方法（TorchScript/ONNX）和实验管理工具（TensorBoard/W&B）
5. 封装一个可复用的通用Trainer类

---

## 二、时间安排

### 工作日（每晚2小时）

| 日期 | 主题 | 时间分配 |
|------|------|----------|
| Day 1 (周一) | Dataset与DataLoader进阶 | 理论30min + 代码实现90min |
| Day 2 (周二) | 模型构建进阶 | 理论40min + 代码练习80min |
| Day 3 (周三) | 训练技巧 | 理论50min + 代码实现70min |
| Day 4 (周四) | 分布式训练基础 | 理论60min + 概念理解60min |
| Day 5 (周五) | 模型导出与实验管理 | 理论50min + 工具实践70min |

### 周末（6-8小时）

| 日期 | 主题 | 时间分配 |
|------|------|----------|
| Day 6 (周六) | Trainer类设计与实现 | 设计60min + 编码180min |
| Day 7 (周日) | Trainer完善与测试 | 测试60min + 集成TB/W&B120min + 总结60min |

---

## 三、详细学习内容

### Day 1: Dataset与DataLoader进阶

#### 1. 自定义Dataset

继承 `torch.utils.data.Dataset`，实现 `__len__` 和 `__getitem__`：

```python
from torch.utils.data import Dataset
from PIL import Image
import json

class ImageTextDataset(Dataset):
    """同时加载图片和文本的自定义Dataset"""
    def __init__(self, annotation_file, img_dir, transform=None, tokenizer=None):
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        self.img_dir = img_dir
        self.transform = transform
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        item = self.annotations[idx]

        # 加载图片
        img_path = f"{self.img_dir}/{item['image']}"
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # 处理文本
        text = item['caption']
        if self.tokenizer:
            text_ids = self.tokenizer.encode(text)
        else:
            text_ids = text

        label = item.get('label', 0)
        return image, text_ids, label
```

#### 2. collate_fn自定义batch组装

当样本包含变长数据（如文本序列）时，需要自定义collate_fn：

```python
from torch.nn.utils.rnn import pad_sequence

def custom_collate_fn(batch):
    """
    batch: list of (image, text_ids, label)
    """
    images, texts, labels = zip(*batch)

    # 图片直接stack
    images = torch.stack(images)

    # 文本padding
    text_lengths = torch.tensor([len(t) for t in texts])
    texts_padded = pad_sequence(
        [torch.tensor(t) for t in texts],
        batch_first=True,
        padding_value=0
    )

    labels = torch.tensor(labels)
    return images, texts_padded, text_lengths, labels

# 使用
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=custom_collate_fn,
    num_workers=4,
    pin_memory=True
)
```

#### 3. DataLoader性能优化参数

```python
DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,        # 多进程加载（CPU核心数的一半）
    pin_memory=True,      # 锁页内存，加速CPU→GPU传输
    prefetch_factor=2,    # 每个worker预取的batch数
    persistent_workers=True  # 保持worker进程存活，避免重复创建
)
```

各参数说明：
- **num_workers**：数据加载的子进程数。0表示主进程加载。推荐设为CPU核心数的1/4到1/2
- **pin_memory**：使用锁页内存(page-locked memory)，允许快速DMA传输到GPU。仅在CUDA训练时有效
- **prefetch_factor**：每个worker预先准备的batch数，减少训练等待
- **persistent_workers**：避免每个epoch重启worker进程

#### 4. WeightedRandomSampler处理类别不平衡

```python
from torch.utils.data import WeightedRandomSampler
import numpy as np

# 假设有1000个样本，类别0有900个，类别1有100个
labels = np.array([0]*900 + [1]*100)

# 计算每个样本的权重（类别频率的倒数）
class_counts = np.bincount(labels)
class_weights = 1.0 / class_counts  # [1/900, 1/100]
sample_weights = class_weights[labels]  # 每个样本的权重

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),  # 一个epoch采样总数
    replacement=True  # 允许重复采样
)

dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
# 不再使用shuffle=True（sampler和shuffle互斥）
```

---

### Day 2: 模型构建进阶

#### 1. nn.Sequential链式构建

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    nn.Conv2d(64, 128, 3, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    nn.Flatten(),
    nn.Linear(128 * 8 * 8, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 10),
)

# 也可以给每层命名
from collections import OrderedDict
model = nn.Sequential(OrderedDict([
    ('conv1', nn.Conv2d(3, 64, 3, padding=1)),
    ('bn1', nn.BatchNorm2d(64)),
    ('relu1', nn.ReLU()),
    ('pool1', nn.MaxPool2d(2, 2)),
]))
```

#### 2. nn.ModuleList与nn.ModuleDict

```python
class DynamicModel(nn.Module):
    def __init__(self, hidden_sizes):
        super().__init__()
        # ModuleList: 动态数量的子模块
        self.layers = nn.ModuleList([
            nn.Linear(in_f, out_f)
            for in_f, out_f in zip(hidden_sizes[:-1], hidden_sizes[1:])
        ])
        # ModuleDict: 按名字索引
        self.norms = nn.ModuleDict({
            f'bn_{i}': nn.BatchNorm1d(size)
            for i, size in enumerate(hidden_sizes[1:])
        })

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.norms[f'bn_{i}'](x)
            x = F.relu(x)
        return x

model = DynamicModel([784, 512, 256, 10])
```

#### 3. 自定义Layer

```python
class Swish(nn.Module):
    """自定义Swish激活函数层"""
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

class残差块(nn.Module):
    """自定义残差块"""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))  # 残差连接 + LayerNorm
```

#### 4. 参数初始化

```python
import torch.nn.init as init

def init_weights(module):
    """统一的权重初始化函数"""
    if isinstance(module, nn.Linear):
        # Xavier初始化（适用于tanh/sigmoid）
        init.xavier_uniform_(module.weight)
        if module.bias is not None:
            init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        # He初始化（适用于ReLU）
        init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
        init.ones_(module.weight)
        init.zeros_(module.bias)

# 应用初始化
model.apply(init_weights)
```

为什么初始化重要：
- **全零初始化**：所有神经元输出相同，梯度相同，对称性无法打破
- **过大初始化**：激活值饱和，梯度消失
- **Xavier初始化**：保持每层输出的方差与输入相同
- **He初始化**：针对ReLU的Xavier变体，考虑ReLU让一半神经元归零

#### 5. nn.Parameter

```python
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # nn.Parameter会自动注册为模型参数
        self.attention_weights = nn.Parameter(
            torch.randn(hidden_dim) * 0.01
        )

    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        scores = x @ self.attention_weights  # (batch, seq_len)
        weights = F.softmax(scores, dim=1)   # (batch, seq_len)
        return (x * weights.unsqueeze(-1)).sum(dim=1)  # (batch, hidden_dim)
```

`nn.Parameter` 与普通Tensor的区别：它会自动被 `model.parameters()` 包含，从而参与梯度更新和模型保存。

---

### Day 3: 训练技巧

#### 1. 学习率Warmup

```python
from torch.optim.lr_scheduler import LambdaLR

def get_linear_warmup_scheduler(optimizer, warmup_steps, total_steps):
    """线性Warmup + 余弦衰减"""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Warmup阶段：线性增加
            return float(current_step) / float(max(1, warmup_steps))
        # 衰减阶段：余弦衰减
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = get_linear_warmup_scheduler(optimizer, warmup_steps=500, total_steps=10000)
```

为什么需要Warmup：训练初期模型参数随机，大的学习率可能导致训练不稳定。Warmup让模型先用小学习率"热身"，逐步过渡到正常学习率。

#### 2. 梯度累积

当GPU显存不够使用大batch_size时，可以用梯度累积模拟大batch：

```python
accumulation_steps = 4  # 等效batch_size = batch_size * 4

optimizer.zero_grad()
for i, (inputs, labels) in enumerate(dataloader):
    outputs = model(inputs)
    loss = criterion(outputs, labels) / accumulation_steps  # 注意除以累积步数
    loss.backward()  # 梯度累积

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()      # 更新参数
        optimizer.zero_grad()  # 清空梯度
```

原理：PyTorch默认梯度是累加的（不自动清零），多次 `backward()` 后一次性 `step()` 等效于大batch训练。

#### 3. 混合精度训练 (AMP)

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()  # 梯度缩放器

for inputs, labels in dataloader:
    inputs, labels = inputs.cuda(), labels.cuda()
    optimizer.zero_grad()

    # 前向传播使用FP16
    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, labels)

    # 反向传播
    scaler.scale(loss).backward()       # 缩放loss防止FP16梯度下溢
    scaler.step(optimizer)               # 反缩放梯度并更新
    scaler.update()                      # 更新缩放因子
```

AMP加速原理：
- **FP16前向**：矩阵运算在Tensor Core上速度提升2-8倍
- **FP16减少显存**：激活值和梯度占用减半，可以用更大batch
- **GradScaler**：防止FP16梯度下溢（值太小变成0）。先放大loss再反向传播，更新时缩小回来

风险与解决：
- FP16精度较低（只有10位有效数字 vs FP32的23位）→ 用FP32做参数更新
- 梯度下溢（小梯度变成0）→ GradScaler自动缩放
- 溢出（值超出FP16范围）→ autocast自动回退到FP32

#### 4. 梯度裁剪

```python
# 按范数裁剪（最常用）
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 按值裁剪
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

---

### Day 4: 分布式训练基础

#### 1. DataParallel (DP) - 单机多卡

```python
# 简单但效率较低
model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
# 就像普通模型一样使用
output = model(input)
```

DP的问题：
- 单进程多线程，受Python GIL限制
- GPU 0是主卡，负责分发和汇总，负载不均衡
- 通信开销大

#### 2. DistributedDataParallel (DDP) - 推荐方式

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# 初始化进程组
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)

# 模型包装
model = model.to(local_rank)
model = DDP(model, device_ids=[local_rank])

# 数据加载
sampler = DistributedSampler(dataset)
dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

# 训练循环
for epoch in range(epochs):
    sampler.set_epoch(epoch)  # 确保每个epoch数据打乱不同
    for batch in dataloader:
        ...
```

#### 3. DP vs DDP对比

| 特性 | DP | DDP |
|------|-----|------|
| 进程模型 | 单进程多线程 | 多进程 |
| GIL瓶颈 | 有 | 无 |
| 通信方式 | Parameter Server | Ring-AllReduce |
| 负载均衡 | GPU 0负载重 | 均衡 |
| 启动方式 | 代码中直接用 | torchrun/accelerate |
| 多机支持 | 不支持 | 支持 |
| 推荐度 | 不推荐 | 推荐 |

**实际使用建议**：直接使用 HuggingFace Accelerate 库，它封装了DDP的复杂度：
```python
from accelerate import Accelerator
accelerator = Accelerator()
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
```

---

### Day 5: 模型导出与实验管理

#### 1. TorchScript导出

```python
import torch

# 方法1：Tracing（追踪，适用于无控制流的模型）
example_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)
traced_model.save("model_traced.pt")

# 方法2：Scripting（脚本化，支持控制流）
scripted_model = torch.jit.script(model)
scripted_model.save("model_scripted.pt")

# 加载
loaded_model = torch.jit.load("model_traced.pt")
output = loaded_model(example_input)
```

TorchScript将PyTorch的动态图转为静态图，可以在没有Python的环境中运行（如C++生产环境）。

#### 2. ONNX导出

```python
torch.onnx.export(
    model,                           # 模型
    example_input,                   # 示例输入
    "model.onnx",                    # 输出文件
    export_params=True,              # 导出参数
    opset_version=14,                # ONNX算子版本
    do_constant_folding=True,        # 常量折叠优化
    input_names=['input'],           # 输入名
    output_names=['output'],         # 输出名
    dynamic_axes={                   # 动态维度
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
```

#### 3. ONNX Runtime推理验证

```python
import onnxruntime as ort
import numpy as np

# 加载ONNX模型
session = ort.InferenceSession("model.onnx")

# 推理
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: example_input.numpy()})

# 与PyTorch对比
with torch.no_grad():
    pt_output = model(example_input)
print(f"最大差异: {np.max(np.abs(output[0] - pt_output.numpy()))}")
# 应该小于1e-5
```

#### 4. TensorBoard实验管理

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment_1')

for epoch in range(num_epochs):
    # 记录标量（loss、accuracy等）
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)

    # 记录学习率
    writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)

    # 记录模型参数分布
    for name, param in model.named_parameters():
        writer.add_histogram(name, param, epoch)

    # 记录图像（如生成的图片、特征图等）
    writer.add_images('predictions', pred_images, epoch)

    # 记录文本
    writer.add_text('hyperparams', f'lr={lr}, batch={batch_size}', epoch)

writer.close()

# 终端启动: tensorboard --logdir=runs
```

#### 5. Weights & Biases

```python
import wandb

# 初始化实验
wandb.init(
    project="my-project",
    config={
        "learning_rate": 1e-3,
        "architecture": "ResNet18",
        "dataset": "CIFAR-10",
        "epochs": 50,
    }
)

# 训练中记录
for epoch in range(epochs):
    wandb.log({
        "loss": train_loss,
        "accuracy": train_acc,
        "epoch": epoch,
    })

wandb.finish()
```

W&B优势：
- 多实验对比（自动生成对比图表）
- 超参数 sweeps（自动搜索最优超参）
- 团队协作（云端共享实验结果）
- 模型版本管理

---

### Day 6-7: 封装Trainer类

```python
"""通用Trainer类设计"""
import os
import yaml
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class Trainer:
    def __init__(self, config_path=None, **kwargs):
        # 从YAML加载配置或使用kwargs
        if config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = kwargs

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.epochs = config.get('epochs', 50)
        self.lr = config.get('learning_rate', 1e-3)
        self.weight_decay = config.get('weight_decay', 1e-4)
        self.grad_clip = config.get('grad_clip', 1.0)
        self.use_amp = config.get('use_amp', False)
        self.accumulation_steps = config.get('accumulation_steps', 1)
        self.patience = config.get('patience', 5)

        # AMP
        self.scaler = GradScaler() if self.use_amp else None

        # 实验记录
        self.writer = SummaryWriter(config.get('log_dir', 'runs/default'))
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def set_model(self, model):
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        self.optimizer.zero_grad()
        progress_bar = tqdm(dataloader, desc='Training')

        for i, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # 前向传播
            if self.use_amp:
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets) / self.accumulation_steps
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets) / self.accumulation_steps
                loss.backward()

            # 梯度累积
            if (i + 1) % self.accumulation_steps == 0:
                if self.grad_clip > 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )

                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.accumulation_steps
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar.set_postfix({
                'loss': f'{total_loss/(i+1):.4f}',
                'acc': f'{100.*correct/total:.1f}%'
            })

        return total_loss / len(dataloader), 100. * correct / total

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            if self.use_amp:
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        return total_loss / len(dataloader), 100. * correct / total

    def fit(self, train_loader, val_loader):
        """完整训练流程"""
        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)

            # 记录
            self.writer.add_scalars('Loss', {
                'train': train_loss, 'val': val_loss
            }, epoch)
            self.writer.add_scalars('Accuracy', {
                'train': train_acc, 'val': val_acc
            }, epoch)

            print(f"Epoch {epoch+1}/{self.epochs}: "
                  f"Train Loss={train_loss:.4f} Acc={train_acc:.1f}% | "
                  f"Val Loss={val_loss:.4f} Acc={val_acc:.1f}%")

            # Early Stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save('best_model.pth')
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        self.writer.close()

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint.get('config')
```

YAML配置文件示例 `config.yaml`：
```yaml
epochs: 50
learning_rate: 0.001
weight_decay: 0.0001
grad_clip: 1.0
use_amp: true
accumulation_steps: 1
patience: 5
log_dir: "runs/resnet18_cifar10"
```

使用方式：
```python
# 从配置文件创建Trainer
trainer = Trainer(config_path='config.yaml')
trainer.set_model(model)
trainer.fit(train_loader, val_loader)
```

---

## 四、代码练习

### Day 1：自定义ImageTextDataset

```python
"""任务：实现一个同时加载图片和文本的Dataset"""
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

class ImageTextDataset(Dataset):
    """图片-文本配对数据集"""
    def __init__(self, data_list, transform=None, max_text_len=50):
        """
        data_list: list of dicts with 'image_path', 'text', 'label'
        """
        self.data = data_list
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.max_text_len = max_text_len

        # 构建简单词表
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        idx = 2
        for item in data_list:
            for word in item['text'].lower().split():
                if word not in self.word2idx:
                    self.word2idx[word] = idx
                    idx += 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['image_path']).convert('RGB')
        image = self.transform(image)

        # 文本编码
        words = item['text'].lower().split()[:self.max_text_len]
        text_ids = [self.word2idx.get(w, 1) for w in words]
        text_len = len(text_ids)

        label = item['label']
        return image, torch.tensor(text_ids, dtype=torch.long), text_len, label

# collate_fn处理变长文本
def collate_fn(batch):
    images, texts, text_lens, labels = zip(*batch)
    images = torch.stack(images)

    # padding
    max_len = max(text_lens)
    padded_texts = torch.zeros(len(texts), max_len, dtype=torch.long)
    for i, t in enumerate(texts):
        padded_texts[i, :len(t)] = t

    text_lens = torch.tensor(text_lens)
    labels = torch.tensor(labels)
    return images, padded_texts, text_lens, labels
```

### Day 3：AMP对比实验

```python
"""任务：对比FP32和FP16的速度和显存"""
import torch
import torch.nn as nn
import time
from torch.cuda.amp import autocast, GradScaler

def benchmark_amp(model, dataloader, use_amp=False, epochs=3):
    device = torch.device('cuda')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler() if use_amp else None

    # 清空缓存
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start_time = time.time()
    for epoch in range(epochs):
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            if use_amp:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

    elapsed = time.time() - start_time
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB

    mode = "AMP (FP16)" if use_amp else "FP32"
    print(f"{mode}: 时间={elapsed:.2f}s, 峰值显存={peak_mem:.0f}MB")
    return elapsed, peak_mem

# 运行对比
# benchmark_amp(model, loader, use_amp=False)  # FP32
# benchmark_amp(model, loader, use_amp=True)   # AMP
```

### Day 6-7：完善Trainer类

参考Day 6-7的完整Trainer实现，在此基础上：
1. 添加W&B集成支持
2. 添加学习率调度器配置
3. 添加模型加载和推理方法
4. 编写单元测试验证各功能

---

## 五、本周产出

### 周末交付物

1. **自定义Dataset**：ImageTextDataset实现，含collate_fn和DataLoader配置 `dataset_utils.py`
2. **AMP对比报告**：FP32 vs FP16的速度和显存对比数据
3. **通用Trainer类**：可复用的训练框架，支持AMP、梯度累积、Early Stopping、TensorBoard `trainer.py`
4. **配置文件**：YAML格式的训练配置模板 `config.yaml`

---

## 六、自测题

### 题1：DataLoader的num_workers和pin_memory的作用？

<details>
<summary>参考答案</summary>

**num_workers**：数据加载使用的子进程数。
- num_workers=0：主进程加载数据，GPU需要等待CPU完成数据准备
- num_workers>0：多个子进程并行加载数据，GPU训练时CPU已经在准备下一个batch
- 推荐：设为CPU核心数的1/4到1/2。太多会导致内存开销大和进程切换开销

**pin_memory**：将数据分配在锁页内存(page-locked/pinned memory)中。
- 锁页内存允许GPU通过DMA直接访问，加速CPU到GPU的数据传输
- 不使用pin_memory时，数据需要先从可分页内存复制到锁页内存，再到GPU
- 仅在使用CUDA训练时有效，CPU训练时无意义
</details>

### 题2：混合精度训练为什么能加速？FP16的风险？

<details>
<summary>参考答案</summary>

加速原因：
1. **计算加速**：FP16在Tensor Core上的矩阵乘法速度是FP32的2-8倍
2. **显存减半**：激活值、梯度占用FP16只需一半显存，可以用更大batch
3. **带宽减半**：GPU显存带宽是瓶颈，FP16数据传输量减半

FP16的风险：
1. **精度损失**：FP16只有10位有效数字（FP32有23位），可能导致累积误差
2. **梯度下溢**：小梯度在FP16中变成0（underflow），参数无法更新
3. **数值溢出**：FP16最大值约65504，大值会溢出变成inf

解决方案：
- GradScaler自动放大loss，防止梯度下溢
- autocast自动对精度敏感的操作（如Softmax、LayerNorm）使用FP32
- 参数主副本保持FP32（FP16只用于前向和反向计算）
</details>

### 题3：DP和DDP的核心区别？

<details>
<summary>参考答案</summary>

核心区别在于进程模型：
- **DP (DataParallel)**：单进程多线程。数据在GPU间传递，但受Python GIL限制，无法真正并行。GPU 0作为主卡负责分发和汇总，负载不均衡。
- **DDP (DistributedDataParallel)**：多进程。每个GPU一个独立进程，绕过GIL。使用Ring-AllReduce通信，各GPU负载均衡。

性能差异：
- DDP比DP快约20-50%
- DDP支持多机多卡，DP仅限单机
- DDP的显存使用更均衡

实际项目中推荐使用DDP（或更高级的封装如HuggingFace Accelerate）。
</details>

### 题4：为什么要用TorchScript/ONNX导出模型？

<details>
<summary>参考答案</summary>

导出模型的目的：
1. **脱离Python依赖**：导出后可在C++/Java/Go等环境中推理，不需要Python运行时
2. **性能优化**：静态图可以进行算子融合、常量折叠等优化
3. **硬件加速**：ONNX可以转换为TensorRT/OpenVINO等专用推理引擎
4. **模型部署**：生产环境通常不用Python+PyTorch部署，而是用轻量级推理引擎

选择：
- TorchScript：PyTorch原生方案，适合C++部署
- ONNX：跨框架标准，支持多种推理后端（TensorRT/ONNX Runtime/OpenVINO）
- 实际项目中ONNX更常用，因为生态更完善
</details>

### 题5：实验管理为什么重要？

<details>
<summary>参考答案</summary>

实验管理的重要性：
1. **可复现**：记录每次实验的超参数、代码版本、随机种子，确保结果可复现
2. **对比分析**：同时可视化多次实验的loss/accuracy曲线，直观对比不同方案
3. **避免重复**：知道哪些超参已经试过，避免做重复实验浪费时间
4. **团队协作**：团队成员可以共享和讨论实验结果
5. **模型选择**：从大量实验中选择最优模型

常见做法：
- 用TensorBoard记录训练指标
- 用W&B管理超参搜索和多实验对比
- 用Git记录代码版本
- 用YAML/JSON记录超参数配置
</details>

---

## 七、Java开发者提示

### 1. DataLoader vs 连接池

| DataLoader概念 | Java中的类比 |
|---------------|-------------|
| Dataset | 数据源（数据库/文件） |
| DataLoader | 连接池（如HikariCP） |
| batch_size | 一次批量查询的条数 |
| num_workers | 连接池大小 |
| prefetch | 预加载/缓存 |
| pin_memory | 直接内存(DirectBuffer) |

```java
// DataLoader的多进程预加载类似数据库连接池
HikariConfig config = new HikariConfig();
config.setMaximumPoolSize(num_workers);  // 工作线程数
config.setConnectionTimeout(30000);       // 超时
// 连接池预先准备好连接，应用需要时直接获取
```

### 2. 混合精度 vs 数据类型的取舍

混合精度的思想和Java中选择数据类型一样：
- 用 `int` 还是 `long`？取决于值域和性能需求
- FP32 vs FP16：精度和性能的权衡
- Java中用 `double` 做精确计算，用 `float` 节省内存
- AMP中用FP32保存参数，FP16做计算

### 3. Trainer类 vs Spring框架的Template模式

Trainer类的设计模式和Spring的JdbcTemplate/RestTemplate相同——封装重复逻辑：
```java
// Spring的Template模式
jdbcTemplate.execute(new ConnectionCallback() {
    public Object doInConnection(Connection conn) {
        // 只关心业务逻辑
    }
});

// Trainer的Template模式
trainer.fit(train_loader, val_loader);
// 训练循环、梯度累积、AMP、日志记录全部封装
// 用户只需提供模型和数据
```

### 4. ONNX导出 vs JVM字节码

ONNX导出类似于Java编译为字节码：
- PyTorch动态图 → ONNX静态图 ≈ Java源码 → 字节码
- ONNX Runtime执行 ≈ JVM执行字节码
- 跨平台：ONNX可以在不同硬件上运行，就像字节码可以在不同OS上运行

### 5. 实验管理 vs CI/CD日志

实验管理和DevOps中的日志管理完全对应：
- TensorBoard ≈ Jenkins/GitLab CI的构建日志
- W&B ≈ Grafana监控面板
- 超参记录 ≈ 构建参数记录
- 模型版本 ≈ Docker镜像版本
