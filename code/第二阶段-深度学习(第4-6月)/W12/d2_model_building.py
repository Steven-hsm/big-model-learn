"""
W12-D2: nn.Sequential, nn.ModuleList, nn.ModuleDict, custom layers, weight init
================================================================================
演示 PyTorch 模型构建的各种方式和自定义层。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 1. nn.Sequential: 顺序模型
# ============================================================
def demo_sequential():
    print("=" * 60)
    print("1. nn.Sequential 用法")
    print("=" * 60)

    # 方式一：直接传入层
    model1 = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )

    # 方式二：使用 OrderedDict 命名
    from collections import OrderedDict
    model2 = nn.Sequential(OrderedDict([
        ("fc1", nn.Linear(784, 256)),
        ("relu1", nn.ReLU()),
        ("fc2", nn.Linear(256, 128)),
        ("relu2", nn.ReLU()),
        ("fc3", nn.Linear(128, 10)),
    ]))

    x = torch.randn(4, 784)
    out1 = model1(x)
    out2 = model2(x)

    print(f"Sequential 方式1 输出: {out1.shape}")  # [4, 10]
    print(f"Sequential 方式2 输出: {out2.shape}")  # [4, 10]
    print(f"命名层 fc2 权重 shape: {model2.fc2.weight.shape}")  # [128, 256]
    print()


# ============================================================
# 2. nn.ModuleList: 动态层列表
# ============================================================
class DeepNetwork(nn.Module):
    """使用 nn.ModuleList 构建可变深度的网络"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # nn.ModuleList: PyTorch 会自动注册其中的参数
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)
        return x


def demo_module_list():
    print("=" * 60)
    print("2. nn.ModuleList 用法")
    print("=" * 60)

    model = DeepNetwork(784, 256, 10, num_layers=5)
    x = torch.randn(4, 784)
    out = model(x)
    print(f"输出 shape: {out.shape}")  # [4, 10]
    print(f"隐藏层数量: {len(model.hidden_layers)}")
    print(f"总参数量: {sum(p.numel() for p in model.parameters()):,}")
    print()


# ============================================================
# 3. nn.ModuleDict: 字典式层管理
# ============================================================
class MultiTaskModel(nn.Module):
    """使用 nn.ModuleDict 管理多个任务头"""

    def __init__(self, shared_dim=128):
        super().__init__()
        self.shared_backbone = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, shared_dim),
            nn.ReLU(),
        )

        # 不同任务有不同的输出头
        self.task_heads = nn.ModuleDict({
            "classification": nn.Linear(shared_dim, 10),
            "regression": nn.Linear(shared_dim, 1),
            "multi_label": nn.Linear(shared_dim, 5),
        })

    def forward(self, x, task="classification"):
        features = self.shared_backbone(x)
        return self.task_heads[task](features)


def demo_module_dict():
    print("=" * 60)
    print("3. nn.ModuleDict 用法")
    print("=" * 60)

    model = MultiTaskModel(shared_dim=128)
    x = torch.randn(4, 784)

    for task in ["classification", "regression", "multi_label"]:
        out = model(x, task=task)
        print(f"任务 '{task}' 输出 shape: {out.shape}")
    print()


# ============================================================
# 4. 自定义层
# ============================================================
class Swish(nn.Module):
    """Swish 激活函数: swish(x) = x * sigmoid(beta * x)"""

    def __init__(self, beta=1.0):
        super().__init__()
        # beta 可以作为可学习参数
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class ResidualBlock(nn.Module):
    """残差块: out = activation(x + F(x))"""

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.block(x))


class GatedLinearUnit(nn.Module):
    """门控线性单元: GLU(x) = x_left * sigmoid(x_right)"""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim * 2)

    def forward(self, x):
        projected = self.linear(x)
        left, right = projected.chunk(2, dim=-1)
        return left * torch.sigmoid(right)


def demo_custom_layers():
    print("=" * 60)
    print("4. 自定义层演示")
    print("=" * 60)

    x = torch.randn(4, 128)

    # Swish
    swish = Swish(beta=1.0)
    out_swish = swish(x)
    print(f"Swish 输出 shape: {out_swish.shape}")
    print(f"Swish beta 参数: {swish.beta.item():.4f}")

    # ResidualBlock
    res_block = ResidualBlock(dim=128, dropout=0.1)
    out_res = res_block(x)
    print(f"ResidualBlock 输出 shape: {out_res.shape}")
    print(f"输入和输出是否接近: {torch.allclose(x, out_res, atol=1e-4)}")

    # GLU
    glu = GatedLinearUnit(128, 64)
    out_glu = glu(x)
    print(f"GLU 输出 shape: {out_glu.shape}")
    print()


# ============================================================
# 5. 权重初始化
# ============================================================
def init_weights_xavier(module):
    """Xavier/Glorot 初始化 — 适合 Sigmoid/Tanh"""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.xavier_uniform_(module.weight)


def init_weights_he(module):
    """He/Kaiming 初始化 — 适合 ReLU 及其变体"""
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def demo_weight_init():
    print("=" * 60)
    print("5. 权重初始化演示")
    print("=" * 60)

    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )

    # 默认初始化（均匀分布）
    print(f"默认初始化 fc1 权重 std: {model[0].weight.std().item():.6f}")

    # Xavier 初始化
    model.apply(init_weights_xavier)
    print(f"Xavier 初始化后 fc1 权重 std: {model[0].weight.std().item():.6f}")

    # He 初始化
    model.apply(init_weights_he)
    print(f"He 初始化后 fc1 权重 std: {model[0].weight.std().item():.6f}")

    # 理论值: Xavier uniform 范围 = sqrt(6 / (fan_in + fan_out))
    fan_in, fan_out = 784, 256
    xavier_limit = math.sqrt(6.0 / (fan_in + fan_out))
    print(f"Xavier 理论范围: +/-{xavier_limit:.6f}")

    # 理论值: He normal std = sqrt(2 / fan_in)
    he_std = math.sqrt(2.0 / fan_in)
    print(f"He 理论 std: {he_std:.6f}")
    print()


# ============================================================
# 6. nn.Parameter: 手动管理参数
# ============================================================
class CustomLinear(nn.Module):
    """手动使用 nn.Parameter 实现线性层"""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


def demo_parameter():
    print("=" * 60)
    print("6. nn.Parameter 演示")
    print("=" * 60)

    layer = CustomLinear(128, 64)
    x = torch.randn(4, 128)
    out = layer(x)
    print(f"输出 shape: {out.shape}")
    print(f"参数列表: {[name for name, _ in layer.named_parameters()]}")
    print(f"weight shape: {layer.weight.shape}")
    print(f"bias shape:   {layer.bias.shape}")

    # 验证与 nn.Linear 一致
    ref = nn.Linear(128, 64)
    ref.weight = nn.Parameter(layer.weight.clone())
    ref.bias = nn.Parameter(layer.bias.clone())
    out_ref = ref(x)
    print(f"与 nn.Linear 结果一致: {torch.allclose(out, out_ref)}")
    print()


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)

    demo_sequential()
    demo_module_list()
    demo_module_dict()
    demo_custom_layers()
    demo_weight_init()
    demo_parameter()

    print("所有演示完成！")
