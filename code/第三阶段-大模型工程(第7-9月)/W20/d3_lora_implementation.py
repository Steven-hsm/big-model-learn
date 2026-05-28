"""
W20-D3 LoRA实现 (LoRA Implementation)
======================================
从零实现LoRA层(LoRALinear), 注入到预训练模型,
只训练LoRA参数, 梯度检查
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W20-D3 LoRA实现 (LoRA Implementation)")
print("=" * 60)

# ============================================================
# 1. 纯NumPy实现LoRA层
# ============================================================
print("\n--- 1. 纯NumPy实现LoRA层 ---")


class LoRALinear:
    """
    LoRA线性层: y = W*x + (alpha/r) * B*A*x
    W: 冻结的原始权重
    B, A: 可训练的低秩矩阵
    """

    def __init__(self, in_features, out_features, rank=8, alpha=16):
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # 原始权重 (冻结)
        self.weight = np.random.randn(out_features, in_features) * 0.02
        self.bias = np.zeros(out_features)

        # LoRA参数 (可训练)
        # A: 随机初始化, B: 全零初始化
        self.lora_A = np.random.randn(rank, in_features) * 0.01
        self.lora_B = np.zeros((out_features, rank))

        # 梯度存储
        self.d_lora_A = None
        self.d_lora_B = None
        self.d_weight = None  # 应该始终为None(冻结)

        # 缓存
        self._input = None
        self._lora_output = None

    def forward(self, x):
        """前向传播"""
        self._input = x.copy()

        # 原始线性变换 (冻结)
        base_output = x @ self.weight.T + self.bias

        # LoRA部分: x @ A^T @ B^T * scaling
        lora_inter = x @ self.lora_A.T          # (batch, rank)
        lora_output = lora_inter @ self.lora_B.T  # (batch, out_features)
        self._lora_output = lora_output

        return base_output + lora_output * self.scaling

    def backward(self, grad_output, lr=0.001):
        """反向传播 (只更新LoRA参数)"""
        batch_size = grad_output.shape[0]

        # LoRA梯度
        grad_scaled = grad_output * self.scaling

        # d(loss)/d(B) = grad_scaled^T @ (x @ A^T)
        self.d_lora_B = grad_scaled.T @ (self._input @ self.lora_A.T) / batch_size

        # d(loss)/d(A) = (grad_scaled @ B)^T @ x
        self.d_lora_A = (grad_scaled @ self.lora_B).T @ self._input / batch_size

        # 更新LoRA参数 (简单SGD)
        self.lora_B -= lr * self.d_lora_B
        self.lora_A -= lr * self.d_lora_A

        # 注意: 不更新 self.weight (冻结!)

    def merge_weights(self):
        """将LoRA权重合并到原始权重"""
        self.weight = self.weight + self.scaling * (self.lora_B @ self.lora_A)
        self.lora_A = np.zeros_like(self.lora_A)
        self.lora_B = np.zeros_like(self.lora_B)

    def trainable_params(self):
        """可训练参数数量"""
        return self.lora_A.size + self.lora_B.size

    def total_params(self):
        """总参数数量"""
        return self.weight.size + self.bias.size + self.lora_A.size + self.lora_B.size

    def lora_params_ratio(self):
        """LoRA参数占比"""
        return self.trainable_params() / self.total_params() * 100


# 测试LoRA层
print("\n  测试LoRA线性层:")
lora_layer = LoRALinear(in_features=64, out_features=32, rank=4, alpha=8)
x = np.random.randn(2, 64)  # batch=2

output = lora_layer.forward(x)
print(f"  输入形状: {x.shape}")
print(f"  输出形状: {output.shape}")
print(f"  总参数: {lora_layer.total_params():,}")
print(f"  可训练参数: {lora_layer.trainable_params():,}")
print(f"  LoRA参数占比: {lora_layer.lora_params_ratio():.2f}%")

# ============================================================
# 2. 简化Transformer + LoRA
# ============================================================
print("\n--- 2. 简化Transformer + LoRA ---")


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def layer_norm(x, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + eps)


class LoRAAttention:
    """带LoRA的多头注意力"""

    def __init__(self, d_model, n_heads, lora_rank=4, lora_alpha=8):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Q, K, V 投影 - Q和V加LoRA, K不加
        self.Wq = LoRALinear(d_model, d_model, rank=lora_rank, alpha=lora_alpha)
        self.Wk = LoRALinear(d_model, d_model, rank=0, alpha=1)  # 无LoRA
        self.Wv = LoRALinear(d_model, d_model, rank=lora_rank, alpha=lora_alpha)
        self.Wo = LoRALinear(d_model, d_model, rank=0, alpha=1)  # 无LoRA

    def forward(self, x, mask=None):
        batch, seq_len, d = x.shape
        x_2d = x.reshape(-1, d)  # (batch*seq, d)

        Q = self.Wq.forward(x_2d).reshape(batch, seq_len, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        K = self.Wk.forward(x_2d).reshape(batch, seq_len, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        V = self.Wv.forward(x_2d).reshape(batch, seq_len, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_head)
        if mask is not None:
            scores += mask * (-1e9)
        attn = softmax(scores)
        out = np.matmul(attn, V)
        out = out.transpose(0, 2, 1, 3).reshape(-1, d)
        return self.Wo.forward(out).reshape(batch, seq_len, d)

    def backward(self, grad, lr=0.001):
        """只更新有LoRA的层"""
        self.Wq.backward(grad, lr)
        self.Wv.backward(grad, lr)


class LoRATransformerBlock:
    """带LoRA的Transformer块"""

    def __init__(self, d_model, n_heads, d_ff, lora_rank=4, lora_alpha=8):
        self.attention = LoRAAttention(d_model, n_heads, lora_rank, lora_alpha)
        # FFN
        self.ff_w1 = np.random.randn(d_model, d_ff) * 0.02  # 冻结
        self.ff_w2 = np.random.randn(d_ff, d_model) * 0.02  # 冻结

    def forward(self, x, mask=None):
        # Self-Attention + 残差
        attn_out = self.attention.forward(x, mask)
        x = layer_norm(x + attn_out)

        # FFN + 残差
        x_2d = x.reshape(-1, x.shape[-1])
        ff_out = (np.maximum(0, x_2d @ self.ff_w1) @ self.ff_w2).reshape(x.shape)
        x = layer_norm(x + ff_out)
        return x


class LoRATransformerModel:
    """简化的带LoRA的Transformer模型"""

    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, lora_rank=4, lora_alpha=8):
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Embedding (冻结)
        self.token_emb = np.random.randn(vocab_size, d_model) * 0.02
        self.pos_emb = np.random.randn(128, d_model) * 0.02

        # Transformer Blocks (带LoRA)
        self.blocks = [
            LoRATransformerBlock(d_model, n_heads, d_ff, lora_rank, lora_alpha)
            for _ in range(n_layers)
        ]

        # LM Head (冻结)
        self.lm_head = np.random.randn(d_model, vocab_size) * 0.02

    def forward(self, token_ids):
        seq_len = len(token_ids)
        x = (self.token_emb[token_ids] + self.pos_emb[:seq_len])[np.newaxis, :]  # (1, seq, d)

        for block in self.blocks:
            x = block.forward(x)

        x = x.squeeze(0)
        logits = x @ self.lm_head
        return logits

    def trainable_params(self):
        """统计可训练参数"""
        total = 0
        for block in self.blocks:
            attn = block.attention
            total += attn.Wq.trainable_params()
            total += attn.Wv.trainable_params()
        return total

    def total_params(self):
        """统计所有参数"""
        total = self.token_emb.size + self.pos_emb.size + self.lm_head.size
        for block in self.blocks:
            attn = block.attention
            total += attn.Wq.total_params()
            total += attn.Wk.total_params()
            total += attn.Wv.total_params()
            total += attn.Wo.total_params()
            total += block.ff_w1.size + block.ff_w2.size
        return total


# 创建模型
print("\n  创建带LoRA的Transformer模型:")
model = LoRATransformerModel(
    vocab_size=1000, d_model=64, n_heads=4,
    n_layers=2, d_ff=256, lora_rank=4, lora_alpha=8
)

trainable = model.trainable_params()
total = model.total_params()
print(f"  总参数: {total:,}")
print(f"  可训练参数: {trainable:,}")
print(f"  可训练比例: {trainable/total*100:.3f}%")

# 前向传播
input_ids = np.array([1, 5, 10, 15, 20])
logits = model.forward(input_ids)
print(f"\n  前向传播测试:")
print(f"  输入: {input_ids}")
print(f"  输出logits形状: {logits.shape}")

# ============================================================
# 3. 梯度检查
# ============================================================
print("\n--- 3. 梯度检查 (Gradient Check) ---")
print("""
  梯度检查: 验证只有LoRA参数有梯度, 冻结参数没有梯度

  检查项:
    1) 原始权重W的梯度应该为None或零
    2) LoRA参数A和B的梯度应该非零
    3) Embedding和LM Head的梯度应该为None
""")


def gradient_check_demo():
    """梯度检查演示"""
    layer = LoRALinear(32, 16, rank=4, alpha=8)
    x = np.random.randn(4, 32)

    # 前向传播
    output = layer.forward(x)
    loss = np.sum(output ** 2)  # 简单损失

    # 反向传播
    grad = 2 * output  # d(loss)/d(output)
    layer.backward(grad, lr=0.001)

    # 检查梯度
    print(f"  原始权重梯度: {layer.d_weight}")  # 应该是None
    print(f"  LoRA_A梯度范数: {np.linalg.norm(layer.d_lora_A):.6f}")  # 应该>0
    print(f"  LoRA_B梯度范数: {np.linalg.norm(layer.d_lora_B):.6f}")  # 应该>0

    # 检查原始权重是否改变
    W_before = layer.weight.copy()
    layer.backward(grad, lr=0.001)
    W_after = layer.weight
    W_changed = np.linalg.norm(W_after - W_before)
    print(f"  原始权重变化量: {W_changed:.10f} (应该≈0)")
    print(f"  梯度检查通过: {W_changed < 1e-10}")


gradient_check_demo()

# ============================================================
# 4. 训练前后对比
# ============================================================
print("\n--- 4. 训练前后对比 ---")

# 模拟训练过程
layer = LoRALinear(32, 16, rank=4, alpha=8)
x_train = np.random.randn(8, 32)
y_target = np.random.randn(8, 16)

print("\n  模拟训练过程:")
print(f"  {'Epoch':>6s} {'Loss':>10s} {'|ΔW|':>10s} {'|B*A|':>10s}")

initial_output = layer.forward(x_train)
initial_loss = np.mean((initial_output - y_target) ** 2)

W_initial = layer.weight.copy()

for epoch in range(20):
    output = layer.forward(x_train)
    loss = np.mean((output - y_target) ** 2)
    grad = 2 * (output - y_target) / y_target.size
    layer.backward(grad, lr=0.01)

    if epoch % 5 == 0 or epoch == 19:
        W_change = np.linalg.norm(layer.weight - W_initial)
        lora_norm = np.linalg.norm(layer.lora_B @ layer.lora_A)
        print(f"  {epoch:>6d} {loss:>10.4f} {W_change:>10.6f} {lora_norm:>10.6f}")

# ============================================================
# 5. 可视化
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 子图1: LoRA参数分布
ax1 = axes[0]
np.random.seed(42)
A = np.random.randn(4, 32) * 0.01
B = np.zeros((16, 4))
# 模拟训练后的B
B_trained = np.random.randn(16, 4) * 0.1
delta_W = B_trained @ A

ax1.hist(A.flatten(), bins=30, alpha=0.7, label='LoRA A (初始化)', color='#3498db')
ax1.hist(B_trained.flatten(), bins=30, alpha=0.7, label='LoRA B (训练后)', color='#e74c3c')
ax1.set_xlabel('参数值', fontsize=12)
ax1.set_ylabel('频率', fontsize=12)
ax1.set_title('LoRA参数分布', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 子图2: ΔW的奇异值
ax2 = axes[1]
U, S, Vt = np.linalg.svd(delta_W)
ax2.bar(range(len(S)), S, color='#9b59b6', edgecolor='black', alpha=0.85)
ax2.set_xlabel('奇异值索引', fontsize=12)
ax2.set_ylabel('奇异值大小', fontsize=12)
ax2.set_title(f'ΔW的奇异值分布 (有效秩={np.sum(S > 0.01 * S[0])})', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# 子图3: 训练loss曲线
ax3 = axes[2]
layer_viz = LoRALinear(32, 16, rank=4, alpha=8)
losses = []
for epoch in range(50):
    output = layer_viz.forward(x_train)
    loss = np.mean((output - y_target) ** 2)
    losses.append(loss)
    grad = 2 * (output - y_target) / y_target.size
    layer_viz.backward(grad, lr=0.01)

ax3.plot(range(50), losses, linewidth=2, color='#e74c3c')
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('MSE Loss', fontsize=12)
ax3.set_title('LoRA训练Loss曲线', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W20/lora_implementation.png', dpi=150, bbox_inches='tight')
print("\n  图表已保存: lora_implementation.png")
plt.close()

# ============================================================
# 6. PyTorch实现要点
# ============================================================
print("\n--- 6. PyTorch实现要点 ---")
print("""
  关键实现要点:
    1) 冻结原始权重: W.requires_grad = False
    2) LoRA参数: lora_A.requires_grad = True, lora_B.requires_grad = True
    3) 初始化: A用kaiming, B用zeros
    4) 前向: output = F.linear(x, W, b) + (alpha/r) * (x @ A.T @ B.T)
    5) 合并: W_merged = W + (alpha/r) * B @ A

  实际使用推荐PEFT库:
    from peft import LoraConfig, get_peft_model
    config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
    model = get_peft_model(base_model, config)
""")

print("\n" + "=" * 60)
print("W20-D3 完成! 本节从零实现了LoRA层并验证了梯度检查")
print("=" * 60)
