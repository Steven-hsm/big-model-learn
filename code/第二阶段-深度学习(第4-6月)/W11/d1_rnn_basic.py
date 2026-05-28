"""
W11-d1: RNN基础 - 从零实现SimpleRNN
- 用NumPy实现SimpleRNN前向传播（tanh激活）
- 处理简单序列，打印每个时间步的隐藏状态
- 演示参数共享（所有时间步共享相同的W）
- 不同输入/隐藏维度的参数计数
- 正弦波预测任务
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 从零实现SimpleRNN
# ============================================================
class SimpleRNN:
    """用NumPy从零实现的简单RNN（单层，单方向）"""

    def __init__(self, input_size, hidden_size, output_size=None,
                 activation='tanh', seed=42):
        np.random.seed(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation_name = activation

        # 初始化权重（Xavier初始化）
        scale_ih = np.sqrt(2.0 / (input_size + hidden_size))
        scale_hh = np.sqrt(2.0 / (hidden_size + hidden_size))

        self.W_ih = np.random.randn(input_size, hidden_size) * scale_ih  # 输入到隐藏
        self.W_hh = np.random.randn(hidden_size, hidden_size) * scale_hh  # 隐藏到隐藏
        self.b_h = np.zeros(hidden_size)  # 隐藏层偏置

        if output_size is not None:
            self.W_ho = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / (hidden_size + output_size))
            self.b_o = np.zeros(output_size)

    def _activate(self, x):
        if self.activation_name == 'tanh':
            return np.tanh(x)
        elif self.activation_name == 'relu':
            return np.maximum(0, x)
        else:
            return np.tanh(x)

    def forward(self, x_seq, h0=None):
        """
        前向传播
        x_seq: (seq_len, input_size) 输入序列
        h0: (hidden_size,) 初始隐藏状态
        返回: hidden_states, outputs
        """
        seq_len = x_seq.shape[0]
        if h0 is None:
            h0 = np.zeros(self.hidden_size)

        hidden_states = [h0.copy()]

        for t in range(seq_len):
            # h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b_h)
            x_t = x_seq[t]
            h_prev = hidden_states[-1]
            h_t = self._activate(x_t @ self.W_ih + h_prev @ self.W_hh + self.b_h)
            hidden_states.append(h_t)

        hidden_states = np.array(hidden_states)  # (seq_len+1, hidden_size)

        outputs = None
        if self.output_size is not None:
            outputs = hidden_states[1:] @ self.W_ho + self.b_o  # (seq_len, output_size)

        return hidden_states, outputs

    def count_parameters(self):
        """统计参数量"""
        total = self.input_size * self.hidden_size  # W_ih
        total += self.hidden_size * self.hidden_size  # W_hh
        total += self.hidden_size  # b_h
        if self.output_size is not None:
            total += self.hidden_size * self.output_size  # W_ho
            total += self.output_size  # b_o
        return total


# ============================================================
# 2. 处理简单序列，打印隐藏状态
# ============================================================
print("=" * 60)
print("1. SimpleRNN 前向传播演示")
print("=" * 60)

np.random.seed(42)
input_size = 3
hidden_size = 4
output_size = 2

rnn = SimpleRNN(input_size, hidden_size, output_size)

# 创建一个长度为5的输入序列
seq_len = 5
x_seq = np.random.randn(seq_len, input_size)
print(f"\n输入序列形状: {x_seq.shape}")
print(f"输入序列:\n{x_seq.round(3)}")

# 前向传播
hidden_states, outputs = rnn.forward(x_seq)

print(f"\n隐藏状态形状: {hidden_states.shape} (seq_len+1, hidden_size)")
print("\n各时间步隐藏状态:")
for t, h in enumerate(hidden_states):
    if t == 0:
        print(f"  h_0 (初始): {h.round(4)}")
    else:
        print(f"  h_{t} (t={t}):   {h.round(4)}")

print(f"\n输出形状: {outputs.shape}")
print(f"输出:\n{outputs.round(4)}")


# ============================================================
# 3. 演示参数共享
# ============================================================
print("\n" + "=" * 60)
print("2. 参数共享演示")
print("=" * 60)

# 所有时间步使用相同的权重矩阵
print(f"\nW_ih 形状: {rnn.W_ih.shape} — 所有时间步共享")
print(f"W_hh 形状: {rnn.W_hh.shape} — 所有时间步共享")
print(f"b_h  形状: {rnn.b_h.shape}  — 所有时间步共享")

# 手动验证: 对每个时间步，相同的权重产生不同的隐藏状态
print("\n手动验证参数共享（逐步计算）:")
h = np.zeros(hidden_size)
for t in range(seq_len):
    pre_act = x_seq[t] @ rnn.W_ih + h @ rnn.W_hh + rnn.b_h
    h = np.tanh(pre_act)
    print(f"  t={t}: pre_activation={pre_act.round(4)}, h_t={h.round(4)}")
    print(f"         使用的 W_ih id={id(rnn.W_ih)}, W_hh id={id(rnn.W_hh)}")


# ============================================================
# 4. 不同配置的参数计数
# ============================================================
print("\n" + "=" * 60)
print("3. 不同配置的RNN参数量对比")
print("=" * 60)

configs = [
    (1, 8, None, "小模型"),
    (4, 16, None, "中等模型"),
    (10, 32, None, "较大模型"),
    (10, 32, 1, "较大模型+输出层"),
    (50, 64, 10, "大模型+输出层"),
    (100, 128, 10, "超大模型+输出层"),
]

print(f"\n{'描述':<16} {'输入':>4} {'隐藏':>4} {'输出':>4} {'参数量':>8}")
print("-" * 48)
for inp, hid, out, desc in configs:
    r = SimpleRNN(inp, hid, out)
    n = r.count_parameters()
    out_str = str(out) if out else "None"
    print(f"{desc:<16} {inp:>4} {hid:>4} {out_str:>4} {n:>8}")

# 参数量公式说明
print("\n参数量公式: input_size * hidden_size + hidden_size^2 + hidden_size")
print("有输出层时额外加: hidden_size * output_size + output_size")


# ============================================================
# 5. 正弦波预测任务
# ============================================================
print("\n" + "=" * 60)
print("4. 正弦波预测任务")
print("=" * 60)

# 生成正弦波数据
np.random.seed(42)
seq_length = 50
t_vals = np.linspace(0, 4 * np.pi, seq_length + 1)
sine_wave = np.sin(t_vals)

# 任务: 给定前N个点，预测下一个点
look_back = 10  # 用前10个点预测第11个

# 准备数据
X_data = []
y_data = []
for i in range(len(sine_wave) - look_back):
    X_data.append(sine_wave[i:i + look_back])
    y_data.append(sine_wave[i + look_back])

X_data = np.array(X_data)  # (samples, look_back)
y_data = np.array(y_data)  # (samples,)

# RNN输入: (seq_len=look_back, input_size=1)
# 为每个样本reshape
X_rnn = X_data.reshape(-1, look_back, 1)  # (samples, look_back, 1)

# 创建RNN用于回归
rnn_sine = SimpleRNN(input_size=1, hidden_size=16, output_size=1, seed=42)

# 简单的梯度下降训练
lr = 0.01
epochs = 100

print(f"\n训练参数:")
print(f"  look_back = {look_back}")
print(f"  hidden_size = 16")
print(f"  learning_rate = {lr}")
print(f"  epochs = {epochs}")
print(f"  训练样本数 = {len(X_data)}")

# 用于记录训练过程
train_losses = []

for epoch in range(epochs):
    total_loss = 0
    for i in range(len(X_rnn)):
        x_seq_i = X_rnn[i]  # (look_back, 1)
        y_true = y_data[i]

        # 前向传播
        hidden_states, outputs = rnn_sine.forward(x_seq_i)
        y_pred = outputs[-1, 0]  # 取最后一个时间步的输出

        # MSE损失
        error = y_pred - y_true
        loss = error ** 2
        total_loss += loss

        # 反向传播（简化的BPTT，只更新最后一步）
        # d_loss/d_y_pred = 2 * error
        # d_y_pred/d_h = W_ho
        d_y_pred = 2 * error

        # 更新输出层权重
        h_last = hidden_states[-1]
        d_W_ho = np.outer(h_last, np.array([d_y_pred]))
        rnn_sine.W_ho -= lr * d_W_ho
        rnn_sine.b_o -= lr * d_y_pred

        # 简化：只更新隐藏层的梯度（截断BPTT）
        d_h = d_y_pred * rnn_sine.W_ho[:, 0]  # (hidden_size,)
        # tanh的导数: 1 - tanh^2
        d_pre = d_h * (1 - h_last ** 2)

        h_prev = hidden_states[-2]
        x_last = x_seq_i[-1]

        rnn_sine.W_ih -= lr * np.outer(x_last, d_pre)
        rnn_sine.W_hh -= lr * np.outer(h_prev, d_pre)
        rnn_sine.b_h -= lr * d_pre

    avg_loss = total_loss / len(X_rnn)
    train_losses.append(avg_loss)

    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch + 1:3d}, Loss: {avg_loss:.6f}")

# 用训练好的RNN做预测
predictions = []
for i in range(len(X_rnn)):
    hidden_states, outputs = rnn_sine.forward(X_rnn[i])
    predictions.append(outputs[-1, 0])

predictions = np.array(predictions)

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图1: RNN隐藏状态随时间步变化
ax = axes[0, 0]
hidden_states_demo, _ = rnn.forward(X_rnn[0])
for h_idx in range(hidden_size):
    ax.plot(range(len(hidden_states_demo)), hidden_states_demo[:, h_idx],
            label=f'h[{h_idx}]', alpha=0.7)
ax.set_xlabel('时间步')
ax.set_ylabel('隐藏状态值')
ax.set_title('RNN隐藏状态随时间步变化')
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

# 图2: 训练损失曲线
ax = axes[0, 1]
ax.plot(range(1, epochs + 1), train_losses, 'b-', linewidth=1.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.set_title('正弦波预测 - 训练损失')
ax.grid(True, alpha=0.3)

# 图3: 正弦波预测结果
ax = axes[1, 0]
t_pred = t_vals[look_back:]
ax.plot(t_pred, y_data, 'b-', label='真实值', linewidth=2)
ax.plot(t_pred, predictions, 'r--', label='RNN预测', linewidth=1.5)
ax.set_xlabel('t')
ax.set_ylabel('sin(t)')
ax.set_title('正弦波预测结果')
ax.legend()
ax.grid(True, alpha=0.3)

# 图4: 预测误差
ax = axes[1, 1]
errors = y_data - predictions
ax.plot(t_pred, errors, 'g-', linewidth=1)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax.fill_between(t_pred, errors, alpha=0.3, color='green')
ax.set_xlabel('t')
ax.set_ylabel('预测误差')
ax.set_title(f'预测误差 (MAE={np.mean(np.abs(errors)):.4f})')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W11/d1_rnn_basic.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\n最终预测 MAE: {np.mean(np.abs(errors)):.4f}")
print(f"最终预测 MSE: {np.mean(errors**2):.6f}")
print("\n图片已保存: d1_rnn_basic.png")
