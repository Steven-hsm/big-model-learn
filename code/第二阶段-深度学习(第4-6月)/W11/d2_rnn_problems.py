"""
W11-d2: RNN的问题 - 梯度消失与梯度爆炸
- 演示梯度消失：在长序列上训练RNN，loss停滞
- 演示梯度爆炸：使用大学习率，出现NaN loss
- 实现梯度裁剪
- 可视化：RNN vs LSTM 随时间步的梯度幅度
- 长期依赖问题具体示例
"""

import numpy as np
import matplotlib.pyplot as plt
import time

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 辅助：SimpleRNN实现（带梯度计算）
# ============================================================
class SimpleRNNWithGrad:
    """带BPTT梯度计算的SimpleRNN"""

    def __init__(self, input_size, hidden_size, output_size, seed=42):
        np.random.seed(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        scale = 0.5
        self.W_ih = np.random.randn(input_size, hidden_size) * scale
        self.W_hh = np.random.randn(hidden_size, hidden_size) * scale
        self.b_h = np.zeros(hidden_size)
        self.W_ho = np.random.randn(hidden_size, output_size) * scale
        self.b_o = np.zeros(output_size)

    def forward(self, x_seq):
        seq_len = x_seq.shape[0]
        self.x_seq = x_seq
        self.h = [np.zeros(self.hidden_size)]
        self.pre_act = []

        for t in range(seq_len):
            pre = x_seq[t] @ self.W_ih + self.h[-1] @ self.W_hh + self.b_h
            self.pre_act.append(pre)
            h_t = np.tanh(pre)
            self.h.append(h_t)

        self.h = np.array(self.h)
        output = self.h[-1] @ self.W_ho + self.b_o
        return output

    def backward(self, y_true, lr=0.001, clip_value=None):
        """简化的BPTT"""
        output = self.h[-1] @ self.W_ho + self.b_o
        error = output - y_true
        loss = np.mean(error ** 2)

        # 输出层梯度
        d_out = 2 * error / error.size
        d_W_ho = np.outer(self.h[-1], d_out)
        d_b_o = d_out.copy()

        # 传播到隐藏层
        d_h = d_out @ self.W_ho.T  # (hidden_size,)

        # BPTT
        grad_W_ih = np.zeros_like(self.W_ih)
        grad_W_hh = np.zeros_like(self.W_hh)
        grad_b_h = np.zeros_like(self.b_h)

        seq_len = len(self.pre_act)
        grad_magnitudes = []

        for t in reversed(range(seq_len)):
            # tanh导数
            d_pre = d_h * (1 - self.h[t + 1] ** 2)
            grad_magnitudes.append(np.linalg.norm(d_pre))

            grad_W_ih += np.outer(self.x_seq[t], d_pre)
            grad_W_hh += np.outer(self.h[t], d_pre)
            grad_b_h += d_pre

            # 传播到前一个时间步
            d_h = d_pre @ self.W_hh.T

        grad_magnitudes = grad_magnitudes[::-1]

        # 梯度裁剪
        if clip_value is not None:
            total_norm = np.sqrt(
                np.sum(grad_W_ih ** 2) + np.sum(grad_W_hh ** 2) + np.sum(grad_b_h ** 2)
            )
            if total_norm > clip_value:
                scale = clip_value / (total_norm + 1e-8)
                grad_W_ih *= scale
                grad_W_hh *= scale
                grad_b_h *= scale

        # 更新参数
        self.W_ih -= lr * grad_W_ih
        self.W_hh -= lr * grad_W_hh
        self.b_h -= lr * grad_b_h
        self.W_ho -= lr * d_W_ho
        self.b_o -= lr * d_b_o

        return loss, grad_magnitudes


# ============================================================
# LSTM辅助（简化版，用于梯度对比）
# ============================================================
class SimpleLSTMWithGrad:
    """简化LSTM实现，用于梯度对比"""

    def __init__(self, input_size, hidden_size, output_size, seed=42):
        np.random.seed(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        scale = 0.5
        # 合并的权重 [input; hidden] -> [4*hidden]
        self.W = np.random.randn(input_size + hidden_size, hidden_size * 4) * scale
        self.b = np.zeros(hidden_size * 4)
        self.W_ho = np.random.randn(hidden_size, output_size) * scale
        self.b_o = np.zeros(output_size)

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, x_seq):
        seq_len = x_seq.shape[0]
        self.x_seq = x_seq
        self.h_list = [np.zeros(self.hidden_size)]
        self.c_list = [np.zeros(self.hidden_size)]
        self.gates = []

        for t in range(seq_len):
            xh = np.concatenate([x_seq[t], self.h_list[-1]])
            pre = xh @ self.W + self.b

            f = self._sigmoid(pre[:self.hidden_size])
            i = self._sigmoid(pre[self.hidden_size:2 * self.hidden_size])
            g = np.tanh(pre[2 * self.hidden_size:3 * self.hidden_size])
            o = self._sigmoid(pre[3 * self.hidden_size:])

            c = f * self.c_list[-1] + i * g
            h = o * np.tanh(c)

            self.gates.append((f, i, g, o))
            self.c_list.append(c)
            self.h_list.append(h)

        self.h_list = np.array(self.h_list)
        self.c_list = np.array(self.c_list)

        output = self.h_list[-1] @ self.W_ho + self.b_o
        return output

    def backward(self, y_true, lr=0.001, clip_value=None):
        output = self.h_list[-1] @ self.W_ho + self.b_o
        error = output - y_true
        loss = np.mean(error ** 2)

        d_out = 2 * error / error.size
        d_W_ho = np.outer(self.h_list[-1], d_out)
        d_b_o = d_out.copy()

        d_h = d_out @ self.W_ho.T
        d_c = np.zeros(self.hidden_size)

        grad_W = np.zeros_like(self.W)
        grad_b = np.zeros_like(self.b)
        grad_magnitudes = []

        seq_len = len(self.gates)

        for t in reversed(range(seq_len)):
            f, i, g, o = self.gates[t]
            tanh_c = np.tanh(self.c_list[t + 1])

            d_h_total = d_h
            d_o = d_h_total * tanh_c
            d_c_total = d_h_total * o * (1 - tanh_c ** 2) + d_c

            grad_magnitudes.append(np.linalg.norm(d_h_total))

            d_f = d_c_total * self.c_list[t]
            d_i = d_c_total * g
            d_g = d_c_total * i
            d_c = d_c_total * f

            d_pre = np.concatenate([
                d_f * f * (1 - f),
                d_i * i * (1 - i),
                d_g * (1 - g ** 2),
                d_o * o * (1 - o)
            ])

            xh = np.concatenate([self.x_seq[t], self.h_list[t]])
            grad_W += np.outer(xh, d_pre)
            grad_b += d_pre

            d_xh = d_pre @ self.W.T
            d_h = d_xh[self.input_size:]

        grad_magnitudes = grad_magnitudes[::-1]

        if clip_value is not None:
            total_norm = np.sqrt(np.sum(grad_W ** 2) + np.sum(grad_b ** 2))
            if total_norm > clip_value:
                scale = clip_value / (total_norm + 1e-8)
                grad_W *= scale
                grad_b *= scale

        self.W -= lr * grad_W
        self.b -= lr * grad_b
        self.W_ho -= lr * d_W_ho
        self.b_o -= lr * d_b_o

        return loss, grad_magnitudes


# ============================================================
# 1. 梯度消失演示
# ============================================================
print("=" * 60)
print("1. 梯度消失演示 - 长序列RNN训练")
print("=" * 60)

np.random.seed(42)
seq_lengths = [10, 50, 100, 200]
results_vanish = {}

for sl in seq_lengths:
    # 生成数据: 用序列第一个值预测目标
    x_data = np.random.randn(20, sl, 1)
    y_data = x_data[:, 0, :1] + 0.5  # 目标依赖第一个时间步

    rnn = SimpleRNNWithGrad(1, 8, 1, seed=42)
    losses = []

    for epoch in range(100):
        epoch_loss = 0
        for i in range(len(x_data)):
            rnn.forward(x_data[i])
            loss, _ = rnn.backward(y_data[i], lr=0.01)
            epoch_loss += loss
        losses.append(epoch_loss / len(x_data))

    results_vanish[sl] = losses
    print(f"  序列长度={sl:>3d}: 初始loss={losses[0]:.4f}, 最终loss={losses[-1]:.4f}")


# ============================================================
# 2. 梯度爆炸演示
# ============================================================
print("\n" + "=" * 60)
print("2. 梯度爆炸演示 - 大学习率")
print("=" * 60)

np.random.seed(42)
x_short = np.random.randn(5, 20, 1)
y_short = np.random.randn(5, 1)

for lr_val in [0.001, 0.01, 0.1, 1.0, 5.0]:
    rnn = SimpleRNNWithGrad(1, 8, 1, seed=42)
    exploded = False
    losses = []

    for epoch in range(50):
        epoch_loss = 0
        for i in range(len(x_short)):
            try:
                out = rnn.forward(x_short[i])
                loss, _ = rnn.backward(y_short[i], lr=lr_val)
                if np.isnan(loss) or np.isinf(loss):
                    exploded = True
                    epoch_loss = float('nan')
                    break
                epoch_loss += loss
            except (FloatingPointError, ValueError):
                exploded = True
                epoch_loss = float('nan')
                break

        if exploded:
            losses.append(float('nan'))
            print(f"  lr={lr_val:>5.3f}: 在epoch {epoch + 1}发生梯度爆炸! NaN detected")
            break
        losses.append(epoch_loss / len(x_short))
    else:
        print(f"  lr={lr_val:>5.3f}: 正常训练, 最终loss={losses[-1]:.4f}")


# ============================================================
# 3. 梯度裁剪演示
# ============================================================
print("\n" + "=" * 60)
print("3. 梯度裁剪 vs 无裁剪")
print("=" * 60)

np.random.seed(42)
x_clip = np.random.randn(10, 30, 1)
y_clip = np.random.randn(10, 1)

for clip_val, label in [(None, "无裁剪"), (1.0, "clip=1.0"), (0.5, "clip=0.5")]:
    rnn = SimpleRNNWithGrad(1, 8, 1, seed=42)
    losses = []
    grad_norms = []

    for epoch in range(80):
        epoch_loss = 0
        epoch_grads = []
        for i in range(len(x_clip)):
            rnn.forward(x_clip[i])
            loss, grad_mags = rnn.backward(y_clip[i], lr=0.05, clip_value=clip_val)
            epoch_loss += loss
            if grad_mags:
                epoch_grads.append(np.mean(grad_mags))

        losses.append(epoch_loss / len(x_clip))
        grad_norms.append(np.mean(epoch_grads) if epoch_grads else 0)

    status = "NaN!" if np.isnan(losses[-1]) else f"loss={losses[-1]:.4f}"
    print(f"  {label:>8s}: 最终 {status}, 平均梯度范数={np.nanmean(grad_norms[-10:]):.6f}")


# ============================================================
# 4. RNN vs LSTM 梯度幅度对比
# ============================================================
print("\n" + "=" * 60)
print("4. RNN vs LSTM 梯度幅度随时间步变化")
print("=" * 60)

np.random.seed(42)
test_seq_len = 50
x_long = np.random.randn(1, test_seq_len, 1)
y_long = np.array([[1.0]])

# RNN
rnn_g = SimpleRNNWithGrad(1, 16, 1, seed=42)
rnn_g.forward(x_long[0])
_, rnn_grad_mags = rnn_g.backward(y_long[0], lr=0.001)

# LSTM
lstm_g = SimpleLSTMWithGrad(1, 16, 1, seed=42)
lstm_g.forward(x_long[0])
_, lstm_grad_mags = lstm_g.backward(y_long[0], lr=0.001)

print(f"\n序列长度: {test_seq_len}")
print(f"{'时间步':>6} {'RNN梯度':>12} {'LSTM梯度':>12} {'比值(LSTM/RNN)':>15}")
print("-" * 50)
for t in [0, 5, 10, 20, 30, 40, 49]:
    rnn_val = rnn_grad_mags[t] if t < len(rnn_grad_mags) else 0
    lstm_val = lstm_grad_mags[t] if t < len(lstm_grad_mags) else 0
    ratio = lstm_val / (rnn_val + 1e-10)
    print(f"{t:>6d} {rnn_val:>12.6f} {lstm_val:>12.6f} {ratio:>15.2f}")


# ============================================================
# 5. 长期依赖问题示例
# ============================================================
print("\n" + "=" * 60)
print("5. 长期依赖问题示例")
print("=" * 60)

print("\n任务: 序列第一个元素 + 最后一个元素 -> 预测它们的和")
print("序列中间的元素是噪声（干扰项）")
print("-" * 50)

np.random.seed(42)

# 准备数据
for seq_len in [5, 20, 50]:
    n_samples = 50
    x_data = np.random.randn(n_samples, seq_len, 1)
    # 目标: 第一个和最后一个元素的和
    y_data = (x_data[:, 0, 0] + x_data[:, -1, 0]).reshape(-1, 1)

    # RNN训练
    rnn = SimpleRNNWithGrad(1, 16, 1, seed=42)
    # LSTM训练
    lstm = SimpleLSTMWithGrad(1, 16, 1, seed=42)

    rnn_losses = []
    lstm_losses = []

    for epoch in range(60):
        r_loss = 0
        l_loss = 0
        for i in range(n_samples):
            # RNN
            rnn.forward(x_data[i])
            rl, _ = rnn.backward(y_data[i], lr=0.005, clip_value=5.0)
            r_loss += rl

            # LSTM
            lstm.forward(x_data[i])
            ll, _ = lstm.backward(y_data[i], lr=0.005, clip_value=5.0)
            l_loss += ll

        rnn_losses.append(r_loss / n_samples)
        lstm_losses.append(l_loss / n_samples)

    rnn_final = rnn_losses[-1] if not np.isnan(rnn_losses[-1]) else float('inf')
    lstm_final = lstm_losses[-1] if not np.isnan(lstm_losses[-1]) else float('inf')
    print(f"  seq_len={seq_len:>2d}: RNN最终loss={rnn_final:.4f}, "
          f"LSTM最终loss={lstm_final:.4f}, "
          f"LSTM优势={((rnn_final - lstm_final) / rnn_final * 100):.1f}%")


# ============================================================
# 可视化
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 图1: 梯度消失 - 不同序列长度的训练曲线
ax = axes[0, 0]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
for idx, sl in enumerate(seq_lengths):
    ax.plot(range(1, 101), results_vanish[sl], color=colors[idx],
            label=f'seq_len={sl}', linewidth=1.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('梯度消失: 不同序列长度的RNN训练')
ax.legend()
ax.set_ylim(0, 5)
ax.grid(True, alpha=0.3)

# 图2: RNN vs LSTM 梯度幅度
ax = axes[0, 1]
time_steps = range(1, test_seq_len + 1)
ax.semilogy(time_steps, rnn_grad_mags, 'r-', label='RNN', linewidth=2, alpha=0.8)
ax.semilogy(time_steps, lstm_grad_mags, 'b-', label='LSTM', linewidth=2, alpha=0.8)
ax.set_xlabel('时间步')
ax.set_ylabel('梯度幅度 (对数刻度)')
ax.set_title('RNN vs LSTM 梯度幅度随时间步变化')
ax.legend()
ax.grid(True, alpha=0.3)

# 图3: 梯度裁剪效果
ax = axes[0, 2]
for clip_val, label, color in [(None, "无裁剪", 'red'), (1.0, "clip=1.0", 'blue'), (0.5, "clip=0.5", 'green')]:
    rnn = SimpleRNNWithGrad(1, 8, 1, seed=42)
    losses = []
    for epoch in range(80):
        epoch_loss = 0
        for i in range(len(x_clip)):
            rnn.forward(x_clip[i])
            loss, _ = rnn.backward(y_clip[i], lr=0.05, clip_value=clip_val)
            epoch_loss += loss
        losses.append(epoch_loss / len(x_clip))
    valid_losses = [l if not np.isnan(l) else None for l in losses]
    ax.plot(range(1, 81), valid_losses, color=color, label=label, linewidth=1.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('梯度裁剪效果对比')
ax.legend()
ax.set_ylim(0, 10)
ax.grid(True, alpha=0.3)

# 图4: 长期依赖 - seq_len=5
ax = axes[1, 0]
np.random.seed(42)
x_dep = np.random.randn(50, 5, 1)
y_dep = (x_dep[:, 0, 0] + x_dep[:, -1, 0]).reshape(-1, 1)
rnn_dep = SimpleRNNWithGrad(1, 16, 1, seed=42)
lstm_dep = SimpleLSTMWithGrad(1, 16, 1, seed=42)
r_losses, l_losses = [], []
for epoch in range(60):
    rl_total, ll_total = 0, 0
    for i in range(50):
        rnn_dep.forward(x_dep[i])
        rl, _ = rnn_dep.backward(y_dep[i], lr=0.005, clip_value=5.0)
        rl_total += rl
        lstm_dep.forward(x_dep[i])
        ll, _ = lstm_dep.backward(y_dep[i], lr=0.005, clip_value=5.0)
        ll_total += ll
    r_losses.append(rl_total / 50)
    l_losses.append(ll_total / 50)
ax.plot(range(1, 61), r_losses, 'r-', label='RNN', linewidth=1.5)
ax.plot(range(1, 61), l_losses, 'b-', label='LSTM', linewidth=1.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('长期依赖任务 (seq_len=5)')
ax.legend()
ax.grid(True, alpha=0.3)

# 图5: 长期依赖 - seq_len=50
ax = axes[1, 1]
np.random.seed(42)
x_dep2 = np.random.randn(50, 50, 1)
y_dep2 = (x_dep2[:, 0, 0] + x_dep2[:, -1, 0]).reshape(-1, 1)
rnn_dep2 = SimpleRNNWithGrad(1, 16, 1, seed=42)
lstm_dep2 = SimpleLSTMWithGrad(1, 16, 1, seed=42)
r_losses2, l_losses2 = [], []
for epoch in range(60):
    rl_total, ll_total = 0, 0
    for i in range(50):
        rnn_dep2.forward(x_dep2[i])
        rl, _ = rnn_dep2.backward(y_dep2[i], lr=0.005, clip_value=5.0)
        rl_total += rl
        lstm_dep2.forward(x_dep2[i])
        ll, _ = lstm_dep2.backward(y_dep2[i], lr=0.005, clip_value=5.0)
        ll_total += ll
    r_losses2.append(rl_total / 50)
    l_losses2.append(ll_total / 50)
ax.plot(range(1, 61), r_losses2, 'r-', label='RNN', linewidth=1.5)
ax.plot(range(1, 61), l_losses2, 'b-', label='LSTM', linewidth=1.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('长期依赖任务 (seq_len=50)')
ax.legend()
ax.grid(True, alpha=0.3)

# 图6: RNN梯度指数衰减示意图
ax = axes[1, 2]
t_range = np.arange(1, 51)
# 不同特征值的衰减曲线
for eigenval, label in [(0.99, 'λ=0.99 (好)'), (0.9, 'λ=0.90'), (0.5, 'λ=0.50'), (0.1, 'λ=0.10 (差)')]:
    ax.semilogy(t_range, eigenval ** t_range, linewidth=2, label=label)
ax.set_xlabel('时间步')
ax.set_ylabel('梯度衰减 (对数刻度)')
ax.set_title('RNN梯度指数衰减示意\nh_t = tanh(W*x + W*h_{t-1})')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W11/d2_rnn_problems.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n图片已保存: d2_rnn_problems.png")
