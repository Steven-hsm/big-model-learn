"""
W11-d4: GRU实现
- 从零实现GRU（重置门、更新门）
- PyTorch nn.GRU使用
- 对比LSTM vs GRU：参数量、前向传播时间
- 打印对比表
- 在相同序列预测任务上训练，比较准确率
"""

import numpy as np
import matplotlib.pyplot as plt
import time

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


# ============================================================
# 1. 从零实现GRU
# ============================================================
print("=" * 60)
print("1. GRU 从零实现")
print("=" * 60)


class GRUCell:
    """用NumPy实现的GRU单元"""

    def __init__(self, input_size, hidden_size, seed=42):
        np.random.seed(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 更新门 z_t
        self.W_iz = np.random.randn(input_size, hidden_size) * 0.5
        self.W_hz = np.random.randn(hidden_size, hidden_size) * 0.5
        self.b_z = np.zeros(hidden_size)

        # 重置门 r_t
        self.W_ir = np.random.randn(input_size, hidden_size) * 0.5
        self.W_hr = np.random.randn(hidden_size, hidden_size) * 0.5
        self.b_r = np.zeros(hidden_size)

        # 候选隐藏状态 h_tilde
        self.W_ih = np.random.randn(input_size, hidden_size) * 0.5
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.5
        self.b_h = np.zeros(hidden_size)

    def forward(self, x_t, h_prev):
        """
        GRU单步前向传播
        z_t = sigma(W_iz @ x_t + W_hz @ h_{t-1} + b_z)     更新门
        r_t = sigma(W_ir @ x_t + W_hr @ h_{t-1} + b_r)     重置门
        h_tilde = tanh(W_ih @ x_t + W_hh @ (r_t * h_{t-1}) + b_h)  候选状态
        h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde           最终状态
        """
        # 更新门
        z_t = sigmoid(x_t @ self.W_iz + h_prev @ self.W_hz + self.b_z)
        # 重置门
        r_t = sigmoid(x_t @ self.W_ir + h_prev @ self.W_hr + self.b_r)
        # 候选隐藏状态
        h_tilde = np.tanh(
            x_t @ self.W_ih + (r_t * h_prev) @ self.W_hh + self.b_h
        )
        # 最终隐藏状态
        h_t = (1 - z_t) * h_prev + z_t * h_tilde

        gate_info = {'z': z_t, 'r': r_t, 'h_tilde': h_tilde}
        return h_t, gate_info

    def count_parameters(self):
        total = 0
        total += self.W_iz.size + self.W_hz.size + self.b_z.size
        total += self.W_ir.size + self.W_hr.size + self.b_r.size
        total += self.W_ih.size + self.W_hh.size + self.b_h.size
        return total


# 演示GRU单步计算
np.random.seed(42)
gru_cell = GRUCell(input_size=3, hidden_size=4)

x_t = np.array([1.0, 0.5, -0.3])
h_prev = np.array([0.1, -0.2, 0.3, -0.1])

print(f"\n输入 x_t = {x_t}")
print(f"前一隐藏状态 h_{{t-1}} = {h_prev}")

h_t, gate_info = gru_cell.forward(x_t, h_prev)

print(f"\n更新门 z_t   = {gate_info['z'].round(4)}")
print(f"  含义: 控制多少新信息混合 (1=全部用新的, 0=保留旧的)")
print(f"\n重置门 r_t   = {gate_info['r'].round(4)}")
print(f"  含义: 控制遗忘多少之前的隐藏状态")
print(f"\n候选值 h_tilde = {gate_info['h_tilde'].round(4)}")
print(f"\n(1-z_t)*h_{{t-1}} = {((1 - gate_info['z']) * h_prev).round(4)}")
print(f"z_t*h_tilde    = {(gate_info['z'] * gate_info['h_tilde']).round(4)}")
print(f"\n最终 h_t = {h_t.round(4)}")

# 完整序列前向传播
print("\n" + "-" * 50)
print("GRU序列前向传播（5个时间步）:")
x_seq = np.random.randn(5, 3)
h = np.zeros(4)
for t in range(5):
    h, gates = gru_cell.forward(x_seq[t], h)
    print(f"  t={t}: z={gates['z'].round(3)}, r={gates['r'].round(3)}, h={h.round(4)}")


# ============================================================
# 2. PyTorch nn.GRU 使用
# ============================================================
print("\n" + "=" * 60)
print("2. PyTorch nn.GRU 使用示例")
print("=" * 60)

pytorch_available = False
try:
    import torch
    import torch.nn as nn

    pytorch_gru = nn.GRU(input_size=3, hidden_size=4, num_layers=1,
                         batch_first=True)

    x_tensor = torch.tensor(x_seq, dtype=torch.float32).unsqueeze(0)
    h0 = torch.zeros(1, 1, 4)

    output, hn = pytorch_gru(x_tensor, h0)

    print(f"\n输入形状: {x_tensor.shape}")
    print(f"输出形状: {output.shape}  (batch_size, seq_len, hidden_size)")
    print(f"最终隐藏状态形状: {hn.shape}  (num_layers, batch_size, hidden_size)")

    # 多层GRU
    multi_gru = nn.GRU(input_size=3, hidden_size=8, num_layers=3,
                       batch_first=True, dropout=0.1)
    output2, hn2 = multi_gru(x_tensor)
    print(f"\n多层GRU (3层):")
    print(f"  输出形状: {output2.shape}")
    print(f"  隐藏状态形状: {hn2.shape}")

    total_params = sum(p.numel() for p in pytorch_gru.parameters())
    print(f"\n单层GRU参数量: {total_params}")
    for name, param in pytorch_gru.named_parameters():
        print(f"  {name}: {param.shape}")

    pytorch_available = True

except ImportError:
    print("\nPyTorch未安装，跳过PyTorch GRU示例")


# ============================================================
# 3. LSTM vs GRU 参数量与速度对比
# ============================================================
print("\n" + "=" * 60)
print("3. LSTM vs GRU 对比")
print("=" * 60)


class LSTMCellNP:
    """简化LSTM用于对比"""

    def __init__(self, input_size, hidden_size, seed=42):
        np.random.seed(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        concat_size = input_size + hidden_size
        self.W = np.random.randn(concat_size, hidden_size * 4) * 0.5
        self.b = np.zeros(hidden_size * 4)
        self.b[hidden_size:2 * hidden_size] = 1.0

    def forward(self, x_t, h_prev, c_prev):
        concat = np.concatenate([x_t, h_prev])
        gates = concat @ self.W + self.b
        hs = self.hidden_size
        i_t = sigmoid(gates[:hs])
        f_t = sigmoid(gates[hs:2 * hs])
        g_t = np.tanh(gates[2 * hs:3 * hs])
        o_t = sigmoid(gates[3 * hs:])
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * np.tanh(c_t)
        return h_t, c_t

    def count_parameters(self):
        return self.W.size + self.b.size


# 对比不同配置
configs = [
    (4, 8),
    (8, 16),
    (16, 32),
    (32, 64),
    (64, 128),
    (128, 256),
]

print(f"\n{'输入维度':>8} {'隐藏维度':>8} {'LSTM参数':>10} {'GRU参数':>10} {'比值(GRU/LSTM)':>15}")
print("-" * 55)

lstm_param_list = []
gru_param_list = []

for inp, hid in configs:
    lstm_c = LSTMCellNP(inp, hid)
    gru_c = GRUCell(inp, hid)
    lstm_p = lstm_c.count_parameters()
    gru_p = gru_c.count_parameters()
    ratio = gru_p / lstm_p
    lstm_param_list.append(lstm_p)
    gru_param_list.append(gru_p)
    print(f"{inp:>8d} {hid:>8d} {lstm_p:>10d} {gru_p:>10d} {ratio:>15.2%}")

print("\n结论: GRU参数量约为LSTM的 3/4 (GRU只有3个门，LSTM有4个)")

# 速度对比
print(f"\n前向传播速度对比 (序列长度=100, 重复1000次):")
speed_results = {}

for inp, hid in [(8, 16), (32, 64)]:
    seq_len = 100
    x_test = np.random.randn(seq_len, inp)

    lstm_c = LSTMCellNP(inp, hid)
    gru_c = GRUCell(inp, hid)

    # LSTM计时
    start = time.time()
    for _ in range(1000):
        h = np.zeros(hid)
        c = np.zeros(hid)
        for t in range(seq_len):
            h, c = lstm_c.forward(x_test[t], h, c)
    lstm_time = time.time() - start

    # GRU计时
    start = time.time()
    for _ in range(1000):
        h = np.zeros(hid)
        for t in range(seq_len):
            h, _ = gru_c.forward(x_test[t], h)
    gru_time = time.time() - start

    speedup = lstm_time / gru_time
    speed_results[(inp, hid)] = (lstm_time, gru_time, speedup)
    print(f"  ({inp:>2},{hid:>2}): LSTM={lstm_time:.3f}s, GRU={gru_time:.3f}s, GRU更快{speedup:.2f}倍")


# ============================================================
# 4. 序列预测任务对比
# ============================================================
print("\n" + "=" * 60)
print("4. 序列预测任务 - LSTM vs GRU 训练对比")
print("=" * 60)

np.random.seed(42)

# 生成正弦波预测数据
seq_len_pred = 20
n_samples = 100
look_back = 10

t_vals = np.linspace(0, 8 * np.pi, n_samples + look_back + 1)
sine_data = np.sin(t_vals)

X_all = []
y_all = []
for i in range(n_samples):
    X_all.append(sine_data[i:i + look_back])
    y_all.append(sine_data[i + look_back])

X_all = np.array(X_all).reshape(n_samples, look_back, 1)
y_all = np.array(y_all).reshape(n_samples, 1)

# 划分训练/测试
split = int(0.8 * n_samples)
X_train, X_test = X_all[:split], X_all[split:]
y_train, y_test = y_all[:split], y_all[split:]

print(f"训练样本: {len(X_train)}, 测试样本: {len(X_test)}")
print(f"序列长度: {look_back}, 输入维度: 1, 隐藏维度: 16")


# 训练LSTM
class LSTMTrainer:
    def __init__(self, input_size=1, hidden_size=16, output_size=1, lr=0.005):
        self.hidden_size = hidden_size
        self.lstm = LSTMCellNP(input_size, hidden_size)
        # 输出层
        np.random.seed(42)
        self.W_out = np.random.randn(hidden_size, output_size) * 0.5
        self.b_out = np.zeros(output_size)
        self.lr = lr

    def predict(self, x_seq):
        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)
        for t in range(len(x_seq)):
            h, c = self.lstm.forward(x_seq[t], h, c)
        return h @ self.W_out + self.b_out

    def train_step(self, x_seq, y_true):
        # 前向
        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)
        for t in range(len(x_seq)):
            h, c = self.lstm.forward(x_seq[t], h, c)
        pred = h @ self.W_out + self.b_out
        error = pred - y_true
        loss = np.mean(error ** 2)

        # 简化反向（只更新输出层和最后一步）
        d_pred = 2 * error
        self.W_out -= self.lr * np.outer(h, d_pred)
        self.b_out -= self.lr * d_pred
        return loss


# 训练GRU
class GRUTrainer:
    def __init__(self, input_size=1, hidden_size=16, output_size=1, lr=0.005):
        self.hidden_size = hidden_size
        self.gru = GRUCell(input_size, hidden_size)
        np.random.seed(42)
        self.W_out = np.random.randn(hidden_size, output_size) * 0.5
        self.b_out = np.zeros(output_size)
        self.lr = lr

    def predict(self, x_seq):
        h = np.zeros(self.hidden_size)
        for t in range(len(x_seq)):
            h, _ = self.gru.forward(x_seq[t], h)
        return h @ self.W_out + self.b_out

    def train_step(self, x_seq, y_true):
        h = np.zeros(self.hidden_size)
        for t in range(len(x_seq)):
            h, _ = self.gru.forward(x_seq[t], h)
        pred = h @ self.W_out + self.b_out
        error = pred - y_true
        loss = np.mean(error ** 2)

        d_pred = 2 * error
        self.W_out -= self.lr * np.outer(h, d_pred)
        self.b_out -= self.lr * d_pred
        return loss


# 训练两个模型
lstm_trainer = LSTMTrainer(lr=0.005)
gru_trainer = GRUTrainer(lr=0.005)

epochs = 80
lstm_train_losses = []
gru_train_losses = []

for epoch in range(epochs):
    lstm_epoch_loss = 0
    gru_epoch_loss = 0
    for i in range(len(X_train)):
        lstm_epoch_loss += lstm_trainer.train_step(X_train[i], y_train[i])
        gru_epoch_loss += gru_trainer.train_step(X_train[i], y_train[i])
    lstm_train_losses.append(lstm_epoch_loss / len(X_train))
    gru_train_losses.append(gru_epoch_loss / len(X_train))

# 测试评估
lstm_preds = []
gru_preds = []
for i in range(len(X_test)):
    lstm_preds.append(lstm_trainer.predict(X_test[i]).item())
    gru_preds.append(gru_trainer.predict(X_test[i]).item())

lstm_preds = np.array(lstm_preds)
gru_preds = np.array(gru_preds)
y_test_flat = y_test.flatten()

lstm_mae = np.mean(np.abs(y_test_flat - lstm_preds))
gru_mae = np.mean(np.abs(y_test_flat - gru_preds))
lstm_mse = np.mean((y_test_flat - lstm_preds) ** 2)
gru_mse = np.mean((y_test_flat - gru_preds) ** 2)


# ============================================================
# 打印对比表
# ============================================================
print(f"\n{'=' * 55}")
print(f"{'LSTM vs GRU 对比总结':^55}")
print(f"{'=' * 55}")
print(f"{'指标':<20} {'LSTM':>15} {'GRU':>15}")
print(f"{'-' * 55}")
print(f"{'参数量':<20} {lstm_trainer.lstm.count_parameters():>15d} {gru_trainer.gru.count_parameters():>15d}")
print(f"{'门数量':<20} {'4 (i,f,g,o)':>15} {'3 (z,r,h)':>15}")
print(f"{'细胞状态':<20} {'有':>15} {'无':>15}")
print(f"{'训练最终Loss':<20} {lstm_train_losses[-1]:>15.6f} {gru_train_losses[-1]:>15.6f}")
print(f"{'测试MAE':<20} {lstm_mae:>15.6f} {gru_mae:>15.6f}")
print(f"{'测试MSE':<20} {lstm_mse:>15.6f} {gru_mse:>15.6f}")
print(f"{'=' * 55}")


# ============================================================
# 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图1: GRU门值变化
ax = axes[0, 0]
x_demo = np.random.randn(20, 3)
h_demo = np.zeros(4)
z_history = []
r_history = []
for t in range(20):
    h_demo, gates = gru_cell.forward(x_demo[t], h_demo)
    z_history.append(gates['z'].copy())
    r_history.append(gates['r'].copy())
z_history = np.array(z_history)
r_history = np.array(r_history)
for i in range(4):
    ax.plot(range(20), z_history[:, i], '-', label=f'z[{i}]', alpha=0.7)
    ax.plot(range(20), r_history[:, i], '--', label=f'r[{i}]', alpha=0.7)
ax.set_xlabel('时间步')
ax.set_ylabel('门值')
ax.set_title('GRU门值变化 (实线=更新门z, 虚线=重置门r)')
ax.legend(fontsize=7, ncol=4)
ax.grid(True, alpha=0.3)

# 图2: 参数量对比
ax = axes[0, 1]
x_pos = np.arange(len(configs))
width = 0.35
ax.bar(x_pos - width / 2, lstm_param_list, width, label='LSTM', color='steelblue')
ax.bar(x_pos + width / 2, gru_param_list, width, label='GRU', color='coral')
ax.set_xlabel('配置 (input_size, hidden_size)')
ax.set_ylabel('参数量')
ax.set_title('LSTM vs GRU 参数量对比')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'({i},{h})' for i, h in configs], fontsize=8)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 图3: 训练曲线
ax = axes[1, 0]
ax.plot(range(1, epochs + 1), lstm_train_losses, 'b-', label='LSTM', linewidth=1.5)
ax.plot(range(1, epochs + 1), gru_train_losses, 'r-', label='GRU', linewidth=1.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.set_title('训练损失对比 (正弦波预测)')
ax.legend()
ax.grid(True, alpha=0.3)

# 图4: 预测结果对比
ax = axes[1, 1]
t_test = np.arange(len(y_test_flat))
ax.plot(t_test, y_test_flat, 'k-', label='真实值', linewidth=2, alpha=0.7)
ax.plot(t_test, lstm_preds, 'b--', label='LSTM预测', linewidth=1.5)
ax.plot(t_test, gru_preds, 'r:', label='GRU预测', linewidth=1.5)
ax.set_xlabel('样本')
ax.set_ylabel('值')
ax.set_title(f'测试集预测对比\nLSTM MAE={lstm_mae:.4f}, GRU MAE={gru_mae:.4f}')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W11/d4_gru.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n图片已保存: d4_gru.png")
