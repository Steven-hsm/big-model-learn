"""
W11-d3: LSTM实现
- 手动LSTM计算：实现一个时间步，显式展示遗忘门/输入门/输出门
- 打印所有门值(f_t, i_t, o_t)和细胞状态更新
- 用NumPy从零实现LSTMCell
- PyTorch nn.LSTM使用示例
- 展示LSTM如何长期���持信息
"""

import numpy as np
import matplotlib.pyplot as plt
import time

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 手动LSTM单步计算（显式展示每个门）
# ============================================================
print("=" * 60)
print("1. LSTM 单步手动计算 - 逐步展示每个门")
print("=" * 60)

np.random.seed(42)
input_size = 3
hidden_size = 4

# LSTM参数
W_ii = np.random.randn(input_size, hidden_size) * 0.5  # 输入门 - 输入权重
W_hi = np.random.randn(hidden_size, hidden_size) * 0.5  # 输入门 - 隐藏权重
b_i = np.zeros(hidden_size)

W_if = np.random.randn(input_size, hidden_size) * 0.5  # 遗忘门 - 输入权重
W_hf = np.random.randn(hidden_size, hidden_size) * 0.5  # 遗忘门 - 隐藏权重
b_f = np.ones(hidden_size)  # 遗忘门偏置通常初始化为1

W_ig = np.random.randn(input_size, hidden_size) * 0.5  # 候选细胞 - 输入权重
W_hg = np.random.randn(hidden_size, hidden_size) * 0.5  # 候选细胞 - 隐藏权重
b_g = np.zeros(hidden_size)

W_io = np.random.randn(input_size, hidden_size) * 0.5  # 输出门 - 输入权重
W_ho = np.random.randn(hidden_size, hidden_size) * 0.5  # 输出门 - 隐藏权重
b_o = np.zeros(hidden_size)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


# 初始状态
h_prev = np.zeros(hidden_size)
c_prev = np.zeros(hidden_size)
x_t = np.array([1.0, 0.5, -0.3])

print(f"\n输入 x_t = {x_t}")
print(f"初始 h_{{t-1}} = {h_prev}")
print(f"初始 c_{{t-1}} = {c_prev}")

# Step 1: 遗忘门
f_t = sigmoid(x_t @ W_if + h_prev @ W_hf + b_f)
print(f"\n--- 遗忘门 f_t ---")
print(f"  f_t = σ(W_if @ x_t + W_hf @ h_{{t-1}} + b_f)")
print(f"  f_t = {f_t.round(4)}")
print(f"  含义: 控制遗忘多少之前的细胞状态 (0=全部遗忘, 1=全部保留)")

# Step 2: 输入门
i_t = sigmoid(x_t @ W_ii + h_prev @ W_hi + b_i)
print(f"\n--- 输入门 i_t ---")
print(f"  i_t = σ(W_ii @ x_t + W_hi @ h_{{t-1}} + b_i)")
print(f"  i_t = {i_t.round(4)}")
print(f"  含义: 控制多少新信息写入细胞状态")

# Step 3: 候选细胞状态
g_t = np.tanh(x_t @ W_ig + h_prev @ W_hg + b_g)
print(f"\n--- 候选细胞状态 g_t ---")
print(f"  g_t = tanh(W_ig @ x_t + W_hg @ h_{{t-1}} + b_g)")
print(f"  g_t = {g_t.round(4)}")
print(f"  含义: 新的候选信息 (范围-1到1)")

# Step 4: 更新细胞状态
c_t = f_t * c_prev + i_t * g_t
print(f"\n--- 细胞状态更新 c_t ---")
print(f"  c_t = f_t * c_{{t-1}} + i_t * g_t")
print(f"  f_t * c_{{t-1}} = {(f_t * c_prev).round(4)} (保留的旧信息)")
print(f"  i_t * g_t     = {(i_t * g_t).round(4)} (写入的新信息)")
print(f"  c_t           = {c_t.round(4)}")

# Step 5: 输出门
o_t = sigmoid(x_t @ W_io + h_prev @ W_ho + b_o)
print(f"\n--- 输出门 o_t ---")
print(f"  o_t = σ(W_io @ x_t + W_ho @ h_{{t-1}} + b_o)")
print(f"  o_t = {o_t.round(4)}")
print(f"  含义: 控制输出多少细胞状态")

# Step 6: 隐藏状态
h_t = o_t * np.tanh(c_t)
print(f"\n--- 最终隐藏状态 h_t ---")
print(f"  h_t = o_t * tanh(c_t)")
print(f"  tanh(c_t) = {np.tanh(c_t).round(4)}")
print(f"  h_t       = {h_t.round(4)}")


# ============================================================
# 2. 用NumPy从零实现LSTMCell
# ============================================================
print("\n" + "=" * 60)
print("2. NumPy LSTMCell 完整实现")
print("=" * 60)


class LSTMCell:
    """用NumPy实现的LSTM单元"""

    def __init__(self, input_size, hidden_size, seed=42):
        np.random.seed(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 合并所有门权重 [input, hidden] -> [4 * hidden]
        # 顺序: [input_gate, forget_gate, cell_gate, output_gate]
        concat_size = input_size + hidden_size
        self.W = np.random.randn(concat_size, hidden_size * 4) * 0.5
        self.b = np.zeros(hidden_size * 4)
        # 遗忘门偏置初始化为1（重要！帮助长期记忆）
        self.b[hidden_size:2 * hidden_size] = 1.0

    def forward(self, x_t, h_prev, c_prev):
        """
        单步前向传播
        返回: h_t, c_t, gate_info
        """
        concat = np.concatenate([x_t, h_prev])

        gates = concat @ self.W + self.b

        # 分割为四个门
        i_t = sigmoid(gates[:self.hidden_size])                   # 输入门
        f_t = sigmoid(gates[self.hidden_size:2 * self.hidden_size])  # 遗忘门
        g_t = np.tanh(gates[2 * self.hidden_size:3 * self.hidden_size])  # 候选细胞
        o_t = sigmoid(gates[3 * self.hidden_size:])               # 输出门

        # 更新细胞状态
        c_t = f_t * c_prev + i_t * g_t
        # 计算隐藏状态
        h_t = o_t * np.tanh(c_t)

        gate_info = {'f': f_t, 'i': i_t, 'g': g_t, 'o': o_t,
                     'c_prev': c_prev, 'c_t': c_t}
        return h_t, c_t, gate_info

    def count_parameters(self):
        return self.W.size + self.b.size


# ============================================================
# 3. 完整LSTM前向传播，打印所有门值
# ============================================================
print("\n完整LSTM前向传播（5个时间步）:")
print("-" * 70)

lstm_cell = LSTMCell(input_size=3, hidden_size=4, seed=42)
np.random.seed(100)
x_seq = np.random.randn(5, 3)

h = np.zeros(4)
c = np.zeros(4)

all_gates = []
for t in range(5):
    h, c, gate_info = lstm_cell.forward(x_seq[t], h, c)
    all_gates.append(gate_info)
    print(f"\n时间步 t={t}:")
    print(f"  输入 x_t    = {x_seq[t].round(3)}")
    print(f"  遗忘门 f_t  = {gate_info['f'].round(4)}")
    print(f"  输入门 i_t  = {gate_info['i'].round(4)}")
    print(f"  候选值 g_t  = {gate_info['g'].round(4)}")
    print(f"  输出门 o_t  = {gate_info['o'].round(4)}")
    print(f"  细胞状态 c_t= {gate_info['c_t'].round(4)}")
    print(f"  隐藏状态 h_t= {h.round(4)}")

print(f"\nLSTM参数量: {lstm_cell.count_parameters()}")


# ============================================================
# 4. PyTorch nn.LSTM 使用示例
# ============================================================
print("\n" + "=" * 60)
print("4. PyTorch nn.LSTM 使用示例")
print("=" * 60)

try:
    import torch
    import torch.nn as nn

    # 创建LSTM
    pytorch_lstm = nn.LSTM(input_size=3, hidden_size=4, num_layers=1,
                           batch_first=True)

    # 输入: (batch_size, seq_len, input_size)
    x_tensor = torch.tensor(x_seq, dtype=torch.float32).unsqueeze(0)

    # 初始隐藏状态和细胞状态
    h0 = torch.zeros(1, 1, 4)
    c0 = torch.zeros(1, 1, 4)

    # 前向传播
    output, (hn, cn) = pytorch_lstm(x_tensor, (h0, c0))

    print(f"\n输入形状: {x_tensor.shape}")
    print(f"输出形状: {output.shape}  (batch_size, seq_len, hidden_size)")
    print(f"最终隐藏状态形状: {hn.shape}  (num_layers, batch_size, hidden_size)")
    print(f"最终细胞状态形状: {cn.shape}  (num_layers, batch_size, hidden_size)")

    # 多层LSTM
    multi_lstm = nn.LSTM(input_size=3, hidden_size=8, num_layers=3,
                         batch_first=True, dropout=0.1)
    output2, (hn2, cn2) = multi_lstm(x_tensor)
    print(f"\n多层LSTM (3层):")
    print(f"  输出形状: {output2.shape}  (最后一层��输出)")
    print(f"  隐藏状态形状: {hn2.shape}  (num_layers, batch, hidden)")
    print(f"  细胞状态形状: {cn2.shape}")

    # 参数统计
    total_params = sum(p.numel() for p in pytorch_lstm.parameters())
    print(f"\n单层LSTM参数量: {total_params}")
    for name, param in pytorch_lstm.named_parameters():
        print(f"  {name}: {param.shape}")

    pytorch_available = True

except ImportError:
    print("\nPyTorch未安装，跳过PyTorch示例")
    print("安装命令: pip install torch")
    pytorch_available = False


# ============================================================
# 5. LSTM如何长期保持信息
# ============================================================
print("\n" + "=" * 60)
print("5. LSTM长期信息保持演示")
print("=" * 60)

# 实验: 在t=0设置一个值，观察它在LSTM中能保持多久
lstm_long = LSTMCell(input_size=1, hidden_size=8, seed=42)

h_init = np.zeros(8)
c_init = np.array([1.0, 2.0, -1.0, 0.5, -0.5, 1.5, -2.0, 0.8])  # 在t=0设置的值

# 之后输入全零（不提供新信息），观察细胞状态是否保持
h = h_init.copy()
c = c_init.copy()

print("\n初始细胞状态 c_0 =", c_init.round(4))
print("之后输入全零，观察信息保持:\n")
print(f"{'时间步':>6} {'|c_t - c_0| (L1距离)':>22} {'遗忘门均值':>12} {'c_t[0]':>10}")
print("-" * 55)

c_history = [c.copy()]
forget_means = []
h_history = [h.copy()]

for t in range(30):
    x_zero = np.array([0.0])
    h, c, gate_info = lstm_long.forward(x_zero, h, c)
    c_history.append(c.copy())
    forget_means.append(np.mean(gate_info['f']))
    h_history.append(h.copy())
    if t % 5 == 0 or t == 29:
        diff = np.abs(c - c_init)
        print(f"{t + 1:>6d} {np.mean(diff):>22.6f} {np.mean(gate_info['f']):>12.4f} {c[0]:>10.4f}")

c_history = np.array(c_history)

print(f"\n30步后 c_30 = {c.round(4)}")
print(f"初始值   c_0  = {c_init.round(4)}")
print(f"L1距离 = {np.mean(np.abs(c - c_init)):.6f}")
print("=> LSTM通过遗忘门控制，即使没有新输入也能保持信息")


# ============================================================
# 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图1: LSTM门值热力图
ax = axes[0, 0]
gate_names = ['遗忘门(f)', '输入门(i)', '候选值(g)', '输出门(o)']
gate_arrays = [np.array([g['f'] for g in all_gates]),
               np.array([g['i'] for g in all_gates]),
               np.array([g['g'] for g in all_gates]),
               np.array([g['o'] for g in all_gates])]
combined = np.vstack(gate_arrays)
im = ax.imshow(combined, aspect='auto', cmap='RdYlBu_r', vmin=-1, vmax=1)
ax.set_yticks(range(4 * 4))
ax.set_yticklabels(
    [f'{name}[{i}]' for name in gate_names for i in range(4)],
    fontsize=7
)
ax.set_xlabel('时间步')
ax.set_title('LSTM各门值随时间步变化')
plt.colorbar(im, ax=ax)

# 图2: 细胞状态和隐藏状态
ax = axes[0, 1]
for i in range(4):
    ax.plot(range(5), [g['c_t'][i] for g in all_gates],
            marker='o', label=f'c[{i}]', linewidth=1.5)
ax.set_xlabel('时间步')
ax.set_ylabel('细胞状态值')
ax.set_title('细胞状态 c_t 随时间步变化')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 图3: LSTM长期信息保持
ax = axes[1, 0]
for i in range(8):
    ax.plot(range(31), c_history[:, i], linewidth=1.5,
            label=f'c[{i}]={c_init[i]:.1f}')
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('时间步 (全零输入)')
ax.set_ylabel('细胞状态值')
ax.set_title('LSTM长期信息保持\n(初始设置c值后输入全零)')
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

# 图4: 遗忘门均值变化
ax = axes[1, 1]
ax.plot(range(1, 31), forget_means, 'r-o', markersize=4, linewidth=1.5)
ax.axhline(y=0.5, color='k', linestyle='--', alpha=0.3, label='0.5阈值')
ax.fill_between(range(1, 31), forget_means, alpha=0.2, color='red')
ax.set_xlabel('时间步')
ax.set_ylabel('遗忘门均值')
ax.set_title('遗忘门均值变化\n(输入全零时)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W11/d3_lstm.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n图片已保存: d3_lstm.png")
