"""
d3_loss_functions.py - 损失函数
===============================
- 从零实现 MSE, BCE, CCE, Focal Loss
- 绘制每种损失函数的值 vs 预测误差
- 展示每种损失函数的梯度
- 与 PyTorch 实现对比
- 可视化每种损失如何惩罚错误预测
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 损失函数实现（NumPy）
# ============================================================

def mse_loss(y_true, y_pred):
    """均方误差损失 (Mean Squared Error)"""
    return np.mean((y_true - y_pred) ** 2)


def mse_gradient(y_true, y_pred):
    """MSE 对预测值的梯度"""
    return 2 * (y_pred - y_true) / len(y_true)


def bce_loss(y_true, y_pred, eps=1e-7):
    """二元交叉熵损失 (Binary Cross Entropy)"""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def bce_gradient(y_true, y_pred, eps=1e-7):
    """BCE 对预测值的梯度"""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return (-y_true / y_pred + (1 - y_true) / (1 - y_pred)) / len(y_true)


def cce_loss(y_true_onehot, y_pred_probs, eps=1e-7):
    """分类交叉熵损失 (Categorical Cross Entropy)"""
    y_pred_probs = np.clip(y_pred_probs, eps, 1 - eps)
    return -np.mean(np.sum(y_true_onehot * np.log(y_pred_probs), axis=1))


def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25, eps=1e-7):
    """
    Focal Loss: 对易分类样本降权，聚焦难分类样本
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    p_t = np.where(y_true == 1, y_pred, 1 - y_pred)
    alpha_t = np.where(y_true == 1, alpha, 1 - alpha)
    return np.mean(-alpha_t * (1 - p_t) ** gamma * np.log(p_t))


def focal_gradient(y_true, y_pred, gamma=2.0, alpha=0.25, eps=1e-7):
    """Focal Loss 对预测值的梯度"""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    p_t = np.where(y_true == 1, y_pred, 1 - y_pred)
    alpha_t = np.where(y_true == 1, alpha, 1 - alpha)
    # 简化的梯度表达
    grad = (alpha_t * (1 - p_t) ** (gamma - 1) *
            ((gamma * y_pred * np.log(p_t)) - (1 - p_t) * (y_true - y_pred) / y_pred))
    return np.mean(grad) / len(y_true)


# ============================================================
# 2. 与 PyTorch 对比
# ============================================================
print("=" * 60)
print("与 PyTorch 损失函数对比")
print("=" * 60)

np.random.seed(42)
y_true_np = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
y_pred_np = np.array([0.9, 0.1, 0.8, 0.3, 0.7])

# MSE
mse_np = mse_loss(y_true_np, y_pred_np)
mse_torch = nn.MSELoss()(torch.tensor(y_pred_np), torch.tensor(y_true_np)).item()
print(f"\nMSE:")
print(f"  NumPy:   {mse_np:.8f}")
print(f"  PyTorch: {mse_torch:.8f}")
print(f"  差异:    {abs(mse_np - mse_torch):.2e}")

# BCE
bce_np = bce_loss(y_true_np, y_pred_np)
bce_torch = nn.BCELoss()(torch.tensor(y_pred_np), torch.tensor(y_true_np)).item()
print(f"\nBCE:")
print(f"  NumPy:   {bce_np:.8f}")
print(f"  PyTorch: {bce_torch:.8f}")
print(f"  差异:    {abs(bce_np - bce_torch):.2e}")

# CCE (需要 one-hot 格式)
y_true_cc = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # 3个样本，3分类
y_pred_cc = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.3, 0.5]])
cce_np = cce_loss(y_true_cc, y_pred_cc)
cce_torch = nn.CrossEntropyLoss()(
    torch.tensor(y_pred_cc), torch.tensor(np.argmax(y_true_cc, axis=1))
).item()
# 注意：PyTorch 的 CrossEntropyLoss 内部包含 softmax，输入是 logits
# 为公平对比，我们用 NLLLoss + log_softmax
cce_torch2 = nn.NLLLoss()(
    torch.tensor(np.log(y_pred_cc)),
    torch.tensor(np.argmax(y_true_cc, axis=1))
).item()
print(f"\nCCE (分类交叉熵):")
print(f"  NumPy:        {cce_np:.8f}")
print(f"  PyTorch NLL:  {cce_torch2:.8f}")
print(f"  差异:         {abs(cce_np - cce_torch2):.2e}")


# ============================================================
# 3. 可视化：损失函数值 vs 预测
# ============================================================
print("\n" + "=" * 60)
print("绘制损失函数可视化...")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# --- (a) MSE 损失 vs 预测误差 ---
predictions = np.linspace(-2, 4, 300)
y_target = 1.0  # 目标值
mse_values = [(y_target - p) ** 2 for p in predictions]

axes[0, 0].plot(predictions, mse_values, 'b-', linewidth=2)
axes[0, 0].axvline(x=y_target, color='r', linestyle='--', label='真实值 y=1')
axes[0, 0].set_xlabel('预测值')
axes[0, 0].set_ylabel('损失')
axes[0, 0].set_title('MSE 损失: $L = (y - \\hat{y})^2$')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# --- (b) BCE 损失 vs 预测概率 ---
pred_probs = np.linspace(0.01, 0.99, 300)

# y=1 时的 BCE
bce_y1 = -np.log(pred_probs)
# y=0 时的 BCE
bce_y0 = -np.log(1 - pred_probs)

axes[0, 1].plot(pred_probs, bce_y1, 'b-', linewidth=2, label='y=1: $-\\log(\\hat{y})$')
axes[0, 1].plot(pred_probs, bce_y0, 'r-', linewidth=2, label='y=0: $-\\log(1-\\hat{y})$')
axes[0, 1].set_xlabel('预测概率')
axes[0, 1].set_ylabel('损失')
axes[0, 1].set_title('BCE 损失: $L = -[y\\log\\hat{y} + (1-y)\\log(1-\\hat{y})]$')
axes[0, 1].legend()
axes[0, 1].set_ylim(0, 5)
axes[0, 1].grid(True, alpha=0.3)

# --- (c) CCE 损失 vs 正确类概率 ---
correct_prob = np.linspace(0.01, 1.0, 300)
cce_values = -np.log(correct_prob)

axes[0, 2].plot(correct_prob, cce_values, 'g-', linewidth=2)
axes[0, 2].axvline(x=1.0, color='r', linestyle='--', alpha=0.5)
axes[0, 2].set_xlabel('正确类别的预测概率')
axes[0, 2].set_ylabel('损失')
axes[0, 2].set_title('CCE 损失: $L = -\\log(p_{correct})$')
axes[0, 2].set_ylim(0, 5)
axes[0, 2].grid(True, alpha=0.3)

# --- (d) Focal Loss vs BCE 对比 ---
pred_probs2 = np.linspace(0.01, 0.99, 300)
bce_curve = -np.log(pred_probs2)  # y=1 时的 BCE
focal_g2 = -(1 - pred_probs2) ** 2 * np.log(pred_probs2)
focal_g5 = -(1 - pred_probs2) ** 5 * np.log(pred_probs2)

axes[1, 0].plot(pred_probs2, bce_curve, 'b-', linewidth=2, label='BCE (gamma=0)')
axes[1, 0].plot(pred_probs2, focal_g2, 'r-', linewidth=2, label='Focal (gamma=2)')
axes[1, 0].plot(pred_probs2, focal_g5, 'g-', linewidth=2, label='Focal (gamma=5)')
axes[1, 0].set_xlabel('预测概率 (y=1)')
axes[1, 0].set_ylabel('损失')
axes[1, 0].set_title('Focal Loss vs BCE\n(对易分类样本降权)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# --- (e) MSE 梯度 vs 预测误差 ---
errors = np.linspace(-3, 3, 300)
mse_grad_vals = 2 * errors

axes[1, 1].plot(errors, mse_grad_vals, 'b-', linewidth=2)
axes[1, 1].axhline(y=0, color='k', linewidth=0.5)
axes[1, 1].axvline(x=0, color='k', linewidth=0.5)
axes[1, 1].set_xlabel('预测误差 (y_pred - y_true)')
axes[1, 1].set_ylabel('梯度')
axes[1, 1].set_title('MSE 梯度: $\\frac{\\partial L}{\\partial \\hat{y}} = 2(\\hat{y} - y)$')
axes[1, 1].grid(True, alpha=0.3)

# --- (f) BCE 梯度 vs 预测概率 ---
bce_grad_y1 = 1 / pred_probs2       # dL/dp when y=1
bce_grad_y0 = -1 / (1 - pred_probs2)  # dL/dp when y=0

axes[1, 2].plot(pred_probs2, bce_grad_y1, 'b-', linewidth=2, label='y=1')
axes[1, 2].plot(pred_probs2, bce_grad_y0, 'r-', linewidth=2, label='y=0')
axes[1, 2].axhline(y=0, color='k', linewidth=0.5)
axes[1, 2].set_xlabel('预测概率')
axes[1, 2].set_ylabel('梯度')
axes[1, 2].set_title('BCE 梯度')
axes[1, 2].set_ylim(-10, 10)
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.suptitle("损失函数全面对比", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("d3_loss_functions.png", dpi=150, bbox_inches='tight')
plt.show()
print("图保存完成: d3_loss_functions.png")


# ============================================================
# 4. 损失函数惩罚行为对比
# ============================================================
print("\n" + "=" * 60)
print("损失函数惩罚行为对比")
print("=" * 60)

# 模拟不同置信度的预测
confidence_levels = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
print(f"\n{'预测概率':>10} | {'MSE':>10} | {'BCE':>10} | {'Focal(γ=2)':>12} | {'Focal(γ=5)':>12}")
print("-" * 65)
for p in confidence_levels:
    y = np.array([1.0])
    p_arr = np.array([p])
    print(f"{p:>10.2f} | {mse_loss(y, p_arr):>10.6f} | "
          f"{bce_loss(y, p_arr):>10.6f} | "
          f"{focal_loss(y, p_arr, gamma=2.0):>12.6f} | "
          f"{focal_loss(y, p_arr, gamma=5.0):>12.6f}")

print("\n观察:")
print("  - MSE: 对极端错误的惩罚是二次增长")
print("  - BCE: 对极端错误的惩罚是对数增长（更平滑）")
print("  - Focal Loss: gamma 越大，对已正确分类的样本惩罚越小")
print("  - CCE: 关注正确类别的概率，概率越低损失越高")

print("\n" + "=" * 60)
print("总结:")
print("  - MSE 适用于回归问题，BCE 适用于二分类，CCE 适用于多分类")
print("  - Focal Loss 适用于类别不平衡场景")
print("  - NumPy 实现与 PyTorch 完全一致")
print("=" * 60)
