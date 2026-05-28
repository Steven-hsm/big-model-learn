"""
W16-D5 知识蒸馏 (Knowledge Distillation)
==========================================
知识蒸馏原理, Teacher-Student 训练循环,
软目标与温度, 蒸馏损失实现, 对比三种模型的准确率
"""

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W16-D5 知识蒸馏 (Knowledge Distillation)")
print("=" * 60)

# ============================================================
# 1. 知识蒸馏原理
# ============================================================
print("\n--- 1. 知识蒸馏原理 ---")
print("""
  知识蒸馏 (Hinton et al., 2015):
    用大模型 (Teacher) 的知识训练小模型 (Student)

    Teacher (大模型):
      - 参数多, 精度高, 速度慢
      - 输出 "软标签" (soft targets) 包含类间关系

    Student (小模型):
      - 参数少, 精度接近 Teacher, 速度快
      - 学习 Teacher 的软标签 + 真实标签

    核心公式:
      L = α * L_hard + (1-α) * L_soft

      L_hard = CE(student_logits, 真实标签)     # 硬标签损失
      L_soft = KL(softmax(s/T) || softmax(t/T)) # 软标签损失

      T = 温度参数 (Temperature)
        T=1:  标准 softmax
        T>1:  分布更平滑, 暴露更多类间信息

    示例 (T=5):
      Teacher logits: [10, 2, 1]   (很确定)
      Soft(T=1):      [0.999, 0.000, 0.000]   → 几乎无信息
      Soft(T=5):      [0.862, 0.077, 0.062]   → 暴露类间关系!
""")

# 演示温度对 softmax 的影响
logits = torch.tensor([10.0, 2.0, 1.0])

print("  温度对 Softmax 分布的影响:")
print(f"  Logits: {logits.tolist()}")
for T in [1, 2, 5, 10, 20]:
    soft = F.softmax(logits / T, dim=-1)
    print(f"  T={T:>2d}: {soft.tolist()}")

fig, ax = plt.subplots(figsize=(10, 5))
temps = [1, 2, 5, 10, 20]
x = np.arange(3)
width = 0.15

for i, T in enumerate(temps):
    soft = F.softmax(logits / T, dim=-1).numpy()
    ax.bar(x + i * width, soft, width, label=f"T={T}")

ax.set_xlabel("类别")
ax.set_ylabel("概率")
ax.set_title("温度参数对 Softmax 分布的影响", fontsize=13)
ax.set_xticks(x + width * 2)
ax.set_xticklabels(["类别0 (logit=10)", "类别1 (logit=2)", "类别2 (logit=1)"])
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("D:/code/big-model-learn/code/q_01/W16/temperature_effect.png", dpi=150, bbox_inches="tight")
plt.close()
print("  温度效果图已保存: W16/temperature_effect.png")

# ============================================================
# 2. 准备数据
# ============================================================
print("\n--- 2. 准备数据 ---")

np.random.seed(42)
torch.manual_seed(42)

# 生成多分类数据 (5 类, 10 维特征)
n_classes = 5
n_features = 10
n_samples = 2000

# 各类中心不同
centers = np.random.randn(n_classes, n_features) * 3
X_list, y_list = [], []
for cls in range(n_classes):
    n_cls = n_samples // n_classes
    X_list.append(centers[cls] + np.random.randn(n_cls, n_features))
    y_list.append(np.full(n_cls, cls))

X = np.vstack(X_list)
y = np.concatenate(y_list)

# 打乱
idx = np.random.permutation(len(y))
X, y = X[idx], y[idx]

X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.long)

# 划分数据集
n_train = int(0.7 * len(y))
n_val = int(0.15 * len(y))
n_test = len(y) - n_train - n_val

dataset = TensorDataset(X_t, y_t)
train_set, val_set, test_set = random_split(
    dataset, [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64)
test_loader = DataLoader(test_set, batch_size=64)

print(f"  数据: {n_samples} 样本, {n_features} 特征, {n_classes} 类")
print(f"  训练: {n_train}, 验证: {n_val}, 测试: {n_test}")

# ============================================================
# 3. 定义模型
# ============================================================
print("\n--- 3. 定义模型 ---")


class TeacherModel(nn.Module):
    """大模型 (Teacher)"""
    def __init__(self, input_dim=10, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class StudentModel(nn.Module):
    """小模型 (Student)"""
    def __init__(self, input_dim=10, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        return self.net(x)


teacher = TeacherModel()
student = StudentModel()

t_params = sum(p.numel() for p in teacher.parameters())
s_params = sum(p.numel() for p in student.parameters())

print(f"  Teacher 参数量: {t_params:,}")
print(f"  Student 参数量: {s_params:,}")
print(f"  压缩比:         {t_params / s_params:.1f}x")

# ============================================================
# 4. 训练 Teacher
# ============================================================
print("\n--- 4. 训练 Teacher ---")


def evaluate(model, loader):
    """评估模型准确率"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)
    return correct / total


def train_teacher(model, train_loader, val_loader, epochs=30, lr=0.001):
    """训练 Teacher 模型"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_accs, val_accs = [], []

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_acc = evaluate(model, train_loader)
        val_acc = evaluate(model, val_loader)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

    return train_accs, val_accs


print("  训练 Teacher (30 epochs)...")
teacher_train_accs, teacher_val_accs = train_teacher(teacher, train_loader, val_loader, epochs=30)
print(f"  Teacher 最终训练准确率: {teacher_train_accs[-1]:.4f}")
print(f"  Teacher 最终验证准确率: {teacher_val_accs[-1]:.4f}")

# ============================================================
# 5. 训练 Student (无蒸馏, 基线)
# ============================================================
print("\n--- 5. 训练 Student (无蒸馏, 基线) ---")

student_baseline = StudentModel()

print("  训练 Student 基线 (30 epochs)...")
baseline_train_accs, baseline_val_accs = train_teacher(
    student_baseline, train_loader, val_loader, epochs=30, lr=0.001
)
print(f"  Student 基线训练准确率: {baseline_train_accs[-1]:.4f}")
print(f"  Student 基线验证准确率: {baseline_val_accs[-1]:.4f}")

# ============================================================
# 6. 知识蒸馏训练 Student
# ============================================================
print("\n--- 6. 知识蒸馏训练 Student ---")


def distillation_loss(student_logits, teacher_logits, labels, temperature=5.0, alpha=0.7):
    """
    蒸馏损失 = α * hard_loss + (1-α) * soft_loss

    hard_loss: 学生预测 vs 真实标签 (标准交叉熵)
    soft_loss: 学生软预测 vs 教师软预测 (KL散度)
    """
    # Hard loss: 标准交叉熵
    hard_loss = F.cross_entropy(student_logits, labels)

    # Soft loss: KL散度
    soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    soft_loss = F.kl_div(soft_student, soft_targets, reduction='batchmean') * (temperature ** 2)

    # 组合
    total_loss = alpha * hard_loss + (1 - alpha) * soft_loss
    return total_loss


def train_student_distillation(student, teacher, train_loader, val_loader,
                                temperature=5.0, alpha=0.7, epochs=30, lr=0.001):
    """用知识蒸馏训练 Student"""
    teacher.eval()  # Teacher 固定

    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        student.train()
        for X_batch, y_batch in train_loader:
            student_logits = student(X_batch)

            with torch.no_grad():
                teacher_logits = teacher(X_batch)

            loss = distillation_loss(
                student_logits, teacher_logits, y_batch,
                temperature=temperature, alpha=alpha
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_acc = evaluate(student, train_loader)
        val_acc = evaluate(student, val_loader)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

    return train_accs, val_accs


student_distilled = StudentModel()

print("  训练蒸馏 Student (30 epochs, T=5, α=0.7)...")
distill_train_accs, distill_val_accs = train_student_distillation(
    student_distilled, teacher, train_loader, val_loader,
    temperature=5.0, alpha=0.7, epochs=30
)
print(f"  蒸馏 Student 训练准确率: {distill_train_accs[-1]:.4f}")
print(f"  蒸馏 Student 验证准确率: {distill_val_accs[-1]:.4f}")

# ============================================================
# 7. 对比三种模型
# ============================================================
print("\n--- 7. 三种模型对比 ---")

test_acc_teacher = evaluate(teacher, test_loader)
test_acc_baseline = evaluate(student_baseline, test_loader)
test_acc_distilled = evaluate(student_distilled, test_loader)

print(f"""
  ┌─────────────────────────────────────────────────────┐
  │  模型对比                                           │
  ├──────────────────────┬──────────┬──────────┬────────┤
  │  模型                │ 参数量   │ 测试准确率│ 压缩比 │
  ├──────────────────────┼──────────┼──────────┼────────┤
  │  Teacher (大模型)     │ {t_params:>7,d} │ {test_acc_teacher:>7.4f}  │  1.0x  │
  │  Student (无蒸馏)     │ {s_params:>7,d} │ {test_acc_baseline:>7.4f}  │ {t_params/s_params:>4.1f}x  │
  │  Student (蒸馏)       │ {s_params:>7,d} │ {test_acc_distilled:>7.4f}  │ {t_params/s_params:>4.1f}x  │
  └──────────────────────┴──────────┴──────────┴────────┘

  蒸馏提升: {(test_acc_distilled - test_acc_baseline) * 100:+.2f}%
  蒸馏 vs Teacher 差距: {(test_acc_teacher - test_acc_distilled) * 100:.2f}%
""")

# ============================================================
# 8. 可视化训练过程
# ============================================================
print("\n--- 8. 可视化训练过程 ---")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

epochs_range = range(1, 31)

ax1.plot(epochs_range, teacher_val_accs, 'b-', linewidth=2, label="Teacher (大模型)")
ax1.plot(epochs_range, baseline_val_accs, 'r--', linewidth=2, label="Student 无蒸馏")
ax1.plot(epochs_range, distill_val_accs, 'g-', linewidth=2, label="Student 蒸馏")
ax1.set_title("验证集准确率对比", fontsize=13)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("准确率")
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# 最终准确率柱状图
models = ["Teacher", "Student\n(无蒸馏)", "Student\n(蒸馏)"]
accs = [test_acc_teacher, test_acc_baseline, test_acc_distilled]
colors = ['#3498db', '#e74c3c', '#2ecc71']

bars = ax2.bar(models, accs, color=colors, width=0.5)
for bar, acc in zip(bars, accs):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f'{acc:.4f}', ha='center', fontsize=12, fontweight='bold')

ax2.set_title("测试集准确率对比", fontsize=13)
ax2.set_ylabel("准确率")
ax2.set_ylim(0, 1.1)
ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("D:/code/big-model-learn/code/q_01/W16/distillation_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  对比图已保存: W16/distillation_comparison.png")

# ============================================================
# 9. 不同温度和 alpha 的影响
# ============================================================
print("\n--- 9. 温度参数影响实验 ---")

temp_results = {}
for T in [1, 2, 3, 5, 10, 20]:
    student_t = StudentModel()
    _, val_accs = train_student_distillation(
        student_t, teacher, train_loader, val_loader,
        temperature=T, alpha=0.7, epochs=20, lr=0.001
    )
    temp_results[T] = val_accs[-1]
    print(f"  T={T:>2d}: 验证准确率 = {val_accs[-1]:.4f}")

best_T = max(temp_results, key=temp_results.get)
print(f"\n  最佳温度: T={best_T}, 准确率={temp_results[best_T]:.4f}")

# Alpha 影响实验
print("\n  Alpha 参数影响实验 (T=5):")
alpha_results = {}
for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
    student_a = StudentModel()
    _, val_accs = train_student_distillation(
        student_a, teacher, train_loader, val_loader,
        temperature=5.0, alpha=alpha, epochs=20, lr=0.001
    )
    alpha_results[alpha] = val_accs[-1]
    alpha_desc = "纯软标签" if alpha == 0 else ("纯硬标签" if alpha == 1 else "混合")
    print(f"  α={alpha:.1f} ({alpha_desc:>6s}): 验证准确率 = {val_accs[-1]:.4f}")

# ============================================================
# 10. 总结
# ============================================================
print("\n--- 10. 总结 ---")
print("""
  本节学习了:
  1) 知识蒸馏原理: Teacher 软标签 → Student 学习
  2) 温度参数 T: 越大分布越平滑, 暴露更多类间信息
  3) 蒸馏损失: α * hard_loss + (1-α) * soft_loss
  4) 实验对比: 蒸馏 Student 通常优于无蒸馏 Student
  5) T 和 α 的选择影响蒸馏效果

  最佳实践:
    - 温度 T: 通常 3~10 (太小无效果, 太大信息丢失)
    - α: 通常 0.5~0.9 (保留一定硬标签)
    - Teacher 越强, 蒸馏效果越好
    - Student 结构需要有一定的容量

  应用场景:
    - 模型压缩: 大模型 → 小模型 (部署到边缘设备)
    - 跨架构蒸馏: CNN → MLP, Transformer → CNN
    - 集成蒸馏: 多个 Teacher → 单个 Student

  下一步: d6_hyperparameter_optimization.py - 超参数优化
""")
