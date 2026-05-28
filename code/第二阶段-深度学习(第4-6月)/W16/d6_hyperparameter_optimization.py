"""
W16-D6 超参数优化 (Hyperparameter Optimization)
=================================================
Optuna 基础: 定义目标函数, suggest 超参数,
创建 study, 优化, 获取最佳参数。
示例: 优化学习率, batch_size, hidden_size
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
print("W16-D6 超参数优化 (Hyperparameter Optimization)")
print("=" * 60)

# ============================================================
# 1. 超参数优化概述
# ============================================================
print("\n--- 1. 超参数优化概述 ---")
print("""
  超参数 vs 参数:
    参数 (Parameters):   模型通过训练学习的权重
    超参数 (Hyperparameters): 训练前设定的配置

  常见超参数:
    - 学习率 (learning rate)
    - Batch size
    - 隐藏层维度 (hidden size)
    - Dropout 率
    - 正则化系数
    - 优化器类型 (Adam, SGD, ...)
    - 训练轮数 (epochs)

  搜索策略:
    1) Grid Search (网格搜索): 穷举所有组合
    2) Random Search (随机搜索): 随机采样
    3) Bayesian Optimization (贝叶斯优化): 智能搜索
    4) Optuna: 自动化超参数优化框架
""")

# ============================================================
# 2. 手动实现: Grid Search 和 Random Search
# ============================================================
print("\n--- 2. 手动搜索 ---")

# 准备数据
np.random.seed(42)
torch.manual_seed(42)

n_samples = 1000
X = np.random.randn(n_samples, 10)
y = (X[:, 0] * 2 + X[:, 1] * 0.5 + np.random.randn(n_samples) * 0.5 > 0).astype(int)

X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X_t, y_t)
train_set, val_set = random_split(dataset, [800, 200],
                                   generator=torch.Generator().manual_seed(42))


def train_evaluate(lr, hidden_size, dropout, epochs=20):
    """训练并评估模型, 返回验证准确率"""
    model = nn.Sequential(
        nn.Linear(10, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, 2)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 验证
    model.eval()
    val_loader = DataLoader(val_set, batch_size=200)
    for X_val, y_val in val_loader:
        with torch.no_grad():
            preds = torch.argmax(model(X_val), dim=-1)
        acc = (preds == y_val).float().mean().item()

    return acc


# --- Grid Search ---
print("\n  Grid Search:")
lr_grid = [0.001, 0.01, 0.1]
hidden_grid = [16, 32, 64]
dropout_grid = [0.1, 0.3]

best_acc = 0
best_params = {}
grid_results = []

for lr in lr_grid:
    for hs in hidden_grid:
        for dp in dropout_grid:
            acc = train_evaluate(lr, hs, dp, epochs=15)
            grid_results.append({"lr": lr, "hidden": hs, "dropout": dp, "acc": acc})
            if acc > best_acc:
                best_acc = acc
                best_params = {"lr": lr, "hidden": hs, "dropout": dp}

print(f"    搜索空间: {len(lr_grid)} x {len(hidden_grid)} x {len(dropout_grid)} = {len(grid_results)} 组合")
print(f"    最佳参数: {best_params}")
print(f"    最佳准确率: {best_acc:.4f}")

# --- Random Search ---
print("\n  Random Search:")
n_trials = 10
random_results = []
best_acc_rs = 0
best_params_rs = {}

np.random.seed(123)
for i in range(n_trials):
    lr = 10 ** np.random.uniform(-4, -1)  # 对数均匀采样
    hidden = np.random.choice([16, 32, 64, 128])
    dropout = np.random.uniform(0.1, 0.5)

    acc = train_evaluate(lr, int(hidden), dropout, epochs=15)
    random_results.append({"lr": lr, "hidden": int(hidden), "dropout": dropout, "acc": acc})

    if acc > best_acc_rs:
        best_acc_rs = acc
        best_params_rs = {"lr": lr, "hidden": int(hidden), "dropout": dropout}

print(f"    搜索次数: {n_trials}")
print(f"    最佳参数: lr={best_params_rs['lr']:.5f}, hidden={best_params_rs['hidden']}, dropout={best_params_rs['dropout']:.3f}")
print(f"    最佳准确率: {best_acc_rs:.4f}")

# ============================================================
# 3. Optuna 基础
# ============================================================
print("\n--- 3. Optuna 超参数优化 ---")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
    print("  Optuna 已安装")
except ImportError:
    HAS_OPTUNA = False
    print("  [!] Optuna 未安装, 运行: pip install optuna")
    print("  将展示代码示例而非实际运行")


if HAS_OPTUNA:
    # 定义目标函数
    def objective(trial):
        """Optuna 目标函数: 定义搜索空间并返回评估指标"""
        # suggest 超参数
        lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64, 128])
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])

        # 构建模型
        model = nn.Sequential(
            nn.Linear(10, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2)
        )

        # 选择优化器
        if optimizer_name == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        criterion = nn.CrossEntropyLoss()
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

        # 训练
        model.train()
        for epoch in range(15):
            for X_batch, y_batch in train_loader:
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 评估
        model.eval()
        val_loader = DataLoader(val_set, batch_size=200)
        for X_val, y_val in val_loader:
            with torch.no_grad():
                preds = torch.argmax(model(X_val), dim=-1)
            acc = (preds == y_val).float().mean().item()

        return acc  # 最大化准确率

    # 创建 study 并优化
    print("\n  创建 Optuna Study (direction='maximize')...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20, show_progress_bar=False)

    # 结果
    print(f"\n  Optuna 优化结果 ({len(study.trials)} trials):")
    print(f"    最佳���确率: {study.best_value:.4f}")
    print(f"    最佳参数:")
    for key, val in study.best_params.items():
        if isinstance(val, float):
            print(f"      {key}: {val:.6f}")
        else:
            print(f"      {key}: {val}")

    # 所有 trial 结果
    print(f"\n  所有 Trial 结果:")
    print(f"  {'Trial':>6s}  {'准确率':>8s}  {'lr':>10s}  {'hidden':>8s}  {'dropout':>8s}  {'batch':>6s}  {'optimizer':>8s}")
    for t in study.trials:
        params = t.params
        lr_str = f"{params['lr']:.6f}"
        print(f"  {t.number:>6d}  {t.value:>8.4f}  {lr_str:>10s}  {params['hidden_size']:>8d}  {params['dropout']:>8.4f}  {params['batch_size']:>6d}  {params['optimizer']:>8s}")

    # ============================================================
    # 4. Optuna 可视化
    # ============================================================
    print("\n--- 4. Optuna 分析 ---")

    # 参数重要性
    try:
        importance = optuna.importance.get_param_importances(study)
        print("  参数重要性:")
        for param, imp in importance.items():
            bar = "█" * int(imp * 40)
            print(f"    {param:15s}: {imp:.4f} {bar}")
    except Exception as e:
        print(f"  参数重要性计算失败: {e}")

    # 手动绘制优化历史
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    trial_numbers = [t.number for t in study.trials]
    trial_values = [t.value for t in study.trials]

    ax1.plot(trial_numbers, trial_values, 'o-', markersize=4)
    best_so_far = np.maximum.accumulate(trial_values)
    ax1.plot(trial_numbers, best_so_far, 'r-', linewidth=2, label="最佳累计")
    ax1.set_title("Optuna 优化历史", fontsize=13)
    ax1.set_xlabel("Trial")
    ax1.set_ylabel("验证准确率")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 学习率 vs 准确率
    lrs = [t.params['lr'] for t in study.trials]
    ax2.scatter(lrs, trial_values, c=trial_numbers, cmap='viridis', s=50)
    ax2.set_xscale('log')
    ax2.set_title("学习率 vs 准确率", fontsize=13)
    ax2.set_xlabel("Learning Rate (log)")
    ax2.set_ylabel("验证准确率")
    ax2.grid(True, alpha=0.3)
    plt.colorbar(ax2.collections[0], ax=ax2, label="Trial")

    plt.tight_layout()
    plt.savefig("D:/code/big-model-learn/code/q_01/W16/optuna_optimization.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  优化图已保存: W16/optuna_optimization.png")

else:
    # Optuna 未安装时的代码示例
    print("""
  Optuna 使用示例代码:

  import optuna

  def objective(trial):
      # 1. Suggest 超参数
      lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
      hidden = trial.suggest_categorical("hidden", [16, 32, 64, 128])
      dropout = trial.suggest_float("dropout", 0.1, 0.5)
      batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

      # 2. 构建模型
      model = build_model(lr, hidden, dropout, batch_size)

      # 3. 训练和评估
      accuracy = train_and_evaluate(model)

      return accuracy  # maximize

  # 3. 创建 study 并优化
  study = optuna.create_study(direction="maximize")
  study.optimize(objective, n_trials=50)

  # 4. 获取最佳参数
  print(study.best_params)
  print(study.best_value)

  # 高级功能:
  #   - study = optuna.create_study(sampler=optuna.samplers.TPESampler())
  #   - optuna.integration.PyTorchLightningPruningCallback
  #   - study.optimize(objective, n_trials=100, timeout=3600)
  #   - optuna.visualization.plot_optimization_history(study)
  """)

# ============================================================
# 5. 搜索策略对比
# ============================================================
print("\n--- 5. 搜索策略对比 ---")
print("""
  策略              优点                    缺点
  ──────────────────────────────────────────────────────────
  Grid Search       全面, 可复现             慢, 维度灾难
  Random Search     简单, 比Grid高效         可能错过最优
  Bayesian (Optuna) 智能, 高效, 可并行       需要额外依赖

  Optuna 的优势:
    - TPE 采样器 (Tree-structured Parzen Estimator)
    - 自动剪枝 (Early Stopping)
    - 并行优化
    - 丰富的可视化
    - 易于集成 (PyTorch, TensorFlow, ...)

  实践建议:
    1) 先用 Random Search 快速探索
    2) 再用 Optuna (TPE) 精细搜索
    3) 重要参数优先: lr > hidden_size > dropout > batch_size
    4) 使用 log scale 搜索学习率
""")

# ============================================================
# 6. 总结
# ============================================================
print("\n--- 6. 总结 ---")
print("""
  本节学习了:
  1) 超参数优化的意义和常见策略
  2) Grid Search 和 Random Search 手动实现
  3) Optuna 框架: objective, suggest, study, optimize
  4) 参数重要性分析
  5) 搜索策略对比

  关键要点:
    - 学习率通常是最重要的超参数
    - Optuna TPE 比 Grid/Random 更高效
    - 使用 log scale 搜索学习率
    - 设置合理的搜索空间很重要

  下一步: d7_debug_training.py - 训练调试技巧
""")
