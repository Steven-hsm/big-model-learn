"""
W25-D4 数据标注策略
====================
数据标注策略, 主动学习(Active Learning), 弱监督学习, 标注质量控制, 伪标签
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

try:
    from sklearn.linear_model import LogisticRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W25-D4 数据标注策略")
print("=" * 60)

# ============================================================
# 1. 数据标注概述
# ============================================================
print("\n--- 1. 数据标注概述 ---")
print("""
数据标注是监督学习的基石:

  标注方式:
    - 人工标注:   专家标注, 质量高但成本高
    - 半自动标注: 模型预标注 + 人工校正
    - 弱监督:     用规则/启发式生成标签(有噪声)
    - 主动学习:   模型选择最有价值的样本标注
    - 伪标签:     用高置信度预测作为标签
    - 众包标注:   Amazon Mechanical Turk等平台

  标注策略选择:
  ┌─────────────┬──────────┬──────────┬──────────┐
  │ 方式        │ 质量     │ 成本     │ 速度     │
  ├─────────────┼──────────┼──────────┼──────────┤
  │ 专家标注    │ 很高     │ 很高     │ 慢       │
  │ 众包标注    │ 中等     │ 中等     │ 中等     │
  │ 弱监督      │ 较低     │ 低       │ 快       │
  │ 主动学习    │ 高       │ 中等     │ 中等     │
  │ 伪标签      │ 较低     │ 很低     │ 很快     │
  └─────────────┴──────────┴──────────┴──────────┘
""")

# ============================================================
# 2. 主动学习 (Active Learning)
# ============================================================
print("\n--- 2. 主动学习 (Active Learning) ---")

# 生成模拟数据
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                            n_redundant=2, random_state=42, class_sep=0.8)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


class ActiveLearner:
    """主动学习器"""

    def __init__(self, base_model, strategy='uncertainty'):
        self.base_model = base_model
        self.strategy = strategy
        self.labeled_mask = None
        self.performance_history = []

    def initialize(self, X, y, n_initial=20):
        """初始化: 随机选择少量样本标注"""
        self.X = X
        self.y = y
        self.labeled_mask = np.zeros(len(X), dtype=bool)
        initial_idx = np.random.choice(len(X), n_initial, replace=False)
        self.labeled_mask[initial_idx] = True
        print(f"  初始化: 随机标注 {n_initial} 个样本")

    def _select_samples(self, n_select=10):
        """选择最有价值的样本"""
        unlabeled_idx = np.where(~self.labeled_mask)[0]
        if len(unlabeled_idx) == 0:
            return np.array([])

        X_unlabeled = self.X[unlabeled_idx]

        if self.strategy == 'uncertainty':
            # 不确定性采样: 选择预测概率最接近0.5的样本
            probs = self.base_model.predict_proba(X_unlabeled)
            uncertainty = 1 - np.max(probs, axis=1)  # 最大概率越小越不确定
            selected_local = np.argsort(uncertainty)[-n_select:]

        elif self.strategy == 'margin':
            # 边缘采样: 选择两个最可能类别概率差最小的
            probs = self.base_model.predict_proba(X_unlabeled)
            sorted_probs = np.sort(probs, axis=1)
            margin = sorted_probs[:, -1] - sorted_probs[:, -2]
            selected_local = np.argsort(margin)[:n_select]

        elif self.strategy == 'entropy':
            # 熵采样: 选择熵最大的样本
            probs = self.base_model.predict_proba(X_unlabeled)
            entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
            selected_local = np.argsort(entropy)[-n_select:]

        return unlabeled_idx[selected_local]

    def teach(self, X_test_eval, y_test_eval, n_select=10, n_rounds=10):
        """主动学习训练循环"""
        # 初始训练
        labeled_idx = np.where(self.labeled_mask)[0]
        self.base_model.fit(self.X[labeled_idx], self.y[labeled_idx])
        acc = accuracy_score(y_test_eval, self.base_model.predict(X_test_eval))
        n_labeled = self.labeled_mask.sum()
        self.performance_history.append((n_labeled, acc))
        print(f"  Round  0: 已标注={n_labeled:4d}, 准确率={acc:.4f}")

        for round_i in range(1, n_rounds + 1):
            # 选择样本
            selected = self._select_samples(n_select)
            if len(selected) == 0:
                break

            # "标注"选中的样本(模拟: 使用真实标签)
            self.labeled_mask[selected] = True

            # 重新训练
            labeled_idx = np.where(self.labeled_mask)[0]
            self.base_model.fit(self.X[labeled_idx], self.y[labeled_idx])

            # 评估
            acc = accuracy_score(y_test_eval, self.base_model.predict(X_test_eval))
            n_labeled = self.labeled_mask.sum()
            self.performance_history.append((n_labeled, acc))
            print(f"  Round {round_i:2d}: 已标注={n_labeled:4d}, 准确率={acc:.4f}")

        return self.performance_history


# 运行主动学习
print("\n  === 不确定性采样 ===")
model_al = RandomForestClassifier(n_estimators=50, random_state=42)
al = ActiveLearner(model_al, strategy='uncertainty')
al.initialize(X_train, y_train, n_initial=20)
history_uncertainty = al.teach(X_test, y_test, n_select=20, n_rounds=15)

print("\n  === 随机采样(对照组) ===")
np.random.seed(42)
labeled_mask_random = np.zeros(len(X_train), dtype=bool)
random_history = []
initial_idx = np.random.choice(len(X_train), 20, replace=False)
labeled_mask_random[initial_idx] = True

for round_i in range(16):
    labeled_idx = np.where(labeled_mask_random)[0]
    model_random = RandomForestClassifier(n_estimators=50, random_state=42)
    model_random.fit(X_train[labeled_idx], y_train[labeled_idx])
    acc = accuracy_score(y_test, model_random.predict(X_test))
    random_history.append((len(labeled_idx), acc))

    # 随机选择新样本
    unlabeled = np.where(~labeled_mask_random)[0]
    if len(unlabeled) > 20:
        new_idx = np.random.choice(unlabeled, 20, replace=False)
        labeled_mask_random[new_idx] = True

# ============================================================
# 3. 弱监督学习
# ============================================================
print("\n--- 3. 弱监督学习 ---")


class WeakSupervisor:
    """弱监督标注器"""

    def __init__(self):
        self.labeling_functions = []

    def add_labeling_function(self, name, func, weight=1.0):
        """添加标注函数 (Labeling Function)"""
        self.labeling_functions.append({
            'name': name, 'func': func, 'weight': weight,
        })

    def label(self, X):
        """用多个弱标注函数组合生成标签"""
        n = len(X)
        votes = np.zeros((n, 2))  # 二分类: [否, 是]
        lf_stats = []

        for lf in self.labeling_functions:
            labels = lf['func'](X)
            coverage = (labels != -1).mean()  # -1表示弃权
            agree_pos = (labels == 1).sum()
            agree_neg = (labels == 0).sum()
            lf_stats.append({
                'name': lf['name'],
                'coverage': coverage,
                'positive': agree_pos,
                'negative': agree_neg,
                'abstain': (labels == -1).sum(),
            })

            for i in range(n):
                if labels[i] == 1:
                    votes[i, 1] += lf['weight']
                elif labels[i] == 0:
                    votes[i, 0] += lf['weight']

        # 加权投票
        weak_labels = np.argmax(votes, axis=1)
        confidence = np.max(votes, axis=1) / (votes.sum(axis=1) + 1e-10)

        return weak_labels, confidence, lf_stats


# 创建模拟数据(用户购买预测)
np.random.seed(42)
n_users = 500
X_weak = pd.DataFrame({
    'browse_count': np.random.poisson(10, n_users),
    'cart_count': np.random.poisson(2, n_users),
    'avg_time_on_page': np.random.exponential(5, n_users),
    'has_coupon': np.random.choice([0, 1], n_users, p=[0.7, 0.3]),
    'days_since_last': np.random.exponential(30, n_users),
    'is_vip': np.random.choice([0, 1], n_users, p=[0.8, 0.2]),
})
y_true = ((X_weak['cart_count'] > 2) | (X_weak['has_coupon'] & (X_weak['browse_count'] > 5))).astype(int)

# 定义弱标注函数
ws = WeakSupervisor()
ws.add_labeling_function('LF_cart_high', lambda X: (X['cart_count'] > 3).astype(int).replace(0, -1).values + 1)
ws.add_labeling_function('LF_browse_coupon', lambda X: np.where((X['browse_count'] > 8) & (X['has_coupon'] == 1), 1, -1))
ws.add_labeling_function('LF_vip_active', lambda X: np.where((X['is_vip'] == 1) & (X['days_since_last'] < 10), 1, -1))
ws.add_labeling_function('LF_inactive', lambda X: np.where(X['days_since_last'] > 60, 0, -1))
ws.add_labeling_function('LF_browse_low', lambda X: np.where(X['browse_count'] < 2, 0, -1))

weak_labels, confidence, lf_stats = ws.label(X_weak)

# 评估弱标签质量
agreement = (weak_labels == y_true.values).mean()
print(f"\n  弱标签与真实标签一致性: {agreement:.2%}")
print(f"  平均置信度: {confidence.mean():.3f}")

print(f"\n  {'标注函数':<20} {'覆盖率':<10} {'正例':<8} {'负例':<8} {'弃权':<8}")
print("  " + "-" * 60)
for s in lf_stats:
    print(f"  {s['name']:<20} {s['coverage']:<10.2%} {s['positive']:<8d} {s['negative']:<8d} {s['abstain']:<8d}")

# ============================================================
# 4. 标注质量控制
# ============================================================
print("\n--- 4. 标注质量控制 ---")


class LabelQualityController:
    """标注质量控制"""

    @staticmethod
    def inter_annotator_agreement(annotations_a, annotations_b):
        """标注者间一致性 (Cohen's Kappa简化版)"""
        n = len(annotations_a)
        agree = (annotations_a == annotations_b).sum()

        # 随机一致性
        p_a = (annotations_a == 1).mean()
        p_b = (annotations_b == 1).mean()
        p_random = p_a * p_b + (1 - p_a) * (1 - p_b)

        p_observed = agree / n
        kappa = (p_observed - p_random) / (1 - p_random + 1e-10)

        if kappa > 0.8:
            quality = '优秀'
        elif kappa > 0.6:
            quality = '良好'
        elif kappa > 0.4:
            quality = '中等'
        else:
            quality = '较差'

        return {
            'agreement': p_observed,
            'kappa': kappa,
            'quality': quality,
        }

    @staticmethod
    def detect_label_errors(X, y, model, threshold=0.8):
        """检测可能的标注错误"""
        model.fit(X, y)
        probs = model.predict_proba(X)
        pred = model.predict(X)
        max_prob = np.max(probs, axis=1)

        # 模型预测与标签不一致且高置信度 -> 可能是标注错误
        potential_errors = []
        for i in range(len(y)):
            if pred[i] != y[i] and max_prob[i] > threshold:
                potential_errors.append(i)

        return potential_errors

    @staticmethod
    def majority_vote(*annotations):
        """多数投票"""
        stacked = np.stack(annotations, axis=0)
        voted = np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], 0, stacked)
        return voted


# 模拟标注者
np.random.seed(42)
n_samples = 200
y_gold = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])

# 三个标注者, 准确率不同
annotator_a = y_gold.copy()
annotator_b = y_gold.copy()
annotator_c = y_gold.copy()

# 注入错误
errors_a = np.random.choice(n_samples, 10, replace=False)
errors_b = np.random.choice(n_samples, 30, replace=False)
errors_c = np.random.choice(n_samples, 50, replace=False)

annotator_a[errors_a] = 1 - annotator_a[errors_a]
annotator_b[errors_b] = 1 - annotator_b[errors_b]
annotator_c[errors_c] = 1 - annotator_c[errors_c]

# 评估一致性
qc = LabelQualityController()
for name_a, name_b, data_a, data_b in [
    ('标注者A', '标注者B', annotator_a, annotator_b),
    ('标注者A', '标注者C', annotator_a, annotator_c),
    ('标注者B', '标注者C', annotator_b, annotator_c),
]:
    result = qc.inter_annotator_agreement(data_a, data_b)
    print(f"  {name_a} vs {name_b}: Kappa={result['kappa']:.3f} ({result['quality']})")

# 多数投票
voted = qc.majority_vote(annotator_a, annotator_b, annotator_c)
voted_acc = (voted == y_gold).mean()
print(f"\n  多数投票后准确率: {voted_acc:.2%}")
print(f"  单个标注者准确率: A={1-10/200:.0%}, B={1-30/200:.0%}, C={1-50/200:.0%}")

# ============================================================
# 5. 伪标签 (Pseudo Labeling)
# ============================================================
print("\n--- 5. 伪标签 (Pseudo Labeling) ---")


class PseudoLabeler:
    """伪标签生成器"""

    def __init__(self, base_model, confidence_threshold=0.9):
        self.base_model = base_model
        self.threshold = confidence_threshold

    def generate_pseudo_labels(self, X_labeled, y_labeled, X_unlabeled, n_iter=3):
        """迭代式伪标签"""
        all_X = X_labeled.copy()
        all_y = y_labeled.copy()
        stats = []

        for i in range(n_iter):
            self.base_model.fit(all_X, all_y)
            probs = self.base_model.predict_proba(X_unlabeled)
            max_probs = np.max(probs, axis=1)
            pseudo_labels = np.argmax(probs, axis=1)

            # 选择高置信度样本
            confident_mask = max_probs >= self.threshold
            n_pseudo = confident_mask.sum()

            if n_pseudo == 0:
                print(f"  Iter {i+1}: 无高置信度样本, 停止")
                break

            # 添加伪标签数据
            pseudo_X = X_unlabeled[confident_mask]
            pseudo_y = pseudo_labels[confident_mask]

            all_X = np.vstack([all_X, pseudo_X])
            all_y = np.concatenate([all_y, pseudo_y])

            # 从未标注池中移除
            X_unlabeled = X_unlabeled[~confident_mask]

            stats.append({
                'iteration': i + 1,
                'pseudo_count': n_pseudo,
                'total_labeled': len(all_y),
                'avg_confidence': max_probs[confident_mask].mean(),
            })
            print(f"  Iter {i+1}: 新增伪标签={n_pseudo}, 总数据={len(all_y)}, "
                  f"平均置信度={max_probs[confident_mask].mean():.3f}")

        return all_X, all_y, stats


# 运行伪标签
print("\n  伪标签示例:")
X_l, X_u, y_l, y_u = train_test_split(X, y, test_size=0.5, random_state=42)
pseudo_labeler = PseudoLabeler(LogisticRegression(max_iter=1000, random_state=42),
                                confidence_threshold=0.85)
all_X, all_y, pseudo_stats = pseudo_labeler.generate_pseudo_labels(X_l, y_l, X_u, n_iter=5)

# 伪标签模型 vs 仅标注数据模型
model_pseudo = LogisticRegression(max_iter=1000, random_state=42).fit(all_X, all_y)
model_labeled_only = LogisticRegression(max_iter=1000, random_state=42).fit(X_l, y_l)
X_eval, y_eval = make_classification(n_samples=300, n_features=10, n_informative=5,
                                      n_redundant=2, random_state=99, class_sep=0.8)
acc_pseudo = accuracy_score(y_eval, model_pseudo.predict(X_eval))
acc_labeled = accuracy_score(y_eval, model_labeled_only.predict(X_eval))
print(f"\n  仅标注数据准确率: {acc_labeled:.4f}")
print(f"  +伪标签后准确率:  {acc_pseudo:.4f}")

# ============================================================
# 6. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 左上: 主动学习 vs 随机采样
ax = axes[0, 0]
al_labels, al_accs = zip(*history_uncertainty)
rand_labels, rand_accs = zip(*random_history)
ax.plot(al_labels, al_accs, 'b-o', markersize=4, label='主动学习(不确定性)')
ax.plot(rand_labels, rand_accs, 'r--s', markersize=4, label='随机采样')
ax.set_xlabel('已标注样本数')
ax.set_ylabel('测试准确率')
ax.set_title('主动学习 vs 随机采样')
ax.legend()
ax.grid(True, alpha=0.3)

# 右上: 弱标注函数覆盖率
ax = axes[0, 1]
lf_names = [s['name'] for s in lf_stats]
coverages = [s['coverage'] for s in lf_stats]
positives = [s['positive'] for s in lf_stats]
negatives = [s['negative'] for s in lf_stats]
x_pos = np.arange(len(lf_names))
width = 0.35
ax.bar(x_pos - width/2, positives, width, label='正例', color='#4CAF50')
ax.bar(x_pos + width/2, negatives, width, label='负例', color='#F44336')
ax.set_xticks(x_pos)
ax.set_xticklabels(lf_names, rotation=30, ha='right', fontsize=8)
ax.set_ylabel('标注数')
ax.set_title('弱标注函数统计')
ax.legend()

# 左下: 标注者一致性热力图
ax = axes[1, 0]
annotators = ['A', 'B', 'C']
data = [annotator_a, annotator_b, annotator_c]
kappa_matrix = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        if i == j:
            kappa_matrix[i, j] = 1.0
        else:
            kappa_matrix[i, j] = qc.inter_annotator_agreement(data[i], data[j])['kappa']

im = ax.imshow(kappa_matrix, cmap='RdYlGn', vmin=0, vmax=1)
ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(annotators)
ax.set_yticklabels(annotators)
ax.set_title('标注者间Kappa一致性')
for i in range(3):
    for j in range(3):
        ax.text(j, i, f'{kappa_matrix[i,j]:.3f}', ha='center', va='center', fontsize=12)
plt.colorbar(im, ax=ax)

# 右下: 伪标签迭代过程
ax = axes[1, 1]
if pseudo_stats:
    iters = [s['iteration'] for s in pseudo_stats]
    totals = [s['total_labeled'] for s in pseudo_stats]
    confidences = [s['avg_confidence'] for s in pseudo_stats]

    ax2 = ax.twinx()
    ax.bar(iters, totals, color='#2196F3', alpha=0.7, label='总数据量')
    ax2.plot(iters, confidences, 'r-o', label='平均置信度')
    ax.set_xlabel('迭代次数')
    ax.set_ylabel('总数据量', color='#2196F3')
    ax2.set_ylabel('平均置信度', color='red')
    ax.set_title('伪标签迭代过程')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='center right')

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W25/d4_data_labeling.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存: d4_data_labeling.png")

print("\n完成! 数据标注策略要点:")
print("  1. 主动学习: 用更少的标注达到更高的性能")
print("  2. 弱监督: 用规则/启发式生成标签, 低成本快速")
print("  3. 质量控制: 标注者一致性(Kappa), 多数投票, 错误检测")
print("  4. 伪标签: 利用未标注数据, 高置信度预测作为标签")
