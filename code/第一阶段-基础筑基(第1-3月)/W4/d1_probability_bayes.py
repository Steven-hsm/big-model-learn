"""
W04-D1 概率基础与贝叶斯
========================
内容：
1. 模拟掷骰子验证概率
2. 条件概率示例
3. 贝叶斯定理：医学检测示例
4. 朴素贝叶斯垃圾邮件分类器
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 模拟掷骰子验证概率
# ============================================================
print("=" * 60)
print("1. 模拟掷骰子验证概率")
print("=" * 60)

np.random.seed(42)

# 模拟掷骰子 100000 次
n_rolls = 100000
rolls = np.random.randint(1, 7, size=n_rolls)

# 统计每个点数出现的频率
for face in range(1, 7):
    count = np.sum(rolls == face)
    freq = count / n_rolls
    print(f"  点数 {face}: 出现次数={count}, 频率={freq:.4f}, 理论概率={1/6:.4f}")

# 可视化频率 vs 理论概率
fig, ax = plt.subplots(figsize=(8, 5))
faces = np.arange(1, 7)
frequencies = [np.sum(rolls == f) / n_rolls for f in faces]
ax.bar(faces - 0.15, frequencies, width=0.3, label='模拟频率', color='steelblue', alpha=0.8)
ax.bar(faces + 0.15, [1/6]*6, width=0.3, label='理论概率', color='coral', alpha=0.8)
ax.set_xlabel('骰子点数')
ax.set_ylabel('概率/频率')
ax.set_title(f'掷骰子模拟 (n={n_rolls})')
ax.set_xticks(faces)
ax.legend()
ax.set_ylim(0, 0.22)
plt.tight_layout()
plt.savefig('d1_dice_simulation.png', dpi=150)
plt.close()
print("  [图] 骰子模拟图已保存为 d1_dice_simulation.png")


# ============================================================
# 2. 条件概率示例
# ============================================================
print("\n" + "=" * 60)
print("2. 条件概率示例")
print("=" * 60)

# 示例：两枚骰子的样本空间
# 事件A：第一枚骰子为偶数
# 事件B：两枚骰子之和 >= 8
# 求 P(B|A) = P(A∩B) / P(A)

print("\n  两枚骰子实验：")
print("  事件A：第一枚骰子为偶数")
print("  事件B：两枚骰子之和 >= 8")

# 枚举所有36种结果
total_outcomes = 36  # 6 x 6
event_A = []  # 第一枚为偶数
event_B = []  # 之和 >= 8
event_AB = []  # A且B

for d1 in range(1, 7):
    for d2 in range(1, 7):
        if d1 % 2 == 0:
            event_A.append((d1, d2))
        if d1 + d2 >= 8:
            event_B.append((d1, d2))
        if d1 % 2 == 0 and d1 + d2 >= 8:
            event_AB.append((d1, d2))

p_A = len(event_A) / total_outcomes
p_B = len(event_B) / total_outcomes
p_AB = len(event_AB) / total_outcomes
p_B_given_A = p_AB / p_A

print(f"  P(A) = {len(event_A)}/{total_outcomes} = {p_A:.4f}")
print(f"  P(B) = {len(event_B)}/{total_outcomes} = {p_B:.4f}")
print(f"  P(A∩B) = {len(event_AB)}/{total_outcomes} = {p_AB:.4f}")
print(f"  P(B|A) = P(A∩B)/P(A) = {p_AB:.4f}/{p_A:.4f} = {p_B_given_A:.4f}")

# 用模拟验证
n_sim = 200000
d1_sim = np.random.randint(1, 7, size=n_sim)
d2_sim = np.random.randint(1, 7, size=n_sim)
a_mask = d1_sim % 2 == 0
b_mask = d1_sim + d2_sim >= 8
p_B_given_A_sim = np.sum(a_mask & b_mask) / np.sum(a_mask)
print(f"  模拟验证 P(B|A) = {p_B_given_A_sim:.4f}")


# ============================================================
# 3. 贝叶斯定理：医学检测示例
# ============================================================
print("\n" + "=" * 60)
print("3. 贝叶斯定理：医学检测示例")
print("=" * 60)

# 已知信息
prevalence = 0.01      # 患病率 P(D) = 1%
sensitivity = 0.99     # 灵敏度 P(+|D) = 99%（真阳性率）
false_positive = 0.05  # 假阳性率 P(+|~D) = 5%

# 计算
p_D = prevalence            # P(患病)
p_notD = 1 - p_D            # P(未患病)
p_pos_given_D = sensitivity # P(阳性|患病)
p_pos_given_notD = false_positive  # P(阳性|未患病)

# 全概率公式 P(阳性)
p_pos = p_pos_given_D * p_D + p_pos_given_notD * p_notD

# 贝叶斯定理：P(患病|阳性)
p_D_given_pos = (p_pos_given_D * p_D) / p_pos

print(f"\n  已知条件：")
print(f"    患病率 P(D)       = {p_D:.2%}")
print(f"    灵敏度 P(+|D)     = {p_pos_given_D:.2%}")
print(f"    假阳性率 P(+|~D)  = {p_pos_given_notD:.2%}")
print(f"\n  计算过程：")
print(f"    P(+) = P(+|D)·P(D) + P(+|~D)·P(~D)")
print(f"         = {p_pos_given_D:.2f}×{p_D:.2f} + {p_pos_given_notD:.2f}×{p_notD:.2f}")
print(f"         = {p_pos:.4f}")
print(f"\n    P(D|+) = P(+|D)·P(D) / P(+)")
print(f"           = {p_pos_given_D:.2f}×{p_D:.2f} / {p_pos:.4f}")
print(f"           = {p_D_given_pos:.4f} = {p_D_given_pos:.2%}")
print(f"\n  结论：即使检测为阳性，真正患病的概率只有 {p_D_given_pos:.2%}！")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左图：100000人中的检测结果分布
n_people = 100000
n_sick = int(n_people * p_D)
n_healthy = n_people - n_sick
tp = int(n_sick * sensitivity)      # 真阳性
fn = n_sick - tp                     # 假阴性
fp = int(n_healthy * false_positive) # 假阳性
tn = n_healthy - fp                  # 真阴性

categories = ['真阳性\n(有病且阳性)', '假阴性\n(有病但阴性)', '假阳性\n(无病但阳性)', '真阴性\n(无病且阴性)']
values = [tp, fn, fp, tn]
colors = ['#e74c3c', '#f39c12', '#e67e22', '#2ecc71']
axes[0].bar(categories, values, color=colors, alpha=0.85)
axes[0].set_ylabel('人数')
axes[0].set_title(f'医学检测结果分布 (n={n_people})')
for i, v in enumerate(values):
    axes[0].text(i, v + 200, str(v), ha='center', fontweight='bold')

# 右图：贝叶斯定理各概率
labels = ['P(D)', 'P(+|D)', 'P(+|~D)', 'P(D|+)']
probs = [p_D, p_pos_given_D, p_pos_given_notD, p_D_given_pos]
bar_colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
bars = axes[1].bar(labels, probs, color=bar_colors, alpha=0.85)
axes[1].set_ylabel('概率')
axes[1].set_title('贝叶斯定理相关概率')
for bar, prob in zip(bars, probs):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{prob:.4f}', ha='center', fontweight='bold')
axes[1].set_ylim(0, 1.15)

plt.tight_layout()
plt.savefig('d1_bayes_medical.png', dpi=150)
plt.close()
print("\n  [图] 贝叶斯医学检测图已保存为 d1_bayes_medical.png")


# ============================================================
# 4. 朴素贝叶斯垃圾邮件分类器
# ============================================================
print("\n" + "=" * 60)
print("4. 朴素贝叶斯垃圾邮件分类器")
print("=" * 60)


class NaiveBayesSpamFilter:
    """
    朴素贝叶斯垃圾邮件分类器
    使用拉普拉斯平滑和对数概率避免下溢
    """

    def __init__(self, alpha=1.0):
        """
        初始化分类器

        Parameters:
            alpha: 拉普拉斯平滑参数，默认为1
        """
        self.alpha = alpha  # 拉普拉斯平滑参数
        self.class_log_prior = {}   # 类别的对数先验概率
        self.feature_log_prob = {}  # 给定类别下各单词的对数条件概率
        self.vocab = set()          # 词汇表
        self.classes = []           # 类别列表

    def _tokenize(self, text):
        """简单分词：转小写并按空格切分"""
        return text.lower().split()

    def train(self, emails, labels):
        """
        训练分类器

        Parameters:
            emails: 邮件文本列表
            labels: 对应标签列表 ('spam' 或 'ham')
        """
        self.classes = list(set(labels))

        # 统计每个类别的文档数
        class_counts = defaultdict(int)
        for label in labels:
            class_counts[label] += 1

        # 计算类别的对数先验概率 log P(c)
        total_docs = len(labels)
        for c in self.classes:
            self.class_log_prior[c] = np.log(class_counts[c] / total_docs)

        # 构建词汇表和统计每个类别中各单词的出现次数
        word_counts = {c: defaultdict(int) for c in self.classes}
        total_words_in_class = {c: 0 for c in self.classes}

        for email, label in zip(emails, labels):
            words = self._tokenize(email)
            for word in words:
                self.vocab.add(word)
                word_counts[label][word] += 1
                total_words_in_class[label] += 1

        # 计算对数条件概率 log P(w|c)，使用拉普拉斯平滑
        vocab_size = len(self.vocab)
        self.feature_log_prob = {c: {} for c in self.classes}

        for c in self.classes:
            for word in self.vocab:
                # P(w|c) = (count(w,c) + alpha) / (total_words_in_c + alpha * |V|)
                prob = (word_counts[c][word] + self.alpha) / \
                       (total_words_in_class[c] + self.alpha * vocab_size)
                self.feature_log_prob[c][word] = np.log(prob)

        print(f"  训练完成：词汇表大小={vocab_size}, 类别数={len(self.classes)}")

    def predict(self, email):
        """
        预测邮件类别

        Parameters:
            email: 邮件文本

        Returns:
            预测的类别标签
        """
        words = self._tokenize(email)
        best_class = None
        best_log_prob = -np.inf

        for c in self.classes:
            # log P(c|email) ∝ log P(c) + Σ log P(w|c)
            log_prob = self.class_log_prior[c]
            for word in words:
                if word in self.vocab:
                    log_prob += self.feature_log_prob[c].get(word, 0.0)
                else:
                    # 未见过的词，使用平滑概率
                    log_prob += np.log(
                        self.alpha / (self.alpha * len(self.vocab))
                    )

            if log_prob > best_log_prob:
                best_log_prob = log_prob
                best_class = c

        return best_class

    def predict_proba(self, email):
        """返回各类别的对数概率（用于分析）"""
        words = self._tokenize(email)
        log_probs = {}

        for c in self.classes:
            log_prob = self.class_log_prior[c]
            for word in words:
                if word in self.vocab:
                    log_prob += self.feature_log_prob[c].get(word, 0.0)
            log_probs[c] = log_prob

        return log_probs


# ---- 训练数据 ----
train_emails = [
    "免费 赢得 大奖 点击 链接 领取",
    "优惠 促销 折扣 限时 免费 购买",
    "赚钱 投资 高回报 日入 万元",
    "恭喜 中奖 免费领取 点击 确认",
    "贷款 低利息 无抵押 快速 到账",
    "明天 开会 讨论 项目 进度",
    "请 查看 附件 中的 报告 文件",
    "周末 一起 吃饭 聚会 联系",
    "项目 进度 更新 请 查看",
    "会议 时间 改为 下午 三点",
    "请 完成 本周 工作 总结",
    "转发 会议 纪要 给 团队",
    "免费 体验 VIP 会员 点击 注册",
    "紧急 通知 账户 异常 点击 验证",
    "附件 是 合同 文件 请 签字",
    "下午 讨论 技术 方案 会议室",
]

train_labels = [
    "spam", "spam", "spam", "spam", "spam",
    "ham", "ham", "ham", "ham", "ham",
    "ham", "ham", "spam", "spam",
    "ham", "ham",
]

# ---- 训练分类器 ----
print("\n  训练朴素贝叶斯分类器...")
filter = NaiveBayesSpamFilter(alpha=1.0)
filter.train(train_emails, train_labels)

# ---- 测试 ----
test_emails = [
    "免费 领取 VIP 优惠 大奖",
    "请 查看 项目 报告 并 回复",
    "赚钱 好 项目 投资 回报 高",
    "明天 上午 开会 讨论 方案",
    "恭喜 赢得 大奖 点击 领取 免费",
    "附件 为 合同 请 签字 确认",
    "限时 促销 免费 体验 点击 注册",
    "本周 工作 总结 请 完成",
]

test_true_labels = ["spam", "ham", "spam", "ham", "spam", "ham", "spam", "ham"]

print("\n  测试结果：")
print(f"  {'邮件内容':<30s} {'预测':<8s} {'实际':<8s} {'正确'}")
print("  " + "-" * 65)

correct = 0
for email, true_label in zip(test_emails, test_true_labels):
    pred = filter.predict(email)
    log_probs = filter.predict_proba(email)
    is_correct = "V" if pred == true_label else "X"
    if pred == true_label:
        correct += 1
    print(f"  {email:<30s} {pred:<8s} {true_label:<8s} {is_correct}  "
          f"(log_p: spam={log_probs['spam']:.2f}, ham={log_probs['ham']:.2f})")

accuracy = correct / len(test_emails)
print(f"\n  准确率: {correct}/{len(test_emails)} = {accuracy:.2%}")

print("\n" + "=" * 60)
print("D1 完成！")
print("=" * 60)
