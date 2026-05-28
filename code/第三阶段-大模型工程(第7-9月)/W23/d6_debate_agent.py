### Day 6（周六）：辩论式Multi-Agent
# 多角度分析, Judge Agent, 迭代优化, 答案聚合策略

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 辩论式Multi-Agent架构
# ============================================================
print("=" * 60)
print("1. 辩论式Multi-Agent架构")
print("=" * 60)

print("""
  辩论式架构:
    1. Proposer (提议者)  → 提出初始答案/方案
    2. Opposer (反对者)   → 提出反对意见和替代方案
    3. Judge   (裁判)     → 综合评判, 选择最优方案

  优势:
    - 多角度分析问题
    - 减少单一视角的偏见
    - 通过辩论发现隐藏的问题

  应用:
    - 决策分析
    - 代码审查
    - 方案评估
    - 知识问答
""")


# ============================================================
# 2. 辩论Agent实现
# ============================================================
class DebateAgent:
    """辩论Agent"""

    def __init__(self, name, stance, expertise):
        self.name = name
        self.stance = stance  # 'proposer', 'opposer', 'judge'
        self.expertise = expertise
        self.arguments = []
        self.score = 0

    def argue(self, topic, previous_args=None):
        """生成论点"""
        if self.stance == 'proposer':
            arg = self._propose(topic, previous_args)
        elif self.stance == 'opposer':
            arg = self._oppose(topic, previous_args)
        else:
            arg = self._judge(topic, previous_args)

        self.arguments.append(arg)
        return arg

    def _propose(self, topic, prev):
        proposals = {
            'default': {
                'claim': f"我主张{topic}是可行的",
                'evidence': ['数据支持方案A', '已有成功案例', '成本可控'],
                'confidence': 0.8,
            },
        }
        return proposals.get('default', proposals['default'])

    def _oppose(self, topic, prev):
        return {
            'claim': f"我反对关于{topic}的观点, 有以下风险",
            'counter_evidence': ['方案忽略了隐私风险', '成本被低估', '缺乏长期数据'],
            'alternative': '建议采用更保守的方案B',
            'confidence': 0.7,
        }

    def _judge(self, topic, prev):
        return {
            'decision': '综合评判',
            'proposer_score': 0.75,
            'opposer_score': 0.70,
            'winner': 'proposer',
            'reasoning': '提议者提供了更多数据支撑, 但反对者指出了重要风险',
            'recommendation': '采用方案A, 但加入反对者提出的隐私保护措施',
        }


class DebateSystem:
    """辩论系统"""

    def __init__(self, proposer, opposer, judge, max_rounds=3):
        self.proposer = proposer
        self.opposer = opposer
        self.judge = judge
        self.max_rounds = max_rounds
        self.debate_log = []
        self.evolution_scores = []

    def run_debate(self, topic):
        """运行辩论"""
        print(f"\n  辩论主题: {topic}")
        print(f"  {'=' * 50}")

        for round_num in range(1, self.max_rounds + 1):
            print(f"\n  --- Round {round_num} ---")

            # 提议者发言
            prop_arg = self.proposer.argue(topic, self.debate_log)
            print(f"  [{self.proposer.name}] {prop_arg['claim']}")
            if 'evidence' in prop_arg:
                for e in prop_arg.get('evidence', []):
                    print(f"    证据: {e}")

            # 反对者发言
            opp_arg = self.opposer.argue(topic, self.debate_log)
            print(f"  [{self.opposer.name}] {opp_arg['claim']}")
            if 'counter_evidence' in opp_arg:
                for e in opp_arg.get('counter_evidence', []):
                    print(f"    反驳: {e}")

            # 裁判评判
            judge_arg = self.judge.argue(topic, self.debate_log)
            print(f"  [{self.judge.name}] {judge_arg['decision']}")
            print(f"    提议者: {judge_arg['proposer_score']:.2f}, "
                  f"反对者: {judge_arg['opposer_score']:.2f}")
            print(f"    推荐: {judge_arg['recommendation']}")

            self.debate_log.append({
                'round': round_num,
                'proposer': prop_arg,
                'opposer': opp_arg,
                'judge': judge_arg,
            })

            self.evolution_scores.append({
                'round': round_num,
                'proposer': judge_arg['proposer_score'],
                'opposer': judge_arg['opposer_score'],
            })

        # 最终裁决
        final = self.judge.argue(topic, self.debate_log)
        print(f"\n  === 最终裁决 ===")
        print(f"  胜者: {final.get('winner', 'proposer')}")
        print(f"  推荐方案: {final.get('recommendation', '综合方案')}")

        return final


# ============================================================
# 3. 运行辩论
# ============================================================
print("=" * 60)
print("2. 辩论系统演示")
print("=" * 60)

proposer = DebateAgent("技术专家", "proposer", "技术架构")
opposer = DebateAgent("安全专家", "opposer", "安全与风险")
judge = DebateAgent("首席架构师", "judge", "综合评判")

debate = DebateSystem(proposer, opposer, judge, max_rounds=3)
final_result = debate.run_debate("在公司内部部署LLM服务")


# ============================================================
# 4. 答案聚合策略
# ============================================================
print("\n" + "=" * 60)
print("3. 答案聚合策略")
print("=" * 60)

class AnswerAggregator:
    """多Agent答案聚合"""

    def __init__(self):
        self.strategies = {
            'majority_vote': self._majority_vote,
            'weighted_average': self._weighted_average,
            'best_of_n': self._best_of_n,
            'cascade': self._cascade,
        }

    def aggregate(self, answers, strategy='majority_vote', **kwargs):
        """聚合多个答案"""
        print(f"\n  使用策略: {strategy}")
        print(f"  输入答案: {len(answers)}个")

        result = self.strategies[strategy](answers, **kwargs)
        return result

    def _majority_vote(self, answers, **kwargs):
        """多数投票"""
        from collections import Counter
        vote_counts = Counter(answers)
        winner = vote_counts.most_common(1)[0]
        print(f"    投票结果: {dict(vote_counts)}")
        print(f"    多数选择: {winner[0]} ({winner[1]}票)")
        return {'answer': winner[0], 'confidence': winner[1] / len(answers)}

    def _weighted_average(self, answers, weights=None, **kwargs):
        """加权平均"""
        if weights is None:
            weights = [1.0] * len(answers)

        weighted_sum = sum(a * w for a, w in zip(answers, weights))
        total_weight = sum(weights)
        result = weighted_sum / total_weight

        print(f"    加权结果: {result:.3f}")
        return {'answer': result, 'method': 'weighted_average'}

    def _best_of_n(self, answers, scores=None, **kwargs):
        """选择最佳"""
        if scores is None:
            scores = [0.5] * len(answers)

        best_idx = scores.index(max(scores))
        print(f"    各答案分数: {list(zip(answers, scores))}")
        print(f"    最佳答案: {answers[best_idx]} (分数: {scores[best_idx]})")
        return {'answer': answers[best_idx], 'score': scores[best_idx]}

    def _cascade(self, answers, **kwargs):
        """级联验证"""
        print(f"    级联验证:")
        validated = answers[0]
        for i, ans in enumerate(answers[1:], 1):
            if ans != validated:
                print(f"      级{i}: 发现不一致 '{validated}' vs '{ans}'")
                validated = ans  # 以后面的为准
            else:
                print(f"      级{i}: 一致确认 '{ans}'")
        return {'answer': validated, 'method': 'cascade'}


# 测试聚合策略
aggregator = AnswerAggregator()

# 多数投票
aggregator.aggregate(['方案A', '方案A', '方案B', '方案A', '方案B'], 'majority_vote')

# 加权平均
aggregator.aggregate([0.8, 0.7, 0.85, 0.75], 'weighted_average', weights=[3, 2, 3, 2])

# Best-of-N
aggregator.aggregate(
    ['微服务', '单体', '微服务'],
    'best_of_n',
    scores=[0.85, 0.6, 0.9],
)

# 级联验证
aggregator.aggregate(['正确', '正确', '正确', '错误', '正确'], 'cascade')


# ============================================================
# 5. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 5.1 辩论分数演化
ax1 = axes[0, 0]
if debate.evolution_scores:
    rounds = [s['round'] for s in debate.evolution_scores]
    prop_scores = [s['proposer'] for s in debate.evolution_scores]
    opp_scores = [s['opposer'] for s in debate.evolution_scores]
    ax1.plot(rounds, prop_scores, 'g-o', linewidth=2, markersize=10, label='提议者')
    ax1.plot(rounds, opp_scores, 'r-s', linewidth=2, markersize=10, label='反对者')
    ax1.set_xlabel('辩论轮次')
    ax1.set_ylabel('得分')
    ax1.set_title('辩论得分演化')
    ax1.legend()
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

# 5.2 聚合策略对比
ax2 = axes[0, 1]
strategies = ['Majority\nVote', 'Weighted\nAverage', 'Best-of-N', 'Cascade']
accuracy = [0.82, 0.85, 0.88, 0.80]
speed = [0.95, 0.90, 0.85, 0.70]
x = np.arange(len(strategies))
width = 0.35
ax2.bar(x - width / 2, accuracy, width, label='准确率', color='#4CAF50', edgecolor='black')
ax2.bar(x + width / 2, speed, width, label='速度', color='#2196F3', edgecolor='black')
ax2.set_xticks(x)
ax2.set_xticklabels(strategies)
ax2.set_ylabel('分数')
ax2.set_title('聚合策略对比')
ax2.legend()
ax2.set_ylim(0, 1.1)

# 5.3 Agent数量 vs 辩论质量
ax3 = axes[1, 0]
agent_nums = [2, 3, 5, 7, 10]
quality_no_judge = [0.65, 0.72, 0.78, 0.80, 0.81]
quality_with_judge = [0.70, 0.80, 0.87, 0.91, 0.93]
time_cost = [2, 4, 8, 14, 25]

ax3_twin = ax3.twinx()
l1 = ax3.plot(agent_nums, quality_no_judge, 'b-o', linewidth=2, label='无裁判质量')
l2 = ax3.plot(agent_nums, quality_with_judge, 'g-s', linewidth=2, label='有裁判质量')
l3 = ax3_twin.plot(agent_nums, time_cost, 'r-^', linewidth=2, label='时间成本')
ax3.set_xlabel('Agent数量')
ax3.set_ylabel('回答质量', color='blue')
ax3_twin.set_ylabel('时间成本 (s)', color='red')
ax3.set_title('Agent数量 vs 辩论效果')
lines = l1 + l2 + l3
ax3.legend(lines, [l.get_label() for l in lines])

# 5.4 迭代优化效果
ax4 = axes[1, 1]
iterations = range(1, 8)
initial_quality = 0.60
qualities = []
q = initial_quality
for i in iterations:
    improvement = 0.05 * np.exp(-i / 5) + np.random.normal(0, 0.01)
    q = min(q + improvement, 0.95)
    qualities.append(q)

ax4.plot(list(iterations), qualities, 'go-', linewidth=2, markersize=10)
ax4.axhline(y=0.90, color='r', linestyle='--', label='目标质量')
ax4.fill_between(list(iterations), qualities, alpha=0.15, color='green')
ax4.set_xlabel('迭代轮次')
ax4.set_ylabel('回答质量')
ax4.set_title('迭代优化效果')
ax4.legend()
ax4.set_ylim(0.5, 1.0)
ax4.grid(True, alpha=0.3)

plt.suptitle('W23-D6: 辩论式Multi-Agent', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W23/d6_debate_agent.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存!")
