"""
W27-D5 自动重训练策略
======================
自动重训练策略, 触发条件(定时/漂移/性能下降), A/B测试, 灰度发布, 模型回滚
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W27-D5 自动重训练策略")
print("=" * 60)

# ============================================================
# 1. 自动重训练概述
# ============================================================
print("\n--- 1. 自动重训练概述 ---")
print("""
  重训练触发条件:
  ┌──────────────┬─────────────────────────────────────┐
  │ 触发类型     │ 描述                                │
  ├──────────────┼─────────────────────────────────────┤
  │ 定时触发     │ 每天/每周/每月自动训练              │
  │ 漂移触发     │ 数据漂移检测到时触发                │
  │ 性能下降     │ 模型准确率低于阈值时触发            │
  │ 数据量触发   │ 新增数据达到阈值时触发              │
  │ 手动触发     │ 人工判断需要重训练                  │
  └──────────────┴─────────────────────────────────────┘

  重训练后部署流程:
    1. 训练新模型
    2. 通过测试套件
    3. 影子模式验证 (Shadow)
    4. 灰度发布 (Canary)
    5. A/B测试
    6. 全量发布 / 回滚
""")

# ============================================================
# 2. 触发条件管理器
# ============================================================
print("\n--- 2. 触发条件管理器 ---")


class RetrainingTrigger:
    """重训练触发器"""

    def __init__(self):
        self.triggers = []
        self.trigger_history = []

    def add_trigger(self, name, condition_fn, description=''):
        self.triggers.append({
            'name': name,
            'condition': condition_fn,
            'description': description,
        })

    def check_all(self, context):
        """检查所有触发条件"""
        fired = []
        for trigger in self.triggers:
            try:
                should_fire = trigger['condition'](context)
                if should_fire:
                    fired.append(trigger['name'])
                    self.trigger_history.append({
                        'trigger': trigger['name'],
                        'timestamp': datetime.now().isoformat(),
                        'context_snapshot': str(context)[:100],
                    })
            except Exception as e:
                print(f"  [错误] 触发器 {trigger['name']} 检查失败: {e}")
        return fired


class ScheduledTrigger:
    """定时触发"""

    def __init__(self, interval_days=7):
        self.interval = timedelta(days=interval_days)
        self.last_training = datetime.now() - timedelta(days=interval_days - 1)

    def should_trigger(self, context):
        return datetime.now() - self.last_training >= self.interval


class DriftTrigger:
    """漂移触发"""

    def __init__(self, psi_threshold=0.2):
        self.psi_threshold = psi_threshold

    def should_trigger(self, context):
        psi_values = context.get('psi_values', {})
        return any(v > self.psi_threshold for v in psi_values.values())


class PerformanceTrigger:
    """性能下降触发"""

    def __init__(self, threshold=0.85, window=5):
        self.threshold = threshold
        self.window = window

    def should_trigger(self, context):
        recent_scores = context.get('recent_accuracy', [])
        if len(recent_scores) < self.window:
            return False
        return np.mean(recent_scores[-self.window:]) < self.threshold


# ============================================================
# 3. A/B测试框架
# ============================================================
print("\n--- 3. A/B测试框架 ---")


class ABTestFramework:
    """A/B测试框架"""

    def __init__(self, model_a, model_b, name='ab_test'):
        self.model_a = model_a  # 对照组 (当前模型)
        self.model_b = model_b  # 实验组 (新模型)
        self.name = name
        self.results_a = []
        self.results_b = []
        self.assignment_log = []

    def assign(self, user_id):
        """分配用户到A或B组"""
        # 基于用户ID哈希分配, 确保同一用户始终在同一组
        group = 'A' if hash(str(user_id)) % 2 == 0 else 'B'
        self.assignment_log.append({'user_id': user_id, 'group': group})
        return group

    def predict(self, X, user_id):
        """根据分组选择模型"""
        group = self.assign(user_id)
        if group == 'A':
            return self.model_a.predict(X), group
        else:
            return self.model_b.predict(X), group

    def record_result(self, group, y_true, y_pred):
        """记录结果"""
        correct = (y_true == y_pred).sum()
        total = len(y_true)
        if group == 'A':
            self.results_a.append({'correct': correct, 'total': total})
        else:
            self.results_b.append({'correct': correct, 'total': total})

    def analyze(self):
        """分析A/B测试结果"""
        if not self.results_a or not self.results_b:
            print("  数据不足, 无法分析")
            return None

        acc_a = sum(r['correct'] for r in self.results_a) / sum(r['total'] for r in self.results_a)
        acc_b = sum(r['correct'] for r in self.results_b) / sum(r['total'] for r in self.results_b)

        total_a = sum(r['total'] for r in self.results_a)
        total_b = sum(r['total'] for r in self.results_b)

        # 简单的Z检验
        p_a = acc_a
        p_b = acc_b
        p_pool = (acc_a * total_a + acc_b * total_b) / (total_a + total_b)
        se = np.sqrt(p_pool * (1 - p_pool) * (1/total_a + 1/total_b))
        z_score = (p_b - p_a) / (se + 1e-10)
        p_value = 2 * (1 - 0.5 * (1 + np.math.erf(abs(z_score) / np.sqrt(2))))

        result = {
            'accuracy_a': acc_a,
            'accuracy_b': acc_b,
            'total_a': total_a,
            'total_b': total_b,
            'z_score': z_score,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'b_better': acc_b > acc_a,
        }

        print(f"\n  A/B测试结果:")
        print(f"    模型A (当前): acc={acc_a:.4f} (n={total_a})")
        print(f"    模型B (新):   acc={acc_b:.4f} (n={total_b})")
        print(f"    Z-score: {z_score:.3f}, p-value: {p_value:.4f}")
        print(f"    统计显著: {'是' if result['significant'] else '否'}")
        print(f"    B更好: {'是' if result['b_better'] else '否'}")

        return result


# ============================================================
# 4. 灰度发布
# ============================================================
print("\n--- 4. 灰度发布 ---")


class CanaryDeployment:
    """灰度发布管理器"""

    def __init__(self, model_old, model_new):
        self.model_old = model_old
        self.model_new = model_new
        self.traffic_percentage = 0  # 新模型流量百分比
        self.stages = [1, 5, 10, 25, 50, 100]
        self.current_stage_idx = 0
        self.metrics_old = []
        self.metrics_new = []
        self.rollback_threshold = 0.03  # 性能下降3%触发回滚

    def promote(self):
        """提升新模型流量"""
        if self.current_stage_idx < len(self.stages):
            self.traffic_percentage = self.stages[self.current_stage_idx]
            self.current_stage_idx += 1
            print(f"  灰度发布: 新模型流量提升至 {self.traffic_percentage}%")
            return True
        return False

    def route_request(self, request_id):
        """路由请求"""
        if np.random.random() * 100 < self.traffic_percentage:
            return self.model_new, 'new'
        return self.model_old, 'old'

    def record_metrics(self, model_version, accuracy):
        """记录指标"""
        if model_version == 'new':
            self.metrics_new.append(accuracy)
        else:
            self.metrics_old.append(accuracy)

    def check_rollback(self):
        """检查是否需要回滚"""
        if not self.metrics_new:
            return False

        avg_new = np.mean(self.metrics_new[-10:])
        avg_old = np.mean(self.metrics_old[-10:]) if self.metrics_old else 1.0

        if avg_old - avg_new > self.rollback_threshold:
            print(f"  [回滚] 新模型性能下降: old={avg_old:.4f}, new={avg_new:.4f}")
            return True
        return False

    def rollback(self):
        """执行回滚"""
        self.traffic_percentage = 0
        self.current_stage_idx = 0
        print(f"  [回滚] 已回滚到旧模型, 新模型流量=0%")

    def status(self):
        print(f"\n  灰度发布状态:")
        print(f"    新模型流量: {self.traffic_percentage}%")
        print(f"    旧模型样本: {len(self.metrics_old)}")
        print(f"    新模型样本: {len(self.metrics_new)}")
        if self.metrics_old:
            print(f"    旧模型平均: {np.mean(self.metrics_old[-10:]):.4f}")
        if self.metrics_new:
            print(f"    新模型平均: {np.mean(self.metrics_new[-10:]):.4f}")


# ============================================================
# 5. 模拟完整的重训练流程
# ============================================================
print("\n--- 5. 模拟完整的重训练流程 ---")

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# "旧"模型
model_old = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
model_old.fit(X_train, y_train)
acc_old = accuracy_score(y_test, model_old.predict(X_test))
print(f"\n  当前模型准确率: {acc_old:.4f}")

# 模拟"新"训练的模型 (更好的超参数)
model_new = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)
model_new.fit(X_train, y_train)
acc_new = accuracy_score(y_test, model_new.predict(X_test))
print(f"  新训练模型准确率: {acc_new:.4f}")

# --- A/B测试 ---
print("\n  === A/B测试 ===")
ab = ABTestFramework(model_old, model_new, 'retrain_ab_test')

np.random.seed(42)
for i in range(100):
    user_id = f'user_{i:04d}'
    y_pred, group = ab.predict(X_test, user_id)
    ab.record_result(group, y_test, y_pred)

ab_result = ab.analyze()

# --- 灰度发布 ---
print("\n  === 灰度发布 ===")
canary = CanaryDeployment(model_old, model_new)

np.random.seed(42)
for stage in range(len(canary.stages)):
    canary.promote()

    # 模拟该阶段的流量
    for _ in range(50):
        model, version = canary.route_request(f'req_{stage}_{_}')
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        canary.record_metrics(version, acc)

    if canary.check_rollback():
        canary.rollback()
        break

canary.status()

# ============================================================
# 6. 模型版本管理与回滚
# ============================================================
print("\n--- 6. 模型版本管理与回滚 ---")


class ModelVersionManager:
    """模型版本管理器"""

    def __init__(self):
        self.versions = []
        self.active_version = None

    def add_version(self, model, metrics, description=''):
        version = {
            'version': len(self.versions) + 1,
            'model': model,
            'metrics': metrics,
            'description': description,
            'timestamp': datetime.now().isoformat(),
            'status': 'active' if self.active_version is None else 'standby',
        }
        self.versions.append(version)
        if self.active_version is None:
            self.active_version = version['version']
        print(f"  添加版本 v{version['version']}: {description}")
        return version['version']

    def get_active(self):
        for v in self.versions:
            if v['version'] == self.active_version:
                return v
        return None

    def rollback(self, target_version=None):
        """回滚到指定版本(默认上一版本)"""
        if target_version is None:
            target_version = self.active_version - 1

        if target_version < 1 or target_version > len(self.versions):
            print(f"  无效版本: v{target_version}")
            return False

        old_active = self.active_version
        self.active_version = target_version
        print(f"  回滚: v{old_active} -> v{target_version}")
        return True

    def promote(self, version):
        """将指定版本设为活跃"""
        old = self.active_version
        self.active_version = version
        print(f"  推进: v{old} -> v{version}")

    def list_versions(self):
        print(f"\n  模型版本:")
        for v in self.versions:
            active = ' [ACTIVE]' if v['version'] == self.active_version else ''
            acc = v['metrics'].get('accuracy', 'N/A')
            print(f"    v{v['version']}: acc={acc}, {v['description']}{active}")


# 使用版本管理器
vm = ModelVersionManager()
vm.add_version(model_old, {'accuracy': acc_old}, '原始模型 (n_est=50)')
vm.add_version(model_new, {'accuracy': acc_new}, '重训练模型 (n_est=200)')
vm.list_versions()

# ============================================================
# 7. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 左上: A/B测试结果
ax = axes[0, 0]
if ab_result:
    bars = ax.bar(['模型A (当前)', '模型B (新)'],
                   [ab_result['accuracy_a'], ab_result['accuracy_b']],
                   color=['#2196F3', '#4CAF50'])
    ax.set_ylabel('准确率')
    ax.set_title(f'A/B测试 (p={ab_result["p_value"]:.4f})')
    for bar, val in zip(bars, [ab_result['accuracy_a'], ab_result['accuracy_b']]):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', fontweight='bold')
    ax.set_ylim(0.85, 1.0)

# 右上: 灰度发布进度
ax = axes[0, 1]
stages_completed = min(canary.current_stage_idx, len(canary.stages))
if canary.metrics_old and canary.metrics_new:
    ax.plot(canary.metrics_old, label='旧模型', alpha=0.5, color='#2196F3')
    ax.plot(canary.metrics_new, label='新模型', alpha=0.5, color='#4CAF50')
    ax.set_xlabel('请求序号')
    ax.set_ylabel('准确率')
    ax.set_title(f'灰度发布 (流量={canary.traffic_percentage}%)')
    ax.legend()

# 左下: 触发条件决策树
ax = axes[1, 0]
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_title('重训练决策流程')

decisions = [
    (5, 9, '监控指标'),
    (5, 7.5, '性能下降?'),
    (2.5, 6, '数据漂移?'),
    (7.5, 6, '触发重训练'),
    (2.5, 4, '数据量变化?'),
    (7.5, 4, '通过测试'),
    (5, 2, '灰度发布'),
    (5, 0.5, '全量发布 / 回滚'),
]

for x, y, text in decisions:
    color = '#BBDEFB' if '?' in text else '#C8E6C9' if text in ['触发重训练', '全量发布 / 回滚'] else '#FFF9C4'
    ax.add_patch(plt.Rectangle((x - 1.2, y - 0.4), 2.4, 0.8,
                                facecolor=color, edgecolor='#333', linewidth=1))
    ax.text(x, y, text, ha='center', va='center', fontsize=9)

# 箭头
arrows = [(5, 8.6, 5, 7.9), (5, 7.1, 2.5, 6.4), (5, 7.1, 7.5, 6.4),
          (2.5, 5.6, 2.5, 4.4), (7.5, 5.6, 7.5, 4.4), (7.5, 3.6, 5, 2.4),
          (5, 1.6, 5, 0.9)]
for x1, y1, x2, y2 in arrows:
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#666', lw=1.5))

# 右下: 模型版本时间线
ax = axes[1, 1]
versions_data = [(v['version'], v['metrics'].get('accuracy', 0), v['description']) for v in vm.versions]
for ver, acc, desc in versions_data:
    color = '#4CAF50' if ver == vm.active_version else '#BBDEFB'
    ax.barh(ver, acc, color=color, height=0.5, edgecolor='#333')
    ax.text(acc + 0.002, ver, f'v{ver}: {acc:.4f} - {desc[:25]}',
            va='center', fontsize=9)

ax.set_xlabel('准确率')
ax.set_ylabel('版本')
ax.set_title('模型版本历史')
ax.set_yticks([v[0] for v in versions_data])
ax.set_yticklabels([f'v{v[0]}' for v in versions_data])
ax.set_xlim(0.85, 1.0)

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W27/d5_retraining.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存: d5_retraining.png")

print("\n完成! 自动重训练策略要点:")
print("  1. 触发条件: 定时/漂移/性能下降/数据量变化")
print("  2. A/B测试: 统计显著性验证新模型是否更好")
print("  3. 灰度发布: 逐步增加新模型流量百分比")
print("  4. 自动回滚: 性能下降超过阈值时自动回退")
print("  5. 版本管理: 维护模型版本历史, 支持快速回滚")
