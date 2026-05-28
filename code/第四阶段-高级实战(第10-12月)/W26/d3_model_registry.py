"""
W26-D3 模型注册中心
====================
模型注册中心概念, 模型版本管理, 模型阶段(Staging/Production), 模型审批流程
"""

import os
import json
import hashlib
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W26-D3 模型注册中心")
print("=" * 60)

# ============================================================
# 1. 模型注册中心概念
# ============================================================
print("\n--- 1. 模型注册中心概念 ---")
print("""
模型注册中心 (Model Registry) 是MLOps中的模型管理核心:

  模型生命周期:
    None -> Registered -> Staging -> Production -> Archived

  ┌────────────┐
  │  Registered │  模型注册到中心
  └─────┬──────┘
        v
  ┌────────────┐
  │   Staging   │  预发布环境, A/B测试
  └─────┬──────┘
        v
  ┌────────────┐
  │ Production  │  生产环境, 服务用户
  └─────┬──────┘
        v
  ┌────────────┐
  │  Archived   │  已归档, 不再使用
  └────────────┘

  核心功能:
    - 版本管理: 同一模型的多版本共存
    - 阶段转换: Staging -> Production 需审批
    - 元数据:   训练数据、参数、指标、签名
    - 审批流程: 多人审核, 确保模型质量
    - 回滚:     快速回滚到上一个版本

  工具: MLflow Model Registry, SageMaker Model Registry, Vertex AI Model Registry
""")

# ============================================================
# 2. 简化版模型注册中心
# ============================================================
print("\n--- 2. 简化版模型注册中心 ---")


class ModelVersion:
    """模型版本"""

    def __init__(self, model_name, version, run_id=None):
        self.model_name = model_name
        self.version = version
        self.run_id = run_id
        self.stage = 'None'
        self.status = 'READY'
        self.description = ''
        self.tags = {}
        self.metrics = {}
        self.params = {}
        self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
        self.approvals = []
        self.model_object = None

    def to_dict(self):
        return {
            'model_name': self.model_name,
            'version': self.version,
            'run_id': self.run_id,
            'stage': self.stage,
            'status': self.status,
            'description': self.description,
            'tags': self.tags,
            'metrics': self.metrics,
            'params': self.params,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'approvals': self.approvals,
        }


class ModelRegistry:
    """模型注册中心"""

    STAGES = ['None', 'Staging', 'Production', 'Archived']

    def __init__(self):
        self.models = {}  # model_name -> [ModelVersion]

    def register_model(self, model_name, model_object, metrics, params,
                       description='', run_id=None):
        """注册模型"""
        if model_name not in self.models:
            self.models[model_name] = []

        version = len(self.models[model_name]) + 1
        mv = ModelVersion(model_name, version, run_id)
        mv.model_object = model_object
        mv.metrics = metrics
        mv.params = params
        mv.description = description
        mv.stage = 'None'

        self.models[model_name].append(mv)
        print(f"  注册模型: {model_name} v{version} (accuracy={metrics.get('accuracy', 'N/A')})")
        return mv

    def transition_stage(self, model_name, version, new_stage, approver=''):
        """转换模型阶段"""
        if model_name not in self.models:
            print(f"  模型 {model_name} 不存在")
            return None

        versions = self.models[model_name]
        mv = None
        for v in versions:
            if v.version == version:
                mv = v
                break

        if mv is None:
            print(f"  版本 {version} 不存在")
            return None

        old_stage = mv.stage

        # 审批逻辑
        if new_stage == 'Production':
            if mv.metrics.get('accuracy', 0) < 0.85:
                print(f"  [拒绝] 准确率 {mv.metrics['accuracy']:.4f} 低于阈值 0.85")
                return None
            mv.approvals.append({
                'from': old_stage,
                'to': new_stage,
                'approver': approver,
                'timestamp': datetime.now().isoformat(),
            })

        # 如果推进到Production, 将旧版本归档
        if new_stage == 'Production':
            for v in versions:
                if v.stage == 'Production' and v.version != version:
                    v.stage = 'Archived'
                    v.updated_at = datetime.now().isoformat()
                    print(f"  旧版本 v{v.version} 已归档")

        mv.stage = new_stage
        mv.updated_at = datetime.now().isoformat()
        print(f"  {model_name} v{version}: {old_stage} -> {new_stage} (审批人: {approver})")
        return mv

    def get_production_model(self, model_name):
        """获取生产环境模型"""
        if model_name not in self.models:
            return None
        for mv in reversed(self.models[model_name]):
            if mv.stage == 'Production':
                return mv
        return None

    def list_models(self):
        """列出所有模型"""
        print(f"\n  {'模型名':<25} {'版本':<6} {'阶段':<12} {'准确率':<10} {'描述'}")
        print("  " + "-" * 75)
        for name, versions in self.models.items():
            for mv in versions:
                acc = mv.metrics.get('accuracy', 'N/A')
                if isinstance(acc, float):
                    acc = f'{acc:.4f}'
                print(f"  {name:<25} v{mv.version:<5} {mv.stage:<12} {acc:<10} {mv.description[:20]}")

    def get_model_history(self, model_name):
        """获取模型版本历史"""
        if model_name not in self.models:
            return []
        return [(mv.version, mv.stage, mv.metrics, mv.approvals)
                for mv in self.models[model_name]]


# ============================================================
# 3. 注册和管理工作流
# ============================================================
print("\n--- 3. 注册和管理工作流 ---")

# 准备数据
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

registry = ModelRegistry()

# --- 模型1: Logistic Regression ---
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_s, y_train)
acc_lr = accuracy_score(y_test, lr.predict(X_test_s))

registry.register_model(
    'breast_cancer_classifier', lr,
    metrics={'accuracy': acc_lr, 'model_type': 'LogisticRegression'},
    params={'C': 1.0, 'max_iter': 1000},
    description='基线Logistic Regression模型',
    run_id='run_001',
)

# --- 模型2: Random Forest v1 ---
rf1 = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
rf1.fit(X_train_s, y_train)
acc_rf1 = accuracy_score(y_test, rf1.predict(X_test_s))

registry.register_model(
    'breast_cancer_classifier', rf1,
    metrics={'accuracy': acc_rf1, 'model_type': 'RandomForest'},
    params={'n_estimators': 50, 'max_depth': 5},
    description='Random Forest v1',
    run_id='run_002',
)

# --- 模型3: Random Forest v2 ---
rf2 = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf2.fit(X_train_s, y_train)
acc_rf2 = accuracy_score(y_test, rf2.predict(X_test_s))

registry.register_model(
    'breast_cancer_classifier', rf2,
    metrics={'accuracy': acc_rf2, 'model_type': 'RandomForest'},
    params={'n_estimators': 200, 'max_depth': 10},
    description='Random Forest v2 - 更多树',
    run_id='run_003',
)

# --- 模型4: Gradient Boosting ---
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train_s, y_train)
acc_gb = accuracy_score(y_test, gb.predict(X_test_s))

registry.register_model(
    'breast_cancer_classifier', gb,
    metrics={'accuracy': acc_gb, 'model_type': 'GradientBoosting'},
    params={'n_estimators': 100, 'learning_rate': 0.1},
    description='Gradient Boosting',
    run_id='run_004',
)

# --- 模型5: 低质量模型 (测试审批拒绝) ---
lr_bad = LogisticRegression(C=0.001, max_iter=10, random_state=42)
lr_bad.fit(X_train_s, y_train)
acc_bad = accuracy_score(y_test, lr_bad.predict(X_test_s))

registry.register_model(
    'breast_cancer_classifier', lr_bad,
    metrics={'accuracy': acc_bad, 'model_type': 'LogisticRegression'},
    params={'C': 0.001, 'max_iter': 10},
    description='低质量模型(测试审批)',
    run_id='run_005',
)

# ============================================================
# 4. 阶段转换与审批
# ============================================================
print("\n--- 4. 阶段转换与审批 ---")

# v1 -> Staging
registry.transition_stage('breast_cancer_classifier', 1, 'Staging', 'engineer_A')

# v2 -> Staging -> Production
registry.transition_stage('breast_cancer_classifier', 2, 'Staging', 'engineer_A')
registry.transition_stage('breast_cancer_classifier', 2, 'Production', 'manager_B')

# v3 -> Staging -> Production (替换v2)
registry.transition_stage('breast_cancer_classifier', 3, 'Staging', 'engineer_A')
registry.transition_stage('breast_cancer_classifier', 3, 'Production', 'manager_B')

# v5 -> Production (应该被拒绝)
print("\n  尝试将低质量模型推到Production:")
registry.transition_stage('breast_cancer_classifier', 5, 'Production', 'manager_B')

# 列出所有模型
print("\n--- 当前模型状态 ---")
registry.list_models()

# 获取当前生产模型
prod = registry.get_production_model('breast_cancer_classifier')
if prod:
    print(f"\n  当前生产模型: v{prod.version} ({prod.params}), 准确率={prod.metrics['accuracy']:.4f}")

# ============================================================
# 5. 模型审批流程可视化
# ============================================================
print("\n--- 5. 模型审批流程 ---")
print("""
  典型审批流程:

  1. 数据科学家训练新模型
     └─> 注册到Registry (None)
  2. 通过初步验证
     └─> 推到Staging
  3. 在Staging环境进行A/B测试/影子测试
     └─> 验证通过, 请求审批
  4. ML工程师/经理审核
     └─> 审批通过 -> Production
     └─> 审批拒绝 -> 打回修改
  5. 生产环境监控
     └─> 性能下降 -> 回滚到上一版本

  审批检查项:
    - 准确率/F1是否达标
    - 公平性指标是否合格
    - 延迟是否在SLA内
    - 数据版本是否正确
    - 测试报告是否完整
""")

# ============================================================
# 6. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 左上: 模型版本性能
ax = axes[0, 0]
versions_data = registry.get_model_history('breast_cancer_classifier')
versions = [f'v{v[0]}' for v in versions_data]
accuracies = [v[2].get('accuracy', 0) for v in versions_data]
stages = [v[1] for v in versions_data]
stage_colors = {'None': '#9E9E9E', 'Staging': '#FF9800', 'Production': '#4CAF50',
                'Archived': '#BDBDBD'}
colors = [stage_colors.get(s, '#9E9E9E') for s in stages]

bars = ax.bar(versions, accuracies, color=colors)
ax.axhline(0.95, color='red', linestyle='--', alpha=0.7, label='目标阈值 0.95')
ax.axhline(0.85, color='orange', linestyle='--', alpha=0.7, label='最低阈值 0.85')
ax.set_ylabel('准确率')
ax.set_title('模型版本准确率')
ax.legend(fontsize=8)
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.003,
            f'{acc:.3f}', ha='center', fontsize=9)

# 右上: 模型生命周期时间线
ax = axes[0, 1]
ax.set_xlim(-1, 6)
ax.set_ylim(-0.5, 5.5)
ax.set_title('模型生命周期')
ax.axis('off')

stages_list = ['Registered', 'Staging', 'Production', 'Archived']
stage_y = {s: i for i, s in enumerate(reversed(stages_list))}

for i, stage in enumerate(reversed(stages_list)):
    y = i
    ax.add_patch(plt.Rectangle((-0.5, y - 0.3), 2, 0.6,
                                facecolor=list(stage_colors.values())[list(stage_colors.keys()).index(
                                    stage if stage != 'Registered' else 'None')],
                                edgecolor='#333', linewidth=1))
    ax.text(0.5, y, stage, ha='center', va='center', fontweight='bold')

# 绘制转换箭头
for v_idx, (ver, stage, metrics, approvals) in enumerate(versions_data):
    for approval in approvals:
        from_stage = approval['from'] if approval['from'] != 'None' else 'Registered'
        to_stage = approval['to']
        if from_stage in stage_y and to_stage in stage_y:
            ax.annotate('', xy=(2.5 + v_idx * 0.5, stage_y[to_stage]),
                        xytext=(2.5 + v_idx * 0.5, stage_y[from_stage]),
                        arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))

# 左下: 准确率趋势
ax = axes[1, 0]
valid_versions = [(v, m) for v, s, m, a in versions_data if m.get('accuracy')]
if valid_versions:
    v_nums = [v[0] for v in valid_versions]
    accs = [v[1]['accuracy'] for v in valid_versions]
    ax.plot(v_nums, accs, 'bo-', markersize=10, linewidth=2)
    ax.set_xlabel('版本号')
    ax.set_ylabel('准确率')
    ax.set_title('模型准确率演进')
    ax.grid(True, alpha=0.3)
    for v, a in zip(v_nums, accs):
        ax.annotate(f'{a:.3f}', (v, a), textcoords="offset points",
                    xytext=(0, 10), ha='center')

# 右下: 审批统计
ax = axes[1, 1]
approval_counts = {'通过': 0, '拒绝': 0, '待审批': 0}
for v, s, m, a in versions_data:
    if a:
        if s in ['Production', 'Staging']:
            approval_counts['通过'] += len(a)
        else:
            approval_counts['待审批'] += 1

# 尝试拒绝的次数
refused = 1
approval_counts['拒绝'] = refused

ax.bar(approval_counts.keys(), approval_counts.values(),
       color=['#4CAF50', '#F44336', '#FF9800'])
ax.set_ylabel('次数')
ax.set_title('审批统计')
for i, (k, v) in enumerate(approval_counts.items()):
    ax.text(i, v + 0.1, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W26/d3_model_registry.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存: d3_model_registry.png")

print("\n完成! 模型注册中心要点:")
print("  1. 版本管理: 同一模型的多版本共存和追溯")
print("  2. 阶段管理: None -> Staging -> Production -> Archived")
print("  3. 审批流程: 质量门控, 多人审核, 自动化检查")
print("  4. 回滚机制: 快速回退到稳定版本")
print("  5. 生产就绪: 确保只有经过验证的模型才能上线")
