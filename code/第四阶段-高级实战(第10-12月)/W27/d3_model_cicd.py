"""
W27-D3 模型CI/CD
=================
ML CI/CD流水线, GitHub Actions for ML, 模型测试(单元/集成/性能), 自动化训练Pipeline
"""

import os
import json
import time
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W27-D3 模型CI/CD")
print("=" * 60)

# ============================================================
# 1. ML CI/CD概念
# ============================================================
print("\n--- 1. ML CI/CD概念 ---")
print("""
  传统CI/CD vs ML CI/CD:
  ┌────────────┬──────────────────────┬──────────────────────┐
  │ 阶段       │ 传统软件             │ ML系统               │
  ├────────────┼──────────────────────┼──────────────────────┤
  │ 持续集成CI │ 代码测试+构建        │ 数据验证+模型训练+测试│
  │ 持续交付CD │ 部署应用             │ 部署模型+监控         │
  │ 持续训练CT │ 无                   │ 自动重训练(漂移时)    │
  └────────────┴──────────────────────┴──────────────────────┘

  ML CI/CD Pipeline:
  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │ 数据验证 │->│ 模型训练 │->│ 模型测试 │->│ 模型注册 │->│ 模型部署 │
  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘
       │              │              │              │              │
   Schema检查    参数记录       性能测试       版本管理       灰度发布
   质量门控      指标记录       公平性测试     审批流程       A/B测试
""")

# ============================================================
# 2. GitHub Actions for ML (概念)
# ============================================================
print("\n--- 2. GitHub Actions for ML ---")
print("""
  .github/workflows/ml-pipeline.yml 示例:

  name: ML Pipeline
  on:
    push:
      paths:
        - 'data/**'
        - 'src/**'
    schedule:
      - cron: '0 2 * * *'  # 每天凌晨2点

  jobs:
    data-validation:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - run: python scripts/validate_data.py

    model-training:
      needs: data-validation
      runs-on: ubuntu-latest
      steps:
        - run: python scripts/train.py

    model-testing:
      needs: model-training
      steps:
        - run: python scripts/test_model.py

    model-deployment:
      needs: model-testing
      if: success()
      steps:
        - run: python scripts/deploy.py
""")

# ============================================================
# 3. 模型测试框架
# ============================================================
print("\n--- 3. 模型测试框架 ---")


class ModelTestSuite:
    """模型测试套件"""

    def __init__(self, model, X_test, y_test, feature_names=None):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names
        self.results = []

    def test_prediction_shape(self):
        """单元测试: 预测输出形状"""
        y_pred = self.model.predict(self.X_test)
        passed = len(y_pred) == len(self.y_test)
        self.results.append({
            'test': 'prediction_shape',
            'type': '单元测试',
            'passed': passed,
            'detail': f'预测数量={len(y_pred)}, 期望={len(self.y_test)}',
        })
        return passed

    def test_prediction_values(self):
        """单元测试: 预测值范围"""
        y_pred = self.model.predict(self.X_test)
        unique = np.unique(y_pred)
        expected = np.unique(self.y_test)
        passed = set(unique).issubset(set(expected))
        self.results.append({
            'test': 'prediction_values',
            'type': '单元测试',
            'passed': passed,
            'detail': f'预测类别={unique}, 期望类别={expected}',
        })
        return passed

    def test_predict_proba(self):
        """单元测试: 概率输出"""
        try:
            proba = self.model.predict_proba(self.X_test)
            passed = proba.shape[1] == len(np.unique(self.y_test))
            self.results.append({
                'test': 'predict_proba',
                'type': '单元测试',
                'passed': passed,
                'detail': f'概率矩阵形状={proba.shape}',
            })
            return passed
        except Exception as e:
            self.results.append({
                'test': 'predict_proba',
                'type': '单元测试',
                'passed': False,
                'detail': str(e),
            })
            return False

    def test_accuracy_threshold(self, threshold=0.85):
        """性能测试: 准确率阈值"""
        y_pred = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        passed = acc >= threshold
        self.results.append({
            'test': f'accuracy >= {threshold}',
            'type': '性能测试',
            'passed': passed,
            'detail': f'accuracy={acc:.4f}',
        })
        return passed

    def test_cv_stability(self, cv=5, max_std=0.05):
        """性能测试: 交叉验证稳定性"""
        scores = cross_val_score(self.model, self.X_test, self.y_test, cv=cv)
        passed = scores.std() <= max_std
        self.results.append({
            'test': f'cv_stability (std <= {max_std})',
            'type': '性能测试',
            'passed': passed,
            'detail': f'cv_mean={scores.mean():.4f}, cv_std={scores.std():.4f}',
        })
        return passed

    def test_latency(self, max_latency_ms=100, n_samples=100):
        """延迟测试"""
        sample = self.X_test[:n_samples]
        latencies = []
        for _ in range(n_samples):
            t0 = time.time()
            self.model.predict(sample[:1])
            latencies.append((time.time() - t0) * 1000)

        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        passed = p95_latency <= max_latency_ms
        self.results.append({
            'test': f'latency_p95 <= {max_latency_ms}ms',
            'type': '延迟测试',
            'passed': passed,
            'detail': f'avg={avg_latency:.2f}ms, p95={p95_latency:.2f}ms',
        })
        return passed

    def test_robustness(self, noise_level=0.1):
        """鲁棒性测试: 加噪声后性能下降是否在可接受范围"""
        y_pred_clean = self.model.predict(self.X_test)
        acc_clean = accuracy_score(self.y_test, y_pred_clean)

        X_noisy = self.X_test.copy()
        if isinstance(X_noisy, pd.DataFrame):
            for col in X_noisy.select_dtypes(include=[np.number]).columns:
                X_noisy[col] += np.random.randn(len(X_noisy)) * noise_level * X_noisy[col].std()
        else:
            noise = np.random.randn(*X_noisy.shape) * noise_level
            X_noisy = X_noisy + noise

        y_pred_noisy = self.model.predict(X_noisy)
        acc_noisy = accuracy_score(self.y_test, y_pred_noisy)

        degradation = acc_clean - acc_noisy
        passed = degradation < 0.1  # 性能下降不超过10%
        self.results.append({
            'test': f'robustness (noise={noise_level})',
            'type': '鲁棒性测试',
            'passed': passed,
            'detail': f'clean_acc={acc_clean:.4f}, noisy_acc={acc_noisy:.4f}, degradation={degradation:.4f}',
        })
        return passed

    def test_feature_importance_stability(self):
        """特征重要性稳定性测试"""
        if not hasattr(self.model, 'feature_importances_'):
            self.results.append({
                'test': 'feature_importance',
                'type': '稳定性测试',
                'passed': True,
                'detail': '模型不支持feature_importances_, 跳过',
            })
            return True

        importances = self.model.feature_importances_
        # 检查没有单一特征占主导 (>90%)
        max_importance = importances.max()
        passed = max_importance < 0.9
        self.results.append({
            'test': 'feature_importance_balance',
            'type': '稳定性测试',
            'passed': passed,
            'detail': f'max_importance={max_importance:.4f}',
        })
        return passed

    def run_all(self):
        """运行所有测试"""
        self.results = []
        self.test_prediction_shape()
        self.test_prediction_values()
        self.test_predict_proba()
        self.test_accuracy_threshold()
        self.test_cv_stability()
        self.test_latency()
        self.test_robustness()
        self.test_feature_importance_stability()
        return self.report()

    def report(self):
        """生成测试报告"""
        passed = sum(1 for r in self.results if r['passed'])
        total = len(self.results)

        print(f"\n  模型测试报告 ({passed}/{total} 通过):")
        print(f"  {'测试名':<35} {'类型':<10} {'状态':<6} {'详情'}")
        print("  " + "-" * 80)

        for r in self.results:
            status = 'PASS' if r['passed'] else 'FAIL'
            print(f"  {r['test']:<35} {r['type']:<10} {status:<6} {r['detail']}")

        all_passed = passed == total
        print(f"\n  结果: {'全部通过' if all_passed else '存在失败'}")
        return all_passed


# ============================================================
# 4. 运行模型测试
# ============================================================
print("\n--- 4. 运行模型测试 ---")

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

suite = ModelTestSuite(model, X_test, y_test, feature_names=data.feature_names)
all_passed = suite.run_all()

# ============================================================
# 5. 自动化训练Pipeline
# ============================================================
print("\n--- 5. 自动化训练Pipeline ---")


class MLPipeline:
    """自动化ML Pipeline"""

    def __init__(self, config):
        self.config = config
        self.stages = []
        self.model = None
        self.metrics = {}

    def _add_stage(self, name, status, detail=''):
        self.stages.append({
            'stage': name, 'status': status,
            'detail': detail, 'timestamp': datetime.now().isoformat(),
        })
        symbol = 'OK' if status == 'success' else 'FAIL'
        print(f"  [{symbol}] {name}: {detail}")

    def run(self, X, y):
        """运行完整Pipeline"""
        # Stage 1: 数据验证
        try:
            assert len(X) > 0, "数据为空"
            assert len(X) == len(y), "X和y长度不匹配"
            null_pct = np.isnan(X).mean() if isinstance(X, np.ndarray) else pd.DataFrame(X).isna().mean().mean()
            assert null_pct < 0.1, f"空值率过高: {null_pct:.2%}"
            self._add_stage('数据验证', 'success', f'{len(X)} 条记录, 空值率={null_pct:.2%}')
        except Exception as e:
            self._add_stage('数据验证', 'failure', str(e))
            return False

        # Stage 2: 数据分割
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.get('test_size', 0.2),
                random_state=self.config.get('random_state', 42),
                stratify=y,
            )
            self._add_stage('数据分割', 'success',
                            f'训练={len(X_train)}, 测试={len(X_test)}')
        except Exception as e:
            self._add_stage('数据分割', 'failure', str(e))
            return False

        # Stage 3: 模型训练
        try:
            self.model = RandomForestClassifier(
                n_estimators=self.config.get('n_estimators', 100),
                max_depth=self.config.get('max_depth', 10),
                random_state=self.config.get('random_state', 42),
            )
            self.model.fit(X_train, y_train)
            self._add_stage('模型训练', 'success', f'{type(self.model).__name__}')
        except Exception as e:
            self._add_stage('模型训练', 'failure', str(e))
            return False

        # Stage 4: 模型评估
        try:
            y_pred = self.model.predict(X_test)
            self.metrics['accuracy'] = accuracy_score(y_test, y_pred)
            self.metrics['train_accuracy'] = accuracy_score(y_train, self.model.predict(X_train))

            threshold = self.config.get('accuracy_threshold', 0.8)
            if self.metrics['accuracy'] >= threshold:
                self._add_stage('模型评估', 'success',
                                f'acc={self.metrics["accuracy"]:.4f} (threshold={threshold})')
            else:
                self._add_stage('模型评估', 'failure',
                                f'acc={self.metrics["accuracy"]:.4f} < {threshold}')
                return False
        except Exception as e:
            self._add_stage('模型评估', 'failure', str(e))
            return False

        # Stage 5: 模型测试
        try:
            test_suite = ModelTestSuite(self.model, X_test, y_test)
            test_suite.test_accuracy_threshold(threshold)
            test_suite.test_latency()
            test_results = all(r['passed'] for r in test_suite.results)
            if test_results:
                self._add_stage('模型测试', 'success', f'{len(test_suite.results)} 项测试全部通过')
            else:
                self._add_stage('模型测试', 'failure', '部分测试未通过')
                return False
        except Exception as e:
            self._add_stage('模型测试', 'failure', str(e))
            return False

        self._add_stage('Pipeline', 'success', '全部阶段通过')
        return True

    def get_pipeline_summary(self):
        """获取Pipeline摘要"""
        success = sum(1 for s in self.stages if s['status'] == 'success')
        total = len(self.stages)
        return {
            'total_stages': total,
            'success': success,
            'failed': total - success,
            'all_passed': success == total,
            'stages': self.stages,
        }


# 运行Pipeline
print("\n--- 运行自动化Pipeline ---")
config = {
    'test_size': 0.2,
    'random_state': 42,
    'n_estimators': 100,
    'max_depth': 10,
    'accuracy_threshold': 0.90,
}

pipeline = MLPipeline(config)
success = pipeline.run(X, y)

summary = pipeline.get_pipeline_summary()
print(f"\n  Pipeline结果: {'成功' if summary['all_passed'] else '失败'} "
      f"({summary['success']}/{summary['total_stages']} 阶段通过)")

# ============================================================
# 6. 可视化
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 左: Pipeline阶段状态
ax = axes[0]
stages = [s['stage'] for s in pipeline.stages]
statuses = [s['status'] for s in pipeline.stages]
colors = ['#4CAF50' if s == 'success' else '#F44336' for s in statuses]
ax.barh(range(len(stages)), [1]*len(stages), color=colors)
ax.set_yticks(range(len(stages)))
ax.set_yticklabels(stages, fontsize=9)
ax.set_xlim(0, 1.5)
ax.set_xticks([])
ax.set_title('Pipeline阶段状态')
for i, (stage, status) in enumerate(zip(stages, statuses)):
    symbol = 'PASS' if status == 'success' else 'FAIL'
    ax.text(1.1, i, symbol, va='center', fontweight='bold',
            color='#4CAF50' if status == 'success' else '#F44336')

# 中: 模型测试结果
ax = axes[1]
test_passed = sum(1 for r in suite.results if r['passed'])
test_failed = len(suite.results) - test_passed
ax.pie([test_passed, test_failed],
       labels=[f'通过 ({test_passed})', f'失败 ({test_failed})'],
       colors=['#4CAF50', '#F44336'],
       autopct='%1.0f%%', startangle=90)
ax.set_title(f'模型测试 ({len(suite.results)}项)')

# 右: 测试详情
ax = axes[2]
ax.axis('off')
ax.set_title('测试详情')
for i, r in enumerate(suite.results):
    symbol = 'PASS' if r['passed'] else 'FAIL'
    color = '#4CAF50' if r['passed'] else '#F44336'
    ax.text(0.05, 0.95 - i * 0.12, f'[{symbol}] {r["test"]}',
            transform=ax.transAxes, fontsize=9, color=color, fontfamily='monospace')

plt.suptitle('模型CI/CD Pipeline', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W27/d3_model_cicd.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存: d3_model_cicd.png")

print("\n完成! 模型CI/CD要点:")
print("  1. Pipeline: 数据验证 -> 训练 -> 测试 -> 注册 -> 部署")
print("  2. 模型测试: 单元/性能/延迟/鲁棒性/稳定性")
print("  3. 质量门控: 准确率/延迟阈值检查")
print("  4. 自动化: 触发条件(推送/定时/事件)")
print("  5. 持续训练: 数据变化时自动重训练")
