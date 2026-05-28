"""
W28-D7 MLOps综合项目
=====================
MLOps综合项目: 完整ML系统(数据Pipeline+训练+部署+监控), 配置文件, 架构文档
"""

import os
import json
import time
import hashlib
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W28-D7 MLOps综合项目")
print("=" * 60)

# ============================================================
# 项目: 完整的ML系统 - 乳腺癌分类
# ============================================================
print("""
项目: 乳腺癌分类 - 完整MLOps系统

  涵盖所有MLOps组件:
  1. 配置管理
  2. 数据Pipeline (采集/验证/清洗)
  3. 特征工程
  4. 模型训练与实验追踪
  5. 模型评估与选择
  6. 模型注册
  7. 模型服务 (API)
  8. 监控系统
""")

# ============================================================
# 1. 配置管理
# ============================================================
print("\n=== 1. 配置管理 ===")


class Config:
    """项目配置"""

    def __init__(self):
        self.config = {
            'project': {
                'name': 'breast_cancer_classifier',
                'version': '1.0.0',
                'description': '乳腺癌分类MLOps项目',
            },
            'data': {
                'dataset': 'sklearn_breast_cancer',
                'test_size': 0.2,
                'random_state': 42,
                'validation': {
                    'max_null_pct': 0.05,
                    'min_samples': 100,
                },
            },
            'features': {
                'scaler': 'StandardScaler',
                'selection': {
                    'method': 'importance',
                    'top_k': 15,
                },
            },
            'training': {
                'models': ['LogisticRegression', 'RandomForest', 'GradientBoosting'],
                'cv_folds': 5,
                'random_state': 42,
                'hyperparameters': {
                    'RandomForest': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [5, 10, 15],
                    },
                    'GradientBoosting': {
                        'n_estimators': [50, 100],
                        'learning_rate': [0.05, 0.1, 0.2],
                    },
                },
            },
            'deployment': {
                'accuracy_threshold': 0.90,
                'latency_p99_max_ms': 50,
                'max_batch_size': 100,
            },
            'monitoring': {
                'drift_psi_threshold': 0.2,
                'accuracy_min': 0.85,
                'check_interval_hours': 24,
                'alert_channels': ['email', 'slack'],
            },
        }

    def get(self, key_path, default=None):
        """获取配置值 (支持点分隔路径)"""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def to_json(self):
        return json.dumps(self.config, indent=2, ensure_ascii=False)


config = Config()
print(f"  项目: {config.get('project.name')}")
print(f"  模型: {config.get('training.models')}")
print(f"  准确率阈值: {config.get('deployment.accuracy_threshold')}")
print(f"  漂移PSI阈值: {config.get('monitoring.drift_psi_threshold')}")

# ============================================================
# 2. 实验追踪器
# ============================================================
print("\n=== 2. 实验追踪器 ===")


class ExperimentTracker:
    def __init__(self):
        self.runs = []
        self.current_run = None

    def start_run(self, name, params):
        self.current_run = {
            'run_id': hashlib.md5(f"{time.time()}{name}".encode()).hexdigest()[:8],
            'name': name, 'params': params,
            'metrics': {}, 'start_time': datetime.now().isoformat(),
        }

    def log_metric(self, key, value):
        if self.current_run:
            self.current_run['metrics'][key] = value

    def end_run(self, status='FINISHED'):
        if self.current_run:
            self.current_run['status'] = status
            self.current_run['end_time'] = datetime.now().isoformat()
            self.runs.append(self.current_run)
            self.current_run = None

    def get_best_run(self, metric='test_accuracy'):
        return max(self.runs, key=lambda r: r['metrics'].get(metric, 0))

    def print_leaderboard(self):
        print(f"\n  {'模型':<25} {'参数':<30} {'测试准确率':<12} {'CV准确率'}")
        print("  " + "-" * 80)
        for run in sorted(self.runs, key=lambda r: r['metrics'].get('test_accuracy', 0), reverse=True):
            params_str = str(run['params'])[:28]
            ta = run['metrics'].get('test_accuracy', 0)
            ca = run['metrics'].get('cv_accuracy', 0)
            print(f"  {run['name']:<25} {params_str:<30} {ta:<12.4f} {ca:.4f}")


tracker = ExperimentTracker()

# ============================================================
# 3. 数据Pipeline
# ============================================================
print("\n=== 3. 数据Pipeline ===")


class DataPipeline:
    def __init__(self, config):
        self.config = config
        self.validation_results = []

    def load_data(self):
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        print(f"  数据加载: {df.shape}")
        return df, data.feature_names

    def validate(self, df):
        checks = []
        null_pct = df.isna().mean().max()
        min_samples = self.config.get('data.validation.min_samples', 100)
        checks.append(('空值率', null_pct < 0.05, f'{null_pct:.2%}'))
        checks.append(('数据量', len(df) >= min_samples, f'{len(df)}'))
        checks.append(('目标列存在', 'target' in df.columns, 'OK'))

        self.validation_results = checks
        print(f"  数据验证:")
        for name, passed, detail in checks:
            status = 'PASS' if passed else 'FAIL'
            print(f"    [{status}] {name}: {detail}")
        return all(c[1] for c in checks)

    def split(self, df):
        X = df.drop('target', axis=1).values
        y = df['target'].values
        test_size = self.config.get('data.test_size', 0.2)
        rs = self.config.get('data.random_state', 42)
        return train_test_split(X, y, test_size=test_size, random_state=rs, stratify=y)

    def scale(self, X_train, X_test):
        scaler = StandardScaler()
        return scaler.fit_transform(X_train), scaler.transform(X_test), scaler


data_pipeline = DataPipeline(config)
df, feature_names = data_pipeline.load_data()
data_pipeline.validate(df)
X_train, X_test, y_train, y_test = data_pipeline.split(df)
X_train_s, X_test_s, scaler = data_pipeline.scale(X_train, X_test)
print(f"  训练集: {X_train_s.shape}, 测试集: {X_test_s.shape}")

# ============================================================
# 4. 模型训练与实验
# ============================================================
print("\n=== 4. 模型训练与实验 ===")

models_to_train = {
    'LogisticRegression': (LogisticRegression(max_iter=1000, random_state=42),
                            {'max_iter': 1000}),
    'RandomForest_50_10': (RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42),
                            {'n_estimators': 50, 'max_depth': 10}),
    'RandomForest_100_10': (RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
                             {'n_estimators': 100, 'max_depth': 10}),
    'RandomForest_200_15': (RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42),
                             {'n_estimators': 200, 'max_depth': 15}),
    'GradientBoosting_100': (GradientBoostingClassifier(n_estimators=100, random_state=42),
                              {'n_estimators': 100}),
    'GradientBoosting_200': (GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42),
                              {'n_estimators': 200, 'learning_rate': 0.1}),
}

cv_folds = config.get('training.cv_folds', 5)

for name, (model, params) in models_to_train.items():
    tracker.start_run(name, params)

    # 训练
    model.fit(X_train_s, y_train)

    # 评估
    y_pred = model.predict(X_test_s)
    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_train_s, y_train, cv=cv_folds)

    tracker.log_metric('test_accuracy', test_acc)
    tracker.log_metric('test_f1', test_f1)
    tracker.log_metric('cv_accuracy', cv_scores.mean())
    tracker.log_metric('cv_std', cv_scores.std())
    tracker.end_run()

    print(f"  {name:<25} test_acc={test_acc:.4f}, cv_acc={cv_scores.mean():.4f}")

tracker.print_leaderboard()

# ============================================================
# 5. 模型注册
# ============================================================
print("\n=== 5. 模型注册 ===")


class ModelRegistry:
    def __init__(self):
        self.models = {}
        self.production_model = None

    def register(self, name, model, metrics, scaler):
        version = len(self.models) + 1
        self.models[version] = {
            'name': name, 'model': model, 'metrics': metrics,
            'scaler': scaler, 'stage': 'registered',
            'registered_at': datetime.now().isoformat(),
        }
        print(f"  注册模型: {name} v{version}")
        return version

    def promote_to_production(self, version, threshold=0.9):
        model_info = self.models[version]
        if model_info['metrics']['test_accuracy'] >= threshold:
            # 归档旧模型
            if self.production_model:
                self.models[self.production_model]['stage'] = 'archived'
            model_info['stage'] = 'production'
            self.production_model = version
            print(f"  推进到Production: v{version} ({model_info['name']})")
            return True
        else:
            print(f"  [拒绝] 准确率不达标: {model_info['metrics']['test_accuracy']:.4f}")
            return False

    def get_production_model(self):
        if self.production_model:
            return self.models[self.production_model]
        return None


registry = ModelRegistry()
best_run = tracker.get_best_run()
best_model = models_to_train[best_run['name']][0]

version = registry.register(
    best_run['name'], best_model,
    best_run['metrics'], scaler
)
registry.promote_to_production(version, config.get('deployment.accuracy_threshold', 0.9))

# ============================================================
# 6. 模型服务
# ============================================================
print("\n=== 6. 模型服务 ===")


class ModelServer:
    """模型服务"""

    def __init__(self, registry):
        self.registry = registry
        self.request_log = []
        self.error_count = 0

    def predict(self, features):
        """单条预测"""
        t0 = time.time()
        prod = self.registry.get_production_model()
        if not prod:
            return {'error': 'No production model'}

        try:
            X = prod['scaler'].transform([features])
            prediction = int(prod['model'].predict(X)[0])
            probability = float(prod['model'].predict_proba(X)[0][prediction])
            latency = (time.time() - t0) * 1000

            self.request_log.append({
                'prediction': prediction,
                'probability': probability,
                'latency_ms': latency,
            })

            return {
                'prediction': prediction,
                'probability': probability,
                'label': 'malignant' if prediction == 0 else 'benign',
                'latency_ms': latency,
                'model_version': self.registry.production_model,
            }
        except Exception as e:
            self.error_count += 1
            return {'error': str(e)}

    def batch_predict(self, features_list):
        """批量预测"""
        t0 = time.time()
        prod = self.registry.get_production_model()
        if not prod:
            return {'error': 'No production model'}

        X = prod['scaler'].transform(features_list)
        predictions = prod['model'].predict(X)
        probabilities = prod['model'].predict_proba(X)

        results = []
        for pred, probs in zip(predictions, probabilities):
            results.append({
                'prediction': int(pred),
                'probability': float(probs[pred]),
            })

        latency = (time.time() - t0) * 1000
        return {'results': results, 'count': len(results), 'latency_ms': latency}

    def health(self):
        return {
            'model_loaded': self.registry.get_production_model() is not None,
            'total_requests': len(self.request_log),
            'error_count': self.error_count,
        }

    def get_latency_stats(self):
        if not self.request_log:
            return {}
        latencies = [r['latency_ms'] for r in self.request_log]
        return {
            'count': len(latencies),
            'mean': np.mean(latencies),
            'p50': np.percentile(latencies, 50),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99),
        }


server = ModelServer(registry)

# 测试预测
sample = X_test[0].tolist()
result = server.predict(sample)
print(f"  预测结果: {result['label']} (概率={result['probability']:.4f})")

# 批量预测
batch = X_test[:10].tolist()
batch_result = server.batch_predict(batch)
print(f"  批量预测: {batch_result['count']} 条, 耗时={batch_result['latency_ms']:.2f}ms")

# 压测
np.random.seed(42)
for _ in range(500):
    idx = np.random.randint(0, len(X_test))
    server.predict(X_test[idx].tolist())

latency_stats = server.get_latency_stats()
print(f"\n  延迟统计 (500次请求):")
print(f"    平均: {latency_stats['mean']:.2f}ms")
print(f"    P50: {latency_stats['p50']:.2f}ms")
print(f"    P95: {latency_stats['p95']:.2f}ms")
print(f"    P99: {latency_stats['p99']:.2f}ms")

# ============================================================
# 7. 监控系统
# ============================================================
print("\n=== 7. 监控系统 ===")


class MonitoringSystem:
    def __init__(self, model_server, config):
        self.server = model_server
        self.config = config
        self.alerts = []
        self.daily_checks = []

    def daily_check(self, day, X_current, y_true):
        """每日监控检查"""
        prod = self.server.registry.get_production_model()
        if not prod:
            return

        model = prod['model']
        scaler = prod['scaler']
        X_scaled = scaler.transform(X_current)

        y_pred = model.predict(X_scaled)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        # PSI漂移检测 (简化)
        ref = prod['scaler'].transform(
            load_breast_cancer().data[:len(X_current)]
        )
        max_psi = 0
        for i in range(min(5, X_scaled.shape[1])):
            psi = self._compute_psi(ref[:, i], X_scaled[:, i])
            max_psi = max(max_psi, psi)

        check = {
            'day': day, 'accuracy': acc, 'f1_score': f1,
            'max_psi': max_psi, 'timestamp': datetime.now().isoformat(),
        }
        self.daily_checks.append(check)

        # 告警检查
        acc_threshold = self.config.get('monitoring.accuracy_min', 0.85)
        psi_threshold = self.config.get('monitoring.drift_psi_threshold', 0.2)

        if acc < acc_threshold:
            self.alerts.append({
                'day': day, 'level': 'CRITICAL',
                'message': f'准确率 {acc:.4f} < {acc_threshold}',
            })

        if max_psi > psi_threshold:
            self.alerts.append({
                'day': day, 'level': 'WARNING',
                'message': f'PSI {max_psi:.4f} > {psi_threshold}',
            })

        return check

    @staticmethod
    def _compute_psi(ref, cur, bins=10):
        all_data = np.concatenate([ref, cur])
        edges = np.percentile(all_data, np.linspace(0, 100, bins + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        rh = np.clip(np.histogram(ref, bins=edges)[0] / len(ref), 1e-6, None)
        ch = np.clip(np.histogram(cur, bins=edges)[0] / len(cur), 1e-6, None)
        return float(np.sum((ch - rh) * np.log(ch / rh)))

    def print_alerts(self):
        if self.alerts:
            print(f"\n  告警 ({len(self.alerts)} 条):")
            for a in self.alerts:
                print(f"    Day {a['day']}: [{a['level']}] {a['message']}")
        else:
            print(f"\n  无告警")


monitor = MonitoringSystem(server, config)

# 模拟30天监控
np.random.seed(42)
for day in range(30):
    drift = day * 0.003
    X_sim = X_test + np.random.randn(*X_test.shape) * drift
    monitor.daily_check(day, X_sim, y_test)

if monitor.daily_checks:
    last = monitor.daily_checks[-1]
    print(f"  Day 29: acc={last['accuracy']:.4f}, psi={last['max_psi']:.4f}")
    monitor.print_alerts()

# ============================================================
# 8. 架构文档 (输出)
# ============================================================
print("\n=== 8. 架构文档 ===")

architecture_doc = f"""
  项目: {config.get('project.name')} v{config.get('project.version')}
  描述: {config.get('project.description')}

  系统架构:
  ┌──────────────────────────────────────────────────┐
  │ 客户端 -> API -> 模型服务(v{registry.production_model}) -> 监控 │
  └──────────────────────────────────────────────────┘

  数据Pipeline:
    1. 数据加载: sklearn breast_cancer
    2. 数据验证: 空值率/数据量/Schema检查
    3. 数据分割: 训练/测试 = {int((1-config.get('data.test_size'))*100)}/{int(config.get('data.test_size')*100)}
    4. 标准化: StandardScaler

  模型训练:
    - 候选模型: {config.get('training.models')}
    - CV折数: {config.get('training.cv_folds')}
    - 最佳模型: {best_run['name']}
    - 测试准确率: {best_run['metrics']['test_accuracy']:.4f}
    - CV准确率: {best_run['metrics']['cv_accuracy']:.4f}

  部署配置:
    - 准确率阈值: {config.get('deployment.accuracy_threshold')}
    - 最大延迟P99: {config.get('deployment.latency_p99_max_ms')}ms

  监控配置:
    - 漂移PSI阈值: {config.get('monitoring.drift_psi_threshold')}
    - 最低准确率: {config.get('monitoring.accuracy_min')}
    - 检查间隔: {config.get('monitoring.check_interval_hours')}小时

  MLOps成熟度: Level 2 (CI/CD自动化)
"""
print(architecture_doc)

# ============================================================
# 9. 综合可视化
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. 实验排行榜
ax = axes[0, 0]
run_names = [r['name'][:20] for r in tracker.runs]
run_accs = [r['metrics']['test_accuracy'] for r in tracker.runs]
colors = ['#4CAF50' if a == max(run_accs) else '#2196F3' for a in run_accs]
ax.barh(run_names, run_accs, color=colors)
ax.set_xlabel('测试准确率')
ax.set_title('模型实验排行榜')
ax.set_xlim(0.9, 1.0)

# 2. CV准确率箱线图
ax = axes[0, 1]
cv_data = []
cv_labels = []
for name, (model, _) in models_to_train.items():
    scores = cross_val_score(model, X_train_s, y_train, cv=5)
    cv_data.append(scores)
    cv_labels.append(name[:15])
ax.boxplot(cv_data, labels=cv_labels)
ax.set_ylabel('CV准确率')
ax.set_title('5折交叉验证分布')
ax.tick_params(axis='x', rotation=45)

# 3. API延迟分布
ax = axes[0, 2]
latencies = [r['latency_ms'] for r in server.request_log]
ax.hist(latencies, bins=50, color='#2196F3', alpha=0.7)
ax.axvline(np.percentile(latencies, 99), color='red', linestyle='--', label='P99')
ax.set_xlabel('延迟 (ms)')
ax.set_ylabel('请求数')
ax.set_title('API延迟分布')
ax.legend()

# 4. 监控准确率趋势
ax = axes[1, 0]
if monitor.daily_checks:
    days = [c['day'] for c in monitor.daily_checks]
    accs = [c['accuracy'] for c in monitor.daily_checks]
    ax.plot(days, accs, 'b-', linewidth=2)
    ax.axhline(config.get('monitoring.accuracy_min', 0.85), color='red',
               linestyle='--', label='告警阈值')
    ax.set_xlabel('天数')
    ax.set_ylabel('准确率')
    ax.set_title('30天准确率监控')
    ax.legend()
    ax.grid(True, alpha=0.3)

# 5. 漂移PSI趋势
ax = axes[1, 1]
if monitor.daily_checks:
    days = [c['day'] for c in monitor.daily_checks]
    psis = [c['max_psi'] for c in monitor.daily_checks]
    ax.plot(days, psis, 'r-', linewidth=2)
    ax.axhline(config.get('monitoring.drift_psi_threshold', 0.2),
               color='orange', linestyle='--', label='PSI阈值')
    ax.set_xlabel('天数')
    ax.set_ylabel('最大PSI')
    ax.set_title('30天漂移监控')
    ax.legend()
    ax.grid(True, alpha=0.3)

# 6. 系统总结
ax = axes[1, 2]
ax.axis('off')
summary = [
    f"项目: {config.get('project.name')}",
    f"最佳模型: {best_run['name']}",
    f"测试准确率: {best_run['metrics']['test_accuracy']:.4f}",
    f"CV准确率: {best_run['metrics']['cv_accuracy']:.4f}",
    f"生产版本: v{registry.production_model}",
    f"API请求: {len(server.request_log)}",
    f"P99延迟: {latency_stats['p99']:.2f}ms",
    f"监控天数: {len(monitor.daily_checks)}",
    f"告警数: {len(monitor.alerts)}",
    f"MLOps等级: Level 2",
]
for i, line in enumerate(summary):
    ax.text(0.1, 0.9 - i * 0.09, line, fontsize=10,
            transform=ax.transAxes, fontfamily='monospace')
ax.set_title('系统总结')

plt.suptitle('MLOps综合项目 - 乳腺癌分类系统', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W28/d7_mlops_project.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存: d7_mlops_project.png")

print("\n" + "=" * 60)
print("MLOps综合项目完成!")
print("=" * 60)
print("""
  涵盖组件:
    1. 配置管理:    YAML/JSON配置文件
    2. 数据Pipeline: 加载/验证/清洗/分割
    3. 实验追踪:    多模型对比, 参数和指标记录
    4. 模型注册:    版本管理, 生产推广
    5. 模型服务:    单条/批量API, 延迟监控
    6. 监控系统:    准确率/漂移/告警
    7. 架构文档:    系统设计文档

  关键成果:
    - 最佳模型: """ + best_run['name'] + """
    - 测试准确率: """ + f"{best_run['metrics']['test_accuracy']:.4f}" + """
    - API P99延迟: """ + f"{latency_stats['p99']:.2f}ms" + """
    - MLOps成熟度: Level 2
""")
