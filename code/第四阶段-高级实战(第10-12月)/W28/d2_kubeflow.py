"""
W28-D2 Kubeflow
================
Kubeflow概念, Pipeline编排, 组件定义, 实验管理, 简化版Pipeline实现
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W28-D2 Kubeflow")
print("=" * 60)

# ============================================================
# 1. Kubeflow概念
# ============================================================
print("\n--- 1. Kubeflow概念 ---")
print("""
  Kubeflow 是Kubernetes原生的ML平台:

  核心组件:
    - Kubeflow Pipelines: ML工作流编排
    - Katib:            超参数调优(NAS)
    - Training Operator: 分布式训练(TF/PyTorch/MPICH)
    - KServe:           模型服务(推理)
    - Notebook Servers: Jupyter开发环境
    - ML Metadata:      元数据管理

  Pipeline核心概念:
    - Component:  一个独立的处理步骤(Docker容器)
    - Pipeline:   由多个Component组成的有向无环图(DAG)
    - Experiment: 一组相关的Pipeline运行
    - Run:        Pipeline的一次执行
    - Artifact:   步骤间的数据传递(输入/输出)

  Pipeline示例:
    @dsl.pipeline(name='ML Pipeline')
    def ml_pipeline(data_path):
        ingest = ingest_data(data_path=data_path)
        train = train_model(data=ingest.output)
        evaluate = evaluate_model(model=train.output)
        deploy = deploy_model(model=train.output, metrics=evaluate.output)
""")

# ============================================================
# 2. 简化版Pipeline编排
# ============================================================
print("\n--- 2. 简化版Pipeline编排 ---")


class PipelineComponent:
    """Pipeline组件"""

    def __init__(self, name, func, inputs=None, outputs=None):
        self.name = name
        self.func = func
        self.inputs = inputs or []
        self.outputs = outputs or []
        self.result = None
        self.execution_time = 0

    def execute(self, *args, **kwargs):
        t0 = time.time()
        self.result = self.func(*args, **kwargs)
        self.execution_time = time.time() - t0
        print(f"  [完成] {self.name}: 耗时 {self.execution_time:.3f}s")
        return self.result

    def __repr__(self):
        return f"Component({self.name})"


class KubeflowPipelineSimulator:
    """Kubeflow Pipeline模拟器"""

    def __init__(self, name):
        self.name = name
        self.components = []
        self.edges = []  # (from_component, to_component, data_key)
        self.execution_log = []

    def add_component(self, component):
        self.components.append(component)
        return component

    def add_edge(self, from_comp, to_comp, data_key='output'):
        self.edges.append((from_comp, to_comp, data_key))
        return self

    def compile(self):
        """编译Pipeline (验证DAG)"""
        # 拓扑排序
        visited = set()
        order = []

        def dfs(comp):
            if comp in visited:
                return
            visited.add(comp)
            for src, dst, _ in self.edges:
                if src == comp:
                    dfs(dst)
            order.append(comp)

        for comp in self.components:
            dfs(comp)

        order.reverse()
        self.execution_order = order

        print(f"\n  Pipeline编译: {self.name}")
        print(f"  组件数: {len(self.components)}")
        print(f"  执行顺序: {' -> '.join(c.name for c in order)}")
        return order

    def run(self, initial_inputs=None):
        """执行Pipeline"""
        print(f"\n  === 执行Pipeline: {self.name} ===")

        results = {}
        initial_inputs = initial_inputs or {}

        for comp in self.execution_order:
            # 收集输入
            inputs = {}
            for src, dst, key in self.edges:
                if dst == comp and src.result is not None:
                    inputs[key] = src.result

            # 执行
            try:
                result = comp.execute(**inputs)
                results[comp.name] = {
                    'status': 'success',
                    'result_type': type(result).__name__,
                    'execution_time': comp.execution_time,
                }
            except Exception as e:
                results[comp.name] = {
                    'status': 'failed',
                    'error': str(e),
                    'execution_time': 0,
                }
                print(f"  [失败] {comp.name}: {e}")
                break

            self.execution_log.append({
                'component': comp.name,
                'status': 'success',
                'time': comp.execution_time,
                'timestamp': datetime.now().isoformat(),
            })

        print(f"\n  Pipeline执行完成")
        return results


# ============================================================
# 3. 构建ML Pipeline
# ============================================================
print("\n--- 3. 构建ML Pipeline ---")


# 定义组件函数
def ingest_data(dataset_name='breast_cancer'):
    """数据摄取"""
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    print(f"    数据摄取: {dataset_name}, {df.shape}")
    return df


def preprocess_data(output, test_size=0.2):
    """数据预处理"""
    df = output
    X = df.drop('target', axis=1).values
    y = df['target'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42
    )
    print(f"    预处理: 训练={X_train.shape}, 测试={X_test.shape}")
    return {'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test, 'scaler': scaler}


def train_model(output, n_estimators=100, max_depth=10):
    """模型训练"""
    data = output
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )
    model.fit(data['X_train'], data['y_train'])

    y_pred = model.predict(data['X_train'])
    train_acc = accuracy_score(data['y_train'], y_pred)
    print(f"    训练: n_estimators={n_estimators}, 训练准确率={train_acc:.4f}")

    return {'model': model, 'data': data, 'train_accuracy': train_acc}


def evaluate_model(output):
    """模型评估"""
    model = output['model']
    data = output['data']

    y_pred = model.predict(data['X_test'])
    test_acc = accuracy_score(data['y_test'], y_pred)

    cv_scores = cross_val_score(model, data['X_train'], data['y_train'], cv=5)

    metrics = {
        'test_accuracy': test_acc,
        'cv_accuracy': cv_scores.mean(),
        'cv_std': cv_scores.std(),
    }
    print(f"    评估: test_acc={test_acc:.4f}, cv_acc={cv_scores.mean():.4f}")
    return {'model': model, 'metrics': metrics, 'data': data}


def deploy_decision(output, accuracy_threshold=0.9):
    """部署决策"""
    metrics = output['metrics']
    deploy = metrics['test_accuracy'] >= accuracy_threshold

    if deploy:
        print(f"    部署决策: 批准部署 (acc={metrics['test_accuracy']:.4f} >= {accuracy_threshold})")
    else:
        print(f"    部署决策: 拒绝部署 (acc={metrics['test_accuracy']:.4f} < {accuracy_threshold})")

    return {
        'deploy': deploy,
        'metrics': metrics,
        'model': output['model'],
    }


# 创建Pipeline
pipeline = KubeflowPipelineSimulator('breast_cancer_ml_pipeline')

comp_ingest = PipelineComponent('ingest_data', ingest_data, outputs=['data'])
comp_preprocess = PipelineComponent('preprocess_data', preprocess_data,
                                     inputs=['data'], outputs=['processed_data'])
comp_train = PipelineComponent('train_model', train_model,
                                inputs=['processed_data'], outputs=['model'])
comp_evaluate = PipelineComponent('evaluate_model', evaluate_model,
                                   inputs=['model_data'], outputs=['metrics'])
comp_deploy = PipelineComponent('deploy_decision', deploy_decision,
                                 inputs=['evaluation'], outputs=['decision'])

pipeline.add_component(comp_ingest)
pipeline.add_component(comp_preprocess)
pipeline.add_component(comp_train)
pipeline.add_component(comp_evaluate)
pipeline.add_component(comp_deploy)

pipeline.add_edge(comp_ingest, comp_preprocess, 'output')
pipeline.add_edge(comp_preprocess, comp_train, 'output')
pipeline.add_edge(comp_train, comp_evaluate, 'output')
pipeline.add_edge(comp_evaluate, comp_deploy, 'output')

# 编译并运行
pipeline.compile()
results = pipeline.run()

# ============================================================
# 4. 实验管理
# ============================================================
print("\n--- 4. 实验管理 ---")


class ExperimentManager:
    """简化版实验管理"""

    def __init__(self):
        self.experiments = {}

    def run_experiment(self, name, pipeline_fn, params):
        """运行实验"""
        if name not in self.experiments:
            self.experiments[name] = []

        t0 = time.time()
        result = pipeline_fn(**params)
        elapsed = time.time() - t0

        run = {
            'params': params,
            'result': result,
            'elapsed': elapsed,
            'timestamp': datetime.now().isoformat(),
        }
        self.experiments[name].append(run)

        print(f"\n  实验 {name} Run {len(self.experiments[name])}:")
        print(f"    参数: {params}")
        print(f"    结果: test_acc={result['metrics']['test_accuracy']:.4f}")
        print(f"    耗时: {elapsed:.3f}s")
        return run

    def compare(self, name):
        """对比实验"""
        if name not in self.experiments:
            return

        runs = self.experiments[name]
        print(f"\n  实验 {name} 对比 ({len(runs)} 次运行):")
        print(f"  {'Run':<5} {'n_est':<8} {'depth':<8} {'准确率':<10} {'CV准确率':<10}")
        print("  " + "-" * 50)

        for i, run in enumerate(runs, 1):
            params = run['params']
            metrics = run['result']['metrics']
            print(f"  {i:<5} {params.get('n_estimators', '-'):<8} "
                  f"{params.get('max_depth', '-'):<8} "
                  f"{metrics['test_accuracy']:<10.4f} "
                  f"{metrics['cv_accuracy']:<10.4f}")


def run_ml_pipeline(n_estimators=100, max_depth=10):
    """完整ML Pipeline函数"""
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)

    return {
        'metrics': {
            'test_accuracy': test_acc,
            'cv_accuracy': cv_scores.mean(),
        }
    }


em = ExperimentManager()
for n_est, depth in [(50, 5), (100, 10), (150, 15), (200, None)]:
    em.run_experiment('rf_tuning', run_ml_pipeline,
                      {'n_estimators': n_est, 'max_depth': depth})

em.compare('rf_tuning')

# ============================================================
# 5. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 左上: Pipeline DAG可视化
ax = axes[0, 0]
ax.set_xlim(-1, 5)
ax.set_ylim(-1, 5)
ax.axis('off')
ax.set_title('Pipeline DAG')

comp_positions = {
    'ingest_data': (2, 4),
    'preprocess_data': (2, 3),
    'train_model': (2, 2),
    'evaluate_model': (2, 1),
    'deploy_decision': (2, 0),
}

comp_colors = {
    'ingest_data': '#BBDEFB',
    'preprocess_data': '#C8E6C9',
    'train_model': '#FFE0B2',
    'evaluate_model': '#E1BEE7',
    'deploy_decision': '#FFCDD2',
}

for name, (x, y) in comp_positions.items():
    color = comp_colors[name]
    ax.add_patch(plt.Rectangle((x - 1.2, y - 0.3), 2.4, 0.6,
                                facecolor=color, edgecolor='#333', linewidth=1.5))
    ax.text(x, y, name, ha='center', va='center', fontsize=9, fontweight='bold')

for src_name, (sx, sy) in comp_positions.items():
    for dst_name, (dx, dy) in comp_positions.items():
        if dy == sy - 1 and dx == sx:
            ax.annotate('', xy=(dx, dy + 0.3), xytext=(sx, sy - 0.3),
                        arrowprops=dict(arrowstyle='->', color='#666', lw=2))

# 右上: 组件执行耗时
ax = axes[0, 1]
comp_names = [c.name for c in pipeline.execution_order]
comp_times = [c.execution_time for c in pipeline.execution_order]
colors = [comp_colors.get(n, '#2196F3') for n in comp_names]
ax.barh(comp_names, comp_times, color=colors)
ax.set_xlabel('耗时 (秒)')
ax.set_title('组件执行耗时')
for i, t in enumerate(comp_times):
    ax.text(t + 0.001, i, f'{t:.3f}s', va='center')

# 左下: 实验结果对比
ax = axes[1, 0]
runs = em.experiments['rf_tuning']
params_list = [str(r['params']) for r in runs]
accs = [r['result']['metrics']['test_accuracy'] for r in runs]
cv_accs = [r['result']['metrics']['cv_accuracy'] for r in runs]

x = np.arange(len(runs))
width = 0.35
ax.bar(x - width/2, accs, width, label='Test Accuracy', color='#2196F3')
ax.bar(x + width/2, cv_accs, width, label='CV Accuracy', color='#4CAF50')
ax.set_xticks(x)
ax.set_xticklabels([f'Run {i+1}' for i in range(len(runs))])
ax.set_ylabel('准确率')
ax.set_title('实验结果对比')
ax.legend()
ax.set_ylim(0.9, 1.0)

# 右下: Pipeline执行时间线
ax = axes[1, 1]
cumulative_time = np.cumsum(comp_times)
ax.barh(comp_names, comp_times, left=[0] + list(cumulative_time[:-1]),
        color=colors)
ax.set_xlabel('时间 (秒)')
ax.set_title('Pipeline执行时间线')
for i, (name, t, ct) in enumerate(zip(comp_names, comp_times, cumulative_time)):
    ax.text(ct - t/2, i, f'{t:.3f}s', ha='center', va='center', fontsize=8, color='white')

plt.suptitle('Kubeflow Pipeline编排', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W28/d2_kubeflow.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存: d2_kubeflow.png")

print("\n完成! Kubeflow要点:")
print("  1. Pipeline: 由组件(Component)组成的DAG")
print("  2. 组件: 独立的处理步骤, Docker容器化")
print("  3. 编排: 拓扑排序确定执行顺序")
print("  4. 实验: 多参数组合的自动化运行")
print("  5. 元数据: 自动记录执行过程和产物")
