"""
W28-D3 Airflow for ML
======================
Airflow for ML, DAG设计, 任务依赖, 定时调度, ML Pipeline示例
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W28-D3 Airflow for ML")
print("=" * 60)

# ============================================================
# 1. Airflow概念
# ============================================================
print("\n--- 1. Airflow概念 ---")
print("""
  Apache Airflow 核心概念:

    - DAG (有向无环图):   工作流定义, 任务及其依赖关系
    - Task:              DAG中的一个节点(任务)
    - Operator:          任务的类型(Python/Spark/Email...)
    - Task Instance:     任务的一次运行实例
    - Sensor:            等待条件的特殊Operator

  Airflow DAG示例 (Python):

    from airflow import DAG
    from airflow.operators.python import PythonOperator

    with DAG('ml_pipeline', start_date=datetime(2024,1,1),
             schedule_interval='@daily') as dag:

        task_ingest = PythonOperator(
            task_id='ingest_data',
            python_callable=ingest_data,
        )

        task_train = PythonOperator(
            task_id='train_model',
            python_callable=train_model,
        )

        task_ingest >> task_train >> task_deploy

  调度表达式:
    - '@daily':   每天执行
    - '@hourly':  每小时执行
    - '0 2 * * *': Cron表达式, 每天凌晨2点
    - None:       不自动调度, 手动触发
""")

# ============================================================
# 2. 简化版Airflow DAG实现
# ============================================================
print("\n--- 2. 简化版Airflow DAG实现 ---")


class AirflowTask:
    """Airflow任务"""

    def __init__(self, task_id, python_callable, retries=2):
        self.task_id = task_id
        self.callable = python_callable
        self.retries = retries
        self.downstream = []
        self.status = 'pending'
        self.result = None
        self.start_time = None
        self.end_time = None
        self.duration = 0
        self.attempt = 0

    def __rshift__(self, other):
        """支持 >> 语法"""
        self.downstream.append(other)
        return other

    def __lshift__(self, other):
        """支持 << 语法"""
        other.downstream.append(self)
        return other

    def execute(self, context=None):
        """执行任务"""
        self.start_time = datetime.now()
        context = context or {}

        for attempt in range(1, self.retries + 1):
            self.attempt = attempt
            try:
                self.result = self.callable(context)
                self.status = 'success'
                break
            except Exception as e:
                self.status = 'retry' if attempt < self.retries else 'failed'
                print(f"    [重试 {attempt}/{self.retries}] {self.task_id}: {e}")
                if attempt == self.retries:
                    raise

        self.end_time = datetime.now()
        self.duration = (self.end_time - self.start_time).total_seconds()
        return self.result


class AirflowDAG:
    """简化版Airflow DAG"""

    def __init__(self, dag_id, schedule=None, start_date=None, catchup=False):
        self.dag_id = dag_id
        self.schedule = schedule
        self.start_date = start_date or datetime.now()
        self.catchup = catchup
        self.tasks = {}
        self.run_history = []

    def add_task(self, task):
        self.tasks[task.task_id] = task
        return task

    def _topological_sort(self):
        """拓扑排序"""
        visited = set()
        order = []

        def dfs(task_id):
            if task_id in visited:
                return
            visited.add(task_id)
            task = self.tasks[task_id]
            for ds in task.downstream:
                if ds.task_id in self.tasks:
                    dfs(ds.task_id)
            order.append(task_id)

        for tid in self.tasks:
            dfs(tid)

        order.reverse()
        return order

    def run(self, context=None):
        """执行DAG"""
        context = context or {}
        context['dag_id'] = self.dag_id
        context['execution_date'] = datetime.now().isoformat()

        order = self._topological_sort()
        run_record = {
            'dag_id': self.dag_id,
            'execution_date': context['execution_date'],
            'tasks': [],
            'status': 'running',
        }

        print(f"\n  === DAG运行: {self.dag_id} ===")
        print(f"  执行顺序: {' -> '.join(order)}")

        for task_id in order:
            task = self.tasks[task_id]
            # 将上游结果传入context
            context['task_instance'] = task
            context['prev_results'] = {
                tid: self.tasks[tid].result for tid in order
                if self.tasks[tid].result is not None
            }

            try:
                t0 = time.time()
                task.execute(context)
                elapsed = time.time() - t0
                print(f"  [{task.status.upper()}] {task_id}: "
                      f"耗时 {task.duration:.3f}s")

                run_record['tasks'].append({
                    'task_id': task_id,
                    'status': task.status,
                    'duration': task.duration,
                })
            except Exception as e:
                print(f"  [FAILED] {task_id}: {e}")
                run_record['tasks'].append({
                    'task_id': task_id,
                    'status': 'failed',
                    'duration': 0,
                })
                run_record['status'] = 'failed'
                break

        if run_record['status'] != 'failed':
            run_record['status'] = 'success'

        run_record['total_duration'] = sum(t['duration'] for t in run_record['tasks'])
        self.run_history.append(run_record)
        return run_record

    def print_tree(self):
        """打印DAG树"""
        print(f"\n  DAG: {self.dag_id}")
        print(f"  调度: {self.schedule or '手动触发'}")
        for tid, task in self.tasks.items():
            ds = [d.task_id for d in task.downstream]
            ds_str = f' -> {ds}' if ds else ''
            print(f"    {tid}{ds_str}")


# ============================================================
# 3. 构建ML Pipeline DAG
# ============================================================
print("\n--- 3. 构建ML Pipeline DAG ---")


# 定义任务函数
def task_validate_data(context):
    """数据验证"""
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    assert df.isna().sum().sum() == 0, "存在空值"
    assert len(df) > 100, "数据量不足"
    return {'data': df, 'feature_names': data.feature_names}


def task_preprocess(context):
    """数据预处理"""
    prev = context.get('prev_results', {})
    validate_result = prev.get('validate_data')
    if validate_result is None:
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
    else:
        df = validate_result['data']

    X = df.drop('target', axis=1).values
    y = df['target'].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return {'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test}


def task_train(context):
    """模型训练"""
    prev = context.get('prev_results', {})
    preprocess_result = prev.get('preprocess')
    if preprocess_result is None:
        data = load_breast_cancer()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        X_train = preprocess_result['X_train']
        y_train = preprocess_result['y_train']
        X_test = preprocess_result['X_test']
        y_test = preprocess_result['y_test']

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    return {'model': model, 'X_test': X_test, 'y_test': y_test, 'X_train': X_train, 'y_train': y_train}


def task_evaluate(context):
    """模型评估"""
    prev = context.get('prev_results', {})
    train_result = prev.get('train')
    if train_result is None:
        return {'accuracy': 0}

    model = train_result['model']
    X_test = train_result['X_test']
    y_test = train_result['y_test']

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return {'accuracy': acc, 'passed': acc >= 0.9}


def task_deploy(context):
    """模型部署"""
    prev = context.get('prev_results', {})
    eval_result = prev.get('evaluate')
    if eval_result and eval_result.get('passed'):
        return {'deployed': True, 'version': 'v1.0'}
    return {'deployed': False}


def task_notify(context):
    """发送通知"""
    prev = context.get('prev_results', {})
    deploy_result = prev.get('deploy', {})
    eval_result = prev.get('evaluate', {})

    if deploy_result.get('deployed'):
        return {'notification': f"模型部署成功! accuracy={eval_result.get('accuracy', 0):.4f}"}
    return {'notification': "模型未部署"}


# 创建DAG
dag = AirflowDAG('ml_training_pipeline', schedule='@daily')

# 创建任务
t_validate = AirflowTask('validate_data', task_validate_data)
t_preprocess = AirflowTask('preprocess', task_preprocess)
t_train = AirflowTask('train', task_train)
t_evaluate = AirflowTask('evaluate', task_evaluate)
t_deploy = AirflowTask('deploy', task_deploy)
t_notify = AirflowTask('notify', task_notify)

dag.add_task(t_validate)
dag.add_task(t_preprocess)
dag.add_task(t_train)
dag.add_task(t_evaluate)
dag.add_task(t_deploy)
dag.add_task(t_notify)

# 设置依赖
t_validate >> t_preprocess >> t_train >> t_evaluate >> t_deploy >> t_notify

# 打印DAG结构
dag.print_tree()

# 运行DAG
dag.run()

# ============================================================
# 4. 多DAG管理
# ============================================================
print("\n--- 4. 多DAG管理 ---")

# 创建每日重训练DAG
daily_dag = AirflowDAG('daily_retrain', schedule='0 2 * * *')

t_daily_check = AirflowTask('check_performance', lambda ctx: {'needs_retrain': True})
t_daily_retrain = AirflowTask('retrain', lambda ctx: {'accuracy': 0.95})
t_daily_register = AirflowTask('register', lambda ctx: {'registered': True})

t_daily_check >> t_daily_retrain >> t_daily_register

daily_dag.add_task(t_daily_check)
daily_dag.add_task(t_daily_retrain)
daily_dag.add_task(t_daily_register)

daily_dag.print_tree()
daily_dag.run()

# 创建每周报告DAG
weekly_dag = AirflowDAG('weekly_report', schedule='0 9 * * 1')

t_weekly_collect = AirflowTask('collect_metrics', lambda ctx: {'metrics_collected': True})
t_weekly_report = AirflowTask('generate_report', lambda ctx: {'report_generated': True})
t_weekly_notify = AirflowTask('send_report', lambda ctx: {'report_sent': True})

t_weekly_collect >> t_weekly_report >> t_weekly_notify

weekly_dag.add_task(t_weekly_collect)
weekly_dag.add_task(t_weekly_report)
weekly_dag.add_task(t_weekly_notify)

weekly_dag.print_tree()
weekly_dag.run()

# ============================================================
# 5. 可视化
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# ML Pipeline DAG
ax = axes[0]
ax.set_xlim(-1, 7)
ax.set_ylim(-1, 7)
ax.axis('off')
ax.set_title('ML训练Pipeline DAG')

dag_tasks = [
    ('validate_data', 3, 5.5),
    ('preprocess', 3, 4.5),
    ('train', 3, 3.5),
    ('evaluate', 3, 2.5),
    ('deploy', 3, 1.5),
    ('notify', 3, 0.5),
]

task_colors = ['#BBDEFB', '#C8E6C9', '#FFE0B2', '#E1BEE7', '#FFCDD2', '#B2DFDB']

for i, (name, x, y) in enumerate(dag_tasks):
    ax.add_patch(plt.Rectangle((x - 1.5, y - 0.3), 3, 0.6,
                                facecolor=task_colors[i], edgecolor='#333', linewidth=1.5))
    ax.text(x, y, name, ha='center', va='center', fontsize=9)

for i in range(len(dag_tasks) - 1):
    x1, y1 = dag_tasks[i][1], dag_tasks[i][2]
    x2, y2 = dag_tasks[i+1][1], dag_tasks[i+1][2]
    ax.annotate('', xy=(x2, y2 + 0.3), xytext=(x1, y1 - 0.3),
                arrowprops=dict(arrowstyle='->', color='#666', lw=2))

# 任务执行甘特图
ax = axes[1]
if dag.run_history:
    run = dag.run_history[-1]
    tasks_info = run['tasks']
    cumulative = 0
    for i, t in enumerate(tasks_info):
        ax.barh(t['task_id'], t['duration'], left=cumulative,
                color=task_colors[i], edgecolor='#333')
        ax.text(cumulative + t['duration']/2, i, f"{t['duration']:.3f}s",
                ha='center', va='center', fontsize=8)
        cumulative += t['duration']
    ax.set_xlabel('时间 (秒)')
    ax.set_title('任务执行甘特图')

# 多DAG概览
ax = axes[2]
dags = [dag.dag_id, daily_dag.dag_id, weekly_dag.dag_id]
task_counts = [len(d.tasks) for d in [dag, daily_dag, weekly_dag]]
schedules = ['@daily', '0 2 * * *', '0 9 * * 1']

colors_dag = ['#2196F3', '#4CAF50', '#FF9800']
bars = ax.bar(dags, task_counts, color=colors_dag)
ax.set_ylabel('任务数')
ax.set_title('DAG概览')
for bar, schedule in zip(bars, schedules):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
            schedule, ha='center', fontsize=8, color='#666')

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W28/d3_airflow_ml.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存: d3_airflow_ml.png")

print("\n完成! Airflow for ML要点:")
print("  1. DAG: 有向无环图定义工作流")
print("  2. Task: DAG中的节点, 支持Python/Bash等")
print("  3. 依赖: >> 运算符定义任务执行顺序")
print("  4. 调度: Cron表达式定义定时执行")
print("  5. 重试: 任务失败自动重试机制")
