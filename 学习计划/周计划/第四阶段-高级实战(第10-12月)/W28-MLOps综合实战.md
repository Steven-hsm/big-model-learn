# 第28周 - MLOps综合实战

> **阶段**: MLOps综合实战
> **时间**: 工作日每晚2小时 + 周末6-8小时
> **前置知识**: 第25-27周数据工程、实验管理、模型监控与CI/CD

---

## 一、本周目标

1. 设计端到端ML平台架构，掌握各层技术选型与组件协作方式
2. 使用Airflow编排ML训练DAG，实现从数据检查到模型部署的自动化流水线
3. 理解GPU资源管理与弹性伸缩策略，掌握Ray分布式计算框架基础
4. 掌握Spot实例训练、模型缓存、推理弹性伸缩等成本优化手段
5. 理解ML系统安全与合规要求，实现API认证、数据脱敏和审计日志
6. 将前几周学到的所有MLOps组件集成到一个可运行的平台中

---

## 二、时间安排

### 工作日（每晚2小时）

| 日期 | 主题 | 时长 |
|------|------|------|
| Day 1 (周一) | 端到端ML平台架构设计 | 2h |
| Day 2 (周二) | Airflow编排 | 2h |
| Day 3 (周三) | GPU资源管理 | 2h |
| Day 4 (周四) | 成本优化 | 2h |
| Day 5 (周五) | 安全与合规 | 2h |

### 周末（6-8小时/天）

| 日期 | 主题 | 时长 |
|------|------|------|
| Day 6 (周六) | 综合集成 | 6-8h |
| Day 7 (周日) | 文档与Demo | 6-8h |

---

## 三、详细学习内容

### Day 1: 端到端ML平台架构设计 (2h)

#### ML平台四层架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    端到端ML平台架构                               │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              Monitoring Layer (监控层)                       │ │
│  │   Prometheus + Grafana / Evidently / 自定义告警规则          │ │
│  └──────────────────────────┬──────────────────────────────────┘ │
│                              │ 指标反馈                           │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              Serving Layer (服务层)                          │ │
│  │   Docker + Kubernetes / vLLM / Triton / FastAPI + Redis     │ │
│  └──────────────────────────┬──────────────────────────────────┘ │
│                              │ 模型分发                           │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              Training Layer (训练层)                         │ │
│  │   MLflow + Airflow / Ray / Weights & Biases / DVC Pipeline  │ │
│  └──────────────────────────┬──────────────────────────────────┘ │
│                              │ 数据供给                           │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              Data Layer (数据层)                             │ │
│  │   DVC + Feast / Great Expectations / Label Studio           │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

#### 各层技术选型对比

**Data Layer（数据层）**

| 组件 | 功能 | 工具选型 | 备选方案 |
|------|------|---------|---------|
| 数据版本管理 | 追踪数据集变更 | DVC | LakeFS, Delta Lake |
| 特征管理 | 统一离线/在线特征 | Feast | Tecton, Hopsworks |
| 数据质量 | 自动化质量检查 | Great Expectations | Soda, Deequ |
| 数据标注 | 标注平台 | Label Studio | Prodigy, CVAT |
| 数据存储 | 数据湖/仓库 | MinIO/S3 | GCS, Azure Blob |

**Training Layer（训练层）**

| 组件 | 功能 | 工具选型 | 备选方案 |
|------|------|---------|---------|
| 实验管理 | 追踪参数和指标 | MLflow | W&B, Neptune |
| 工作流编排 | DAG调度 | Airflow | Prefect, Dagster |
| 分布式训练 | 多GPU/多节点 | Ray | DeepSpeed, Horovod |
| Pipeline定义 | 可复现流水线 | DVC Pipeline | Kubeflow, ZenML |
| 超参搜索 | 自动调参 | Optuna | Ray Tune, SigOpt |

**Serving Layer（服务层）**

| 组件 | 功能 | 工具选型 | 备选方案 |
|------|------|---------|---------|
| 容器化 | 环境隔离 | Docker | Podman |
| 编排调度 | 弹性伸缩 | Kubernetes | Docker Compose(开发) |
| LLM推理 | 高性能推理 | vLLM | TGI, Triton |
| API服务 | REST接口 | FastAPI | Flask, gRPC |
| 缓存 | 推理加速 | Redis | Memcached |

**Monitoring Layer（监控层）**

| 组件 | 功能 | 工具选型 | 备选方案 |
|------|------|---------|---------|
| 指标采集 | 时序数据 | Prometheus | Datadog |
| 可视化 | 监控面板 | Grafana | Kibana |
| 数据漂移 | 分布检测 | Evidently | NannyML |
| 告警管理 | 通知路由 | AlertManager | PagerDuty |

#### 架构设计原则

```
1. 模块化：每个组件可独立替换和升级
2. 可复现：任何实验可以通过DVC+MLflow完全复现
3. 自动化：从训练到部署的流程通过Airflow自动化
4. 可观测：所有组件的状态通过Prometheus+Grafana监控
5. 弹性：根据流量自动扩缩容
6. 安全：认证+授权+审计覆盖所有操作
```

---

### Day 2: Airflow编排 (2h)

#### Airflow安装与初始化

```bash
# 方式1：pip安装（开发环境）
pip install apache-airflow
airflow db init
airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com

# 启动Web服务器
airflow webserver --port 8080

# 启动调度器（另一个终端）
airflow scheduler

# 方式2：Docker Compose（推荐生产环境）
curl -LfO 'https://airflow.apache.org/docs/apache-airflow/2.8.0/docker-compose.yaml'
mkdir -p ./dags ./logs ./plugins
docker-compose up -d
```

#### DAG定义基础

```python
# dags/ml_training_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# 默认参数
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'email': ['ml-alerts@company.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# 定义DAG
with DAG(
    dag_id='ml_training_pipeline',
    default_args=default_args,
    description='端到端ML训练流水线',
    schedule_interval='0 2 * * *',  # 每天凌晨2点运行
    start_date=datetime(2024, 1, 1),
    catchup=False,  # 不补跑历史任务
    tags=['ml', 'training'],
) as dag:

    # Task 1: 数据检查
    def check_data_quality(**kwargs):
        """检查数据质量"""
        import pandas as pd
        from great_expectations.dataset import PandasDataset

        df = pd.read_csv('/data/raw/latest.csv')
        dataset = PandasDataset(df)

        # 检查行数
        assert len(df) > 100, f"数据量不足: {len(df)}"

        # 检查缺失值
        null_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        assert null_ratio < 0.1, f"缺失值比例过高: {null_ratio:.2%}"

        logger.info(f"数据质量检查通过: {len(df)}行, {len(df.columns)}列")
        # 通过XCom传递数据路径
        kwargs['ti'].xcom_push(key='data_path', value='/data/raw/latest.csv')
        kwargs['ti'].xcom_push(key='row_count', value=len(df))

    check_data = PythonOperator(
        task_id='check_data_quality',
        python_callable=check_data_quality,
    )

    # Task 2: 数据预处理
    def preprocess_data(**kwargs):
        """数据预处理"""
        import pandas as pd
        from sklearn.model_selection import train_test_split

        # 从XCom获取上游数据
        data_path = kwargs['ti'].xcom_pull(
            task_ids='check_data_quality', key='data_path'
        )
        row_count = kwargs['ti'].xcom_pull(
            task_ids='check_data_quality', key='row_count'
        )
        logger.info(f"处理数据: {data_path}, 行数: {row_count}")

        df = pd.read_csv(data_path)

        # 清洗和特征工程
        df = df.drop_duplicates()
        df = df.fillna(df.median(numeric_only=True))

        # 切分数据
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        train_df.to_csv('/data/processed/train.csv', index=False)
        test_df.to_csv('/data/processed/test.csv', index=False)

        kwargs['ti'].xcom_push(key='train_size', value=len(train_df))
        kwargs['ti'].xcom_push(key='test_size', value=len(test_df))
        logger.info(f"预处理完成: train={len(train_df)}, test={len(test_df)}")

    preprocess = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data,
    )

    # Task 3: 模型训练
    def train_model(**kwargs):
        """模型训练"""
        import mlflow
        import mlflow.sklearn
        from sklearn.ensemble import RandomForestClassifier
        import pandas as pd
        import json

        mlflow.set_tracking_uri('http://localhost:5000')
        mlflow.set_experiment('production-models')

        train_df = pd.read_csv('/data/processed/train.csv')
        X_train = train_df.drop(columns=['target'])
        y_train = train_df['target']

        with mlflow.start_run(run_name=f'train_{datetime.now().strftime("%Y%m%d_%H%M")}'):
            # 训练模型
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)

            # 记录参数
            mlflow.log_params({
                'n_estimators': 200,
                'max_depth': 15,
                'train_size': len(X_train),
            })

            # 保存模型
            mlflow.sklearn.log_model(model, 'model')

            run_id = mlflow.active_run().info.run_id
            kwargs['ti'].xcom_push(key='run_id', value=run_id)
            logger.info(f"训练完成, run_id={run_id}")

    train = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
    )

    # Task 4: 模型评估
    def evaluate_model(**kwargs):
        """模型评估"""
        import mlflow
        import mlflow.sklearn
        from sklearn.metrics import accuracy_score, f1_score, classification_report
        import pandas as pd
        import json

        run_id = kwargs['ti'].xcom_pull(task_ids='train_model', key='run_id')

        # 加载模型
        model_uri = f'runs:/{run_id}/model'
        model = mlflow.sklearn.load_model(model_uri)

        # 评估
        test_df = pd.read_csv('/data/processed/test.csv')
        X_test = test_df.drop(columns=['target'])
        y_test = test_df['target']
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        # 记录指标到MLflow
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics({
                'test_accuracy': accuracy,
                'test_f1': f1,
            })

        # 质量门禁
        assert accuracy >= 0.85, f"准确率不达标: {accuracy:.4f} < 0.85"
        assert f1 >= 0.80, f"F1不达标: {f1:.4f} < 0.80"

        kwargs['ti'].xcom_push(key='accuracy', value=float(accuracy))
        kwargs['ti'].xcom_push(key='f1', value=float(f1))
        logger.info(f"评估通过: accuracy={accuracy:.4f}, f1={f1:.4f}")

    evaluate = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model,
    )

    # Task 5: 模型注册
    def register_model(**kwargs):
        """注册模型到Model Registry"""
        import mlflow

        run_id = kwargs['ti'].xcom_pull(task_ids='train_model', key='run_id')
        accuracy = kwargs['ti'].xcom_pull(task_ids='evaluate_model', key='accuracy')

        # 注册模型
        model_name = 'production-classifier'
        model_uri = f'runs:/{run_id}/model'

        result = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
        )

        # 过渡到Staging
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=result.version,
            stage='Staging',
        )

        logger.info(f"模型已注册: {model_name} v{result.version} (accuracy={accuracy:.4f})")

    register = PythonOperator(
        task_id='register_model',
        python_callable=register_model,
    )

    # Task 6: 部署通知
    deploy_notification = BashOperator(
        task_id='deploy_notification',
        bash_command='echo "模型已就绪，等待部署审批" && curl -X POST https://hooks.slack.example.com/notify -d "ML模型已更新到Staging"',
    )

    # 定义任务依赖
    check_data >> preprocess >> train >> evaluate >> register >> deploy_notification
```

#### XCom任务间通信

```python
"""
XCom (Cross-Communication): Airflow任务间传递数据

限制：
- 默认使用数据库后端，适合小数据（< 48KB）
- 大数据应使用共享存储（S3/GCS/文件系统）
- 只传递元信息和引用，不传递数据本身

推（Push）：
    kwargs['ti'].xcom_push(key='my_key', value='my_value')

拉（Pull）：
    value = kwargs['ti'].xcom_pull(task_ids='upstream_task', key='my_key')
"""

# XCom最佳实践
def upstream_task(**kwargs):
    """上游任务：传递元信息"""
    # 不要传递大对象，只传递路径或引用
    kwargs['ti'].xcom_push(key='model_path', value='/models/latest/model.pkl')
    kwargs['ti'].xcom_push(key='metrics', value={'accuracy': 0.92, 'f1': 0.89})
    return  # 返回值也会自动通过XCom传递（key='return_value'）

def downstream_task(**kwargs):
    """下游任务：获取上游结果"""
    model_path = kwargs['ti'].xcom_pull(
        task_ids='upstream_task', key='model_path'
    )
    metrics = kwargs['ti'].xcom_pull(
        task_ids='upstream_task', key='metrics'
    )
    print(f"模型路径: {model_path}")
    print(f"指标: {metrics}")
```

---

### Day 3: GPU资源管理 (2h)

#### nvidia-smi监控

```bash
# 基础监控
nvidia-smi

# 持续监控（每2秒刷新）
nvidia-smi -l 2

# 只看GPU利用率
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv

# 查看GPU上的进程
nvidia-smi pmon -c 5  # 刷新5次

# 查看详细信息
nvidia-smi --query-gpu=index,name,temperature.gpu,power.draw,power.limit --format=csv

# 常用监控脚本
watch -n 1 nvidia-smi  # 每秒刷新GPU状态
```

```python
# Python监控GPU
import subprocess
import json

def get_gpu_stats():
    """获取GPU状态信息"""
    result = subprocess.run(
        [
            'nvidia-smi',
            '--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw',
            '--format=csv,noheader,nounits',
        ],
        capture_output=True,
        text=True,
    )

    gpus = []
    for line in result.stdout.strip().split('\n'):
        parts = [p.strip() for p in line.split(',')]
        gpus.append({
            'index': int(parts[0]),
            'name': parts[1],
            'memory_used_mb': float(parts[2]),
            'memory_total_mb': float(parts[3]),
            'gpu_util_pct': float(parts[4]),
            'temperature_c': float(parts[5]),
            'power_draw_w': float(parts[6]),
        })
    return gpus

# 使用
stats = get_gpu_stats()
for gpu in stats:
    mem_pct = gpu['memory_used_mb'] / gpu['memory_total_mb'] * 100
    print(f"GPU {gpu['index']}: {gpu['name']}")
    print(f"  显存: {gpu['memory_used_mb']:.0f}/{gpu['memory_total_mb']:.0f}MB ({mem_pct:.1f}%)")
    print(f"  利用率: {gpu['gpu_util_pct']}%")
    print(f"  温度: {gpu['temperature_c']}C")
    print(f"  功耗: {gpu['power_draw_w']}W")
```

#### Ray分布式计算框架

```bash
# 安装Ray
pip install ray

# 启动本地Ray集群（单机多核）
ray start --head --port=6379

# 查看集群状态
ray status

# 停止集群
ray stop
```

```python
# Ray基础用法
import ray
import time

# 初始化Ray
ray.init(address='auto')  # 连接到已有集群
# 或 ray.init()  # 启动本地集群

# 普通函数
def normal_process(data):
    """普通串行处理"""
    results = []
    for item in data:
        # 模拟耗时计算
        time.sleep(1)
        results.append(item ** 2)
    return results

# Ray远程函数
@ray.remote
def ray_process(item):
    """Ray远程处理：会被分发到集群中的不同节点"""
    time.sleep(1)
    return item ** 2

# 对比
data = list(range(8))

# 串行执行：~8秒
start = time.time()
normal_result = normal_process(data)
print(f"串行耗时: {time.time() - start:.2f}s")

# Ray并行执行：~2秒（假设有4个CPU）
start = time.time()
futures = [ray_process.remote(item) for item in data]
ray_result = ray.get(futures)
print(f"Ray并行耗时: {time.time() - start:.2f}s")
```

```python
# Ray分布式训练示例
import ray
from ray import train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

def train_func(config):
    """分布式训练函数"""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    # 获取分布式信息
    world_size = train.get_context().get_world_size()
    rank = train.get_context().get_world_rank()

    # 创建模型
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 2),
    )

    # 准备数据
    X = torch.randn(1000, 10)
    y = torch.randint(0, 2, (1000,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Ray Train会自动处理DDP包装
    model = train.torch.prepare_model(model)

    # 训练循环
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', 0.001))
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        train.report({'loss': avg_loss, 'epoch': epoch})

# 配置并启动分布式训练
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config={'lr': 0.001},
    scaling_config=ScalingConfig(
        num_workers=2,           # 使用2个worker
        use_gpu=True,            # 使用GPU
        resources_per_worker={'GPU': 1},
    ),
)

result = trainer.fit()
print(f"最佳checkpoint: {result.checkpoint}")
print(f"最终指标: {result.metrics}")
```

#### SLURM概念

```
SLURM (Simple Linux Utility for Resource Management)
- HPC集群常用的工作负载管理器
- AI训练集群中也广泛使用

核心概念：
┌─────────────────────────────────────────────────┐
│  SLURM集群架构                                    │
│                                                   │
│  ┌──────────┐    ┌───────────────────────────┐   │
│  │ slurmctld │    │ slurmd (计算节点1)         │   │
│  │ (控制守护) │───▶│   GPU 0, GPU 1            │   │
│  │ 调度+管理  │    ├───────────────────────────┤   │
│  └──────────┘    │ slurmd (计算节点2)         │   │
│                   │   GPU 0, GPU 1            │   │
│                   └───────────────────────────┘   │
└─────────────────────────────────────────────────┘

常用命令：
sinfo          # 查看集群和分区状态
squeue -u $USER  # 查看自己的任务
srun           # 提交交互式任务
sbatch         # 提交批处理任务
scancel        # 取消任务
sacct          # 查看已完成任务的资源使用

提交脚本示例 (train.slurm)：
#!/bin/bash
#SBATCH --job-name=llm-finetune
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4          # 申请4块GPU
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00       # 最长运行24小时
#SBATCH --output=logs/%j.out

python train.py --config configs/finetune.yaml

# 提交
sbatch train.slurm
```

#### 弹性伸缩策略

```python
# autoscaler_config.py
"""
Kubernetes HPA (Horizontal Pod Autoscaler) 配置
根据负载自动扩缩推理服务副本数
"""

# 推理服务部署配置
k8s_deployment = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-inference
spec:
  replicas: 2  # 初始副本数
  selector:
    matchLabels:
      app: llm-inference
  template:
    metadata:
      labels:
        app: llm-inference
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        resources:
          requests:
            nvidia.com/gpu: 1
          limits:
            nvidia.com/gpu: 1
        env:
        - name: MODEL_NAME
          value: "meta-llama/Llama-2-7b-chat-hf"
"""

# HPA自动伸缩配置
hpa_config = """
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-inference-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-inference
  minReplicas: 2        # 最少2个副本
  maxReplicas: 10       # 最多10个副本
  metrics:
  - type: Resource
    resource:
      name: nvidia.com/gpu-utilization
      target:
        type: Utilization
        averageUtilization: 70  # GPU利用率>70%时扩容
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "100"  # 每个Pod QPS>100时扩容
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60   # 扩容冷却时间
      policies:
      - type: Percent
        value: 100        # 每次最多翻倍
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300  # 缩容冷却时间5分钟
      policies:
      - type: Percent
        value: 25         # 每次最多缩25%
        periodSeconds: 60
"""
```

---

### Day 4: 成本优化 (2h)

#### Spot实例训练

```python
# spot_training.py
"""
Spot实例：利用云厂商的闲置计算资源，价格可低至按需实例的20-30%
风险：随时可能被回收

关键策略：
1. 频繁保存checkpoint
2. 实现断点续训
3. 使用多可用区分散风险
"""

import os
import torch
from pathlib import Path

class SpotTrainingManager:
    """Spot实例训练管理器"""

    def __init__(self, checkpoint_dir='/tmp/checkpoints'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.latest_checkpoint = self._find_latest_checkpoint()

    def _find_latest_checkpoint(self):
        """查找最新的checkpoint"""
        checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_*.pt'))
        if checkpoints:
            latest = checkpoints[-1]
            print(f"找到checkpoint: {latest}")
            return latest
        return None

    def save_checkpoint(self, model, optimizer, epoch, metrics, step):
        """保存checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
        }
        path = self.checkpoint_dir / f'checkpoint_epoch{epoch}_step{step}.pt'
        torch.save(checkpoint, path)
        print(f"Checkpoint已保存: {path}")

        # 清理旧checkpoint，只保留最近3个
        all_checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_*.pt'))
        for old_ckpt in all_checkpoints[:-3]:
            old_ckpt.unlink()
            print(f"清理旧checkpoint: {old_ckpt}")

    def load_checkpoint(self, model, optimizer=None):
        """加载checkpoint，实现断点续训"""
        if self.latest_checkpoint is None:
            print("未找到checkpoint，从头开始训练")
            return 0, 0, {}

        checkpoint = torch.load(self.latest_checkpoint, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epoch = checkpoint['epoch']
        step = checkpoint['step']
        metrics = checkpoint.get('metrics', {})
        print(f"从checkpoint恢复: epoch={epoch}, step={step}")
        return epoch, step, metrics

    def train_with_checkpoint(self, model, optimizer, train_loader, num_epochs, save_every_n_steps=100):
        """带checkpoint的训练循环"""
        # 尝试恢复
        start_epoch, start_step, metrics = self.load_checkpoint(model, optimizer)
        global_step = start_step

        for epoch in range(start_epoch, num_epochs):
            model.train()
            for batch_idx, batch in enumerate(train_loader):
                # 如果是恢复的，跳过已训练的batch
                if epoch == start_epoch and batch_idx * len(batch) < start_step:
                    continue

                optimizer.zero_grad()
                loss = model(**batch).loss
                loss.backward()
                optimizer.step()
                global_step += 1

                # 频繁保存checkpoint
                if global_step % save_every_n_steps == 0:
                    self.save_checkpoint(
                        model, optimizer, epoch,
                        {'loss': loss.item()}, global_step
                    )

            print(f"Epoch {epoch} 完成, loss={loss.item():.4f}")

        return model


# AWS Spot实例训练脚本
aws_spot_script = """
# train_on_spot.sh - AWS Spot实例训练脚本
# 使用AWS SageMaker Spot实例

import sagemaker
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point='train.py',
    role=sagemaker.get_execution_role(),
    instance_type='ml.g4dn.xlarge',
    instance_count=1,
    framework_version='2.1.0',
    py_version='py310',
    # Spot实例配置
    use_spot_instances=True,          # 使用Spot实例
    max_run=24 * 60 * 60,             # 最大运行时间24小时
    max_wait=48 * 60 * 60,            # 最大等待时间48小时（含排队）
    checkpoint_s3_uri='s3://my-bucket/checkpoints/',  # checkpoint保存到S3
)

estimator.fit({'train': 's3://my-bucket/data/'})
"""
```

#### 模型缓存LRU

```python
# model_cache.py
"""模型缓存：避免重复加载模型，加速推理启动"""

from collections import OrderedDict
import threading
import time

class ModelCache:
    """LRU模型缓存"""

    def __init__(self, max_size=5, max_memory_gb=16):
        self.cache = OrderedDict()  # 有序字典，维护访问顺序
        self.max_size = max_size
        self.max_memory_bytes = max_memory_gb * 1024**3
        self.current_memory = 0
        self.lock = threading.Lock()
        self.stats = {'hits': 0, 'misses': 0}

    def get(self, model_name):
        """获取模型（命中则移到末尾表示最近使用）"""
        with self.lock:
            if model_name in self.cache:
                # 命中：移到末尾（最近使用）
                self.cache.move_to_end(model_name)
                self.stats['hits'] += 1
                return self.cache[model_name]
            self.stats['misses'] += 1
            return None

    def put(self, model_name, model, model_size_bytes=0):
        """缓存模型"""
        with self.lock:
            if model_name in self.cache:
                # 已存在，更新
                self.cache.move_to_end(model_name)
                return

            # 淘汰旧模型直到有足够空间
            while (len(self.cache) >= self.max_size or
                   self.current_memory + model_size_bytes > self.max_memory_bytes):
                if not self.cache:
                    break
                # 淘汰最久未使用的
                evicted_name, evicted_model = self.cache.popitem(last=False)
                evicted_size = getattr(evicted_model, '_cache_size', 0)
                self.current_memory -= evicted_size
                print(f"淘汰模型: {evicted_name}")

            self.cache[model_name] = model
            self.current_memory += model_size_bytes
            print(f"缓存模型: {model_name} ({model_size_bytes / 1024**3:.2f}GB)")

    def get_stats(self):
        """获取缓存统计"""
        total = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / max(total, 1)
        return {
            'cache_size': len(self.cache),
            'memory_used_gb': self.current_memory / 1024**3,
            'hit_rate': hit_rate,
            **self.stats,
        }

# 使用示例
cache = ModelCache(max_size=3, max_memory_gb=8)

# 模拟加载模型
model = cache.get("llama-7b")
if model is None:
    # 缓存未命中，加载模型
    print("加载模型 llama-7b...")
    # model = load_model("llama-7b")
    class FakeModel:
        _cache_size = 14 * 1024**3  # 14GB
    model = FakeModel()
    cache.put("llama-7b", model, model._cache_size)
```

#### 推理弹性伸缩与成本监控

```python
# cost_monitor.py
"""推理成本监控"""

import time
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class CostConfig:
    """成本配置"""
    gpu_cost_per_hour: float = 2.5       # GPU实例每小时成本(美元)
    cpu_cost_per_hour: float = 0.1        # CPU实例每小时成本
    spot_discount: float = 0.7            # Spot实例折扣

@dataclass
class ServiceMetrics:
    """服务指标"""
    total_requests: int = 0
    total_inference_time: float = 0.0
    total_tokens: int = 0
    gpu_hours: float = 0.0
    start_time: float = field(default_factory=time.time)

class CostMonitor:
    """推理成本监控器"""

    def __init__(self, config: CostConfig):
        self.config = config
        self.metrics = ServiceMetrics()

    def record_request(self, inference_time: float, tokens: int, gpu_time: float = 0):
        """记录单次推理"""
        self.metrics.total_requests += 1
        self.metrics.total_inference_time += inference_time
        self.metrics.total_tokens += tokens
        self.metrics.gpu_hours += gpu_time / 3600

    def get_cost_report(self) -> Dict:
        """生成成本报告"""
        hours = (time.time() - self.metrics.start_time) / 3600
        gpu_cost = self.metrics.gpu_hours * self.config.gpu_cost_per_hour

        cost_per_1k_tokens = (gpu_cost / max(self.metrics.total_tokens, 1)) * 1000
        avg_latency = self.metrics.total_inference_time / max(self.metrics.total_requests, 1)

        return {
            'total_requests': self.metrics.total_requests,
            'total_tokens': self.metrics.total_tokens,
            'avg_latency_ms': avg_latency * 1000,
            'gpu_hours': self.metrics.gpu_hours,
            'total_cost_usd': gpu_cost,
            'cost_per_1k_tokens_usd': cost_per_1k_tokens,
            'spot_savings_usd': gpu_cost * self.config.spot_discount,
        }

    def recommend_optimization(self) -> list:
        """根据指标推荐优化策略"""
        report = self.get_cost_report()
        recommendations = []

        if report['avg_latency_ms'] > 2000:
            recommendations.append("延迟偏高，建议：开启KV Cache / 使用量化(INT8) / 增加GPU")

        if report['cost_per_1k_tokens_usd'] > 0.01:
            recommendations.append("成本偏高，建议：使用Spot实例 / 模型蒸馏 / Batch推理")

        utilization = report['gpu_hours'] / max((time.time() - self.metrics.start_time) / 3600, 0.01)
        if utilization < 0.3:
            recommendations.append("GPU利用率低，建议：缩容 / 合并推理服务 / 使用Serverless")

        return recommendations
```

---

### Day 5: 安全与合规 (2h)

#### API Key认证与RBAC

```python
# auth.py
"""API Key认证与RBAC权限控制"""

from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from functools import wraps
from typing import List, Optional
import hashlib
import secrets
import time

app = FastAPI()

# API Key管理
api_key_header = APIKeyHeader(name='X-API-Key')

# 模拟数据库
API_KEYS_DB = {}  # {api_key_hash: {user_id, role, permissions}}
USERS_DB = {}

class Role:
    ADMIN = 'admin'
    DATA_SCIENTIST = 'data_scientist'
    ENGINEER = 'engineer'
    VIEWER = 'viewer'

class Permission:
    TRAIN = 'train'
    DEPLOY = 'deploy'
    VIEW_METRICS = 'view_metrics'
    MANAGE_USERS = 'manage_users'
    DELETE_MODEL = 'delete_model'

# 角色-权限映射
ROLE_PERMISSIONS = {
    Role.ADMIN: [Permission.TRAIN, Permission.DEPLOY, Permission.VIEW_METRICS,
                 Permission.MANAGE_USERS, Permission.DELETE_MODEL],
    Role.DATA_SCIENTIST: [Permission.TRAIN, Permission.VIEW_METRICS],
    Role.ENGINEER: [Permission.DEPLOY, Permission.VIEW_METRICS],
    Role.VIEWER: [Permission.VIEW_METRICS],
}

def generate_api_key(user_id: str, role: str) -> str:
    """生成API Key"""
    api_key = f"ml-{secrets.token_hex(32)}"
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()

    API_KEYS_DB[key_hash] = {
        'user_id': user_id,
        'role': role,
        'permissions': ROLE_PERMISSIONS.get(role, []),
        'created_at': time.time(),
    }

    return api_key  # 只在创建时返回一次

def verify_api_key(api_key: str = Security(api_key_header)) -> dict:
    """验证API Key"""
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()

    if key_hash not in API_KEYS_DB:
        raise HTTPException(status_code=401, detail='Invalid API Key')

    key_info = API_KEYS_DB[key_hash]

    # 检查是否过期（30天）
    if time.time() - key_info['created_at'] > 30 * 24 * 3600:
        raise HTTPException(status_code=401, detail='API Key expired')

    return key_info

def require_permission(permission: str):
    """权限检查装饰器"""
    def decorator(func):
        async def wrapper(*args, current_user: dict = Depends(verify_api_key), **kwargs):
            if permission not in current_user['permissions']:
                raise HTTPException(
                    status_code=403,
                    detail=f'Permission denied: {permission}'
                )
            return await func(*args, current_user=current_user, **kwargs)
        wrapper.__name__ = func.__name__
        return wrapper
    return decorator

# API端点
@app.post('/api/v1/train')
@require_permission(Permission.TRAIN)
async def train_model(config: dict, current_user: dict = Depends(verify_api_key)):
    """训练模型（需要train权限）"""
    return {'status': 'training', 'user': current_user['user_id']}

@app.post('/api/v1/deploy')
@require_permission(Permission.DEPLOY)
async def deploy_model(model_name: str, current_user: dict = Depends(verify_api_key)):
    """部署模型（需要deploy权限）"""
    return {'status': 'deploying', 'model': model_name}

@app.get('/api/v1/metrics')
async def get_metrics(current_user: dict = Depends(verify_api_key)):
    """查看指标（所有角色可访问）"""
    return {'accuracy': 0.92, 'qps': 150}
```

#### 数据隐私脱敏

```python
# data_anonymizer.py
"""数据隐私脱敏工具"""

import re
import hashlib
from typing import Dict, List

class DataAnonymizer:
    """数据脱敏处理器"""

    def __init__(self, salt: str = 'your-secret-salt'):
        self.salt = salt

    def hash_pii(self, value: str) -> str:
        """PII字段哈希（不可逆）"""
        return hashlib.sha256(f'{value}{self.salt}'.encode()).hexdigest()[:16]

    def mask_phone(self, phone: str) -> str:
        """手机号脱敏: 13812345678 → 138****5678"""
        return re.sub(r'(\d{3})\d{4}(\d{4})', r'\1****\2', phone)

    def mask_email(self, email: str) -> str:
        """邮箱脱敏: test@example.com → t***@example.com"""
        parts = email.split('@')
        if len(parts) == 2:
            name = parts[0]
            masked = name[0] + '***' if len(name) > 1 else '***'
            return f'{masked}@{parts[1]}'
        return '***'

    def mask_id_card(self, id_card: str) -> str:
        """身份证脱敏: 110101199001011234 → 110101****1234"""
        if len(id_card) >= 14:
            return id_card[:6] + '********' + id_card[-4:]
        return '***'

    def anonymize_text(self, text: str) -> str:
        """文本自动脱敏：识别并替换PII"""
        # 手机号
        text = re.sub(r'1[3-9]\d{9}', '[PHONE]', text)
        # 邮箱
        text = re.sub(r'[\w.-]+@[\w.-]+\.\w+', '[EMAIL]', text)
        # 身份证号
        text = re.sub(r'\d{17}[\dXx]', '[ID_CARD]', text)
        # 银行卡号
        text = re.sub(r'\d{16,19}', '[BANK_CARD]', text)
        # IP地址
        text = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '[IP]', text)
        return text

    def anonymize_record(self, record: Dict, pii_fields: List[str]) -> Dict:
        """结构化数据脱敏"""
        anonymized = record.copy()
        for field in pii_fields:
            if field in anonymized:
                anonymized[field] = self.hash_pii(str(anonymized[field]))
        return anonymized

# 使用示例
anonymizer = DataAnonymizer()

# 文本脱敏
text = "用户张三的手机号是13812345678，邮箱test@example.com，身份证110101199001011234"
print(anonymizer.anonymize_text(text))
# 输出: 用户张三的手机号是[PHONE]，邮箱[EMAIL]，身份证[ID_CARD]

# 结构化数据脱敏
user_record = {
    'user_id': '12345',
    'name': '张三',
    'phone': '13812345678',
    'email': 'test@example.com',
    'purchase_amount': 999.99,
}
anonymized = anonymizer.anonymize_record(user_record, pii_fields=['user_id', 'name', 'phone', 'email'])
print(anonymized)
# 输出: {'user_id': 'a1b2c3d4...', 'name': 'e5f6g7h8...', ...}
```

#### 审计日志

```python
# audit_logger.py
"""审计日志系统"""

import logging
import json
import time
from datetime import datetime
from typing import Optional
from fastapi import Request

class AuditLogger:
    """审计日志记录器"""

    def __init__(self, log_file: str = 'logs/audit.jsonl'):
        self.logger = logging.getLogger('audit')
        self.logger.setLevel(logging.INFO)

        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)

    def log_event(
        self,
        action: str,
        user_id: str,
        resource_type: str,
        resource_id: str,
        details: Optional[dict] = None,
        status: str = 'success',
        request: Optional[Request] = None,
    ):
        """记录审计事件"""
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'action': action,           # train, deploy, predict, delete, login
            'user_id': user_id,
            'resource_type': resource_type,  # model, dataset, experiment
            'resource_id': resource_id,
            'status': status,           # success, failure, denied
            'details': details or {},
        }

        if request:
            event['client_ip'] = request.client.host if request.client else None
            event['user_agent'] = request.headers.get('user-agent')

        self.logger.info(json.dumps(event, ensure_ascii=False))

# FastAPI中间件集成
from fastapi import FastAPI

app = FastAPI()
audit = AuditLogger()

@app.middleware('http')
async def audit_middleware(request: Request, call_next):
    """自动记录所有API请求"""
    start_time = time.time()

    response = await call_next(request)

    duration = time.time() - start_time

    # 记录到审计日志
    audit.log_event(
        action='api_request',
        user_id=request.headers.get('x-user-id', 'anonymous'),
        resource_type='api',
        resource_id=request.url.path,
        details={
            'method': request.method,
            'status_code': response.status_code,
            'duration_ms': round(duration * 1000, 2),
        },
        status='success' if response.status_code < 400 else 'failure',
        request=request,
    )

    return response
```

#### 对抗攻击防御

```python
# adversarial_defense.py
"""对抗攻击防御"""

class AdversarialDefense:
    """LLM对抗攻击防御"""

    def __init__(self):
        self.blocked_patterns = [
            # 提示注入
            r'ignore\s+(all\s+)?previous\s+instructions',
            r'forget\s+(all\s+)?previous',
            r'you\s+are\s+now\s+a',
            r'system\s*:\s*',
            # 信息泄露
            r'(show|reveal|display)\s+(your|the)\s+(prompt|system|instructions)',
            r'(what|how)\s+(are|is)\s+you\s+(programmed|configured)',
            # 越狱
            r'DAN\s+mode',
            r'jailbreak',
        ]

    def check_input(self, user_input: str) -> dict:
        """检查用户输入是否为对抗攻击"""
        import re

        threats = []
        for pattern in self.blocked_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                threats.append(f'匹配规则: {pattern}')

        # 检查输入长度异常
        if len(user_input) > 10000:
            threats.append('输入长度异常 (>10000字符)')

        # 检查特殊字符比例
        special_chars = sum(1 for c in user_input if not c.isalnum() and c not in ' .,!?;:')
        if special_chars / max(len(user_input), 1) > 0.5:
            threats.append('特殊字符比例异常')

        return {
            'is_safe': len(threats) == 0,
            'threats': threats,
            'input_length': len(user_input),
        }

    def sanitize_input(self, user_input: str) -> str:
        """清理用户输入"""
        # 移除控制字符
        import re
        sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', user_input)
        # 限制长度
        sanitized = sanitized[:5000]
        return sanitized

    def check_output(self, model_output: str) -> dict:
        """检查模型输出是否泄露敏感信息"""
        sensitive_patterns = [
            r'api[_-]?key\s*[=:]\s*\S+',
            r'password\s*[=:]\s*\S+',
            r'secret\s*[=:]\s*\S+',
            r'token\s*[=:]\s*\S+',
        ]

        leaked = []
        for pattern in sensitive_patterns:
            if re.search(pattern, model_output, re.IGNORECASE):
                leaked.append(pattern)

        return {
            'is_safe': len(leaked) == 0,
            'leaked_patterns': leaked,
        }
```

---

### Day 6: 综合集成 (6-8h)

#### 完整docker-compose.yml

```yaml
# docker-compose.yml - MLOps平台一键部署
version: '3.8'

services:
  # ===== Data Layer =====
  minio:
    image: minio/minio:latest
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes:
      - minio_data:/data
    command: server /data --console-address ":9001"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  postgres:
    image: postgres:15-alpine
    ports:
      - "5432:5432"
    environment:
      POSTGRES_USER: mlops
      POSTGRES_PASSWORD: mlops
      POSTGRES_DB: mlops
    volumes:
      - postgres_data:/var/lib/postgresql/data

  # ===== Training Layer =====
  mlflow:
    image: python:3.10-slim
    ports:
      - "5000:5000"
    environment:
      - MLFLOW_S3_ENDPOINT_URL=http://minio:9000
      - AWS_ACCESS_KEY_ID=minioadmin
      - AWS_SECRET_ACCESS_KEY=minioadmin
    volumes:
      - ./mlflow:/mlflow
    command: >
      bash -c "pip install mlflow boto3 &&
               mlflow server
               --host 0.0.0.0
               --port 5000
               --backend-store-uri postgresql://mlops:mlops@postgres:5432/mlops
               --default-artifact-root s3://mlflow/artifacts
               --serve-artifacts"
    depends_on:
      - postgres
      - minio

  airflow-webserver:
    image: apache/airflow:2.8.0
    ports:
      - "8080:8080"
    environment:
      - AIRFLOW__CORE__EXECUTOR=LocalExecutor
      - AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql://mlops:mlops@postgres:5432/airflow
      - AIRFLOW__CORE__LOAD_EXAMPLES=False
    volumes:
      - ./dags:/opt/airflow/dags
      - ./data:/data
    command: >
      bash -c "airflow db init &&
               airflow users create --username admin --password admin
               --firstname Admin --lastname User --role Admin --email admin@example.com || true &&
               airflow webserver"
    depends_on:
      - postgres

  airflow-scheduler:
    image: apache/airflow:2.8.0
    environment:
      - AIRFLOW__CORE__EXECUTOR=LocalExecutor
      - AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql://mlops:mlops@postgres:5432/airflow
    volumes:
      - ./dags:/opt/airflow/dags
      - ./data:/data
    command: airflow scheduler
    depends_on:
      - postgres

  # ===== Serving Layer =====
  llm-api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./src:/app/src
      - ./models:/app/models
    depends_on:
      - redis
      - mlflow

  # ===== Monitoring Layer =====
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
    depends_on:
      - prometheus

  alertmanager:
    image: prom/alertmanager:latest
    ports:
      - "9093:9093"
    volumes:
      - ./monitoring/alertmanager.yml:/etc/alertmanager/config.yml

volumes:
  minio_data:
  redis_data:
  postgres_data:
  prometheus_data:
  grafana_data:
```

#### 集成验证脚本

```python
# integration_test.py
"""MLOps平台集成测试"""

import requests
import time
import sys

def wait_for_service(name: str, url: str, timeout: int = 120):
    """等待服务启动"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code < 500:
                print(f"  [OK] {name} 已就绪")
                return True
        except:
            pass
        time.sleep(5)
    print(f"  [FAIL] {name} 启动超时")
    return False

def test_all_services():
    """测试所有服务"""
    print("=" * 60)
    print("MLOps平台集成测试")
    print("=" * 60)

    services = [
        ('MinIO', 'http://localhost:9000/minio/health/live'),
        ('Redis', 'http://localhost:6379'),  # Redis不是HTTP，这里简化
        ('PostgreSQL', 'http://localhost:5432'),  # 同理
        ('MLflow', 'http://localhost:5000/health'),
        ('Airflow', 'http://localhost:8080/health'),
        ('LLM API', 'http://localhost:8000/health'),
        ('Prometheus', 'http://localhost:9090/-/healthy'),
        ('Grafana', 'http://localhost:3000/api/health'),
    ]

    results = {}
    for name, url in services:
        print(f"\n检查 {name}...")
        results[name] = wait_for_service(name, url)

    print("\n" + "=" * 60)
    print("测试结果:")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
    print("=" * 60)

    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)

if __name__ == '__main__':
    test_all_services()
```

---

### Day 7: 文档与Demo (6-8h)

#### MLOps平台使用文档

```markdown
# MLOps平台使用文档

## 1. 快速开始

### 前置条件
- Docker & Docker Compose
- Python 3.10+
- NVIDIA GPU + CUDA（训练服务需要）

### 一键启动
```bash
git clone https://github.com/your-org/mlops-platform.git
cd mlops-platform
docker-compose up -d
```

### 服务地址
| 服务 | URL | 用户名/密码 |
|------|-----|------------|
| Airflow | http://localhost:8080 | admin/admin |
| MLflow | http://localhost:5000 | - |
| API | http://localhost:8000 | API Key认证 |
| Grafana | http://localhost:3000 | admin/admin |
| MinIO | http://localhost:9001 | minioadmin/minioadmin |

## 2. 使用流程

### 2.1 数据管理
```bash
# 追踪数据
dvc add data/raw/dataset.csv
git add data/raw/dataset.csv.dvc .gitignore
git commit -m "add dataset v1"
dvc push
```

### 2.2 配置训练Pipeline
1. 在 `dags/` 目录创建训练DAG
2. Airflow Web UI 中启用DAG
3. 手动触发或等待定时调度

### 2.3 查看实验结果
1. 访问 MLflow UI: http://localhost:5000
2. 查看实验对比和参数指标
3. 选择最佳模型注册到Model Registry

### 2.4 部署模型
```bash
curl -X POST http://localhost:8000/api/v1/deploy \
  -H "X-API-Key: your-api-key" \
  -d '{"model_name": "my-model", "version": "1"}'
```

### 2.5 监控
1. 访问 Grafana: http://localhost:3000
2. 查看推理延迟、QPS、错误率等面板
3. 配置告警规则

## 3. API文档
访问 http://localhost:8000/docs 查看Swagger API文档
```

#### Demo录制清单

```
Demo录制步骤（建议使用OBS录制，导出为MP4）：

1. [0:00-0:30] 介绍
   - 平台总览，展示四层架构图

2. [0:30-2:00] 数据管理
   - DVC数据版本管理演示
   - 数据质量检查报告展示

3. [2:00-4:00] 训练Pipeline
   - Airflow DAG展示
   - 触发训练，查看执行过程
   - MLflow实验对比

4. [4:00-5:30] 模型部署
   - Model Registry注册
   - API部署和测试
   - 推理请求演示

5. [5:30-7:00] 监控
   - Grafana监控面板
   - 触发告警演示
   - 数据漂移报告

6. [7:00-8:00] 总结
   - 平台亮点总结
   - 后续优化方向

总时长: 8-10分钟
```

---

## 四、代码练习

| Day | 练习内容 | 预计时间 |
|-----|---------|---------|
| Day 1 | 画出端到端ML平台架构图，完成各层技术选型对比表 | 30min |
| Day 2 | 编写一个完整的Airflow训练DAG，包含5个以上任务节点 | 60min |
| Day 3 | 使用Ray实现一个简单的并行数据处理任务，对比串行vs并行耗时 | 60min |
| Day 4 | 实现Spot实例训练管理器，包含checkpoint保存和断点续训逻辑 | 60min |
| Day 5 | 实现API Key认证+RBAC权限控制，至少3个角色和3个权限 | 60min |
| Day 6 | 完成docker-compose.yml编写，所有服务可正常启动并通过集成测试 | 3-4h |
| Day 7 | 编写平台使用文档，录制Demo视频 | 3-4h |

---

## 五、本周产出

1. **ML平台架构图** - 四层架构及各组件技术选型文档
2. **Airflow训练DAG** - 完整的端到端训练编排流水线
3. **Spot训练管理器** - 支持断点续训的弹性训练脚本
4. **安全认证模块** - API Key认证+RBAC+审计日志+数据脱敏
5. **docker-compose.yml** - 一键启动MLOps平台全部服务
6. **平台使用文档** - 完整的使用说明和API文档
7. **Demo视频** - 8-10分钟的平台功能演示

---

## 六、自测题

### 题目

1. **端到端ML平台通常包含哪几层？每层的主要职责和代表工具是什么？**

2. **Airflow DAG中如何定义任务依赖？XCom的作用是什么，有什么使用限制？**

3. **Spot实例训练有哪些风险？如何通过技术手段降低这些风险？**

4. **ML系统常见的安全威胁有哪些？分别如何防御？**

5. **MLOps和传统CI/CD的核心区别是什么？ML CI/CD有哪些额外步骤？**

### 参考答案

<details>
<summary>点击展开答案</summary>

1. **四层架构**：(1) **Data Layer** - 数据版本管理(DVC)、特征管理(Feast)、数据质量(GE)。负责数据的采集、存储、版本控制和质量保障。(2) **Training Layer** - 实验管理(MLflow)、工作流编排(Airflow)、分布式训练(Ray)。负责模型训练、超参搜索和实验追踪。(3) **Serving Layer** - 容器化(Docker)、编排(K8s)、推理引擎(vLLM)。负责模型部署、弹性伸缩和高性能推理。(4) **Monitoring Layer** - 指标采集(Prometheus)、可视化(Grafana)、漂移检测(Evidently)。负责运行监控、漂移告警和性能追踪。

2. **任务依赖**：通过 `>>` 运算符或 `set_downstream()/set_upstream()` 方法定义。如 `task_a >> task_b >> task_c`。**XCom** (Cross-Communication) 用于任务间传递小量元数据（如数据路径、指标值），限制是默认最大48KB，不适合传递大数据。大数据应通过共享存储（S3/文件系统）传递，XCom只传递引用路径。

3. **风险**：Spot实例随时可能被云厂商回收，导致训练中断、进度丢失。**防御手段**：(1) 频繁保存checkpoint（每N步保存一次）(2) 实现断点续训逻辑，从最新checkpoint恢复 (3) checkpoint保存到持久存储（S3/NFS）(4) 使用多可用区分散风险 (5) 设置max_wait参数，在实例被回收后自动重新请求 (6) 监控Spot市场价格，避免出价过低。

4. **安全威胁**：(1) **提示注入攻击** - 恶意用户通过输入操纵LLM行为。防御：输入过滤+规则匹配。(2) **数据泄露** - 模型输出包含训练数据中的敏感信息。防御：数据脱敏+输出检查。(3) **未授权访问** - API被未授权用户调用。防御：API Key认证+RBAC权限控制。(4) **对抗样本** - 精心构造的输入欺骗模型。防御：输入验证+异常检测。(5) **模型窃取** - 通过大量查询复制模型。防御：速率限制+查询频率监控。

5. **核心区别**：传统CI/CD关注代码质量和功能正确性，ML CI/CD还需要关注**数据质量**和**模型效果**。**额外步骤**：(1) 数据质量验证（检查数据分布和完整性）(2) 模型评估门禁（精度必须超过阈值）(3) 模型注册到Registry（版本管理和阶段管理）(4) A/B测试/灰度发布（渐进式验证模型效果）(5) 持续监控数据漂移和模型退化 (6) 自动重训机制（检测到漂移时触发重新训练）。
</details>

---

## 七、Java开发者提示

> 帮助有Java背景的开发者理解MLOps综合实战概念

| MLOps概念 | Java类比 | 说明 |
|-----------|---------|------|
| Airflow DAG | Spring Batch Job/Step | 工作流编排概念相同，都是定义任务和依赖 |
| Airflow XCom | Spring Integration MessageChannel | 任务间通信机制 |
| MLflow Model Registry | Maven仓库(Nexus/Artifactory) | 模型版本管理，类似jar包版本管理 |
| Docker Compose | Docker Compose(通用) | 微服务编排方式完全相同 |
| RBAC权限 | Spring Security RBAC | 基于角色的访问控制概念相同 |
| API Key认证 | JWT Token认证 | 认证机制类似 |
| 审计日志 | SLF4J + ELK | 日志记录和分析方式类似 |
| Spot实例 | 云竞价实例(Java也用) | 成本优化概念相同 |
| 弹性伸缩 | K8s HPA(Java也用) | 自动扩缩容概念相同 |
| Ray分布式 | Spark/Flink分布式计算 | 分布式计算框架类比 |
| nvidia-smi | JVM监控(jstat/jmap) | 资源监控类比 |
| 对抗攻击防御 | WAF(Web应用防火墙) | 安全防护概念相同 |
| 数据脱敏 | 数据脱敏(Java也有) | 隐私保护概念相同 |

### 关键思维转换

1. **平台思维**: MLOps平台就像一个微服务架构系统，每个组件都是一个服务，通过API通信
2. **数据是Pipeline的血液**: 传统软件关注代码流转，ML还需要关注数据流转
3. **模型也是制品**: 模型文件(.pt/.onnx)就像Java的jar包，需要版本管理、仓库存储和分发
4. **训练即构建**: 模型训练相当于Java项目的编译构建，需要自动化和可复现
5. **推理即服务**: 模型推理相当于Spring Boot应用，需要部署、监控和弹性伸缩
