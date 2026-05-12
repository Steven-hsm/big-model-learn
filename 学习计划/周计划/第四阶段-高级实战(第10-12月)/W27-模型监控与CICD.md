# 第27周 - 模型监控与CI/CD

> **阶段**: MLOps监控与自动化
> **时间**: 工作日每晚2小时 + 周末6-8小时
> **前置知识**: 第25-26周数据工程与实验管理

---

## 一、本周目标

1. 理解数据漂移、概念漂移和预测漂移的区别，掌握漂移检测方法
2. 使用Evidently生成数据漂移报告，设置漂移告警阈值
3. 搭建Prometheus+Grafana模型性能监控体系
4. 实现ML CI/CD流水线，自动化训练、评估和部署流程
5. 为之前的RAG/Agent项目搭建完整的CI/CD和监控

---

## 二、时间安排

### 工作日（每晚2小时）

| 日期 | 主题 | 时长 |
|------|------|------|
| Day 1 (周一) | 模型监控概念 | 2h |
| Day 2 (周二) | 漂移检测实践 | 2h |
| Day 3 (周三) | 模型性能监控 | 2h |
| Day 4 (周四) | ML CI/CD基础 | 2h |
| Day 5 (周五) | ML CI/CD进阶 | 2h |

### 周末（6-8小时/天）

| 日期 | 主题 | 时长 |
|------|------|------|
| Day 6 (周六) | 告警与应急 | 6-8h |
| Day 7 (周日) | 实战：RAG/Agent项目CI/CD | 6-8h |

---

## 三、详细学习内容

### Day 1: 模型监控概念 (2h)

#### 三种漂移类型

```
1. 数据漂移 Data Drift（输入数据分布变化）
   P(X) ≠ P_train(X)

   原因：用户画像变化、季节性变化、数据源变更
   示例：训练时用户年龄集中在20-40岁，上线后突然大量50-60岁用户

2. 概念漂移 Concept Drift（标签与特征的关系变化）
   P(Y|X) 发生变化

   原因：用户偏好变化、市场环境变化、社会事件
   示例：疫情前后的消费行为模型，"外出就餐"的预测逻辑完全改变

3. 预测漂移 Prediction Drift（模型输出分布变化）
   P(Y_pred) 发生变化

   原因：数据漂移或概念漂移的间接结果
   示例：模型原本预测50%正面/50%负面，突然变成80%正面/20%负面
```

#### 监控架构设计

```
┌─────────────────────────────────────────────────┐
│                  ML监控系统                       │
│                                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │ 数据监控  │  │ 模型监控  │  │ 系统性能监控  │   │
│  │          │  │          │  │              │   │
│  │ 数据漂移  │  │ 预测漂移  │  │ 延迟P50/P99  │   │
│  │ 特征分布  │  │ 准确率    │  │ 吞吐量QPS    │   │
│  │ 数据质量  │  │ 置信度    │  │ 错误率       │   │
│  │ 缺失率   │  │ 反馈标签  │  │ GPU利用率    │   │
│  └─────┬────┘  └─────┬────┘  └──────┬───────┘   │
│        │             │              │             │
│        ▼             ▼              ▼             │
│  ┌─────────────────────────────────────────┐     │
│  │            告警系统 (AlertManager)        │     │
│  │  P0紧急 / P1高 / P2中 / P3低             │     │
│  └──────────────────┬──────────────────────┘     │
│                     │                             │
│                     ▼                             │
│  ┌─────────────────────────────────────────┐     │
│  │          可视化面板 (Grafana)             │     │
│  └─────────────────────────────────────────┘     │
└─────────────────────────────────────────────────┘
```

#### 为什么模型上线后效果退化

```
模型效果退化的常见原因：

1. 数据漂移（占40%）
   - 用户群体变化
   - 数据采集方式变更
   - 季节性波动

2. 概念漂移（占30%）
   - 业务规则变化
   - 用户偏好转移
   - 外部环境变化

3. 数据质量问题（占20%）
   - 特征缺失
   - 数据格式变化
   - 上游数据源异常

4. 系统问题（占10%）
   - 特征计算错误
   - 模型加载异常
   - 网络延迟

结论：模型上线不是结束，而是运维的开始
```

---

### Day 2: 漂移检测实践 (2h)

#### Evidently安装与基础

```bash
# 安装
pip install evidently

# 安装含可视化依赖
pip install evidently[visualization]
```

#### 生成数据漂移报告

```python
import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import *

# 准备参考数据（训练数据）和当前数据（线上数据）
reference_data = pd.DataFrame({
    "age": np.random.normal(35, 10, 1000),
    "income": np.random.normal(50000, 15000, 1000),
    "score": np.random.uniform(0, 100, 1000),
})

# 模拟数据漂移：年龄分布发生变化
current_data = pd.DataFrame({
    "age": np.random.normal(50, 12, 1000),       # 年龄偏移
    "income": np.random.normal(50000, 15000, 1000), # 收入不变
    "score": np.random.uniform(0, 100, 1000),       # 分数不变
})

# 方法1：使用预设报告
drift_report = Report(metrics=[
    DataDriftPreset(),
])
drift_report.run(
    reference_data=reference_data,
    current_data=current_data,
)
drift_report.save_html("data_drift_report.html")

# 方法2：使用具体指标
detailed_report = Report(metrics=[
    # 整体数据漂移
    DataDriftTable(),

    # 单列漂移检测
    ColumnDriftMetric(column_name="age"),
    ColumnDriftMetric(column_name="income"),
    ColumnDriftMetric(column_name="score"),

    # 数值列分布统计
    ColumnSummaryMetric(column_name="age"),
    ColumnSummaryMetric(column_name="income"),
])
detailed_report.run(
    reference_data=reference_data,
    current_data=current_data,
)
detailed_report.save_html("detailed_drift_report.html")

# 查看结果
result = detailed_report.as_dict()
for metric in result["metrics"]:
    if "drift_score" in str(metric):
        print(metric)
```

#### 统计检验方法详解

```python
# 漂移检测的三大统计方法

"""
1. KS检验 (Kolmogorov-Smirnov Test)
   - 适用: 连续数值型特征
   - 原理: 比较两个分布的累积分布函数(CDF)最大差异
   - 统计量: D = max|F_ref(x) - F_cur(x)|
   - 判断: p_value < 0.05 认为存在显著漂移
"""

from scipy import stats
import numpy as np

ref_data = np.random.normal(35, 10, 1000)
cur_data = np.random.normal(50, 12, 1000)

ks_stat, p_value = stats.ks_2samp(ref_data, cur_data)
print(f"KS统计量: {ks_stat:.4f}, p值: {p_value:.6f}")
# KS统计量越大，漂移越严重

"""
2. Wasserstein距离 (Earth Mover's Distance)
   - 适用: 连续数值型特征
   - 原理: 将一个分布"搬运"到另一个分布所需的最小代价
   - 判断: 距离越大，漂移越严重
   - 直觉: 两个分布的"重心"距离
"""

from scipy.stats import wasserstein_distance

w_dist = wasserstein_distance(ref_data, cur_data)
print(f"Wasserstein距离: {w_dist:.4f}")

"""
3. PSI (Population Stability Index) 群体稳定性指标
   - 适用: 任何特征（分箱后比较）
   - 原理: 比较两个分布在各分箱中的占比差异
   - 计算: PSI = sum((actual_pct - expected_pct) * ln(actual_pct / expected_pct))
   - 判断标准:
     PSI < 0.1  : 无显著变化（绿灯）
     0.1 <= PSI < 0.2 : 需要关注（黄灯）
     PSI >= 0.2 : 显著漂移（红灯，需要重新训练）
"""

def calculate_psi(expected, actual, bins=10):
    """计算PSI"""
    # 分箱
    breakpoints = np.linspace(min(expected.min(), actual.min()),
                              max(expected.max(), actual.max()), bins + 1)

    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    # 避免0值
    expected_pct = np.clip(expected_pct, 1e-6, None)
    actual_pct = np.clip(actual_pct, 1e-6, None)

    # 计算PSI
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi

psi = calculate_psi(ref_data, cur_data)
print(f"PSI: {psi:.4f}")
if psi < 0.1:
    print("状态: 稳定（绿灯）")
elif psi < 0.2:
    print("状态: 需关注（黄灯）")
else:
    print("状态: 显著漂移（红灯）- 需要重新训练模型")
```

#### Alibi Detect库

```python
# pip install alibi-detect
from alibi_detect.cd import MMDDrift, LSDDDrift
import numpy as np

# 准备数据
X_ref = np.random.randn(1000, 10)  # 参考数据
X_test = np.random.randn(500, 10)   # 测试数据（无漂移）
X_drift = np.random.randn(500, 10) + 2  # 漂移数据

# MMD (Maximum Mean Discrepancy) 漂移检测
mmd_detector = MMDDrift(X_ref, backend='tensorflow', p_val=0.05)
pred = mmd_detector.predict(X_test)
print(f"无漂移数据: drift={pred['data']['is_drift']}, p_val={pred['data']['p_val']}")

pred = mmd_detector.predict(X_drift)
print(f"漂移数据: drift={pred['data']['is_drift']}, p_val={pred['data']['p_val']}")

# LSDD (Least-Squares Density Difference) 漂移检测
lsdd_detector = LSDDDrift(X_ref, backend='tensorflow', p_val=0.05)
pred = lsdd_detector.predict(X_drift)
print(f"LSDD检测: drift={pred['data']['is_drift']}")
```

---

### Day 3: 模型性能监控 (2h)

#### 关键监控指标

```
推理性能指标：
┌──────────────────────────────────────────────┐
│ 延迟 (Latency)                                │
│   P50: 50%的请求在此时间内完成  → 用户体验基线  │
│   P95: 95%的请求在此时间内完成  → 可接受上限    │
│   P99: 99%的请求在此时间内完成  → 尾部延迟      │
│                                                │
│ 吞吐量 (Throughput)                            │
│   QPS: 每秒处理请求数                           │
│   TPS: 每秒处理事务数                           │
│                                                │
│ 错误率 (Error Rate)                            │
│   4xx: 客户端错误（参数错误等）                  │
│   5xx: 服务端错误（模型异常等）                  │
│                                                │
│ 资源使用                                       │
│   GPU显存利用率: 模型占用的显存比例              │
│   GPU计算利用率: GPU计算单元的繁忙程度           │
│   CPU利用率: CPU使用百分比                      │
│   内存利用率: 内存使用百分比                     │
└──────────────────────────────────────────────┘
```

#### Prometheus指标采集

```bash
# 安装Prometheus客户端
pip install prometheus_client
```

```python
# monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import FastAPI, Response
import time
import threading

app = FastAPI()

# 定义指标

# 1. 请求计数器
REQUEST_COUNT = Counter(
    'model_request_total',
    'Total model inference requests',
    ['model_name', 'model_version', 'status']  # 标签维度
)

# 2. 推理延迟直方图
INFERENCE_LATENCY = Histogram(
    'model_inference_seconds',
    'Model inference latency in seconds',
    ['model_name'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]  # 延迟桶
)

# 3. 当前处理的请求数
ACTIVE_REQUESTS = Gauge(
    'model_active_requests',
    'Number of active inference requests',
    ['model_name']
)

# 4. 模型预测分布
PREDICTION_DISTRIBUTION = Counter(
    'model_prediction_total',
    'Distribution of model predictions',
    ['model_name', 'predicted_class']
)

# 5. GPU指标（需要nvidia-ml-py3）
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_MEMORY_USED = Gauge('gpu_memory_used_bytes', 'GPU memory used in bytes', ['gpu_id'])
    GPU_UTILIZATION = Gauge('gpu_utilization_percent', 'GPU utilization percentage', ['gpu_id'])
except:
    pass

def update_gpu_metrics():
    """定期更新GPU指标"""
    try:
        import pynvml
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            GPU_MEMORY_USED.labels(gpu_id=str(i)).set(info.used)
            GPU_UTILIZATION.labels(gpu_id=str(i)).set(util.gpu)
    except:
        pass

# GPU指标更新线程
def gpu_metrics_thread():
    while True:
        update_gpu_metrics()
        time.sleep(5)

threading.Thread(target=gpu_metrics_thread, daemon=True).start()

# 推理接口（带监控）
@app.post("/predict")
def predict(text: str):
    model_name = "sentiment-classifier"

    ACTIVE_REQUESTS.labels(model_name=model_name).inc()
    start_time = time.time()

    try:
        # 模型推理
        result = model.predict(text)
        predicted_class = result["label"]

        # 记录预测分布
        PREDICTION_DISTRIBUTION.labels(
            model_name=model_name, predicted_class=predicted_class
        ).inc()

        # 记录成功请求
        REQUEST_COUNT.labels(
            model_name=model_name, model_version="v2", status="success"
        ).inc()

        return {"prediction": predicted_class, "confidence": result["score"]}

    except Exception as e:
        REQUEST_COUNT.labels(
            model_name=model_name, model_version="v2", status="error"
        ).inc()
        raise

    finally:
        latency = time.time() - start_time
        INFERENCE_LATENCY.labels(model_name=model_name).observe(latency)
        ACTIVE_REQUESTS.labels(model_name=model_name).dec()

# Prometheus指标暴露端点
@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type="text/plain"
    )
```

#### Grafana可视化面板

```yaml
# docker-compose.yml - Prometheus + Grafana
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
    depends_on:
      - prometheus

volumes:
  prometheus_data:
  grafana_data:
```

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'ml-model'
    metrics_path: '/metrics'
    static_configs:
      - targets: ['host.docker.internal:8000']  # 模型服务地址
```

```
Grafana Dashboard关键面板：

1. 请求概览
   - 总请求数（计数器）
   - 成功/失败比例（饼图）
   - 每分钟请求量（时序图）

2. 延迟面板
   - P50延迟（折线图）
   - P95延迟（折线图）
   - P99延迟（折线图）
   - 延迟分布（直方图）

3. 资源面板
   - GPU显存使用（折线图）
   - GPU计算利用率（折线图）
   - CPU利用率（折线图）
   - 内存使用（折线图）

4. 预测面板
   - 预测类别分布（饼图）
   - 各类别预测趋势（折线图）

配置PromQL查询示例：
- P50延迟: histogram_quantile(0.5, rate(model_inference_seconds_bucket[5m]))
- P95延迟: histogram_quantile(0.95, rate(model_inference_seconds_bucket[5m]))
- QPS: rate(model_request_total[1m])
- 错误率: rate(model_request_total{status="error"}[5m]) / rate(model_request_total[5m])
- GPU利用率: gpu_utilization_percent{gpu_id="0"}
```

---

### Day 4: ML CI/CD基础 (2h)

#### GitHub Actions工作流文件

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Training Pipeline

on:
  push:
    branches: [main]
    paths:
      - 'src/**'
      - 'data/**'
      - 'configs/**'
  workflow_dispatch:  # 手动触发
    inputs:
      model_name:
        description: 'Model name to train'
        default: 'sentiment-classifier'

jobs:
  # 阶段1：数据验证
  data-validation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install great-expectations pandas

      - name: Run data quality checks
        run: python src/validate_data.py

  # 阶段2：模型训练
  model-training:
    needs: data-validation  # 依赖数据验证通过
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'
          cache: 'pip'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install mlflow scikit-learn

      - name: Train model
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
          MLFLOW_EXPERIMENT: ${{ github.event.inputs.model_name || 'default' }}
        run: |
          python src/train.py

      - name: Upload model artifact
        uses: actions/upload-artifact@v4
        with:
          name: trained-model
          path: models/

  # 阶段3：模型评估
  model-evaluation:
    needs: model-training
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Download model
        uses: actions/download-artifact@v4
        with:
          name: trained-model
          path: models/

      - name: Evaluate model
        run: |
          python src/evaluate.py \
            --model-path models/ \
            --threshold 0.85 \
            --output evaluation_report.json

      - name: Check model quality gate
        run: |
          python src/check_quality_gate.py \
            --report evaluation_report.json \
            --min-accuracy 0.85 \
            --min-f1 0.80

  # 阶段4：模型注册
  model-registration:
    needs: model-evaluation
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'  # 只在main分支注册
    steps:
      - uses: actions/checkout@v4

      - name: Register model to MLflow
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
        run: |
          python src/register_model.py \
            --model-name sentiment-classifier \
            --stage Staging

  # 阶段5：自动部署
  model-deployment:
    needs: model-registration
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build Docker image
        run: |
          docker build -t sentiment-classifier:${{ github.sha }} .

      - name: Push to registry
        run: |
          docker push your-registry/sentiment-classifier:${{ github.sha }}

      - name: Deploy to staging
        run: |
          python src/deploy.py \
            --image-tag ${{ github.sha }} \
            --environment staging
```

#### 模型评估门禁

```python
# src/check_quality_gate.py
import json
import sys
import argparse

def check_quality_gate(report_path: str, min_accuracy: float, min_f1: float):
    """模型质量门禁检查"""
    with open(report_path) as f:
        report = json.load(f)

    accuracy = report["accuracy"]
    f1 = report["f1_score"]

    print(f"模型评估结果:")
    print(f"  准确率: {accuracy:.4f} (阈值: {min_accuracy})")
    print(f"  F1分数: {f1:.4f} (阈值: {min_f1})")

    passed = True
    if accuracy < min_accuracy:
        print(f"  FAIL: 准确率 {accuracy:.4f} < {min_accuracy}")
        passed = False
    else:
        print(f"  PASS: 准确率达标")

    if f1 < min_f1:
        print(f"  FAIL: F1分数 {f1:.4f} < {min_f1}")
        passed = False
    else:
        print(f"  PASS: F1分数达标")

    if not passed:
        print("\n质量门禁未通过，模型不会被注册到Production")
        sys.exit(1)
    else:
        print("\n质量门禁通过，模型可以注册到Staging")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--min-accuracy", type=float, default=0.85)
    parser.add_argument("--min-f1", type=float, default=0.80)
    args = parser.parse_args()
    check_quality_gate(args.report, args.min_accuracy, args.min_f1)
```

---

### Day 5: ML CI/CD进阶 (2h)

#### A/B测试框架

```python
# ab_testing.py
import hashlib
import random
from typing import Dict, Optional

class ABTestRouter:
    """A/B测试路由器：将流量分配到不同模型版本"""

    def __init__(self):
        self.experiments = {}

    def create_experiment(
        self,
        name: str,
        model_a: str,        # 对照组模型（当前生产版本）
        model_b: str,        # 实验组模型（新版本）
        traffic_ratio: float = 0.5,  # B组流量比例
    ):
        self.experiments[name] = {
            "model_a": model_a,
            "model_b": model_b,
            "traffic_ratio": traffic_ratio,
            "results_a": {"count": 0, "positive": 0},
            "results_b": {"count": 0, "positive": 0},
        }

    def route(self, experiment_name: str, user_id: str) -> str:
        """根据user_id分配到A组或B组"""
        exp = self.experiments[experiment_name]

        # 使用hash确保同一用户始终分到同一组
        hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16) % 100

        if hash_val < exp["traffic_ratio"] * 100:
            return exp["model_b"]  # 实验组
        else:
            return exp["model_a"]  # 对照组

    def record_result(self, experiment_name: str, model: str, is_positive: bool):
        """记录结果"""
        exp = self.experiments[experiment_name]
        group = "results_a" if model == exp["model_a"] else "results_b"
        exp[group]["count"] += 1
        if is_positive:
            exp[group]["positive"] += 1

    def get_results(self, experiment_name: str) -> Dict:
        """获取实验结果"""
        exp = self.experiments[experiment_name]
        results_a = exp["results_a"]
        results_b = exp["results_b"]

        rate_a = results_a["positive"] / max(results_a["count"], 1)
        rate_b = results_b["positive"] / max(results_b["count"], 1)

        return {
            "model_a": {"name": exp["model_a"], "rate": rate_a, "count": results_a["count"]},
            "model_b": {"name": exp["model_b"], "rate": rate_b, "count": results_b["count"]},
            "improvement": rate_b - rate_a,
        }

# 使用示例
router = ABTestRouter()
router.create_experiment("sentiment_v2_test", "sentiment-v1", "sentiment-v2", traffic_ratio=0.2)

# 每次请求
user_id = "user_123"
model_name = router.route("sentiment_v2_test", user_id)
# 调用对应模型...
# 记录结果
router.record_result("sentiment_v2_test", model_name, is_positive=True)
```

#### 灰度发布（Canary Deployment）

```python
# canary_deployment.py

class CanaryDeployment:
    """灰度发布：逐步增加新版本流量"""

    def __init__(self, config: dict):
        self.model_old = config["model_old"]
        self.model_new = config["model_new"]
        self.stages = [1, 5, 10, 25, 50, 100]  # 流量百分比阶段
        self.current_stage = 0
        self.error_threshold = config.get("error_threshold", 0.05)

    def get_current_traffic_percent(self) -> int:
        return self.stages[self.current_stage]

    def route(self, request_id: str) -> str:
        """路由请求到旧模型或新模型"""
        traffic_percent = self.get_current_traffic_percent()

        # 简单取模分流
        if hash(request_id) % 100 < traffic_percent:
            return self.model_new
        else:
            return self.model_old

    def check_metrics_and_promote(self, error_rate: float):
        """检查指标，决定是否推进到下一阶段"""
        if error_rate > self.error_threshold:
            print(f"错误率 {error_rate:.4f} 超过阈值 {self.error_threshold}，回滚！")
            self.rollback()
            return False

        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            new_percent = self.get_current_traffic_percent()
            print(f"灰度推进: 新版本流量增加到 {new_percent}%")
            return True
        else:
            print("新版本已全量上线")
            return True

    def rollback(self):
        """回滚到旧版本"""
        self.current_stage = 0
        print(f"回滚: 切换回旧版本 {self.model_old}")

# 使用示例
canary = CanaryDeployment({
    "model_old": "sentiment-v1",
    "model_new": "sentiment-v2",
    "error_threshold": 0.05,
})
```

#### Shadow Mode（影子模式）

```python
# shadow_mode.py
import time
import logging

logger = logging.getLogger("shadow_mode")

class ShadowModeDeployment:
    """影子模式：新模型并行运行但不影响用户"""

    def __init__(self, production_model, shadow_model):
        self.production_model = production_model
        self.shadow_model = shadow_model
        self.comparison_log = []

    def predict(self, input_data: dict) -> dict:
        """生产模型返回结果，影子模型静默运行"""

        # 1. 生产模型正常返回
        start = time.time()
        prod_result = self.production_model.predict(input_data)
        prod_latency = time.time() - start

        # 2. 影子模型静默运行（不影响用户）
        try:
            start = time.time()
            shadow_result = self.shadow_model.predict(input_data)
            shadow_latency = time.time() - start

            # 3. 记录对比数据
            comparison = {
                "timestamp": time.time(),
                "input_hash": hash(str(input_data)),
                "prod_prediction": prod_result["label"],
                "prod_confidence": prod_result["confidence"],
                "prod_latency": prod_latency,
                "shadow_prediction": shadow_result["label"],
                "shadow_confidence": shadow_result["confidence"],
                "shadow_latency": shadow_latency,
                "predictions_match": prod_result["label"] == shadow_result["label"],
            }
            self.comparison_log.append(comparison)

            # 4. 如果差异较大，记录警告
            if not comparison["predictions_match"]:
                logger.warning(
                    f"预测不一致: prod={prod_result['label']}, "
                    f"shadow={shadow_result['label']}, "
                    f"input={input_data}"
                )

        except Exception as e:
            logger.error(f"影子模型异常: {e}")

        # 只返回生产模型的结果
        return prod_result

    def get_comparison_report(self) -> dict:
        """获取影子模式对比报告"""
        total = len(self.comparison_log)
        if total == 0:
            return {"total": 0}

        match_count = sum(1 for c in self.comparison_log if c["predictions_match"])
        avg_prod_latency = sum(c["prod_latency"] for c in self.comparison_log) / total
        avg_shadow_latency = sum(c["shadow_latency"] for c in self.comparison_log) / total

        return {
            "total_requests": total,
            "predictions_match_rate": match_count / total,
            "avg_prod_latency": avg_prod_latency,
            "avg_shadow_latency": avg_shadow_latency,
            "latency_diff": avg_shadow_latency - avg_prod_latency,
        }
```

---

### Day 6: 告警与应急 (6-8h)

#### 告警规则配置

```yaml
# alertmanager/alert_rules.yml
groups:
  - name: ml_model_alerts
    interval: 30s
    rules:
      # P0 紧急 - 模型服务不可用
      - alert: ModelServiceDown
        expr: up{job="ml-model"} == 0
        for: 1m
        labels:
          severity: P0
        annotations:
          summary: "模型服务 {{ $labels.instance }} 不可用"
          description: "模型服务已宕机超过1分钟"

      # P1 高 - 错误率飙升
      - alert: HighErrorRate
        expr: rate(model_request_total{status="error"}[5m]) / rate(model_request_total[5m]) > 0.1
        for: 5m
        labels:
          severity: P1
        annotations:
          summary: "模型错误率超过10%"
          description: "当前错误率: {{ $value | humanizePercentage }}"

      # P1 高 - 数据漂移
      - alert: DataDriftDetected
        expr: ml_data_drift_psi > 0.2
        for: 10m
        labels:
          severity: P1
        annotations:
          summary: "检测到显著数据漂移"
          description: "PSI值: {{ $value }}，超过0.2阈值"

      # P2 中 - 延迟升高
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(model_inference_seconds_bucket[5m])) > 2.0
        for: 10m
        labels:
          severity: P2
        annotations:
          summary: "P95延迟超过2秒"
          description: "当前P95延迟: {{ $value }}s"

      # P2 中 - GPU利用率过高
      - alert: GPUOverutilization
        expr: gpu_utilization_percent > 95
        for: 15m
        labels:
          severity: P2
        annotations:
          summary: "GPU利用率持续超过95%"
          description: "可能需要扩容"

      # P3 低 - 模型预测分布异常
      - alert: PredictionDrift
        expr: abs(rate(model_prediction_total{predicted_class="positive"}[1h]) - 0.5) > 0.2
        for: 1h
        labels:
          severity: P3
        annotations:
          summary: "模型预测分布偏离正常"
          description: "正面预测占比偏离50%超过20%"
```

#### 告警级别与响应

```
告警级别定义：

P0 紧急（立即响应，< 15分钟）
  - 模型服务完全不可用
  - 大量用户受影响
  - 处理：立即回滚 + 通知全员

P1 高（30分钟内响应）
  - 错误率超过10%
  - 数据漂移显著（PSI > 0.2）
  - 处理：评估影响 + 决定是否回滚

P2 中（2小时内响应）
  - 延迟升高
  - GPU利用率过高
  - 处理：排查原因 + 优化

P3 低（24小时内响应）
  - 预测分布轻微变化
  - 资源利用率波动
  - 处理：记录 + 后续优化
```

#### 模型降级策略

```python
# degradation.py

class ModelDegradationHandler:
    """模型降级策略"""

    def __init__(self, config: dict):
        self.primary_model = config["primary_model"]
        self.fallback_model = config.get("fallback_model")       # 备用模型（上一版本）
        self.rule_engine = config.get("rule_engine")              # 规则引擎
        self.default_response = config.get("default_response")    # 默认响应

        self.error_count = 0
        self.max_errors = config.get("max_errors", 5)
        self.current_level = "normal"  # normal/fallback/rule/default

    def predict(self, input_data: dict) -> dict:
        """根据当前降级级别返回结果"""
        try:
            if self.current_level == "normal":
                return self.primary_model.predict(input_data)
            elif self.current_level == "fallback":
                return self.fallback_model.predict(input_data)
            elif self.current_level == "rule":
                return self.rule_engine.predict(input_data)
            else:
                return self.default_response

        except Exception as e:
            self.error_count += 1
            if self.error_count >= self.max_errors:
                self._degrade()
            return self.predict(input_data)  # 重试

    def _degrade(self):
        """降级到下一级"""
        if self.current_level == "normal":
            self.current_level = "fallback"
            print(f"降级: 主模型 → 备用模型({self.fallback_model})")
        elif self.current_level == "fallback":
            self.current_level = "rule"
            print("降级: 备用模型 → 规则引擎")
        else:
            self.current_level = "default"
            print("降级: 规则引擎 → 默认响应")

        self.error_count = 0

    def recover(self):
        """恢复到正常级别"""
        self.current_level = "normal"
        self.error_count = 0
        print("恢复: 切换回主模型")
```

#### On-call手册模板

```markdown
# ML模型On-call手册

## 1. 模型服务信息
- 服务名称: sentiment-classifier
- 当前生产版本: v2.3
- 备用版本: v2.2
- MLflow地址: http://mlflow.company.com
- Grafana面板: http://grafana.company.com/d/ml-dashboard

## 2. 快速回滚步骤
```bash
# 1. 确认当前Production版本
mlflow models list --model-name sentiment-classifier

# 2. 切换到上一版本
python scripts/rollback.py --model sentiment-classifier --version 2.2

# 3. 验证服务恢复
curl http://model-service:8000/health
```

## 3. 常见故障排查

### 故障1: 推理延迟飙升
- 检查GPU显存: nvidia-smi
- 检查请求量: Grafana QPS面板
- 解决: 扩容 or 限流

### 故障2: 数据漂移告警
- 检查上游数据源: 数据格式是否变更
- 查看漂移报告: Evidently Dashboard
- 解决: 数据修复 or 模型重训

### 故障3: 模型OOM
- 检查输入数据大小
- 检查batch_size配置
- 解决: 减小batch_size or 增加GPU

## 4. 应急联系人
- 模型负责人: @zhangsan
- 数据工程师: @lisi
- 基础设施: @wangwu
```

---

### Day 7: 实战 - RAG/Agent项目CI/CD (6-8h)

#### 完整CI/CD Pipeline

```yaml
# .github/workflows/rag-pipeline.yml
name: RAG System CI/CD

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v --cov=src --cov-report=xml
      - run: bandit -r src/ -f json -o security_report.json

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - name: Build Docker image
        run: docker build -t rag-system:${{ github.sha }} .
      - name: Push to registry
        run: |
          echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USER }} --password-stdin
          docker tag rag-system:${{ github.sha }} your-registry/rag-system:${{ github.sha }}
          docker push your-registry/rag-system:${{ github.sha }}

  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Deploy
        run: python scripts/deploy.py --tag ${{ github.sha }} --env staging
```

#### Docker部署配置

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY configs/ ./configs/

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  rag-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
      - prometheus

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    depends_on:
      - prometheus

  mlflow:
    image: python:3.10-slim
    command: >
      bash -c "pip install mlflow &&
               mlflow server --host 0.0.0.0 --port 5000
               --backend-store-uri sqlite:///mlflow.db"
    ports:
      - "5000:5000"
```

---

## 四、代码练习

| Day | 练习内容 | 预计时间 |
|-----|---------|---------|
| Day 1 | 画出你的模型监控架构图，列出需要监控的指标 | 30min |
| Day 2 | 用Evidently对一个数据集生成漂移报���，计算PSI值 | 60min |
| Day 3 | 搭建Prometheus+Grafana，配置至少4个监控面板 | 60min |
| Day 4 | 创建GitHub Actions工作流，自动化训练+评估+注册 | 60min |
| Day 5 | 实现A/B测试路由器或灰度发布逻辑 | 60min |
| Day 6 | 编写告警规则和On-call手册 | 2h |
| Day 7 | 完成RAG/Agent项目的完整CI/CD和监控搭建 | 4-5h |

---

## 五、本周产出

1. **数据漂移报告** - Evidently生成的HTML报告
2. **Prometheus+Grafana监控** - Docker Compose一键启动的监控体系
3. **GitHub Actions CI/CD** - 自动化训练、评估、注册、部署流水线
4. **A/B测试或灰度发布代码** - 流量分配和模型切换逻辑
5. **告警规则和On-call手册** - 完整的应急响应流程
6. **Docker Compose部署文件** - 一键部署全部服务

---

## 六、自测题

### 题目

1. **数据漂移(Data Drift)和概念漂移(Concept Drift)的区别是什么？各举一个例子。**

2. **PSI指标大于0.2意味着什么？应该采取什么行动？**

3. **Prometheus+Grafana的监控架构是怎样的？数据如何从应用流向Grafana面板？**

4. **ML CI/CD和传统软件CI/CD有什么区别？有哪些额外步骤？**

5. **灰度发布(Canary)和A/B测试的区别是什么？各自的适用场景？**

### 参考答案

<details>
<summary>点击展开答案</summary>

1. **数据漂移**是输入数据分布P(X)发生变化。例如：模型训练时用户年龄集中在20-40岁，上线后大量50-60岁用户涌入。**概念漂移**是特征与标签的关系P(Y|X)发生变化。例如：疫情前后，"外出就餐"特征对"是否会感染"的预测关系完全改变。数据漂移是"数据变了"，概念漂移是"规律变了"。

2. PSI > 0.2意味着数据分布发生了**显著变化**（红灯状态），模型可能已经不适应当前数据。应该：(1) 确认漂移是否为正常波动（季节性等）；(2) 如果是持续漂移，需要用新数据重新训练模型；(3) 评估当前模型效果是否下降；(4) 制定模型更新计划。

3. 架构：**应用暴露/metrics端点（prometheus_client）→ Prometheus定时拉取(scrape)指标数据 → Prometheus存储时序数据 → Grafana查询Prometheus(PromQL) → Grafana渲染可视化面板**。数据流：应用代码中的Counter/Histogram/Gauge → /metrics HTTP端点 → Prometheus存储 → Grafana Dashboard。

4. **ML CI/CD额外步骤**：(1) **数据验证** - 检查数据质量和分布；(2) **模型评估门禁** - 精度必须超过阈值；(3) **模型注册** - 注册到Model Registry；(4) **A/B测试/灰度发布** - 渐进式上线验证；(5) **模型监控** - 上线后持续监控漂移和性能；(6) **模型回滚** - 效果退化时快速回滚。传统CI/CD主要关注代码质量和功能测试，ML CI/CD还需要关注数据质量和模型效果。

5. **灰度发布**: 按流量比例逐步扩大新版本，关注系统稳定性（延迟、错误率），目标是安全上线。适合所有模型更新。**A/B测试**: 将用户分为A/B两组，对比业务指标（点击率、转化率等），目标是验证业务效果。适合需要验证业务价值的新模型。灰度发布关注"会不会出问题"，A/B测试关注"效果会不会更好"。
</details>

---

## 七、Java开发者提示

> 帮助有Java背景的开发者理解ML监控与CI/CD概念

| ML监控/CI/CD概念 | Java类比 | 说明 |
|-----------------|---------|------|
| 数据漂移 | API输入参数格式变化 | 用户传的数据和开发时预期不同了 |
| 概念漂移 | 业务规则变化 | 之前正确的逻辑现在不适用了 |
| PSI指标 | SLA偏离度 | 衡量当前状态和基线的差距 |
| Prometheus | Micrometer + Prometheus | 指标采集方式完全相同 |
| Grafana | Grafana（Java也用） | 可视化工具通用 |
| ML CI/CD | Jenkins/GitHub Actions | 基础CI/CD相同，ML有额外步骤 |
| 质量门禁 | SonarQube质量门 | 代码质量门禁 vs 模型质量门禁 |
| A/B测试 | Feature Flag(功能开关) | 都是分流到不同逻辑 |
| 灰度发布 | 蓝绿部署/滚动部署 | 发布策略概念相同 |
| Shadow Mode | 并行运行测试 | 新旧系统并行，对比结果 |
| On-call手册 | 运维手册 | 应急响应流程类似 |
| 模型降级 | 服务降级(Hystrix) | 主服务不可用时降级到备选方案 |

### 关键思维转换

1. **模型也会"坏"**: 传统软件只要代码不变就能稳定运行，但模型会因数据变化而"退化"
2. **监控是必需品**: 模型上线后必须持续监控，就像微服务必须有健康检查
3. **发布要渐进**: 模型部署不能一刀切，需要灰度→A/B→全量的渐进流程
4. **回滚是常态**: 模型效果不佳时需要快速回滚，就像代码出bug时需要回滚版本
5. **数据也是资产**: 监控不仅要看系统指标，还要看数据分布变化
