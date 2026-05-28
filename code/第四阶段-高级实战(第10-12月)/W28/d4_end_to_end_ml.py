"""
W28-D4 端到端ML系统设计
========================
端到端ML系统设计, 系统架构, 数据流设计, API设计, 扩展性考虑
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import json
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W28-D4 端到端ML系统设计")
print("=" * 60)

# ============================================================
# 1. 端到端ML系统架构
# ============================================================
print("\n--- 1. 端到端ML系统架构 ---")
print("""
  ┌──────────────────────────────────────────────────────────────┐
  │                    端到端ML系统架构                           │
  │                                                              │
  │  客户端 -> API网关 -> 负载均衡 -> 模型服务(多副本)          │
  │                                                              │
  │  数据层:  数据湖 + 特征存储 + 缓存(Redis)                  │
  │  训练层:  实验追踪 + Pipeline + 超参优化                    │
  │  服务层:  模型服务 + A/B测试 + 监控                         │
  │  基础设施: K8s + Docker + CI/CD + 日志                      │
  └──────────────────────────────────────────────────────────────┘

  关键设计决策:
    1. 同步 vs 异步推理
    2. 批量 vs 实时处理
    3. 模型序列化格式 (Pickle/ONNX/PMML)
    4. 特征计算位置 (在线 vs 离线)
    5. 扩展策略 (水平 vs 垂直)
""")

# ============================================================
# 2. API设计
# ============================================================
print("\n--- 2. API设计 ---")


class MLPredictionAPI:
    """ML预测API (模拟REST API)"""

    def __init__(self, model, scaler, feature_names):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.request_count = 0
        self.error_count = 0
        self.latencies = []

    def _validate_input(self, features):
        """输入验证"""
        if not isinstance(features, list):
            raise ValueError("features must be a list")
        if len(features) != len(self.feature_names):
            raise ValueError(f"Expected {len(self.feature_names)} features, got {len(features)}")
        for i, f in enumerate(features):
            if not isinstance(f, (int, float)):
                raise ValueError(f"Feature {i} must be numeric, got {type(f)}")
            if np.isnan(f) or np.isinf(f):
                raise ValueError(f"Feature {i} has invalid value: {f}")

    def predict(self, request_body):
        """POST /predict - 单条预测"""
        self.request_count += 1
        t0 = time.time()

        try:
            features = request_body.get('features', [])
            self._validate_input(features)

            X = self.scaler.transform([features])
            prediction = int(self.model.predict(X)[0])
            probability = float(self.model.predict_proba(X)[0][prediction])

            latency = (time.time() - t0) * 1000
            self.latencies.append(latency)

            return {
                'status': 200,
                'body': {
                    'prediction': prediction,
                    'probability': round(probability, 4),
                    'label': 'malignant' if prediction == 0 else 'benign',
                    'latency_ms': round(latency, 2),
                    'request_id': self.request_count,
                }
            }

        except ValueError as e:
            self.error_count += 1
            return {'status': 400, 'body': {'error': str(e)}}
        except Exception as e:
            self.error_count += 1
            return {'status': 500, 'body': {'error': f'Internal error: {str(e)}'}}

    def predict_batch(self, request_body):
        """POST /predict/batch - 批量预测"""
        self.request_count += 1
        t0 = time.time()

        try:
            features_list = request_body.get('features_list', [])
            if not features_list:
                raise ValueError("features_list is empty")

            for features in features_list:
                self._validate_input(features)

            X = self.scaler.transform(features_list)
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)

            results = []
            for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
                results.append({
                    'prediction': int(pred),
                    'probability': float(probs[pred]),
                    'label': 'malignant' if pred == 0 else 'benign',
                })

            latency = (time.time() - t0) * 1000
            self.latencies.append(latency)

            return {
                'status': 200,
                'body': {
                    'predictions': results,
                    'count': len(results),
                    'latency_ms': round(latency, 2),
                }
            }
        except Exception as e:
            self.error_count += 1
            return {'status': 400, 'body': {'error': str(e)}}

    def health(self):
        """GET /health - 健康检查"""
        return {
            'status': 200,
            'body': {
                'status': 'healthy',
                'model_loaded': self.model is not None,
                'total_requests': self.request_count,
                'error_rate': self.error_count / max(self.request_count, 1),
            }
        }

    def metrics(self):
        """GET /metrics - 性能指标"""
        if not self.latencies:
            return {'status': 200, 'body': {'message': 'No data'}}

        return {
            'status': 200,
            'body': {
                'total_requests': self.request_count,
                'error_count': self.error_count,
                'error_rate': round(self.error_count / max(self.request_count, 1), 4),
                'latency': {
                    'mean_ms': round(np.mean(self.latencies), 2),
                    'p50_ms': round(np.percentile(self.latencies, 50), 2),
                    'p95_ms': round(np.percentile(self.latencies, 95), 2),
                    'p99_ms': round(np.percentile(self.latencies, 99), 2),
                }
            }
        }


# ============================================================
# 3. 初始化并测试API
# ============================================================
print("\n--- 3. 初始化并测试API ---")

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_s, y_train)

api = MLPredictionAPI(model, scaler, data.feature_names)

# 单条预测
print("\n  POST /predict:")
sample = X_test[0].tolist()
response = api.predict({'features': sample})
print(f"    状态: {response['status']}")
print(f"    预测: {response['body']['prediction']}")
print(f"    概率: {response['body']['probability']}")
print(f"    标签: {response['body']['label']}")

# 批量预测
print("\n  POST /predict/batch:")
batch_samples = X_test[:5].tolist()
response = api.predict_batch({'features_list': batch_samples})
print(f"    状态: {response['status']}")
print(f"    预测数: {response['body']['count']}")
for r in response['body']['predictions']:
    print(f"      {r['label']} (prob={r['probability']:.4f})")

# 错误处理
print("\n  POST /predict (错误输入):")
response = api.predict({'features': [1, 2, 3]})  # 特征数不对
print(f"    状态: {response['status']}")
print(f"    错误: {response['body']['error']}")

# 健康检查
print("\n  GET /health:")
response = api.health()
print(f"    {response['body']}")

# ============================================================
# 4. 性能测试
# ============================================================
print("\n--- 4. 性能测试 ---")

# 模拟1000次请求
np.random.seed(42)
for _ in range(1000):
    idx = np.random.randint(0, len(X_test))
    api.predict({'features': X_test[idx].tolist()})

# 性能报告
response = api.metrics()
print(f"\n  性能报告:")
print(f"    总请求: {response['body']['total_requests']}")
print(f"    错误率: {response['body']['error_rate']:.4%}")
print(f"    延迟统计:")
for k, v in response['body']['latency'].items():
    print(f"      {k}: {v}ms")

# ============================================================
# 5. 扩展性设计
# ============================================================
print("\n--- 5. 扩展性设计 ---")
print("""
  扩展策略:

  1. 水平扩展 (增加副本):
     - K8s HPA (Horizontal Pod Autoscaler)
     - 基于CPU/内存/自定义指标自动扩缩
     - 无状态设计, 易于扩展

  2. 垂直扩展 (增加资源):
     - 增大CPU/内存/GPU
     - 适合大模型推理

  3. 缓存策略:
     - 相同输入缓存结果 (Redis)
     - 特征预计算缓存
     - 模型预热

  4. 异步处理:
     - 高延迟操作异步化 (Celery/RMQ)
     - 批量预测队列
     - 流式处理 (Kafka)

  5. 模型优化:
     - ONNX Runtime加速
     - 模型蒸馏/量化
     - TensorRT (GPU推理)

  API设计最佳实践:
    - 无状态: 请求间不共享状态
    - 幂等性: 相同请求相同结果
    - 超时控制: 防止长时间阻塞
    - 限流: 保护系统不过载
    - 版本化: /v1/predict, /v2/predict
""")

# ============================================================
# 6. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 左上: API延迟分布
ax = axes[0, 0]
ax.hist(api.latencies, bins=50, color='#2196F3', alpha=0.7, edgecolor='white')
ax.axvline(np.percentile(api.latencies, 50), color='green', linestyle='--', label='P50')
ax.axvline(np.percentile(api.latencies, 95), color='orange', linestyle='--', label='P95')
ax.axvline(np.percentile(api.latencies, 99), color='red', linestyle='--', label='P99')
ax.set_xlabel('延迟 (ms)')
ax.set_ylabel('请求数')
ax.set_title('API延迟分布')
ax.legend()

# 右上: 系统架构图
ax = axes[0, 1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('端到端ML系统架构')

arch_components = [
    ('客户端', 1, 9, '#FFCDD2'),
    ('API网关', 1, 7.5, '#FFE0B2'),
    ('负载均衡', 1, 6, '#BBDEFB'),
    ('模型服务A', 3, 4.5, '#C8E6C9'),
    ('模型服务B', 5, 4.5, '#C8E6C9'),
    ('模型服务C', 7, 4.5, '#C8E6C9'),
    ('特征存储', 3, 2.5, '#E1BEE7'),
    ('Redis缓存', 5, 2.5, '#F8BBD0'),
    ('监控', 7, 2.5, '#FFF9C4'),
    ('数据湖', 5, 0.5, '#B2EBF2'),
]

for name, x, y, color in arch_components:
    ax.add_patch(plt.Rectangle((x - 0.8, y - 0.35), 1.6, 0.7,
                                facecolor=color, edgecolor='#333', linewidth=1))
    ax.text(x, y, name, ha='center', va='center', fontsize=8)

# 箭头
arrows = [(1, 8.6, 1, 7.85), (1, 7.15, 1, 6.35), (1, 5.65, 3, 4.85),
          (1, 5.65, 5, 4.85), (1, 5.65, 7, 4.85)]
for x1, y1, x2, y2 in arrows:
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#666', lw=1))

# 左下: 请求量趋势 (模拟)
ax = axes[1, 0]
hours = np.arange(24)
traffic = 500 + 300 * np.sin(np.pi * (hours - 6) / 12) + np.random.randn(24) * 50
traffic = np.clip(traffic, 100, 1000)
ax.plot(hours, traffic, 'b-', linewidth=2)
ax.fill_between(hours, traffic, alpha=0.2)
ax.set_xlabel('小时')
ax.set_ylabel('请求量')
ax.set_title('24小时请求量趋势 (模拟)')
ax.grid(True, alpha=0.3)

# 右下: 扩展策略对比
ax = axes[1, 1]
strategies = ['单实例', '水平扩展\n(3副本)', '水平扩展\n(5副本)', '+缓存', '+异步']
capacities = [100, 300, 500, 800, 1200]
costs = [1, 3, 5, 4, 5.5]
colors_s = ['#F44336', '#FF9800', '#FFC107', '#4CAF50', '#2196F3']

ax2 = ax.twinx()
bars = ax.bar(strategies, capacities, color=colors_s, alpha=0.7)
ax2.plot(strategies, costs, 'r-o', label='相对成本')
ax.set_ylabel('QPS容量')
ax2.set_ylabel('相对成本')
ax.set_title('扩展策略对比')
ax2.legend(loc='upper left')
for bar, cap in zip(bars, capacities):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 20,
            f'{cap}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W28/d4_end_to_end_ml.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存: d4_end_to_end_ml.png")

print("\n完成! 端到端ML系统设计要点:")
print("  1. 系统架构: API网关+负载均衡+模型服务+存储+监控")
print("  2. API设计: RESTful, 单条/批量预测, 健康检查, 指标")
print("  3. 输入验证: 类型/长度/范围检查, 错误处理")
print("  4. 扩展性: 水平/垂直扩展, 缓存, 异步")
print("  5. 性能: P99延迟, 吞吐量, 错误率监控")
