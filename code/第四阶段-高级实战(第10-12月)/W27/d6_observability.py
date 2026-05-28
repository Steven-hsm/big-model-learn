"""
W27-D6 ML可观测性
==================
ML可观测性, 日志记录, 分布式追踪(概念), 指标收集(Prometheus概念), Grafana仪表盘模板
"""

import os
import json
import time
import logging
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W27-D6 ML可观测性")
print("=" * 60)

# ============================================================
# 1. ML可观测性概述
# ============================================================
print("\n--- 1. ML可观测性概述 ---")
print("""
  可观测性三大支柱:
  ┌──────────────┬────────────────────────────────────────┐
  │ 支柱         │ ML场景中的应用                         │
  ├──────────────┼────────────────────────────────────────┤
  │ 日志(Logs)   │ 推理请求/响应、错误、数据变更         │
  │ 指标(Metrics)│ 延迟、准确率、请求量、资源使用        │
  │ 追踪(Traces) │ 请求在Pipeline中的完整路径            │
  └──────────────┴────────────────────────────────────────┘

  工具链:
    - Prometheus: 指标收集和告警
    - Grafana:    可视化仪表盘
    - ELK Stack:  日志聚合和搜索
    - Jaeger:     分布式追踪
    - OpenTelemetry: 统一可观测性框架
""")

# ============================================================
# 2. 结构化日志
# ============================================================
print("\n--- 2. 结构化日志 ---")


class MLLogger:
    """ML结构化日志记录器"""

    def __init__(self, service_name='ml-service'):
        self.service_name = service_name
        self.logs = []
        self.request_counter = 0

    def _log(self, level, event, data=None):
        entry = {
            'timestamp': datetime.now().isoformat(),
            'service': self.service_name,
            'level': level,
            'event': event,
            'request_id': self.request_counter,
            **(data or {}),
        }
        self.logs.append(entry)

    def log_prediction(self, input_features, prediction, probability, latency_ms):
        """记录预测请求"""
        self.request_counter += 1
        self._log('INFO', 'prediction', {
            'input_hash': hash(str(input_features[:5])) & 0xFFFFFFFF,
            'prediction': int(prediction),
            'probability': float(probability),
            'latency_ms': round(latency_ms, 3),
        })

    def log_error(self, error_type, message, stack_trace=''):
        """记录错误"""
        self._log('ERROR', 'error', {
            'error_type': error_type,
            'message': message,
            'stack_trace': stack_trace[:200],
        })

    def log_data_quality(self, issues):
        """记录数据质量问题"""
        self._log('WARNING', 'data_quality', {'issues': issues})

    def log_model_event(self, event, details):
        """记录模型事件 (加载/更新/回滚)"""
        self._log('INFO', f'model_{event}', details)

    def search(self, level=None, event=None, limit=10):
        """搜索日志"""
        results = self.logs
        if level:
            results = [l for l in results if l['level'] == level]
        if event:
            results = [l for l in results if event in l['event']]
        return results[-limit:]

    def print_recent(self, n=10):
        """打印最近的日志"""
        print(f"\n  最近 {n} 条日志:")
        for entry in self.logs[-n:]:
            ts = entry['timestamp'][:19]
            level = entry['level']
            event = entry['event']
            data_str = ', '.join(f'{k}={v}' for k, v in entry.items()
                                  if k not in ['timestamp', 'service', 'level', 'event', 'request_id'])
            print(f"  [{ts}] {level:<7} {event:<20} req={entry['request_id']:<5} {data_str}")


# ============================================================
# 3. 指标收集 (Prometheus概念)
# ============================================================
print("\n--- 3. 指标收集 (Prometheus概念) ---")
print("""
  Prometheus 指标类型:
    - Counter:   单调递增计数器 (如: 请求总数, 错误总数)
    - Gauge:     可增减的值 (如: 当前内存使用, 活跃连接数)
    - Histogram: 分布统计 (如: 请求延迟分布)
    - Summary:   分位数统计 (如: P50, P95, P99延迟)

  示例:
    prediction_requests_total{model="v1",status="success"} 12345
    prediction_latency_seconds{quantile="0.99"} 0.023
    model_accuracy_gauge{model="v1"} 0.9543
""")


class MetricsCollector:
    """简化版指标收集器"""

    def __init__(self):
        self.counters = defaultdict(int)
        self.gauges = {}
        self.histograms = defaultdict(list)

    def inc_counter(self, name, labels=None, value=1):
        key = f"{name}_{labels}" if labels else name
        self.counters[key] += value

    def set_gauge(self, name, value, labels=None):
        key = f"{name}_{labels}" if labels else name
        self.gauges[key] = value

    def observe_histogram(self, name, value, labels=None):
        key = f"{name}_{labels}" if labels else name
        self.histograms[key].append(value)

    def get_counter(self, name, labels=None):
        key = f"{name}_{labels}" if labels else name
        return self.counters.get(key, 0)

    def get_gauge(self, name, labels=None):
        key = f"{name}_{labels}" if labels else name
        return self.gauges.get(key, 0)

    def get_percentile(self, name, percentile, labels=None):
        key = f"{name}_{labels}" if labels else name
        values = self.histograms.get(key, [])
        if not values:
            return 0
        return np.percentile(values, percentile)

    def print_metrics(self):
        """打印所有指标"""
        print(f"\n  === Counters ===")
        for key, value in sorted(self.counters.items()):
            print(f"    {key}: {value}")

        print(f"\n  === Gauges ===")
        for key, value in sorted(self.gauges.items()):
            print(f"    {key}: {value}")

        print(f"\n  === Histograms ===")
        for key, values in sorted(self.histograms.items()):
            if values:
                print(f"    {key}: count={len(values)}, "
                      f"p50={np.percentile(values, 50):.4f}, "
                      f"p95={np.percentile(values, 95):.4f}, "
                      f"p99={np.percentile(values, 99):.4f}")


# ============================================================
# 4. 分布式追踪 (概念)
# ============================================================
print("\n--- 4. 分布式追踪 ---")


class SimpleTracer:
    """简化版分布式追踪器"""

    def __init__(self):
        self.traces = {}
        self.trace_counter = 0

    def start_trace(self, operation_name):
        self.trace_counter += 1
        trace_id = f'trace_{self.trace_counter:06d}'
        self.traces[trace_id] = {
            'trace_id': trace_id,
            'operation': operation_name,
            'start_time': time.time(),
            'spans': [],
        }
        return trace_id

    def start_span(self, trace_id, span_name):
        span = {
            'name': span_name,
            'start_time': time.time(),
            'end_time': None,
            'duration_ms': None,
        }
        self.traces[trace_id]['spans'].append(span)
        return span

    def end_span(self, span):
        span['end_time'] = time.time()
        span['duration_ms'] = (span['end_time'] - span['start_time']) * 1000

    def end_trace(self, trace_id):
        trace = self.traces[trace_id]
        trace['duration_ms'] = (time.time() - trace['start_time']) * 1000

    def print_trace(self, trace_id):
        trace = self.traces[trace_id]
        print(f"\n  Trace: {trace['operation']} (总耗时: {trace['duration_ms']:.2f}ms)")
        for span in trace['spans']:
            indent = '    '
            print(f"  {indent}{span['name']}: {span['duration_ms']:.2f}ms")

    def get_waterfall_data(self, trace_id):
        """获取瀑布图数据"""
        trace = self.traces[trace_id]
        data = []
        t0 = trace['start_time']
        for span in trace['spans']:
            data.append({
                'name': span['name'],
                'start_offset_ms': (span['start_time'] - t0) * 1000,
                'duration_ms': span['duration_ms'],
            })
        return data


# ============================================================
# 5. 模拟完整的可观测性系统
# ============================================================
print("\n--- 5. 模拟完整的可观测性系统 ---")

# 准备模型
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 初始化组件
logger = MLLogger('breast-cancer-predictor')
metrics = MetricsCollector()
tracer = SimpleTracer()

# 模拟推理请求
np.random.seed(42)
for i in range(50):
    # 开始追踪
    trace_id = tracer.start_trace('prediction_request')

    # Span 1: 特征预处理
    span_preprocess = tracer.start_span(trace_id, 'feature_preprocessing')
    time.sleep(0.001)
    tracer.end_span(span_preprocess)

    # Span 2: 特征验证
    span_validate = tracer.start_span(trace_id, 'feature_validation')
    sample = X_test[i % len(X_test):i % len(X_test) + 1]
    tracer.end_span(span_validate)

    # Span 3: 模型推理
    span_infer = tracer.start_span(trace_id, 'model_inference')
    t0 = time.time()
    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0][prediction]
    latency = (time.time() - t0) * 1000
    tracer.end_span(span_infer)

    # 记录日志
    logger.log_prediction(sample[0], prediction, probability, latency)

    # 记录指标
    status = 'success'
    metrics.inc_counter('prediction_requests_total', labels=status)
    metrics.observe_histogram('prediction_latency_ms', latency)
    metrics.set_gauge('last_prediction_probability', probability)

    # Span 4: 后处理
    span_post = tracer.start_span(trace_id, 'postprocessing')
    tracer.end_span(span_post)

    tracer.end_trace(trace_id)

# 模拟一些错误
for _ in range(3):
    logger.log_error('ValueError', 'Invalid input shape')
    metrics.inc_counter('prediction_requests_total', labels='error')
    metrics.inc_counter('prediction_errors_total')

# 更新模型指标
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
metrics.set_gauge('model_accuracy', acc)

# 打印结果
logger.print_recent(5)
metrics.print_metrics()

# 打印一个追踪
if tracer.traces:
    first_trace = list(tracer.traces.keys())[0]
    tracer.print_trace(first_trace)

# ============================================================
# 6. Grafana仪表盘模板 (概念)
# ============================================================
print("\n--- 6. Grafana仪表盘模板 ---")
print("""
  推荐的ML监控仪表盘面板:

  Row 1: 概览
    - 请求总数 (Stat)
    - 错误率 (Stat)
    - 平均延迟 (Stat)
    - 模型准确率 (Stat)

  Row 2: 性能
    - 延迟分布 (Histogram -> Heatmap)
    - P50/P95/P99趋势 (Time series)
    - 请求QPS (Time series)

  Row 3: 模型质量
    - 预测分布 (Pie chart)
    - 置信度分布 (Histogram)
    - 准确率趋势 (Time series)

  Row 4: 资源
    - CPU使用率 (Gauge)
    - 内存使用 (Time series)
    - GPU利用率 (Gauge)

  告警规则:
    - 错误率 > 1%: WARNING
    - P99延迟 > 200ms: WARNING
    - 准确率 < 90%: CRITICAL
    - 请求量下降 > 50%: INFO
""")

# ============================================================
# 7. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 左上: 请求延迟分布
ax = axes[0, 0]
latencies = metrics.histograms.get('prediction_latency_ms_success', [])
if not latencies:
    latencies = list(metrics.histograms.values())[0] if metrics.histograms else []
if latencies:
    ax.hist(latencies, bins=20, color='#2196F3', alpha=0.7, edgecolor='white')
    ax.axvline(np.percentile(latencies, 50), color='green', linestyle='--', label='P50')
    ax.axvline(np.percentile(latencies, 95), color='orange', linestyle='--', label='P95')
    ax.axvline(np.percentile(latencies, 99), color='red', linestyle='--', label='P99')
    ax.set_xlabel('延迟 (ms)')
    ax.set_ylabel('请求数')
    ax.set_title('推理延迟分布')
    ax.legend()

# 右上: 追踪瀑布图
ax = axes[0, 1]
if tracer.traces:
    first_trace_id = list(tracer.traces.keys())[0]
    waterfall = tracer.get_waterfall_data(first_trace_id)
    names = [w['name'] for w in waterfall]
    starts = [w['start_offset_ms'] for w in waterfall]
    durations = [w['duration_ms'] for w in waterfall]
    ax.barh(names, durations, left=starts, color=['#2196F3', '#4CAF50', '#FF9800', '#9C27B0'])
    ax.set_xlabel('时间 (ms)')
    ax.set_title('请求追踪瀑布图')

# 左下: 请求量趋势
ax = axes[1, 0]
pred_logs = logger.search(event='prediction')
if pred_logs:
    time_buckets = defaultdict(int)
    for log in pred_logs:
        bucket = log['timestamp'][:16]  # 按分钟聚合
        time_buckets[bucket] += 1

    times = list(time_buckets.keys())
    counts = list(time_buckets.values())
    ax.plot(range(len(counts)), counts, 'b-', linewidth=2)
    ax.fill_between(range(len(counts)), counts, alpha=0.2, color='#2196F3')
    ax.set_xlabel('时间')
    ax.set_ylabel('请求数')
    ax.set_title('请求量趋势')

# 右下: 错误率
ax = axes[1, 1]
total = metrics.get_counter('prediction_requests_total', 'success')
errors = metrics.get_counter('prediction_requests_total', 'error')
error_rate = errors / (total + errors) if (total + errors) > 0 else 0

labels = ['成功', '错误']
sizes = [total, errors]
colors = ['#4CAF50', '#F44336']
ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
ax.set_title(f'请求状态分布 (错误率={error_rate:.1%})')

plt.suptitle('ML可观测性仪表盘', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W27/d6_observability.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存: d6_observability.png")

print("\n完成! ML可观测性要点:")
print("  1. 日志: 结构化记录推理请求和错误")
print("  2. 指标: Counter/Gauge/Histogram收集性能数据")
print("  3. 追踪: 请求在Pipeline中的完整路径")
print("  4. Grafana: 可视化仪表盘和告警")
print("  5. 三大支柱结合: 日志+指标+追踪=完整的可观测性")
