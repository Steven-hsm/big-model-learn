"""
W31-D5 监控系统
================
实现RAG系统的监控功能, 包括:
- 请求日志记录
- 性能指标收集
- 错误追踪
- 健康检查

监控是保障系统稳定运行的基础。
"""

import time
import json
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from enum import Enum

try:
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    HAS_PLT = True
except ImportError:
    HAS_PLT = False


# ============================================================
# 1. 请求日志
# ============================================================

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class RequestLog:
    """请求日志"""
    request_id: str
    method: str
    path: str
    status_code: int
    latency_ms: float
    timestamp: float = field(default_factory=time.time)
    user_id: str = "anonymous"
    error: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


class RequestLogger:
    """请求日志记录器"""

    def __init__(self, max_logs: int = 10000):
        self.logs: List[RequestLog] = []
        self.max_logs = max_logs
        self._lock = threading.Lock()

    def log_request(self, method: str, path: str, status_code: int,
                     latency_ms: float, **kwargs) -> RequestLog:
        """记录请求"""
        import uuid
        log = RequestLog(
            request_id=str(uuid.uuid4())[:8],
            method=method,
            path=path,
            status_code=status_code,
            latency_ms=latency_ms,
            **kwargs
        )
        with self._lock:
            self.logs.append(log)
            if len(self.logs) > self.max_logs:
                self.logs = self.logs[-self.max_logs:]
        return log

    def get_recent(self, n: int = 20) -> List[RequestLog]:
        return self.logs[-n:]

    def get_errors(self, n: int = 20) -> List[RequestLog]:
        return [log for log in self.logs if log.status_code >= 400][-n:]

    def get_stats(self) -> Dict:
        if not self.logs:
            return {'total': 0}
        latencies = [l.latency_ms for l in self.logs]
        status_counts = defaultdict(int)
        for l in self.logs:
            status_counts[l.status_code] += 1
        return {
            'total': len(self.logs),
            'avg_latency_ms': sum(latencies) / len(latencies),
            'max_latency_ms': max(latencies),
            'status_codes': dict(status_counts),
            'error_rate': status_counts.get(500, 0) / len(self.logs),
        }


# ============================================================
# 2. 性能指标
# ============================================================

class MetricsCollector:
    """性能指标收集器"""

    def __init__(self):
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def increment(self, name: str, value: float = 1):
        """递增计数器"""
        with self._lock:
            self.counters[name] += value

    def set_gauge(self, name: str, value: float):
        """设置仪表值"""
        with self._lock:
            self.gauges[name] = value

    def observe(self, name: str, value: float):
        """记录观测值(用于直方图/摘要)"""
        with self._lock:
            self.histograms[name].append(value)
            # 限制历史数据量
            if len(self.histograms[name]) > 10000:
                self.histograms[name] = self.histograms[name][-5000:]

    def get_metrics(self) -> Dict:
        """获取所有指标"""
        import statistics
        result = {
            'counters': dict(self.counters),
            'gauges': dict(self.gauges),
            'summaries': {},
        }

        for name, values in self.histograms.items():
            if values:
                sorted_vals = sorted(values)
                result['summaries'][name] = {
                    'count': len(values),
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'p95': sorted_vals[int(len(sorted_vals) * 0.95)],
                    'p99': sorted_vals[int(len(sorted_vals) * 0.99)],
                    'min': min(values),
                    'max': max(values),
                }

        return result

    def print_report(self):
        """打印指标报告"""
        metrics = self.get_metrics()
        print(f"\n{'='*60}")
        print("性能指标报告")
        print(f"{'='*60}")

        print("\n[计数器]")
        for name, value in metrics['counters'].items():
            print(f"  {name}: {value:.0f}")

        print("\n[仪表]")
        for name, value in metrics['gauges'].items():
            print(f"  {name}: {value:.2f}")

        print("\n[延迟摘要]")
        for name, s in metrics['summaries'].items():
            print(f"  {name}:")
            print(f"    平均={s['mean']:.2f}ms, P95={s['p95']:.2f}ms, "
                  f"P99={s['p99']:.2f}ms")


# ============================================================
# 3. 错误追踪
# ============================================================

@dataclass
class ErrorRecord:
    """错误记录"""
    error_id: str
    error_type: str
    message: str
    traceback: str
    timestamp: float = field(default_factory=time.time)
    path: str = ""
    request_data: Dict = field(default_factory=dict)
    resolved: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)


class ErrorTracker:
    """错误追踪器"""

    def __init__(self, max_errors: int = 1000):
        self.errors: List[ErrorRecord] = []
        self.max_errors = max_errors
        self.error_counts: Dict[str, int] = defaultdict(int)

    def record_error(self, error_type: str, message: str,
                      traceback: str = "", **kwargs) -> ErrorRecord:
        """记录错误"""
        import uuid
        record = ErrorRecord(
            error_id=str(uuid.uuid4())[:8],
            error_type=error_type,
            message=message,
            traceback=traceback,
            **kwargs
        )
        self.errors.append(record)
        self.error_counts[error_type] += 1

        if len(self.errors) > self.max_errors:
            self.errors = self.errors[-self.max_errors:]

        return record

    def get_recent(self, n: int = 10) -> List[ErrorRecord]:
        return self.errors[-n:]

    def get_by_type(self, error_type: str) -> List[ErrorRecord]:
        return [e for e in self.errors if e.error_type == error_type]

    def get_error_summary(self) -> Dict:
        """错误摘要"""
        return {
            'total_errors': len(self.errors),
            'unresolved': sum(1 for e in self.errors if not e.resolved),
            'error_types': dict(self.error_counts),
            'most_common': max(self.error_counts.items(), key=lambda x: x[1])[0]
                if self.error_counts else None,
        }

    def resolve(self, error_id: str):
        """标记错误为已解决"""
        for e in self.errors:
            if e.error_id == error_id:
                e.resolved = True
                break


# ============================================================
# 4. 健康检查
# ============================================================

class HealthChecker:
    """健康检查器"""

    def __init__(self):
        self.checks: Dict[str, Callable] = {}
        self.start_time = time.time()

    def register_check(self, name: str, check_func: Callable):
        """注册健康检查"""
        self.checks[name] = check_func

    def check_health(self) -> Dict:
        """执行所有健康检查"""
        results = {}
        overall_status = "healthy"

        for name, check_func in self.checks.items():
            try:
                start = time.time()
                result = check_func()
                latency = (time.time() - start) * 1000

                if result.get('status') == 'unhealthy':
                    overall_status = "degraded"

                results[name] = {
                    'status': result.get('status', 'healthy'),
                    'latency_ms': latency,
                    'details': result.get('details', ''),
                }
            except Exception as e:
                results[name] = {
                    'status': 'unhealthy',
                    'error': str(e),
                }
                overall_status = "degraded"

        return {
            'status': overall_status,
            'uptime_seconds': time.time() - self.start_time,
            'timestamp': time.time(),
            'components': results,
        }


# ============================================================
# 5. 监控系统(整合)
# ============================================================

class MonitoringSystem:
    """监控系统(整合所有监控组件)"""

    def __init__(self):
        self.logger = RequestLogger()
        self.metrics = MetricsCollector()
        self.errors = ErrorTracker()
        self.health = HealthChecker()

        # 注册健康检查
        self.health.register_check("logger", self._check_logger)
        self.health.register_check("metrics", self._check_metrics)
        self.health.register_check("memory", self._check_memory)

    def _check_logger(self) -> Dict:
        stats = self.logger.get_stats()
        error_rate = stats.get('error_rate', 0)
        return {
            'status': 'unhealthy' if error_rate > 0.1 else 'healthy',
            'details': f"错误率: {error_rate:.2%}",
        }

    def _check_metrics(self) -> Dict:
        return {'status': 'healthy', 'details': '指标收集正常'}

    def _check_memory(self) -> Dict:
        import sys
        return {
            'status': 'healthy',
            'details': f'日志数: {len(self.logger.logs)}, '
                       f'错误数: {len(self.errors.errors)}'
        }

    def record_request(self, method: str, path: str, status: int,
                        latency_ms: float, **kwargs):
        """记录请求(同时更新日志和指标)"""
        self.logger.log_request(method, path, status, latency_ms, **kwargs)
        self.metrics.increment('requests_total')
        self.metrics.increment(f'requests_{method.lower()}')
        self.metrics.observe('request_latency_ms', latency_ms)

        if status >= 400:
            self.metrics.increment('requests_error')

    def get_dashboard_data(self) -> Dict:
        """获取仪表盘数据"""
        return {
            'health': self.health.check_health(),
            'request_stats': self.logger.get_stats(),
            'error_summary': self.errors.get_error_summary(),
            'metrics': self.metrics.get_metrics(),
        }


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("W31-D5 监控系统")
    print("=" * 60)

    monitor = MonitoringSystem()

    # --- 模拟请求日志 ---
    print("\n--- 1. 请求日志 ---")
    import random
    random.seed(42)

    paths = ["/api/query", "/api/documents", "/api/health", "/api/stats"]
    methods = ["GET", "POST", "DELETE"]

    for i in range(100):
        path = random.choice(paths)
        method = random.choice(methods)
        status = random.choices([200, 200, 200, 200, 404, 500], weights=[50, 20, 10, 10, 5, 5])[0]
        latency = random.uniform(5, 200)
        monitor.record_request(method, path, status, latency)

    stats = monitor.logger.get_stats()
    print(f"总请求数: {stats['total']}")
    print(f"平均延迟: {stats['avg_latency_ms']:.1f}ms")
    print(f"最大延迟: {stats['max_latency_ms']:.1f}ms")
    print(f"状态码分布: {stats['status_codes']}")
    print(f"错误率: {stats['error_rate']:.2%}")

    # --- 性能指标 ---
    print(f"\n{'='*60}")
    print("--- 2. 性能指标 ---")
    print(f"{'='*60}")
    monitor.metrics.print_report()

    # --- 错误追踪 ---
    print(f"\n{'='*60}")
    print("--- 3. 错误追踪 ---")
    print(f"{'='*60}")

    error_types = [
        ("ValueError", "查询参数无效"),
        ("TimeoutError", "检索超时"),
        ("ConnectionError", "数据库连接失败"),
        ("ValueError", "文档格式不支持"),
        ("TimeoutError", "LLM调用超时"),
    ]
    for etype, msg in error_types:
        monitor.errors.record_error(etype, msg, path="/api/query")

    summary = monitor.errors.get_error_summary()
    print(f"总错误数: {summary['total_errors']}")
    print(f"未解决: {summary['unresolved']}")
    print(f"错误类型分布: {summary['error_types']}")
    print(f"最常见错误: {summary['most_common']}")

    # --- 健康检查 ---
    print(f"\n{'='*60}")
    print("--- 4. 健康检查 ---")
    print(f"{'='*60}")
    health = monitor.health.check_health()
    print(f"整体状态: {health['status']}")
    print(f"运行时间: {health['uptime_seconds']:.0f}秒")
    for name, result in health['components'].items():
        print(f"  {name}: {result['status']}")

    # --- 可视化 ---
    if HAS_PLT:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 请求延迟分布
        latencies = [l.latency_ms for l in monitor.logger.logs]
        axes[0, 0].hist(latencies, bins=30, color='#3498db', edgecolor='white')
        axes[0, 0].set_title('请求延迟分布', fontweight='bold')
        axes[0, 0].set_xlabel('延迟(ms)')
        axes[0, 0].set_ylabel('请求数')

        # 状态码分布
        status_codes = stats['status_codes']
        axes[0, 1].pie(status_codes.values(), labels=status_codes.keys(),
                        autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c', '#f39c12'])
        axes[0, 1].set_title('状态码分布', fontweight='bold')

        # 请求路径统计
        path_counts = defaultdict(int)
        for log in monitor.logger.logs:
            path_counts[log.path] += 1
        axes[1, 0].barh(list(path_counts.keys()), list(path_counts.values()),
                         color='#9b59b6')
        axes[1, 0].set_title('请求路径统计', fontweight='bold')
        axes[1, 0].set_xlabel('请求数')

        # 错误类型统计
        error_types_dict = summary['error_types']
        if error_types_dict:
            axes[1, 1].bar(error_types_dict.keys(), error_types_dict.values(),
                            color='#e74c3c')
            axes[1, 1].set_title('错误类型分布', fontweight='bold')
            axes[1, 1].set_ylabel('次数')
            axes[1, 1].tick_params(axis='x', rotation=15)

        plt.tight_layout()
        plt.savefig('D:/code/big-model-learn/code/q_01/W31/d5_monitoring.png', dpi=150)
        print("\n图表已保存为 d5_monitoring.png")
        plt.close()

    print("\n完成!")
