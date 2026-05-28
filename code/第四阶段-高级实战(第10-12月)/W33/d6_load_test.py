"""
W33-D6 压力测试
================
实现RAG系统的压力测试, 包括:
- 并发测试模拟
- 延迟分布分析
- 瓶颈识别

压力测试帮助发现系统在高负载下的表现。
"""

import time
import threading
import statistics
import hashlib
from typing import List, Dict, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

import numpy as np


# ============================================================
# 1. 负载生成器
# ============================================================

@dataclass
class LoadTestConfig:
    """压力测试配置"""
    name: str
    concurrency: int = 10       # 并发数
    total_requests: int = 100   # 总请求数
    ramp_up_seconds: float = 1  # 逐步增加并发的时间
    think_time_ms: float = 0    # 请求间隔(模拟用户思考时间)


@dataclass
class RequestResult:
    """请求结果"""
    request_id: int
    start_time: float
    end_time: float
    latency_ms: float
    status: str  # 'success' / 'error'
    error: str = ""

    @property
    def success(self) -> bool:
        return self.status == 'success'


class LoadGenerator:
    """负载生成器"""

    def __init__(self, target_func: Callable):
        """
        参数:
            target_func: 被测函数, 接受任意参数, 返回结果
        """
        self.target_func = target_func
        self.results: List[RequestResult] = []
        self._lock = threading.Lock()

    def run(self, config: LoadTestConfig) -> Dict:
        """运行压力测试"""
        self.results = []
        print(f"\n开始压力测试: {config.name}")
        print(f"  并发数: {config.concurrency}, 总请求: {config.total_requests}")

        start_time = time.time()
        completed = 0

        with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
            futures = {}

            for i in range(config.total_requests):
                # 逐步增加并发(ramp up)
                if config.ramp_up_seconds > 0 and config.concurrency > 1:
                    progress = i / config.total_requests
                    if progress < 1.0:
                        delay = config.ramp_up_seconds / config.total_requests
                        time.sleep(delay)

                # 思考时间
                if config.think_time_ms > 0:
                    time.sleep(config.think_time_ms / 1000)

                future = executor.submit(self._execute_request, i)
                futures[future] = i

            for future in as_completed(futures):
                result = future.result()
                with self._lock:
                    self.results.append(result)
                    completed += 1
                    if completed % 20 == 0:
                        print(f"  进度: {completed}/{config.total_requests}")

        total_time = time.time() - start_time
        return self._analyze(config, total_time)

    def _execute_request(self, request_id: int) -> RequestResult:
        """执行单个请求"""
        start = time.time()
        try:
            self.target_func()
            status = 'success'
            error = ""
        except Exception as e:
            status = 'error'
            error = str(e)
        end = time.time()

        return RequestResult(
            request_id=request_id,
            start_time=start,
            end_time=end,
            latency_ms=(end - start) * 1000,
            status=status,
            error=error,
        )

    def _analyze(self, config: LoadTestConfig, total_time: float) -> Dict:
        """分析测试结果"""
        if not self.results:
            return {'error': 'no results'}

        latencies = [r.latency_ms for r in self.results]
        successes = [r for r in self.results if r.success]
        errors = [r for r in self.results if not r.success]

        sorted_latencies = sorted(latencies)

        analysis = {
            'config': config.name,
            'total_requests': len(self.results),
            'total_time_s': total_time,
            'success_count': len(successes),
            'error_count': len(errors),
            'error_rate': len(errors) / len(self.results),
            'throughput_rps': len(self.results) / total_time,
            'latency': {
                'mean_ms': statistics.mean(latencies),
                'median_ms': statistics.median(latencies),
                'p50_ms': sorted_latencies[int(len(sorted_latencies) * 0.50)],
                'p90_ms': sorted_latencies[int(len(sorted_latencies) * 0.90)],
                'p95_ms': sorted_latencies[int(len(sorted_latencies) * 0.95)],
                'p99_ms': sorted_latencies[int(len(sorted_latencies) * 0.99)],
                'min_ms': min(latencies),
                'max_ms': max(latencies),
            },
        }

        return analysis

    @staticmethod
    def print_report(analysis: Dict):
        """打印测试报告"""
        print(f"\n{'='*60}")
        print(f"压力测试报告: {analysis['config']}")
        print(f"{'='*60}")
        print(f"总请求数: {analysis['total_requests']}")
        print(f"总耗时: {analysis['total_time_s']:.2f}秒")
        print(f"成功: {analysis['success_count']}, 失败: {analysis['error_count']}")
        print(f"错误率: {analysis['error_rate']:.2%}")
        print(f"吞吐量: {analysis['throughput_rps']:.1f} req/s")
        print(f"\n延迟分布:")
        lat = analysis['latency']
        print(f"  平均: {lat['mean_ms']:.1f}ms")
        print(f"  中位: {lat['median_ms']:.1f}ms")
        print(f"  P90:  {lat['p90_ms']:.1f}ms")
        print(f"  P95:  {lat['p95_ms']:.1f}ms")
        print(f"  P99:  {lat['p99_ms']:.1f}ms")
        print(f"  最小: {lat['min_ms']:.1f}ms, 最大: {lat['max_ms']:.1f}ms")


# ============================================================
# 2. 瓶颈分析器
# ============================================================

class BottleneckAnalyzer:
    """瓶颈分析器"""

    @staticmethod
    def analyze_latency_distribution(latencies: List[float]) -> Dict:
        """分析延迟分布"""
        sorted_lat = sorted(latencies)
        n = len(sorted_lat)

        # 分段分析
        segments = {
            'fast': sum(1 for l in sorted_lat if l < 100),
            'normal': sum(1 for l in sorted_lat if 100 <= l < 500),
            'slow': sum(1 for l in sorted_lat if 500 <= l < 1000),
            'very_slow': sum(1 for l in sorted_lat if l >= 1000),
        }

        return {
            'distribution': segments,
            'tail_latency_ratio': segments['slow'] + segments['very_slow'],
            'tail_latency_pct': (segments['slow'] + segments['very_slow']) / n,
        }

    @staticmethod
    def identify_bottleneck(component_latencies: Dict[str, List[float]]) -> str:
        """识别瓶颈组件"""
        avg_latencies = {
            name: statistics.mean(lat) for name, lat in component_latencies.items()
        }
        bottleneck = max(avg_latencies, key=avg_latencies.get)
        return f"瓶颈组件: {bottleneck} (平均{avg_latencies[bottleneck]:.1f}ms)"


# ============================================================
# 3. 模拟被测系统
# ============================================================

class MockRAGSystem:
    """模拟RAG系统(用于压力测试)"""

    def __init__(self, base_latency_ms: float = 50, error_rate: float = 0.02):
        self.base_latency = base_latency_ms
        self.error_rate = error_rate
        self.request_count = 0
        self._lock = threading.Lock()

    def query(self) -> Dict:
        """模拟查询"""
        with self._lock:
            self.request_count += 1

        # 模拟处理延迟
        latency = self.base_latency_ms * np.random.uniform(0.5, 3.0) / 1000
        time.sleep(max(latency, 0.001))

        # 模拟偶发错误
        if np.random.random() < self.error_rate:
            raise Exception("模拟服务错误")

        return {'answer': '测试回答'}


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("W33-D6 压力测试")
    print("=" * 60)

    mock_system = MockRAGSystem(base_latency_ms=30, error_rate=0.01)
    generator = LoadGenerator(target_func=mock_system.query)

    # --- 1. 不同并发级别测试 ---
    print("\n--- 1. 并发级别对比 ---")
    all_results = {}

    for concurrency in [1, 5, 10, 20, 50]:
        config = LoadTestConfig(
            name=f"并发{concurrency}",
            concurrency=concurrency,
            total_requests=100,
            ramp_up_seconds=0.5,
        )
        result = generator.run(config)
        all_results[f"c={concurrency}"] = result
        LoadGenerator.print_report(result)

    # --- 2. 瓶颈分析 ---
    print(f"\n{'='*60}")
    print("--- 2. 瓶颈分析 ---")
    print(f"{'='*60}")

    component_latencies = {
        '检索': [np.random.uniform(20, 80) for _ in range(50)],
        '重排序': [np.random.uniform(10, 30) for _ in range(50)],
        'LLM生成': [np.random.uniform(100, 500) for _ in range(50)],
        '后处理': [np.random.uniform(5, 15) for _ in range(50)],
    }

    for name, latencies in component_latencies.items():
        print(f"  {name}: 平均={statistics.mean(latencies):.1f}ms")

    bottleneck = BottleneckAnalyzer.identify_bottleneck(component_latencies)
    print(f"\n  {bottleneck}")

    # --- 可视化 ---
    if HAS_PLT:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 吞吐量 vs 并发
        ax = axes[0, 0]
        concurrencies = [1, 5, 10, 20, 50]
        throughputs = [all_results[f'c={c}']['throughput_rps'] for c in concurrencies]
        ax.plot(concurrencies, throughputs, 'o-', color='#3498db', linewidth=2, markersize=8)
        ax.set_xlabel('并发数')
        ax.set_ylabel('吞吐量(req/s)')
        ax.set_title('吞吐量 vs 并发数', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 延迟分布
        ax = axes[0, 1]
        for c in [1, 10, 50]:
            key = f'c={c}'
            lat = all_results[key]['latency']
            metrics = ['mean_ms', 'p50_ms', 'p90_ms', 'p95_ms', 'p99_ms']
            values = [lat[m] for m in metrics]
            ax.plot(metrics, values, 'o-', label=f'并发{c}', linewidth=2)
        ax.set_ylabel('延迟(ms)')
        ax.set_title('延迟分布(不同并发)', fontweight='bold')
        ax.legend()

        # 错误率
        ax = axes[1, 0]
        error_rates = [all_results[f'c={c}']['error_rate'] * 100 for c in concurrencies]
        ax.bar([str(c) for c in concurrencies], error_rates, color='#e74c3c')
        ax.set_xlabel('并发数')
        ax.set_ylabel('错误率(%)')
        ax.set_title('错误率 vs 并发数', fontweight='bold')

        # 组件延迟
        ax = axes[1, 1]
        names = list(component_latencies.keys())
        means = [statistics.mean(v) for v in component_latencies.values()]
        ax.bar(names, means, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
        ax.set_ylabel('平均延迟(ms)')
        ax.set_title('组件延迟分析', fontweight='bold')

        plt.tight_layout()
        plt.savefig('D:/code/big-model-learn/code/q_01/W33/d6_load_test.png', dpi=150)
        print("\n图表已保存为 d6_load_test.png")
        plt.close()

    print("\n完成!")
