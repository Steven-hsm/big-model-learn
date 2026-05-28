"""
W31-D4 性能优化
================
实现RAG系统的性能优化策略, 包括:
- 批量处理
- 缓存策略(LRU/TTL)
- 异步处理模式
- 性能分析与调优

性能优化是生产级RAG系统的关键。
"""

import time
import hashlib
import json
import threading
import statistics
from typing import Dict, List, Any, Optional, Callable
from collections import OrderedDict
from functools import wraps
from dataclasses import dataclass, field

try:
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    HAS_PLT = True
except ImportError:
    HAS_PLT = False


# ============================================================
# 1. 批量处理
# ============================================================

class BatchProcessor:
    """批量处理器

    将多个小请求合并为一个批量请求, 减少IO开销。
    适用于: 批量embedding、批量检索等场景。
    """

    def __init__(self, batch_size: int = 32, max_wait: float = 0.1):
        self.batch_size = batch_size
        self.max_wait = max_wait  # 最大等待时间(秒)
        self.pending = []
        self.results = {}
        self._lock = threading.Lock()

    def process_batch(self, items: List[Any]) -> List[Any]:
        """模拟批量处理(如批量embedding)"""
        # 模拟处理时间: 固定开销 + 每项时间
        base_time = 0.01   # 固定开销
        per_item_time = 0.001  # 每项时间
        time.sleep(base_time + per_item_time * len(items))

        # 模拟返回结果
        return [f"result_{i}" for i in range(len(items))]

    def process_single_vs_batch(self, count: int):
        """对比单个处理 vs 批量处理"""
        items = [f"item_{i}" for i in range(count)]

        # 单个处理
        start = time.time()
        for item in items:
            self.process_batch([item])
        single_time = time.time() - start

        # 批量处理
        start = time.time()
        self.process_batch(items)
        batch_time = time.time() - start

        speedup = single_time / batch_time if batch_time > 0 else 0
        return {
            'count': count,
            'single_time_ms': single_time * 1000,
            'batch_time_ms': batch_time * 1000,
            'speedup': speedup,
        }


# ============================================================
# 2. 缓存策略
# ============================================================

class LRUCache:
    """LRU (Least Recently Used) 缓存

    当缓存满时, 淘汰最久未使用的数据。
    适用于: 查询结果缓存, embedding缓存。
    """

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.cache: OrderedDict = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, key: str, value: Any):
        """写入缓存"""
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0

    def stats(self) -> Dict:
        return {
            'size': len(self.cache),
            'capacity': self.capacity,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hit_rate,
        }


class TTLCache:
    """TTL (Time To Live) 缓存

    数据在指定时间后自动失效。
    适用于: 实时性要求较高的查询结果缓存。
    """

    def __init__(self, capacity: int = 1000, ttl: float = 300):
        self.capacity = capacity
        self.ttl = ttl  # 生存时间(秒)
        self.cache: Dict[str, tuple] = {}  # key -> (value, expire_time)
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            value, expire_time = self.cache[key]
            if time.time() < expire_time:
                self.hits += 1
                return value
            else:
                del self.cache[key]
        self.misses += 1
        return None

    def put(self, key: str, value: Any, ttl: float = None):
        expire_time = time.time() + (ttl or self.ttl)
        self.cache[key] = (value, expire_time)
        if len(self.cache) > self.capacity:
            self._evict()

    def _evict(self):
        """清理过期和超出容量的缓存"""
        now = time.time()
        expired = [k for k, (_, exp) in self.cache.items() if now >= exp]
        for k in expired:
            del self.cache[k]
        if len(self.cache) > self.capacity:
            oldest = list(self.cache.keys())[:len(self.cache) - self.capacity]
            for k in oldest:
                del self.cache[k]

    def stats(self) -> Dict:
        return {
            'size': len(self.cache),
            'capacity': self.capacity,
            'ttl': self.ttl,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0,
        }


class CachedRAGQuery:
    """带缓存的RAG查询"""

    def __init__(self, cache_capacity: int = 500, ttl: float = 60):
        self.cache = TTLCache(capacity=cache_capacity, ttl=ttl)
        self.query_count = 0

    def _compute_cache_key(self, query: str, top_k: int) -> str:
        """计算缓存键"""
        return hashlib.md5(f"{query}_{top_k}".encode()).hexdigest()

    def query(self, query: str, top_k: int = 5) -> Dict:
        """执行查询(带缓存)"""
        self.query_count += 1
        cache_key = self._compute_cache_key(query, top_k)

        # 检查缓存
        cached = self.cache.get(cache_key)
        if cached is not None:
            cached['from_cache'] = True
            return cached

        # 模拟实际查询
        start = time.time()
        time.sleep(0.05)  # 模拟查询延迟
        result = {
            'query': query,
            'answer': f"关于「{query}」的回答...",
            'sources': [{'title': '文档A', 'score': 0.95}],
            'latency_ms': (time.time() - start) * 1000,
            'from_cache': False,
        }

        self.cache.put(cache_key, result)
        return result


# ============================================================
# 3. 异步处理模式
# ============================================================

class AsyncTaskQueue:
    """异步任务队列

    使用线程池模拟异步处理。
    适用于: 文档索引、批量处理等耗时操作。
    """

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.tasks: Dict[str, Dict] = {}
        self.completed: List[Dict] = []

    def submit(self, task_id: str, func: Callable, *args, **kwargs) -> str:
        """提交任务"""
        self.tasks[task_id] = {
            'id': task_id,
            'status': 'pending',
            'started_at': None,
            'completed_at': None,
            'result': None,
            'error': None,
        }

        def run_task():
            self.tasks[task_id]['status'] = 'running'
            self.tasks[task_id]['started_at'] = time.time()
            try:
                result = func(*args, **kwargs)
                self.tasks[task_id]['status'] = 'completed'
                self.tasks[task_id]['result'] = result
            except Exception as e:
                self.tasks[task_id]['status'] = 'failed'
                self.tasks[task_id]['error'] = str(e)
            self.tasks[task_id]['completed_at'] = time.time()
            self.completed.append(self.tasks[task_id])

        thread = threading.Thread(target=run_task)
        thread.start()
        return task_id

    def get_status(self, task_id: str) -> Dict:
        return self.tasks.get(task_id, {'status': 'not_found'})

    def wait_all(self, timeout: float = 30) -> bool:
        """等待所有任务完成"""
        start = time.time()
        while time.time() - start < timeout:
            if all(t['status'] in ('completed', 'failed')
                   for t in self.tasks.values()):
                return True
            time.sleep(0.1)
        return False


# ============================================================
# 4. 性能分析器
# ============================================================

class Profiler:
    """性能分析器"""

    def __init__(self):
        self.records: Dict[str, List[float]] = {}

    def record(self, name: str, duration_ms: float):
        if name not in self.records:
            self.records[name] = []
        self.records[name].append(duration_ms)

    def time_it(self, name: str):
        """装饰器: 记录函数执行时间"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start = time.time()
                result = func(*args, **kwargs)
                duration = (time.time() - start) * 1000
                self.record(name, duration)
                return result
            return wrapper
        return decorator

    def summary(self) -> Dict:
        """生成分析摘要"""
        summary = {}
        for name, durations in self.records.items():
            summary[name] = {
                'count': len(durations),
                'mean_ms': statistics.mean(durations),
                'median_ms': statistics.median(durations),
                'p95_ms': sorted(durations)[int(len(durations) * 0.95)] if durations else 0,
                'min_ms': min(durations),
                'max_ms': max(durations),
                'total_ms': sum(durations),
            }
        return summary

    def print_report(self):
        """打印性能报告"""
        print(f"\n{'='*70}")
        print(f"{'性能分析报告':^70}")
        print(f"{'='*70}")
        print(f"{'操作':<20} {'次数':<8} {'平均(ms)':<12} {'P95(ms)':<12} {'总计(ms)':<12}")
        print("-" * 70)
        for name, s in self.summary().items():
            print(f"{name:<20} {s['count']:<8} {s['mean_ms']:<12.2f} "
                  f"{s['p95_ms']:<12.2f} {s['total_ms']:<12.2f}")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("W31-D4 性能优化")
    print("=" * 60)

    profiler = Profiler()

    # --- 1. 批量处理 ---
    print("\n--- 1. 批量处理对比 ---")
    processor = BatchProcessor()
    for count in [10, 50, 100, 500]:
        result = processor.process_single_vs_batch(count)
        print(f"  {count}项: 单个={result['single_time_ms']:.1f}ms, "
              f"批量={result['batch_time_ms']:.1f}ms, 加速={result['speedup']:.1f}x")

    # --- 2. 缓存策略 ---
    print(f"\n{'='*60}")
    print("--- 2. 缓存策略 ---")
    print(f"{'='*60}")

    # LRU缓存测试
    lru = LRUCache(capacity=100)
    for i in range(200):
        lru.put(f"key_{i % 150}", f"value_{i}")
    for i in range(300):
        lru.get(f"key_{i % 100}")
    print(f"\nLRU缓存: {lru.stats()}")

    # 带缓存的查询
    print("\n--- 缓存查询对比 ---")
    cached_rag = CachedRAGQuery()
    queries = ["RAG是什么", "Python优势", "RAG是什么", "向量检索", "Python优势", "RAG是什么"]

    for q in queries:
        result = cached_rag.query(q)
        cache_status = "缓存命中" if result['from_cache'] else "实际查询"
        print(f"  查询「{q}」: {cache_status}, 延迟={result['latency_ms']:.1f}ms")

    print(f"\n缓存统计: {cached_rag.cache.stats()}")

    # --- 3. 异步处理 ---
    print(f"\n{'='*60}")
    print("--- 3. 异步处理 ---")
    print(f"{'='*60}")

    queue = AsyncTaskQueue(max_workers=4)

    def mock_index_task(doc_name, duration):
        time.sleep(duration)
        return f"{doc_name} 索引完成"

    # 提交多个任务
    for i in range(5):
        queue.submit(f"task_{i}", mock_index_task, f"文档{i}", 0.1 * (i + 1))

    print("已提交5个索引任务...")
    queue.wait_all(timeout=10)

    for task_id, task in queue.tasks.items():
        status = task['status']
        duration = task['completed_at'] - task['started_at'] if task['completed_at'] else 0
        print(f"  {task_id}: {status} ({duration:.2f}s)")

    # --- 4. 性能分析 ---
    print(f"\n{'='*60}")
    print("--- 4. 性能分析 ---")
    print(f"{'='*60}")

    # 模拟各种操作
    for _ in range(50):
        start = time.time()
        time.sleep(0.001 * (hash(str(_)) % 10 + 1))
        profiler.record("检索", (time.time() - start) * 1000)

    for _ in range(30):
        start = time.time()
        time.sleep(0.005 * (hash(str(_)) % 5 + 1))
        profiler.record("生成", (time.time() - start) * 1000)

    for _ in range(20):
        start = time.time()
        time.sleep(0.0005)
        profiler.record("缓存查询", (time.time() - start) * 1000)

    profiler.print_report()

    # --- 可视化 ---
    if HAS_PLT:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 批量处理加速比
        counts = [10, 50, 100, 500]
        speedups = [processor.process_single_vs_batch(c)['speedup'] for c in counts]
        ax = axes[0]
        ax.plot(counts, speedups, 'o-', color='#3498db', linewidth=2, markersize=8)
        ax.set_xlabel('处理数量')
        ax.set_ylabel('加速比')
        ax.set_title('批量处理加速效果', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 性能热力图
        summary = profiler.summary()
        ops = list(summary.keys())
        metrics = ['mean_ms', 'p95_ms', 'max_ms']
        data = [[summary[op][m] for m in metrics] for op in ops]

        ax = axes[1]
        x_pos = range(len(metrics))
        width = 0.25
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        for i, (op, color) in enumerate(zip(ops, colors)):
            ax.bar([x + i * width for x in x_pos], data[i], width, label=op, color=color)
        ax.set_xticks([x + width for x in x_pos])
        ax.set_xticklabels(['平均', 'P95', '最大'])
        ax.set_ylabel('延迟(ms)')
        ax.set_title('各操作延迟分布', fontweight='bold')
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig('D:/code/big-model-learn/code/q_01/W31/d4_performance.png', dpi=150)
        print("\n图表已保存为 d4_performance.png")
        plt.close()

    print("\n完成!")
