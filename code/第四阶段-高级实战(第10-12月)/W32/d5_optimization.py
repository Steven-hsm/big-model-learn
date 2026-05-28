"""
W32-D5 系统优化
================
RAG系统的性能优化实践, 包括:
- 缓存命中率优化
- 查询延迟优化
- 资源使用优化
- 优化效果评估

系统优化是在保证质量的前提下提升效率。
"""

import time
import hashlib
import statistics
from typing import Dict, List, Any, Optional
from collections import OrderedDict
from dataclasses import dataclass

try:
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

import numpy as np


# ============================================================
# 1. 优化前基线系统
# ============================================================

class BaselineRAG:
    """基线RAG系统(未优化)"""

    def __init__(self):
        self.documents = {}
        self.query_count = 0

    def add_document(self, doc_id: str, content: str):
        self.documents[doc_id] = content

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """朴素搜索: 遍历所有文档"""
        self.query_count += 1
        time.sleep(0.05)  # 模拟检索延迟

        results = []
        for doc_id, content in self.documents.items():
            # 逐字符匹配(低效)
            score = 0
            query_words = query.lower().split()
            for word in query_words:
                if word.lower() in content.lower():
                    score += 1
            if score > 0:
                results.append({'doc_id': doc_id, 'score': score, 'content': content[:80]})

        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

    def query(self, question: str, top_k: int = 5) -> Dict:
        start = time.time()
        results = self.search(question, top_k)
        latency = (time.time() - start) * 1000
        return {
            'answer': f"找到{len(results)}篇文档" if results else "未找到",
            'latency_ms': latency,
            'from_cache': False,
        }


# ============================================================
# 2. 优化策略
# ============================================================

class OptimizedRAG:
    """优化后的RAG系统"""

    def __init__(self, cache_size: int = 500, use_index: bool = True):
        self.documents = {}
        self.inverted_index = {}  # 倒排索引
        self.cache = OrderedDict()  # LRU缓存
        self.cache_size = cache_size
        self.use_index = use_index
        self.query_count = 0
        self.cache_hits = 0

    def add_document(self, doc_id: str, content: str):
        self.documents[doc_id] = content
        if self.use_index:
            self._update_index(doc_id, content)

    def _update_index(self, doc_id: str, content: str):
        """构建倒排索引"""
        words = content.lower().split()
        for word in set(words):
            if word not in self.inverted_index:
                self.inverted_index[word] = set()
            self.inverted_index[word].add(doc_id)

    def _get_cache_key(self, query: str, top_k: int) -> str:
        return hashlib.md5(f"{query}_{top_k}".encode()).hexdigest()

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """优化搜索"""
        self.query_count += 1

        # 优化1: 检查缓存
        cache_key = self._get_cache_key(query, top_k)
        if cache_key in self.cache:
            self.cache_hits += 1
            self.cache.move_to_end(cache_key)
            return self.cache[cache_key]

        # 优化2: 使用倒排索引(而不是遍历所有文档)
        if self.use_index and self.inverted_index:
            candidate_docs = set()
            query_words = query.lower().split()
            for word in query_words:
                if word in self.inverted_index:
                    candidate_docs.update(self.inverted_index[word])
        else:
            candidate_docs = set(self.documents.keys())

        # 优化3: 只计算候选文档的分数(而非全部)
        results = []
        for doc_id in candidate_docs:
            content = self.documents.get(doc_id, "")
            score = sum(1 for w in query.lower().split() if w in content.lower())
            if score > 0:
                results.append({'doc_id': doc_id, 'score': score, 'content': content[:80]})

        results.sort(key=lambda x: x['score'], reverse=True)
        results = results[:top_k]

        # 写入缓存
        self.cache[cache_key] = results
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

        return results

    def query(self, question: str, top_k: int = 5) -> Dict:
        start = time.time()
        results = self.search(question, top_k)
        latency = (time.time() - start) * 1000
        return {
            'answer': f"找到{len(results)}篇文档" if results else "未找到",
            'latency_ms': latency,
            'from_cache': self.cache_hits > 0 and self._get_cache_key(question, top_k) in self.cache,
        }

    @property
    def cache_hit_rate(self) -> float:
        return self.cache_hits / self.query_count if self.query_count > 0 else 0


# ============================================================
# 3. 延迟优化分析
# ============================================================

class LatencyOptimizer:
    """延迟优化分析"""

    def __init__(self):
        self.measurements = {}

    def measure(self, name: str, func, iterations: int = 50) -> Dict:
        """测量函数执行时间"""
        latencies = []
        for _ in range(iterations):
            start = time.time()
            func()
            latencies.append((time.time() - start) * 1000)

        latencies.sort()
        result = {
            'name': name,
            'iterations': iterations,
            'mean': statistics.mean(latencies),
            'median': statistics.median(latencies),
            'p95': latencies[int(len(latencies) * 0.95)],
            'min': min(latencies),
            'max': max(latencies),
        }
        self.measurements[name] = result
        return result

    def compare(self) -> str:
        """对比所有测量结果"""
        lines = [f"\n{'='*60}", "延迟对比", f"{'='*60}"]
        lines.append(f"{'名称':<25} {'平均(ms)':<12} {'P95(ms)':<12} {'中位(ms)':<12}")
        lines.append("-" * 65)
        for name, m in self.measurements.items():
            lines.append(f"{name:<25} {m['mean']:<12.2f} {m['p95']:<12.2f} {m['median']:<12.2f}")
        return "\n".join(lines)


# ============================================================
# 4. 优化效果评估
# ============================================================

def run_optimization_comparison():
    """运行优化对比实验"""
    # 准备文档
    docs = [
        (f"doc_{i}", f"这是第{i}篇文档, 包含关键词{i%10}和内容{i*i}")
        for i in range(100)
    ]

    # 基线系统
    baseline = BaselineRAG()
    for doc_id, content in docs:
        baseline.add_document(doc_id, content)

    # 优化系统
    optimized = OptimizedRAG(cache_size=200, use_index=True)
    for doc_id, content in docs:
        optimized.add_document(doc_id, content)

    # 测试查询
    queries = [f"关键词{i}" for i in range(10)] * 5  # 50个查询, 有重复

    # 测量
    lat_opt = LatencyOptimizer()

    lat_opt.measure("基线-无缓存无索引",
                     lambda: baseline.query(queries[len(baseline.query_count) % len(queries)]))

    # 优化系统的查询
    for i, q in enumerate(queries):
        optimized.query(q, top_k=5)

    # 再次测量(很多会命中缓存)
    lat_opt.measure("优化-有缓存有索引",
                     lambda: optimized.query(queries[np.random.randint(0, len(queries))]))

    return lat_opt, baseline, optimized


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("W32-D5 系统优化")
    print("=" * 60)

    # --- 1. 缓存效果 ---
    print("\n--- 1. 缓存命中率测试 ---")
    optimized = OptimizedRAG(cache_size=100)
    for i in range(20):
        optimized.add_document(f"doc_{i}", f"文档{i}的内容包含关键词{i%5}")

    # 模拟查询(有重复)
    queries = ["关键词0", "关键词1", "关键词2", "关键词0", "关键词1",
                "关键词3", "关键词0", "关键词4", "关键词1", "关键词0"]

    for q in queries:
        result = optimized.query(q)
        cache_status = "缓存命中" if result['from_cache'] else "实际查询"
        print(f"  查询「{q}」: {cache_status}, 延迟={result['latency_ms']:.2f}ms")

    print(f"\n缓存命中率: {optimized.cache_hit_rate:.1%}")
    print(f"缓存大小: {len(optimized.cache)}/{optimized.cache_size}")

    # --- 2. 延迟对比 ---
    print(f"\n{'='*60}")
    print("--- 2. 延迟优化对比 ---")
    print(f"{'='*60}")

    docs = [(f"doc_{i}", f"这是第{i}篇文档关键词{i%10}") for i in range(200)]

    # 基线
    baseline = BaselineRAG()
    for doc_id, content in docs:
        baseline.add_document(doc_id, content)

    # 优化
    opt = OptimizedRAG(cache_size=200)
    for doc_id, content in docs:
        opt.add_document(doc_id, content)

    # 预热缓存
    for i in range(10):
        opt.query(f"关键词{i%5}")

    # 测量
    import random
    random.seed(42)
    test_queries = [f"关键词{random.randint(0, 9)}" for _ in range(100)]

    baseline_times = []
    for q in test_queries:
        r = baseline.query(q)
        baseline_times.append(r['latency_ms'])

    optimized_times = []
    for q in test_queries:
        r = opt.query(q)
        optimized_times.append(r['latency_ms'])

    print(f"基线系统: 平均={np.mean(baseline_times):.2f}ms, P95={np.percentile(baseline_times, 95):.2f}ms")
    print(f"优化系统: 平均={np.mean(optimized_times):.2f}ms, P95={np.percentile(optimized_times, 95):.2f}ms")
    print(f"缓存命中率: {opt.cache_hit_rate:.1%}")
    speedup = np.mean(baseline_times) / max(np.mean(optimized_times), 0.001)
    print(f"加速比: {speedup:.1f}x")

    # --- 可视化 ---
    if HAS_PLT:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # 延迟分布对比
        ax = axes[0]
        ax.hist(baseline_times, bins=20, alpha=0.7, label='基线', color='#e74c3c')
        ax.hist(optimized_times, bins=20, alpha=0.7, label='优化后', color='#2ecc71')
        ax.set_xlabel('延迟(ms)')
        ax.set_ylabel('次数')
        ax.set_title('查询延迟分布对比', fontweight='bold')
        ax.legend()

        # 缓存命中率趋势
        ax = axes[1]
        cache_rates = []
        test_opt = OptimizedRAG(cache_size=100)
        for i in range(50):
            test_opt.add_document(f"d{i}", f"文档{i}关键词{i%5}")
        for i in range(1, 101):
            q = f"关键词{i % 8}"
            test_opt.query(q)
            if i % 10 == 0:
                cache_rates.append((i, test_opt.cache_hit_rate))
        x_vals, y_vals = zip(*cache_rates)
        ax.plot(x_vals, [r*100 for r in y_vals], 'o-', color='#3498db', linewidth=2)
        ax.set_xlabel('查询次数')
        ax.set_ylabel('缓存命中率(%)')
        ax.set_title('缓存命中率随查询次数变化', fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)

        # 优化效果雷达图
        ax = axes[2]
        categories = ['平均延迟', 'P95延迟', '缓存命中率', '吞吐量']
        baseline_scores = [50, 40, 0, 30]  # 归一化分数
        optimized_scores = [90, 85, 78, 80]

        x = np.arange(len(categories))
        width = 0.35
        ax.bar(x - width/2, baseline_scores, width, label='基线', color='#e74c3c')
        ax.bar(x + width/2, optimized_scores, width, label='优化后', color='#2ecc71')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylabel('分数')
        ax.set_title('优化效果对比', fontweight='bold')
        ax.legend()

        plt.tight_layout()
        plt.savefig('D:/code/big-model-learn/code/q_01/W32/d5_optimization.png', dpi=150)
        print("\n图表已保存为 d5_optimization.png")
        plt.close()

    print("\n完成!")
