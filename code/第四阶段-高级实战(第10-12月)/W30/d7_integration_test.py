"""
W30-D7 集成测试
================
实现RAG系统的集成测试, 包括:
- API接口测试
- 端到端Pipeline测试
- 性能基准测试

测试是保障系统质量的关键环节。
"""

import time
import json
import statistics
from typing import List, Dict, Tuple
from dataclasses import dataclass, field

try:
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    HAS_PLT = True
except ImportError:
    HAS_PLT = False


# ============================================================
# 1. 测试框架
# ============================================================

class TestResult:
    """单个测试结果"""

    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = None
        self.duration_ms = 0
        self.details = {}

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name} ({self.duration_ms:.1f}ms)"


class TestSuite:
    """测试套件"""

    def __init__(self, name: str):
        self.name = name
        self.results: List[TestResult] = []
        self.start_time = None

    def run(self, func, name: str = None, **kwargs):
        """运行单个测试"""
        result = TestResult(name or func.__name__)
        start = time.time()
        try:
            func(result=result, **kwargs)
            result.passed = True
        except AssertionError as e:
            result.error = str(e)
        except Exception as e:
            result.error = f"异常: {type(e).__name__}: {e}"
        result.duration_ms = (time.time() - start) * 1000
        self.results.append(result)
        return result

    def summary(self) -> Dict:
        """生成测试摘要"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        total_time = sum(r.duration_ms for r in self.results)

        return {
            'suite': self.name,
            'total': total,
            'passed': passed,
            'failed': failed,
            'pass_rate': passed / total if total > 0 else 0,
            'total_time_ms': total_time,
            'results': self.results,
        }

    def print_report(self):
        """打印测试报告"""
        summary = self.summary()
        print(f"\n{'='*60}")
        print(f"测试套件: {self.name}")
        print(f"{'='*60}")

        for result in self.results:
            status = "PASS" if result.passed else "FAIL"
            icon = "+" if result.passed else "x"
            print(f"  [{icon}] {result.name} ({result.duration_ms:.1f}ms)")
            if result.error:
                print(f"      错误: {result.error[:80]}")

        print(f"\n总计: {summary['total']} | "
              f"通过: {summary['passed']} | "
              f"失败: {summary['failed']} | "
              f"通过率: {summary['pass_rate']:.1%} | "
              f"总耗时: {summary['total_time_ms']:.1f}ms")


# ============================================================
# 2. 简化的被测系统 (从d6复用核心逻辑)
# ============================================================

class SimpleDocumentStore:
    """文档存储"""

    def __init__(self):
        self.documents = {}

    def add(self, doc_id: str, title: str, content: str, tags: List[str] = None):
        self.documents[doc_id] = {
            'doc_id': doc_id, 'title': title, 'content': content,
            'tags': tags or [], 'created_at': time.time()
        }
        return doc_id

    def get(self, doc_id: str):
        return self.documents.get(doc_id)

    def search(self, query: str, top_k: int = 5):
        query_words = set(query.lower().split())
        results = []
        for doc_id, doc in self.documents.items():
            content_words = set(doc['content'].lower().split())
            overlap = query_words & content_words
            if overlap:
                score = len(overlap) / len(query_words) if query_words else 0
                results.append({'doc_id': doc_id, 'title': doc['title'],
                               'score': score, 'content': doc['content'][:100]})
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

    def delete(self, doc_id: str) -> bool:
        if doc_id in self.documents:
            del self.documents[doc_id]
            return True
        return False

    def count(self) -> int:
        return len(self.documents)


class SimpleRAGSystem:
    """简化的RAG系统"""

    def __init__(self):
        self.store = SimpleDocumentStore()
        self.query_count = 0

    def upload_document(self, title: str, content: str, tags: List[str] = None) -> str:
        import hashlib
        doc_id = hashlib.md5(f"{title}_{time.time()}".encode()).hexdigest()[:10]
        self.store.add(doc_id, title, content, tags)
        return doc_id

    def query(self, question: str, top_k: int = 5) -> Dict:
        start = time.time()
        self.query_count += 1

        results = self.store.search(question, top_k=top_k)
        if results:
            answer = f"根据检索到的{len(results)}篇文档, {results[0]['content']}"
        else:
            answer = "抱歉, 未找到相关文档。"

        return {
            'question': question,
            'answer': answer,
            'sources': results,
            'latency_ms': (time.time() - start) * 1000,
        }

    def health_check(self) -> Dict:
        return {
            'status': 'healthy',
            'documents': self.store.count(),
            'queries_served': self.query_count,
        }


# ============================================================
# 3. API测试
# ============================================================

def run_api_tests():
    """API接口测试"""
    suite = TestSuite("API接口测试")
    system = SimpleRAGSystem()

    # 测试: 文档上传
    def test_upload(result, **kwargs):
        doc_id = system.upload_document("测试文档", "这是一篇关于Python编程的测试文档", ["test"])
        assert doc_id is not None, "上传应返回doc_id"
        assert len(doc_id) > 0, "doc_id不能为空"
        result.details['doc_id'] = doc_id

    suite.run(test_upload, "文档上传 - 正常")

    # 测试: 查询
    def test_query(result, **kwargs):
        resp = system.query("Python编程")
        assert 'answer' in resp, "响应应包含answer字段"
        assert 'sources' in resp, "响应应包含sources字段"
        assert 'latency_ms' in resp, "响应应包含latency_ms字段"
        assert len(resp['answer']) > 0, "回答不能为空"
        result.details['latency'] = resp['latency_ms']

    suite.run(test_query, "查询接口 - 正常")

    # 测试: 无结果查询
    def test_empty_query(result, **kwargs):
        resp = system.query("量子计算xyz123")
        assert 'answer' in resp, "即使无结果也应返回answer"
        assert len(resp['sources']) == 0, "无结果时sources应为空"

    suite.run(test_empty_query, "查询接口 - 无匹配结果")

    # 测试: 健康检查
    def test_health(result, **kwargs):
        health = system.health_check()
        assert health['status'] == 'healthy', "健康状态应为healthy"
        assert health['documents'] > 0, "应有文档存在"

    suite.run(test_health, "健康检查")

    # 测试: 文档删除
    def test_delete(result, **kwargs):
        doc_id = system.upload_document("临时文档", "待删除的内容")
        deleted = system.store.delete(doc_id)
        assert deleted == True, "删除应返回True"
        assert system.store.get(doc_id) is None, "删除后不应能获取"

    suite.run(test_delete, "文档删除 - 正常")

    # 测试: 删除不存在的文档
    def test_delete_nonexistent(result, **kwargs):
        deleted = system.store.delete("nonexistent_id")
        assert deleted == False, "删除不存在的文档应返回False"

    suite.run(test_delete_nonexistent, "文档删除 - 不存在")

    # 测试: 多文档上传与查询
    def test_multiple_docs(result, **kwargs):
        docs = [
            ("机器学习基础", "机器学习是人工智能的核心技术"),
            ("深度学习进阶", "深度学习使用神经网络进行特征学习"),
            ("NLP入门", "自然语言处理研究计算机理解和生成文本"),
        ]
        for title, content in docs:
            system.upload_document(title, content)

        resp = system.query("人工智能")
        assert len(resp['sources']) > 0, "多文档场景应有检索结果"

    suite.run(test_multiple_docs, "多文档上传与查询")

    return suite


# ============================================================
# 4. 端到端Pipeline测试
# ============================================================

def run_e2e_tests():
    """端到端Pipeline测试"""
    suite = TestSuite("端到端Pipeline测试")
    system = SimpleRAGSystem()

    # 完整流程: 上传 -> 检索 -> 回答
    def test_full_pipeline(result, **kwargs):
        # Step 1: 上传知识库
        docs = [
            ("RAG系统设计", "RAG系统包含检索器、生成器和重排序器三个核心组件", ["RAG"]),
            ("向量检索原理", "向量检索通过计算查询和文档的语义相似度来匹配", ["检索"]),
            ("Prompt工程", "好的Prompt设计能显著提升LLM的输出质量", ["Prompt"]),
        ]
        doc_ids = []
        for title, content, tags in docs:
            doc_id = system.upload_document(title, content, tags)
            doc_ids.append(doc_id)

        assert system.store.count() == 3, f"应有3篇文档, 实际{system.store.count()}"

        # Step 2: 查询
        queries = ["RAG系统", "检索方法", "Prompt优化"]
        for q in queries:
            resp = system.query(q)
            assert 'answer' in resp, f"查询'{q}'应返回answer"
            assert resp['latency_ms'] < 1000, f"查询'{q}'延迟应<1000ms"

        result.details['doc_ids'] = doc_ids

    suite.run(test_full_pipeline, "完整Pipeline: 上传->检索->回答")

    # 测试: 中文查询兼容性
    def test_chinese_query(result, **kwargs):
        system.upload_document("中文测试", "自然语言处理是人工智能的重要研究方向")
        resp = system.query("什么是自然语言处理?")
        assert len(resp['answer']) > 0, "中文查询应有回答"

    suite.run(test_chinese_query, "中文查询兼容性")

    # 测试: 连续对话场景
    def test_consecutive_queries(result, **kwargs):
        system.upload_document("Python教程", "Python是一种优雅的编程语言, 适合初学者")

        # 第一轮
        r1 = system.query("Python是什么")
        assert r1['answer'], "第一轮查询应有回答"

        # 第二轮
        r2 = system.query("Python适合谁")
        assert r2['answer'], "第二轮查询应有回答"

        # 验证查询计数
        assert system.query_count >= 2, "应记录多次查询"

    suite.run(test_consecutive_queries, "连续对话场景")

    # 测试: 大文档处理
    def test_large_document(result, **kwargs):
        large_content = "这是大文档测试。" * 1000
        doc_id = system.upload_document("大文档", large_content)
        assert doc_id is not None, "大文档上传应成功"

        resp = system.query("大文档")
        assert resp['answer'], "大文档查询应有回答"

    suite.run(test_large_document, "大文档处理")

    return suite


# ============================================================
# 5. 性能基准测试
# ============================================================

@dataclass
class BenchmarkResult:
    """基准测试结果"""
    name: str
    iterations: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    ops_per_second: float


class PerformanceBenchmark:
    """性能基准测试"""

    def __init__(self):
        self.results: List[BenchmarkResult] = []

    def run_benchmark(self, name: str, func, iterations: int = 100,
                       warmup: int = 10) -> BenchmarkResult:
        """运行基准测试"""
        # 预热
        for _ in range(warmup):
            func()

        # 正式测试
        latencies = []
        for _ in range(iterations):
            start = time.time()
            func()
            latency = (time.time() - start) * 1000
            latencies.append(latency)

        latencies.sort()
        result = BenchmarkResult(
            name=name,
            iterations=iterations,
            mean_ms=statistics.mean(latencies),
            median_ms=statistics.median(latencies),
            p95_ms=latencies[int(len(latencies) * 0.95)],
            p99_ms=latencies[int(len(latencies) * 0.99)],
            min_ms=min(latencies),
            max_ms=max(latencies),
            ops_per_second=1000 / statistics.mean(latencies),
        )
        self.results.append(result)
        return result

    def print_report(self):
        """打印基准报告"""
        print(f"\n{'='*70}")
        print(f"{'性能基准测试报告':^70}")
        print(f"{'='*70}")
        print(f"{'测试名称':<20} {'平均(ms)':<10} {'P95(ms)':<10} "
              f"{'P99(ms)':<10} {'QPS':<10}")
        print("-" * 70)

        for r in self.results:
            print(f"{r.name:<20} {r.mean_ms:<10.2f} {r.p95_ms:<10.2f} "
                  f"{r.p99_ms:<10.2f} {r.ops_per_second:<10.1f}")


def run_performance_tests():
    """运行性能基准测试"""
    bench = PerformanceBenchmark()
    system = SimpleRAGSystem()

    # 准备数据
    for i in range(50):
        system.upload_document(f"文档{i}", f"这是第{i}篇文档, 包含一些测试内容关键词{i%10}")

    # 基准1: 文档上传
    bench.run_benchmark("文档上传", lambda: system.upload_document(
        "bench", "基准测试文档", ["bench"]), iterations=50)

    # 基准2: 查询
    bench.run_benchmark("查询", lambda: system.query("测试内容"), iterations=100)

    # 基准3: 健康检查
    bench.run_benchmark("健康检查", lambda: system.health_check(), iterations=200)

    # 基准4: 混合操作
    counter = [0]
    def mixed_op():
        counter[0] += 1
        if counter[0] % 3 == 0:
            system.upload_document("mixed", f"混合测试{counter[0]}")
        else:
            system.query(f"查询{counter[0] % 10}")

    bench.run_benchmark("混合操作", mixed_op, iterations=100)

    return bench


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("W30-D7 集成测试")
    print("=" * 60)

    # --- 1. API测试 ---
    print("\n--- 1. API接口测试 ---")
    api_suite = run_api_tests()
    api_suite.print_report()

    # --- 2. 端到端测试 ---
    print(f"\n--- 2. 端到端Pipeline测试 ---")
    e2e_suite = run_e2e_tests()
    e2e_suite.print_report()

    # --- 3. 性能基准 ---
    print(f"\n--- 3. 性能基准测试 ---")
    bench = run_performance_tests()
    bench.print_report()

    # --- 可视化 ---
    if HAS_PLT and bench.results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 延迟分布
        names = [r.name for r in bench.results]
        means = [r.mean_ms for r in bench.results]
        p95s = [r.p95_ms for r in bench.results]
        p99s = [r.p99_ms for r in bench.results]

        x = range(len(names))
        width = 0.25
        ax = axes[0]
        ax.bar([i - width for i in x], means, width, label='平均', color='#3498db')
        ax.bar(x, p95s, width, label='P95', color='#e74c3c')
        ax.bar([i + width for i in x], p99s, width, label='P99', color='#f39c12')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15)
        ax.set_ylabel('延迟(ms)')
        ax.set_title('接口延迟分布', fontweight='bold')
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # QPS
        qps = [r.ops_per_second for r in bench.results]
        ax = axes[1]
        bars = ax.bar(names, qps, color='#2ecc71', edgecolor='white')
        for bar, val in zip(bars, qps):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{val:.0f}', ha='center', fontsize=10)
        ax.set_ylabel('QPS (操作/秒)')
        ax.set_title('吞吐量对比', fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig('D:/code/big-model-learn/code/q_01/W30/d7_integration_test.png', dpi=150)
        print("\n图表已保存为 d7_integration_test.png")
        plt.close()

    # --- 最终汇总 ---
    print(f"\n{'='*60}")
    print("测试汇总")
    print(f"{'='*60}")
    api_summary = api_suite.summary()
    e2e_summary = e2e_suite.summary()
    total_tests = api_summary['total'] + e2e_summary['total']
    total_passed = api_summary['passed'] + e2e_summary['passed']
    print(f"总测试数: {total_tests}")
    print(f"通过: {total_passed}")
    print(f"失败: {total_tests - total_passed}")
    print(f"通过率: {total_passed/total_tests:.1%}")

    print("\n完成!")
