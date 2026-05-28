### Day 5（周五）：FastAPI部署LLM
# 流式响应(SSE), 批量推理, 请求队列, 健康检查

import numpy as np
import matplotlib.pyplot as plt
import time
import json

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# FastAPI概念演示 (无需安装FastAPI)
print("=" * 60)
print("1. FastAPI部署LLM API服务")
print("=" * 60)

print("""
  典型API设计:
    POST /v1/chat/completions  - 聊天补全 (流式/非流式)
    POST /v1/completions       - 文本补全
    GET  /v1/models            - 列出可用模型
    GET  /health               - 健康检查
    GET  /metrics              - 性能指标

  代码结构:
    app/
    ├── main.py          # FastAPI应用入口
    ├── model.py         # 模型加载和推理
    ├── schemas.py       # 请求/响应模型
    ├── queue.py         # 请求队列管理
    └── config.py        # 配置
""")


# ============================================================
# 2. 模拟API服务
# ============================================================

class MockLLMModel:
    """模拟LLM模型"""

    def __init__(self, model_name="mock-llm-7b"):
        self.model_name = model_name
        self.loaded = True
        self.inference_count = 0

    def generate(self, prompt, max_tokens=50, temperature=0.7):
        """模拟文本生成"""
        self.inference_count += 1
        # 模拟生成时间
        time.sleep(0.01)
        tokens = [f"token_{i}" for i in range(max_tokens)]
        return {
            'text': f"生成文本: 基于输入'{prompt[:20]}...'的回答",
            'tokens': tokens,
            'usage': {
                'prompt_tokens': len(prompt.split()),
                'completion_tokens': max_tokens,
                'total_tokens': len(prompt.split()) + max_tokens,
            },
        }

    def generate_stream(self, prompt, max_tokens=50):
        """模拟流式生成"""
        self.inference_count += 1
        for i in range(max_tokens):
            time.sleep(0.005)
            yield {
                'token': f"token_{i}",
                'index': i,
                'finish_reason': 'stop' if i == max_tokens - 1 else None,
            }


class RequestQueue:
    """请求队列管理器"""

    def __init__(self, max_concurrent=8):
        self.max_concurrent = max_concurrent
        self.current = 0
        self.queue = []
        self.completed = []
        self.rejected = 0

    def enqueue(self, request_id, priority=0):
        """加入队列"""
        if self.current < self.max_concurrent:
            self.current += 1
            self.queue.append({
                'id': request_id,
                'priority': priority,
                'status': 'processing',
                'start_time': time.time(),
            })
            return True
        else:
            self.rejected += 1
            return False

    def complete(self, request_id):
        """完成请求"""
        for req in self.queue:
            if req['id'] == request_id:
                req['status'] = 'completed'
                req['end_time'] = time.time()
                self.completed.append(req)
                self.queue.remove(req)
                self.current -= 1
                return True
        return False

    def get_stats(self):
        return {
            'processing': self.current,
            'max_concurrent': self.max_concurrent,
            'completed': len(self.completed),
            'rejected': self.rejected,
            'queue_length': len([q for q in self.queue if q['status'] == 'processing']),
        }


class LLMAPIService:
    """模拟LLM API服务"""

    def __init__(self, model_name="mock-llm-7b"):
        self.model = MockLLMModel(model_name)
        self.queue = RequestQueue(max_concurrent=8)
        self.request_log = []
        self.start_time = time.time()

    def chat_completions(self, messages, max_tokens=50, stream=False):
        """聊天补全API"""
        request_id = f"req_{len(self.request_log)}"
        prompt = messages[-1]['content'] if messages else ""

        # 入队
        if not self.queue.enqueue(request_id):
            return {'error': 'Service overloaded', 'status': 503}

        start = time.time()

        if stream:
            # 流式响应
            result = {'type': 'stream', 'chunks': []}
            for chunk in self.model.generate_stream(prompt, max_tokens):
                result['chunks'].append(chunk)
        else:
            # 非流式响应
            result = self.model.generate(prompt, max_tokens)

        elapsed = time.time() - start

        self.queue.complete(request_id)
        self.request_log.append({
            'id': request_id,
            'latency': elapsed,
            'tokens': max_tokens,
            'tps': max_tokens / elapsed if elapsed > 0 else 0,
        })

        return result

    def health_check(self):
        """健康检查"""
        uptime = time.time() - self.start_time
        return {
            'status': 'healthy' if self.model.loaded else 'unhealthy',
            'model': self.model.model_name,
            'uptime': f'{uptime:.0f}s',
            'queue': self.queue.get_stats(),
            'total_requests': len(self.request_log),
        }

    def get_metrics(self):
        """获取性能指标"""
        if not self.request_log:
            return {'error': 'No requests yet'}

        latencies = [r['latency'] for r in self.request_log]
        tps_list = [r['tps'] for r in self.request_log]

        return {
            'total_requests': len(self.request_log),
            'avg_latency': np.mean(latencies),
            'p50_latency': np.percentile(latencies, 50),
            'p99_latency': np.percentile(latencies, 99),
            'avg_tps': np.mean(tps_list),
            'queue_stats': self.queue.get_stats(),
        }


# ============================================================
# 3. 运行API服务模拟
# ============================================================
print("\n" + "=" * 60)
print("2. API服务模拟测试")
print("=" * 60)

service = LLMAPIService("MyLLM-7B")

# 测试聊天补全
test_messages = [
    [{"role": "user", "content": "什么是机器学习?"}],
    [{"role": "user", "content": "解释Transformer架构"}],
    [{"role": "user", "content": "Python和Java的区别"}],
]

print("\n  非流式请求测试:")
for msgs in test_messages:
    result = service.chat_completions(msgs, max_tokens=30)
    print(f"    输入: {msgs[0]['content'][:30]}")
    print(f"    输出: {result.get('text', 'stream')[:50]}")

print("\n  流式请求测试:")
stream_result = service.chat_completions(
    [{"role": "user", "content": "写一首诗"}],
    max_tokens=10, stream=True
)
print(f"    流式chunks: {len(stream_result.get('chunks', []))}个")

# 健康检查
print(f"\n  健康检查:")
health = service.health_check()
for key, val in health.items():
    print(f"    {key}: {val}")

# 性能指标
print(f"\n  性能指标:")
metrics = service.get_metrics()
for key, val in metrics.items():
    print(f"    {key}: {val}")


# ============================================================
# 4. 并发性能测试
# ============================================================
print("\n" + "=" * 60)
print("3. 并发性能测试")
print("=" * 60)

np.random.seed(42)
concurrency_levels = [1, 4, 8, 16, 32]
concurrency_results = {}

for conc in concurrency_levels:
    service_test = LLMAPIService(f"test-{conc}")
    latencies = []

    for i in range(50):
        start = time.time()
        result = service_test.chat_completions(
            [{"role": "user", "content": f"测试请求{i}"}],
            max_tokens=20,
        )
        latencies.append(time.time() - start)

    concurrency_results[conc] = {
        'latencies': latencies,
        'avg_latency': np.mean(latencies),
        'p99_latency': np.percentile(latencies, 99),
        'throughput': 50 / sum(latencies),
    }
    print(f"  并发={conc:>2}: 平均延迟={np.mean(latencies) * 1000:.1f}ms, "
          f"P99={np.percentile(latencies, 99) * 1000:.1f}ms, "
          f"吞吐={50 / sum(latencies):.1f}req/s")


# ============================================================
# 5. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 5.1 API架构图
ax1 = axes[0, 0]
api_layers = [
    ('Client / SDK', 0.85, '#4CAF50'),
    ('Load Balancer', 0.70, '#2196F3'),
    ('FastAPI Server', 0.55, '#FF9800'),
    ('Request Queue', 0.40, '#9C27B0'),
    ('Model Worker', 0.25, '#E91E63'),
    ('GPU', 0.10, '#607D8B'),
]
for label, y, color in api_layers:
    ax1.add_patch(plt.Rectangle((0.1, y - 0.05), 0.8, 0.1,
                                 facecolor=color, alpha=0.6, edgecolor='black', linewidth=2))
    ax1.text(0.5, y, label, ha='center', va='center', fontsize=10, fontweight='bold')

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_title('LLM API服务架构')
ax1.axis('off')

# 5.2 延迟分布
ax2 = axes[0, 1]
latencies_ms = [r['latency'] * 1000 for r in service.request_log]
if latencies_ms:
    ax2.hist(latencies_ms, bins=15, color='#2196F3', edgecolor='black', alpha=0.7)
    ax2.axvline(x=np.mean(latencies_ms), color='red', linestyle='--',
                label=f'均值={np.mean(latencies_ms):.1f}ms')
    ax2.axvline(x=np.percentile(latencies_ms, 99), color='orange', linestyle='--',
                label=f'P99={np.percentile(latencies_ms, 99):.1f}ms')
    ax2.set_xlabel('延迟 (ms)')
    ax2.set_ylabel('请求数')
    ax2.set_title('请求延迟分布')
    ax2.legend()

# 5.3 并发 vs 延迟
ax3 = axes[1, 0]
concs = list(concurrency_results.keys())
avg_lats = [concurrency_results[c]['avg_latency'] * 1000 for c in concs]
p99_lats = [concurrency_results[c]['p99_latency'] * 1000 for c in concs]
ax3.plot(concs, avg_lats, 'bo-', linewidth=2, label='平均延迟')
ax3.plot(concs, p99_lats, 'rs-', linewidth=2, label='P99延迟')
ax3.set_xlabel('并发数')
ax3.set_ylabel('延迟 (ms)')
ax3.set_title('并发数 vs 延迟')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 5.4 并发 vs 吞吐量
ax4 = axes[1, 1]
throughputs = [concurrency_results[c]['throughput'] for c in concs]
ax4.plot(concs, throughputs, 'g^-', linewidth=2, markersize=10)
ax4.set_xlabel('并发数')
ax4.set_ylabel('吞吐量 (req/s)')
ax4.set_title('并发数 vs 吞吐量')
ax4.grid(True, alpha=0.3)
for i, v in enumerate(throughputs):
    ax4.annotate(f'{v:.1f}', (concs[i], v), textcoords="offset points",
                 xytext=(0, 10), ha='center')

plt.suptitle('W24-D5: FastAPI部署LLM', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W24/d5_api_deployment.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存!")
