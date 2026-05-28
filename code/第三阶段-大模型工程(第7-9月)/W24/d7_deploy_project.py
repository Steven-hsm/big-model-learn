### Day 7（周日）：部署项目 - LLM API服务
# 模型API封装, Docker配置, 性能压测, 监控指标

import numpy as np
import matplotlib.pyplot as plt
import time
import json
from collections import deque

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. LLM服务完整封装
# ============================================================
print("=" * 60)
print("1. LLM API服务完整封装")
print("=" * 60)


class LLMService:
    """完整LLM API服务封装"""

    def __init__(self, model_name="MyLLM-7B", max_batch_size=8, max_seq_length=2048):
        self.model_name = model_name
        self.max_batch_size = max_batch_size
        self.max_seq_length = max_seq_length

        # 模型状态
        self.is_ready = False
        self.model_loaded_at = None

        # 请求管理
        self.active_requests = 0
        self.request_queue = deque(maxlen=100)

        # 统计
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens_generated': 0,
            'total_input_tokens': 0,
        }

        # 性能历史
        self.latency_history = deque(maxlen=1000)
        self.throughput_history = deque(maxlen=100)
        self.error_history = []

        # 监控
        self.metrics = {
            'avg_latency': 0,
            'p50_latency': 0,
            'p99_latency': 0,
            'current_qps': 0,
            'gpu_utilization': 0,
            'memory_usage': 0,
        }

    def load_model(self):
        """加载模型"""
        print(f"  [启动] 加载模型 {self.model_name}...")
        time.sleep(0.1)  # 模拟加载时间
        self.is_ready = True
        self.model_loaded_at = time.time()
        print(f"  [启动] 模型加载完成!")

    # ---------- API端点 ----------

    def chat_completions(self, messages, max_tokens=100, temperature=0.7, stream=False):
        """POST /v1/chat/completions"""
        if not self.is_ready:
            return {'error': 'Model not ready', 'status_code': 503}

        request_id = f"req_{self.stats['total_requests']}"
        start_time = time.time()

        try:
            self.stats['total_requests'] += 1
            self.active_requests += 1

            prompt = messages[-1]['content'] if messages else ""
            input_tokens = len(prompt.split())

            # 模拟推理
            time.sleep(np.random.uniform(0.02, 0.15))
            output_tokens = max_tokens

            latency = time.time() - start_time
            self.latency_history.append(latency)

            self.stats['successful_requests'] += 1
            self.stats['total_tokens_generated'] += output_tokens
            self.stats['total_input_tokens'] += input_tokens

            response = {
                'id': request_id,
                'object': 'chat.completion',
                'model': self.model_name,
                'choices': [{
                    'message': {'role': 'assistant', 'content': f'回复: {prompt[:30]}...'},
                    'finish_reason': 'stop',
                }],
                'usage': {
                    'prompt_tokens': input_tokens,
                    'completion_tokens': output_tokens,
                    'total_tokens': input_tokens + output_tokens,
                },
                'latency_ms': latency * 1000,
            }

            return response

        except Exception as e:
            self.stats['failed_requests'] += 1
            self.error_history.append({'time': time.time(), 'error': str(e)})
            return {'error': str(e), 'status_code': 500}

        finally:
            self.active_requests -= 1

    def health_check(self):
        """GET /health"""
        return {
            'status': 'healthy' if self.is_ready else 'unhealthy',
            'model': self.model_name,
            'uptime': time.time() - self.model_loaded_at if self.model_loaded_at else 0,
            'active_requests': self.active_requests,
        }

    def get_metrics(self):
        """GET /metrics"""
        if self.latency_history:
            latencies = list(self.latency_history)
            self.metrics['avg_latency'] = np.mean(latencies)
            self.metrics['p50_latency'] = np.percentile(latencies, 50)
            self.metrics['p99_latency'] = np.percentile(latencies, 99)

        uptime = time.time() - self.model_loaded_at if self.model_loaded_at else 1
        self.metrics['current_qps'] = self.stats['successful_requests'] / uptime
        self.metrics['gpu_utilization'] = min(self.active_requests / self.max_batch_size, 1.0) * 100
        self.metrics['memory_usage'] = np.random.uniform(60, 90)

        return {
            'stats': self.stats,
            'performance': {k: f'{v:.4f}' if isinstance(v, float) else v
                           for k, v in self.metrics.items()},
        }

    def list_models(self):
        """GET /v1/models"""
        return {
            'data': [{
                'id': self.model_name,
                'object': 'model',
                'owned_by': 'local',
                'max_seq_length': self.max_seq_length,
            }]
        }


# ============================================================
# 2. 启动服务并测试
# ============================================================
print("\n" + "=" * 60)
print("2. API服务测试")
print("=" * 60)

service = LLMService("MyLLM-7B-Q4", max_batch_size=8)
service.load_model()

# 测试基本API
print("\n  基本功能测试:")
test_messages = [
    [{"role": "user", "content": "你好"}],
    [{"role": "user", "content": "解释一下什么是API"}],
    [{"role": "user", "content": "用Python写一个hello world"}],
]

for msgs in test_messages:
    response = service.chat_completions(msgs, max_tokens=50)
    print(f"    请求: {msgs[0]['content']}")
    print(f"    延迟: {response.get('latency_ms', 0):.1f}ms")
    print(f"    Tokens: {response.get('usage', {}).get('total_tokens', 0)}")

print(f"\n  健康检查: {service.health_check()}")
print(f"  模型列表: {service.list_models()}")


# ============================================================
# 3. 性能压测
# ============================================================
print("\n" + "=" * 60)
print("3. 性能压测")
print("=" * 60)

class LoadTester:
    """性能压测工具"""

    def __init__(self, service):
        self.service = service
        self.results = {}

    def run_load_test(self, num_requests=100, concurrency=1):
        """运行负载测试"""
        latencies = []
        errors = 0

        start_time = time.time()

        for i in range(num_requests):
            msgs = [{"role": "user", "content": f"压测请求{i}: 测试内容"}]
            result = self.service.chat_completions(msgs, max_tokens=20)

            if 'error' in result:
                errors += 1
            else:
                latencies.append(result['latency_ms'] / 1000)

        total_time = time.time() - start_time

        self.results[concurrency] = {
            'num_requests': num_requests,
            'errors': errors,
            'total_time': total_time,
            'latencies': latencies,
            'avg_latency': np.mean(latencies) if latencies else 0,
            'p50_latency': np.percentile(latencies, 50) if latencies else 0,
            'p99_latency': np.percentile(latencies, 99) if latencies else 0,
            'qps': num_requests / total_time,
            'error_rate': errors / num_requests,
        }

        return self.results[concurrency]

    def generate_report(self):
        """生成压测报告"""
        print(f"\n  {'=' * 60}")
        print(f"  压测报告")
        print(f"  {'=' * 60}")

        print(f"\n  {'并发':>6} {'QPS':>8} {'平均延迟':>10} {'P50':>10} {'P99':>10} {'错误率':>8}")
        print(f"  {'-' * 55}")

        for conc, result in sorted(self.results.items()):
            print(f"  {conc:>6} {result['qps']:>8.1f} "
                  f"{result['avg_latency'] * 1000:>8.1f}ms "
                  f"{result['p50_latency'] * 1000:>8.1f}ms "
                  f"{result['p99_latency'] * 1000:>8.1f}ms "
                  f"{result['error_rate']:>8.1%}")


# 运行压测
load_tester = LoadTester(service)

print("\n  运行负载测试...")
for conc in [1, 4, 8, 16, 32]:
    result = load_tester.run_load_test(num_requests=50, concurrency=conc)
    print(f"    并发={conc:>2}: QPS={result['qps']:.1f}, "
          f"平均延迟={result['avg_latency'] * 1000:.1f}ms, "
          f"错误率={result['error_rate']:.1%}")

load_tester.generate_report()


# ============================================================
# 4. Docker配置示例
# ============================================================
print("\n" + "=" * 60)
print("4. Docker部署配置")
print("=" * 60)

dockerfile = """
  # Dockerfile示例 (概念)
  FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

  # 安装Python和依赖
  RUN apt-get update && apt-get install -y python3-pip
  COPY requirements.txt .
  RUN pip3 install -r requirements.txt

  # 复制应用代码
  COPY . /app
  WORKDIR /app

  # 暴露端口
  EXPOSE 8000

  # 健康检查
  HEALTHCHECK --interval=30s CMD curl -f http://localhost:8000/health

  # 启动命令
  CMD ["python3", "main.py", "--host", "0.0.0.0", "--port", "8000"]
"""

docker_compose = """
  # docker-compose.yml示例
  services:
    llm-api:
      build: .
      ports:
        - "8000:8000"
      deploy:
        resources:
          reservations:
            devices:
              - driver: nvidia
                count: 1
                capabilities: [gpu]
      environment:
        - MODEL_NAME=MyLLM-7B
        - MAX_BATCH_SIZE=8
      healthcheck:
        test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
        interval: 30s
        timeout: 10s
        retries: 3
"""

print(dockerfile)
print(docker_compose)


# ============================================================
# 5. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 5.1 延迟分布
ax1 = axes[0, 0]
latencies_ms = [l * 1000 for l in service.latency_history]
if latencies_ms:
    ax1.hist(latencies_ms, bins=20, color='#2196F3', edgecolor='black', alpha=0.7)
    ax1.axvline(x=np.mean(latencies_ms), color='red', linestyle='--',
                label=f'均值={np.mean(latencies_ms):.1f}ms')
    ax1.axvline(x=np.percentile(latencies_ms, 99), color='orange', linestyle='--',
                label=f'P99={np.percentile(latencies_ms, 99):.1f}ms')
    ax1.set_xlabel('延迟 (ms)')
    ax1.set_ylabel('请求数')
    ax1.set_title('请求延迟分布')
    ax1.legend()

# 5.2 QPS vs 并发数
ax2 = axes[0, 1]
if load_tester.results:
    concs = sorted(load_tester.results.keys())
    qps_values = [load_tester.results[c]['qps'] for c in concs]
    ax2.plot(concs, qps_values, 'go-', linewidth=2, markersize=10)
    ax2.set_xlabel('并发数')
    ax2.set_ylabel('QPS (请求/秒)')
    ax2.set_title('QPS vs 并发数')
    ax2.grid(True, alpha=0.3)
    for c, q in zip(concs, qps_values):
        ax2.annotate(f'{q:.1f}', (c, q), textcoords="offset points", xytext=(0, 10), ha='center')

# 5.3 延迟随并发变化
ax3 = axes[1, 0]
if load_tester.results:
    concs = sorted(load_tester.results.keys())
    avg_lats = [load_tester.results[c]['avg_latency'] * 1000 for c in concs]
    p99_lats = [load_tester.results[c]['p99_latency'] * 1000 for c in concs]
    ax3.plot(concs, avg_lats, 'b-o', linewidth=2, label='平均延迟')
    ax3.plot(concs, p99_lats, 'r-s', linewidth=2, label='P99延迟')
    ax3.set_xlabel('并发数')
    ax3.set_ylabel('延迟 (ms)')
    ax3.set_title('延迟 vs 并发数')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

# 5.4 监控仪表板
ax4 = axes[1, 1]
metrics = service.get_metrics()
monitor_items = ['QPS', 'GPU利用率', '成功率', '内存使用']
values = [
    float(metrics['performance'].get('current_qps', 0)) / 50 * 100,
    float(metrics['performance'].get('gpu_utilization', 0)),
    service.stats['successful_requests'] / max(service.stats['total_requests'], 1) * 100,
    float(metrics['performance'].get('memory_usage', 0)),
]
colors_mon = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']
bars = ax4.bar(monitor_items, values, color=colors_mon, edgecolor='black')
ax4.set_ylabel('百分比 (%)')
ax4.set_title('服务监控仪表板')
ax4.set_ylim(0, 110)
for bar, val in zip(bars, values):
    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
             f'{val:.1f}%', ha='center', fontsize=10)

plt.suptitle('W24-D7: 部署项目 - LLM API服务', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W24/d7_deploy_project.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存!")
