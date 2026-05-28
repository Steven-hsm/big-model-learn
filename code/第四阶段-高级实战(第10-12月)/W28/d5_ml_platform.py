"""
W28-D5 ML平台设计
==================
ML平台设计, 多租户, 资源管理, GPU调度, 平台架构模板
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W28-D5 ML平台设计")
print("=" * 60)

# ============================================================
# 1. ML平台架构
# ============================================================
print("\n--- 1. ML平台架构 ---")
print("""
  ML平台核心模块:
  ┌────────────────────────────────────────────────────────────┐
  │                     ML平台 (自建)                           │
  │                                                             │
  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐        │
  │  │用户界面 │ │API服务  │ │认证授权 │ │租户管理 │        │
  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘        │
  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐        │
  │  │数据管理 │ │训练服务 │ │部署服务 │ │监控服务 │        │
  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘        │
  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐        │
  │  │调度器   │ │资源管理 │ │GPU池    │ │存储管理 │        │
  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘        │
  └────────────────────────────────────────────────────────────┘

  参考: Kubeflow, MLflow, SageMaker, Vertex AI, Azure ML
""")

# ============================================================
# 2. 多租户管理
# ============================================================
print("\n--- 2. 多租户管理 ---")


class Tenant:
    """租户"""

    def __init__(self, tenant_id, name, tier='standard', quotas=None):
        self.tenant_id = tenant_id
        self.name = name
        self.tier = tier  # standard, premium, enterprise
        self.quotas = quotas or self._default_quotas(tier)
        self.usage = defaultdict(float)
        self.projects = []

    def _default_quotas(self, tier):
        quotas = {
            'standard': {
                'max_projects': 5,
                'max_experiments': 50,
                'max_models': 20,
                'max_gpu_hours': 100,
                'max_storage_gb': 50,
                'max_concurrent_jobs': 2,
            },
            'premium': {
                'max_projects': 20,
                'max_experiments': 200,
                'max_models': 100,
                'max_gpu_hours': 500,
                'max_storage_gb': 200,
                'max_concurrent_jobs': 5,
            },
            'enterprise': {
                'max_projects': -1,
                'max_experiments': -1,
                'max_models': -1,
                'max_gpu_hours': -1,
                'max_storage_gb': 1000,
                'max_concurrent_jobs': 20,
            },
        }
        return quotas.get(tier, quotas['standard'])

    def check_quota(self, resource):
        """检查配额"""
        limit = self.quotas.get(resource, -1)
        if limit == -1:
            return True
        current = self.usage.get(resource, 0)
        return current < limit

    def consume(self, resource, amount=1):
        """消耗资源"""
        self.usage[resource] += amount

    def get_usage_report(self):
        """获取使用报告"""
        print(f"\n  租户: {self.name} ({self.tier})")
        print(f"  {'资源':<20} {'已用':<12} {'配额':<12} {'使用率'}")
        print("  " + "-" * 55)
        for resource, limit in self.quotas.items():
            used = self.usage.get(resource, 0)
            if limit == -1:
                rate = '无限'
            else:
                rate = f'{used/limit:.1%}'
            print(f"  {resource:<20} {used:<12} {limit:<12} {rate}")


class MultiTenantManager:
    """多租户管理器"""

    def __init__(self):
        self.tenants = {}

    def register_tenant(self, tenant_id, name, tier='standard'):
        tenant = Tenant(tenant_id, name, tier)
        self.tenants[tenant_id] = tenant
        print(f"  注册租户: {name} ({tier})")
        return tenant

    def get_tenant(self, tenant_id):
        return self.tenants.get(tenant_id)

    def check_access(self, tenant_id, resource):
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return False
        return tenant.check_quota(resource)

    def list_tenants(self):
        print(f"\n  租户列表 ({len(self.tenants)} 个):")
        for tid, t in self.tenants.items():
            print(f"    {tid}: {t.name} ({t.tier})")


# 创建租户
mtm = MultiTenantManager()
mtm.register_tenant('team_a', '推荐算法团队', 'premium')
mtm.register_tenant('team_b', 'NLP团队', 'standard')
mtm.register_tenant('team_c', '计算机视觉团队', 'enterprise')

# 模拟使用
team_a = mtm.get_tenant('team_a')
team_a.consume('max_projects', 3)
team_a.consume('max_experiments', 45)
team_a.consume('max_gpu_hours', 120)

team_b = mtm.get_tenant('team_b')
team_b.consume('max_projects', 4)
team_b.consume('max_experiments', 30)
team_b.consume('max_gpu_hours', 80)

team_a.get_usage_report()
team_b.get_usage_report()

# ============================================================
# 3. 资源管理器
# ============================================================
print("\n--- 3. 资源管理器 ---")


class Resource:
    """计算资源"""

    def __init__(self, resource_id, resource_type, specs):
        self.resource_id = resource_id
        self.resource_type = resource_type  # cpu, gpu, tpu
        self.specs = specs
        self.status = 'idle'  # idle, busy, maintenance
        self.assigned_to = None
        self.job_history = []


class ResourceManager:
    """资源管理器"""

    def __init__(self):
        self.resources = {}
        self.job_queue = []

    def add_resource(self, resource_id, resource_type, specs):
        resource = Resource(resource_id, resource_type, specs)
        self.resources[resource_id] = resource
        print(f"  添加资源: {resource_id} ({resource_type}, {specs})")

    def allocate(self, requirements, tenant_id, job_id):
        """分配资源"""
        for rid, resource in self.resources.items():
            if resource.status == 'idle' and self._meets_requirements(resource, requirements):
                resource.status = 'busy'
                resource.assigned_to = {'tenant': tenant_id, 'job': job_id}
                print(f"  分配 {rid} -> {tenant_id}/{job_id}")
                return rid
        # 加入队列
        self.job_queue.append({
            'requirements': requirements,
            'tenant_id': tenant_id,
            'job_id': job_id,
            'queued_at': datetime.now().isoformat(),
        })
        print(f"  资源不足, {job_id} 加入等待队列")
        return None

    def release(self, resource_id):
        """释放资源"""
        if resource_id in self.resources:
            resource = self.resources[resource_id]
            resource.status = 'idle'
            resource.assigned_to = None
            print(f"  释放 {resource_id}")

            # 尝试分配队列中的任务
            if self.job_queue:
                next_job = self.job_queue.pop(0)
                self.allocate(next_job['requirements'],
                              next_job['tenant_id'], next_job['job_id'])

    def _meets_requirements(self, resource, requirements):
        """检查资源是否满足需求"""
        for key, value in requirements.items():
            if resource.specs.get(key, 0) < value:
                return False
        return True

    def status(self):
        """资源状态"""
        idle = sum(1 for r in self.resources.values() if r.status == 'idle')
        busy = sum(1 for r in self.resources.values() if r.status == 'busy')
        print(f"\n  资源状态: 空闲={idle}, 繁忙={busy}, 总计={len(self.resources)}")
        print(f"  等待队列: {len(self.job_queue)} 个任务")


# 创建资源
rm = ResourceManager()
rm.add_resource('gpu-01', 'gpu', {'gpu_count': 1, 'gpu_memory': 16, 'cpu': 8, 'memory': 32})
rm.add_resource('gpu-02', 'gpu', {'gpu_count': 1, 'gpu_memory': 24, 'cpu': 8, 'memory': 64})
rm.add_resource('gpu-03', 'gpu', {'gpu_count': 4, 'gpu_memory': 80, 'cpu': 32, 'memory': 128})
rm.add_resource('cpu-01', 'cpu', {'cpu': 16, 'memory': 64})
rm.add_resource('cpu-02', 'cpu', {'cpu': 32, 'memory': 128})

# 分配资源
rm.allocate({'gpu_count': 1, 'gpu_memory': 16}, 'team_a', 'train_bert')
rm.allocate({'gpu_count': 1, 'gpu_memory': 16}, 'team_b', 'train_gpt')
rm.allocate({'gpu_count': 4, 'gpu_memory': 40}, 'team_c', 'train_resnet')
rm.allocate({'cpu': 16}, 'team_a', 'feature_engineering')

rm.status()

# 释放
rm.release('gpu-01')
rm.status()

# ============================================================
# 4. GPU调度策略
# ============================================================
print("\n--- 4. GPU调度策略 ---")
print("""
  GPU调度策略:

  1. FIFO (先进先出):
     - 简单公平
     - 不考虑优先级

  2. 优先级调度:
     - 按任务优先级排序
     - 生产 > 预发布 > 实验

  3. 抢占式调度:
     - 高优先级任务可抢占低优先级
     - 适合混合工作负载

  4. 分时调度 (MPS/MIG):
     - GPU分时共享
     - 适合小任务

  5. 成本优化:
     - 使用Spot/Preemptible实例
     - 自动缩容空闲GPU

  GPU利用率优化:
    - 梯度累积: 小batch模拟大batch
    - 混合精度: FP16加速
    - 数据预取: 避免GPU空闲
    - 多任务共享: MPS/MIG
""")


class GPUScheduler:
    """GPU调度器 (简化版)"""

    def __init__(self):
        self.gpu_pool = {}
        self.job_queue = []

    def add_gpu(self, gpu_id, memory_gb, gpu_type='A100'):
        self.gpu_pool[gpu_id] = {
            'memory': memory_gb,
            'type': gpu_type,
            'status': 'idle',
            'current_job': None,
        }

    def submit_job(self, job_id, gpu_memory_needed, priority=1, tenant_id=''):
        """提交任务 (优先级1-3, 3最高)"""
        self.job_queue.append({
            'job_id': job_id,
            'gpu_memory': gpu_memory_needed,
            'priority': priority,
            'tenant_id': tenant_id,
            'submit_time': datetime.now(),
        })
        # 按优先级排序
        self.job_queue.sort(key=lambda j: -j['priority'])

    def schedule(self):
        """调度任务"""
        scheduled = []
        remaining = []

        for job in self.job_queue:
            placed = False
            for gpu_id, gpu in self.gpu_pool.items():
                if gpu['status'] == 'idle' and gpu['memory'] >= job['gpu_memory']:
                    gpu['status'] = 'busy'
                    gpu['current_job'] = job['job_id']
                    scheduled.append({
                        'job': job['job_id'],
                        'gpu': gpu_id,
                        'priority': job['priority'],
                        'tenant': job['tenant_id'],
                    })
                    placed = True
                    break
            if not placed:
                remaining.append(job)

        self.job_queue = remaining
        return scheduled


# GPU调度示例
scheduler = GPUScheduler()
scheduler.add_gpu('gpu-0', 16)
scheduler.add_gpu('gpu-1', 24)
scheduler.add_gpu('gpu-2', 80)

scheduler.submit_job('train_llm', 70, priority=3, tenant_id='team_c')
scheduler.submit_job('train_bert', 16, priority=2, tenant_id='team_a')
scheduler.submit_job('train_gpt', 20, priority=2, tenant_id='team_b')
scheduler.submit_job('train_resnet', 16, priority=1, tenant_id='team_a')
scheduler.submit_job('train_yolo', 24, priority=1, tenant_id='team_b')

scheduled = scheduler.schedule()
print(f"\n  GPU调度结果:")
for s in scheduled:
    print(f"    {s['job']:<15} -> {s['gpu']:<8} (优先级={s['priority']}, 租户={s['tenant']})")

if scheduler.job_queue:
    print(f"  等待中: {[j['job_id'] for j in scheduler.job_queue]}")

# ============================================================
# 5. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 左上: 租户资源使用
ax = axes[0, 0]
tenants = ['team_a\n(premium)', 'team_b\n(standard)']
resources = ['projects', 'experiments', 'GPU hours']
usage_a = [3, 45, 120]
quota_a = [20, 200, 500]
usage_b = [4, 30, 80]
quota_b = [5, 50, 100]

x = np.arange(len(resources))
width = 0.2
ax.bar(x - 1.5*width, usage_a, width, label='A已用', color='#2196F3')
ax.bar(x - 0.5*width, quota_a, width, label='A配额', color='#90CAF9')
ax.bar(x + 0.5*width, usage_b, width, label='B已用', color='#F44336')
ax.bar(x + 1.5*width, quota_b, width, label='B配额', color='#FFCDD2')
ax.set_xticks(x)
ax.set_xticklabels(resources)
ax.set_ylabel('数量')
ax.set_title('多租户资源使用')
ax.legend(fontsize=8)

# 右上: GPU利用率
ax = axes[0, 1]
gpu_ids = list(scheduler.gpu_pool.keys())
gpu_mem = [scheduler.gpu_pool[g]['memory'] for g in gpu_ids]
gpu_jobs = [scheduler.gpu_pool[g].get('current_job', 'idle') for g in gpu_ids]
colors = ['#4CAF50' if j == 'idle' else '#F44336' for j in gpu_jobs]
ax.bar(gpu_ids, gpu_mem, color=colors, edgecolor='#333')
for i, (gid, mem, job) in enumerate(zip(gpu_ids, gpu_mem, gpu_jobs)):
    ax.text(i, mem + 1, job or 'idle', ha='center', fontsize=9)
ax.set_ylabel('GPU内存 (GB)')
ax.set_title('GPU池状态')

# 左下: 调度甘特图
ax = axes[1, 0]
job_names = [s['job'] for s in scheduled]
gpu_names = [s['gpu'] for s in scheduled]
durations = [4, 2, 3][:len(scheduled)]  # 模拟时长
colors_job = plt.cm.Set2(np.linspace(0, 1, len(scheduled)))
cumulative = 0
for i, (job, gpu, dur) in enumerate(zip(job_names, gpu_names, durations)):
    ax.barh(gpu, dur, left=cumulative, color=colors_job[i], edgecolor='#333')
    ax.text(cumulative + dur/2, gpu, job, ha='center', va='center', fontsize=8)
    cumulative_start = cumulative
    cumulative += dur

ax.set_xlabel('时间 (小时, 模拟)')
ax.set_title('GPU调度甘特图')

# 右下: 平台架构层次
ax = axes[1, 1]
ax.axis('off')
ax.set_title('ML平台架构层次')

layers = [
    ('用户层', ['Jupyter', 'VS Code', 'Web UI'], '#FFCDD2'),
    ('服务层', ['训练API', '部署API', '特征API'], '#FFE0B2'),
    ('调度层', ['任务队列', 'GPU调度', '资源管理'], '#BBDEFB'),
    ('基础设施', ['K8s', 'Docker', '存储'], '#C8E6C9'),
]

for i, (layer_name, components, color) in enumerate(layers):
    y = 3.5 - i * 1.0
    ax.add_patch(plt.Rectangle((0.5, y - 0.35), 9, 0.7,
                                facecolor=color, edgecolor='#333', linewidth=1))
    ax.text(1.5, y, layer_name, ha='center', va='center', fontsize=10, fontweight='bold')
    for j, comp in enumerate(components):
        ax.text(4 + j * 2, y, comp, ha='center', va='center', fontsize=9)

ax.set_xlim(0, 10)
ax.set_ylim(-0.5, 4.5)

plt.tight_layout()
plt.savefig('D:/code/big-model-learn/code/q_01/W28/d5_ml_platform.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存: d5_ml_platform.png")

print("\n完成! ML平台设计要点:")
print("  1. 多租户: 配额管理, 资源隔离, 计费")
print("  2. 资源管理: CPU/GPU池, 动态分配")
print("  3. GPU调度: 优先级, 抢占, 分时共享")
print("  4. 平台架构: 用户层/服务层/调度层/基础设施")
print("  5. 渐进建设: 从小开始, 逐步完善")
