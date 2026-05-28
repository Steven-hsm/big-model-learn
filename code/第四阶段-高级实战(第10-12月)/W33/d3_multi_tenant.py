"""
W33-D3 多租户支持
==================
实现多租户架构, 包括:
- 租户数据隔离
- 知识库管理
- 配额控制

多租户让一套系统服务多个组织。
"""

import time
import hashlib
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

import numpy as np


# ============================================================
# 1. 租户模型
# ============================================================

@dataclass
class TenantQuota:
    """租户配额"""
    max_documents: int = 1000        # 最大文档数
    max_storage_mb: int = 5000       # 最大存储空间(MB)
    max_queries_per_day: int = 10000 # 每日查询上限
    max_users: int = 50              # 最大用户数
    max_collections: int = 10        # 最大知识库数


@dataclass
class Tenant:
    """租户"""
    tenant_id: str
    name: str
    plan: str = "free"               # free / pro / enterprise
    quota: TenantQuota = field(default_factory=TenantQuota)
    created_at: float = field(default_factory=time.time)
    is_active: bool = True
    settings: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'tenant_id': self.tenant_id,
            'name': self.name,
            'plan': self.plan,
            'quota': {
                'max_documents': self.quota.max_documents,
                'max_queries_per_day': self.quota.max_queries_per_day,
            },
            'is_active': self.is_active,
        }


# ============================================================
# 2. 知识库管理
# ============================================================

@dataclass
class KnowledgeBase:
    """知识库"""
    kb_id: str
    tenant_id: str
    name: str
    description: str = ""
    documents: List[str] = field(default_factory=list)  # document IDs
    created_at: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)

    @property
    def document_count(self) -> int:
        return len(self.documents)


class KnowledgeBaseManager:
    """知识库管理器"""

    def __init__(self):
        self.knowledge_bases: Dict[str, KnowledgeBase] = {}

    def create_kb(self, tenant_id: str, name: str,
                   description: str = "", tags: List[str] = None) -> KnowledgeBase:
        """创建知识库"""
        kb_id = hashlib.md5(f"{tenant_id}_{name}_{time.time()}".encode()).hexdigest()[:10]
        kb = KnowledgeBase(
            kb_id=kb_id,
            tenant_id=tenant_id,
            name=name,
            description=description,
            tags=tags or [],
        )
        self.knowledge_bases[kb_id] = kb
        return kb

    def get_kb(self, kb_id: str) -> Optional[KnowledgeBase]:
        return self.knowledge_bases.get(kb_id)

    def list_tenant_kbs(self, tenant_id: str) -> List[KnowledgeBase]:
        return [kb for kb in self.knowledge_bases.values() if kb.tenant_id == tenant_id]

    def add_document(self, kb_id: str, doc_id: str) -> bool:
        kb = self.knowledge_bases.get(kb_id)
        if kb:
            kb.documents.append(doc_id)
            return True
        return False

    def delete_kb(self, kb_id: str) -> bool:
        if kb_id in self.knowledge_bases:
            del self.knowledge_bases[kb_id]
            return True
        return False


# ============================================================
# 3. 租户管理器
# ============================================================

class TenantManager:
    """多租户管理器"""

    # 套餐配额
    PLAN_QUOTAS = {
        'free': TenantQuota(max_documents=100, max_storage_mb=500,
                             max_queries_per_day=100, max_users=5, max_collections=3),
        'pro': TenantQuota(max_documents=5000, max_storage_mb=10000,
                            max_queries_per_day=5000, max_users=20, max_collections=10),
        'enterprise': TenantQuota(max_documents=100000, max_storage_mb=100000,
                                   max_queries_per_day=100000, max_users=500, max_collections=50),
    }

    def __init__(self):
        self.tenants: Dict[str, Tenant] = {}
        self.usage: Dict[str, Dict] = defaultdict(lambda: {
            'documents': 0, 'storage_mb': 0, 'queries_today': 0, 'users': 0
        })
        self.kb_manager = KnowledgeBaseManager()

    def create_tenant(self, name: str, plan: str = "free") -> Tenant:
        """创建租户"""
        tid = hashlib.md5(f"{name}_{time.time()}".encode()).hexdigest()[:10]
        quota = self.PLAN_QUOTAS.get(plan, TenantQuota())
        tenant = Tenant(tid, name, plan, quota)
        self.tenants[tid] = tenant
        return tenant

    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        return self.tenants.get(tenant_id)

    def check_quota(self, tenant_id: str, resource: str) -> bool:
        """检查配额"""
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return False

        usage = self.usage[tenant_id]
        quota = tenant.quota

        if resource == 'document':
            return usage['documents'] < quota.max_documents
        elif resource == 'storage':
            return usage['storage_mb'] < quota.max_storage_mb
        elif resource == 'query':
            return usage['queries_today'] < quota.max_queries_per_day
        elif resource == 'user':
            return usage['users'] < quota.max_users
        elif resource == 'collection':
            return len(self.kb_manager.list_tenant_kbs(tenant_id)) < quota.max_collections
        return True

    def record_usage(self, tenant_id: str, resource: str, amount: int = 1):
        """记录使用量"""
        if resource == 'document':
            self.usage[tenant_id]['documents'] += amount
        elif resource == 'query':
            self.usage[tenant_id]['queries_today'] += amount
        elif resource == 'storage':
            self.usage[tenant_id]['storage_mb'] += amount
        elif resource == 'user':
            self.usage[tenant_id]['users'] += amount

    def get_usage_report(self, tenant_id: str) -> Dict:
        """获取使用报告"""
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return {}

        usage = self.usage[tenant_id]
        quota = tenant.quota

        return {
            'tenant': tenant.name,
            'plan': tenant.plan,
            'usage': {
                'documents': {'used': usage['documents'], 'limit': quota.max_documents,
                              'usage_rate': usage['documents'] / quota.max_documents},
                'queries_today': {'used': usage['queries_today'], 'limit': quota.max_queries_per_day,
                                   'usage_rate': usage['queries_today'] / quota.max_queries_per_day},
                'storage_mb': {'used': usage['storage_mb'], 'limit': quota.max_storage_mb,
                                'usage_rate': usage['storage_mb'] / quota.max_storage_mb},
            },
            'knowledge_bases': len(self.kb_manager.list_tenant_kbs(tenant_id)),
        }

    def list_tenants(self) -> List[Dict]:
        return [t.to_dict() for t in self.tenants.values()]


# ============================================================
# 4. 数据隔离的文档存储
# ============================================================

class IsolatedDocumentStore:
    """租户隔离的文档存储"""

    def __init__(self, tenant_manager: TenantManager):
        self.tm = tenant_manager
        self.documents: Dict[str, Dict] = {}  # doc_id -> {tenant_id, ...}

    def add_document(self, tenant_id: str, kb_id: str,
                      title: str, content: str) -> Dict:
        """添加文档(带隔离检查)"""
        # 检查配额
        if not self.tm.check_quota(tenant_id, 'document'):
            return {'error': '文档数量已达上限', 'status': 403}

        # 创建文档
        doc_id = hashlib.md5(f"{tenant_id}_{title}_{time.time()}".encode()).hexdigest()[:10]
        self.documents[doc_id] = {
            'doc_id': doc_id,
            'tenant_id': tenant_id,
            'kb_id': kb_id,
            'title': title,
            'content': content,
            'created_at': time.time(),
        }

        # 更新知识库
        self.tm.kb_manager.add_document(kb_id, doc_id)
        self.tm.record_usage(tenant_id, 'document')
        self.tm.record_usage(tenant_id, 'storage', amount=len(content) / 1024 / 1024)

        return {'doc_id': doc_id, 'status': 200}

    def search(self, tenant_id: str, query: str, top_k: int = 5) -> List[Dict]:
        """搜索(自动隔离)"""
        if not self.tm.check_quota(tenant_id, 'query'):
            return []

        self.tm.record_usage(tenant_id, 'query')

        # 只搜索该租户的文档
        results = []
        for doc_id, doc in self.documents.items():
            if doc['tenant_id'] != tenant_id:
                continue  # 关键: 跳过其他租户的数据
            if query.lower() in doc['content'].lower():
                results.append({
                    'doc_id': doc['doc_id'],
                    'title': doc['title'],
                    'score': 1.0,
                })

        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

    def get_document(self, tenant_id: str, doc_id: str) -> Optional[Dict]:
        """获取文档(带隔离)"""
        doc = self.documents.get(doc_id)
        if doc and doc['tenant_id'] == tenant_id:
            return doc
        return None  # 不允许访问其他租户的文档


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("W33-D3 多租户支持")
    print("=" * 60)

    tm = TenantManager()
    store = IsolatedDocumentStore(tm)

    # --- 创建租户 ---
    print("\n--- 1. 创建租户 ---")
    tenants = {
        'free': tm.create_tenant("个人用户A", "free"),
        'pro': tm.create_tenant("中小企业B", "pro"),
        'enterprise': tm.create_tenant("大型企业C", "enterprise"),
    }

    for plan, tenant in tenants.items():
        print(f"  {tenant.name} (计划: {plan}): ID={tenant.tenant_id}")
        print(f"    配额: 文档={tenant.quota.max_documents}, "
              f"查询/天={tenant.quota.max_queries_per_day}")

    # --- 创建知识库 ---
    print(f"\n{'='*60}")
    print("--- 2. 知识库管理 ---")
    print(f"{'='*60}")

    for plan, tenant in tenants.items():
        kb = tm.kb_manager.create_kb(tenant.tenant_id, f"{tenant.name}的知识库")
        print(f"  {tenant.name}: 创建知识库 {kb.kb_id}")

    # --- 添加文档 ---
    print(f"\n{'='*60}")
    print("--- 3. 数据隔离 ---")
    print(f"{'='*60}")

    docs_data = {
        'free': [("RAG入门", "RAG是检索增强生成技术"), ("Python基础", "Python编程入门")],
        'pro': [("企业文档", "公司内部规章制度"), ("技术手册", "系统使用说明"), ("项目文档", "项目架构设计")],
        'enterprise': [("金融报告", "2024年财务分析"), ("产品文档", "产品使用手册"),
                        ("合规文档", "合规政策说明"), ("培训资料", "新员工培训材料")],
    }

    for plan, tenant in tenants.items():
        kbs = tm.kb_manager.list_tenant_kbs(tenant.tenant_id)
        kb_id = kbs[0].kb_id if kbs else None
        for title, content in docs_data[plan]:
            result = store.add_document(tenant.tenant_id, kb_id, title, content)
            if result.get('status') == 200:
                print(f"  [{tenant.name}] 添加: {title}")

    # --- 搜索隔离验证 ---
    print(f"\n--- 搜索隔离验证 ---")
    for plan, tenant in tenants.items():
        results = store.search(tenant.tenant_id, "文档")
        print(f"  {tenant.name} 搜索「文档」: {len(results)}条结果 (只能看到自己的)")

    # 尝试跨租户访问
    free_doc_ids = [d['doc_id'] for d in store.documents.values()
                     if d['tenant_id'] == tenants['free'].tenant_id]
    if free_doc_ids:
        cross_result = store.get_document(tenants['pro'].tenant_id, free_doc_ids[0])
        print(f"\n  跨租户访问: {'成功(错误!)' if cross_result else '拒绝(正确!)'}")

    # --- 使用报告 ---
    print(f"\n{'='*60}")
    print("--- 4. 使用报告 ---")
    print(f"{'='*60}")

    for plan, tenant in tenants.items():
        report = tm.get_usage_report(tenant.tenant_id)
        usage = report['usage']
        print(f"\n  {tenant.name} ({plan}):")
        for resource, info in usage.items():
            rate = info['usage_rate']
            bar = '#' * int(rate * 20)
            print(f"    {resource:<15s}: {info['used']}/{info['limit']} ({rate:.0%}) {bar}")

    # --- 可视化 ---
    if HAS_PLT:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 各租户使用量
        ax = axes[0]
        plans = ['free', 'pro', 'enterprise']
        doc_usage = [tm.get_usage_report(tenants[p].tenant_id)['usage']['documents']['usage_rate'] * 100
                      for p in plans]
        query_usage = [tm.get_usage_report(tenants[p].tenant_id)['usage']['queries_today']['usage_rate'] * 100
                        for p in plans]

        x = np.arange(len(plans))
        width = 0.35
        ax.bar(x - width/2, doc_usage, width, label='文档使用率', color='#3498db')
        ax.bar(x + width/2, query_usage, width, label='查询使用率', color='#e74c3c')
        ax.set_xticks(x)
        ax.set_xticklabels(plans)
        ax.set_ylabel('使用率(%)')
        ax.set_title('各租户资源使用率', fontweight='bold')
        ax.legend()

        # 配额对比
        ax = axes[1]
        max_docs = [tenants[p].quota.max_documents for p in plans]
        colors = ['#2ecc71', '#3498db', '#9b59b6']
        bars = ax.bar(plans, max_docs, color=colors)
        for bar, val in zip(bars, max_docs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    str(val), ha='center', va='bottom')
        ax.set_ylabel('最大文档数')
        ax.set_title('各套餐配额对比', fontweight='bold')

        plt.tight_layout()
        plt.savefig('D:/code/big-model-learn/code/q_01/W33/d3_multi_tenant.png', dpi=150)
        print("\n图表已保存为 d3_multi_tenant.png")
        plt.close()

    print("\n完成!")
