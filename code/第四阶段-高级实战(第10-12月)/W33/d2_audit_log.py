"""
W33-D2 审计日志
================
实现审计日志系统, 包括:
- 操作记录
- 查询历史追踪
- 日志分析与统计

审计日志保障系统操作的合规性和可追溯性。
"""

import time
import json
import hashlib
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter
from enum import Enum

try:
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

import numpy as np


# ============================================================
# 1. 审计日志模型
# ============================================================

class ActionType(Enum):
    """操作类型"""
    QUERY = "query"               # 查询
    UPLOAD = "upload"             # 上传文档
    DELETE = "delete"             # 删除文档
    LOGIN = "login"               # 登录
    LOGOUT = "logout"             # 登出
    CONFIG_CHANGE = "config"      # 配置变更
    PERMISSION_CHANGE = "perm"    # 权限变更
    EXPORT = "export"             # 数据导出
    SYSTEM = "system"             # 系统操作


class AuditLevel(Enum):
    """日志级别"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class AuditLog:
    """审计日志条目"""
    log_id: str
    timestamp: float
    action: str                    # ActionType.value
    level: str                     # AuditLevel.value
    user_id: str
    user_role: str
    description: str
    resource: str = ""             # 操作的资源
    ip_address: str = "127.0.0.1"
    request_data: Dict = field(default_factory=dict)
    response_status: int = 200
    duration_ms: float = 0
    extra: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================================
# 2. 审计日志管理器
# ============================================================

class AuditLogger:
    """审计日志管理器"""

    def __init__(self, max_logs: int = 50000):
        self.logs: List[AuditLog] = []
        self.max_logs = max_logs

    def log(self, action: str, user_id: str, user_role: str,
             description: str, level: str = "info", **kwargs) -> AuditLog:
        """记录审计日志"""
        log_id = hashlib.md5(f"{action}_{time.time()}_{user_id}".encode()).hexdigest()[:10]

        entry = AuditLog(
            log_id=log_id,
            timestamp=time.time(),
            action=action,
            level=level,
            user_id=user_id,
            user_role=user_role,
            description=description,
            **kwargs,
        )

        self.logs.append(entry)
        if len(self.logs) > self.max_logs:
            self.logs = self.logs[-self.max_logs:]

        return entry

    def get_recent(self, n: int = 20) -> List[AuditLog]:
        return self.logs[-n:]

    def get_by_user(self, user_id: str, n: int = 50) -> List[AuditLog]:
        return [l for l in self.logs if l.user_id == user_id][-n:]

    def get_by_action(self, action: str, n: int = 50) -> List[AuditLog]:
        return [l for l in self.logs if l.action == action][-n:]

    def get_by_time_range(self, start: float, end: float) -> List[AuditLog]:
        return [l for l in self.logs if start <= l.timestamp <= end]

    def get_critical(self, n: int = 20) -> List[AuditLog]:
        return [l for l in self.logs if l.level == "critical"][-n:]


# ============================================================
# 3. 查询历史追踪
# ============================================================

class QueryHistory:
    """查询历史追踪"""

    def __init__(self, audit_logger: AuditLogger):
        self.audit = audit_logger

    def record_query(self, user_id: str, user_role: str,
                      query: str, answer: str, latency_ms: float,
                      sources_count: int = 0):
        """记录查询"""
        self.audit.log(
            action=ActionType.QUERY.value,
            user_id=user_id,
            user_role=user_role,
            description=f"查询: {query[:50]}",
            request_data={'query': query},
            duration_ms=latency_ms,
            extra={'sources_count': sources_count, 'answer_length': len(answer)},
        )

    def get_user_history(self, user_id: str, n: int = 20) -> List[Dict]:
        """获取用户查询历史"""
        logs = self.audit.get_by_user(user_id, n)
        return [
            {
                'query': l.request_data.get('query', ''),
                'time': l.timestamp,
                'latency_ms': l.duration_ms,
                'sources': l.extra.get('sources_count', 0),
            }
            for l in reversed(logs) if l.action == ActionType.QUERY.value
        ]

    def get_popular_queries(self, n: int = 10) -> List[Dict]:
        """获取热门查询"""
        query_counts = Counter()
        for log in self.audit.logs:
            if log.action == ActionType.QUERY.value:
                query = log.request_data.get('query', '')
                if query:
                    query_counts[query] += 1

        return [{'query': q, 'count': c} for q, c in query_counts.most_common(n)]


# ============================================================
# 4. 日志分析器
# ============================================================

class AuditAnalyzer:
    """审计日志分析器"""

    def __init__(self, audit_logger: AuditLogger):
        self.audit = audit_logger

    def action_distribution(self) -> Dict[str, int]:
        """操作类型分布"""
        return dict(Counter(l.action for l in self.audit.logs))

    def user_activity(self, top_n: int = 10) -> Dict[str, int]:
        """用户活跃度"""
        return dict(Counter(l.user_id for l in self.audit.logs).most_common(top_n))

    def hourly_distribution(self) -> Dict[int, int]:
        """按小时分布"""
        hourly = defaultdict(int)
        for log in self.audit.logs:
            hour = time.localtime(log.timestamp).tm_hour
            hourly[hour] += 1
        return dict(sorted(hourly.items()))

    def error_rate(self) -> float:
        """错误率"""
        if not self.audit.logs:
            return 0.0
        errors = sum(1 for l in self.audit.logs if l.response_status >= 400)
        return errors / len(self.audit.logs)

    def avg_latency(self) -> float:
        """平均延迟"""
        latencies = [l.duration_ms for l in self.audit.logs if l.duration_ms > 0]
        return np.mean(latencies) if latencies else 0.0

    def security_summary(self) -> Dict:
        """安全摘要"""
        critical = self.audit.get_critical()
        login_attempts = [l for l in self.audit.logs if l.action == ActionType.LOGIN.value]
        failed_logins = [l for l in login_attempts if l.response_status != 200]

        return {
            'critical_events': len(critical),
            'total_login_attempts': len(login_attempts),
            'failed_logins': len(failed_logins),
            'unique_users': len(set(l.user_id for l in self.audit.logs)),
            'data_exports': len([l for l in self.audit.logs if l.action == ActionType.EXPORT.value]),
        }

    def print_report(self):
        """打印分析报告"""
        print(f"\n{'='*60}")
        print("审计日志分析报告")
        print(f"{'='*60}")
        print(f"总日志数: {len(self.audit.logs)}")

        print(f"\n[操作分布]")
        for action, count in self.action_distribution().items():
            bar = '*' * (count // 5)
            print(f"  {action:<15s}: {count:4d} {bar}")

        print(f"\n[用户活跃度TOP5]")
        for user, count in list(self.user_activity(5).items()):
            print(f"  {user:<15s}: {count:4d}次")

        print(f"\n[系统指标]")
        print(f"  平均延迟: {self.avg_latency():.1f}ms")
        print(f"  错误率: {self.error_rate():.2%}")

        sec = self.security_summary()
        print(f"\n[安全摘要]")
        print(f"  关键事件: {sec['critical_events']}")
        print(f"  登录尝试: {sec['total_login_attempts']} (失败: {sec['failed_logins']})")
        print(f"  活跃用户: {sec['unique_users']}")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("W33-D2 审计日志")
    print("=" * 60)

    audit = AuditLogger()
    query_history = QueryHistory(audit)

    # --- 模拟操作日志 ---
    import random
    random.seed(42)

    users = [
        ("user_001", "admin"), ("user_002", "editor"),
        ("user_003", "viewer"), ("user_004", "editor"), ("user_005", "viewer"),
    ]

    actions = [
        (ActionType.QUERY, "info", 200),
        (ActionType.UPLOAD, "info", 200),
        (ActionType.DELETE, "warning", 200),
        (ActionType.LOGIN, "info", 200),
        (ActionType.CONFIG_CHANGE, "warning", 200),
        (ActionType.EXPORT, "info", 200),
        (ActionType.LOGIN, "warning", 401),  # 登录失败
        (ActionType.PERMISSION_CHANGE, "critical", 200),
    ]

    queries = ["什么是RAG?", "Python教程", "向量检索", "如何部署", "Prompt工程"]

    for _ in range(200):
        user_id, role = random.choice(users)
        action, level, status = random.choice(actions)

        if action == ActionType.QUERY:
            query = random.choice(queries)
            latency = random.uniform(50, 500)
            query_history.record_query(user_id, role, query, "回答...", latency, random.randint(1, 5))
        else:
            audit.log(
                action=action.value,
                user_id=user_id,
                user_role=role,
                description=f"执行了{action.value}操作",
                level=level.value,
                response_status=status,
                duration_ms=random.uniform(10, 200),
            )

    # --- 查询历史 ---
    print("\n--- 查询历史 ---")
    history = query_history.get_user_history("user_001", n=5)
    for h in history:
        print(f"  查询: {h['query']}, 延迟: {h['latency_ms']:.0f}ms")

    print("\n热门查询:")
    popular = query_history.get_popular_queries(5)
    for item in popular:
        print(f"  「{item['query']}」: {item['count']}次")

    # --- 日志分析 ---
    analyzer = AuditAnalyzer(audit)
    analyzer.print_report()

    # --- 可视化 ---
    if HAS_PLT:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 操作分布
        dist = analyzer.action_distribution()
        ax = axes[0, 0]
        ax.bar(dist.keys(), dist.values(), color='#3498db')
        ax.set_title('操作类型分布', fontweight='bold')
        ax.set_ylabel('次数')
        ax.tick_params(axis='x', rotation=15)

        # 用户活跃度
        activity = analyzer.user_activity(5)
        ax = axes[0, 1]
        ax.barh(list(activity.keys()), list(activity.values()), color='#2ecc71')
        ax.set_title('用户活跃度', fontweight='bold')
        ax.set_xlabel('操作次数')

        # 按小时分布
        hourly = analyzer.hourly_distribution()
        if hourly:
            ax = axes[1, 0]
            ax.bar(hourly.keys(), hourly.values(), color='#9b59b6')
            ax.set_xlabel('小时')
            ax.set_ylabel('操作数')
            ax.set_title('操作按小时分布', fontweight='bold')

        # 查询热度
        if popular:
            ax = axes[1, 1]
            queries_names = [p['query'] for p in popular]
            counts = [p['count'] for p in popular]
            ax.barh(queries_names, counts, color='#e74c3c')
            ax.set_title('热门查询TOP5', fontweight='bold')
            ax.set_xlabel('查询次数')

        plt.tight_layout()
        plt.savefig('D:/code/big-model-learn/code/q_01/W33/d2_audit_log.png', dpi=150)
        print("\n图表已保存为 d2_audit_log.png")
        plt.close()

    print("\n完成!")
