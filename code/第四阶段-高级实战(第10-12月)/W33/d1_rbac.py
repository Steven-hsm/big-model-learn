"""
W33-D1 RBAC权限控制
====================
实现基于角色的访问控制(RBAC), 包括:
- 用户角色定义
- 权限检查机制
- API鉴权装饰器

RBAC是保障系统安全的基础。
"""

import time
import hashlib
import json
from typing import List, Dict, Set, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps


# ============================================================
# 1. 角色与权限定义
# ============================================================

class Permission(Enum):
    """权限枚举"""
    # 文档权限
    DOC_READ = "doc:read"
    DOC_WRITE = "doc:write"
    DOC_DELETE = "doc:delete"
    DOC_UPLOAD = "doc:upload"

    # 查询权限
    QUERY_BASIC = "query:basic"
    QUERY_ADVANCED = "query:advanced"

    # 管理权限
    USER_MANAGE = "user:manage"
    SYSTEM_CONFIG = "system:config"
    SYSTEM_MONITOR = "system:monitor"
    LOG_VIEW = "log:view"


class Role(Enum):
    """角色枚举"""
    ADMIN = "admin"
    EDITOR = "editor"
    VIEWER = "viewer"
    GUEST = "guest"


# 角色权限映射
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.ADMIN: {
        # 管理员拥有所有权限
        Permission.DOC_READ, Permission.DOC_WRITE, Permission.DOC_DELETE, Permission.DOC_UPLOAD,
        Permission.QUERY_BASIC, Permission.QUERY_ADVANCED,
        Permission.USER_MANAGE, Permission.SYSTEM_CONFIG, Permission.SYSTEM_MONITOR, Permission.LOG_VIEW,
    },
    Role.EDITOR: {
        Permission.DOC_READ, Permission.DOC_WRITE, Permission.DOC_UPLOAD,
        Permission.QUERY_BASIC, Permission.QUERY_ADVANCED,
        Permission.LOG_VIEW,
    },
    Role.VIEWER: {
        Permission.DOC_READ,
        Permission.QUERY_BASIC,
    },
    Role.GUEST: {
        Permission.QUERY_BASIC,
    },
}


# ============================================================
# 2. 用户模型
# ============================================================

@dataclass
class User:
    """用户"""
    user_id: str
    username: str
    role: Role
    email: str = ""
    created_at: float = field(default_factory=time.time)
    is_active: bool = True
    extra_permissions: Set[Permission] = field(default_factory=set)

    @property
    def permissions(self) -> Set[Permission]:
        """获取用户的所有权限"""
        perms = ROLE_PERMISSIONS.get(self.role, set())
        return perms | self.extra_permissions

    def has_permission(self, permission: Permission) -> bool:
        """检查是否有某项权限"""
        return permission in self.permissions and self.is_active

    def to_dict(self) -> Dict:
        return {
            'user_id': self.user_id,
            'username': self.username,
            'role': self.role.value,
            'email': self.email,
            'is_active': self.is_active,
        }


# ============================================================
# 3. 权限管理器
# ============================================================

class RBACManager:
    """RBAC权限管理器"""

    def __init__(self):
        self.users: Dict[str, User] = {}
        self.api_keys: Dict[str, str] = {}  # api_key -> user_id
        self._init_default_users()

    def _init_default_users(self):
        """初始化默认用户"""
        defaults = [
            ("admin_001", "admin", Role.ADMIN, "admin@example.com"),
            ("editor_001", "editor", Role.EDITOR, "editor@example.com"),
            ("viewer_001", "viewer", Role.VIEWER, "viewer@example.com"),
            ("guest_001", "guest", Role.GUEST, "guest@example.com"),
        ]
        for uid, name, role, email in defaults:
            self.users[uid] = User(uid, name, role, email)
            # 生成API Key
            api_key = hashlib.md5(f"{uid}_{time.time()}".encode()).hexdigest()[:16]
            self.api_keys[api_key] = uid

    def create_user(self, username: str, role: Role, email: str = "") -> User:
        """创建用户"""
        uid = hashlib.md5(f"{username}_{time.time()}".encode()).hexdigest()[:10]
        user = User(uid, username, role, email)
        self.users[uid] = user
        api_key = hashlib.md5(f"{uid}_{time.time()}".encode()).hexdigest()[:16]
        self.api_keys[api_key] = uid
        return user

    def authenticate(self, api_key: str) -> Optional[User]:
        """通过API Key认证"""
        user_id = self.api_keys.get(api_key)
        if user_id and user_id in self.users:
            user = self.users[user_id]
            if user.is_active:
                return user
        return None

    def check_permission(self, user_id: str, permission: Permission) -> bool:
        """检查权限"""
        user = self.users.get(user_id)
        if user:
            return user.has_permission(permission)
        return False

    def get_user(self, user_id: str) -> Optional[User]:
        return self.users.get(user_id)

    def list_users(self) -> List[Dict]:
        return [u.to_dict() for u in self.users.values()]

    def update_role(self, user_id: str, new_role: Role) -> bool:
        """更新用户角色"""
        user = self.users.get(user_id)
        if user:
            user.role = new_role
            return True
        return False

    def deactivate_user(self, user_id: str) -> bool:
        """停用用户"""
        user = self.users.get(user_id)
        if user:
            user.is_active = False
            return True
        return False


# ============================================================
# 4. API鉴权装饰器
# ============================================================

class AuthorizationError(Exception):
    """授权错误"""
    pass


class AuthenticationError(Exception):
    """认证错误"""
    pass


def require_permission(permission: Permission):
    """权限检查装饰器"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 从kwargs中获取用户
            user = kwargs.get('current_user')
            if not user:
                raise AuthenticationError("未认证")

            if not user.has_permission(permission):
                raise AuthorizationError(
                    f"用户'{user.username}'没有权限'{permission.value}'"
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ============================================================
# 5. 模拟API路由(带权限控制)
# ============================================================

class SecureAPI:
    """带权限控制的API"""

    def __init__(self, rbac: RBACManager):
        self.rbac = rbac
        self.access_log = []

    def _log_access(self, user: User, action: str, success: bool):
        self.access_log.append({
            'user': user.username,
            'role': user.role.value,
            'action': action,
            'success': success,
            'timestamp': time.time(),
        })

    def query(self, api_key: str, question: str) -> Dict:
        """查询接口 - 需要query:basic权限"""
        user = self.rbac.authenticate(api_key)
        if not user:
            return {'error': '认证失败', 'status': 401}

        if not user.has_permission(Permission.QUERY_BASIC):
            self._log_access(user, 'query', False)
            return {'error': '权限不足', 'status': 403}

        self._log_access(user, 'query', True)
        return {'answer': f"关于「{question}」的回答...", 'status': 200}

    def upload_document(self, api_key: str, title: str) -> Dict:
        """上传文档 - 需要doc:upload权限"""
        user = self.rbac.authenticate(api_key)
        if not user:
            return {'error': '认证失败', 'status': 401}

        if not user.has_permission(Permission.DOC_UPLOAD):
            self._log_access(user, 'upload', False)
            return {'error': f'{user.username}({user.role.value})无上传权限', 'status': 403}

        self._log_access(user, 'upload', True)
        return {'message': f'文档"{title}"已上传', 'status': 200}

    def delete_document(self, api_key: str, doc_id: str) -> Dict:
        """删除文档 - 需要doc:delete权限"""
        user = self.rbac.authenticate(api_key)
        if not user:
            return {'error': '认证失败', 'status': 401}

        if not user.has_permission(Permission.DOC_DELETE):
            self._log_access(user, 'delete', False)
            return {'error': '权限不足', 'status': 403}

        self._log_access(user, 'delete', True)
        return {'message': f'文档{doc_id}已删除', 'status': 200}

    def view_logs(self, api_key: str) -> Dict:
        """查看日志 - 需要log:view权限"""
        user = self.rbac.authenticate(api_key)
        if not user:
            return {'error': '认证失败', 'status': 401}

        if not user.has_permission(Permission.LOG_VIEW):
            self._log_access(user, 'view_logs', False)
            return {'error': '权限不足', 'status': 403}

        self._log_access(user, 'view_logs', True)
        return {'logs': self.access_log[-20:], 'status': 200}


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("W33-D1 RBAC权限控制")
    print("=" * 60)

    rbac = RBACManager()
    api = SecureAPI(rbac)

    # --- 角色权限矩阵 ---
    print("\n--- 角色权限矩阵 ---")
    print(f"{'权限':<20} {'ADMIN':<8} {'EDITOR':<8} {'VIEWER':<8} {'GUEST':<8}")
    print("-" * 52)
    for perm in Permission:
        row = f"{perm.value:<20}"
        for role in Role:
            has = "Y" if perm in ROLE_PERMISSIONS[role] else "-"
            row += f"{has:<8}"
        print(row)

    # --- 获取API Keys ---
    keys = {}
    for uid, user in rbac.users.items():
        for key, u in rbac.api_keys.items():
            if u == uid:
                keys[user.username] = key
                break

    # --- 测试各角色权限 ---
    print(f"\n{'='*60}")
    print("--- 权限测试 ---")
    print(f"{'='*60}")

    test_cases = [
        ("admin", "query", lambda k: api.query(k, "什么是RAG?")),
        ("admin", "upload", lambda k: api.upload_document(k, "测试文档")),
        ("admin", "delete", lambda k: api.delete_document(k, "doc_001")),
        ("viewer", "query", lambda k: api.query(k, "什么是RAG?")),
        ("viewer", "upload", lambda k: api.upload_document(k, "测试文档")),
        ("viewer", "delete", lambda k: api.delete_document(k, "doc_001")),
        ("guest", "query", lambda k: api.query(k, "什么是RAG?")),
        ("guest", "upload", lambda k: api.upload_document(k, "测试文档")),
    ]

    for username, action, func in test_cases:
        key = keys.get(username, "")
        result = func(key)
        status = "成功" if result.get('status') == 200 else f"失败({result.get('error', '')})"
        print(f"  {username:<8} {action:<10} -> {status}")

    # --- 查看访问日志 ---
    print(f"\n{'='*60}")
    print("--- 访问日志 ---")
    print(f"{'='*60}")
    admin_key = keys.get("admin", "")
    log_result = api.view_logs(admin_key)
    if log_result.get('status') == 200:
        for entry in log_result['logs'][-8:]:
            status = "通过" if entry['success'] else "拒绝"
            print(f"  [{entry['user']:<8}] {entry['action']:<10} -> {status}")

    print("\n完成!")
