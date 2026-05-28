"""
W33-D7 发布准备
================
准备项目发布, 包括:
- 版本管理
- 变更日志生成
- 部署文档

规范的发布流程保障项目质量。
"""

import time
import json
from typing import List, Dict
from dataclasses import dataclass, field


# ============================================================
# 1. 版本管理
# ============================================================

@dataclass
class Version:
    """语义化版本号"""
    major: int = 1
    minor: int = 0
    patch: int = 0
    prerelease: str = ""  # alpha, beta, rc.1 等

    def __str__(self):
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        return version

    def bump_major(self):
        return Version(self.major + 1, 0, 0)

    def bump_minor(self):
        return Version(self.major, self.minor + 1, 0)

    def bump_patch(self):
        return Version(self.major, self.minor, self.patch + 1)

    @staticmethod
    def parse(version_str: str) -> 'Version':
        parts = version_str.split('-', 1)
        version_parts = parts[0].split('.')
        major = int(version_parts[0])
        minor = int(version_parts[1]) if len(version_parts) > 1 else 0
        patch = int(version_parts[2]) if len(version_parts) > 2 else 0
        prerelease = parts[1] if len(parts) > 1 else ""
        return Version(major, minor, patch, prerelease)


class VersionManager:
    """版本管理器"""

    def __init__(self, current_version: str = "1.0.0"):
        self.current = Version.parse(current_version)
        self.history: List[Dict] = []

    def release(self, version_type: str = "patch",
                 changes: List[str] = None, author: str = "") -> Version:
        """发布新版本"""
        old_version = str(self.current)

        if version_type == "major":
            self.current = self.current.bump_major()
        elif version_type == "minor":
            self.current = self.current.bump_minor()
        else:
            self.current = self.current.bump_patch()

        release_info = {
            'version': str(self.current),
            'previous_version': old_version,
            'type': version_type,
            'changes': changes or [],
            'author': author,
            'timestamp': time.time(),
            'date': time.strftime('%Y-%m-%d'),
        }
        self.history.append(release_info)

        return self.current

    def get_history(self) -> List[Dict]:
        return self.history


# ============================================================
# 2. 变更日志生成器
# ============================================================

@dataclass
class Change:
    """变更条目"""
    type: str       # added, changed, fixed, removed, deprecated
    description: str
    component: str = ""
    issue: str = ""


class ChangelogGenerator:
    """变更日志生成器"""

    def __init__(self):
        self.changes: List[Change] = []

    def add_change(self, change_type: str, description: str,
                    component: str = "", issue: str = ""):
        self.changes.append(Change(change_type, description, component, issue))

    def generate(self, version: str = "", date: str = "") -> str:
        """生成变更日志"""
        if not date:
            date = time.strftime('%Y-%m-%d')

        lines = []
        if version:
            lines.append(f"## [{version}] - {date}")
        else:
            lines.append(f"## {date}")
        lines.append("")

        # 按类型分组
        type_labels = {
            'added': '新增',
            'changed': '变更',
            'fixed': '修复',
            'removed': '移除',
            'deprecated': '弃用',
        }
        type_icons = {
            'added': '+', 'changed': '~', 'fixed': '!',
            'removed': '-', 'deprecated': 'x',
        }

        grouped = {}
        for change in self.changes:
            grouped.setdefault(change.type, []).append(change)

        for change_type, label in type_labels.items():
            if change_type in grouped:
                lines.append(f"### {label}")
                for change in grouped[change_type]:
                    line = f"- {change.description}"
                    if change.component:
                        line = f"- [{change.component}] {change.description}"
                    if change.issue:
                        line += f" (#{change.issue})"
                    lines.append(line)
                lines.append("")

        return "\n".join(lines)


# ============================================================
# 3. 部署文档生成器
# ============================================================

class DeploymentDocGenerator:
    """部署文档生成器"""

    def __init__(self):
        self.project_name = "RAG知识库问答系统"
        self.version = "1.0.0"

    def generate(self) -> str:
        """生成部署文档"""
        return f"""
# {self.project_name} 部署文档 v{self.version}

## 环境要求

### 硬件要求
| 组件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| CPU | 2核 | 4核+ |
| 内存 | 4GB | 8GB+ |
| 磁盘 | 20GB | 100GB+ SSD |
| 网络 | 10Mbps | 100Mbps+ |

### 软件要求
- Python 3.11+
- Docker 24.0+
- Docker Compose v2.0+
- PostgreSQL 15+
- Redis 7+

## 部署步骤

### 方式一: Docker部署(推荐)

```bash
# 1. 克隆项目
git clone https://github.com/your-org/rag-system.git
cd rag-system

# 2. 配置环境变量
cp .env.example .env
# 编辑.env, 填写必要配置

# 3. 启动所有服务
docker-compose up -d

# 4. 验证服务
curl http://localhost:8000/api/health
```

### 方式二: 手动部署

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 初始化数据库
python manage.py init-db

# 3. 启动服务
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## 环境变量说明

| 变量 | 说明 | 默认值 | 必填 |
|------|------|--------|------|
| APP_HOST | 监听地址 | 0.0.0.0 | 否 |
| APP_PORT | 监听端口 | 8000 | 否 |
| OPENAI_API_KEY | OpenAI API密钥 | - | 是 |
| DATABASE_URL | 数据库连接 | - | 是 |
| REDIS_URL | Redis连接 | redis://localhost:6379/0 | 否 |
| LOG_LEVEL | 日志级别 | INFO | 否 |

## 健康检查

```bash
# API健康检查
curl http://localhost:8000/api/health

# 预期响应
{{"status": "healthy", "version": "{self.version}"}}
```

## 常见问题

### Q: 启动失败, 提示端口被占用
A: 修改.env中的APP_PORT, 或停止占用8000端口的进程

### Q: 向量检索超时
A: 检查向量数据库连接, 增加超时时间

### Q: 内存不足
A: 减小batch_size, 增加服务器内存

## 监控

访问 http://localhost:8000/api/stats 查看系统状态。

## 回滚

```bash
# 回滚到上一版本
docker-compose down
git checkout v0.9.0
docker-compose up -d
```
"""


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("W33-D7 发布准备")
    print("=" * 60)

    # --- 1. 版本管理 ---
    print("\n--- 1. 版本管理 ---")
    vm = VersionManager("0.9.0")
    print(f"当前版本: {vm.current}")

    # 发布历史
    releases = [
        ("patch", ["修复查询结果为空的问题"], "dev1"),
        ("patch", ["优化缓存策略, 提升命中率"], "dev2"),
        ("minor", ["新增流式输出功能", "新增文档批量上传"], "dev1"),
        ("patch", ["修复并发查询的线程安全问题"], "dev3"),
        ("minor", ["新增重排序模块", "支持HyDE检索", "新增多租户功能"], "dev1"),
        ("major", ["全新架构重构", "API v2版本", "不兼容v1接口"], "dev1"),
    ]

    for version_type, changes, author in releases:
        new_version = vm.release(version_type, changes, author)
        print(f"  发布 v{new_version} ({version_type}): {changes[0]}")

    # --- 2. 变更日志 ---
    print(f"\n{'='*60}")
    print("--- 2. 变更日志 ---")
    print(f"{'='*60}")

    changelog = ChangelogGenerator()
    changelog.add_change('added', '实现BM25+向量混合检索', '检索模块', '001')
    changelog.add_change('added', '支持流式SSE输出', 'API模块', '005')
    changelog.add_change('added', '新增Gradio Web界面', '前端', '008')
    changelog.add_change('changed', '优化检索延迟, P95从800ms降至300ms', '检索模块', '010')
    changelog.add_change('changed', '升级embedding模型至BGE-large', '嵌入模块')
    changelog.add_change('fixed', '修复长文档截断后丢失上下文的问题', '分块模块', '003')
    changelog.add_change('fixed', '修复并发场景下的缓存竞争问题', '缓存模块', '007')
    changelog.add_change('removed', '移除已弃用的v1检索接口', 'API模块')

    log_content = changelog.generate("1.0.0", time.strftime('%Y-%m-%d'))
    print(log_content)

    # --- 3. 部署文档 ---
    print(f"{'='*60}")
    print("--- 3. 部署文档 ---")
    print(f"{'='*60}")

    deploy_doc = DeploymentDocGenerator()
    doc_content = deploy_doc.generate()
    print(doc_content[:1000])
    print(f"... (共{len(doc_content)}字符)")

    print("\n完成!")
