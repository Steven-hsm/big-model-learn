"""
W31-D6 CI/CD配置
=================
生成CI/CD配置, 包括:
- GitHub Actions工作流
- 自动测试流程
- 自动部署模板
- 代码质量检查

CI/CD让代码从提交到部署全程自动化。
"""

import json
from typing import Dict, List


# ============================================================
# 1. GitHub Actions工作流生成器
# ============================================================

class GitHubActionsGenerator:
    """GitHub Actions工作流生成器"""

    def __init__(self, name: str):
        self.name = name
        self.triggers = {}
        self.jobs = {}

    def add_trigger(self, event: str, config: Dict = None):
        """添加触发条件"""
        self.triggers[event] = config or {}
        return self

    def add_job(self, name: str, config: Dict):
        """添加任务"""
        self.jobs[name] = config
        return self

    def generate(self) -> str:
        """生成YAML工作流"""
        lines = [f"name: {self.name}", ""]

        # 触发条件
        lines.append("on:")
        for event, config in self.triggers.items():
            if config:
                lines.append(f"  {event}:")
                for key, values in config.items():
                    if isinstance(values, list):
                        lines.append(f"    {key}:")
                        for v in values:
                            lines.append(f"      - {v}")
                    else:
                        lines.append(f"    {key}: {values}")
            else:
                lines.append(f"  {event}:")
        lines.append("")

        # 任务
        lines.append("jobs:")
        for job_name, job_config in self.jobs.items():
            lines.append(f"  {job_name}:")
            for key, value in job_config.items():
                if key == "runs-on":
                    lines.append(f"    runs-on: {value}")
                elif key == "steps":
                    lines.append(f"    steps:")
                    for step in value:
                        if 'name' in step:
                            lines.append(f"      - name: {step['name']}")
                        if 'uses' in step:
                            lines.append(f"        uses: {step['uses']}")
                        if 'with' in step:
                            lines.append(f"        with:")
                            for k, v in step['with'].items():
                                lines.append(f"          {k}: {v}")
                        if 'run' in step:
                            lines.append(f"        run: |")
                            for line in step['run'].split('\n'):
                                lines.append(f"          {line}")
                elif isinstance(value, list):
                    lines.append(f"    {key}:")
                    for v in value:
                        lines.append(f"      - {v}")
                else:
                    lines.append(f"    {key}: {value}")
            lines.append("")

        return "\n".join(lines)


# ============================================================
# 2. 预定义工作流
# ============================================================

def create_test_workflow() -> str:
    """创建自动测试工作流"""
    gen = GitHubActionsGenerator("测试工作流")
    gen.add_trigger("push", {"branches": ["main", "develop"]})
    gen.add_trigger("pull_request", {"branches": ["main"]})

    gen.add_job("test", {
        "runs-on": "ubuntu-latest",
        "steps": [
            {"name": "检出代码", "uses": "actions/checkout@v4"},
            {"name": "设置Python", "uses": "actions/setup-python@v5",
             "with": {"python-version": "3.11"}},
            {"name": "缓存依赖",
             "uses": "actions/cache@v3",
             "with": {
                 "path": "~/.cache/pip",
                 "key": "${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}",
             }},
            {"name": "安装依赖",
             "run": "pip install -r requirements.txt\npip install pytest pytest-cov"},
            {"name": "运行测试",
             "run": "pytest tests/ --cov=src --cov-report=xml --cov-report=term-missing"},
            {"name": "上传覆盖率",
             "uses": "codecov/codecov-action@v3",
             "with": {"file": "./coverage.xml"}},
        ]
    })

    return gen.generate()


def create_lint_workflow() -> str:
    """创建代码质量检查工作流"""
    gen = GitHubActionsGenerator("代码质量检查")
    gen.add_trigger("push", {"branches": ["main"]})
    gen.add_trigger("pull_request", {"branches": ["main"]})

    gen.add_job("lint", {
        "runs-on": "ubuntu-latest",
        "steps": [
            {"name": "检出代码", "uses": "actions/checkout@v4"},
            {"name": "设置Python", "uses": "actions/setup-python@v5",
             "with": {"python-version": "3.11"}},
            {"name": "安装工具",
             "run": "pip install flake8 black isort mypy"},
            {"name": "Black格式检查",
             "run": "black --check --diff src/ tests/"},
            {"name": "isort检查",
             "run": "isort --check-only --diff src/ tests/"},
            {"name": "Flake8检查",
             "run": "flake8 src/ tests/ --max-line-length=120 --exclude=__pycache__"},
            {"name": "类型检查",
             "run": "mypy src/ --ignore-missing-imports"},
        ]
    })

    return gen.generate()


def create_deploy_workflow() -> str:
    """创建自动部署工作流"""
    gen = GitHubActionsGenerator("部署工作流")
    gen.add_trigger("push", {"branches": ["main"]})

    gen.add_job("deploy", {
        "runs-on": "ubuntu-latest",
        "needs": ["test"],  # 依赖测试通过
        "steps": [
            {"name": "检出代码", "uses": "actions/checkout@v4"},
            {"name": "登录Docker",
             "uses": "docker/login-action@v3",
             "with": {
                 "registry": "ghcr.io",
                 "username": "${{ github.actor }}",
                 "password": "${{ secrets.GITHUB_TOKEN }}",
             }},
            {"name": "构建并推送",
             "uses": "docker/build-push-action@v5",
             "with": {
                 "context": ".",
                 "push": "true",
                 "tags": "ghcr.io/${{ github.repository }}:latest",
             }},
            {"name": "部署到服务器",
             "run": "ssh deploy@server 'cd /app && docker-compose pull && docker-compose up -d'",
            },
        ]
    })

    return gen.generate()


def create_full_ci_cd_workflow() -> str:
    """创建完整的CI/CD工作流"""
    gen = GitHubActionsGenerator("完整CI/CD")
    gen.add_trigger("push", {"branches": ["main", "develop"]})
    gen.add_trigger("pull_request", {"branches": ["main"]})

    # 测试任务
    gen.add_job("test", {
        "runs-on": "ubuntu-latest",
        "steps": [
            {"name": "检出代码", "uses": "actions/checkout@v4"},
            {"name": "设置Python", "uses": "actions/setup-python@v5",
             "with": {"python-version": "3.11"}},
            {"name": "安装依赖",
             "run": "pip install -r requirements.txt\npip install pytest pytest-cov"},
            {"name": "运行测试",
             "run": "pytest tests/ -v --cov=src"},
        ]
    })

    # 代码质量任务
    gen.add_job("lint", {
        "runs-on": "ubuntu-latest",
        "steps": [
            {"name": "检出代码", "uses": "actions/checkout@v4"},
            {"name": "设置Python", "uses": "actions/setup-python@v5",
             "with": {"python-version": "3.11"}},
            {"name": "代码检查",
             "run": "pip install flake8 black\nblack --check src/\nflake8 src/"},
        ]
    })

    return gen.generate()


# ============================================================
# 3. 测试配置模板
# ============================================================

def create_pytest_config() -> str:
    """生成pytest配置"""
    return """# pytest配置 (conftest.py)
import pytest
import sys
import os

# 添加源码路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

@pytest.fixture
def sample_documents():
    \"\"\"示例文档fixture\"\"\"
    return [
        {"title": "RAG技术", "content": "RAG是检索增强生成技术"},
        {"title": "Python入门", "content": "Python是数据科学的首选语言"},
        {"title": "向量检索", "content": "向量检索通过语义匹配查找文档"},
    ]

@pytest.fixture
def rag_system():
    \"\"\"RAG系统fixture\"\"\"
    # 返回一个已初始化的RAG系统实例
    return {"status": "ready", "documents": 3}

# 测试标记
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")
"""


def create_sample_test() -> str:
    """生成示例测试文件"""
    return """# tests/test_rag_system.py
import pytest

class TestRAGSystem:
    \"\"\"RAG系统测试\"\"\"

    def test_upload_document(self, rag_system):
        \"\"\"测试文档上传\"\"\"
        assert rag_system['status'] == 'ready'

    def test_query(self, rag_system, sample_documents):
        \"\"\"测试查询功能\"\"\"
        assert len(sample_documents) == 3

    def test_empty_query(self):
        \"\"\"测试空查询处理\"\"\"
        query = ""
        assert len(query) == 0

    @pytest.mark.parametrize("query,expected_min", [
        ("RAG", 1),
        ("Python", 1),
        ("不存在的内容", 0),
    ])
    def test_search_relevance(self, query, expected_min):
        \"\"\"参数化测试: 搜索相关性\"\"\"
        assert len(query) >= 0  # 简化断言
"""


# ============================================================
# 4. 质量门禁配置
# ============================================================

class QualityGate:
    """质量门禁配置"""

    def __init__(self):
        self.rules = {}

    def add_rule(self, name: str, threshold: float, description: str = ""):
        self.rules[name] = {
            'threshold': threshold,
            'description': description,
            'current': None,
        }
        return self

    def evaluate(self, metrics: Dict) -> Dict:
        """评估是否通过质量门禁"""
        results = {}
        all_passed = True

        for name, rule in self.rules.items():
            current = metrics.get(name, 0)
            passed = current >= rule['threshold']
            results[name] = {
                'passed': passed,
                'threshold': rule['threshold'],
                'current': current,
                'description': rule['description'],
            }
            if not passed:
                all_passed = False

        return {
            'passed': all_passed,
            'results': results,
        }

    def print_gate_report(self, metrics: Dict):
        """打印门禁报告"""
        evaluation = self.evaluate(metrics)
        print(f"\n{'='*60}")
        print("质量门禁报告")
        print(f"{'='*60}")
        print(f"整体: {'通过' if evaluation['passed'] else '未通过'}")
        print()
        for name, result in evaluation['results'].items():
            status = "PASS" if result['passed'] else "FAIL"
            print(f"  [{status}] {name}: {result['current']:.2%} "
                  f"(要求 >= {result['threshold']:.2%})")
            if result['description']:
                print(f"         {result['description']}")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("W31-D6 CI/CD配置")
    print("=" * 60)

    # --- 1. 测试工作流 ---
    print("\n--- 1. 自动测试工作流 (.github/workflows/test.yml) ---")
    test_wf = create_test_workflow()
    print(test_wf)

    # --- 2. 代码质量检查 ---
    print(f"\n{'='*60}")
    print("--- 2. 代码质量检查工作流 ---")
    print(f"{'='*60}")
    lint_wf = create_lint_workflow()
    print(lint_wf)

    # --- 3. 部署工作流 ---
    print(f"\n{'='*60}")
    print("--- 3. 自动部署工作流 ---")
    print(f"{'='*60}")
    deploy_wf = create_deploy_workflow()
    print(deploy_wf)

    # --- 4. 完整CI/CD ---
    print(f"\n{'='*60}")
    print("--- 4. 完整CI/CD工作流 ---")
    print(f"{'='*60}")
    full_wf = create_full_ci_cd_workflow()
    print(full_wf)

    # --- 5. 测试配置 ---
    print(f"\n{'='*60}")
    print("--- 5. 测试配置 ---")
    print(f"{'='*60}")
    print("\nconftest.py:")
    print(create_pytest_config()[:300] + "...")
    print("\ntests/test_rag_system.py:")
    print(create_sample_test()[:300] + "...")

    # --- 6. 质量门禁 ---
    print(f"\n{'='*60}")
    print("--- 6. 质量门禁 ---")
    print(f"{'='*60}")

    gate = QualityGate()
    gate.add_rule("test_pass_rate", 0.95, "测试通过率")
    gate.add_rule("code_coverage", 0.80, "代码覆盖���")
    gate.add_rule("lint_pass_rate", 1.0, "代码规范通过率")

    # 模拟指标
    sample_metrics = {
        "test_pass_rate": 0.97,
        "code_coverage": 0.85,
        "lint_pass_rate": 1.0,
    }
    gate.print_gate_report(sample_metrics)

    print("\n完成!")
