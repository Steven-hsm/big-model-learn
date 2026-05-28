"""
W34-D1 代码审查
================
实现代码审查工具, 包括:
- 代码审查清单
- 安全审查要点
- 性能审查指南

代码审查是保障代码质量的重要实践。
"""

import re
import ast
import os
from typing import List, Dict, Tuple
from dataclasses import dataclass, field


# ============================================================
# 1. 代码审查清单
# ============================================================

class CodeReviewChecklist:
    """代码审查清单"""

    CATEGORIES = {
        '功能正确性': [
            '代码是否实现了预期功能?',
            '边界条件是否正确处理?',
            '错误处理是否完善?',
            '是否有明显的逻辑错误?',
            '返回值是否正确?',
        ],
        '代码质量': [
            '命名是否清晰有意义?',
            '函数是否过长(>50行)?',
            '是否有重复代码可以提取?',
            '注释是否充分且准确?',
            '代码结构是否清晰?',
        ],
        '性能': [
            '是否有不必要的循环?',
            '数据结构选择是否合理?',
            '是否有可以缓存的计算?',
            '是否有潜在的内存泄漏?',
            '异步操作是否正确处理?',
        ],
        '安全性': [
            '输入验证是否充分?',
            'SQL注入风险是否存在?',
            '敏感数据是否���密处理?',
            '权限检查是否到位?',
            '日志中是否泄露敏感信息?',
        ],
        '可维护性': [
            '配置是否外置?',
            '是否便于测试?',
            '依赖是否合理?',
            '接口是否向后兼容?',
            '文档是否需要更新?',
        ],
    }

    def print_checklist(self):
        """打印审查清单"""
        print("代码审查清单")
        print("=" * 50)
        for category, items in self.CATEGORIES.items():
            print(f"\n[{category}]")
            for item in items:
                print(f"  [ ] {item}")


# ============================================================
# 2. 静态代码分析器
# ============================================================

@dataclass
class CodeIssue:
    """代码问题"""
    file: str
    line: int
    severity: str    # info, warning, error
    category: str    # security, performance, style, bug
    message: str
    suggestion: str = ""


class StaticAnalyzer:
    """静态代码分析器(简化版)"""

    def __init__(self):
        self.issues: List[CodeIssue] = []

    def analyze_code(self, code: str, filename: str = "unknown.py") -> List[CodeIssue]:
        """分析代码"""
        self.issues = []
        lines = code.split('\n')

        for i, line in enumerate(lines, 1):
            self._check_security(line, i, filename)
            self._check_performance(line, i, filename)
            self._check_style(line, i, filename)
            self._check_bugs(line, i, filename)

        return self.issues

    def _check_security(self, line: str, lineno: int, filename: str):
        """安全检查"""
        # 检查硬编码密码
        if re.search(r'(password|secret|api_key|token)\s*=\s*["\'][^"\']+["\']',
                      line, re.IGNORECASE):
            self.issues.append(CodeIssue(
                filename, lineno, 'error', 'security',
                '检测到硬编码的密钥或密码',
                '使用环境变量或配置文件管理敏感信息'
            ))

        # 检查eval使用
        if 'eval(' in line:
            self.issues.append(CodeIssue(
                filename, lineno, 'error', 'security',
                '使用了eval(), 存在代码注入风险',
                '使用ast.literal_eval()替代'
            ))

        # 检查SQL拼接
        if re.search(r'(SELECT|INSERT|UPDATE|DELETE).*%s.*%', line, re.IGNORECASE):
            if '+' in line or 'format' in line:
                self.issues.append(CodeIssue(
                    filename, lineno, 'warning', 'security',
                    '可能存在SQL注入风险',
                    '使用参数化查询'
                ))

        # 检查异常捕获过于宽泛
        if 'except:' in line and 'Exception' not in line:
            self.issues.append(CodeIssue(
                filename, lineno, 'warning', 'security',
                '使用了裸except, 可能隐藏真正的错误',
                '捕获具体的异常类型'
            ))

    def _check_performance(self, line: str, lineno: int, filename: str):
        """性能检查"""
        # 检查循环中的重复计算
        if 'for ' in line and 'append' in line:
            pass  # 正常模式, 不报告

        # 检查列表推导使用
        if re.search(r'for\s+\w+\s+in\s+range.*\.append', line):
            self.issues.append(CodeIssue(
                filename, lineno, 'info', 'performance',
                '考虑使用列表推导替代循环append',
                '使用 [expr for x in range(n)]'
            ))

        # 检查sleep
        if 'time.sleep(' in line:
            self.issues.append(CodeIssue(
                filename, lineno, 'info', 'performance',
                '使用了time.sleep(), 可能影响性能',
                '在异步环境中使用asyncio.sleep()'
            ))

    def _check_style(self, line: str, lineno: int, filename: str):
        """风格检查"""
        # 行长度
        if len(line) > 120:
            self.issues.append(CodeIssue(
                filename, lineno, 'info', 'style',
                f'行长度{len(line)}超过120字符',
                '拆分为多行'
            ))

        # TODO/FIXME
        if re.search(r'#\s*(TODO|FIXME|HACK|XXX)', line, re.IGNORECASE):
            self.issues.append(CodeIssue(
                filename, lineno, 'info', 'style',
                '存在TODO/FIXME标记',
                '跟踪并解决遗留问题'
            ))

    def _check_bugs(self, line: str, lineno: int, filename: str):
        """Bug检查"""
        # 可变默认参数
        if re.search(r'def\s+\w+\(.*=\[\]', line):
            self.issues.append(CodeIssue(
                filename, lineno, 'warning', 'bug',
                '使用了可变默认参数(如[])',
                '使用None作为默认值, 在函数内初始化'
            ))

        if re.search(r'def\s+\w+\(.*=\{\}', line):
            self.issues.append(CodeIssue(
                filename, lineno, 'warning', 'bug',
                '使用了可变默认参数(如{})',
                '使用None作为默认值, 在函数内初始化'
            ))

        # == 比较None
        if '== None' in line:
            self.issues.append(CodeIssue(
                filename, lineno, 'info', 'style',
                '使用==比较None',
                '使用 is None 替代'
            ))

    def print_report(self):
        """打印分析报告"""
        if not self.issues:
            print("未发现问题!")
            return

        # 按严重度分组
        by_severity = {}
        for issue in self.issues:
            by_severity.setdefault(issue.severity, []).append(issue)

        print(f"\n{'='*60}")
        print(f"代码审查报告 ({len(self.issues)}个问题)")
        print(f"{'='*60}")

        for severity in ['error', 'warning', 'info']:
            issues = by_severity.get(severity, [])
            if issues:
                icon = {'error': 'x', 'warning': '!', 'info': 'i'}[severity]
                print(f"\n[{icon}] {severity.upper()} ({len(issues)}个)")
                for issue in issues:
                    print(f"  L{issue.line}: [{issue.category}] {issue.message}")
                    if issue.suggestion:
                        print(f"       建议: {issue.suggestion}")


# ============================================================
# 3. 安全审查
# ============================================================

class SecurityReviewer:
    """安全审查器"""

    SECURITY_CHECKLIST = {
        '输入验证': [
            '所有外部输入是否经过验证?',
            '文件上传是否有类型和大小限制?',
            'API参数是否有范围检查?',
        ],
        '认证与授权': [
            'API是否有认证机制?',
            '敏感操作是否有权限检查?',
            '密码是否使用bcrypt/scrypt存储?',
        ],
        '数据保护': [
            '敏感数据是否加密存储?',
            'HTTPS是否强制使用?',
            '日志是否过滤了敏感信息?',
        ],
        '依赖安全': [
            '依赖库是否有已知漏洞?',
            '是否使用了过时的加密算法?',
            '第三方库版本是否锁定?',
        ],
    }

    def print_checklist(self):
        print("\n安全审查清单")
        print("=" * 50)
        for category, items in self.SECURITY_CHECKLIST.items():
            print(f"\n[{category}]")
            for item in items:
                print(f"  [ ] {item}")

    def scan_code(self, code: str, filename: str = "") -> List[CodeIssue]:
        """扫描代码安全问题"""
        issues = []
        lines = code.split('\n')

        security_patterns = [
            (r'eval\s*\(', '使用了eval(), 存在代码注入风险', 'error'),
            (r'exec\s*\(', '使用了exec(), 存在代码注入风险', 'error'),
            (r'__import__\s*\(', '动态导入可能存在风险', 'warning'),
            (r'subprocess.*shell\s*=\s*True', 'shell=True存在命令注入风险', 'error'),
            (r'pickle\.loads?\s*\(', 'pickle反序列化不安全', 'error'),
            (r'yaml\.load\s*\([^)]*\)(?!.*Loader)', 'yaml.load不安全, 使用yaml.safe_load', 'warning'),
            (r'assert\s+', 'assert在生产环境可能被禁用', 'warning'),
            (r'random\.\w+', 'random模块不安全, 使用secrets', 'info'),
        ]

        for i, line in enumerate(lines, 1):
            for pattern, message, severity in security_patterns:
                if re.search(pattern, line):
                    issues.append(CodeIssue(
                        filename, i, severity, 'security', message
                    ))

        return issues


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("W34-D1 代码审查")
    print("=" * 60)

    # --- 1. 审查清单 ---
    print("\n--- 1. 代码审查清单 ---")
    checklist = CodeReviewChecklist()
    checklist.print_checklist()

    # --- 2. 静态分析 ---
    print(f"\n{'='*60}")
    print("--- 2. 静态代码分析 ---")
    print(f"{'='*60}")

    sample_code = '''
import time

# TODO: 优化这个函数
def process_data(data=[]):
    password = "my_secret_password"
    api_key = "sk-1234567890"

    result = []
    for item in data:
        if item == None:
            continue
        try:
            processed = eval(item)
            result.append(processed)
        except:
            pass

    time.sleep(5)
    return result

def very_long_function_name_that_exceeds_the_line_limit_and_should_be_refactored_to_be_shorter(x, y, z, a, b, c):
    return x + y + z + a + b + c
'''

    analyzer = StaticAnalyzer()
    analyzer.analyze_code(sample_code, "sample.py")
    analyzer.print_report()

    # --- 3. 安全审查 ---
    print(f"\n{'='*60}")
    print("--- 3. 安全审查 ---")
    print(f"{'='*60}")

    reviewer = SecurityReviewer()
    reviewer.print_checklist()

    print("\n--- 安全扫描结果 ---")
    sec_issues = reviewer.scan_code(sample_code, "sample.py")
    for issue in sec_issues:
        print(f"  [{issue.severity}] L{issue.line}: {issue.message}")

    print("\n完成!")
