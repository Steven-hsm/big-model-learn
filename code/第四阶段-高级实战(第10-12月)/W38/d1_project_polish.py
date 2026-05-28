"""
W38 Day 1 - 项目打磨
====================
主题: 项目打磨检查清单, 代码质量分析脚本, 性能优化建议
"""

import os
import ast
import re
from datetime import datetime


def print_section(title):
    """打印分隔线"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# =============================================
# 1. 项目打磨检查清单
# =============================================
def project_polish_checklist():
    """项目打磨检查清单"""
    print_section("项目打磨检查清单")

    checklist = {
        "README.md检查": [
            "[ ] 项目名称和一句话描述",
            "[ ] 功能特点(至少3点)",
            "[ ] 安装说明(可复制粘贴的命令)",
            "[ ] 快速开始示例(可运行的代码)",
            "[ ] 项目截图或GIF演示",
            "[ ] 技术栈说明",
            "[ ] 项目架构图",
            "[ ] 贡献指南",
            "[ ] 许可证",
            "[ ] 联系方式"
        ],
        "代码质量检查": [
            "[ ] 所有代码有中文/英文注释",
            "[ ] 函数有docstring(参数、返回值说明)",
            "[ ] 变量命名有意义(不使用a, b, c)",
            "[ ] 没有硬编码的配置(使用配置文件/环境变量)",
            "[ ] 错误处理完善(try/except)",
            "[ ] 没有未使用的import和变量",
            "[ ] 代码格式一致(PEP8)",
            "[ ] 类型注解(type hints)"
        ],
        "测试检查": [
            "[ ] 有单元测试",
            "[ ] 核心功能测试覆盖",
            "[ ] 边界情况测试",
            "[ ] 测试可以独立运行",
            "[ ] CI/CD流水线配置"
        ],
        "文档检查": [
            "[ ] API文档(如果有)",
            "[ ] 使用教程",
            "[ ] 配置说明",
            "[ ] 常见问题FAQ",
            "[ ] CHANGELOG.md"
        ],
        "项目配置检查": [
            "[ ] requirements.txt 或 pyproject.toml",
            "[ ] .gitignore 文件完整",
            "[ ] .env.example (不提交真实密钥)",
            "[ ] Docker支持(可选)",
            "[ ] GitHub Actions / CI配置"
        ],
        "演示检查": [
            "[ ] 在线Demo可访问(如Hugging Face Spaces)",
            "[ ] 示例数据包含",
            "[ ] 运行结果截图",
            "[ ] 性能基准数据"
        ]
    }

    for category, items in checklist.items():
        print(f"\n【{category}】")
        for item in items:
            print(f"  {item}")

    # 计算项目健康度
    total = sum(len(items) for items in checklist.values())
    print(f"\n  总检查项: {total}项")
    print(f"  目标: 至少完成80%的检查项")


# =============================================
# 2. 代码质量分析脚本
# =============================================
def analyze_code_quality():
    """代码质量分析脚本"""
    print_section("代码质量分析")

    # 简单的代码质量分析器
    class CodeAnalyzer:
        """简单的Python代码质量分析器"""

        def __init__(self):
            self.results = {}

        def analyze_file(self, filepath):
            """分析单个Python文件"""
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()

                result = {
                    "文件": os.path.basename(filepath),
                    "总行数": len(content.split('\n')),
                    "代码行数": 0,
                    "注释行数": 0,
                    "空行数": 0,
                    "函数数": 0,
                    "类数": 0,
                    "有docstring的函数": 0,
                    "平均函数长度": 0,
                    "最长行": 0,
                    "TODO数": 0,
                    "问题": []
                }

                # 逐行分析
                in_docstring = False
                for line in content.split('\n'):
                    stripped = line.strip()

                    if not stripped:
                        result["空行数"] += 1
                    elif stripped.startswith('#'):
                        result["注释行数"] += 1
                    elif '"""' in stripped or "'''" in stripped:
                        result["注释行数"] += 1
                        in_docstring = not in_docstring
                    elif in_docstring:
                        result["注释行数"] += 1
                    else:
                        result["代码行数"] += 1

                    # 最长行
                    if len(line) > result["最长行"]:
                        result["最长行"] = len(line)

                    # TODO检查
                    if 'TODO' in stripped or 'FIXME' in stripped:
                        result["TODO数"] += 1

                # AST分析
                try:
                    tree = ast.parse(content)
                    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
                    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]

                    result["函数数"] = len(functions)
                    result["类数"] = len(classes)

                    # 检查docstring
                    for func in functions:
                        if (func.body and isinstance(func.body[0], ast.Expr)
                                and isinstance(func.body[0].value, ast.Constant)):
                            result["有docstring的函数"] += 1

                    # 平均函数长度
                    if functions:
                        func_lengths = []
                        for func in functions:
                            start = func.lineno
                            end = func.end_lineno if hasattr(func, 'end_lineno') else start
                            func_lengths.append(end - start + 1)
                        result["平均函数长度"] = round(
                            sum(func_lengths) / len(func_lengths), 1
                        )
                except SyntaxError:
                    result["问题"].append("语法错误, 无法进行AST分析")

                # 质量检查
                if result["最长行"] > 120:
                    result["问题"].append(f"存在超长行({result['最长行']}字符)")
                if result["平均函数长度"] > 50:
                    result["问题"].append(f"函数平均长度过长({result['平均函数长度']}行)")
                if result["函数数"] > 0 and result["有docstring的函数"] < result["函数数"] * 0.5:
                    result["问题"].append("超过50%的函数缺少docstring")

                return result

            except Exception as e:
                return {"文件": os.path.basename(filepath), "错误": str(e)}

        def print_report(self, results):
            """打印分析报告"""
            print(f"\n  {'文件名':<30s} {'行数':>6s} {'函数':>6s} "
                  f"{'类':>4s} {'注释率':>8s} {'问题数':>6s}")
            print("  " + "-" * 70)

            for r in results:
                if "错误" in r:
                    print(f"  {r['文件']:<30s} {'ERROR':>6s}")
                    continue

                total_lines = r["代码行数"] + r["注释行数"]
                comment_rate = f"{r['注释行数']/total_lines*100:.1f}%" if total_lines > 0 else "0%"
                problems = len(r.get("问题", []))

                print(f"  {r['文件']:<30s} {r['总行数']:>6d} {r['函数数']:>6d} "
                      f"{r['类数']:>4d} {comment_rate:>8s} {problems:>6d}")

                if r.get("问题"):
                    for p in r["问题"]:
                        print(f"    - {p}")

    # 分析当前目录的Python文件
    analyzer = CodeAnalyzer()
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(current_dir)

    # 查找一些Python文件来分析
    sample_files = []
    for root, dirs, files in os.walk(parent_dir):
        for f in files:
            if f.endswith('.py') and not f.startswith('__'):
                sample_files.append(os.path.join(root, f))
                if len(sample_files) >= 5:
                    break
        if len(sample_files) >= 5:
            break

    if sample_files:
        print("\n  分析示例文件(最多5个):")
        results = [analyzer.analyze_file(f) for f in sample_files[:5]]
        analyzer.print_report(results)
    else:
        print("  未找到Python文件进行分析")

    print("\n--- 代码质量建议 ---")
    suggestions = [
        "注释率目标: 15-30%",
        "函数长度: 建议不超过30行",
        "行宽: 建议不超过120字符",
        "所有公共函数应有docstring",
        "使用类型注解提高可读性"
    ]
    for s in suggestions:
        print(f"  {s}")


# =============================================
# 3. 性能优化建议
# =============================================
def performance_optimization_tips():
    """性能优化建议"""
    print_section("性能优化建议")

    tips = {
        "Python性能优化": [
            "1. 使用列表推导式替代循环: [x*2 for x in data]",
            "2. 使用生成器处理大数据: (x*2 for x in data)",
            "3. 使用集合代替列表做查找: O(1) vs O(n)",
            "4. 使用 collections.defaultdict 简化代码",
            "5. 字符串拼接用 join() 代替 +",
            "6. 使用 multiprocessing 替代 threading(GIL限制)",
            "7. 使用 cProfile 定位性能瓶颈",
            "8. 热点代码考虑 Cython 或 numba 加速"
        ],
        "AI模型性能优化": [
            "1. 批处理(Batching): 合并请求提高GPU利用率",
            "2. 量化(Quantization): FP16 -> INT8 -> INT4",
            "3. 剪枝(Pruning): 移除不重要的权重",
            "4. 知识蒸馏(Distillation): 大模型教小模型",
            "5. KV Cache: 缓存已计算的Key-Value",
            "6. ONNX Runtime: 跨平台推理优化",
            "7. TensorRT: NVIDIA GPU推理加速",
            "8. vLLM: 高吞吐量LLM推理引擎"
        ],
        "数据处理优化": [
            "1. 使用 Pandas 向量化操作代替循环",
            "2. 大文件使用分块读取: pd.read_csv(chunksize=10000)",
            "3. 使用 Polars 替代 Pandas(更快的DataFrame库)",
            "4. 并行处理: multiprocessing / joblib",
            "5. 缓存计算结果: functools.lru_cache",
            "6. 使用数据库索引加速查询"
        ],
        "Web应用优化": [
            "1. 异步处理: FastAPI + async/await",
            "2. 缓存: Redis缓存热点数据",
            "3. 连接池: 数据库和HTTP连接复用",
            "4. CDN: 静态资源分发",
            "5. 负载均衡: Nginx反向代理",
            "6. 水平扩展: Docker + Kubernetes"
        ]
    }

    for category, items in tips.items():
        print(f"\n【{category}】")
        for item in items:
            print(f"  {item}")

    # 性能分析工具推荐
    print("\n--- 性能分析工具 ---")
    tools = [
        "cProfile: Python内置性能分析器",
        "line_profiler: 逐行性能分析",
        "memory_profiler: 内存使用分析",
        "py-spy: 采样分析器(无需修改代码)",
        "torch.profiler: PyTorch模型性能分析",
        "NVIDIA Nsight: GPU性能分析"
    ]
    for tool in tools:
        print(f"  {tool}")


# =============================================
# 主程序
# =============================================
if __name__ == "__main__":
    print("=" * 60)
    print("  W38 Day 1 - 项目打磨")
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    project_polish_checklist()
    analyze_code_quality()
    performance_optimization_tips()

    print("\n" + "=" * 60)
    print("  好的项目需要打磨, 细节决定成败!")
    print("  建议: 今天花2小时按检查清单审查你的重点项目")
    print("=" * 60)
