"""
W35 Day 2 - 阅读源码方法
========================
主题: 阅读源码方法, 调试开源项目, 代码导航技巧, 实例分析(transformers库结构)
"""

import os
import json
from datetime import datetime


def print_section(title):
    """打印分隔线"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# =============================================
# 1. 源码阅读方法论
# =============================================
def demonstrate_code_reading_methods():
    """展示源码阅读方法"""
    print_section("源码阅读方法论")

    methods = {
        "自顶向下法(Top-Down)": {
            "步骤": [
                "1. 阅读README和文档, 理解项目目标",
                "2. 查看项目目录结构, 了解模块划分",
                "3. 找到入口文件(main.py, __init__.py)",
                "4. 追踪核心调用链, 逐层深入",
                "5. 关注接口定义, 理解抽象层"
            ],
            "适用场景": "快速了解项目整体架构",
            "优点": "能快速建立全局视图",
            "缺点": "可能忽略底层细节"
        },
        "自底向上法(Bottom-Up)": {
            "步骤": [
                "1. 从具体功能入手, 找到感兴趣的函数",
                "2. 阅读函数实现, 理解每行代码",
                "3. 追溯调用者, 理解使用场景",
                "4. 了解模块间依赖关系",
                "5. 逐步扩展到整个项目"
            ],
            "适用场景": "修复Bug或理解特定功能",
            "优点": "深入理解具体实现",
            "缺点": "可能缺乏全局视角"
        },
        "问题驱动法": {
            "步骤": [
                "1. 带着具体问题阅读源码",
                "2. 使用搜索定位相关代码",
                "3. 设置断点调试运行流程",
                "4. 修改代码验证假设",
                "5. 记录理解, 写学习笔记"
            ],
            "适用场景": "解决具体问题或学习特定功能",
            "优点": "目标明确, 效率高",
            "缺点": "可能忽略其他重要部分"
        }
    }

    for method, details in methods.items():
        print(f"\n【{method}】")
        print(f"  适用场景: {details['适用场景']}")
        print(f"  优点: {details['优点']}")
        print(f"  缺点: {details['缺点']}")
        print("  步骤:")
        for step in details["步骤"]:
            print(f"    {step}")


# =============================================
# 2. 代码导航技巧
# =============================================
def demonstrate_navigation_techniques():
    """代码导航技巧"""
    print_section("代码导航技巧")

    tools_and_shortcuts = {
        "VS Code导航": {
            "跳转到定义": "F12 或 Ctrl+Click",
            "查看引用": "Shift+F12",
            "查看类型定义": "右键 -> Go to Type Definition",
            "查找所有引用": "Shift+Alt+F12",
            "返回上一位置": "Alt+←",
            "前进到下一位置": "Alt+→",
            "全局搜索": "Ctrl+Shift+F",
            "文件搜索": "Ctrl+P",
            "符号搜索": "Ctrl+Shift+O",
            "大纲视图": "View -> Open View -> Outline"
        },
        "PyCharm导航": {
            "跳转到定义": "Ctrl+B 或 Ctrl+Click",
            "查看实现": "Ctrl+Alt+B",
            "查找用法": "Alt+F7",
            "文件结构": "Ctrl+F12",
            "类型层次": "Ctrl+H",
            "最近文件": "Ctrl+E",
            "全局搜索": "Ctrl+Shift+F"
        },
        "命令行工具": {
            "grep / ripgrep": "rg 'def function_name' --type py",
            "ctags": "生成标签文件, 支持跳转",
            "tree": "tree -d -L 2  # 查看目录结构",
            "grep -r": "grep -rn 'class_name' --include='*.py'"
        },
        "GitHub在线导航": {
            "文件搜索": "按 T 键激活",
            "代码搜索": "GitHub搜索栏",
            "跳转到行": "按 L 键输入行号",
            "查看Blame": "按 B 键",
            "在线编辑": "按 . 键打开Web编辑器"
        }
    }

    for tool, shortcuts in tools_and_shortcuts.items():
        print(f"\n【{tool}】")
        for action, shortcut in shortcuts.items():
            print(f"  {action}: {shortcut}")


# =============================================
# 3. 调试开源项目
# =============================================
def demonstrate_debugging_techniques():
    """调试开源项目技巧"""
    print_section("调试开源项目技巧")

    debugging_approaches = {
        "环境搭建": [
            "1. Fork & Clone项目到本地",
            "2. 创建虚拟环境: python -m venv venv",
            "3. 安装开发依赖: pip install -e '.[dev]'",
            "4. 运行测试确认环境正常: pytest",
            "5. 配置IDE调试器"
        ],
        "调试方法": [
            "断点调试 - 在IDE中设置断点, 逐步执行",
            "日志调试 - 添加print/logging语句",
            "单元测试 - 编写最小复现测试",
            "异常追踪 - 阅读完整的traceback",
            "Git Bisect - 二分查找引入Bug的提交"
        ],
        "常用调试工具": [
            "pdb/ipdb - Python内置调试器",
            "breakpoint() - Python 3.7+推荐方式",
            "VS Code Debugger - 图形化调试",
            "PyCharm Debugger - 功能强大的调试器",
            "strace/dtrace - 系统调用追踪"
        ]
    }

    for category, items in debugging_approaches.items():
        print(f"\n【{category}】")
        for item in items:
            print(f"  {item}")

    # Python调试代码示例
    print("\n--- Python调试代码示例 ---")
    debug_code = '''
import pdb

def debug_transformers():
    """调试transformers库的示例"""
    # 方法1: 使用breakpoint()
    from transformers import AutoModel
    model = AutoModel.from_pretrained("bert-base-uncased")
    breakpoint()  # 在此处暂停, 可以交互式检查变量

    # 方法2: 使用pdb
    import pdb
    pdb.set_trace()

    # 方法3: 使用logging
    import logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.debug("调试信息: model loaded successfully")

    # 方法4: 使用上下文管理器计时
    import time
    class Timer:
        def __enter__(self):
            self.start = time.time()
            return self
        def __exit__(self, *args):
            print(f"耗时: {time.time() - self.start:.4f}秒")
    '''
    print(debug_code)


# =============================================
# 4. 实例分析: Transformers库结构
# =============================================
def analyze_transformers_structure():
    """分析Hugging Face Transformers库结构"""
    print_section("实例分析: Transformers库结构")

    structure = {
        "src/transformers/": {
            "__init__.py": "包初始化, 导出公共API",
            "models/": {
                "描述": "所有模型的实现, 按模型类型分目录",
                "bert/": {
                    "modeling_bert.py": "BertModel核心实现",
                    "tokenization_bert.py": "Bert分词器",
                    "configuration_bert.py": "Bert配置类",
                    "__init__.py": "导出Bert相关类"
                },
                "gpt2/": "GPT-2模型实现",
                "llama/": "LLaMA模型实现",
                "t5/": "T5模型实现",
            },
            "pipelines/": "高层推理管道(text-classification等)",
            "trainer.py": "训练器, 封装训练循环",
            "training_args.py": "训练参数配置",
            "configuration_utils.py": "配置基类",
            "modeling_utils.py": "模型基类(PreTrainedModel)",
            "tokenization_utils.py": "分词器基类",
            "utils/": "工具函数(logging, hub, etc.)",
        }
    }

    def print_structure(data, indent=0):
        """递归打印项目结构"""
        for key, value in data.items():
            if isinstance(value, dict):
                print("  " * indent + f"📁 {key}/")
                print_structure(value, indent + 1)
            else:
                print("  " * indent + f"📄 {key}: {value}")

    print_structure(structure)

    # 核心调用链分析
    print("\n--- 核心调用链: 从加载到推理 ---")
    call_chain = [
        "AutoModel.from_pretrained('bert-base-uncased')",
        "  -> AutoConfig.from_pretrained()  # 加载配置",
        "  -> config_class.from_pretrained()  # 实例化Config",
        "  -> model_class.from_pretrained()  # 实例化Model",
        "    -> download from hub  # 下载模型权重",
        "    -> load_state_dict()  # 加载权重",
        "    -> return model  # 返回模型实例",
        "",
        "model(**inputs)  # 前向推理",
        "  -> Model.forward()  # 调用模型forward方法",
        "    -> embeddings()  # 嵌入层",
        "    -> encoder()  # 编码器(多层Transformer)",
        "      -> layer_1.forward()  # 第一层",
        "        -> self_attention()  # 自注意力",
        "        -> feed_forward()  # 前馈网络",
        "      -> layer_2.forward()  # 第二层",
        "      -> ...",
        "    -> pooler()  # 池化层",
        "  -> return ModelOutput  # 返回结构化输出"
    ]
    for line in call_chain:
        print(f"  {line}")

    # 关键设计模式
    print("\n--- Transformers中的设计模式 ---")
    patterns = {
        "工厂模式(Factory)": "AutoModel/AutoTokenizer根据配置自动选择实现类",
        "注册模式(Registry)": "MODEL_MAPPING字典注册模型与配置的映射关系",
        "策略模式(Strategy)": "不同的Tokenizer使用不同的分词策略",
        "模板方法(Template Method)": "PreTrainedModel定义通用流程, 子类实现具体逻辑",
        "建造者模式(Builder)": "TrainingArguments使用建造者模式配置参数"
    }
    for pattern, desc in patterns.items():
        print(f"  {pattern}: {desc}")


# =============================================
# 5. 源码阅读笔记模板
# =============================================
def generate_reading_notes_template():
    """生成源码阅读笔记模板"""
    print_section("源码阅读笔记模板")

    template = {
        "项目信息": {
            "名称": "",
            "GitHub": "",
            "版本": "",
            "语言": "Python",
            "阅读日期": datetime.now().strftime("%Y-%m-%d")
        },
        "架构概览": {
            "项目目标": "",
            "核心模块": [],
            "依赖关系": "",
            "入口文件": ""
        },
        "关键发现": [
            "1. 发现1: ",
            "2. 发现2: ",
            "3. 发现3: "
        ],
        "代码质量评估": {
            "代码风格": "PEP8/自定义",
            "测试覆盖": "高/中/低",
            "文档质量": "优秀/良好/一般/差",
            "注释密度": "高/中/低"
        },
        "学习要点": [
            "1. 学到的设计模式: ",
            "2. 学到的编码技巧: ",
            "3. 学到的架构思想: "
        ],
        "疑问与TODO": [
            "1. 问题: ",
            "2. 待深入研究: "
        ]
    }

    print(json.dumps(template, ensure_ascii=False, indent=2))

    # 保存模板
    output_file = os.path.join(os.path.dirname(__file__), "code_reading_template.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)
    print(f"\n模板已保存到: {output_file}")


# =============================================
# 主程序
# =============================================
if __name__ == "__main__":
    print("=" * 60)
    print("  W35 Day 2 - 阅读源码方法")
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    demonstrate_code_reading_methods()
    demonstrate_navigation_techniques()
    demonstrate_debugging_techniques()
    analyze_transformers_structure()
    generate_reading_notes_template()

    print("\n" + "=" * 60)
    print("  阅读源码是提升编程能力的最佳方式之一!")
    print("  建议: 每周至少深入阅读一个开源项目的核心模块")
    print("=" * 60)
