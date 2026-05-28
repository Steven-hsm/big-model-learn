"""
W34-D6 开源准备
================
准备项目开源, 包括:
- LICENSE选择指南
- README模板
- 贡献指南

开源让项目获得更多关注和贡献。
"""


# ============================================================
# 1. LICENSE选择指南
# ============================================================

class LicenseGuide:
    """LICENSE选择指南"""

    LICENSES = {
        'MIT': {
            'full_name': 'MIT License',
            'description': '最宽松的开源协议, 允许任意使用、修改和分发',
            'allows': ['商业使用', '修改', '分发', '私人使用'],
            'requires': ['保留版权声明', '保留LICENSE文件'],
            'suitable_for': '个人项目、工具库、希望最大程度被采用的项目',
            'popularity': '高',
        },
        'Apache-2.0': {
            'full_name': 'Apache License 2.0',
            'description': '宽松协议, 包含专利授权条款',
            'allows': ['商业使用', '修改', '分发', '私人使用', '专利授权'],
            'requires': ['保留版权声明', '保留LICENSE', '声明修改部分', '保留NOTICE'],
            'suitable_for': '企业项目、涉及专利的技术、大型开源项目',
            'popularity': '高',
        },
        'GPL-3.0': {
            'full_name': 'GNU General Public License v3.0',
            'description': '强Copyleft协议, 衍生作品必须同样开源',
            'allows': ['商业使用', '修改', '分发'],
            'requires': ['开源衍生作品', '保留版权', '声明修改', '使用相同协议'],
            'suitable_for': '希望确保代码永远开源的项目',
            'popularity': '中',
        },
        'BSD-3-Clause': {
            'full_name': 'BSD 3-Clause License',
            'description': '宽松协议, 类似MIT但禁止使用项目名称背书',
            'allows': ['商业使用', '修改', '分发'],
            'requires': ['保留版权声明', '不使用作者名背书'],
            'suitable_for': '学术项目、研究代码',
            'popularity': '中',
        },
    }

    def recommend(self, project_type: str) -> str:
        """根据项目类型推荐LICENSE"""
        recommendations = {
            '个人工具': 'MIT',
            'AI/ML项目': 'Apache-2.0',
            'Web框架': 'MIT',
            '学术研究': 'BSD-3-Clause',
            '企业产品': 'Apache-2.0',
            '强制开源': 'GPL-3.0',
        }
        return recommendations.get(project_type, 'MIT')

    def print_comparison(self):
        """打印LICENSE对比"""
        print("开源LICENSE对比")
        print("=" * 70)
        print(f"{'协议':<15} {'说明':<25} {'适合场景':<20} {'流行度':<10}")
        print("-" * 70)
        for name, info in self.LICENSES.items():
            print(f"{name:<15} {info['description'][:24]:<25} "
                  f"{info['suitable_for'][:19]:<20} {info['popularity']:<10}")

    def generate_mit_license(self, year: str = "2024", author: str = "Your Name") -> str:
        """生成MIT LICENSE文件内容"""
        return f"""MIT License

Copyright (c) {year} {author}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


# ============================================================
# 2. README模板
# ============================================================

class ReadmeTemplate:
    """README模板生成器"""

    @staticmethod
    def generate(project_name: str = "RAG知识库问答系统",
                  description: str = "",
                  features: list = None,
                  tech_stack: list = None) -> str:
        """生成README"""
        if not description:
            description = "基于检索增强生成(RAG)技术的智能知识库问答系统"
        if not features:
            features = [
                "混合检索: BM25 + 向量检索 + RRF融合",
                "智能问答: 基于知识库的准确回答",
                "流式输出: SSE实时推送",
                "文档管理: 自动分块和索引",
                "权限控制: RBAC + 多租户",
            ]
        if not tech_stack:
            tech_stack = [
                ("Python 3.11", "开发语言"),
                ("FastAPI", "Web框架"),
                ("Gradio", "前端界面"),
                ("sentence-transformers", "文本嵌入"),
                ("Docker", "容器化部署"),
            ]

        lines = [
            f"# {project_name}",
            "",
            f"{description}",
            "",
            "## 功能特性",
            "",
        ]
        for f in features:
            lines.append(f"- {f}")

        lines.extend([
            "",
            "## 快速开始",
            "",
            "```bash",
            "# 安装依赖",
            "pip install -r requirements.txt",
            "",
            "# 配置环境",
            "cp .env.example .env",
            "",
            "# 启动服务",
            "python main.py",
            "```",
            "",
            "## 技术栈",
            "",
            "| 技术 | 用途 |",
            "|------|------|",
        ])
        for name, purpose in tech_stack:
            lines.append(f"| {name} | {purpose} |")

        lines.extend([
            "",
            "## 项目结构",
            "```",
            "src/",
            "  retrieval/    # 检索模块",
            "  generation/   # 生成模块",
            "  api/          # API路由",
            "  core/         # 核心配置",
            "tests/          # 测试",
            "docs/           # 文档",
            "```",
            "",
            "## 贡献",
            "",
            "欢迎贡献! 请阅读 CONTRIBUTING.md 了解详情。",
            "",
            "## 许可证",
            "",
            "MIT License - 详见 LICENSE 文件",
        ])

        return "\n".join(lines)


# ============================================================
# 3. 贡献指南
# ============================================================

class ContributingGuide:
    """贡献指南生成器"""

    @staticmethod
    def generate() -> str:
        return """# 贡献指南

感谢你对本项目的关注! 以下是贡献流程。

## 如何贡献

### 报告Bug
1. 在Issues中搜索是否已有相关问题
2. 如果没有, 创建新Issue, 包含:
   - 问题描述
   - 复现步骤
   - 预期行为 vs 实际行为
   - 环境信息(Python版本、OS等)

### 提交代码
1. Fork本仓库
2. 创建特性分支: `git checkout -b feature/your-feature`
3. 编写代码并添加测试
4. 确保所有测试通过: `pytest tests/`
5. 提交代码: `git commit -m 'Add some feature'`
6. 推送分支: `git push origin feature/your-feature`
7. 创建Pull Request

### 代码规范
- 遵循PEP 8代码风格
- 添加必要的注释和文档字符串
- 编写单元测试
- 保持代码简洁, 避免过度工程

### 提交信息格式
```
<type>(<scope>): <subject>

<body>
```

类型(type):
- feat: 新功能
- fix: 修复Bug
- docs: 文档
- style: 格式
- refactor: 重构
- test: 测试
- chore: 构建/工具

示例:
```
feat(retrieval): 添加混合检索支持

实现BM25+向量检索的混合方案, 使用RRF融合算法。
```


## 开发环境设置

```bash
# 克隆项目
git clone https://github.com/your-org/rag-system.git
cd rag-system

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\\Scripts\\activate  # Windows

# 安装开发依赖
pip install -r requirements-dev.txt

# 运行测试
pytest tests/ -v
```

## 许可证
通过提交代码, 你同意你的贡献将在MIT License下发布。
"""


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("W34-D6 开源准备")
    print("=" * 60)

    # --- 1. LICENSE选择 ---
    print("\n--- 1. LICENSE选择 ---")
    guide = LicenseGuide()
    guide.print_comparison()

    print("\n推荐:")
    for project_type in ['个人工具', 'AI/ML项目', '企业产品']:
        print(f"  {project_type}: {guide.recommend(project_type)}")

    # --- 2. README ---
    print(f"\n{'='*60}")
    print("--- 2. README ---")
    print(f"{'='*60}")
    readme = ReadmeTemplate.generate()
    print(readme[:600])
    print(f"\n... (共{len(readme)}字符)")

    # --- 3. 贡献指南 ---
    print(f"\n{'='*60}")
    print("--- 3. 贡献指南 ---")
    print(f"{'='*60}")
    contributing = ContributingGuide.generate()
    print(contributing[:500])
    print(f"\n... (共{len(contributing)}字符)")

    print("\n完成!")
