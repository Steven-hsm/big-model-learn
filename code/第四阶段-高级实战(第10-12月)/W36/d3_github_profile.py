"""
W36 Day 3 - GitHub Profile优化
===============================
主题: README设计模板, 项目展示, 活动图表生成
"""

import os
from datetime import datetime

try:
    import matplotlib
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def print_section(title):
    """打印分隔线"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# =============================================
# 1. Profile README设计模板
# =============================================
def generate_profile_readme():
    """生成GitHub Profile README模板"""
    print_section("GitHub Profile README模板")

    readme_template = """<!-- 文件位置: ~/.github/README.md -->
<!-- 这个README会显示在你的GitHub Profile页面 -->

# Hi there! 👋 I'm [Your Name]

<!-- 可选: 添加一个动态的打字效果 -->
<!-- 使用: https://github.com/DenverCoder1/readme-typing-svg -->
<img align="right" alt="Coding" width="400" src="https://cdn.dribbble.com/users/1162077/screenshots/3848914/programmer.gif">

## About Me 🚀

I'm a **AI Engineer** transitioning from Java backend development,
passionate about building intelligent applications with Large Language Models.

- 🌱 Currently exploring: **RAG Systems, AI Agents, Multi-modal Models**
- 💻 Tech Stack: **Python, PyTorch, Transformers, LangChain**
- 📝 I write about AI at: [Blog URL](https://yourblog.com)
- 💬 Ask me about: **LLM, NLP, Python, System Design**
- 📫 How to reach me: [Email](mailto:your.email@example.com)

## Tech Stack 🛠️

### Languages & Frameworks
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Java](https://img.shields.io/badge/Java-ED8B00?style=flat-square&logo=java&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)

### AI & Data
![Transformers](https://img.shields.io/badge/Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=langchain&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)

### Tools & Platforms
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=flat-square&logo=git&logoColor=white)
![Linux](https://img.shields.io/badge/Linux-FCC624?style=flat-square&logo=linux&logoColor=black)

## Featured Projects 🌟

<!-- 使用GitHub Readme Stats卡片 -->
[![Readme Card](https://github-readme-stats.vercel.app/api/pin/?username=yourusername&repo=your-best-project)](https://github.com/yourusername/your-best-project)

### 🔥 [RAG Knowledge Base](https://github.com/yourusername/rag-knowledge-base)
A production-ready RAG system with document parsing, vector search, and LLM-powered Q&A.
- **Tech**: Python, LangChain, ChromaDB, FastAPI
- **Features**: Multi-format document support, hybrid search, evaluation framework

### 🤖 [AI Chatbot](https://github.com/yourusername/ai-chatbot)
An intelligent chatbot powered by fine-tuned LLM with conversation memory.
- **Tech**: Python, Transformers, Redis, Docker
- **Features**: Multi-turn dialogue, personality customization, streaming response

### 📊 [ML Pipeline](https://github.com/yourusername/ml-pipeline)
An end-to-end ML pipeline for model training, evaluation, and deployment.
- **Tech**: Python, MLflow, Airflow, Kubernetes
- **Features**: Auto hyperparameter tuning, model versioning, A/B testing

## GitHub Stats 📊

<!-- 使用 github-readme-stats -->
![Your Name's GitHub stats](https://github-readme-stats.vercel.app/api?username=yourusername&show_icons=true&theme=radical)

![Top Langs](https://github-readme-stats.vercel.app/api/top-langs/?username=yourusername&layout=compact&theme=radical)

[![GitHub Streak](https://streak-stats.demolab.com/?user=yourusername&theme=radical)](https://git.io/streak-stats)

## Activity Graph 📈

[![Activity Graph](https://github-readme-activity-graph.vercel.app/graph?username=yourusername&theme=react-dark)](https://github.com/ashutosh00710/github-readme-activity-graph)

## Blog Posts 📝

<!-- 使用博客RSS自动更新 -->
- [构建企业级RAG系统的完整指南](https://yourblog.com/post1)
- [大模型微调实战: LoRA从原理到代码](https://yourblog.com/post2)
- [Java工程师转型AI: 我的39周学习路线](https://yourblog.com/post3)

## Connect with Me 🤝

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/yourusername)
[![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?style=flat-square&logo=twitter&logoColor=white)](https://twitter.com/yourusername)
[![知乎](https://img.shields.io/badge/知乎-0084FF?style=flat-square&logo=zhihu&logoColor=white)](https://zhihu.com/people/yourusername)

---

⭐ From [yourusername](https://github.com/yourusername)

<!-- 可选: 添加访客计数 -->
![Visitor Count](https://profile-counter.glitch.me/yourusername/count.svg)
"""

    print(readme_template)

    # 保存README模板
    output_file = os.path.join(os.path.dirname(__file__), "github_profile_readme_template.md")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(readme_template)
    print(f"\n  Profile README模板已保存: {output_file}")


# =============================================
# 2. 项目展示优化
# =============================================
def demonstrate_project_showcase():
    """展示项目展示优化技巧"""
    print_section("项目展示优化")

    # 项目README模板
    project_readme = """# 项目名称 🚀

> 一句话描述你的项目

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Stars](https://img.shields.io/github/stars/username/repo.svg)](https://github.com/username/repo)

## 📸 Demo

<!-- 添加GIF或截图 -->
![Demo](docs/demo.gif)

## ✨ 功能特点

- 🤖 支持多种LLM后端(OpenAI, Anthropic, 本地模型)
- 📄 多格式文档解析(PDF, Word, Markdown)
- 🔍 混合检索(向量+关键词+语义)
- 💡 流式响应, 支持对话记忆
- 📊 内置评估框架

## 🏗️ 架构

<!-- 架构图 -->
```mermaid
graph LR
    A[用户] --> B[API Gateway]
    B --> C[Document Parser]
    B --> D[Query Engine]
    C --> E[Vector Store]
    D --> E
    D --> F[LLM]
    F --> G[Response]
```

## 🚀 快速开始

```bash
pip install your-package
```

```python
from your_package import App

app = App(model="gpt-4")
app.add_documents("./data")
answer = app.query("你的问题")
print(answer)
```

## 📊 性能基准

| 指标 | 值 |
|------|-----|
| 检索准确率 | 92% |
| 平均响应时间 | 1.2s |
| 支持文档数 | 10,000+ |
| 并发请求 | 100 QPS |

## 🛣️ Roadmap

- [x] 基础RAG功能
- [x] 多格式文档支持
- [ ] 多模态RAG
- [ ] Agent集成
- [ ] 分布式部署

## 🤝 贡献

欢迎贡献! 请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)

## 📄 License

MIT License - 详见 [LICENSE](LICENSE)
"""

    print(project_readme)

    # 保存项目README模板
    output_file = os.path.join(os.path.dirname(__file__), "project_readme_template.md")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(project_readme)
    print(f"\n  项目README模板已保存: {output_file}")

    # Pinned Repositories策略
    print("\n--- Pinned Repositories策略 ---")
    strategies = [
        "1. 展示6个最有代表性的项目",
        "2. 优先展示AI/ML相关项目",
        "3. 确保每个项目都有完整的README",
        "4. 项目类型多样化(应用、库、工具、教程)",
        "5. 添加合适的Topics标签",
        "6. 保持项目活跃(最近有commit)"
    ]
    for s in strategies:
        print(f"  {s}")


# =============================================
# 3. 活动图表生成
# =============================================
def generate_activity_charts():
    """生成GitHub活动分析图表"""
    print_section("GitHub活动图表生成")

    if not HAS_MATPLOTLIB or not HAS_NUMPY:
        print("  需要matplotlib和numpy库来生成图表")
        return

    output_dir = os.path.dirname(__file__)

    # 模拟贡献数据
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 月度贡献趋势
    months = ['1月', '2月', '3月', '4月', '5月', '6月',
              '7月', '8月', '9月', '10月', '11月', '12月']
    commits = [15, 28, 45, 62, 78, 85, 92, 88, 95, 102, 98, 110]
    prs = [1, 2, 5, 8, 10, 12, 15, 14, 18, 20, 22, 25]
    issues = [3, 5, 8, 10, 12, 15, 12, 10, 14, 16, 18, 20]

    axes[0, 0].plot(months, commits, 'b-o', linewidth=2, markersize=6, label='Commits')
    axes[0, 0].plot(months, [x * 5 for x in prs], 'r-s', linewidth=2, markersize=6, label='PRs (x5)')
    axes[0, 0].plot(months, [x * 3 for x in issues], 'g-^', linewidth=2, markersize=6, label='Issues (x3)')
    axes[0, 0].set_title('GitHub年度贡献趋势')
    axes[0, 0].set_xlabel('月份')
    axes[0, 0].set_ylabel('数量')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 贡献类型分布
    types = ['代码提交', 'Pull Request', 'Issue', 'Code Review', '文档', '其他']
    values = [45, 20, 15, 10, 8, 2]
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336', '#607D8B']
    explode = (0.05, 0, 0, 0, 0, 0)

    axes[0, 1].pie(values, labels=types, autopct='%1.1f%%', colors=colors,
                   explode=explode, shadow=True)
    axes[0, 1].set_title('贡献类型分布')

    # 3. 项目Stars增长
    projects = ['RAG System', 'AI Chatbot', 'ML Tools', 'Data Pipeline', 'NLP Utils']
    stars = [520, 380, 250, 180, 120]
    colors2 = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

    axes[1, 0].barh(projects, stars, color=colors2)
    axes[1, 0].set_xlabel('Stars')
    axes[1, 0].set_title('项目Stars排行')
    for i, v in enumerate(stars):
        axes[1, 0].text(v + 5, i, f'★{v}', va='center')

    # 4. 技术栈使用频率
    techs = ['Python', 'Jupyter', 'Markdown', 'Dockerfile', 'YAML', 'Shell']
    usage = [65, 15, 10, 5, 3, 2]
    colors3 = plt.cm.Set3(np.linspace(0, 1, len(techs)))

    axes[1, 1].bar(techs, usage, color=colors3)
    axes[1, 1].set_ylabel('使用频率(%)')
    axes[1, 1].set_title('代码语言使用分布')
    for i, v in enumerate(usage):
        axes[1, 1].text(i, v + 0.5, f'{v}%', ha='center')

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "github_activity_charts.png")
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  GitHub活动图表已保存: {chart_path}")

    # GitHub Stats工具推荐
    print("\n--- GitHub Profile增强工具 ---")
    tools = [
        "github-readme-stats: 统计卡片 - https://github.com/anuraghazra/github-readme-stats",
        "github-readme-streak-stats: 连续贡献 - https://github.com/DenverCoder1/github-readme-streak-stats",
        "github-readme-activity-graph: 活动图 - https://github.com/ashutosh00710/github-readme-activity-graph",
        "skill-icons: 技术栈图标 - https://github.com/tandpfun/skill-icons",
        "readme-typing-svg: 打字效果 - https://github.com/DenverCoder1/readme-typing-svg",
        "profile-counter: 访客计数 - https://github.com/khattakdev/profile-counter"
    ]
    for tool in tools:
        print(f"  {tool}")


# =============================================
# 主程序
# =============================================
if __name__ == "__main__":
    print("=" * 60)
    print("  W36 Day 3 - GitHub Profile优化")
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    generate_profile_readme()
    demonstrate_project_showcase()
    generate_activity_charts()

    print("\n" + "=" * 60)
    print("  一个好的GitHub Profile是技术人员最好的名片!")
    print("  建议: 花半天时间优化你的Profile和项目README")
    print("=" * 60)
