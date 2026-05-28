"""
W37 Day 2 - 个人网站模板生成
==============================
主题: 项目展示页, 技术博客页HTML模板
"""

import os
from datetime import datetime


def print_section(title):
    """打印分隔线"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# =============================================
# 1. 个人网站首页模板
# =============================================
def generate_homepage_template():
    """生成个人网站首页HTML模板"""
    print_section("个人网站首页HTML模板")

    html_template = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>张三 | AI应用工程师</title>
    <meta name="description" content="AI应用工程师 - 专注于LLM应用开发和RAG系统构建">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            line-height: 1.6; color: #333; background: #fafafa;
        }
        .container { max-width: 1100px; margin: 0 auto; padding: 0 20px; }

        /* 导航 */
        nav {
            background: #fff; border-bottom: 1px solid #eee;
            position: sticky; top: 0; z-index: 100;
        }
        nav .container {
            display: flex; justify-content: space-between;
            align-items: center; height: 60px;
        }
        nav a { text-decoration: none; color: #333; margin-left: 20px; }
        nav a:hover { color: #0066cc; }

        /* Hero */
        .hero {
            padding: 80px 0; text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .hero h1 { font-size: 2.5em; margin-bottom: 10px; }
        .hero p { font-size: 1.2em; opacity: 0.9; max-width: 600px; margin: 0 auto; }
        .hero .tags { margin-top: 20px; }
        .hero .tag {
            display: inline-block; background: rgba(255,255,255,0.2);
            padding: 5px 15px; border-radius: 20px; margin: 5px; font-size: 0.9em;
        }

        /* 区域标题 */
        .section { padding: 60px 0; }
        .section h2 { font-size: 1.8em; margin-bottom: 30px; text-align: center; }

        /* 技能卡片 */
        .skills-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }
        .skill-card {
            background: #fff; padding: 25px; border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }
        .skill-card h3 { margin-bottom: 15px; color: #0066cc; }
        .skill-card ul { list-style: none; }
        .skill-card li { padding: 5px 0; }
        .skill-card li::before { content: "▸ "; color: #0066cc; }

        /* 项目卡片 */
        .projects-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 25px; }
        .project-card {
            background: #fff; border-radius: 10px; overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08); transition: transform 0.3s;
        }
        .project-card:hover { transform: translateY(-5px); }
        .project-img { height: 180px; background: linear-gradient(135deg, #667eea, #764ba2); }
        .project-content { padding: 20px; }
        .project-content h3 { margin-bottom: 10px; }
        .project-content p { color: #666; font-size: 0.95em; }
        .project-tags { margin-top: 15px; }
        .project-tags span {
            display: inline-block; background: #f0f0f0; padding: 3px 10px;
            border-radius: 15px; font-size: 0.8em; margin: 2px;
        }
        .project-links { margin-top: 15px; }
        .project-links a { color: #0066cc; text-decoration: none; margin-right: 15px; }

        /* 联系 */
        .contact { text-align: center; padding: 40px 0; background: #333; color: white; }
        .contact a { color: #6699ff; text-decoration: none; margin: 0 15px; }

        footer { text-align: center; padding: 20px; color: #999; font-size: 0.85em; }
    </style>
</head>
<body>
    <nav>
        <div class="container">
            <strong>张三</strong>
            <div>
                <a href="#about">关于</a>
                <a href="#projects">项目</a>
                <a href="#blog">博客</a>
                <a href="#contact">联系</a>
            </div>
        </div>
    </nav>

    <div class="hero">
        <h1>AI应用工程师</h1>
        <p>5年Java后端 + 1年AI应用开发经验, 专注于LLM应用和RAG系统构建</p>
        <div class="tags">
            <span class="tag">Python</span>
            <span class="tag">LLM</span>
            <span class="tag">RAG</span>
            <span class="tag">LangChain</span>
            <span class="tag">PyTorch</span>
        </div>
    </div>

    <div class="section" id="about">
        <div class="container">
            <h2>技术栈</h2>
            <div class="skills-grid">
                <div class="skill-card">
                    <h3>AI/ML</h3>
                    <ul>
                        <li>LangChain / LlamaIndex</li>
                        <li>Transformers / PEFT</li>
                        <li>PyTorch / scikit-learn</li>
                        <li>ChromaDB / Milvus</li>
                    </ul>
                </div>
                <div class="skill-card">
                    <h3>后端开发</h3>
                    <ul>
                        <li>Python (FastAPI / Flask)</li>
                        <li>Java (Spring Boot)</li>
                        <li>MySQL / Redis / MongoDB</li>
                        <li>Docker / Kubernetes</li>
                    </ul>
                </div>
                <div class="skill-card">
                    <h3>工具与平台</h3>
                    <ul>
                        <li>Git / GitHub Actions</li>
                        <li>AWS / 阿里云</li>
                        <li>MLflow / Weights & Biases</li>
                        <li>Grafana / Prometheus</li>
                    </ul>
                </div>
            </div>
        </div>
    </div>

    <div class="section" id="projects">
        <div class="container">
            <h2>精选项目</h2>
            <div class="projects-grid">
                <div class="project-card">
                    <div class="project-img"></div>
                    <div class="project-content">
                        <h3>RAG知识库问答系统</h3>
                        <p>企业级RAG系统, 支持多格式文档, 混合检索, 检索准确率91%</p>
                        <div class="project-tags">
                            <span>Python</span><span>LangChain</span><span>ChromaDB</span>
                        </div>
                        <div class="project-links">
                            <a href="#">GitHub</a><a href="#">Demo</a><a href="#">博客</a>
                        </div>
                    </div>
                </div>
                <div class="project-card">
                    <div class="project-img" style="background: linear-gradient(135deg, #f093fb, #f5576c);"></div>
                    <div class="project-content">
                        <h3>AI Agent工作流引擎</h3>
                        <p>基于ReAct模式的Agent框架, 支持工具调用、记忆和多步推理</p>
                        <div class="project-tags">
                            <span>Python</span><span>LangGraph</span><span>Agent</span>
                        </div>
                        <div class="project-links">
                            <a href="#">GitHub</a><a href="#">Demo</a>
                        </div>
                    </div>
                </div>
                <div class="project-card">
                    <div class="project-img" style="background: linear-gradient(135deg, #4facfe, #00f2fe);"></div>
                    <div class="project-content">
                        <h3>大模型微调平台</h3>
                        <p>LoRA微调流水线, 支持10+种模型, 量化推理, 一键部署</p>
                        <div class="project-tags">
                            <span>PyTorch</span><span>LoRA</span><span>vLLM</span>
                        </div>
                        <div class="project-links">
                            <a href="#">GitHub</a><a href="#">博客</a>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div class="contact" id="contact">
        <h2>联系我</h2>
        <p style="margin-top: 15px;">
            <a href="mailto:email@example.com">邮箱</a>
            <a href="https://github.com/username">GitHub</a>
            <a href="https://linkedin.com/in/username">LinkedIn</a>
            <a href="https://zhihu.com/people/username">知乎</a>
        </p>
    </div>

    <footer>
        <p>&copy; 2026 张三. 使用 ❤️ 和 Python 构建</p>
    </footer>
</body>
</html>"""

    print(f"  HTML模板共 {len(html_template)} 字符")

    output_file = os.path.join(os.path.dirname(__file__), "portfolio_homepage.html")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_template)
    print(f"  首页模板已保存: {output_file}")


# =============================================
# 2. 项目展示页模板
# =============================================
def generate_project_page_template():
    """生成项目展示页HTML模板"""
    print_section("项目展示页HTML模板")

    project_html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>项目名称 - 张三</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, sans-serif; line-height: 1.8; color: #333; }
        .container { max-width: 900px; margin: 0 auto; padding: 0 20px; }

        .project-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 60px 0; text-align: center;
        }
        .project-header h1 { font-size: 2.2em; margin-bottom: 10px; }
        .project-header .meta { opacity: 0.9; }
        .badges { margin-top: 15px; }
        .badges a {
            display: inline-block; padding: 8px 20px; background: rgba(255,255,255,0.2);
            color: white; text-decoration: none; border-radius: 25px; margin: 5px;
        }

        .content { padding: 40px 0; }
        .content h2 { margin: 30px 0 15px; color: #444; border-left: 4px solid #667eea; padding-left: 15px; }
        .content p { margin-bottom: 15px; }
        .content ul { margin: 10px 0 15px 20px; }
        .content code { background: #f4f4f4; padding: 2px 6px; border-radius: 4px; }
        .content pre { background: #1e1e1e; color: #d4d4d4; padding: 20px; border-radius: 8px; overflow-x: auto; margin: 15px 0; }

        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin: 20px 0; }
        .metric { background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; }
        .metric .number { font-size: 2em; font-weight: bold; color: #667eea; }
        .metric .label { color: #666; font-size: 0.9em; }

        .tech-stack { display: flex; flex-wrap: wrap; gap: 10px; margin: 15px 0; }
        .tech-badge {
            background: #e8f4f8; color: #0066cc; padding: 5px 15px;
            border-radius: 20px; font-size: 0.9em;
        }

        .gallery { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 20px 0; }
        .gallery-item { background: #f0f0f0; height: 200px; border-radius: 8px; display: flex; align-items: center; justify-content: center; color: #999; }
    </style>
</head>
<body>
    <div class="project-header">
        <div class="container">
            <h1>RAG知识库问答系统</h1>
            <p class="meta">企业级RAG系统 | 2025.09 - 2025.12</p>
            <div class="badges">
                <a href="#">GitHub源码</a>
                <a href="#">在线Demo</a>
                <a href="#">技术博客</a>
            </div>
        </div>
    </div>

    <div class="container content">
        <h2>项目概述</h2>
        <p>企业级RAG(检索增强生成)知识库问答系统, 支持多格式文档导入、
        混合检索策略和LLM驱动的智能问答。</p>

        <div class="metrics">
            <div class="metric">
                <div class="number">91%</div>
                <div class="label">检索准确率</div>
            </div>
            <div class="metric">
                <div class="number">1.2s</div>
                <div class="label">平均响应时间</div>
            </div>
            <div class="metric">
                <div class="number">10万+</div>
                <div class="label">月查询量</div>
            </div>
            <div class="metric">
                <div class="number">89%</div>
                <div class="label">用户满意度</div>
            </div>
        </div>

        <h2>技术栈</h2>
        <div class="tech-stack">
            <span class="tech-badge">Python</span>
            <span class="tech-badge">LangChain</span>
            <span class="tech-badge">ChromaDB</span>
            <span class="tech-badge">FastAPI</span>
            <span class="tech-badge">Docker</span>
            <span class="tech-badge">Redis</span>
        </div>

        <h2>核心功能</h2>
        <ul>
            <li>多格式文档解析 (PDF, Word, Markdown, HTML)</li>
            <li>智能分块策略 (语义分块 + 重叠窗口)</li>
            <li>混合检索 (向量检索 + BM25关键词检索)</li>
            <li>流式响应和会话记忆</li>
            <li>评估框架 (RAGAS + 自定义指标)</li>
        </ul>

        <h2>架构设计</h2>
        <div class="gallery">
            <div class="gallery-item">架构图</div>
            <div class="gallery-item">流程图</div>
        </div>

        <h2>关键代码</h2>
        <pre><code># 混合检索实现
def hybrid_search(query, top_k=5):
    # 向量检索
    vector_results = vector_store.search(query, top_k=top_k*2)
    # BM25关键词检索
    bm25_results = bm25_index.search(query, top_k=top_k*2)
    # 融合排序 (Reciprocal Rank Fusion)
    merged = rrf_merge(vector_results, bm25_results)
    return merged[:top_k]</code></pre>

        <h2>项目成果</h2>
        <ul>
            <li>检索准确率从72%提升至91%</li>
            <li>自动解决85%的常见问题</li>
            <li>客服工作量减少50%</li>
            <li>用户满意度从65%提升至89%</li>
        </ul>
    </div>
</body>
</html>"""

    output_file = os.path.join(os.path.dirname(__file__), "project_showcase.html")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(project_html)
    print(f"  项目展示页模板已保存: {output_file}")


# =============================================
# 3. 技术博客页模板
# =============================================
def generate_blog_page_template():
    """生成技术博客页HTML模板"""
    print_section("技术博客页HTML模板")

    blog_html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>技术博客 - 张三</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, sans-serif; line-height: 1.8; color: #333; background: #fafafa; }
        .container { max-width: 800px; margin: 0 auto; padding: 0 20px; }

        header { background: #fff; border-bottom: 1px solid #eee; padding: 15px 0; }
        header .container { display: flex; justify-content: space-between; align-items: center; }
        header a { text-decoration: none; color: #333; margin-left: 20px; }

        .blog-list { padding: 40px 0; }
        .blog-post {
            background: #fff; padding: 30px; border-radius: 10px; margin-bottom: 25px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        .blog-post h2 { margin-bottom: 10px; }
        .blog-post h2 a { color: #333; text-decoration: none; }
        .blog-post h2 a:hover { color: #0066cc; }
        .blog-post .meta { color: #999; font-size: 0.9em; margin-bottom: 15px; }
        .blog-post .excerpt { color: #666; }
        .blog-post .tags { margin-top: 15px; }
        .blog-post .tag {
            display: inline-block; background: #f0f0f0; padding: 3px 12px;
            border-radius: 15px; font-size: 0.8em; margin: 2px; color: #666;
        }

        .article { padding: 40px 0; }
        .article h1 { font-size: 2em; margin-bottom: 15px; }
        .article .meta { color: #999; margin-bottom: 30px; }
        .article h2 { margin: 30px 0 15px; }
        .article p { margin-bottom: 15px; }
        .article pre { background: #1e1e1e; color: #d4d4d4; padding: 20px; border-radius: 8px; margin: 15px 0; }
        .article code { background: #f4f4f4; padding: 2px 6px; border-radius: 4px; }
        .article blockquote { border-left: 4px solid #667eea; padding: 10px 20px; background: #f8f9fa; margin: 15px 0; }
    </style>
</head>
<body>
    <header>
        <div class="container">
            <strong>张三的技术博客</strong>
            <div>
                <a href="#">首页</a>
                <a href="#">归档</a>
                <a href="#">关于</a>
                <a href="#">RSS</a>
            </div>
        </div>
    </header>

    <div class="container blog-list">
        <div class="blog-post">
            <h2><a href="#">从零构建RAG问答系统(下): 检索与生成</a></h2>
            <div class="meta">2026-05-20 | 阅读 8 分钟 | AI实战</div>
            <div class="excerpt">
                本文是RAG系统系列的下篇, 重点介绍检索策略和生成优化。
                我们将实现混合检索(向量+BM25)、重排序和流式响应...
            </div>
            <div class="tags">
                <span class="tag">RAG</span>
                <span class="tag">LangChain</span>
                <span class="tag">Python</span>
            </div>
        </div>

        <div class="blog-post">
            <h2><a href="#">大模型微调实战: LoRA原理与代码实现</a></h2>
            <div class="meta">2026-05-13 | 阅读 12 分钟 | 深度分析</div>
            <div class="excerpt">
                详细解析LoRA(Low-Rank Adaptation)的数学原理,
                并手把手实现LoRA微调流程, 包括数据处理、训练和评估...
            </div>
            <div class="tags">
                <span class="tag">微调</span>
                <span class="tag">LoRA</span>
                <span class="tag">PyTorch</span>
            </div>
        </div>

        <div class="blog-post">
            <h2><a href="#">Java工程师转型AI: 我的39周学习路线</a></h2>
            <div class="meta">2026-05-06 | 阅读 10 分钟 | 经验分享</div>
            <div class="excerpt">
                从Java后端开发转型AI应用的完整经历, 包括学习路线、
                踩坑记录和求职建议, 送给同样想转型的工程师...
            </div>
            <div class="tags">
                <span class="tag">转型</span>
                <span class="tag">学习路线</span>
                <span class="tag">经验分享</span>
            </div>
        </div>
    </div>
</body>
</html>"""

    output_file = os.path.join(os.path.dirname(__file__), "blog_page.html")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(blog_html)
    print(f"  博客页模板已保存: {output_file}")

    # 部署建议
    print("\n--- 部署建议 ---")
    deploy_options = [
        "GitHub Pages (免费, 简单, 推荐)",
        "Vercel (免费, 支持Serverless)",
        "Netlify (免费, 自动部署)",
        "Hugo/Jekyll (静态站点生成器, 适合博客)",
        "买域名绑定 (~50元/年)"
    ]
    for option in deploy_options:
        print(f"  {option}")


# =============================================
# 主程序
# =============================================
if __name__ == "__main__":
    print("=" * 60)
    print("  W37 Day 2 - 个人网站模板生成")
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    generate_homepage_template()
    generate_project_page_template()
    generate_blog_page_template()

    print("\n" + "=" * 60)
    print("  个人网站是技术人最好的展示窗口!")
    print("  建议: 使用GitHub Pages免费部署, 绑定自定义域名")
    print("=" * 60)
