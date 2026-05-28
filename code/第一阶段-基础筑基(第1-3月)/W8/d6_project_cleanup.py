"""
W08 Day 6: 项目代码��理 + GitHub
ML 项目结构模板、README 生成、依赖管理、Git 工作流
"""

import os
import json
from datetime import datetime

# ============================================================
# 1. ML 项目目录结构模板
# ============================================================
print("=" * 60)
print("项目代码整理 + GitHub")
print("=" * 60)

project_structure = """
ml-project/
|-- data/
|   |-- raw/                # 原始数据 (不可修改)
|   |-- processed/          # 处理后的数据
|   |-- external/           # 外部数据源
|   +-- README.md           # 数据说明
|
|-- notebooks/
|   |-- 01_eda.ipynb        # 数据探索
|   |-- 02_feature_eng.ipynb # 特征工程
|   |-- 03_modeling.ipynb   # 模型训练
|   +-- 04_evaluation.ipynb  # 模型评估
|
|-- src/
|   |-- __init__.py
|   |-- data/
|   |   |-- make_dataset.py    # 数据加载/处理
|   |   +-- validation.py      # 数据验证
|   |-- features/
|   |   |-- build_features.py  # 特征工程
|   |   +-- selectors.py       # 特征选择
|   |-- models/
|   |   |-- train.py           # 模型训练
|   |   |-- predict.py         # 模型预测
|   |   +-- evaluate.py        # 模型评估
|   +-- visualization/
|       |-- visualize.py       # 可视化工具
|       +-- plots.py           # 图表生成
|
|-- models/                 # 保存的模型文件
|   |-- v1_baseline/
|   |-- v2_tuned/
|   +-- latest/
|
|-- api/                    # 推理服务
|   |-- app.py              # FastAPI 应用
|   |-- requirements.txt    # API 依赖
|   +-- Dockerfile          # 容器化
|
|-- tests/
|   |-- test_data.py
|   |-- test_features.py
|   +-- test_model.py
|
|-- docs/
|   |-- project_report.md   # 项目报告
|   +-- api_docs.md         # API 文档
|
|-- configs/
|   |-- config.yaml         # 配置文件
|   +-- hyperparams.yaml    # 超参数配置
|
|-- .gitignore
|-- README.md
|-- requirements.txt
|-- setup.py
|-- Makefile
+-- LICENSE
"""

print("\n--- 1. 推荐 ML 项目目录结构 ---")
print(project_structure)


def create_project_template(base_dir: str = "ml_project_template"):
    """创建项目目录结构模板"""
    dirs = [
        "data/raw",
        "data/processed",
        "data/external",
        "notebooks",
        "src/data",
        "src/features",
        "src/models",
        "src/visualization",
        "models/v1_baseline",
        "models/v2_tuned",
        "models/latest",
        "api",
        "tests",
        "docs",
        "configs",
    ]

    files = {
        "src/__init__.py": "",
        "src/data/__init__.py": "",
        "src/features/__init__.py": "",
        "src/models/__init__.py": "",
        "src/visualization/__init__.py": "",
        "tests/__init__.py": "",
    }

    os.makedirs(base_dir, exist_ok=True)

    for d in dirs:
        os.makedirs(os.path.join(base_dir, d), exist_ok=True)

    for f, content in files.items():
        filepath = os.path.join(base_dir, f)
        if not os.path.exists(filepath):
            with open(filepath, 'w', encoding='utf-8') as fp:
                fp.write(content)

    print(f"  项目模板已创建于: {base_dir}/")
    print(f"  目录数: {len(dirs)}, 文件数: {len(files)}")


create_project_template()

# ============================================================
# 2. README.md 模板生成
# ============================================================
print("\n--- 2. README.md 模板 ---")

readme_template = f"""# {{PROJECT_NAME}}

> {{PROJECT_DESCRIPTION}}

## 项目概述

- **目标**: {{项目目标}}
- **数据集**: {{数据集描述}}
- **最佳模型**: {{最佳模型名称}}
- **性能指标**: {{RMSE / Accuracy 等指标}}
- **创建日期**: {datetime.now().strftime('%Y-%m-%d')}

## 项目结构

```
ml-project/
|-- data/           # 数据文件
|-- notebooks/      # Jupyter notebooks
|-- src/            # 源代码
|-- models/         # 保存的模型
|-- api/            # 推理服务
|-- tests/          # 单元测试
+-- docs/           # 文档
```

## 快速开始

### 环境配置

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\\Scripts\\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 数据准备

```bash
python src/data/make_dataset.py
```

### 模型训练

```bash
python src/models/train.py --config configs/config.yaml
```

### 模型评估

```bash
python src/models/evaluate.py --model models/latest/
```

### 启动推理服务

```bash
cd api
uvicorn app:app --host 0.0.0.0 --port 8000
```

## 实验记录

| 版本 | 模型 | 特征工程 | CV RMSE | Test RMSE | 备注 |
|------|------|----------|---------|-----------|------|
| v1   | LinearRegression | 基线 | - | - | 基线模型 |
| v2   | RandomForest | +派生特征 | - | - | 特征工程 |
| v3   | GradientBoosting | +调优 | - | - | 超参数调优 |
| v4   | Stacking | +集成 | - | - | 模型集成 |

## 技术栈

- Python 3.10+
- scikit-learn, pandas, numpy
- matplotlib, seaborn
- FastAPI / Flask (推理服务)
- joblib (模型保存)

## 作者

{{YOUR_NAME}}

## License

MIT License
"""

readme_path = os.path.join("ml_project_template", "README.md")
with open(readme_path, 'w', encoding='utf-8') as f:
    f.write(readme_template)
print(f"  README.md 模板已保存: {readme_path}")

# ============================================================
# 3. requirements.txt 生成
# ============================================================
print("\n--- 3. requirements.txt ---")

requirements = {
    "基础科学计算": [
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
    ],
    "机器学习": [
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.0.0",
    ],
    "可视化": [
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ],
    "推理服务": [
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "flask>=3.0.0",
        "pydantic>=2.0.0",
    ],
    "工具": [
        "joblib>=1.3.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.65.0",
    ],
    "开发": [
        "jupyter>=1.0.0",
        "pytest>=7.4.0",
        "black>=23.0.0",
        "flake8>=6.0.0",
    ],
}

all_requirements = []
for category, packages in requirements.items():
    print(f"\n  [{category}]")
    for pkg in packages:
        print(f"    {pkg}")
        all_requirements.append(pkg)

req_path = os.path.join("ml_project_template", "requirements.txt")
with open(req_path, 'w', encoding='utf-8') as f:
    f.write("# ML Project Requirements\n")
    f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d')}\n\n")
    for category, packages in requirements.items():
        f.write(f"# {category}\n")
        for pkg in packages:
            f.write(f"{pkg}\n")
        f.write("\n")

print(f"\n  requirements.txt 已保存: {req_path}")

# ============================================================
# 4. .gitignore 模板
# ============================================================
gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
dist/
build/
*.egg

# Jupyter Notebook
.ipynb_checkpoints

# 数据文件 (通常不上传大文件到 Git)
data/raw/*.csv
data/raw/*.parquet
data/processed/

# 模型文件 (大文件)
models/**/*.joblib
models/**/*.pkl
models/**/*.h5
*.joblib
*.pkl

# 环境变量
.env

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# 虚拟环境
venv/
.venv/
env/

# 日志
*.log
logs/

# MLflow
mlruns/
"""

gitignore_path = os.path.join("ml_project_template", ".gitignore")
with open(gitignore_path, 'w', encoding='utf-8') as f:
    f.write(gitignore_content)
print(f"  .gitignore 已保存: {gitignore_path}")

# ============================================================
# 5. 模型性能汇总脚本
# ============================================================
print("\n--- 5. 模型性能汇总 ---")


def generate_model_summary(models_results: list) -> str:
    """生成模型性能汇总报告"""
    print(f"\n  {'模型':<25} {'CV RMSE':<12} {'Test RMSE':<12} {'训练时间':<12} {'状态'}")
    print(f"  {'-'*75}")

    for result in models_results:
        status = "最佳" if result.get('is_best') else ""
        print(f"  {result['name']:<25} {result['cv_rmse']:<12.4f} "
              f"{result['test_rmse']:<12.4f} {result.get('train_time', 'N/A')!s:<12} {status}")

    best = min(models_results, key=lambda x: x['test_rmse'])
    summary = f"\n  最佳模型: {best['name']} (Test RMSE: {best['test_rmse']:.4f})"
    return summary


# 示例模型结果
model_results = [
    {"name": "LinearRegression", "cv_rmse": 0.7234, "test_rmse": 0.7156, "train_time": "0.5s"},
    {"name": "Ridge", "cv_rmse": 0.7230, "test_rmse": 0.7152, "train_time": "0.3s"},
    {"name": "Lasso", "cv_rmse": 0.7512, "test_rmse": 0.7423, "train_time": "0.8s"},
    {"name": "RandomForest", "cv_rmse": 0.5123, "test_rmse": 0.5034, "train_time": "15s"},
    {"name": "GradientBoosting", "cv_rmse": 0.4856, "test_rmse": 0.4789, "train_time": "25s"},
    {"name": "Stacking (Ridge+RF+GB)", "cv_rmse": 0.4723, "test_rmse": 0.4698,
     "train_time": "120s", "is_best": True},
]

summary = generate_model_summary(model_results)
print(summary)

# 保存性能报告
perf_report = {
    "project": "California Housing Price Prediction",
    "date": datetime.now().isoformat(),
    "models": model_results,
    "best_model": min(model_results, key=lambda x: x['test_rmse'])
}
report_path = os.path.join("ml_project_template", "model_performance.json")
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(perf_report, f, indent=2, ensure_ascii=False)
print(f"\n  性能报告已保存: {report_path}")

# ============================================================
# 6. 第一阶段学习成果总结
# ============================================================
print("\n--- 6. Stage 1 学习成果总结 ---")

stage1_summary = """
============================================================
        Stage 1 (W01-W08) 学习成果总结
============================================================

W01 - Python 基础
  [x] Python 基本语法、数据类型
  [x] 条件语句、循环、函数
  [x] 列表、字典、集合操作
  [x] 文件读写、异常处理

W02 - 数据科学基础
  [x] NumPy: 数组操作、广播、线性代数
  [x] Pandas: DataFrame 操作、数据清洗
  [x] Matplotlib: 基础绑图
  [x] Seaborn: 统计可视化

W03 - 数学基础
  [x] 线性代数: 向量、矩阵、特征分解、SVD
  [x] 微积分: 导数、梯度、链式法则
  [x] 概率统计: 分布、贝叶斯、假设检验

W04 - 机器学习入门
  [x] ML 概念框架: 监督/无监督/强化学习
  [x] 数据预处理: 缺失值、标准化、编码
  [x] 训练/验证/测试集划分
  [x] 评估指标: 准确率、精确率、召回率、F1

W05 - 监督学习 (回归)
  [x] 线性回归、多项式回归
  [x] 正则化: Ridge、Lasso
  [x] 评估指标: MSE、RMSE、R2
  [x] 交叉验证

W06 - 监督学习 (分类)
  [x] 逻辑回归
  [x] SVM (支持向量机)
  [x] 决策树、随机森林
  [x] KNN、朴素贝叶斯

W07 - 无监督学习 + 集成
  [x] K-Means 聚类
  [x] PCA 降维
  [x] Bagging (随机森林)
  [x] Boosting (AdaBoost, GBM)
  [x] Stacking 集成

W08 - 综合实战 + 总结
  [x] Kaggle EDA 全流程
  [x] 特征工程 Pipeline
  [x] 模型调优 (GridSearch/RandomSearch)
  [x] 模型保存 + 推理服务 (FastAPI/Flask)
  [x] ML 数学回顾
  [x] 深度学习预习 (感知机、激活函数)

总计:
  - 8 周学习内容
  - 56+ 个 Python 练习文件
  - 覆盖 ML 核心算法 15+
  - 从数学基础到模型部署全链路
============================================================
"""
print(stage1_summary)

# ============================================================
# 7. Git 工作流回顾
# ============================================================
print("\n--- 7. Git 工作流回顾 ---")
git_workflow = """
常用 Git 命令:

# 初始化
git init                          # 初始化仓库
git clone <url>                   # 克隆远程仓库

# 日常操作
git status                        # 查看状态
git add <file>                    # 添加到暂存区
git add .                         # 添加所有修改
git commit -m "message"           # 提交
git push origin main              # 推送到远程
git pull origin main              # 拉取最新代码

# 分支操作
git branch <name>                 # 创建分支
git checkout <name>               # 切换分支
git checkout -b <name>            # 创建并切换
git merge <name>                  # 合并分支
git branch -d <name>              # 删除分支

# 查看历史
git log --oneline                 # 简洁日志
git log --oneline --graph         # 图形化日志
git diff                          # 查看修改

# 撤销操作
git checkout -- <file>            # 撤销工作区修改
git reset HEAD <file>             # 取消暂存
git stash                         # 暂存当前修改
git stash pop                     # 恢复暂存

ML 项目 Git 最佳实践:
1. .gitignore 排除数据文件和模型文件
2. 使用 Git LFS 管理大文件
3. 提交信息规范: feat/fix/docs/refactor
4. 分支策略: main (稳定) + dev (开发) + feature/*
5. 每次 notebook 运行后清除输出再提交
"""
print(git_workflow)

# 创建 Git 提交模板
commit_template = """
# <type>: <subject>
#
# type 可选值:
#   feat     新功能
#   fix      修复bug
#   docs     文档更新
#   style    代码格式(不影响功能)
#   refactor 重构
#   test     测试相关
#   chore    构建/工具变更
#
# 示例:
#   feat: 添加特征工程Pipeline
#   fix: 修复数据加载编码问题
#   docs: 更新README中的安装说明
"""
commit_path = os.path.join("ml_project_template", ".gitmessage")
with open(commit_path, 'w', encoding='utf-8') as f:
    f.write(commit_template)
print(f"  Git 提交模板已保存: {commit_path}")

print("\n" + "=" * 60)
print("项目整理总结")
print("=" * 60)
print(f"""
1. 项目模板已创建: ml_project_template/
2. 包含: README.md, requirements.txt, .gitignore, .gitmessage
3. 目录结构遵循 ML 项目最佳实践
4. 所有模板文件均可直接使用和定制
""")
