# big-model-learn

AI 开发工程师一年学习计划（39 周）的完整代码仓库，从零基础到 AI 高级开发工程师。

## 学习路线

```
第一阶段：基础筑基（第 1-3 月）
│
├── W01  Python 基础与环境搭建
├── W02  NumPy 与 Pandas 数据处理
├── W03  线性代数（AI 核心）
├── W04  概率统计与微积分（AI 核心）
├── W05  机器学习概论与线性模型
├── W06  分类算法与特征工程
├── W07  无监督学习与数据 Pipeline
└── W08  ML 综合实战与阶段总结
│
第二阶段：深度学习（第 4-6 月）
│
├── W09  神经网络原理
├── W10  卷积神经网络 CNN
├── W11  RNN 与序列模型
├── W12  PyTorch 进阶
├── W13  Transformer 原理深入
├── W14  Transformer 从零实现
���── W15  HuggingFace 生态
└── W16  高级训练技术
│
第三阶段：大模型工程（第 7-9 月）
│
├── W17  大语言模型基础
├── W18  RAG 检索增强生成基础
├── W19  RAG 进阶与优化
├── W20  模型微调 LoRA
├── W21  模型评估与对齐
├── W22  AI Agent 开发
├── W23  Multi-Agent 系统
└── W24  模型部署与推理优化
│
第四阶段：高级实战（��� 10-12 月）
│
├── W25  数据工程与特征管理
├── W26  实验管理与模型注册
├── W27  模型监控与 CI/CD
├── W28  MLOps 综合实战
├── W29-34  大型综合项目（企业级 RAG 系统）
├── W35  开源贡献
├── W36  技术博客与个人品牌
└── W37-39  项目打磨与求职准备
```

## 目录结构

```
big-model-learn/
├── code/q_01/
│   ├── 第一阶段-基础筑基(第1-3月)/      # W01 - W08
│   │   ├── W1/  d1~d7 + project
│   │   ├── W2/  d1~d7
│   │   ├── W3/  d1~d7
│   │   ├── W4/  d1~d7
│   │   ├── W5/  d1~d7
│   │   ├── W6/  d1~d7
│   │   ├── W7/  d1~d7
│   │   └── W8/  d1~d7
│   ├── 第二阶段-深度学习(第4-6月)/      # W09 - W16
│   │   ├── W9/  d1~d7
│   │   └── ... W10-W16
│   ├── 第三阶段-大模型工程(第7-9月)/    # W17 - W24
│   │   ├── W17/ d1~d7
│   │   └── ... W18-W24
│   └── 第四阶段-高级实战(第10-12月)/    # W25 - W39
│       ├── W25/ d1~d7
│       └── ... W26-W39
└── 学习计划/                           # 详细学习计划文档
    └── 周计划/
        ├── 第一阶段-基础筑基(第1-3月)/
        ├── 第二阶段-深度学习(第4-6月)/
        ├── 第三阶段-大模型工程(第7-9月)/
        └── 第四阶段-高级实战(第10-12月)/
```

## 代码说明

- 每周 7 个练习文件（d1 ~ d7），对应每天的学习内容
- 所有代码均可独立运行，包含完整实现和中文注释
- 工作日晚每天 2 小时，周末每天 6-8 小时
- 共 **284 个 Python 文件**，覆盖完整学习路径

## 技术栈

| 阶段 | 核心技术 |
|------|---------|
| 基础筑基 | Python, NumPy, Pandas, Matplotlib, Seaborn, scikit-learn |
| 深度学习 | PyTorch, CNN/RNN/Transformer, HuggingFace Transformers |
| 大模型工程 | LLM, RAG, LoRA/QLoRA, Agent, vLLM, ONNX, FastAPI |
| 高级实战 | MLOps, MLflow, Docker, CI/CD, GitHub Actions |

## 环境配置

```bash
# 创建 conda 环境
conda create -n ai-learn python=3.12 -y
conda activate ai-learn

# 安装基础依赖
pip install numpy pandas matplotlib seaborn scipy scikit-learn jupyter

# 深度学习阶段额外安装
pip install torch torchvision datasets tqdm

# 大模型阶段额外安装
pip install transformers peft accelerate sentence-transformers
pip install faiss-cpu langchain chromadb gradio

# MLOps 阶段额外安装
pip install mlflow optuna joblib
```

## 学习进度

- [x] 第一阶段：基础筑基（W01-W08）
- [ ] 第二阶段：深度学习（W09-W16）
- [ ] 第三阶段：大模型工程（W17-W24）
- [ ] 第四阶段：高级实战（W25-W39）

## License

MIT
