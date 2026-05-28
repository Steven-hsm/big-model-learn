"""
W08 Day 7: 学习总结 + 第二阶段预习
Stage 1 完整总结、自测题、深度学习时间线、神经网络从零实现
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Matplotlib 中文显示设置
# ============================================================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. Stage 1 完整总结
# ============================================================
print("=" * 60)
print("Stage 1 学习总结 + Stage 2 预习")
print("=" * 60)

print("""
+----------------------------------------------------------+
|              Stage 1: ML 基础 (8周) 学习总结              |
+----------------------------------------------------------+

  数学基础:
    - 线性代数: 向量空间、矩阵运算、特征分解、SVD
    - 微积分: 导数、偏导数、梯度、链式法则
    - 概率统计: 概率分布、贝叶斯定理、假设检验、MLE

  编程基础:
    - Python 语法、面向对象编程
    - NumPy (数值计算)、Pandas (数据处理)
    - Matplotlib / Seaborn (数据可视化)

  机器学习算法:
    回归: 线性回归、Ridge、Lasso、多项式回归
    分类: 逻辑回归、SVM、决策树、KNN、朴素贝叶斯
    集成: 随机森林、AdaBoost、GBM、Stacking
    无监督: K-Means、PCA

  工程实践:
    - EDA 全流程 (数据探索、可视化、特征分析)
    - 特征工程 (缺失值处理、编码、标准化、Pipeline)
    - 模型调优 (交叉验证、GridSearch、RandomSearch)
    - 模型部署 (joblib 保存、FastAPI/Flask 推理服务)
""")

# ============================================================
# 2. 关键技能清单
# ============================================================
print("\n--- 2. 关键技能自评清单 ---")

skills = [
    ("Python 编程基础", True),
    ("NumPy 数组操作与线性代数", True),
    ("Pandas 数据清洗与处理", True),
    ("Matplotlib/Seaborn 数据可视化", True),
    ("理解过拟合/欠拟合与偏差-方差权衡", True),
    ("数据预处理 Pipeline 构建", True),
    ("回归模型训练与评估", True),
    ("分类模型训练与评估", True),
    ("无监督学习 (聚类/降维)", True),
    ("集成学习方法", True),
    ("交叉验证与模型选择", True),
    ("超参数调优", True),
    ("特征工程与特征选择", True),
    ("模型保存与部署", True),
    ("Git 版本控制基础", True),
]

print(f"\n  {'技能':<35} {'掌握'}")
print(f"  {'-'*45}")
for skill, mastered in skills:
    status = "[v]" if mastered else "[ ]"
    print(f"  {skill:<35} {status}")

print(f"\n  总计: {sum(1 for _, m in skills if m)}/{len(skills)} 项已掌握")

# ============================================================
# 3. 自测题
# ============================================================
print("\n--- 3. 自测题 (答案在下方) ---")

quiz = [
    {
        "q": "Q1: 什么是偏差-方差权衡 (Bias-Variance Tradeoff)?",
        "a": "A1: 模型复杂度增加时, 偏差减小但方差增大."
             " 简单模型欠拟合(高偏差), 复杂模型过拟合(高方差)."
             " 目标是找到总误差最小的平衡点."
    },
    {
        "q": "Q2: L1 正则化(Lasso)和 L2 正则化(Ridge)的区别?",
        "a": "A2: L1 加入 |w| 惩罚, 产生稀疏解(特征选择); "
             "L2 加入 w^2 惩罚, 缩小权重但不为零. "
             "L1 适合高维稀疏数据, L2 适合共线性特征."
    },
    {
        "q": "Q3: 交叉验证的目的是什么? K折交叉验证如何工作?",
        "a": "A3: 目的是更可靠地评估模型泛化性能, 避免单次划分的随机性."
             " K折: 将数据分为K份, 轮流用K-1份训练、1份验证,"
             " 最终取K次结果的平均值."
    },
    {
        "q": "Q4: SVM 的核函数有什么作用?",
        "a": "A4: 核函数将数据映射到高维空间, 使线性不可分的数据变为可分."
             " 常用核: 线性核、多项式核、RBF(高斯)核."
             " 核技巧避免了显式计算高维映射."
    },
    {
        "q": "Q5: 随机森林和梯度提升的主要区别?",
        "a": "A5: 随机森林: Bagging, 多棵树独立并行训练, 降低方差."
             " 梯度提升: Boosting, 树串行训练, 每棵树修正前一棵的错误, 降低偏差."
             " RF 不易过拟合, GBM 精度更高但需仔细调参."
    },
    {
        "q": "Q6: PCA 降维的原理是什么?",
        "a": "A6: PCA 通过特征分解协方差矩阵, 找到方差最大的方向(主成分)."
             " 数据投影到前k个主成分上, 保留最多信息的同时降低维度."
             " 本质上是最小化重构误差的线性降维."
    },
    {
        "q": "Q7: 什么是梯度消失问题? 如何缓解?",
        "a": "A7: 深层网络中, 反向传播时梯度逐层相乘, sigmoid 的导数最大为0.25,"
             " 多层后梯度趋近于0, 导致浅层参数无法更新."
             " 缓解: 使用 ReLU 激活函数、残差连接、BatchNorm、合适的初始化."
    },
    {
        "q": "Q8: 信息熵和信息增益在决策树中的作用?",
        "a": "A8: 信息熵度量数据的不确定性: H(D) = -sum(p_k * log2(p_k))."
             " 信息增益是划分前后熵的减少量: Gain = H(D) - sum H(D_v)."
             " 决策树每次选择信息增益最大的特征进行划分."
    },
]

for item in quiz:
    print(f"\n  {item['q']}")

print("\n\n" + "  " + "=" * 50)
print("  答案:")
print("  " + "=" * 50)
for item in quiz:
    print(f"\n  {item['a']}")

# ============================================================
# 4. 深度学习历史时间线
# ============================================================
print("\n\n--- 4. 深度学习历史时间线 ---")

timeline = [
    ("1943", "McCulloch-Pitts 神经元模型", "第一个数学神经元模型"),
    ("1957", "Rosenblatt 感知机", "第一个可学习的单层神经网络"),
    ("1969", "Minsky 指出 XOR 问题", "第一次 AI 寒冬的开始"),
    ("1986", "Hinton 反向传播算法", "使多层网络训练成为可能"),
    ("1989", "LeNet (LeCun)", "第一个成功的卷积神经网络"),
    ("1997", "LSTM (Hochreiter)", "解决序列建模的长程依赖问题"),
    ("2006", "深度信念网络 (Hinton)", "深度学习复兴的起点"),
    ("2012", "AlexNet (ImageNet)", "深度学习爆发的标志时刻"),
    ("2014", "GAN (Goodfellow)", "生成对抗网络, AI 创造力的开始"),
    ("2015", "ResNet (He et al.)", "残差连接, 突破深度限制(152层)"),
    ("2017", "Transformer (Vaswani)", "Attention is All You Need"),
    ("2018", "BERT (Google)", "预训练语言模型, NLP 范式转变"),
    ("2020", "GPT-3 (OpenAI)", "1750亿参数, Few-shot 能力"),
    ("2022", "ChatGPT (OpenAI)", "大语言模型走向大众"),
    ("2023", "GPT-4 / LLaMA", "多模态 + 开源大模型"),
    ("2024-26", "AI Agent / 推理模型", "自主决策与复杂推理"),
]

print(f"\n  {'年份':<8} {'事件':<30} {'意义'}")
print(f"  {'-'*80}")
for year, event, meaning in timeline:
    print(f"  {year:<8} {event:<30} {meaning}")

# ============================================================
# 5. Stage 2 预习内容
# ============================================================
print("\n--- 5. Stage 2: 深度学习预习 ---")
print("""
Stage 2 预计学习路线 (深度学习 + 大模型):

  W09 - 深度学习基础
    - 神经网络原理 (前向传播、反向传播)
    - PyTorch 基础 (Tensor、Autograd)
    - 全连接网络 (MNIST 手写数字)

  W10 - CNN 卷积神经网络
    - 卷积、池化、步长、填充
    - 经典架构: LeNet、VGG、ResNet
    - 图像分类实战 (CIFAR-10)

  W11 - RNN 循环神经网络
    - RNN、LSTM、GRU
    - 序列建模 (文本分类、时间序列)
    - 注意力机制初步

  W12 - Transformer 与 NLP
    - Self-Attention 机制
    - Transformer 架构详解
    - 文本分类 / 情感分析实战

  W13 - 预训练模型与大语言模型
    - BERT、GPT 原理
    - HuggingFace Transformers 库
    - Prompt Engineering

  W14 - 生成模型
    - VAE (变分自编码器)
    - GAN (生成对抗网络)
    - Diffusion Model 基础

  W15 - 大模型应用开发
    - LangChain / LlamaIndex
    - RAG (检索增强生成)
    - AI Agent 基础

  W16 - 综合项目 + 部署
    - 端到端 ML 项目
    - Docker 容器化
    - MLOps 概述
""")

# ============================================================
# 6. 从零实现神经网络解决 XOR
# ============================================================
print("--- 6. 从零实现 2 层神经网络 (XOR) ---")


class NeuralNetwork:
    """2 层神经网络, 从零实现"""

    def __init__(self, layer_sizes, learning_rate=1.0):
        """
        layer_sizes: [输入维度, 隐藏层维度, 输出维度]
        """
        self.lr = learning_rate
        self.layers = len(layer_sizes) - 1
        np.random.seed(42)

        # Xavier 初始化
        self.weights = []
        self.biases = []
        for i in range(self.layers):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

        # 存储中间值
        self.activations = []
        self.z_values = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_deriv(self, s):
        return s * (1 - s)

    def forward(self, X):
        self.activations = [X]
        self.z_values = []

        a = X
        for i in range(self.layers):
            z = a @ self.weights[i] + self.biases[i]
            a = self.sigmoid(z)
            self.z_values.append(z)
            self.activations.append(a)

        return a

    def backward(self, y):
        m = y.shape[0]
        grads_w = [None] * self.layers
        grads_b = [None] * self.layers

        # 输出层误差
        delta = self.activations[-1] - y  # MSE 损失的梯度

        for i in range(self.layers - 1, -1, -1):
            grads_w[i] = self.activations[i].T @ delta / m
            grads_b[i] = np.sum(delta, axis=0, keepdims=True) / m

            if i > 0:
                delta = (delta @ self.weights[i].T) * self.sigmoid_deriv(self.activations[i])

        return grads_w, grads_b

    def train(self, X, y, epochs=10000, verbose=True):
        losses = []
        for epoch in range(epochs):
            # 前向传播
            output = self.forward(X)

            # 计算损失
            loss = np.mean((y - output) ** 2)
            losses.append(loss)

            # 反向传播
            grads_w, grads_b = self.backward(y)

            # 更新参数
            for i in range(self.layers):
                self.weights[i] -= self.lr * grads_w[i]
                self.biases[i] -= self.lr * grads_b[i]

            if verbose and (epoch + 1) % 2000 == 0:
                pred = (output > 0.5).astype(int)
                acc = np.mean(pred == y)
                print(f"  Epoch {epoch+1:5d} | Loss: {loss:.6f} | Acc: {acc:.2f}")

        return losses

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)


# XOR 数据
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_xor = np.array([[0], [1], [1], [0]], dtype=float)

# 训练 2 隐藏神经元的网络
print("\n训练 2 层神经网络 (2个隐藏神经元):")
nn = NeuralNetwork(layer_sizes=[2, 2, 1], learning_rate=2.0)
losses = nn.train(X_xor, y_xor, epochs=10000)

# 最终预测
final_pred = nn.predict(X_xor)
print(f"\n  最终预测: {final_pred.flatten()}")
print(f"  期望输出: {y_xor.flatten()}")
print(f"  正确率:   {np.mean(final_pred == y_xor) * 100:.0f}%")

# 打印学到的权重
print(f"\n  隐藏层权重 W1:\n{nn.weights[0].round(3)}")
print(f"  隐藏层偏置 b1: {nn.biases[0].round(3)}")
print(f"  输出层权重 W2: {nn.weights[1].round(3).flatten()}")
print(f"  输出层偏置 b2: {nn.biases[1].round(3).flatten()}")

# ============================================================
# 7. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 7.1 训练损失曲线
axes[0, 0].plot(losses, linewidth=0.5, color='steelblue', alpha=0.7)
axes[0, 0].set_xlabel('Epoch', fontsize=12)
axes[0, 0].set_ylabel('Loss (MSE)', fontsize=12)
axes[0, 0].set_title('训练损失曲线', fontsize=14)
axes[0, 0].grid(True, alpha=0.3)

# 7.2 XOR 决策边界
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200),
                      np.linspace(-0.5, 1.5, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = nn.forward(grid)
Z = Z.reshape(xx.shape)

axes[0, 1].contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.6)
axes[0, 1].contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
colors = ['red' if v == 0 else 'blue' for v in y_xor.flatten()]
axes[0, 1].scatter(X_xor[:, 0], X_xor[:, 1], c=colors,
                   s=300, edgecolors='black', zorder=5, linewidths=2)
for i, (x, y_val) in enumerate(zip(X_xor, y_xor.flatten())):
    axes[0, 1].annotate(f'{int(y_val)}', (x[0], x[1]),
                        textcoords="offset points", xytext=(10, 10),
                        fontsize=14, fontweight='bold')
axes[0, 1].set_title('XOR 决策边界 (2个隐藏神经元)', fontsize=14)
axes[0, 1].set_xlabel('x1')
axes[0, 1].set_ylabel('x2')

# 7.3 隐藏层表示
hidden = nn.sigmoid(X_xor @ nn.weights[0] + nn.biases[0])
axes[1, 0].scatter(hidden[:, 0], hidden[:, 1], c=colors,
                   s=300, edgecolors='black', zorder=5, linewidths=2)
for i in range(4):
    axes[1, 0].annotate(f'({int(X_xor[i,0])},{int(X_xor[i,1])})',
                        (hidden[i, 0], hidden[i, 1]),
                        textcoords="offset points", xytext=(10, 10), fontsize=11)
axes[1, 0].set_xlabel('隐藏神经元 1', fontsize=12)
axes[1, 0].set_ylabel('隐藏神经元 2', fontsize=12)
axes[1, 0].set_title('隐藏层特征空间 (线性可分!)', fontsize=14)
axes[1, 0].grid(True, alpha=0.3)

# 7.4 深度学习时间线
years = [int(t[0]) for t in timeline]
events_short = [t[1] for t in timeline]
y_positions = np.arange(len(timeline))

colors_timeline = plt.cm.viridis(np.linspace(0.1, 0.9, len(timeline)))
axes[1, 1].barh(y_positions, [1]*len(timeline), color=colors_timeline,
                edgecolor='white', height=0.8)
for i, (year, event, _) in enumerate(timeline):
    axes[1, 1].text(0.5, i, f'{year} - {event}', ha='center', va='center',
                    fontsize=8, fontweight='bold', color='white')
axes[1, 1].set_yticks(y_positions)
axes[1, 1].set_yticklabels([t[0] for t in timeline])
axes[1, 1].set_xlim(0, 1)
axes[1, 1].set_xticks([])
axes[1, 1].set_title('深度学习历史时间线', fontsize=14)
axes[1, 1].invert_yaxis()

plt.tight_layout()
plt.savefig('d7_summary_visualization.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[图表已保存] d7_summary_visualization.png")

# ============================================================
# 8. 额外: 不同隐藏神经元数量对比
# ============================================================
print("\n--- 8. 隐藏神经元数量对比 ---")

fig, axes = plt.subplots(1, 4, figsize=(20, 4))
hidden_sizes = [1, 2, 4, 8]

for idx, h_size in enumerate(hidden_sizes):
    nn_test = NeuralNetwork(layer_sizes=[2, h_size, 1], learning_rate=2.0)
    nn_test.train(X_xor, y_xor, epochs=10000, verbose=False)
    pred = nn_test.predict(X_xor)
    acc = np.mean(pred == y_xor) * 100

    Z = nn_test.forward(grid).reshape(xx.shape)
    axes[idx].contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.6)
    axes[idx].contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    axes[idx].scatter(X_xor[:, 0], X_xor[:, 1], c=colors,
                     s=200, edgecolors='black', zorder=5)
    axes[idx].set_title(f'隐藏神经元={h_size}\n准确率={acc:.0f}%', fontsize=12)
    axes[idx].set_xlabel('x1')
    axes[idx].set_ylabel('x2')

plt.suptitle('不同隐藏神经元数量对 XOR 问题的影响', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig('d7_hidden_neurons_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("[图表已保存] d7_hidden_neurons_comparison.png")

# ============================================================
# 9. 最终总结
# ============================================================
print("\n" + "=" * 60)
print("Stage 1 完成! 恭喜!")
print("=" * 60)
print("""
  你已经完成了 Machine Learning 基础阶段的全部学习!

  核心收获:
    - 扎实的数学基础 (线性代数、微积分、概率统计)
    - Python 数据科学全栈 (NumPy, Pandas, Matplotlib)
    - 15+ ML 算法的原理与实践
    - 从数据分析到模型部署的完整流程
    - 从零实现神经网络并解决 XOR 问题

  下一步 (Stage 2):
    - PyTorch 深度学习框架
    - CNN (图像)、RNN/Transformer (序列)
    - 预训练模型与大语言模型
    - AI 应用开发 (RAG, Agent)

  记住:
    "The best way to learn is by doing."
    继续动手实践, 你已经在通往 AI 工程师的路上了!
""")
