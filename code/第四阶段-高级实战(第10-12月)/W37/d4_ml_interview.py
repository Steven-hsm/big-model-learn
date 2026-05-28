"""
W37 Day 4 - ML面试题
====================
主题: 概念题(过拟合/正则化/评估), 手撕代码题, 数学推导题
"""

import os
import math
from datetime import datetime


def print_section(title):
    """打印分隔线"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# =============================================
# 1. ML核心概念题
# =============================================
def ml_concept_questions():
    """ML核心概念题"""
    print_section("ML核心概念题")

    questions = {
        "过拟合与正则化": {
            "Q: 什么是过拟合? 列举5种防止过拟合的方法": [
                "1. 正则化(L1/L2): 在损失函数中加入惩罚项",
                "2. Dropout: 训练时随机丢弃部分神经元",
                "3. 数据增强: 通过变换扩充训练数据",
                "4. 早停(Early Stopping): 验证集性能不再提升时停止",
                "5. 模型简化: 减少层数/参数量"
            ],
            "Q: L1和L2正则化的区别?": [
                "L1 (Lasso): lambda * |w|, 产生稀疏解, 可做特征选择",
                "L2 (Ridge): lambda * w^2, 权重趋向小的值但不会为0",
                "L1的梯度在0附近不连续, L1有特征选择效果",
                "L2可微, 优化更稳定, 更常用"
            ],
            "Q: 为什么L1正则化会产生稀疏解?": [
                "L1在0点不可导, 优化容易在0点停留",
                "L1的等高线是菱形, 容易与损失函数在角上相交",
                "角点意味着某些维度为0 -> 稀疏"
            ]
        },
        "模型评估": {
            "Q: 分类评估指标有哪些?": [
                "混淆矩阵: TP, FP, TN, FN",
                "Accuracy = (TP+TN) / (TP+FP+TN+FN)",
                "Precision = TP / (TP+FP)",
                "Recall = TP / (TP+FN)",
                "F1 = 2*P*R/(P+R)",
                "AUC-ROC: 分类阈值无关的评估"
            ],
            "Q: AUC-ROC的含义?": [
                "ROC曲线: TPR vs FPR在不同阈值下的曲线",
                "AUC: ROC曲线下面积, 衡量分类器整体性能",
                "AUC=1: 完美分类器",
                "AUC=0.5: 随机猜测",
                "AUC对样本不平衡不敏感"
            ],
            "Q: 回归评估指标有哪些?": [
                "MSE = mean((y - y_pred)^2)",
                "RMSE = sqrt(MSE)",
                "MAE = mean(|y - y_pred|)",
                "R^2 = 1 - SS_res/SS_tot",
                "MAPE = mean(|y - y_pred| / |y|) * 100%"
            ]
        },
        "优化算法": {
            "Q: SGD vs Adam的区别?": [
                "SGD: 固定学习率, 简单但需要调学习率",
                "Adam: 自适应学习率, 结合动量和自适应",
                "Adam通常收敛更快, SGD泛化可能更好",
                "AdamW: 修正Adam的权重衰减"
            ],
            "Q: 什么是学习率预热(Warmup)?": [
                "训练初期用小学习率, 逐步增大到目标学习率",
                "防止初期大梯度导致训练不稳定",
                "常见于Transformer训练: 线性预热 + 余弦衰减"
            ]
        }
    }

    for category, items in questions.items():
        print(f"\n【{category}】")
        for question, answers in items.items():
            print(f"\n  {question}")
            for answer in answers:
                print(f"    {answer}")


# =============================================
# 2. 手撕代码题
# =============================================
def coding_implementations():
    """手撕ML代码实现"""
    print_section("手撕代码题")

    # 2.1 余弦相似度
    print("\n--- 1. 余弦相似度实现 ---")
    cos_sim_code = '''
def cosine_similarity(a, b):
    """计算两个向量的余弦相似度"""
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x ** 2 for x in a))
    norm_b = math.sqrt(sum(x ** 2 for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

# 测试
a = [1, 2, 3]
b = [4, 5, 6]
print(f"余弦相似度: {cosine_similarity(a, b):.4f}")
# 输出: 余弦相似度: 0.9746
'''
    print(cos_sim_code)
    a = [1, 2, 3]
    b = [4, 5, 6]
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x**2 for x in a))
    norm_b = math.sqrt(sum(x**2 for x in b))
    result = dot / (norm_a * norm_b)
    print(f"  实际运行结果: {result:.4f}")

    # 2.2 Softmax
    print("\n--- 2. Softmax函数实现 ---")
    softmax_code = '''
def softmax(logits):
    """数值稳定的Softmax实现"""
    # 减去最大值防止溢出
    max_val = max(logits)
    exp_vals = [math.exp(x - max_val) for x in logits]
    total = sum(exp_vals)
    return [x / total for x in exp_vals]

# 测试
logits = [2.0, 1.0, 0.1]
probs = softmax(logits)
print(f"Softmax: {probs}")
# 输出: Softmax: [0.6590, 0.2424, 0.0986]
'''
    print(softmax_code)
    logits = [2.0, 1.0, 0.1]
    max_val = max(logits)
    exp_vals = [math.exp(x - max_val) for x in logits]
    total = sum(exp_vals)
    probs = [x / total for x in exp_vals]
    print(f"  实际运行结果: {[f'{p:.4f}' for p in probs]}")

    # 2.3 TF-IDF
    print("\n--- 3. TF-IDF实现 ---")
    tfidf_code = '''
import math

def compute_tf(word, doc):
    """计算词频TF"""
    words = doc.split()
    return words.count(word) / len(words)

def compute_idf(word, docs):
    """计算逆文档频率IDF"""
    n_docs = len(docs)
    n_containing = sum(1 for doc in docs if word in doc.split())
    return math.log((n_docs + 1) / (n_containing + 1)) + 1

def compute_tfidf(word, doc, docs):
    """计算TF-IDF"""
    return compute_tf(word, doc) * compute_idf(word, docs)

# 测试
docs = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "the cat and the dog"
]
for word in ["cat", "dog", "sat", "the"]:
    tfidf = compute_tfidf(word, docs[0], docs)
    print(f"  '{word}' in doc1: TF-IDF = {tfidf:.4f}")
'''
    print(tfidf_code)

    # 2.4 简单的文本分块
    print("\n--- 4. 文本分块实现 ---")
    chunk_code = '''
def split_text(text, chunk_size=100, overlap=20):
    """将文本分割为重叠的块"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # 重叠部分
    return chunks

# 测试
text = "人工智能是计算机科学的一个分支..." * 10
chunks = split_text(text, chunk_size=50, overlap=10)
print(f"文本长度: {len(text)}, 分块数: {len(chunks)}")
'''
    print(chunk_code)

    # 2.5 简单的前向传播
    print("\n--- 5. 简单神经网络前向传播 ---")
    nn_code = '''
import math

def relu(x):
    return max(0, x)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def forward(x, weights, biases):
    """简单的前向传播(2层)"""
    # 隐藏层
    hidden = []
    for i in range(len(weights[0][0])):
        z = sum(x[j] * weights[0][j][i] for j in range(len(x))) + biases[0][i]
        hidden.append(relu(z))

    # 输出层
    output = []
    for i in range(len(weights[1][0])):
        z = sum(hidden[j] * weights[1][j][i] for j in range(len(hidden))) + biases[1][i]
        output.append(sigmoid(z))

    return output

# 测试
x = [0.5, 0.3]
# 权重: 输入层(2) -> 隐藏层(3) -> 输出层(2)
weights = [
    [[0.1, 0.2, -0.1], [0.3, -0.2, 0.1]],  # 输入->隐藏
    [[0.2, -0.1], [0.1, 0.3], [-0.2, 0.1]]  # 隐藏->输出
]
biases = [[0.1, 0.1, 0.1], [0.1, 0.1]]
output = forward(x, weights, biases)
print(f"输出: {[f'{o:.4f}' for o in output]}")
'''
    print(nn_code)


# =============================================
# 3. 数学推导题
# =============================================
def math_derivations():
    """数学推导题"""
    print_section("数学推导题")

    derivations = {
        "1. Softmax + Cross-Entropy Loss梯度推导": {
            "前向传播": [
                "z_i = w_i * x + b_i  (logits)",
                "p_i = exp(z_i) / sum(exp(z_j))  (softmax)",
                "L = -sum(y_i * log(p_i))  (交叉熵)"
            ],
            "梯度推导": [
                "dL/dz_i = p_i - y_i",
                "推导: dL/dz_i = dL/dp_i * dp_i/dz_i",
                "     = -y_i/p_i * p_i(1-p_i) + sum_{j!=i} y_j/p_j * p_i*p_j",
                "     = -y_i(1-p_i) + p_i * sum_{j!=i} y_j",
                "     = -y_i + y_i*p_i + p_i*(1-y_i) = p_i - y_i",
                "结论: Softmax+CE的梯度非常简洁: pred - target"
            ]
        },
        "2. 反向传播推导": {
            "链式法则": [
                "dL/dw = dL/da * da/dz * dz/dw",
                "其中: a = activation(z), z = w*x + b"
            ],
            "关键公式": [
                "dL/da: 来自上一层的梯度",
                "da/dz: 激活函数的导数",
                "  - ReLU: dz * (z > 0)",
                "  - Sigmoid: sigma(z) * (1 - sigma(z))",
                "dz/dw = x (输入)"
            ]
        },
        "3. Attention公式推导": {
            "Scaled Dot-Product Attention": [
                "Q = X * W_q, K = X * W_k, V = X * W_v",
                "Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V",
                "除以sqrt(d_k)的原因: 防止点积过大导致softmax梯度消失",
                "当d_k较大时, 点积的方差也大, softmax趋向one-hot"
            ],
            "Multi-Head Attention": [
                "head_i = Attention(QW_q_i, KW_k_i, VW_v_i)",
                "MultiHead = Concat(head_1, ..., head_h) * W_o",
                "每个头在不同的子空间中学习注意力模式"
            ]
        },
        "4. 梯度下降推导": {
            "基本形式": [
                "theta_new = theta - lr * gradient",
                "梯度: dL/dtheta, 损失函数对参数的导数",
                "负梯度方向是函数值下降最快的方向"
            ],
            "变体": [
                "SGD: theta -= lr * grad",
                "Momentum: v = beta*v + grad; theta -= lr*v",
                "Adam: 一阶矩m和二阶矩v的自适应估计"
            ]
        },
        "5. KL散度推导": {
            "定义": [
                "KL(P||Q) = sum(p_i * log(p_i / q_i))",
                "衡量分布P和Q的差异",
                "非对称: KL(P||Q) != KL(Q||P)",
                "非负: KL(P||Q) >= 0, 等号当且仅当P=Q"
            ],
            "与交叉熵的关系": [
                "H(P, Q) = -sum(p_i * log(q_i))",
                "KL(P||Q) = H(P, Q) - H(P)",
                "交叉熵 = KL散度 + 熵",
                "最小化交叉熵等价于最小化KL散度(因为H(P)固定)"
            ]
        }
    }

    for title, details in derivations.items():
        print(f"\n【{title}】")
        for sub_title, steps in details.items():
            print(f"\n  {sub_title}:")
            for step in steps:
                print(f"    {step}")


# =============================================
# 主程序
# =============================================
if __name__ == "__main__":
    print("=" * 60)
    print("  W37 Day 4 - ML面试题")
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    ml_concept_questions()
    coding_implementations()
    math_derivations()

    print("\n" + "=" * 60)
    print("  理解数学推导能让你在面试中脱颖而出!")
    print("  建议: 用自己的话推导一遍, 不要死记公式")
    print("=" * 60)
