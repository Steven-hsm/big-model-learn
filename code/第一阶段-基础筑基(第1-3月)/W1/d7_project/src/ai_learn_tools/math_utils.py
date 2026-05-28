import numpy as np

def cosine_similarity(a, b):
    """计算两个向量的余弦相似度"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 使用
from ai_learn_tools import cosine_similarity
from ai_learn_tools.math_utils import cosine_similarity  # 等效
from ai_learn_tools import math_utils as mu      