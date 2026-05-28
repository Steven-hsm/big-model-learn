### Day 1（周一）：机器学习概论
# 画出偏差-方差权衡的示意图
import numpy as np
import matplotlib.pyplot as plt

# 模拟不同模型复杂度下的误差
complexity = np.arange(1, 20)
bias_sq = 10 / complexity       # 复杂度↑ → 偏差↓
variance = 0.5 * complexity     # 复杂度↑ →
total_error = bias_sq + variance

plt.figure(figsize=(8, 5))
plt.plot(complexity, bias_sq, 'b-', label='Bias²', linewidth=2)
plt.plot(complexity, variance, 'r-', label='Variance', linewidth=2)
plt.plot(complexity, total_error, 'g-', label='Total Error', linewidth=3)
plt.xlabel('Model Complexity')
plt.ylabel('Error')
plt.title('Bias-Variance Tradeoff')
plt.legend()
plt.axvline(x=complexity[np.argmin(total_error)], color='gray', linestyle='--')
plt.show()
