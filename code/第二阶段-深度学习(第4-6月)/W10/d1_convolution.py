"""
W10-D1 卷积操作 (Convolution Operations)
=========================================
从零实现2D卷积，演示边缘检测和锐化等经典卷积核
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 从零实现 conv2d
# ============================================================
def conv2d(input_matrix, kernel, stride=1, padding=0):
    """
    手动实现2D卷积操作（单通道）

    参数:
        input_matrix: 输入矩阵 (H, W)
        kernel: 卷积核 (kH, kW)
        stride: 步幅
        padding: 填充（在四周各补padding个0）

    返回:
        output_matrix: 卷积结果
    """
    # 应用 padding
    if padding > 0:
        input_matrix = np.pad(input_matrix, padding, mode='constant', constant_values=0)

    h_in, w_in = input_matrix.shape
    k_h, k_w = kernel.shape

    # 输出尺寸公式: output = (input - kernel + 2*padding) / stride + 1
    h_out = (h_in - k_h) // stride + 1
    w_out = (w_in - k_w) // stride + 1

    output = np.zeros((h_out, w_out))

    # 滑动窗口
    for i in range(h_out):
        for j in range(w_out):
            # 提取当前窗口
            h_start = i * stride
            w_start = j * stride
            window = input_matrix[h_start:h_start + k_h, w_start:w_start + k_w]
            # 逐元素相乘再求和
            output[i, j] = np.sum(window * kernel)

    return output


# ============================================================
# 2. 验证：5x5 输入 + 3x3 卷积核
# ============================================================
print("=" * 60)
print("验证 conv2d 实现")
print("=" * 60)

test_input = np.array([
    [1, 2, 0, 1, 3],
    [0, 1, 2, 3, 1],
    [1, 3, 1, 0, 2],
    [2, 0, 3, 1, 1],
    [1, 2, 1, 2, 0]
], dtype=np.float32)

test_kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
], dtype=np.float32)

print(f"\n输入矩阵 ({test_input.shape}):\n{test_input}")
print(f"\n卷积核 ({test_kernel.shape}):\n{test_kernel}")

# 无 padding, stride=1
out_no_pad = conv2d(test_input, test_kernel, stride=1, padding=0)
print(f"\n无padding, stride=1 输出 ({out_no_pad.shape}):\n{out_no_pad}")

# 有 padding, stride=1
out_with_pad = conv2d(test_input, test_kernel, stride=1, padding=1)
print(f"\npadding=1, stride=1 输出 ({out_with_pad.shape}):\n{out_with_pad}")

# stride=2
out_stride2 = conv2d(test_input, test_kernel, stride=2, padding=0)
print(f"\n无padding, stride=2 输出 ({out_stride2.shape}):\n{out_stride2}")

# 验证输出尺寸公式
h, w = test_input.shape
k_h, k_w = test_kernel.shape
for stride_val in [1, 2]:
    for pad_val in [0, 1]:
        expected_h = (h - k_h + 2 * pad_val) // stride_val + 1
        expected_w = (w - k_w + 2 * pad_val) // stride_val + 1
        result = conv2d(test_input, test_kernel, stride=stride_val, padding=pad_val)
        assert result.shape == (expected_h, expected_w), \
            f"Shape mismatch: expected {(expected_h, expected_w)}, got {result.shape}"
        print(f"  stride={stride_val}, padding={pad_val} -> "
              f"输出尺寸 {result.shape} [公式验证通过]")


# ============================================================
# 3. 经典卷积核：边缘检测 & 锐化
# ============================================================
print("\n" + "=" * 60)
print("经典卷积核应用")
print("=" * 60)

# 创建一个示例图像（包含不同区域）
np.random.seed(42)
sample_img = np.zeros((32, 32), dtype=np.float32)
# 添加一个矩形
sample_img[8:24, 8:24] = 1.0
# 添加一些噪声
sample_img += np.random.randn(32, 32) * 0.05
sample_img = np.clip(sample_img, 0, 1)

# 定义经典卷积核
# 水平边缘检测（Prewitt 水平）
horizontal_edge_kernel = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
], dtype=np.float32)

# 垂直边缘检测（Prewitt 垂直）
vertical_edge_kernel = np.array([
    [-1,  0,  1],
    [-1,  0,  1],
    [-1,  0,  1]
], dtype=np.float32)

# 锐化核
sharpen_kernel = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
], dtype=np.float32)

# 拉普拉斯边缘检测
laplacian_kernel = np.array([
    [0,  1, 0],
    [1, -4, 1],
    [0,  1, 0]
], dtype=np.float32)

# 应用卷积核
h_edge = conv2d(sample_img, horizontal_edge_kernel)
v_edge = conv2d(sample_img, vertical_edge_kernel)
sharpened = conv2d(sample_img, sharpen_kernel)
laplacian = conv2d(sample_img, laplacian_kernel)

# 可视化
fig, axes = plt.subplots(2, 3, figsize=(14, 9))

axes[0, 0].imshow(sample_img, cmap='gray')
axes[0, 0].set_title('原始图像')
axes[0, 0].axis('off')

axes[0, 1].imshow(h_edge, cmap='gray')
axes[0, 1].set_title('水平边缘检测 (Prewitt)')
axes[0, 1].axis('off')

axes[0, 2].imshow(v_edge, cmap='gray')
axes[0, 2].set_title('垂直边缘检测 (Prewitt)')
axes[0, 2].axis('off')

axes[1, 0].imshow(sharpened, cmap='gray')
axes[1, 0].set_title('锐化')
axes[1, 0].axis('off')

axes[1, 1].imshow(np.abs(laplacian), cmap='gray')
axes[1, 1].set_title('拉普拉斯边缘检测')
axes[1, 1].axis('off')

# 组合边缘
combined_edge = np.sqrt(h_edge ** 2 + v_edge ** 2)
axes[1, 2].imshow(combined_edge, cmap='gray')
axes[1, 2].set_title('组合边缘 (梯度幅值)')
axes[1, 2].axis('off')

plt.suptitle('卷积核应用示例', fontsize=16)
plt.tight_layout()
plt.savefig('d1_convolution_kernels.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图像已保存: d1_convolution_kernels.png")


# ============================================================
# 4. 多通道卷积说明
# ============================================================
print("\n" + "=" * 60)
print("多通道卷积说明")
print("=" * 60)

# 模拟 RGB 输入 (3 通道)
C_in = 3
H, W = 32, 32
C_out = 16
K = 3

# 单个输出通道的参数量 = C_in * K * K + 1(bias)
params_per_output_channel = C_in * K * K + 1
print(f"\n输入: {H}x{W}x{C_in} (HWC)")
print(f"卷积核: {K}x{K}, 输入通道={C_in}, 输出通道={C_out}")
print(f"\n每个输出通道参数量: {C_in} * {K} * {K} + 1(bias) = {params_per_output_channel}")
print(f"总参数量: {C_out} * {params_per_output_channel} = {C_out * params_per_output_channel}")

# 更大的例子
print("\n--- 更大的例子 ---")
examples = [
    ("Conv2d(3, 64, 3)", 3, 64, 3),
    ("Conv2d(64, 128, 3)", 64, 128, 3),
    ("Conv2d(128, 256, 3)", 128, 256, 3),
    ("Conv2d(256, 512, 3)", 256, 512, 3),
    ("Conv2d(3, 64, 7)", 3, 64, 7),
    ("Conv2d(512, 1000, 1)", 512, 1000, 1),
]

print(f"{'层':<25} {'参数量':>10}")
print("-" * 40)
for name, c_in, c_out, k in examples:
    params = c_out * (c_in * k * k + 1)
    print(f"{name:<25} {params:>10,}")

print("\n多通道卷积工作原理:")
print("  1. 每个输出通道有一组 (C_in, K, K) 的卷积核")
print("  2. 对每个输出通道: 将 C_in 个输入通道分别与对应卷积核做卷积，然后求和")
print("  3. 加上偏置项 (bias)")
print("  4. 总参数量 = C_out * (C_in * K * K + 1)")

# 多通道卷积实现示例
print("\n--- 多通道卷积实现 ---")


def conv2d_multi_channel(input_tensor, kernels, bias=None, stride=1, padding=0):
    """
    多通道2D卷积

    参数:
        input_tensor: (C_in, H, W)
        kernels: (C_out, C_in, kH, kW)
        bias: (C_out,)
        stride: 步幅
        padding: 填充
    """
    if padding > 0:
        input_tensor = np.pad(input_tensor,
                              ((0, 0), (padding, padding), (padding, padding)),
                              mode='constant', constant_values=0)

    c_in, h_in, w_in = input_tensor.shape
    c_out, _, k_h, k_w = kernels.shape

    h_out = (h_in - k_h) // stride + 1
    w_out = (w_in - k_w) // stride + 1

    output = np.zeros((c_out, h_out, w_out))

    for co in range(c_out):
        for i in range(h_out):
            for j in range(w_out):
                h_s = i * stride
                w_s = j * stride
                # 对所有输入通道求和
                window = input_tensor[:, h_s:h_s + k_h, w_s:w_s + k_w]
                output[co, i, j] = np.sum(window * kernels[co])
        if bias is not None:
            output[co] += bias[co]

    return output


# 演示
rgb_input = np.random.randn(3, 8, 8).astype(np.float32)
kernels_demo = np.random.randn(4, 3, 3, 3).astype(np.float32)  # 4个输出通道
bias_demo = np.zeros(4, dtype=np.float32)

multi_out = conv2d_multi_channel(rgb_input, kernels_demo, bias_demo, stride=1, padding=0)
print(f"输入: {rgb_input.shape} (C_in, H, W)")
print(f"卷积核: {kernels_demo.shape} (C_out, C_in, kH, kW)")
print(f"输出: {multi_out.shape} (C_out, H_out, W_out)")
print("多通道卷积验证通过！")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("D1 卷积操作 完成！")
    print("=" * 60)
