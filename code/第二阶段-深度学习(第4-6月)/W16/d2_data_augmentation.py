"""
W16-D2 数据增强 (Data Augmentation)
=====================================
文本增强 (同义词替换, 随机删除, 回译概念),
图像增强 (Mixup, CutMix), 展示增强样本
"""

import random
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("W16-D2 数据增强 (Data Augmentation)")
print("=" * 60)

# ============================================================
# 1. 数据增强概述
# ============================================================
print("\n--- 1. 数据增强概述 ---")
print("""
  数据增强的目的:
    - 扩大训练数据量 (生成变体)
    - 提高模型泛化能力
    - 防止过拟合
    - 处理数据不平衡

  文本增强方法:
    - 同义词替换 (Synonym Replacement)
    - 随机删除 (Random Deletion)
    - 随机交换 (Random Swap)
    - 随机插入 (Random Insertion)
    - 回译 (Back Translation)

  图像增强方法:
    - 几何变换 (翻转, 旋转, 缩放)
    - 颜色变换 (亮度, 对比度, 饱和度)
    - Mixup:    两张图混合
    - CutMix:   裁切并混合区域
""")

# ============================================================
# 2. 文本增强: 同义词替换
# ============================================================
print("\n--- 2. 文本增强: 同义词替换 ---")

# 简易同义词字典 (实际项目中可用 WordNet 或 embedding 相似度)
SYNONYM_DICT = {
    "good": ["great", "excellent", "fine", "wonderful", "nice"],
    "bad": ["terrible", "awful", "poor", "horrible", "dreadful"],
    "happy": ["glad", "pleased", "joyful", "cheerful", "delighted"],
    "sad": ["unhappy", "sorrowful", "melancholy", "gloomy"],
    "big": ["large", "huge", "enormous", "vast", "massive"],
    "small": ["tiny", "little", "miniature", "compact"],
    "fast": ["quick", "rapid", "swift", "speedy"],
    "slow": ["sluggish", "gradual", "unhurried"],
    "beautiful": ["gorgeous", "stunning", "lovely", "attractive"],
    "smart": ["intelligent", "clever", "brilliant", "wise"],
    "love": ["adore", "enjoy", "appreciate", "cherish"],
    "hate": ["despise", "detest", "loathe", "dislike"],
    "important": ["significant", "crucial", "vital", "essential"],
    "easy": ["simple", "straightforward", "effortless"],
    "hard": ["difficult", "challenging", "tough", "demanding"],
}


def synonym_replacement(text, replace_ratio=0.3):
    """同义词替换: 随机选择一些词替换为其同义词"""
    words = text.split()
    n_replace = max(1, int(len(words) * replace_ratio))
    new_words = words.copy()

    # 找到可以替换的词
    replaceable = [i for i, w in enumerate(words) if w.lower() in SYNONYM_DICT]
    if not replaceable:
        return text

    # 随机选择要替换的位置
    to_replace = random.sample(replaceable, min(n_replace, len(replaceable)))

    for idx in to_replace:
        word = words[idx].lower()
        synonyms = SYNONYM_DICT[word]
        new_word = random.choice(synonyms)
        # 保持原词大小写
        if words[idx][0].isupper():
            new_word = new_word.capitalize()
        new_words[idx] = new_word

    return " ".join(new_words)


# 演示
original_text = "This is a good and beautiful movie with smart actors"
random.seed(42)

print(f"  原文: {original_text}")
print(f"  增强变体:")
for i in range(5):
    augmented = synonym_replacement(original_text, replace_ratio=0.3)
    print(f"    [{i+1}] {augmented}")

# ============================================================
# 3. 文本增强: 随机删除
# ============================================================
print("\n--- 3. 文本增强: 随机删除 ---")


def random_deletion(text, delete_prob=0.15):
    """随机删除: 以一定概率删除每个词"""
    words = text.split()
    if len(words) <= 1:
        return text

    new_words = [w for w in words if random.random() > delete_prob]
    if not new_words:
        # 至少保留一个词
        new_words = [random.choice(words)]

    return " ".join(new_words)


print(f"  原文: {original_text}")
print(f"  随机删除变体 (p=0.15):")
for i in range(5):
    augmented = random_deletion(original_text, delete_prob=0.15)
    deleted_count = len(original_text.split()) - len(augmented.split())
    print(f"    [{i+1}] {augmented}  (删除了 {deleted_count} 个词)")

# ============================================================
# 4. 文本增强: 随机交换
# ============================================================
print("\n--- 4. 文本增强: 随机交换 ---")


def random_swap(text, n_swaps=2):
    """随机交换: 随机交换两个词的位置"""
    words = text.split()
    if len(words) <= 1:
        return text

    new_words = words.copy()
    for _ in range(n_swaps):
        idx1, idx2 = random.sample(range(len(new_words)), 2)
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]

    return " ".join(new_words)


print(f"  原文: {original_text}")
print(f"  随机交换变体:")
for i in range(5):
    augmented = random_swap(original_text, n_swaps=2)
    print(f"    [{i+1}] {augmented}")

# ============================================================
# 5. 文本增强: 随机插入
# ============================================================
print("\n--- 5. 文本增强: 随机插入 ---")


def random_insertion(text, n_inserts=2):
    """随机插入: 随机在某位置插入某词的同义词"""
    words = text.split()
    new_words = words.copy()

    for _ in range(n_inserts):
        # 随机选择一个词
        idx = random.randint(0, len(words) - 1)
        word = words[idx].lower()

        if word in SYNONYM_DICT:
            synonym = random.choice(SYNONYM_DICT[word])
            insert_pos = random.randint(0, len(new_words))
            new_words.insert(insert_pos, synonym)

    return " ".join(new_words)


print(f"  原文: {original_text}")
print(f"  随机插入变体:")
for i in range(3):
    augmented = random_insertion(original_text, n_inserts=2)
    print(f"    [{i+1}] {augmented}")

# ============================================================
# 6. 文本增强: EDA (Easy Data Augmentation) 组合
# ============================================================
print("\n--- 6. EDA (Easy Data Augmentation) 组合 ---")


def eda_augment(text, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=4):
    """
    EDA: 随机选择一种增强方法生成变体
    每种方法各生成 num_aug//4 个变体
    """
    augmented_texts = []

    # 同义词替换
    for _ in range(num_aug):
        aug = synonym_replacement(text, replace_ratio=alpha_sr)
        if aug != text:
            augmented_texts.append(aug)

    # 随机插入
    for _ in range(num_aug):
        aug = random_insertion(text, n_inserts=max(1, int(alpha_ri * len(text.split()))))
        augmented_texts.append(aug)

    # 随机交换
    for _ in range(num_aug):
        aug = random_swap(text, n_swaps=max(1, int(alpha_rs * len(text.split()))))
        augmented_texts.append(aug)

    # 随机删除
    for _ in range(num_aug):
        aug = random_deletion(text, delete_prob=p_rd)
        augmented_texts.append(aug)

    return augmented_texts


text = "Machine learning is a fast growing and important field"
augmented = eda_augment(text, num_aug=2)

print(f"  原文: {text}")
print(f"  EDA 增强变体 (共 {len(augmented)} 个):")
for i, aug in enumerate(augmented[:8]):
    print(f"    [{i+1}] {aug}")

# ============================================================
# 7. 回译 (Back Translation) 概念
# ============================================================
print("\n--- 7. 回译 (Back Translation) 概念 ---")
print("""
  回译增强流程:
    原文 (英文) → 翻译为法文 → 翻译回英文 → 增强样本

  例:
    "The movie was great" → "Le film etait genial"
                          → "The film was excellent"

  优点: 生成自然的语言变体, 保持语义不变
  缺点: 需要翻译模型, 速度较慢

  实现方式:
    - 使用 HuggingFace 翻译 pipeline
    - 使用 googletrans 库
    - 使用专用翻译 API

  from transformers import pipeline
  translator = pipeline("translation_en_to_fr")
  back_translator = pipeline("translation_fr_to_en")

  french = translator("The movie was great")[0]['translation_text']
  english = back_translator(french)[0]['translation_text']
""")

# ============================================================
# 8. 图像增强: Mixup
# ============================================================
print("\n--- 8. 图像增强: Mixup ---")
print("""
  Mixup:
    - 将两张图片按比例混合
    - 标签也按相同比例混合
    - x_mixed = λ * x_1 + (1 - λ) * x_2
    - y_mixed = λ * y_1 + (1 - λ) * y_2
    - λ ~ Beta(α, α), 通常 α=1.0 (即均匀分布)
""")


def mixup_data(x1, x2, y1, y2, alpha=1.0):
    """Mixup 数据增强"""
    lam = np.random.beta(alpha, alpha)
    x_mixed = lam * x1 + (1 - lam) * x2
    y_mixed = lam * y1 + (1 - lam) * y2
    return x_mixed, y_mixed, lam


# 用模拟数据演示 Mixup
np.random.seed(42)
img1 = np.zeros((32, 32)) + 0.2  # 暗图
img2 = np.ones((32, 32)) * 0.9   # 亮图
# 添加一些纹理
img1[8:24, 8:24] = 0.8
img2[4:28, 4:28] = 0.3

label1 = np.array([1, 0])  # 类别 0
label2 = np.array([0, 1])  # 类别 1

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
fig.suptitle("Mixup 数据增强示例", fontsize=14, fontweight="bold")

axes[0][0].imshow(img1, cmap="gray", vmin=0, vmax=1)
axes[0][0].set_title(f"图片1 (label=[1,0])")
axes[0][1].imshow(img2, cmap="gray", vmin=0, vmax=1)
axes[0][1].set_title(f"图片2 (label=[0,1])")

mixup_results = []
for i, alpha in enumerate([0.2, 0.5, 0.8]):
    mixed_img, mixed_label, lam = mixup_data(img1, img2, label1, label2, alpha=alpha)
    mixup_results.append((mixed_img, mixed_label, lam))
    axes[0][2 + i].imshow(mixed_img, cmap="gray", vmin=0, vmax=1)
    axes[0][2 + i].set_title(f"Mixup (λ={lam:.2f})\nlabel={mixed_label}")

# 打印 Mixup 结果
print("  Mixup 演示:")
for mixed_img, mixed_label, lam in mixup_results:
    print(f"    λ={lam:.4f}, mixed_label={mixed_label}")

# ============================================================
# 9. 图像增强: CutMix
# ============================================================
print("\n--- 9. 图像增强: CutMix ---")
print("""
  CutMix:
    - 从图片1中裁切一个矩形区域
    - 用图片2的对应区域替换
    - 标签按面积比例混合
    - y_mixed = λ * y_1 + (1 - λ) * y_2
    - λ = (W*H - 裁切面积) / (W*H)
""")


def cutmix_data(x1, x2, y1, y2, alpha=1.0):
    """CutMix 数据增强"""
    h, w = x1.shape[:2]
    lam = np.random.beta(alpha, alpha)

    # 计算裁切区域大小
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)

    # 随机选择裁切中心
    cy = random.randint(0, h - 1)
    cx = random.randint(0, w - 1)

    # 计算裁切边界
    y1_cut = max(0, cy - cut_h // 2)
    y2_cut = min(h, cy + cut_h // 2)
    x1_cut = max(0, cx - cut_w // 2)
    x2_cut = min(w, cx + cut_w // 2)

    # 复制图片1, 用图片2的区域替换
    x_mixed = x1.copy()
    x_mixed[y1_cut:y2_cut, x1_cut:x2_cut] = x2[y1_cut:y2_cut, x1_cut:x2_cut]

    # 按实际面积比例调整 λ
    actual_area = (y2_cut - y1_cut) * (x2_cut - x1_cut)
    total_area = h * w
    adjusted_lam = 1.0 - actual_area / total_area
    y_mixed = adjusted_lam * y1 + (1 - adjusted_lam) * y2

    return x_mixed, y_mixed, adjusted_lam


# CutMix 演示
random.seed(42)
np.random.seed(42)

for i in range(4):
    mixed_img, mixed_label, lam = cutmix_data(img1, img2, label1, label2, alpha=1.0)
    axes[1][i].imshow(mixed_img, cmap="gray", vmin=0, vmax=1)
    axes[1][i].set_title(f"CutMix (λ={lam:.2f})\nlabel={mixed_label}")

plt.tight_layout()
plt.savefig("D:/code/big-model-learn/code/q_01/W16/augmentation_demo.png", dpi=150, bbox_inches="tight")
plt.close()
print("  增强示例图已保存: W16/augmentation_demo.png")

# ============================================================
# 10. PyTorch 中的实现
# ============================================================
print("\n--- 10. PyTorch 实现参考 ---")
print("""
  # Mixup in PyTorch Training Loop
  def mixup_criterion(criterion, pred, y_a, y_b, lam):
      return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

  for inputs, targets in dataloader:
      # 生成 Mixup 样本
      lam = np.random.beta(alpha, alpha)
      index = torch.randperm(inputs.size(0))
      mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
      targets_a, targets_b = targets, targets[index]

      # 前向传播
      outputs = model(mixed_inputs)
      loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
      loss.backward()
      optimizer.step()

  # PyTorch torchvision 增强变换
  from torchvision import transforms
      transforms.Compose([
          transforms.RandomHorizontalFlip(p=0.5),
          transforms.RandomRotation(degrees=15),
          transforms.ColorJitter(brightness=0.2, contrast=0.2),
          transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
      ])
""")

# ============================================================
# 11. 总结
# ============================================================
print("\n--- 11. 总结 ---")
print("""
  本节学习了:
  1) 文本增强: 同义词替换, 随机删除, 随机交换, 随机插入
  2) EDA (Easy Data Augmentation) 组合方法
  3) 回译 (Back Translation) 概念
  4) 图像增强: Mixup (像素混合) 和 CutMix (区域裁切混合)
  5) PyTorch 训练循环中集成增强

  实践建议:
    - 文本分类: EDA 或回译增强
    - 图像分类: Mixup/CutMix + 几何/颜色变换
    - 注意: 验证集不要使用增强!
    - 增强不能弥补差数据, 先保证数据质量

  下一步: d3_regularization_advanced.py - 高级正则化技术
""")
