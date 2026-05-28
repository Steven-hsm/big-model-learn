"""
W29-D5: 文本切分器
==================
功能:
  - 固定长度切分
  - 递归切分
  - 按段落切分
  - 切分策略对比
  - 切分效果评估
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from abc import ABC, abstractmethod
import time


# ============================================================
# 1. 数据模型
# ============================================================

@dataclass
class Chunk:
    """文本块"""
    id: str
    content: str
    index: int           # 在原文中的顺序
    start_char: int      # 起始字符位置
    end_char: int        # 结束字符位置
    char_count: int = 0
    word_count: int = 0
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.char_count = len(self.content)
        self.word_count = len(self.content.split())


@dataclass
class SplitResult:
    """切分结果"""
    chunks: List[Chunk]
    strategy: str
    original_length: int
    split_time_ms: float = 0.0
    params: Dict = field(default_factory=dict)

    def summary(self) -> str:
        """生成摘要"""
        if not self.chunks:
            return "无切分结果"

        lengths = [c.char_count for c in self.chunks]
        avg_len = sum(lengths) / len(lengths)
        min_len = min(lengths)
        max_len = max(lengths)

        return (
            f"策略: {self.strategy}\n"
            f"  原文长度: {self.original_length} 字符\n"
            f"  切分数量: {len(self.chunks)} 块\n"
            f"  块大小: 平均={avg_len:.0f}, "
            f"最小={min_len}, 最大={max_len}\n"
            f"  耗时: {self.split_time_ms:.2f}ms\n"
            f"  参数: {self.params}"
        )


# ============================================================
# 2. 切分器基类与实现
# ============================================================

class BaseSplitter(ABC):
    """切分器基类"""

    @abstractmethod
    def split(self, text: str) -> SplitResult:
        pass

    def _create_chunk(
        self, content: str, index: int,
        start: int, end: int
    ) -> Chunk:
        """创建文本块"""
        return Chunk(
            id=f"chunk_{index:04d}",
            content=content,
            index=index,
            start_char=start,
            end_char=end,
        )


class FixedLengthSplitter(BaseSplitter):
    """固定长度切分器"""

    def __init__(
        self, chunk_size: int = 500,
        overlap: int = 50
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str) -> SplitResult:
        """按固定长度切分"""
        start_time = time.time()
        chunks = []
        step = self.chunk_size - self.overlap

        i = 0
        index = 0
        while i < len(text):
            end = min(i + self.chunk_size, len(text))
            chunk_text = text[i:end]

            # 去除首尾空白
            chunk_text = chunk_text.strip()
            if chunk_text:
                chunks.append(
                    self._create_chunk(chunk_text, index, i, end)
                )
                index += 1

            i += step
            if i >= len(text):
                break

        elapsed = (time.time() - start_time) * 1000

        return SplitResult(
            chunks=chunks,
            strategy="固定长度切分",
            original_length=len(text),
            split_time_ms=elapsed,
            params={
                "chunk_size": self.chunk_size,
                "overlap": self.overlap,
            },
        )


class RecursiveSplitter(BaseSplitter):
    """递归切分器 (按分隔符层级递归切分)"""

    def __init__(
        self, chunk_size: int = 500,
        overlap: int = 50,
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators or [
            "\n\n",   # 段落
            "\n",     # 换行
            "。",     # 中文句号
            ".",      # 英文句号
            " ",      # 空格
            "",       # 字符级
        ]

    def split(self, text: str) -> SplitResult:
        """递归切分"""
        start_time = time.time()
        chunks = self._recursive_split(text, self.separators)
        elapsed = (time.time() - start_time) * 1000

        # 添加索引和位置信息
        result_chunks = []
        pos = 0
        for i, chunk_text in enumerate(chunks):
            start = text.find(chunk_text, pos)
            if start == -1:
                start = pos
            end = start + len(chunk_text)
            result_chunks.append(
                self._create_chunk(chunk_text.strip(), i, start, end)
            )
            pos = end

        return SplitResult(
            chunks=result_chunks,
            strategy="递归切分",
            original_length=len(text),
            split_time_ms=elapsed,
            params={
                "chunk_size": self.chunk_size,
                "overlap": self.overlap,
                "separators": self.separators[:3],
            },
        )

    def _recursive_split(
        self, text: str, separators: List[str]
    ) -> List[str]:
        """递归切分核心逻辑"""
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        # 尝试当前分隔符
        if not separators:
            # 没有更多分隔符, 强制切分
            return self._force_split(text)

        sep = separators[0]
        remaining_seps = separators[1:]

        if sep == "":
            return self._force_split(text)

        # 按分隔符切分
        parts = text.split(sep)
        parts = [p for p in parts if p.strip()]

        if len(parts) <= 1:
            # 当前分隔符无法有效切分, 尝试下一个
            return self._recursive_split(text, remaining_seps)

        # 合并小的片段
        chunks = []
        current = ""

        for part in parts:
            candidate = current + sep + part if current else part

            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    if len(current) > self.chunk_size:
                        # 递归切分过大的片段
                        chunks.extend(
                            self._recursive_split(
                                current, remaining_seps
                            )
                        )
                    else:
                        chunks.append(current)
                current = part

        if current:
            if len(current) > self.chunk_size:
                chunks.extend(
                    self._recursive_split(current, remaining_seps)
                )
            else:
                chunks.append(current)

        return chunks

    def _force_split(self, text: str) -> List[str]:
        """强制切分 (字符级别)"""
        chunks = []
        step = self.chunk_size - self.overlap
        i = 0
        while i < len(text):
            end = min(i + self.chunk_size, len(text))
            chunks.append(text[i:end])
            i += step
        return chunks


class ParagraphSplitter(BaseSplitter):
    """按段落切分器"""

    def __init__(
        self, max_chunk_size: int = 1000,
        min_chunk_size: int = 50
    ):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def split(self, text: str) -> SplitResult:
        """按段落切分"""
        start_time = time.time()

        # 按双换行分段
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks = []
        current_parts = []
        current_len = 0
        index = 0

        for para in paragraphs:
            if current_len + len(para) > self.max_chunk_size and current_parts:
                # 当前块已满, 保存
                chunk_text = "\n\n".join(current_parts)
                chunks.append(
                    self._create_chunk(
                        chunk_text, index, 0, len(chunk_text)
                    )
                )
                index += 1
                current_parts = []
                current_len = 0

            current_parts.append(para)
            current_len += len(para)

        # 处理剩余
        if current_parts:
            chunk_text = "\n\n".join(current_parts)
            chunks.append(
                self._create_chunk(
                    chunk_text, index, 0, len(chunk_text)
                )
            )

        elapsed = (time.time() - start_time) * 1000

        return SplitResult(
            chunks=chunks,
            strategy="按段落切分",
            original_length=len(text),
            split_time_ms=elapsed,
            params={
                "max_chunk_size": self.max_chunk_size,
                "min_chunk_size": self.min_chunk_size,
            },
        )


# ============================================================
# 3. 切分效果评估器
# ============================================================

class SplitEvaluator:
    """切分效果评估"""

    @staticmethod
    def evaluate(result: SplitResult) -> Dict:
        """评估切分效果"""
        if not result.chunks:
            return {"error": "无切分结果"}

        lengths = [c.char_count for c in result.chunks]
        avg_len = sum(lengths) / len(lengths)

        # 长度标准差 (越小越均匀)
        variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)
        std_dev = variance ** 0.5

        # 完整性: 总字符数/原文字符数
        total_chars = sum(lengths)
        coverage = total_chars / result.original_length if result.original_length > 0 else 0

        # 空块比例
        empty_ratio = sum(1 for l in lengths if l < 10) / len(lengths)

        return {
            "strategy": result.strategy,
            "chunk_count": len(result.chunks),
            "avg_length": round(avg_len, 1),
            "std_deviation": round(std_dev, 1),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "coverage": round(coverage, 3),
            "empty_ratio": round(empty_ratio, 3),
            "time_ms": round(result.split_time_ms, 2),
        }


# ============================================================
# 4. 演示: 切分策略对比
# ============================================================

def demo():
    """演示文本切分器"""

    print("=" * 60)
    print("W29-D5: 文本切分器 - 策略对比")
    print("=" * 60)

    # 示例文本
    sample_text = """人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，致力于研究和开发能够模拟人类智能行为的系统和技术。

机器学习是人工智能的核心领域之一。它通过算法和统计模型，使计算机系统能够从数据中学习并改进性能，而无需进行明确的编程。常见的机器学习方法包括监督学习、无监督学习和强化学习。

深度学习是机器学习的一个子集，它使用多层神经网络来处理复杂的模式识别任务。深度学习在图像识别、自然语言处理和语音识别等领域取得了突破性进展。

自然语言处理（NLP）是AI的重要应用领域，它使计算机能够理解、解释和生成人类语言。现代NLP技术主要基于Transformer架构，如BERT、GPT等预训练模型。

检索增强生成（RAG）是一种结合信息检索和文本生成的技术。它通过检索相关文档来增强LLM的回答质量，使生成的答案更加准确和有据可依。RAG系统通常包括文档加载、文本切分、向量化、检索和生成等步骤。

在构建RAG系统时，文本切分是一个关键环节。好的切分策略应该保持语义完整性，同时确保文本块大小适中，便于嵌入模型处理。"""

    print(f"\n原始文本长度: {len(sample_text)} 字符\n")

    # 三种切分策略
    splitters = [
        ("固定长度 (500字, 重叠50)", FixedLengthSplitter(500, 50)),
        ("递归切分 (500字, 重叠50)", RecursiveSplitter(500, 50)),
        ("按段落 (最大1000字)", ParagraphSplitter(1000)),
    ]

    evaluator = SplitEvaluator()
    results = []

    for name, splitter in splitters:
        result = splitter.split(sample_text)
        results.append(result)

        print(f"--- {name} ---")
        print(result.summary())

        # 展示前两个块
        for chunk in result.chunks[:2]:
            preview = chunk.content[:60].replace("\n", " ")
            print(f"  块{chunk.index}: [{chunk.char_count}字] {preview}...")
        print("")

    # ---- 对比评估 ----
    print("=" * 60)
    print("切分效果对比")
    print("=" * 60)

    headers = ["指标", "固定长度", "递归切分", "按段落"]
    print(f"\n{'指标':<15} {'固定长度':<15} {'递归切分':<15} {'按段落':<15}")
    print("-" * 60)

    evals = [evaluator.evaluate(r) for r in results]
    metrics = ["chunk_count", "avg_length", "std_deviation",
               "coverage", "time_ms"]

    metric_names = {
        "chunk_count": "块数量",
        "avg_length": "平均长度",
        "std_deviation": "长度标准差",
        "coverage": "覆盖率",
        "time_ms": "耗时(ms)",
    }

    for metric in metrics:
        row = f"{metric_names[metric]:<15}"
        for ev in evals:
            row += f"{str(ev[metric]):<15}"
        print(row)

    # ---- 推荐策略 ----
    print("\n" + "=" * 60)
    print("推荐策略:")
    print("=" * 60)
    print("  短文档 (<1000字): 按段落切分, 保持语义完整")
    print("  中等文档: 递归切分, chunk_size=500, overlap=50")
    print("  长文档: 递归切分 + 段落感知, chunk_size=300-500")
    print("  RAG最佳实践: chunk_size=500, overlap=50-100")


# ============================================================
# 运行演示
# ============================================================

if __name__ == "__main__":
    demo()
