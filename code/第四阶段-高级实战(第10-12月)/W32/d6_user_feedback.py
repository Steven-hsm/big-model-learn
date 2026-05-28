"""
W32-D6 用户反馈系统
====================
实现用户反馈收集和分析, 包括:
- 反馈数据收集
- 评分系统
- 反馈分析与报告

用户反馈是衡量系统实际效果的重要维度。
"""

import time
import json
import hashlib
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter

try:
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

import numpy as np


# ============================================================
# 1. 反馈数据模型
# ============================================================

@dataclass
class Feedback:
    """用户反馈"""
    feedback_id: str
    session_id: str
    question: str
    answer: str
    rating: int                  # 1-5分
    comment: str = ""            # 文字反馈
    feedback_type: str = "rating"  # rating / thumbs / comment
    tags: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    @property
    def is_positive(self) -> bool:
        return self.rating >= 4

    @property
    def is_negative(self) -> bool:
        return self.rating <= 2


class FeedbackCollector:
    """反馈收集器"""

    def __init__(self):
        self.feedbacks: List[Feedback] = []
        self.thumbs_up_count = 0
        self.thumbs_down_count = 0

    def add_rating_feedback(self, session_id: str, question: str,
                             answer: str, rating: int, comment: str = "",
                             tags: List[str] = None) -> Feedback:
        """添加评分反馈"""
        fb = Feedback(
            feedback_id=hashlib.md5(f"{session_id}_{time.time()}".encode()).hexdigest()[:10],
            session_id=session_id,
            question=question,
            answer=answer,
            rating=rating,
            comment=comment,
            feedback_type="rating",
            tags=tags or [],
        )
        self.feedbacks.append(fb)
        return fb

    def add_thumbs_feedback(self, session_id: str, question: str,
                             answer: str, is_positive: bool) -> Feedback:
        """添加点赞/点踩反馈"""
        if is_positive:
            self.thumbs_up_count += 1
        else:
            self.thumbs_down_count += 1

        return self.add_rating_feedback(
            session_id, question, answer,
            rating=5 if is_positive else 1,
            feedback_type="thumbs"
        )

    def get_all(self) -> List[Feedback]:
        return self.feedbacks

    def get_negative_feedbacks(self) -> List[Feedback]:
        return [fb for fb in self.feedbacks if fb.is_negative]

    def get_positive_feedbacks(self) -> List[Feedback]:
        return [fb for fb in self.feedbacks if fb.is_positive]


# ============================================================
# 2. 评分系统
# ============================================================

class RatingSystem:
    """评分系统"""

    def __init__(self, collector: FeedbackCollector):
        self.collector = collector

    def get_average_rating(self) -> float:
        """获取平均评分"""
        if not self.collector.feedbacks:
            return 0.0
        ratings = [fb.rating for fb in self.collector.feedbacks]
        return np.mean(ratings)

    def get_rating_distribution(self) -> Dict[int, int]:
        """获取评分分布"""
        return dict(Counter(fb.rating for fb in self.collector.feedbacks))

    def get_satisfaction_rate(self) -> float:
        """获取满意度(4-5分占比)"""
        if not self.collector.feedbacks:
            return 0.0
        positive = sum(1 for fb in self.collector.feedbacks if fb.rating >= 4)
        return positive / len(self.collector.feedbacks)

    def get_nps_score(self) -> float:
        """获取NPS(净推荐值)

        NPS = 推荐者比例(9-10分映射到5分制的5分) - 贬损者比例(1-2分)
        范围: -100 到 +100
        """
        if not self.collector.feedbacks:
            return 0.0
        promoters = sum(1 for fb in self.collector.feedbacks if fb.rating == 5)
        detractors = sum(1 for fb in self.collector.feedbacks if fb.rating <= 2)
        total = len(self.collector.feedbacks)
        return (promoters - detractors) / total * 100

    def get_trend(self, window: int = 10) -> List[float]:
        """获取评分趋势(滑动平均)"""
        ratings = [fb.rating for fb in self.collector.feedbacks]
        if len(ratings) < window:
            return [np.mean(ratings)] if ratings else []

        trends = []
        for i in range(len(ratings) - window + 1):
            trends.append(np.mean(ratings[i:i+window]))
        return trends

    def summary(self) -> Dict:
        """评分摘要"""
        return {
            'total_feedbacks': len(self.collector.feedbacks),
            'average_rating': self.get_average_rating(),
            'satisfaction_rate': self.get_satisfaction_rate(),
            'nps_score': self.get_nps_score(),
            'distribution': self.get_rating_distribution(),
            'thumbs_up': self.collector.thumbs_up_count,
            'thumbs_down': self.collector.thumbs_down_count,
        }


# ============================================================
# 3. 反馈分析器
# ============================================================

class FeedbackAnalyzer:
    """反馈分析器"""

    def __init__(self, collector: FeedbackCollector):
        self.collector = collector

    def analyze_negative_patterns(self) -> Dict:
        """分析负面反馈模式"""
        negative = self.collector.get_negative_feedbacks()
        if not negative:
            return {'count': 0, 'patterns': []}

        # 关键词提取(从评论中)
        tag_counts = Counter()
        for fb in negative:
            tag_counts.update(fb.tags)

        # 评论关键词
        comment_words = Counter()
        for fb in negative:
            if fb.comment:
                words = fb.comment.lower().split()
                comment_words.update(words)

        return {
            'count': len(negative),
            'percentage': len(negative) / len(self.collector.feedbacks) * 100
                if self.collector.feedbacks else 0,
            'common_tags': dict(tag_counts.most_common(10)),
            'common_comment_words': dict(comment_words.most_common(10)),
            'low_rated_questions': [
                {'question': fb.question, 'rating': fb.rating, 'comment': fb.comment}
                for fb in sorted(negative, key=lambda x: x.rating)[:5]
            ],
        }

    def analyze_by_question_type(self) -> Dict:
        """按问题类型分析满意度"""
        type_stats = defaultdict(lambda: {'ratings': [], 'count': 0})

        for fb in self.collector.feedbacks:
            qtype = self._classify_question(fb.question)
            type_stats[qtype]['ratings'].append(fb.rating)
            type_stats[qtype]['count'] += 1

        result = {}
        for qtype, stats in type_stats.items():
            result[qtype] = {
                'count': stats['count'],
                'avg_rating': np.mean(stats['ratings']),
                'satisfaction': sum(1 for r in stats['ratings'] if r >= 4) / len(stats['ratings']),
            }
        return result

    def _classify_question(self, question: str) -> str:
        """简单的问题分类"""
        if '?' in question or '？' in question or '什么' in question:
            return "事实查询"
        elif '如何' in question or '怎么' in question:
            return "方法指导"
        elif '为什么' in question:
            return "原因分析"
        elif '对比' in question or '区别' in question:
            return "对比分析"
        else:
            return "其他"

    def generate_improvement_report(self) -> str:
        """生成改进报告"""
        rating_sys = RatingSystem(self.collector)
        summary = rating_sys.summary()
        negative_analysis = self.analyze_negative_patterns()

        lines = [
            "用户反馈改进报告",
            "=" * 40,
            f"总反馈数: {summary['total_feedbacks']}",
            f"平均评分: {summary['average_rating']:.2f}/5",
            f"满意度: {summary['satisfaction_rate']:.1%}",
            f"NPS: {summary['nps_score']:.0f}",
            "",
            f"负面反馈: {negative_analysis['count']}条 ({negative_analysis['percentage']:.1f}%)",
        ]

        if negative_analysis.get('low_rated_questions'):
            lines.append("\n低分问题TOP5:")
            for item in negative_analysis['low_rated_questions'][:5]:
                lines.append(f"  [{item['rating']}分] {item['question']}")
                if item['comment']:
                    lines.append(f"         反馈: {item['comment']}")

        return "\n".join(lines)


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("W32-D6 用户反馈系统")
    print("=" * 60)

    # --- 收集模拟反馈 ---
    collector = FeedbackCollector()

    feedback_data = [
        ("sess_1", "什么是RAG?", "RAG是检索增强生成技术...", 5, "回答准确详细", ["准确性"]),
        ("sess_2", "如何优化检索?", "优化检索可以...", 4, "还不错", ["有用"]),
        ("sess_3", "量子计算是什么?", "抱歉, 未找到相关信息", 2, "没有回答我的问题", ["无结果"]),
        ("sess_4", "Python有哪些库?", "Python有很多库", 2, "太简略了", ["不完整"]),
        ("sess_5", "什么是BM25?", "BM25是一种检索算法", 4, "基本满足", ["准确性"]),
        ("sess_6", "如何部署?", "可以用Docker", 3, "希望能更详细", ["不完整"]),
        ("sess_7", "RAG和微调区别?", "RAG通过检索增强...", 5, "非常好!", ["完整性", "准确性"]),
        ("sess_8", "为什么用向量检索?", "向量检索能理解语义", 4, "", ["有用"]),
        ("sess_9", "怎么评估系统?", "可以用Recall等指标", 1, "完全没帮助", ["不相关"]),
        ("sess_10", "如何处理长文档?", "可以使用分块策略", 4, "有用但不够深入", ["不完整"]),
        ("sess_11", "深度学习入门?", "深度学习使用神经网络", 5, "清晰明了", ["有用"]),
        ("sess_12", "对比BM25和向量?", "BM25基于关键词...", 3, "对比不够明显", ["不完整"]),
        ("sess_13", "什么是Prompt?", "Prompt是提示词", 2, "太简单", ["不完整"]),
        ("sess_14", "Python适合AI吗?", "Python是AI首选", 5, "很全面", ["准确性"]),
        ("sess_15", "如何设计系统?", "需要考虑多个方面", 1, "太空泛", ["不相关"]),
    ]

    for sid, q, a, rating, comment, tags in feedback_data:
        collector.add_rating_feedback(sid, q, a, rating, comment, tags)

    # 模拟点赞/点踩
    collector.add_thumbs_feedback("sess_16", "示例问题1", "好回答", True)
    collector.add_thumbs_feedback("sess_17", "示例问题2", "差回答", False)
    collector.add_thumbs_feedback("sess_18", "示例问题3", "好回答", True)

    # --- 评分分析 ---
    print("\n--- 1. 评分统计 ---")
    rating_sys = RatingSystem(collector)
    summary = rating_sys.summary()
    print(f"总反馈: {summary['total_feedbacks']}")
    print(f"平均评分: {summary['average_rating']:.2f}/5")
    print(f"满意度: {summary['satisfaction_rate']:.1%}")
    print(f"NPS: {summary['nps_score']:.0f}")
    print(f"评分分布: {summary['distribution']}")

    # --- 反馈分析 ---
    print(f"\n{'='*60}")
    print("--- 2. 反馈分析 ---")
    print(f"{'='*60}")
    analyzer = FeedbackAnalyzer(collector)
    print(analyzer.generate_improvement_report())

    # 按问题类型分析
    type_analysis = analyzer.analyze_by_question_type()
    print("\n按问题类型分析:")
    for qtype, stats in type_analysis.items():
        print(f"  {qtype}: 数量={stats['count']}, 均分={stats['avg_rating']:.2f}, 满意度={stats['satisfaction']:.0%}")

    # --- 可视化 ---
    if HAS_PLT:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 评分分布
        dist = summary['distribution']
        labels = [str(i) for i in sorted(dist.keys())]
        counts = [dist.get(int(l), 0) for l in labels]
        colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#27ae60']
        ax = axes[0, 0]
        ax.bar(labels, counts, color=colors[:len(labels)])
        ax.set_xlabel('评分')
        ax.set_ylabel('数量')
        ax.set_title('评分分布', fontweight='bold')

        # 满意度饼图
        ax = axes[0, 1]
        sat = summary['satisfaction_rate']
        labels_pie = ['满意(4-5分)', '一般(3分)', '不满意(1-2分)']
        dist_vals = summary['distribution']
        sizes = [
            dist_vals.get(4, 0) + dist_vals.get(5, 0),
            dist_vals.get(3, 0),
            dist_vals.get(1, 0) + dist_vals.get(2, 0),
        ]
        ax.pie(sizes, labels=labels_pie, autopct='%1.0f%%',
                colors=['#2ecc71', '#f1c40f', '#e74c3c'])
        ax.set_title(f'满意度: {sat:.0%}', fontweight='bold')

        # 评分趋势
        ax = axes[1, 0]
        trends = rating_sys.get_trend(window=5)
        ax.plot(trends, color='#3498db', linewidth=2)
        ax.axhline(y=3, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('反馈序号')
        ax.set_ylabel('评分(滑动平均)')
        ax.set_title('评分趋势', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 问题类型分析
        ax = axes[1, 1]
        types = list(type_analysis.keys())
        avg_ratings = [type_analysis[t]['avg_rating'] for t in types]
        ax.barh(types, avg_ratings, color='#9b59b6')
        ax.set_xlabel('平均评分')
        ax.set_title('各问题类型满意度', fontweight='bold')
        ax.set_xlim(0, 5)

        plt.tight_layout()
        plt.savefig('D:/code/big-model-learn/code/q_01/W32/d6_user_feedback.png', dpi=150)
        print("\n图表已保存为 d6_user_feedback.png")
        plt.close()

    print("\n完成!")
