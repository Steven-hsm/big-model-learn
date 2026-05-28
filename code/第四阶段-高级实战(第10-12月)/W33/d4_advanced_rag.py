"""
W33-D4 高级RAG技术
===================
实现高级RAG检索策略, 包括:
- 查询改写(Query Rewriting)
- HyDE (Hypothetical Document Embeddings)
- 多路召回(Multi-route Retrieval)

高级技术可以显著提升检索质量和最终回答效果。
"""

import re
import hashlib
import time
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

import numpy as np


# ============================================================
# 1. 查询改写 (Query Rewriting)
# ============================================================

class QueryRewriter:
    """查询改写器

    将用户的自然语言查询改写为更适合检索的形式:
    1. 去除停用词
    2. 提取核心概念
    3. 扩展同义词
    4. 生成多个变体
    """

    # 简单停用词表
    STOP_WORDS = {'的', '了', '是', '在', '有', '和', '与', '或', '就', '不',
                  '也', '很', '都', '这', '那', '为', '以', '及', '等', '被',
                  '到', '把', '从', '让', '给', '上', '下', '中', '里'}

    # 同义词扩展表
    SYNONYMS = {
        'RAG': ['检索增强生成', 'retrieval augmented generation'],
        '检索': ['搜索', '查找', '召回', 'retrieval', 'search'],
        '优化': ['改进', '提升', '改善', 'optimize', 'improve'],
        '部署': ['上线', '发布', 'deploy', 'deployment'],
        '评估': ['评价', '测试', '衡量', 'evaluate', 'assess'],
        '模型': ['大语言模型', 'LLM', 'language model'],
        '向量': ['embedding', '嵌入', '向量表示'],
    }

    def rewrite(self, query: str) -> Dict:
        """改写查询"""
        # 1. 基础清理
        cleaned = self._clean_query(query)

        # 2. 提取关键词
        keywords = self._extract_keywords(cleaned)

        # 3. 生成变体
        variants = self._generate_variants(query, keywords)

        return {
            'original': query,
            'cleaned': cleaned,
            'keywords': keywords,
            'variants': variants,
            'recommended': variants[0] if variants else query,
        }

    def _clean_query(self, query: str) -> str:
        """清理查询"""
        # 去除多余空格和标点
        cleaned = re.sub(r'[^\w\s一-鿿]', ' ', query)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        words = []
        # 中文分词(简化: 单字+连续英文)
        for seg in re.findall(r'[一-鿿]|[a-zA-Z]+', text):
            if seg not in self.STOP_WORDS and len(seg) > 0:
                words.append(seg)
        return words

    def _generate_variants(self, original: str, keywords: List[str]) -> List[str]:
        """生成查询变体"""
        variants = [original]

        # 变体1: 同义词替换
        expanded = original
        for keyword in keywords:
            if keyword in self.SYNONYMS:
                syn = self.SYNONYMS[keyword][0]
                expanded = expanded.replace(keyword, syn)
                break  # 只替换一个
        if expanded != original:
            variants.append(expanded)

        # 变体2: 只保留关键词
        if keywords:
            variants.append(' '.join(keywords))

        # 变体3: 英文版本(简化)
        en_terms = []
        for kw in keywords:
            if kw in self.SYNONYMS:
                en_terms.extend([s for s in self.SYNONYMS[kw] if re.match(r'^[a-z]', s)])
        if en_terms:
            variants.append(' '.join(en_terms))

        return variants


# ============================================================
# 2. HyDE (Hypothetical Document Embeddings)
# ============================================================

class HyDERetriever:
    """HyDE检索器

    核心思想:
    1. 让LLM根据查询生成一个"假设性回答"
    2. 用这个假设回答(而非原始查询)去检索
    3. 假设回答在语义空间中更接近真实文档

    优势: 弥补查询和文档之间的语义鸿沟
    """

    def __init__(self):
        self.documents = {}

    def add_document(self, doc_id: str, title: str, content: str):
        self.documents[doc_id] = {'title': title, 'content': content}

    def generate_hypothetical_answer(self, query: str) -> str:
        """生成假设性回答(模拟LLM)"""
        # 实际应用中应调用LLM生成
        # 这里用模板模拟
        templates = {
            'RAG': "RAG(检索增强生成)是一种结合信息检索和文本生成的AI技术。"
                   "它通过先检索相关文档来增强大语言模型的回答质量。",
            'Python': "Python是一种广泛使用的高级编程语言, 在数据科学和AI领域尤其流行。",
            '检索': "检索是从大量文档中找到与查询最相关的文档的过程。"
                    "常用方法包括关键词检索(BM25)和向量语义检索。",
            '部署': "系统部署是将应用发布到生产环境的过程, 常用Docker容器化方案。",
        }

        for keyword, template in templates.items():
            if keyword.lower() in query.lower():
                return template

        return f"关于{query}的详细信息, 包括定义、原理、应用场景和最佳实践。"

    def search_with_hyde(self, query: str, top_k: int = 5) -> Dict:
        """使用HyDE进行检索"""
        # Step 1: 生成假设性回答
        hypothetical_answer = self.generate_hypothetical_answer(query)

        # Step 2: 用假设回答检索(模拟)
        results = self._search(hypothetical_answer, top_k)

        # Step 3: 也用原始查询检索, 然后融合
        original_results = self._search(query, top_k)

        # 融合结果(去重)
        seen = set()
        merged = []
        for r in results + original_results:
            if r['doc_id'] not in seen:
                seen.add(r['doc_id'])
                merged.append(r)

        return {
            'query': query,
            'hypothetical_answer': hypothetical_answer,
            'hyde_results': results[:3],
            'original_results': original_results[:3],
            'merged_results': merged[:top_k],
        }

    def _search(self, text: str, top_k: int) -> List[Dict]:
        """基于关键词匹配的搜索"""
        words = set(text.lower().split())
        results = []

        for doc_id, doc in self.documents.items():
            doc_words = set(doc['content'].lower().split())
            overlap = words & doc_words
            if overlap:
                score = len(overlap) / max(len(words), 1)
                results.append({
                    'doc_id': doc_id,
                    'title': doc['title'],
                    'score': score,
                })

        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]


# ============================================================
# 3. 多路召回 (Multi-route Retrieval)
# ============================================================

class MultiRouteRetriever:
    """多路召回检索器

    同时使用多种检索策略, 然后融合结果:
    - 路径1: 关键词检索(BM25风格)
    - 路径2: 语义检索(向量风格)
    - 路径3: HyDE检索
    """

    def __init__(self):
        self.documents = {}
        self.rewriter = QueryRewriter()
        self.hyde = HyDERetriever()

    def add_document(self, doc_id: str, title: str, content: str):
        self.documents[doc_id] = {'title': title, 'content': content}
        self.hyde.add_document(doc_id, title, content)

    def search(self, query: str, top_k: int = 5) -> Dict:
        """多路召回检索"""
        # Step 1: 查询改写
        rewritten = self.rewriter.rewrite(query)

        # Step 2: 多路检索
        # 路径1: 关键词检索
        keyword_results = self._keyword_search(rewritten['recommended'], top_k * 2)

        # 路径2: 语义检索(模拟)
        semantic_results = self._semantic_search(query, top_k * 2)

        # 路径3: HyDE检索
        hyde_results = self.hyde.search_with_hyde(query, top_k)
        hyde_docs = hyde_results['merged_results']

        # Step 3: RRF融合
        all_routes = [keyword_results, semantic_results, hyde_docs]
        fused = self._rrf_fusion(all_routes, k=60)

        return {
            'query': query,
            'rewritten': rewritten['recommended'],
            'variants': rewritten['variants'],
            'route_results': {
                'keyword': len(keyword_results),
                'semantic': len(semantic_results),
                'hyde': len(hyde_docs),
            },
            'final_results': fused[:top_k],
        }

    def _keyword_search(self, query: str, top_k: int) -> List[Dict]:
        """关键词检索"""
        words = set(query.lower().split())
        results = []
        for doc_id, doc in self.documents.items():
            doc_words = set(doc['content'].lower().split())
            overlap = words & doc_words
            if overlap:
                results.append({'doc_id': doc_id, 'title': doc['title'],
                               'score': len(overlap) / len(words)})
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

    def _semantic_search(self, query: str, top_k: int) -> List[Dict]:
        """语义检索(模拟)"""
        # 模拟: 对关键词检索结果加入随机偏移
        results = self._keyword_search(query, top_k)
        for r in results:
            r['score'] = r['score'] * np.random.uniform(0.8, 1.2)
        results.sort(key=lambda x: x['score'], reverse=True)
        return results

    def _rrf_fusion(self, result_lists: List[List[Dict]], k: int = 60) -> List[Dict]:
        """RRF融合"""
        scores = defaultdict(float)
        doc_info = {}

        for results in result_lists:
            for rank, doc in enumerate(results, 1):
                did = doc['doc_id']
                scores[did] += 1.0 / (k + rank)
                doc_info[did] = doc

        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [{'doc_id': did, 'score': score,
                 'title': doc_info[did].get('title', '')}
                for did, score in fused]


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("W33-D4 高级RAG技术")
    print("=" * 60)

    # --- 1. 查询改写 ---
    print("\n--- 1. 查询改写 ---")
    rewriter = QueryRewriter()
    queries = ["什么是RAG技术?", "如何优化检索的质量?", "Python的模型怎么部署?"]

    for q in queries:
        result = rewriter.rewrite(q)
        print(f"\n原始查询: {q}")
        print(f"关键词: {result['keywords']}")
        print(f"变体:")
        for v in result['variants']:
            print(f"  - {v}")

    # --- 2. HyDE ---
    print(f"\n{'='*60}")
    print("--- 2. HyDE检索 ---")
    print(f"{'='*60}")

    hyde = HyDERetriever()
    docs = [
        ("d1", "RAG技术概述", "RAG是检索增强生成技术, 结合了检索和生成的优势"),
        ("d2", "Python AI库", "Python拥有丰富的AI库如PyTorch和TensorFlow"),
        ("d3", "检索方法对比", "BM25基于关键词, 向量检索基于语义, 各有优势"),
    ]
    for did, title, content in docs:
        hyde.add_document(did, title, content)

    query = "什么是RAG?"
    hyde_result = hyde.search_with_hyde(query)
    print(f"查询: {query}")
    print(f"假设回答: {hyde_result['hypothetical_answer'][:80]}...")
    print(f"HyDE检索结果: {len(hyde_result['hyde_results'])}条")
    print(f"原始检索结果: {len(hyde_result['original_results'])}条")

    # --- 3. 多路召回 ---
    print(f"\n{'='*60}")
    print("--- 3. 多路召回 ---")
    print(f"{'='*60}")

    multi = MultiRouteRetriever()
    all_docs = [
        ("d1", "RAG系统设计", "RAG系统包含检索器、生成器和重排序器"),
        ("d2", "向量检索", "向量检索通过计算语义相似度来匹配文档"),
        ("d3", "BM25算法", "BM25是基于词频的经典检索算法"),
        ("d4", "Python编程", "Python在数据科学和AI领域广泛应用"),
        ("d5", "深度学习框架", "PyTorch和TensorFlow是主流深度学习框架"),
        ("d6", "Prompt工程", "好的Prompt设计能显著提升LLM输出质量"),
    ]
    for did, title, content in all_docs:
        multi.add_document(did, title, content)

    query = "如何提升RAG系统的检索质量?"
    result = multi.search(query, top_k=5)
    print(f"查询: {query}")
    print(f"改写后: {result['rewritten']}")
    print(f"各路召回: {result['route_results']}")
    print(f"最终结果:")
    for r in result['final_results']:
        print(f"  - {r['title']} (分数: {r['score']:.4f})")

    print("\n完成!")
