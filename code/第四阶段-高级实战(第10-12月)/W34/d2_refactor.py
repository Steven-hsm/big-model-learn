"""
W34-D2 代码重构
================
演示代码重构技巧, 包括:
- 通用组件模板
- 清理冗余代码
- 重构前后对比

重构让代码更易维护、更健壮。
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from functools import wraps
import time


# ============================================================
# 1. 重构前后对比
# ============================================================

def demonstrate_refactoring():
    """展示重构前后对比"""

    # === 重构前: 重复代码 ===
    print("--- 重构案例1: 消除重复代码 ---")
    print("重构前:")
    print("""
    def process_user(data):
        if not data:
            return None
        if 'name' not in data:
            return None
        return {'name': data['name'], 'type': 'user'}

    def process_product(data):
        if not data:
            return None
        if 'name' not in data:
            return None
        return {'name': data['name'], 'type': 'product'}
    """)

    print("重构后(通用函数):")
    print("""
    def process_entity(data, entity_type):
        if not data or 'name' not in data:
            return None
        return {'name': data['name'], 'type': entity_type}

    process_user = lambda d: process_entity(d, 'user')
    process_product = lambda d: process_entity(d, 'product')
    """)

    # === 重构案例2: 策略模式 ===
    print("\n--- 重构案例2: 策略模式替代if-else ---")
    print("重构前:")
    print("""
    def search(query, method):
        if method == 'bm25':
            return bm25_search(query)
        elif method == 'vector':
            return vector_search(query)
        elif method == 'hybrid':
            return hybrid_search(query)
        else:
            raise ValueError("未知方法")
    """)

    print("重构后(策略模式):")
    print("""
    SEARCH_STRATEGIES = {
        'bm25': bm25_search,
        'vector': vector_search,
        'hybrid': hybrid_search,
    }

    def search(query, method='hybrid'):
        strategy = SEARCH_STRATEGIES.get(method)
        if not strategy:
            raise ValueError(f"未知方法: {method}")
        return strategy(query)
    """)


# ============================================================
# 2. 通用组件模板
# ============================================================

class ConfigManager:
    """通用配置管理器

    重构要点:
    - 集中管理所有配置
    - 支持默认值
    - 支持环境变量覆盖
    - 支持验证
    """

    def __init__(self):
        self._config: Dict[str, Any] = {}
        self._defaults: Dict[str, Any] = {}
        self._validators: Dict[str, Callable] = {}

    def set_default(self, key: str, value: Any, validator: Callable = None):
        """设置默认值"""
        self._defaults[key] = value
        if validator:
            self._validators[key] = validator

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置"""
        return self._config.get(key, self._defaults.get(key, default))

    def set(self, key: str, value: Any) -> bool:
        """设置配置"""
        if key in self._validators:
            if not self._validators[key](value):
                return False
        self._config[key] = value
        return True

    def load_from_dict(self, config_dict: Dict):
        """从字典加载"""
        for key, value in config_dict.items():
            self.set(key, value)

    def get_all(self) -> Dict:
        """获取所有配置(合并默认值)"""
        result = dict(self._defaults)
        result.update(self._config)
        return result


class Repository:
    """通用数据仓库模板

    重构要点:
    - 抽象通用的CRUD操作
    - 支持不同后端存储
    - 统一错误处理
    """

    def __init__(self):
        self._data: Dict[str, Dict] = {}
        self._counter = 0

    def create(self, data: Dict) -> str:
        """创建记录"""
        self._counter += 1
        record_id = str(self._counter)
        data['id'] = record_id
        data['created_at'] = time.time()
        self._data[record_id] = data
        return record_id

    def get(self, record_id: str) -> Optional[Dict]:
        """获取记录"""
        return self._data.get(record_id)

    def update(self, record_id: str, data: Dict) -> bool:
        """更新记录"""
        if record_id not in self._data:
            return False
        self._data[record_id].update(data)
        self._data[record_id]['updated_at'] = time.time()
        return True

    def delete(self, record_id: str) -> bool:
        """删除记录"""
        if record_id in self._data:
            del self._data[record_id]
            return True
        return False

    def list_all(self, filter_func: Callable = None) -> List[Dict]:
        """列出所有记录"""
        records = list(self._data.values())
        if filter_func:
            records = [r for r in records if filter_func(r)]
        return records

    def count(self) -> int:
        return len(self._data)


class Timer:
    """通用计时器装饰器

    重构要点:
    - 将重复的计时逻辑提取为装饰器
    - 支持函数和方法
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.elapsed_ms = 0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.time() - self.start) * 1000
        if self.name:
            print(f"  [{self.name}] 耗时: {self.elapsed_ms:.2f}ms")

    @staticmethod
    def decorate(func):
        """装饰器模式"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = (time.time() - start) * 1000
            print(f"  [{func.__name__}] 耗时: {elapsed:.2f}ms")
            return result
        return wrapper


# ============================================================
# 3. 重构后的RAG组件
# ============================================================

class RefactoredRetriever:
    """重构后的检索器

    使用策略模式, 支持多种检索方法的统一接口。
    """

    def __init__(self):
        self._strategies: Dict[str, Callable] = {}
        self._documents: List[Dict] = []

    def register_strategy(self, name: str, func: Callable):
        """注册检索策略"""
        self._strategies[name] = func

    def add_documents(self, documents: List[Dict]):
        """添加文档"""
        self._documents.extend(documents)

    def search(self, query: str, strategy: str = "keyword",
               top_k: int = 5) -> List[Dict]:
        """统一检索接口"""
        search_func = self._strategies.get(strategy)
        if not search_func:
            raise ValueError(f"未知检索策略: {strategy}")
        return search_func(self._documents, query, top_k)


# 预定义的检索策略函数
def keyword_search(documents: List[Dict], query: str, top_k: int) -> List[Dict]:
    """关键词检索策略"""
    query_words = set(query.lower().split())
    results = []
    for doc in documents:
        content_words = set(doc.get('content', '').lower().split())
        overlap = query_words & content_words
        if overlap:
            results.append({**doc, 'score': len(overlap) / len(query_words)})
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_k]


def semantic_search_mock(documents: List[Dict], query: str, top_k: int) -> List[Dict]:
    """模拟语义检索策略"""
    # 实际应用中替换为真正的向量检索
    import random
    results = []
    for doc in documents:
        score = random.uniform(0.5, 1.0)
        results.append({**doc, 'score': score})
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_k]


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("W34-D2 代码重构")
    print("=" * 60)

    # --- 1. 重构演示 ---
    demonstrate_refactoring()

    # --- 2. 通用组件 ---
    print(f"\n{'='*60}")
    print("--- 2. 通用组件模板 ---")
    print(f"{'='*60}")

    # 配置管理器
    config = ConfigManager()
    config.set_default('host', '0.0.0.0')
    config.set_default('port', 8000)
    config.set_default('debug', False)
    config.set_default('top_k', 5, validator=lambda v: 1 <= v <= 20)

    config.set('port', 9000)
    config.set('top_k', 10)
    print(f"配置: {config.get_all()}")

    # 数据仓库
    repo = Repository()
    doc_id = repo.create({'title': '测试文档', 'content': '内容'})
    print(f"创建文档: ID={doc_id}")
    print(f"获取文档: {repo.get(doc_id)}")
    print(f"文档数: {repo.count()}")

    # 计时器
    print("\n--- 计时器使用 ---")
    with Timer("数据处理"):
        time.sleep(0.01)
        sum(range(10000))

    # --- 3. 重构后的检索器 ---
    print(f"\n{'='*60}")
    print("--- 3. 策略模式检索器 ---")
    print(f"{'='*60}")

    retriever = RefactoredRetriever()
    retriever.register_strategy('keyword', keyword_search)
    retriever.register_strategy('semantic', semantic_search_mock)

    docs = [
        {'title': 'RAG技术', 'content': 'RAG是检索增强生成技术'},
        {'title': 'Python', 'content': 'Python是AI首选语言'},
        {'title': '向量检索', 'content': '向量检索基于语义匹配'},
    ]
    retriever.add_documents(docs)

    # 使用不同策略
    for strategy in ['keyword', 'semantic']:
        results = retriever.search('RAG检索', strategy=strategy, top_k=3)
        print(f"\n  策略[{strategy}]:")
        for r in results:
            print(f"    - {r['title']} (score: {r['score']:.3f})")

    print("\n完成!")
