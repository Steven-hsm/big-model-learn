import time
import functools

def timer(func):
    """记录函数执行时间的装饰器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[TIMER] {func.__name__}() 耗时 {elapsed:.6f} 秒")
        return result
    return wrapper

def retry(max_attempts=3, delay=1.0, exceptions=(Exception,)):
    """失败自动重试的装饰器"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        print(f"[RETRY] 第{attempt}次失败({e.__class__.__name__}: {e})，"
                              f"{delay}秒后重试...")
                        time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator

# 测试
@timer
def compute():
    return sum(i**2 for i in range(1000000))

@retry(max_attempts=5, delay=0.5, exceptions=(ConnectionError,))
def fetch_data():
    import random
    if random.random() < 0.6:
        raise ConnectionError("网络超时")
    return "数据获取成功"

if __name__ == '__main__':
    print(compute())
    print(fetch_data())