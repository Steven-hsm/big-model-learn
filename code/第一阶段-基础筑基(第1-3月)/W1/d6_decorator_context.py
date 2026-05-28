#实现AOP和资源管理的核心机制#
################装饰器

# 第一步：理解函数是对象
def greet():
    return "Hello"

say_hello = greet        # 函数可以赋值给变量
print(say_hello())       # Hello

# 第二步：理解函数可以作为参数
def call_func(func):
    return func()

call_func(greet)  # "Hello"

# 第三步：理解闭包（Closure）
def make_greeter(greeting):
    def greeter(name):
        return f"{greeting}, {name}!"
    return greeter

hi = make_greeter("Hi")
hi("Alice")  # "Hi, Alice!"

# 第四步：手写装饰器
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("函数执行前")
        result = func(*args, **kwargs)  # 调用原函数
        print("函数执行后")
        return result
    return wrapper

    # 第四步：手写装饰器
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("函数执行前")
        result = func(*args, **kwargs)  # 调用原函数
        print("函数执行后")
        return result
    return wrapper

# 使用@语法糖
@my_decorator
def say_hello(name):
    print(f"Hello, {name}!")
    return "done"  

say_hello("Alice")

#实用装饰器
import time
import functools

# @timer：记录函数执行时间
def timer(func):
    @functools.wraps(func)  # 保留原函数的元信息（__name__, __doc__）
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} 执行耗时: {elapsed:.4f}秒")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)
    return "done"

# @retry：失败自动重试
def retry(max_attempts=3, delay=1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts:
                        raise
                    print(f"第{attempt}次失败({e})，{delay}秒后重试...")
                    time.sleep(delay)
        return wrapper
    return decorator

@retry(max_attempts=3, delay=2)
def unstable_api_call():
    import random
    if random.random() < 0.7:
        raise ConnectionError("网络超时")
    return "成功"

# 多个装饰器叠加（执行顺序：从下到上包装，从上到下执行）
@timer        # 第2层包装
@retry(3)     # 第1层包装
def my_function():
    pass

###########上下文管理器（with语句）
# with语句：确保资源正确释放（类似Java的try-with-resources）
# 方式1：类实现__enter__/__exit__
class DatabaseConnection:
    def __init__(self, url):
        self.url = url

    def __enter__(self):
        print(f"连接数据库: {self.url}")
        self.connection = {"url": self.url, "status": "connected"}
        return self.connection  # with ... as conn 中的conn就是这个返回值

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("关闭数据库连接")
        self.connection["status"] = "closed"
        return False  # 返回True表示异常已处理，不再传播

with DatabaseConnection("localhost:5432") as conn:
    print(conn["status"])  # connected
# 自动调用__exit__，即使发生异常

# 方式2：contextlib.contextmanager装饰器（更简洁）
from contextlib import contextmanager

@contextmanager
def timer_context(label):
    start = time.perf_counter()
    yield  # yield处是with块内的代码执行位置
    elapsed = time.perf_counter() - start
    print(f"[{label}] 耗时: {elapsed:.4f}秒")

with timer_context("数据处理"):
    time.sleep(0.5)

###异常处理
# try / except / else / finally
try:
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"除零错误: {e}")
except (TypeError, ValueError) as e:
    print(f"类型或值错误: {e}")
else:
    print("没有异常时执行")
finally:
    print("总是执行")

# 主动抛出异常
def set_age(age):
    if age < 0:
        raise ValueError("年龄不能为负数")

# 自定义异常
class ModelError(Exception):
    def __init__(self, model_name, reason):
        self.model_name = model_name
        self.reason = reason
        super().__init__(f"模型 {model_name} 错误: {reason}")