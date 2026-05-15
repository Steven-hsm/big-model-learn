################列表推导式
# 基本语法：[表达式 for 变量 in 可迭代对象 if 条件]
squares = [x**2 for x in range(10)]
# [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# 带条件过滤
even_squares = [x**2 for x in range(10) if x % 2 == 0]
# [0, 4, 16, 36, 64]

# 带if-else（注意位置不同！）
result = [x if x > 0 else 0 for x in [-2, -1, 0, 1, 2]]
# [0, 0, 0, 1, 2]

# 嵌套循环（等价于双层for）
pairs = [(x, y) for x in range(3) for y in range(3) if x != y]
# [(0,1),(0,2),(1,0),(1,2),(2,0),(2,1)]

# 嵌套推导式（创建矩阵）
matrix = [[i*3+j for j in range(3)] for i in range(3)]
# [[0,1,2],[3,4,5],[6,7,8]]

# 展平矩阵
flat = [x for row in matrix for x in row]
# [0,1,2,3,4,5,6,7,8]

################字典推导式
# 基本语法：{key_expr: value_expr for item in iterable if condition}
word_lengths = {word: len(word) for word in ['hello', 'world', 'python']}
# {'hello': 5, 'world': 5, 'python': 6}

# 翻转字典
original = {'a': 1, 'b': 2, 'c': 3}
flipped = {v: k for k, v in original.items()}
# {1: 'a', 2: 'b', 3: 'c'}

# 从两个列表创建字典
keys = ['name', 'age', 'city']
values = ['Alice', 25, 'Beijing']
d = dict(zip(keys, values))  # 简洁写法

################集合推导式
# 基本语法：{expr for item in iterable if condition}
unique_lengths = {len(word) for word in ['hello', 'world', 'hi', 'hey']}

################生成器表达式 vs 列表推导式
import sys# 列表推导式：[] 立即计算，返回列表，占内存
list_comp = [x**2 for x in range(1000000)]
print(sys.getsizeof(list_comp))  # ~8MB

# 生成器表达式：() 惰性计算，返回生成器，省内存
gen_expr = (x**2 for x in range(1000000))
print(sys.getsizeof(gen_expr))   # ~200字节！

# 生成器只能遍历一次
for val in gen_expr:
    pass  # 第二次遍历会为空

################yield关键字与生成器函数
# 生成器函数：用yield代替return，每次yield产生一个值
def fibonacci(n):
    """生成前n个斐波那契数"""
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b
fib = fibonacci(10)
print(list(fib))  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

# 生成器的优势：处理大数据集时不会一次性加载到内存
def read_large_file(file_path):
    """逐行读取大文件"""
    with open(file_path) as f:
        for line in f:
            yield line.strip()

# yield from：委托给另一个生成器
def chain(*iterables):
    """将多个可迭代对象串联成一个连续的生成器"""
    for iterable in iterables:
        yield from iterable

list(chain([1,2], [3,4], [5]))

################惰性求值的实际应用
# 管道式数据处理（类似Java Stream）
def process_data(data):
    return (
        x * 2 for x in data        # 第一步：乘以2
        if x > 0                    # 过滤掉非正数
    )

list(process_data([-1,1,2]))

# 无限序列
from itertools import count, islice
evens = (x for x in count() if x % 2 == 0)
first_10_evens = list(islice(evens, 10))
# [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]