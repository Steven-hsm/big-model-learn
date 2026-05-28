############################函数定义与参数
# 基本函数定义
def greet(name):
    """文档字符串（docstring），类似Java的Javadoc"""
    return f"Hello, {name}!"

# 默认参数（Java不支持，类似方法重载的效果）
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

greet("Alice")                # "Hello, Alice!"
greet("Alice", "Hi")          # "Hi, Alice!"

# 关键字参数（调用时指定参数名，提高可读性）
def create_user(name, age, city="Beijing"):
    return {"name": name, "age": age, "city": city}
create_user(name="Alice", age=25, city="Shanghai")
create_user(age=30, name="Bob")  # 顺序无关

# *args：可变位置参数（收集为元组）
def sum_all(*args):
    return sum(args)
sum_all(1, 2, 3, 4)  # 10

# **kwargs：可变关键字参数（收集为字典）
def print_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")
print_info(name="Alice", age=25)

# 参数顺序规则：必选 → 默认 → *args → **kwargs
def func(a, b, c=10, *args, **kwargs):
    pass

# 返回多个值（实际返回元组，然后解包）
def min_max(numbers):
    return min(numbers), max(numbers)
lo, hi = min_max([3, 1, 4, 1, 5])  # lo=1, hi=5

#################Lambda表达式与高阶函数
# Lambda：匿名函数，类似Java的lambda，但更简洁
square = lambda x: x ** 2
add = lambda a, b: a + b
square(5)  # 25

# map：对每个元素应用函数（类似Java Stream.map）
list(map(lambda x: x ** 2, [1, 2, 3, 4]))  # [1, 4, 9, 16]

# filter：过滤元素（类似Java Stream.filter）
list(filter(lambda x: x > 3, [1, 2, 3, 4, 5]))  # [4, 5]

# reduce：累积计算（类似Java Stream.reduce）
from functools import reduce
reduce(lambda acc, x: acc + x, [1, 2, 3, 4], 0)  # 

# sorted的key参数
students = [('Alice', 85), ('Bob', 92), ('Charlie', 78)]
sorted(students, key=lambda s: s[1], reverse=True)
# [('Bob', 92), ('Alice', 85), ('Charlie', 78)]

# 函数作为参数传递（Python函数是一等公民）
def apply_operation(data, operation):
    return [operation(x) for x in data]

apply_operation([1, 2, 3], lambda x: x * 2)   # [2, 4, 6]
apply_operation([1, 2, 3], lambda x: x ** 2)  # [1, 4, 9]