# W01 - Python基础与环境搭建

> 第1周 | Java开发工程师转AI开发学习计划
> 工作日每晚2小时 | 周末6-8小时

---

## 一、本周目标

1. 搭建完整的Python开发环境（Python 3.11+ / VS Code / Git / Jupyter）
2. 掌握Python核心语法：数据类型、控制流、函数、面向对象
3. 理解Python特有概念：推导式、生成器、装饰器、上下文管理器
4. 掌握Python包管理和项目结构，能创建标准Python包
5. 完成5个练习项目，建立Python编程手感

---

## 二、时间安排

| 日期 | 类型 | 时长 | 主题 |
|------|------|------|------|
| Day 1 | 工作日晚 | 2h | 环境搭建 |
| Day 2 | 工作日晚 | 2h | 数据类型与控制流 |
| Day 3 | 工作日晚 | 2h | 函数与Lambda |
| Day 4 | 工作日晚 | 2h | 推导式与生成器 |
| Day 5 | 工作日晚 | 2h | 面向对象编程 |
| Day 6 | 周末 | 3-4h | 装饰器与上下文管理器 |
| Day 7 | 周末 | 3-4h | 项目结构与包管理 |

---

## 三、详细学习内容

### Day 1 - 环境搭建（2h）

**目标**：搭建完整的Python开发环境，能运行第一个Python程序。

**具体步骤**：

1. **安装 Miniconda（推荐）**
   - Miniconda 是 Anaconda 的精简版，自带 conda 包管理器和 Python，是 AI/数据科学领域的主流环境管理工具
   - 相比直接安装 Python 的优势：多版本 Python 共存、一键创建隔离环境、预装科学计算库、依赖冲突更少
   - 下载地址：https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/ （清华镜像，国内速度快）
   - Windows 选择 `Miniconda3-latest-Windows-x86_64.exe`
   - 安装时勾选 "Add Miniconda to my PATH environment variable"
   - **如果安装时没有勾选 PATH，手动配置方法**：
     - 找到 Miniconda 安装路径，默认为：
       - 当前用户安装：`C:\Users\你的用户名\miniconda3`
       - 所有用户安装：`C:\ProgramData\miniconda3`
     - 将以下路径添加到系统环境变量 PATH（按顺序）：
       ```
       C:\Users\你的用户名\miniconda3
       C:\Users\你的用户名\miniconda3\Scripts
       C:\Users\你的用户名\miniconda3\Library\bin
       ```
     - 添加方法：右键"此电脑" → 属性 → 高级系统设置 → 环境变量 → 系统变量 → Path → 编辑 → 新建 → 依次添加上面三个路径 → 确定保存
     - 如果安装时勾选了 "Register Miniconda as my default Python"，还需要将以下路径也加入 PATH：
       ```
       C:\Users\你的用户名\miniconda3\Library\mingw-w64\bin
       ```
     - **验证配置**：关闭所有终端窗口，重新打开一个新的终端，执行：
       ```bash
       conda --version      # 应输出 conda 版本号
       python --version     # 应输出 Python 版本号
       where conda          # 应输出 conda 所在路径
       ```
     - 如果提示 "conda 不是内部或外部命令"，说明 PATH 没配置对，检查路径是否正确
   - 验证安装：
     ```bash
     conda --version      # conda 25.x.x
     python --version     # Python 3.12.x
     ```
   - 配置 conda 镜像源（加速下载）：
     ```bash
     conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
     conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
     conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
     conda config --set show_channel_urls yes
     ```
   - 配置 pip 镜像源：
     ```bash
     pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
     ```
   - 创建学习用的虚拟环境：
     ```bash
     conda create -n ai-learn python=3.12 -y
     conda activate ai-learn          # 激活环境
     conda install numpy pandas matplotlib jupyter -y   # 安装常用库
     ```
   - 常用 conda 命令：
     ```bash
     conda env list                   # 查看所有环境
     conda activate ai-learn          # 激活环境
     conda deactivate                 # 退出环境
     conda install package_name       # 安装包
     conda list                       # 查看已安装包
     conda env export > env.yml       # 导出环境配置
     conda env create -f env.yml      # 从配置文件创建环境
     ```

2. **配置VS Code**
   - 安装扩展：Python（Microsoft官方）、Pylance（类型检查）、Jupyter（notebook支持）、GitLens
   - 配置 Python 解释器：Ctrl+Shift+P → "Python: Select Interpreter" → 选择 `ai-learn` 环境的 Python
   - 配置终端默认使用 PowerShell 或 Git Bash
   - 推荐设置（settings.json）：
     ```json
     {
       "python.defaultInterpreterPath": "~/miniconda3/envs/ai-learn/python.exe",
       "python.terminal.activateEnvironment": true,
       "editor.formatOnSave": true,
       "editor.tabSize": 4
     }
     ```

3. **配置Git**
   - `git config --global user.name "你的名字"`
   - `git config --global user.email "你的邮箱"`
   - 生成SSH密钥：`ssh-keygen -t rsa -b 4096`
   - 创建学习仓库：`mkdir big-model-learn && cd big-model-learn && git init`

4. **创建第一个程序**
   ```python
   # hello.py
   print("Hello, AI World!")

   # Python是动态类型语言，不需要声明变量类型
   name = "Java开发者"
   weeks = 8
   print(f"欢迎{name}，开始{weeks}周AI学习之旅！")  # f-string格式化

   # 查看Python版本和路径
   import sys
   print(f"Python版本: {sys.version}")
   print(f"Python路径: {sys.executable}")
   ```

---

### Day 2 - 数据类型与控制流（2h）

**目标**：掌握Python所有内置数据类型和流程控制语句。

**1. 基本数据类型**

```python
# 数字类型
age = 25              # int，自动推断类型
price = 19.99         # float
complex_num = 3 + 4j  # complex（Java没有的复数类型）

# 类型检查
type(age)             # <class 'int'>
isinstance(age, int)  # True，推荐用isinstance而非type()==

# 数学运算
10 / 3    # 3.3333...  注意：Python3的/总是返回float
10 // 3   # 3          整除
10 % 3    # 1          取模
2 ** 10   # 1024       幂运算，Java中是Math.pow(2,10)
```

**2. 字符串（str）**

```python
s = "Hello, Python"
s[0]           # 'H'    索引访问
s[-1]          # 'n'    负索引（从末尾开始）
s[0:5]         # 'Hello' 切片 [start:end]（左闭右开）
s[::2]         # 'Hlo yhn' 步长切片

# 常用方法
s.upper()      # 'HELLO, PYTHON'
s.lower()      # 'hello, python'
s.split(', ')  # ['Hello', 'Python']  按分隔符拆分
', '.join(['a', 'b'])  # 'a, b'       用分隔符连接
s.replace('Python', 'AI')  # 'Hello, AI'
f"结果是{42}"   # f-string格式化（Python 3.6+，推荐）

# Java对比：Python字符串方法直接用s.method()，而非String.method(s)
```

**3. 列表（list）—— 最常用，类似Java的ArrayList**

```python
fruits = ['apple', 'banana', 'cherry']

# 增删改查
fruits.append('date')        # 末尾添加 → ['apple','banana','cherry','date']
fruits.insert(1, 'blueberry')# 指定位置插入
fruits.pop()                  # 删除并返回最后一个元素
fruits.pop(0)                 # 删除并返回指定索引
fruits.remove('banana')       # 按值删除（只删第一个）
fruits[0] = 'avocado'         # 修改元素
'cherry' in fruits             # True，成员检查

# 切片（非常强大，Java没有的原生功能）
nums = [0, 1, 2, 3, 4, 5]
nums[1:4]     # [1, 2, 3]
nums[::2]     # [0, 2, 4]    步长2
nums[::-1]    # [5, 4, 3, 2, 1, 0]  反转
nums[2:]      # [2, 3, 4, 5]

# 列表操作
len(fruits)                   # 长度
sorted(fruits)                # 返回排序后的新列表
fruits.sort()                 # 原地排序（修改原列表）
list(range(10))               # [0, 1, 2, ..., 9]
```

**4. 元组（tuple）—— 不可变列表**

```python
point = (3, 4)        # 创建后不能修改
x, y = point          # 解包（unpacking）
single = (1,)         # 单元素元组必须加逗号

# 元组作为字典的key（列表不行，因为列表可变）
coords = {(0,0): 'origin', (1,0): 'right'}
```

**5. 字典（dict）—— 类似Java的HashMap**

```python
student = {
    'name': '张三',
    'age': 22,
    'scores': [85, 90, 78]
}

# 增删改查
student['gender'] = 'male'          # 添加/修改
student.get('gpa', 0.0)             # 安全获取，不存在返回默认值
student.pop('age')                   # 删除并返回
student.update({'gpa': 3.8})        # 批量更新

# 遍历（三种方式）
for key in student:                          # 遍历key
    print(key)
for key, value in student.items():           # 遍历键值对
    print(f"{key}: {value}")
for value in student.values():               # 遍历值
    print(value)

# 字典推导式
squared = {x: x**2 for x in range(5)}  # {0:0, 1:1, 2:4, 3:9, 4:16}
```

**6. 集合（set）—— 类似Java的HashSet**

```python
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}

a | b    # {1,2,3,4,5,6}  并集
a & b    # {3,4}           交集
a - b    # {1,2}           差集
a ^ b    # {1,2,5,6}       对称差集

# 去重
list(set([1, 2, 2, 3, 3, 3]))  # [1, 2, 3]
```

**7. 控制流**

```python
# if / elif / else（注意冒号和缩进！Python用缩进代替大括号）
score = 85
if score >= 90:
    grade = 'A'
elif score >= 80:
    grade = 'B'
elif score >= 70:   
    grade = 'C'
else:
    grade = 'F'

# for循环（Python的for是for-each，不是Java的for(;;)）
for fruit in ['apple', 'banana']:
    print(fruit)

# enumerate：同时获取索引和值（Java需要手动维护index变量）
for i, fruit in enumerate(['apple', 'banana']):
    print(f"{i}: {fruit}")

# zip：同时遍历多个列表
names = ['Alice', 'Bob']
ages = [25, 30]
for name, age in zip(names, ages):
    print(f"{name} is {age}")

# range
for i in range(5):       # 0,1,2,3,4
for i in range(2, 8):    # 2,3,4,5,6,7
for i in range(0, 10, 2):# 0,2,4,6,8

# while循环
count = 0
while count < 5:
    count += 1
```

---

### Day 3 - 函数与Lambda（2h）

**目标**：掌握Python函数的所有参数形式和高阶函数用法。

**1. 函数定义与参数**

```python
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
```

**2. Lambda表达式与高阶函数**

```python
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
reduce(lambda acc, x: acc + x, [1, 2, 3, 4], 0)  # 10

# sorted的key参数
students = [('Alice', 85), ('Bob', 92), ('Charlie', 78)]
sorted(students, key=lambda s: s[1], reverse=True)
# [('Bob', 92), ('Alice', 85), ('Charlie', 78)]

# 函数作为参数传递（Python函数是一等公民）
def apply_operation(data, operation):
    return [operation(x) for x in data]

apply_operation([1, 2, 3], lambda x: x * 2)   # [2, 4, 6]
apply_operation([1, 2, 3], lambda x: x ** 2)  # [1, 4, 9]
```

---

### Day 4 - 推导式与生成器（2h）

**目标**：掌握Python中极具表现力的推导式语法和惰性求值的生成器。

**1. 列表推导式（List Comprehension）**

```python
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
```

**2. 字典推导式**

```python
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
```

**3. 集合推导式**

```python
# 基本语法：{expr for item in iterable if condition}
unique_lengths = {len(word) for word in ['hello', 'world', 'hi', 'hey']}
# {5, 2}  自动去重
```

**4. 生成器表达式 vs 列表推导式**

```python
import sys

# 列表推导式：[] 立即计算，返回列表，占内存
list_comp = [x**2 for x in range(1000000)]
print(sys.getsizeof(list_comp))  # ~8MB

# 生成器表达式：() 惰性计算，返回生成器，省内存
gen_expr = (x**2 for x in range(1000000))
print(sys.getsizeof(gen_expr))   # ~200字节！

# 生成器只能遍历一次
for val in gen_expr:
    pass  # 第二次遍历会为空
```

**5. yield关键字与生成器函数**

```python
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
    for iterable in iterables:
        yield from iterable

list(chain([1,2], [3,4], [5]))  # [1, 2, 3, 4, 5]
```

**6. 惰性求值的实际应用**

```python
# 管道式数据处理（类似Java Stream）
def process_data(data):
    return (
        x * 2 for x in data        # 第一步：乘以2
        if x > 0                    # 过滤掉非正数
    )

# 无限序列
from itertools import count, islice
evens = (x for x in count() if x % 2 == 0)
first_10_evens = list(islice(evens, 10))
# [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
```

---

### Day 5 - 面向对象编程（2h）

**目标**：掌握Python OOP，理解与Java OOP的异同。

**1. 类定义与基本用法**

```python
class Student:
    """学生类 - 演示Python OOP"""

    # 类变量（类似Java static字段）
    school = "AI大学"

    def __init__(self, name, age):
        """构造函数（类似Java的构造器）
        self 类似 Java的 this，但必须显式写出
        """
        self.name = name    # 实例变量（直接赋值即可创建）
        self.age = age
        self._scores = []   # _前缀表示"protected"（约定，非强制）
        self.__id = "123"   # __前缀触发名称修饰（name mangling）

    def add_score(self, score):
        """实例方法"""
        self._scores.append(score)

    def average_score(self):
        return sum(self._scores) / len(self._scores) if self._scores else 0

    def __repr__(self):
        """类似Java的toString()，用于开发者调试"""
        return f"Student(name='{self.name}', age={self.age})"

    def __str__(self):
        """类似Java的toString()，用于用户展示"""
        return f"{self.name}({self.age}岁)"

    def __len__(self):
        """支持len()函数"""
        return len(self._scores)

    def __call__(self, greeting):
        """让对象可以像函数一样调用"""
        return f"{greeting}, 我是{self.name}"


# 使用
s = Student("张三", 22)
s.add_score(85)
s.add_score(90)
print(s)           # 张三(22岁)     → 调用__str__
print(repr(s))     # Student(name='张三', age=22) → 调用__repr__
print(len(s))      # 2              → 调用__len__
print(s("你好"))    # 你好, 我是张三  → 调用__call__
```

**2. 类方法、静态方法、property**

```python
class Circle:
    PI = 3.14159

    def __init__(self, radius):
        self._radius = radius  # "私有"属性

    @property
    def radius(self):
        """getter（类似Java的getRadius()）"""
        return self._radius

    @radius.setter
    def radius(self, value):
        """setter（类似Java的setRadius()）"""
        if value <= 0:
            raise ValueError("半径必须为正数")
        self._radius = value

    @property
    def area(self):
        """只读属性（类似Java的getArea()，但没有对应setter）"""
        return self.PI * self._radius ** 2

    @classmethod
    def from_diameter(cls, diameter):
        """工厂方法（类方法，类似Java的static工厂方法）
        cls是类本身，类似self是实例本身
        """
        return cls(diameter / 2)

    @staticmethod
    def is_valid_radius(radius):
        """静态方法（不依赖实例也不依赖类）
        类似Java的static方法
        """
        return radius > 0


# 使用
c = Circle(5)
print(c.radius)        # 5（通过@property getter）
c.radius = 10          # 通过@property setter
print(c.area)          # 314.159...（通过@property）
c2 = Circle.from_diameter(20)  # 通过@classmethod工厂
Circle.is_valid_radius(5)      # 通过@staticmethod
```

**3. 继承与多态**

```python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        """子类应该重写此方法（Python没有abstract关键字）"""
        raise NotImplementedError("子类必须实现speak方法")

class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name)  # 调用父类构造（类似Java的super()）
        self.breed = breed

    def speak(self):
        return f"{self.name}说：汪汪！"

class Cat(Animal):
    def speak(self):
        return f"{self.name}说：喵~"

# 多态（Python的鸭子类型，不需要共同接口）
animals = [Dog("旺财", "柴犬"), Cat("咪咪")]
for animal in animals:
    print(animal.speak())  # 运行时根据对象类型调用正确方法

# isinstance检查（类似Java的instanceof）
isinstance(Dog("旺财", "柴犬"), Animal)  # True

# 多重继承（Java不支持）
class Runnable:
    def run(self):
        return "running"

class Swimmable:
    def swim(self):
        return "swimming"

class Duck(Animal, Runnable, Swimmable):
    def speak(self):
        return "嘎嘎！"

duck = Duck("唐老鸭")
duck.speak()  # 嘎嘎！
duck.run()    # running
duck.swim()   # swimming
```

---

### Day 6 - 装饰器与上下文管理器（3-4h）

**目标**：理解装饰器和上下文管理器，这是Python中实现AOP和资源管理的核心机制。

**1. 装饰器原理**

```python
# 装饰器本质：一个接收函数并返回新函数的高阶函数
# 类似Java的注解+AOP，但更加灵活和透明

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

# 使用@语法糖
@my_decorator
def say_hello(name):
    print(f"Hello, {name}!")
    return "done"

say_hello("Alice")
# 输出：
# 函数执行前
# Hello, Alice!
# 函数执行后
```

**2. 实用装饰器**

```python
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
```

**3. 上下文管理器（with语句）**

```python
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
```

**4. 异常处理**

```python
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
```

---

### Day 7 - 项目结构与包管理（3-4h）

**目标**：掌握Python包管理工具，能创建标准项目结构。

**1. 虚拟环境（conda）**

```bash
# conda 是 AI/数据科学的标准环境管理工具
# 创建新环境（每个项目一个独立环境，避免依赖冲突）
conda create -n ai-learn python=3.12 -y
conda activate ai-learn
conda install numpy pandas matplotlib -y

# 管理环境
conda env list                          # 查看所有环境
conda deactivate                        # 退出当前环境
conda env remove -n ai-learn            # 删除环境
conda env export > environment.yml      # 导出环境配置（可复现）
conda env create -f environment.yml     # 从配置文件创建环境

# environment.yml 格式（替代 requirements.txt，conda 生态的标准格式）
# name: ai-learn
# channels:
#   - defaults
# dependencies:
#   - python=3.12
#   - numpy>=1.24
#   - pandas>=2.0
#   - pip:
#     - scikit-learn>=1.3
#     - torch>=2.0
```

**2. 包管理**

```bash
# pip基本操作
pip install numpy              # 安装包
pip install numpy==1.24.0      # 指定版本
pip install -r requirements.txt # 从文件安装
pip freeze > requirements.txt   # 导出当前环境的包
pip list                       # 列出已安装的包
pip show numpy                 # 查看包信息

# requirements.txt格式
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
```

**3. pyproject.toml（现代Python项目配置）**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "ai-learn-tools"
version = "0.1.0"
description = "AI学习工具包"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.24.0",
    "pandas>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "black>=23.0",
    "ruff>=0.1.0",
]

# 开发模式安装（修改代码立即生效，不用重新安装）
# pip install -e .
```

**4. 标准项目结构**

```
ai-learn-tools/
├── pyproject.toml          # 项目配置
├── README.md               # 项目说明
├── requirements.txt        # 依赖列表
├── src/                    # 源代码
│   └── ai_learn_tools/     # Python包（与项目名对应）
│       ├── __init__.py     # 包初始化，定义public API
│       ├── data_utils.py   # 数据处理��具
│       ├── math_utils.py   # 数学工具
│       └── viz_utils.py    # 可视化工具
├── tests/                  # 测试代码
│   ├── __init__.py
│   ├── test_data_utils.py
│   └── test_math_utils.py
└── docs/                   # 文档
    └── tutorial.md
```

**5. 模块导入**

```python
# ai_learn_tools/__init__.py
from .data_utils import load_csv, clean_data
from .math_utils import cosine_similarity

__all__ = ['load_csv', 'clean_data', 'cosine_similarity']
__version__ = '0.1.0'

# ai_learn_tools/math_utils.py
import numpy as np

def cosine_similarity(a, b):
    """计算两个向量的余弦相似度"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 使用
from ai_learn_tools import cosine_similarity
from ai_learn_tools.math_utils import cosine_similarity  # 等效
from ai_learn_tools import math_utils as mu              # 别名导入
```

---

## 四、代码练习

### Day 2 练习：学生信息管理器

```python
# student_manager.py - 用dict存储学生信息，实现CRUD操作
# 要求：
# 1. 每个学生包含：id, name, age, scores(list), grade
# 2. 实现以下功能：
#    - add_student(students, student_data) -> 添加学生
#    - find_student(students, student_id) -> 按ID查找
#    - update_student(students, student_id, **kwargs) -> 更新信息
#    - delete_student(students, student_id) -> 删除学生
#    - list_students(students, sort_by='name') -> 列出所有学生
#    - get_statistics(students) -> 返回统计信息（平均分、最高分等）

def add_student(students, student_data):
    sid = student_data['id']
    if sid in students:
        raise ValueError(f"学生ID {sid} 已存在")
    students[sid] = student_data.copy()
    return students[sid]

def find_student(students, student_id):
    return students.get(student_id, None)

def update_student(students, student_id, **kwargs):
    if student_id not in students:
        raise KeyError(f"学生ID {student_id} 不存在")
    students[student_id].update(kwargs)
    return students[student_id]

def delete_student(students, student_id):
    if student_id in students:
        return students.pop(student_id)
    return None

def list_students(students, sort_by='name'):
    return sorted(students.values(), key=lambda s: s.get(sort_by, ''))

def get_statistics(students):
    all_scores = [s for stu in students.values() for s in stu.get('scores', [])]
    return {
        'count': len(students),
        'avg_score': sum(all_scores) / len(all_scores) if all_scores else 0,
        'max_score': max(all_scores) if all_scores else 0,
        'min_score': min(all_scores) if all_scores else 0,
    }

# 测试
if __name__ == '__main__':
    students = {}
    add_student(students, {'id': 1, 'name': '张三', 'age': 22, 'scores': [85, 90], 'grade': 'A'})
    add_student(students, {'id': 2, 'name': '李四', 'age': 23, 'scores': [78, 82], 'grade': 'B'})
    print(list_students(students))
    print(get_statistics(students))
```

### Day 3 练习：5个工具函数

```python
# utils.py - 5个常用工具函数

# 1. 字符串反转（多种实现）
def reverse_string(s):
    return s[::-1]  # 切片反转

# 2. 列表去重（保持顺序）
def unique_list(lst):
    seen = set()
    return [x for x in lst if not (x in seen or seen.add(x))]

# 3. 字典按值排序
def sort_dict_by_value(d, reverse=False):
    return dict(sorted(d.items(), key=lambda item: item[1], reverse=reverse))

# 4. 文件读写
def read_write_file(filepath, content=None):
    if content is not None:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

# 5. CSV解析（不使用csv模块，手动实现）
def parse_csv(csv_text):
    lines = csv_text.strip().split('\n')
    headers = lines[0].split(',')
    data = []
    for line in lines[1:]:
        values = line.split(',')
        data.append(dict(zip(headers, values)))
    return data
```

### Day 5 练习：SimpleORM类

```python
# simple_orm.py - 简易ORM实现
# 要求：支持 define_table / insert / find / delete 操作

class SimpleORM:
    def __init__(self):
        self._tables = {}

    def define_table(self, table_name, columns):
        """定义表结构"""
        self._tables[table_name] = {
            'columns': columns,
            'rows': [],
            'next_id': 1
        }

    def insert(self, table_name, **data):
        """插入一条记录"""
        if table_name not in self._tables:
            raise ValueError(f"表 {table_name} 不存在")
        table = self._tables[table_name]
        row = {'id': table['next_id']}
        for col in table['columns']:
            row[col] = data.get(col, None)
        table['rows'].append(row)
        table['next_id'] += 1
        return row

    def find(self, table_name, **conditions):
        """按条件查询"""
        if table_name not in self._tables:
            raise ValueError(f"表 {table_name} 不存在")
        rows = self._tables[table_name]['rows']
        if not conditions:
            return rows
        return [r for r in rows if all(r.get(k) == v for k, v in conditions.items())]

    def delete(self, table_name, **conditions):
        """按条件删除"""
        if table_name not in self._tables:
            raise ValueError(f"表 {table_name} 不存在")
        table = self._tables[table_name]
        original_count = len(table['rows'])
        table['rows'] = [r for r in table['rows']
                         if not all(r.get(k) == v for k, v in conditions.items())]
        return original_count - len(table['rows'])

# 测试
if __name__ == '__main__':
    db = SimpleORM()
    db.define_table('students', ['name', 'age', 'grade'])
    db.insert('students', name='张三', age=22, grade='A')
    db.insert('students', name='李四', age=23, grade='B')
    print(db.find('students', grade='A'))
    print(db.delete('students', name='张三'))
    print(db.find('students'))
```

### Day 6 练习：@timer 和 @retry 装饰器

```python
# decorators.py
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
```

### 周末综合练习：创建完整Python包

创建一个名为 `ai_learn_tools` 的完整Python包项目，包含：
1. 标准目录结构（src/tests/docs）
2. 至少2个功能模块（data_utils.py, math_utils.py）
3. `__init__.py` 定义公共API
4. requirements.txt
5. pyproject.toml
6. 基本测试文件
7. 用 `pip install -e .`（开发模式安装）或 `conda develop .` 安装并验证能正常导入

---

## 五、本周产出

| 产出物 | 说明 |
|--------|------|
| Python开发环境 | Python 3.11+ / VS Code / Git / Jupyter 全部配置好 |
| student_manager.py | 学生信息CRUD管理器 |
| utils.py | 5个工具函数集合 |
| simple_orm.py | 简易ORM实现 |
| decorators.py | @timer和@retry装饰器 |
| ai_learn_tools包 | 完整的Python包项目 |
| Git仓库 | 所有代码提交到GitHub |

---

## 六、自测题

**答题后对照上方学习内容检验理解程度。**

1. **Python中list和tuple的区别？什么时候用tuple？**

   <details>
   <summary>参考答案</summary>

   - list可变（mutable），tuple不可变（immutable）
   - list有append/insert/pop等修改方法，tuple没有
   - tuple可作为dict的key，list不行（可变对象不能hash）
   - 用tuple的场景：函数返回多个值、字典的key、常量数据、数据不需要修改时
   </details>

2. ***args和**kwargs的区别？**

   <details>
   <summary>参考答案</summary>

   - `*args`收集多余的位置参数为元组，如 `def f(*args): print(args)` → `f(1,2,3)` 输出 `(1,2,3)`
   - `**kwargs`收集多余的关键字参数为字典，如 `def f(**kwargs): print(kwargs)` → `f(a=1,b=2)` 输出 `{'a':1,'b':2}`
   - 可以同时使用：`def f(*args, **kwargs)`
   </details>

3. **列表推导式和生成器表达式的区别？内存占用差异？**

   <details>
   <summary>参考答案</summary>

   - 列表推导式用 `[]`，生成器表达式用 `()`
   - 列表推导式立即计算所有值并存储在内存中
   - 生成器表达式惰性求值，每次只产生一个值，内存占用极小（约200字节 vs 几十MB）
   - 生成器只能遍历一次，列表可以反复遍历
   - 当数据量大且只需遍历一次时，优先用生成器
   </details>

4. **装饰器的执行顺序（多个装饰器叠加时）？**

   <details>
   <summary>参考答案</summary>

   ```python
   @decorator_a   # 等价于 func = decorator_a(decorator_b(func))
   @decorator_b
   def func():
       pass
   ```
   - 包装顺序：从下到上（先decorator_b包装，再decorator_a包装）
   - 执行顺序：从外到内（先decorator_a的逻辑，再decorator_b的逻辑，最后原函数）
   </details>

5. **__str__和__repr__的区别？**

   <details>
   <summary>参考答案</summary>

   - `__str__`：面向用户，`print(obj)` 或 `str(obj)` 时调用，要求可读性好
   - `__repr__`：面向开发者，`repr(obj)` 或在交互式环境中直接输入变量名时调用，要求精确无歧义
   - 如果只实现一个，优先实现 `__repr__`（因为当 `__str__` 未定义时会fallback到 `__repr__`）
   - 理想情况下 `eval(repr(obj)) == obj`
   </details>

---

## 七、Java开发者提示

| Java概念 | Python对应 | 说明 |
|----------|-----------|------|
| `public`/`private`/`protected` | 无关键字，用约定 | `_var` = protected, `__var` = private (名称修饰) |
| `interface` | 无interface，用鸭子类型 | "如果它走起来像鸭子，叫起来像鸭子，那它就是鸭子" |
| 注解 + AOP | 装饰器 `@decorator` | 装饰器更灵活，本质是高阶函数 |
| `try-with-resources` | `with` 语句 | 上下文管理器，自动释放资源 |
| `ArrayList` | `list` | Python列表更强大，支持切片 |
| `HashMap` | `dict` | 字面量语法更简洁 `{}` |
| `HashSet` | `set` | 同样是哈希集合 |
| `static方法` | `@staticmethod` | 不依赖实例状态 |
| `toString()` | `__str__`/`__repr__` | 两个方法，分工不同 |
| 泛型 | 无泛型 | 动态类型，运行时检查 |
| 方法重载 | 不支持 | 用默认参数和 `*args`/`**kwargs` 替代 |
| `this` | `self` | `self` 必须显式写在参数列表中 |
| `new Object()` | `Object()` | 不需要 `new` 关键字 |
| `==` 比较引用 | `==` 比较值 | Python中 `==` 比较值相等，`is` 比较引用相同 |
| 块用 `{}` | 缩进 | Python用4空格缩进表示代码块，没有大括号 |

**重要思维转换**：
- Python是动态类型语言，变量不需要声明类型（但可以用类型注解：`def greet(name: str) -> str`）
- Python的哲学是"简单优于复杂"（import this 查看Python之禅）
- Python没有编译阶段，错误在运行时才发现。建议使用类型注解 + mypy做静态检查。
