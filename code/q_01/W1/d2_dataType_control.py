######### 第二天 ################
#############数字类型
age = 25              # int，自动推断类型
price = 19.99         # float
complex_num = 3 + 4j  # 

print(f"age = {age},price = {price},complex_num = {complex_num}")

# 类型检查
print(type(age))             # <class 'int'>
print(isinstance(age, int))  # True，推荐用isinstance而非type()==

# 数学运算
print(f"10 / 3 = {10 / 3}")    # 3.3333...  注意：Python3的/总是返回float
print(f"10 // 3 = {10 // 3}")   # 3          整除
print(f"10 % 3 = {10 % 3}")    # 1          取模
print(f"2 ** 10 = {2 ** 10}")   # 1024       幂运算，Java中是Math.pow(2,10)

################字符串
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

################列表
fruits = ['apple', 'banana', 'cherry']
fruits.append('date')        # 末尾添加 → ['apple','banana','cherry','date']
fruits.insert(1, 'blueberry')# 指定位置插入
fruits.pop()                  # 删除并返回最后一个元素
fruits.pop(0)                 # 删除并返回指定索引
fruits.remove('banana')       # 按值删除（只删第一个）
fruits[0] = 'avocado'         # 修改元素
'cherry' in fruits             # True，成员检查
print(fruits)

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

################元组（tuple）—— 不可变列表
point = (3, 4)        # 创建后不能修改
x, y = point          # 解包（unpacking）
single = (1,)         # 单元素元组必须加逗号
# 元组作为字典的key（列表不行，因为列表可变）
coords = {(0,0): 'origin', (1,0): 'right'}

####################字典（dict）—— 类似Java的HashMap
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

####################集合（set）—— 类似Java的HashSet
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}

a | b    # {1,2,3,4,5,6}  并集
a & b    # {3,4}           交集
a - b    # {1,2}           差集
a ^ b    # {1,2,5,6}       对称差集

# 去重
list(set([1, 2, 2, 3, 3, 3]))  # [1, 2, 3]

####################控制流
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
for i in range(5):       # 0,1,2,3,
    print(i)
for i in range(2, 8):    # 2,3,4,5,6,7
    print(i)
for i in range(0, 10, 2):# 0,2,4,6,8
    print(i)

# while循环
count = 0
while count < 5:
    count += 1