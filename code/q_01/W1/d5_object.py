###################类定义与基本用法
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
        # Python会把__id自动改名为_Student__id，防止子类意外覆盖
        # 访问方式：s._Student__id（外部仍可访问，Python没有真正的私有）
        # 区别：_single是约定私有，__double会触发名称修饰

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

s = Student("张三", 22)
s.add_score(85)
s.add_score(90)
print(s)           # 张三(22岁)     → 调用__str__
print(s.__repr__())     # Student(name='张三', age=22) → 调用__repr__
print(len(s))      # 2              → 调用__len__
print(s("你好"))    # 你好, 我是张三  → 调用__call__


###################类方法、静态方法、property
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

c = Circle(5)
print(c.radius)        # 5（通过@property getter）
c.radius = 10          # 通过@property setter
print(c.area)          # 314.159...（通过@property）
c2 = Circle.from_diameter(20)  # 通过@classmethod工厂
Circle.is_valid_radius(5)      # 通过@staticmethod


###################继承与多态
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
print(duck.speak())  # 嘎嘎！
print(duck.run())    # running
print(duck.swim())   # swimming


