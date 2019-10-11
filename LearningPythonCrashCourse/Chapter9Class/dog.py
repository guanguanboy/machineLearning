#在Python中，首字母大写的名称指的是类。这个类定义中括号是空的，因为我们要从空白创建这个类

class Dog():

    #如下是类的构造函数
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def sit(self):
        print(self.name.title() + " is now sitting.")

    def roll_over(self):
        print(self.name.title() + " roled over!")


my_dog = Dog('willie', 6)

print("My dog's name is " + my_dog.name.title() + ".")
print("My dog is " + str(my_dog.age) + " years old.") #句中的str将age的值6转换为字符串

my_dog.sit()
my_dog.roll_over()

your_dog = Dog('lucy', 3)

your_dog.sit()