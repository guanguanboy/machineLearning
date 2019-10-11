#演示传递变长的参数列表，传递任意数量的实参
#形参名*toppings 中的星号让Python创建一个名为toppings 的空元组， 并将收到的所有值都封装到这个元组中。
def make_pizza(*toppings):
    """打印顾客点的所有配料"""
    print(toppings)


make_pizza('pepperoni')
make_pizza('mushrooms', 'green peppers', 'extra cheese')

def make_pizza_1(*toppings):
    """概述要制作的比萨"""
    print("\nMaking a pizza with the following toppings:")
    for topping in toppings:
        print("- " + topping)


make_pizza_1('pepperoni')
make_pizza_1('mushrooms', 'green peppers', 'extra cheese')


#如下代码演示结合使用位置实参和任意数量实参
#如果要让函数接受不同类型的实参， 必须在函数定义中将接纳任意数量实参的形参放在最后。 Python先匹配位置实参和关键字实参， 再将余下的实参都收集到最后一个形参中。
def make_pizza_2(size, *toppings):
    """概述要制作的比萨"""
    print("\nMaking a " + str(size) +"-inch pizza with the following toppings:")
    for topping in toppings:
        print("- " + topping)


make_pizza_2(16, 'pepperoni')
make_pizza_2(12, 'mushrooms', 'green peppers', 'extra cheese')