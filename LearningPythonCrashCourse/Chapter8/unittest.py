from greeter import greet_user
from greeter import greet_user_with_para
from pets import describe_pet
from pets import describe_pet_with_default_param
from formatted_name import get_formatted_name
from person import build_person

greet_user()
greet_user_with_para('Jesse')

#如下演示基于顺序的传参方式:在函数中， 可根据需要使用任意数量的位置实参， Python将按顺序将函数调用中的实参关联到函数定义中相应的形参
describe_pet('hamster', 'harry')
describe_pet('dog', 'willie')

#如下演示基于关键字的实参传递方式：是传递给函数的名称—值对。 你直接在实参中将名称和值关联起来了， 因此向函数传递实参时不会混淆
describe_pet(animal_type='hamster', pet_name='harry')
describe_pet(pet_name='harry', animal_type='hamster') #关键字实参的顺序无关紧要， 因为Python知道各个值该存储到哪个形参中。


describe_pet_with_default_param('willie')

musician = get_formatted_name('jimi', 'hendrix') #用返回值的函数时， 需要提供一个变量， 用于存储返回的值。 在这里， 将返回值存储在了变量musician 中
print(musician)

musician = build_person('jimi', 'hendrix', age=27)
print(musician)

while True:
    print("\nPlease tell me your name:")
    print("(ente 'q' at any time to quit)")

    f_name = input("First name: ") #接受用户输入的方式
    if f_name == 'q':
        break

    l_name = input("Last name: ")
    if l_name == 'q':
        break

    formatted_name = get_formatted_name(f_name, l_name)
    print("\nHell, " + formatted_name + "!")