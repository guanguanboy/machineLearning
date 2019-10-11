#如下演示基于顺序的传参方式:在函数中， 可根据需要使用任意数量的位置实参， Python将按顺序将函数调用中的实参关联到函数定义中相应的形参
def describe_pet(animal_type, pet_name):
    print("\nI have a " + animal_type + ".")
    print("My " + animal_type + "'s name is " + pet_name.title() + ".") #Python title() 方法返回"标题化"的字符串,就是说所有单词都是以大写开始，其余字母均为小写(见 istitle())。


#下面函数中，然而， Python依然将这个实参视为位置实参， 因此如果函数调用中只包含宠物的名字， 这个实参将关联到函数定义中的第一个形参。 这就是需要将pet_name 放在形参列表
#开头
def describe_pet_with_default_param(pet_name, animal_type='dog'):
    print("\nI have a " + animal_type + ".")
    print("My " + animal_type + "'s name is " + pet_name.title() + ".")