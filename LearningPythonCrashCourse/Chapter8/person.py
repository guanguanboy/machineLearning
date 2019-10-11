#演示返回一个字典:函数可返回任何类型的值， 包括列表和字典等较复杂的数据结构。 例如， 下面的函数接受姓名的组成部分， 并返回一个表示人的字典：
#可选形参age ， 并将其默认值设置为空字符串。 如果函数调用中包含这个形参的值， 这个值将存储到字典中。
def build_person(first_name, last_name, age=''):
    person = {'first':first_name, 'last':last_name}
    if age: #Python将非空字符串解读为True ， 因此如果函数调用中提供了age， if age 将为True
        person['age'] = age
    return person