#a = "我是真正全局变量"
def showvariable():
    b = "我一直是局部变量"
    print(a)

def showb():
    print(b)

#print(b) #报错 在函数showvariable的外部访问局部变量b
#showb() #报错 在函数showvariable的外部访问局部变量b

#下面讲讲global, global第一次是只能定义不能赋值的
def showglobalvariable():
    global a
    a = "我是global"
    print(a)

#showglobalvariable()

#当前global的变量是可以在函数外访问的
def show_global():
    print(a)

#showglobalvariable()
#show_global()

#但是必须是赋值之后访问才有意义，否则会报错
print(a) #还没有调用showglobalvariable对global变量a进行赋值就去访问，当然不能打印，报错了