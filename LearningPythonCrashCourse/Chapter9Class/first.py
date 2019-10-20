PI = 3.14

def print_PI():
    print("PI: ", PI)

#如果模块是被直接运行的，则代码块print_PI(),如果模块是被导入的，则代码块不会被执行
if (__name__ == "__main__"):
    print_PI()