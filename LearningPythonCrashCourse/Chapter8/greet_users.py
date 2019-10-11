#该程序演示向函数传递列表
def greet_users(names):
    for name in names:
        msg = "Hello, " + name.title() + "!"
        print(msg)


usernames = {'hannah', 'ty', 'margot'} #定义了一个用户列表 usernames
greet_users(usernames)
