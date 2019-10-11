def print_models(unprinted_designs, completed_models):
    """
    模拟打印每个设计，直到没有未打印的设计为止
    打印每个设计后，都将其移到列表completed_models中
    :param unprinted_designs:
    :param completed_models:
    :return:
    """
    while unprinted_designs:
        current_design = unprinted_designs.pop()

        #模拟根据设计制作3D打印模型的过程
        print("Printing model: " + current_design)
        completed_models.append(current_design)


def show_completed_models(completed_models):
    """显示打印好的所有模型"""
    print("\nThe following models have been printed:")
    for completed_model in completed_models:
        print(completed_model)


#以下为主程序
unprinted_designs = ['iphone case', 'robot pendant', 'dodecahedron']
completed_models = []
print_models(unprinted_designs, completed_models)
show_completed_models(completed_models)



"""
要将列表的副本传递给函数，可以像下面这样做：
function_name(list_name[:])
切片表示法[:] 创建列表的副本
在print_models.py中， 如果不想清空未打印的设计列表， 可像下面这样调用print_models() 
print_models(unprinted_designs[:], completed_models)

"""