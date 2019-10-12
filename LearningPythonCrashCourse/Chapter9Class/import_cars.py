#导入整个模块
import electric_car

#导入整个模块时，需要使用句点表示法访问需要的类，这种导入方法很简单， 代码也易于阅读。 由于创建类实例的代码都包含模块名， 因此不会与当前文件使用的任何名称
#发生冲突。
my_beetle = electric_car.Car('voklwagen', 'beetle', 2016)
print(my_beetle.get_descriptive_name())

my_tesla = electric_car.ElectricCar('tesla', 'roadster', 2016)
print(my_tesla.get_descriptive_name())