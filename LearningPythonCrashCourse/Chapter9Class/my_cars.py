#从electric_car模块中导入多个类
from electric_car import Car, ElectricCar

my_beetle = Car('voklwagen', 'beetle', 2016)
print(my_beetle.get_descriptive_name())

my_tesla = ElectricCar('tesla', 'roadster', 2016)
print(my_tesla.get_descriptive_name())