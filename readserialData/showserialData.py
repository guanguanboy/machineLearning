import time
import serial
import matplotlib.pyplot as plt
import numpy as np

def convertBytesStringToIntArray(data):
    data_str = str(data, "utf-8")  # 将字节字符串转换为字符字符串

    data_str = data_str.lstrip()  # 用于截掉字符串左边的空格或指定字符。
    data_str = data_str.strip()  # 用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列，注意：该方法只能删除开头或是结尾的字符，不能删除中间部分的字符。

    splited_data_str = data_str.split(' ')  # 以空格为分隔符对字符串进行分割

    num_data = [int(x) for x in splited_data_str]  # 将字符串转换为int型数组

    return num_data

def IsNeedInterupt(record_count):
    if (record_count % 1000 == 0):
        isQ = input("是否要退出程序？（q：退出；其他：不退出，继续）")
        if isQ == "q":
            return True
        else:
            return False
    else:
        return False

def PlotRecord(record_data):
    # 获取数的类型：只有两种类型0和1
    record_type = record_data[2]

    # 去掉数据头，提取需要显示的数据
    plotted_data = record_data[4:]

    # 获取需要显示的数据的长度
    data_len = len(plotted_data)

    # 生成需要显示的数据的横坐标
    index_data = range(1, data_len + 1)

    if 0 == record_type:
        plt.plot(index_data, plotted_data, color = 'orange', linewidth = 5, label='series1')
    else:
        plt.plot(index_data, plotted_data, color ='blue', linewidth = 5, label='series2') #blueviolet

def ShowChart():
    plt.ion() #打开交互模式
    ax = plt.axes()
    plt.yticks([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000])
    ax.set_ylim(0, 8000) #将y坐标轴固定到0到8000

    # 绘制X轴，Y轴名称
    plt.xlabel('Index')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.show()
    plt.pause(0.02)
    plt.clf() #清空当前的绘制

ser = serial.Serial(
    port='COM6',
    baudrate=921600,
    parity=serial.PARITY_ODD,  # 校验位
    stopbits=serial.STOPBITS_TWO,  # 停止位
    bytesize=serial.SEVENBITS  # 数据位
)
data = ''

#实现每两条数据显示一次的方案：
#如果两条数据属于同一种类型，则只显示第二条数据。如果两条数据属于不同类型，则同时把两条数据画出来后再显示。 缺点：有些记录没有显示
#实现方法2：记录上条数据的类型和上条数据解析成整型数组的结果，如果当前条数据与上条数据属于不同的类型，则将当前条数据与上条数据一起显示，如果属于
#同一类型，则只显示当前条数据。缺点：处于变换点的记录显示了两次
record_count = 0 #记录总共收到多少条串口数据

last_record_type = -1
last_integer_array_record = []

while True:
    record_count += 1

    data = ser.readline() #从串口读取一条数据
    num_data = convertBytesStringToIntArray(data)
    record_type = num_data[2]
    PlotRecord(num_data)

    #判断是否把上条记录的数据也打印出来
    if (record_type != last_record_type) and (last_record_type != -1):
        PlotRecord(last_integer_array_record)

    ShowChart()

    #每一百条记录判断一下是否需要退出，如果是，则退出循环
    isInterupt = IsNeedInterupt(record_count)
    if(True == isInterupt):
        break

    #记录当前已显示的记录的类型和整型列表数据
    last_record_type = record_type
    last_integer_array_record = num_data





