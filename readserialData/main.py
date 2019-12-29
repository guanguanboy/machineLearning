import time
import serial
import matplotlib.pyplot as plt
import numpy as np

ser = serial.Serial(
    port='COM6',
    baudrate=921600,
    parity=serial.PARITY_ODD,  # 校验位
    stopbits=serial.STOPBITS_TWO,  # 停止位
    bytesize=serial.SEVENBITS  # 数据位
)
data = ''

count = 0
while True:
    data = ser.readline()
    t = time.time()
    ct = time.ctime(t)
    print(ct, ':')
    print(data)
    print(data.__len__())
    print(type(data))

    data_str = str(data, "utf-8")
    print(type(data_str))
    print("data_str = " + data_str)
    #print(data_str)
    data_str = data_str.lstrip()
    data_str = data_str.strip()

    print("striped data str = " + data_str)

    splited_data_str = data_str.split(' ')

    print("splited data str = " + str(splited_data_str))
    print("splited data str size = ", len(splited_data_str))

    num_data = [int(x) for x in splited_data_str]
    print("num_data = " + str(num_data))
    #print("num data size =" + len(num_data))
    # f = open('D:/test.txt', 'a')
    # f.writelines(ct)
    # f.writelines(':\n')
    # f.writelines(data.decode('utf-8'))
    # f.close()

    curve_type = num_data[2];
    plotted_data = num_data[4:]
    print(plotted_data)
    data_len = len(plotted_data)
    print("data len = " + str(data_len))
    index_data = range(1, data_len+1)
    print(index_data)

    plt.ion() #打开交互模式

    if 0 == curve_type:
        plt.plot(index_data, plotted_data, color = 'orange', linewidth = 5, label='series1')
    else:
        plt.plot(index_data, plotted_data, color ='blue', linewidth = 5, label='series2') #blueviolet

    plt.yticks([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000])

    #绘制X轴，Y轴名称
    plt.xlabel('Index')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.show()
    plt.pause(0.02)
    plt.clf()

    count += 1
    if (count % 1000 == 0):
        isQ = input("是否要退出程序？（q：退出；其他：不退出，继续）")
        if isQ == "q":
            print("感谢您的使用，再见！")
            break




