import numpy as np
import cv2
import matplotlib.pyplot as plt


def rgb2ycbcr(rgb_image):
    rgb_image = rgb_image.astype(np.float32)
    # 1：创建变换矩阵，和偏移量
    transform_matrix = np.array([[0.257, 0.564, 0.098],
                                 [-0.148, -0.291, 0.439],
                                 [0.439, -0.368, -0.071]])
    shift_matrix = np.array([16, 128, 128])
    ycbcr_image = np.zeros(shape=rgb_image.shape)
    w, h, _ = rgb_image.shape
    # 2：遍历每个像素点的三个通道进行变换
    for i in range(w):
        for j in range(h):
            ycbcr_image[i, j, :] = np.dot(transform_matrix, rgb_image[i, j, :]) + shift_matrix
    return ycbcr_image


img = cv2.imread('./2.png')
plt.imshow(img)
plt.show()

img = rgb2ycbcr(img)

plt.imshow(img[:, :, 0])
plt.show()

plt.imshow(img[:, :, 1])
plt.show()

plt.imshow(img[:, :, 2])
plt.show()



img = cv2.resize(img, (224, 224))

dct_list = []
dct_mat = []

for index in range(3):
    img_perchannel = np.float32(img[:, :, index]) #分别获取各个通道的图
    dct = np.zeros_like(img_perchannel) #各个Y，cr，cb各个通道的分量图
    dct_matrix = np.zeros(shape=(28, 28, 64)) # dct_matrix 的shape是28*28*64, 有64个通道；28来自于224/8
    for i in range(0, img_perchannel.shape[0], 8): #以步长为8遍历分量图的宽
        for j in range(0, img_perchannel.shape[1], 8): #以步长为8遍历分量图的高
            dct[i:(i + 8), j:(j + 8)] = np.log(np.abs(cv2.dct(img_perchannel[i:(i + 8), j:(j + 8)])))
            # 对8*8的块做dct变换，其中img_perchannel[i:(i + 8), j:(j + 8)]是一个8*8的块

            dct_matrix[i // 8, j // 8, :] = dct[i:(i + 8), j:(j + 8)].flatten() #将变换后8*8的块展开成64个通道
            flatten_block = dct[i:(i + 8), j:(j + 8)].flatten()
            print(flatten_block.shape)

    dct_list.append(dct) # dct_list 后面没有使用
    print(type(dct_matrix))
    print(dct_matrix.shape)
    dct_mat.append(dct_matrix)

print(type(dct_mat))
print(dct_mat.__len__()) #dct_mat 长度为3，里面存储的是Y，cr，cb变换后的图像

img_num = 9
for i in range(img_num):
    img = dct_mat[0][:, :, i] # 这里只展示Y通道dct变换出来的前9个通道（将Y通道的数据变成了64个通道中的数据）的数据
    plt.subplot(img_num // 3, 3, i + 1)
    plt.imshow(img)

plt.show()

img_num = 9
for i in range(img_num):
    img = dct_mat[1][:, :, i] # 这里只展示cb通道dct变换出来的前9个通道（总共将cb通道的数据变换成了64个通道的数据）的数据
    plt.subplot(img_num // 3, 3, i + 1)
    plt.imshow(img)

plt.show()

img_num = 9
for i in range(img_num):
    img = dct_mat[2][:, :, i] # 这里只展示cr通道dct变换出来的前9个通道（总共将cr通道的数据变换成了64个通道的数据）的数据
    plt.subplot(img_num // 3, 3, i + 1)
    plt.imshow(img)

plt.show()