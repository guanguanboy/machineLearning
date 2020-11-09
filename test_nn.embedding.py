import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable

#https://www.zhihu.com/search?type=content&q=nn.embedding

# 创建一个字典大小为3，词向量维度为50维的Embedding。
embed = nn.Embedding(3, 50)

# 创建一个二维LongTensor索引数据（作为索引数据的时候一般都使用LongTensor）
x = torch.LongTensor([[0, 1, 2]])  # x.size() --> torch.Size([1, 3])
print(x.shape)

# 将索引映射到向量
x_embed = embed(x)  # x_embed.size() --> torch.Size([1, 3, 50])  #1,3 来自于x.shape
#我们就完成了从索引转变成向量的过程
print(x_embed.shape)


#https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html#torch.nn.Embedding
# an Embedding module containing 10 tensors of size 3
embedding = nn.Embedding(10, 3)

# a batch of 2 samples of 4 indices each
input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
print("input_shape:")
print(input.shape) #torch.Size([2, 4])

res = embedding(input)
print(res)

print(res.shape) # torch.Size([2, 4, 3]),其中2，4由input的shape决定

img_shape=(1, 32, 32)
print(np.prod(img_shape))


z = Variable(torch.FloatTensor(np.random.normal(0, 1, (10 ** 2, 100))))

print(type(z)) #<class 'torch.Tensor'>
print(z.shape) #torch.Size([100, 100])

print(*img_shape)
print(type(*img_shape))

b = nn.Linear(1024, 5)
print(type(b))

