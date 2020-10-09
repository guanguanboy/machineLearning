from d2l import torch as d2l

import torch
from torch import nn
import sys
sys.path.append("..")



#generate some real data
#generate data drawn from a Gaussian
X = torch.normal(0.0, 1, (1000, 2))
A = torch.tensor([[1, 2], [-0.1, 0.5]])
b = torch.tensor([1, 2])
data = torch.matmul(X, A) + b

#输出均值为0，标准差为1的 1000个2维的样本

print(X.shape)
print(data.shape)
print(data)
#This should be a Gaussian shifted in some rather arbitary way with mean b and covariance matrix A(T)A
d2l.set_figsize()
d2l.plt.scatter(d2l.numpy(data[:100, 0]), d2l.numpy(data[:100, 1]));
print(f'The covariance matrix is\n{torch.matmul(A.T, A)}')
d2l.plt.show()

batch_size = 8
data_iter = d2l.load_array((data, ), batch_size)

# Our generator network will be the simplest network possible- a single layer linear model.
# This is since we will be drving that linear network with a Gaussian data generator. Hence, it
# literally only needs to learn the parameters to fake things perfectly.
net_G = nn.Sequential(nn.Linear(2,2))

# Discriminator
# For the discriminator we will be a bit more discriminating: we will use an MLP with 3
# layers to make things a bit more interesting.
net_D = nn.Sequential(
    nn.Linear(2, 5),
    nn.Tanh(),
    nn.Linear(5, 3),
    nn.Tanh(),
    nn.Linear(3, 1)
)

#training

# first we define a function to update the discriminator.

def update_D(X, Z, net_D, net_G, loss, trainer_D):
    batch_size = X.shape[0]
    ones = torch.ones((batch_size,), device=X.device)
    zeros = torch.zeros((batch_size,), device=X.device)
    trainer_D.zero_grad()
    real_Y = net_D(X)
    fake_X = net_G(Z)

    # Do not need to compute gradient for 'net_G', detach it from computing gradients
    fake_Y = net_D(fake_X.detach())
    loss_D = (loss(real_Y, ones.reshape(real_Y.shape))
              + loss(fake_Y, zeros.reshape(fake_Y.shape)))/2
    loss_D.backward()
    trainer_D.step()

    return loss_D

#The generator is updated similarly.Here we reuse the cross-entropy loss but change the
#label of the fake data from 0 to 1.
def update_G(Z, net_D, net_G, loss, trainer_G):
    #update generator
    batch_size = Z.shape[0]
    ones = torch.ones((batch_size,), device=Z.device)
    trainer_G.zero_grad()

    # We could reuse 'fake_X' from 'update_D' to save computation
    fake_X = net_G(Z)

    #Recomputing 'fake_Y' is needed since 'net_D' is changed
    fake_Y = net_D(fake_X)
    loss_G = loss(fake_Y, ones.reshape((fake_Y.shape)))
    loss_G.backward()
    trainer_G.step()
    return loss_G

#Both the discriminator and the generator performs a binary logistic regression with
#the cross-entropy loss.
#We use Adam to smooth the trainning process.
#In each iteration, we first update the discriminator and
#then the generator, We visualize both losses and generated examples.
def train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G, latent_dim, data):
    loss = nn.BCEWithLogitsLoss(reduction='sum')
    for w in net_D.parameters():
        nn.init.normal_(w, 0, 0.02)

    for w in net_G.parameters():
        nn.init.normal_(w, 0, 0.02)

    trainer_D = torch.optim.Adam(net_D.parameters(), lr=lr_D)
    trainer_G = torch.optim.Adam(net_G.parameters(), lr=lr_G)

    animator = d2l.Animator(xlabel='epoch',
                            ylabel='loss',
                            xlim=[1, num_epochs],
                            nrows=2,
                            figsize=(5, 5),
                            legend=['discriminator', 'generator'])
    animator.fig.subplots_adjust(hspace=0.3)
    for epoch in range(num_epochs):
        #train one epoch
        timer = d2l.Timer()
        metric = d2l.Accumulator(3) # loss_D, loss_G, num_examples
        for (X,) in data_iter:
            batch_size = X.shape[0]
            Z = torch.normal(0, 1, size=(batch_size, latent_dim))
            metric.add(update_D(X, Z, net_D, net_G, loss, trainer_D),
                       update_G(Z, net_D, net_G, loss, trainer_G),
                       batch_size)

        #visualize generated examples
        Z = torch.normal(0, 1, size=(100, latent_dim))
        fake_X = net_G(Z).detach().numpy()

        animator.axes[1].cla()
        animator.axes[1].scatter(data[:, 0], data[:, 1])
        animator.axes[1].scatter(fake_X[:, 0], fake_X[:, 1])
        animator.axes[1].legend(['real', 'generated'])

        #show the losses
        loss_D, loss_G = metric[0]/metric[2], metric[1]/metric[2]
        animator.add(epoch+1, (loss_D, loss_G))

    print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f}'
          f'{metric[2]/timer.stop():.1f} examples/sec')

    d2l.plt.show()


#Now we specify the hyperparameters to fit the Gaussian distributtion
lr_D, lr_G, latent_dim, num_epochs = 0.05, 0.005, 2, 20
train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G,
      latent_dim, d2l.numpy(data[:100]))

"""
Summary
1,Generative adversarial networks composes of two deep networks,
the generator and the discriminator.

2,The generator generates the image as much closer to the true image as 
possible to fool the discriminator
via maximizing the cross-entropy loss.

3,The discriminator tries to distinguish the generated images
from the true images, via
minimizing the cross-entropy loss.

"""