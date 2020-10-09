from d2l import torch as d2l
import torch
import torchvision
from torch import nn
import warnings

d2l.DATA_HUB['pokemon'] = (d2l.DATA_URL + 'pokemon.zip',
                           'c065c0e2593b8b161a2d7873e42418bf6a21106c')

data_dir = d2l.download_extract('pokemon')
print(data_dir)
pokemon = torchvision.datasets.ImageFolder(data_dir)
print(type(pokemon))

batch_size = 256
transformer = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64, 64)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(0.5, 0.5)
])

pokemon.transform = transformer

data_iter = torch.utils.data.DataLoader(
    pokemon, batch_size=batch_size,
    shuffle=True, num_workers=1
)

#let us visualize the first 20 images
warnings.filterwarnings('ignore')
d2l.set_figsize((4, 4))

for X, y in data_iter:
    imgs = X[0:20, :, :, :].permute(0, 2, 3, 1)/2 + 0.5
    d2l.show_images(imgs, num_rows=4, num_cols=5)
    d2l.plt.show()
    break

class G_block(nn.Module):
    def __init__(self, out_channels, in_channels=3, kernel_size=4, strides=2,
                 padding=1, **kwargs):
        super(G_block, self).__init__(**kwargs)
        self.conv2d_trans = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, strides, padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()


    def forward(self, X):
        return self.activation(self.batch_norm(self.conv2d_trans(X)))


## test convTranspose2d,默认放大1倍
x = torch.zeros((2, 3, 16, 16))
g_blk = G_block(20)
print(g_blk(x).shape)

## 4*4 kernel, 1*1 strides, zero padding, add 3
x = torch.zeros((2, 3, 1, 1))
g_blk = G_block(20, strides=1, padding=0)
print(g_blk(x).shape)


n_G = 64
net_G = nn.Sequential(
    G_block(in_channels=100, out_channels=n_G*8, strides=1, padding=0),
    G_block(in_channels=n_G*8, out_channels=n_G*4),
    G_block(in_channels=n_G*4, out_channels=n_G*2),
    G_block(in_channels=n_G*2, out_channels=n_G),
    nn.ConvTranspose2d(in_channels=n_G, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh()
)

#verify the generator's output shape
x = torch.zeros((1, 100, 1, 1))

print(net_G(x).shape)

class D_block(nn.Module):
    def __init__(self, out_channels, in_channels=3, kernel_size=4, strides=2,
                 padding=1, alpha=0.2, **kwargs):
        super(D_block, self).__init__(**kwargs)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size,
                                strides, padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(alpha, inplace=True)

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv2d(X)))


# test D_block
# A basic block with default setting will halve the width and height of inputs
x = torch.zeros((2, 3, 16, 16))
d_blk = D_block(20)
print(d_blk(x).shape)

# The discriminator is a mirror of the generator
# It uses a convolution layer with output channel 1 as the last layer to
# obtain a single prediction value
n_D = 64
net_D = nn.Sequential(
    D_block(n_D),
    D_block(in_channels=n_D, out_channels=n_D*2),
    D_block(in_channels=n_D*2, out_channels=n_D*4),
    D_block(in_channels=n_D*4, out_channels=n_D*8),
    nn.Conv2d(in_channels=n_D*8, out_channels=1, kernel_size=4, bias=False)
)


#test net_D shape
x = torch.zeros((1, 3, 64, 64))
print(net_D(x).shape)

def train(net_D, net_G, data_iter, num_epochs, lr, latent_dim, device=d2l.try_gpu()):
    loss = nn.BCEWithLogitsLoss(reduction='sum')

    for w in net_D.parameters():
        nn.init.normal_(w, 0, 0.02)

    for w in net_G.parameters():
        nn.init.normal_(w, 0, 0.02)

    net_D, net_G = net_D.to(device), net_G.to(device)
    trainer_hp = {'lr': lr, 'betas':[0.5, 0.999]}
    trainer_D = torch.optim.Adam(net_D.parameters(), **trainer_hp)
    trainer_G = torch.optim.Adam(net_G.parameters(), **trainer_hp)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs],
                            nrows=2, figsize=(5, 5),
                            legend=['discriminator', 'generator'])

    animator.fig.subplots_adjust(hspace=0.3)
    for epoch in range(1, num_epochs + 1):
        # Train one epoch
        timer = d2l.Timer()
        metric = d2l.Accumulator(3) # loss_D, loss_G, num_examples
        for X, _ in data_iter:
            batch_size = X.shape[0]
            Z = torch.normal(0, 1, size=(batch_size, latent_dim, 1, 1))
            X, Z = X.to(device), Z.to(device)
            metric.add(d2l.update_D(X, Z, net_D, net_G, loss, trainer_D),
                       d2l.update_G(Z, net_D, net_G, loss, trainer_G),
                       batch_size)

        # Show generated examples
        Z = torch.normal(0, 1, size=(21, latent_dim, 1, 1), device=device)

        # Normalize the synthetic data to N(0, 1)
        fake_x = net_G(Z).permute(0, 2, 3, 1) / 2 + 0.5
        imgs = torch.cat(
            [torch.cat([fake_x[i*7 + j].cpu().detach() for j in range(7)], dim=1) for i in range(len(fake_x)//7)],
            dim=0
        )

        animator.axes[1].cla()# 清除axes，即当前 figure 中的活动的axes，但其他axes保持不变。
        animator.axes[1].imshow(imgs)

        # Show the losses
        loss_D, loss_G = metric[0]/metric[2], metric[1]/metric[2]
        animator.add(epoch, (loss_D, loss_G))

        if epoch == 1:
            torch.save(net_D, "models/net_D.pth")
            torch.save(net_G, "models/net_G.pth")


    print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, '
          f'{metric[2]/ timer.stop():.1f} examples/sec on {str(device)}')

    d2l.plt.show()
    # print字符串前面加f表示格式化字符串，加f后可以在字符串里面使用花括号括起来的变量和表达式，
    # 如果字符串里面没有表达式，那么前面加不加f输出应该都一样.
    # 包含的{}表达式在程序运行时会被表达式的值代替。

latent_dim, lr, num_epochs = 100, 0.0005, 20
train(net_D, net_G, data_iter, num_epochs, lr, latent_dim)




