from d2l import torch as d2l
import torch
import torchvision
from torch import nn
import warnings

d2l.DATA_HUB['pokemon'] = (d2l.DATA_URL + 'pokemon.zip',
                           'c065c0e2593b8b161a2d7873e42418bf6a21106c')

data_dir = d2l.download_extract('pokemon')

pokemon = torchvision.datasets.ImageFolder(data_dir)

batch_size = 256
transformer = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64, 64)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(0.5, 0.5)
])

pokemon.transforms = transformer
data_dir = torch.utils.data.DataLoader(
    pokemon, batch_size=batch_size,
    shuffle=True, num_workers=1
)

#let us visualize the first 20 images
warnings.filterwarnings('ignore')
d2l.set_figsize((4, 4))

for X, y in data_dir:
    imgs = X[0:20, :, :, :].permute(0, 2, 3, 1)/2 + 0.5
    d2l.show_images(imgs, num_rows=4, num_cols=5)
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

