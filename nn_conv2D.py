import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

daloader = DataLoader(dataset, batch_size=64)


class MyMoul(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


writer = SummaryWriter("nnlogs")
mm = MyMoul()
# print(mm)
step = 0
for data in daloader:
    imgs, targets = data
    output = mm(imgs)
    # print(imgs.shape)
    # print(output.shape)
    # torch.Size([64, 3, 32, 32])
    writer.add_images("input", imgs)

    # torch.Size([64, 6, 30, 30])
    # 6 个channel 无法展示。 img 为3个channel
    # [64, 6, 30, 30] -> [xx,3,30,30]
    output = torch.reshape(output,(-1,3,30,30)) # 不知道第一个值为多少 设置为 -1 会自动根据后面的值计算
    writer.add_images("output", output, step)

    step += 1
writer.close() # tensorboard --logdir="nnlogs" 展示图片