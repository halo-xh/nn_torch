import torchvision

# train_set = torchvision.datasets.CIFAR10(root='./dataset', transform=torchvision.transforms.ToTensor(), train=True, download=False)
from torch.utils.data import DataLoader

# 获取数据集。
from torch.utils.tensorboard import SummaryWriter

test_set = torchvision.datasets.CIFAR10(root='./dataset', transform=torchvision.transforms.ToTensor(), train=False,
                                        download=False)

# 加载数据
test_loader = DataLoader(dataset=test_set, batch_size=4, shuffle=False, num_workers=0, drop_last=False) #  shuffle 摇色子咯

# 测试集中的第一章图片以及target
img, target = test_set[0]
print(img.shape)
print(target)

writer = SummaryWriter("dataloader")

for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        # print(imgs.shape)
        # print(targets)
        writer.add_images("Epoch: {}".format(epoch), imgs, step)
        step += 1
writer.close()
