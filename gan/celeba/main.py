import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':
    test_set = torchvision.datasets.CelebA(root='./dataset', transform=torchvision.transforms.ToTensor(), download=False)
    # 加载数据
    test_loader = DataLoader(dataset=test_set, batch_size=4, shuffle=False, num_workers=0, drop_last=False) #  shuffle 摇色子咯

    tb_writer = SummaryWriter("celebA")

    img, target = test_set[0]
    print(img.shape)
    print(target)