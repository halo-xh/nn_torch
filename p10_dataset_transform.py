import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_set = torchvision.datasets.CIFAR10(root='./dataset', transform=dataset_transform, train=True, download=False)
test_set = torchvision.datasets.CIFAR10(root='./dataset', transform=dataset_transform, train=False, download=False)

# img, target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()

# 画图展示 Tensor -> im
writer = SummaryWriter("p10")
for i in range(10):
    img, target = train_set[i]
    writer.add_image("test_set", img, i)
writer.close()
