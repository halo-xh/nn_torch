import time

import torchvision
from torch.utils.data import DataLoader

from mnist.Classifymni import Classifymni
from mnist.MnistDataset import MnistDataset

if __name__ == '__main__':
    C = Classifymni()
    epochs = 3
    mnist_dataset = MnistDataset('./mnist_train.csv')
    for i in range(epochs):
        for label,image_data_tensor,target_tensor in mnist_dataset:
            C.train2(image_data_tensor,target_tensor)
    C.plot_progress()

    #  测试 效果
    mnist_test_dataset = MnistDataset('./mnist_test.csv')
    score = 0
    items = 0

    for label, image_data_tensor, target_tensor in mnist_test_dataset:
        answer = C.forward(image_data_tensor).detach().numpy()
        if (answer.argmax() == label):
            score += 1
        items += 1

    print(score, items, score / items)
