import torch
import matplotlib.pyplot as plt

from gan.mnist.MnistDataset import MnistDataset
from gan.mnist.MnistDiscriminator import MnistDiscriminator
from gan.mnist.MnistGenerator import MnistGenerator


def generate_random(size):
    # generate_random（4）会返回一个包含4个0～1的值的张量。
    random_data = torch.rand(size)
    return random_data


# 解决模式崩溃
def generate_random_image(size):
    random_data = torch.rand(size)
    return random_data


def generate_random_seed(size):
    return torch.randn(size)


if __name__ == '__main__':
    G = MnistGenerator()
    D = MnistDiscriminator()

    # 测试鉴别器
    # mini_dataset = MnistDataset('../../mnist/mnist_test.csv')
    # for label,image_data_tensor,target_tensor in mini_dataset:
    #     D.strain(image_data_tensor,torch.FloatTensor([1.0]))
    #     D.strain(generate_random(784),torch.FloatTensor([0.0]))
    # D.plot_progress()

    # 测试生成器
    # output = G.forward(generate_random(100))
    # img = output.detach().numpy().reshape(28, 28)
    # plt.imshow(img, interpolation='none', cmap='Blues')
    # plt.show()

    # == GAN ==
    mini_dataset = MnistDataset('../../mnist/mnist_train.csv')
    for label, image_data_tensor, target_tensor in mini_dataset:
        # 真实数据进行鉴别
        D.strain(image_data_tensor, torch.FloatTensor([1.0]))
        # 生成数据进行鉴别
        # 使用detach 避免计算生成器G的梯度
        D.strain(G.forward(generate_random_seed(100)).detach(), torch.FloatTensor([0.0]))
        # 训练生成器
        G.train2(D, generate_random_seed(100), torch.FloatTensor([1.0]))
    D.plot_progress()
    G.plot_progress()

    # 查看生成结果
    f, axarr = plt.subplots(2, 3, figsize=(16, 8))
    for i in range(2):
        for j in range(3):
            output = G.forward(generate_random_seed(100))
            img = output.detach().numpy().reshape(28, 28)
            axarr[i, j].imshow(img, interpolation='none', cmap='Blues')
    plt.show()

    # 种子试验
    seed1 = generate_random_seed(100)
    out1 = G.forward(seed1)
    img1 = out1.detach().numpy().reshape(28, 28)
    plt.imshow(img1, interpolation='none', cmap='Blues')

    seed2 = generate_random_seed(100)
    out2 = G.forward(seed2)
    img2 = out2.detach().numpy().reshape(28, 28)
    plt.imshow(img2, interpolation='none', cmap='Blues')

    # 观察种子之间的变化
    count = 0
    f, a = plt.subplots(3, 4, figsize=(16, 8))
    for i in range(3):
        for j in range(4):
            seed = seed1 + (seed2 - seed1) / 11 * count
            output = G.forward(seed)
            img = output.detach().numpy().reshape(28, 28)
            a[i, j].imshow(img, interpolation='none', cmap='Blues')
            count += 1

    # 观察相加
    seed = seed1 + seed2
    output = G.forward(seed)
    img = output.detach().numpy().reshape(28, 28)
    plt.imshow(img, interpolation='none', cmap='Blues')

    # 观察相减
    seed = seed1 - seed2
    output = G.forward(seed)
    img = output.detach().numpy().reshape(28, 28)
    plt.imshow(img, interpolation='none', cmap='Blues')

    plt.show()
