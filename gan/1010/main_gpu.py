import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_real():
    real_data = torch.FloatTensor(
        [
            random.uniform(0.8, 1.0),
            random.uniform(0.0, 0.2),
            random.uniform(0.8, 1.0),
            random.uniform(0.0, 0.2),
        ]
    )
    return real_data


# 鉴别器
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        # 定义神经网络层
        self.model = nn.Sequential(
            nn.Linear(4, 3),
            nn.Sigmoid(),
            nn.Linear(3, 1),
            nn.Sigmoid()
        )

        # 创建损失函数
        self.loss_function = nn.MSELoss()

        # 创建优化器， 使用随机梯度下降
        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)

        # 计数器和进程记录
        self.counter = 0
        self.progress = []

    def forward(self, inputs):
        return self.model(inputs)

    def strain(self, inputs, targets):
        # 计算网络的输出
        outputs = self.forward(inputs)

        # 计算损失值
        loss = self.loss_function(outputs, targets)

        self.counter += 1
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())
        if (self.counter % 10000 == 0):
            print("counter = ", self.counter)

        # 归零梯度，反向传播，更新权重
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    def plot_progress(self):
        df = pd.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0, 1.0), figsize=(16, 8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))
        plt.show()


# 鉴别器
class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        # 定义神经网络层
        self.model = nn.Sequential(
            nn.Linear(1, 3),
            nn.Sigmoid(),
            nn.Linear(3, 4),
            nn.Sigmoid()
        )

        # 创建优化器， 使用随机梯度下降
        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)

        # 计数器和进程记录
        self.counter = 0
        self.progress = []

    def forward(self, inputs):
        return self.model(inputs)

    def strain(self, D, inputs, targets):
        # 计算网络的输出
        g_outputs = self.forward(inputs)

        # 输入鉴别器
        d_output = D.forward(g_outputs)

        # 计算损失值
        loss = D.loss_function(d_output, targets)

        self.counter += 1
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())

        # 归零梯度，反向传播，更新权重
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    def plot_progress(self):
        df = pd.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0, 1.0), figsize=(16, 8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))
        plt.show()


# 测试鉴别器 - 随机生成假数据
def generate_random(size):
    # generate_random（4）会返回一个包含4个0～1的值的张量。
    random_data = torch.rand(size)
    return random_data


if __name__ == '__main__':

    D = Discriminator()

    # 鉴别器 测试
    # for i in range(10000):
    #     # real data
    #     D.strain(generate_real(), torch.FloatTensor([1.0]))
    #     # random data
    #     D.strain(generate_random(4), torch.FloatTensor([0.0]))
    # D.plot_progress()

    # 测试
    # print(D.forward(generate_real()).item())
    # print(D.forward(generate_random(4)).item())

    G = Generator()

    # 生成器 测试
    # print(G.forward(torch.FloatTensor([0.5])))

    # ======== GAN ========
    image_list = []
    for i in range(10000):
        # real data - 第一步 用真实的数据训练鉴别器。
        D.strain(generate_real(), torch.FloatTensor([1.0]).cuda())

        # 用生成样本训练鉴别器 第2步 用一组生成数据来训练鉴别器
        # 使用detach() 以避免计算生成器中的梯度
        generated_data = G.forward(torch.FloatTensor([0.5]).cuda()).cpu().detach()
        D.strain(generated_data, torch.FloatTensor([0.0]).cuda())

        # 训练生成器
        G.strain(D, torch.FloatTensor([0.5]).cuda(), torch.FloatTensor([1.0]).cuda())

        # 记录生成器演变
        if (i % 1000 == 0):
            # 在这里，为了将生成器的输出张量以numpy数组的形式保存，我们需要在使用numpy()之前使用detach()将输出张量从计算图中分离出来。
            image_list.append(G.forward(torch.FloatTensor([0.5]).cuda()).cpu().detach().numpy())
    D.plot_progress()

    #演变 记录 可视化
    '''
    在训练之后，我们的image_list中应该有10个输出数组，每个数组包含4个值。
    下面，我们将每个输出转换成10 × 4的numpy数组，再将它对角翻转。
    这样做的目的是，方便我们观察它从左向右的演化过程。
    '''
    plt.figure(figsize=(16,8))
    print(image_list)
    plt.imshow(np.array(image_list).T,interpolation='none',cmap='Blues')
    plt.show()