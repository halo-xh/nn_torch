import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt


class Classifymni(nn.Module):

    def __init__(self):
        super().__init__()

        '''
        对于简单的网络，我们可以使用nn.Sequential()，它允许我们提供一个网络模块的列表。
        '''
        self.modle = nn.Sequential(
            # nn.Linear(784, 200)是一个从784个节点到200个节点的全连接映射
            nn.Linear(784, 200),

            # 激活函数 带泄漏线性整流函数（Leaky ReLU）
            nn.LeakyReLU(0.02),
            # nn.Sigmoid()将S型逻辑激活函数应用于前一个模块的输出，也就是本例中200个节点的输出。
            # nn.Sigmoid(),

            # 神经网络中的权重和信号（向网络输入的数据）的取值范围都很大。之前，我们看到较大的输入值会导致饱和，使学习变得困难。
            # 大量研究表明，减少神经网络中参数和信号的取值范围，以及将均值转换为0，是有好处的。我们称这种方法为标准化 （normalization）。
            # 一种常见的做法是，在信号进入一个神经网络层之前将它标准化。
            nn.LayerNorm(200),

            # nn.Linear(200, 10)是将200个节点映射到10个节点的全连接映射。它包含中间隐藏层与输出层10个节点之间所有链接的权重。
            nn.Linear(200, 10),

            # 激活函数 带泄漏线性整流函数（Leaky ReLU）
            nn.LeakyReLU(0.02)
            # nn.Sigmoid()再将S型逻辑激活函数应用于10个节点的输出。其结果就是网络的最终输出。
            # nn.Sigmoid()
        )

        # '''
        # 定义网络误差的方法有多种，PyTorch为常用的方法提供了方便的函数支持。其中，最简单的是均方误差（mean squared error）。
        # 均方误差先计算每个输出节点的实际输出和预期输出之差的平方，再计算平均值。PyTorch将其定义为torch.nn.MSELoss()。
        # '''
        # self.loss_function = nn.MSELoss()

        '''
        一个分类（classification）任务，更适合使用其他损失函数。
        一种常用的损失函数是二元交叉熵损失 （binary cross entropy loss），
        它同时惩罚置信度（confidence）高的错误输出和置信值低的正确输出。PyTorch将其定义为nn.BCELoss()。
        '''
        self.loss_function = nn.BCELoss()

        '''
        创建优化器
        随机梯度下降（stochastic gradientdescent，SGD），将学习率设置为0.01
        '''
        # self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)

        '''
        随机梯度下降法的缺点之一是，它会陷入损失函数的局部最小值 （localminima）。另一个缺点是，它对所有可学习的参数都使用单一的学习率。
        可替代的方案有许多，其中最常见的是Adam。它直接解决了以上两个缺点。首先，它利用动量 （momentum）的概念，减少陷入局部最小值的可能性。
        我们可以想象一下，一个沉重的球如何利用动量滚过一个小坑。同时，它对每个可学习参数使用单独的学习率，这些学习率随着每个参数在训练期间的变化而改变。
        '''
        self.optimiser = torch.optim.Adam(self.parameters())

        '''
        计数。 查看训练进展
        '''
        self.counter = 0
        self.progress = []

    def forward(self, inputs):
        # 直接运行模型
        return self.modle(inputs)

    def train2(self, inputs, targets):
        outputs = self.forward(inputs)
        loss = self.loss_function(outputs, targets)
        self.counter += 1
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())
        if (self.counter % 10000 == 0):
            print('counter = %d', self.counter)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    def plot_progress(self):
        df = pd.DataFrame(self.progress, columns=['loss'])
        plt.plot(data=df, ylim=(0, 1.0), figsize=(16, 8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))
        plt.show()
