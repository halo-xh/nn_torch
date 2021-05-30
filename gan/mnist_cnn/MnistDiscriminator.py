import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt


class MnistDiscriminator(nn.Module):

    def __init__(self):
        super().__init__()

        '''
        对于简单的网络，我们可以使用nn.Sequential()，它允许我们提供一个网络模块的列表。
        '''
        self.model = nn.Sequential(
            # nn.Linear(784, 200)是一个从784个节点到200个节点的全连接映射
            nn.Linear(784, 200),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(200),
            nn.Linear(200, 1),
            nn.Sigmoid()
        )

        # 创建损失函数
        self.loss_function = nn.BCELoss()

        # 创建优化器， 使用随机梯度下降
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.01)

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
        df.plot(ylim=(0), figsize=(16, 8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))
        plt.show()
