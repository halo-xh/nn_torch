import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt


class MnistGenerator(nn.Module):

    def __init__(self):
        super().__init__()

        '''
        对于简单的网络，我们可以使用nn.Sequential()，它允许我们提供一个网络模块的列表。
        '''
        self.modle = nn.Sequential(
            nn.Linear(100, 200),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(200),
            nn.Linear(200, 784),
            nn.Sigmoid()
        )

        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.01)

        '''
        计数。 查看训练进展
        '''
        self.counter = 0
        self.progress = []

    def forward(self, inputs):
        # 直接运行模型
        return self.modle(inputs)

    def train2(self, D, inputs, targets):
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
        plt.plot(data=df, ylim=(0), figsize=(16, 8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))
        plt.show()
