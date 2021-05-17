import torch
from torch import nn


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()

    # 实现 forward 方法。 前馈操作
    def forward(self, input):
        input += 1
        return input

m =MyModule()
x = torch.tensor(1.0)
output =m(x)
print(output)

