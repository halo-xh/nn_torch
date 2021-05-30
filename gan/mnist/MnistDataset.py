import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
import pandas as pd
import matplotlib.pyplot as plt


class MnistDataset(Dataset):

    def __init__(self, csv_file) -> None:
        super().__init__()
        self.data_df = pd.read_csv(csv_file, header=None)

    def __le__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        # 目标图像(标签)
        label = self.data_df.iloc[index, 0]
        target = torch.zeros((10))
        target[label] = 1.0
        # 图像数据，取值范围是0-255，标准化为0-1
        image_values = torch.FloatTensor(self.data_df.iloc[index, 1:].values) / 255.0
        return label, image_values, target


    def plot_image(self, index):
        data = self.data_df.iloc[index, 1:].values.reshape(28, 28)
        plt.title('label = ' + str(self.data_df.iloc[index,0]))
        plt.imshow(data, interpolation='none', cmap='Blues')
        plt.show()
