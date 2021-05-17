from torch.utils.data import Dataset, ConcatDataset
from PIL import Image
from torch.utils.data.dataset import T_co
import os


class MyData(Dataset):

    def __init__(self,root_dir,label_dir) -> None:
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir)
        self.img_path = os.listdir(self.path)


    def __getitem__(self, index) -> T_co:
        img_name = self.img_path[index]
        img_item_path = os.path.join(self.root_dir,self.label_dir,img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img,label

    def __sizeof__(self) -> int:
        return len(self.img_path)


