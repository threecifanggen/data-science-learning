# import cv2
from torch.utils.data import Dataset
from os import listdir
from os.path import join
from torchvision.io import read_image
import plotly.express as px

class LocalImageLoader(Dataset):
    """本地图片loader
    """

    def __init__(self, img_dir, target=None, img_filter='jpg'):
        self.img_dir = img_dir
        self.target = target
        self.img_name_list = [join(self.img_dir, file) for file in listdir(img_dir) if file[-3:] == img_filter]

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        image = read_image(self.img_name_list[idx])
        label = self.target[idx] if self.target is not None else None
        return image, label

    def plot_image(self, idx=0):
        image = read_image(self.img_name_list[idx])
        return px.imshow(image.T)
