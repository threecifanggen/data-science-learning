"""数据载入工具
"""
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from os import listdir
from os.path import join, isfile
from torchvision.io import image, read_image
import plotly.express as px

class LocalImageLoader(Dataset):
    """本地图片loader
    """

    def __init__(
            self,
            img_dir,
            target=None,
            img_filter='jpg',
            hd5_file_dir=None,
            using_hdf5=False,
            crop_size=None):
        self.img_dir = img_dir
        self.target = target # label
        self.img_name_list = [
            join(self.img_dir, file)
            for file in listdir(img_dir)
            if file[-3:] == img_filter
        ]
        self.crop_size = crop_size

        self.hdf5_file_dir = hd5_file_dir
        self.using_hdf5 = using_hdf5
        if hd5_file_dir is None:
            self.hdf5_file = None
        else:
            if isfile(hd5_file_dir):
                self.hdf5_file = h5py.File(hd5_file_dir, 'r')
            else:
                self.hdf5_file = None            

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        if self.using_hdf5:
            img = np.array(self.hdf5_file[idx])
            label = self.target[idx] if self.target is not None else None
            return torch.FloatTensor(img), label
        else:
            image = read_image(self.img_name_list[idx])
            label = self.target[idx] if self.target is not None else None
            if self.crop_size is not None:
                image = self.crop_center(
                    image,
                    self.crop_size[0],
                    self.crop_size[1]
                )
            return image, label

    @classmethod
    def crop_center(cls, img, new_width, new_height):
        _, height, width = img.shape
        startx = width // 2 - new_width // 2
        starty = height // 2 - new_height // 2
        return img[
            :,
            starty:starty + new_height,
            startx:startx + new_width,
        ]
        

    def plot_image(self, idx=0, is_cropped=False):
        """按idx绘制图像
        """
        if is_cropped and self.crop_size is not None:
            image = read_image(self.img_name_list[idx])
            image = self.crop_center(
                image,
                self.crop_size[0],
                self.crop_size[1]
            )
        else:
            image = read_image(self.img_name_list[idx])
        return px.imshow(image.T.transpose(0, 1))

    def load_in_hdf(self, file_dir=None):
        """将数据load到hdf文件中
        """
        if file_dir is None:
            file_dir = self.hdf5_file_dir
        else:
            self.hdf5_file_dir = file_dir
        with h5py.File(file_dir,'w') as h5f_file:
            image1 = read_image(self.img_name_list[0])
            img_ds = h5f_file.create_dataset(
                'images',
                shape=(self.__len__(), *image1.shape),
                dtype=int
            )
            for cnt, ifile in enumerate(self.img_name_list) :
                img = read_image(ifile)
                img_ds[cnt:cnt+1:,:,:] = img
            self.hdf5_file = h5f_file