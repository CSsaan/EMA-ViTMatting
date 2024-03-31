import cv2
import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from config import *
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.utils.data as Data

# 自定义数据集加载的迭代器
# cv2.setNumThreads(1)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# class VimeoDataset(Dataset):
#     def __init__(self, dataset_name, path, batch_size=32, model="RIFE"):
#         self.batch_size = batch_size
#         self.dataset_name = dataset_name
#         self.model = model
#         self.data_root = f'{os.getcwd()}/{path}'
#         self.load_data()

#     def __len__(self):
#         return len(self.meta_data) # 数据集大小

#     def load_data(self):
#         normalize = transforms.Normalize([0.162, 0.151, 0.138], [0.058, 0.052, 0.048])
#         # 定义数据集处理方法变量
#         train_transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), normalize])
#         # 加载数据集
#         train_data = ImageFolder(self.data_root, transform=train_transform)
#         # split数据集
#         train_data, val_data = Data.random_split(train_data, [round(0.8*len(train_data)), round(0.2*len(train_data))])
        
#         if self.dataset_name != 'test':
#             self.meta_data = train_data
#         else:
#             self.meta_data = val_data
        
            
#     def __getitem__(self, index):
#         return self.meta_data(index)

# 图像分类任务，直接加载文件夹，自动根据文件夹来分成不同类的集合
def load_classification_data(dataset_name, path):
    data_root = f'{os.getcwd()}/{path}'
    normalize = transforms.Normalize([0.162, 0.151, 0.138], [0.058, 0.052, 0.048])
    # 定义数据集处理方法变量
    train_transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), normalize])
    # 加载数据集
    train_data = ImageFolder(data_root, transform=train_transform)
    # split数据集
    train_data, val_data = Data.random_split(train_data, [round(0.8*len(train_data)), round(0.2*len(train_data))])
        
    if dataset_name != 'test':
        return train_data
    else:
        return val_data