import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

class AIM500Dataset(Dataset):
    def __init__(self, dataset_name, root_dir):
        self.root_dir = root_dir
        if dataset_name == 'train':
            self.image_folder = os.path.join(root_dir, 'train', 'original')
            self.label_folder = os.path.join(root_dir, 'train', 'mask')
        elif dataset_name == 'test':
            self.image_folder = os.path.join(root_dir, 'test', 'original')
            self.label_folder = os.path.join(root_dir, 'test', 'mask')
        else:
            raise ValueError("Invalid dataset_name. Choose 'train' or 'test'.")
        
        self.image_filenames = os.listdir(self.image_folder)
        self.label_filenames = os.listdir(self.label_folder)
        
        # Transformations for images and labels
        self.transform_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.50542366, 0.46995255, 0.44692866), (0.28501507, 0.27542947, 0.28659645)),
            # transforms.Normalize((0.5,), (0.5,))  # 如果你使用的是灰度图像，这里应该是 (0.5,)
            transforms.Resize((256, 256)),  # 如果你需要调整图像大小
            # transforms.RandomHorizontalFlip(),  # 如果你需要随机水平翻转
            # transforms.RandomRotation(10),  # 如果你需要随机旋转
            # transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # 如果你需要随机仿射变换
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 如果你需要颜色抖动
            # transforms.RandomGrayscale(p=0.2),  # 如果你需要随机灰度
            # transforms.RandomPerspective(distortion_scale=0.5, p=0.5),  # 如果你需要随机透视变换
            # transforms.RandomCrop(224, padding=4),  # 如果你需要随机裁剪
            # transforms.RandomResizedCrop(224),  # 如果你需要随机调整大小和裁剪
            # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),  # 如果你需要随机擦除
            # transforms.RandomSizedCrop(224),  # 如果你需要随机大小裁剪
        ])
        
        self.transform_label = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Resize((256, 256)),
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.image_filenames[idx])
        label_name = os.path.join(self.label_folder, self.label_filenames[idx])
        image = Image.open(img_name).convert('RGB')
        label = Image.open(label_name).convert('L')
        
        image = self.transform_image(image)
        label = self.transform_label(label)
        
        return image, label

if __name__ == '__main__':
    choose_run = 1

    # ---------------- 计算数据集的 Mean & Std -----------------
    if(choose_run == 1):
        data_dir = 'C:/Users/11795/Desktop/PPM-100/image'  # 数据集所在目录
        image_files = os.listdir(data_dir)
        # 初始化总和
        total_sum = np.zeros(3)
        total_sum_sq = np.zeros(3)
        total_count = 0
        # 遍历所有图像文件
        for image_file in  tqdm(image_files):
            image_path = os.path.join(data_dir, image_file)
            image = np.array(Image.open(image_path)) / 255.0  # 读取图像并将像素值缩放到[0, 1]
            total_sum += np.sum(image, axis=(0, 1))
            total_sum_sq += np.sum(image ** 2, axis=(0, 1))
            total_count += image.shape[0] * image.shape[1]
        # 计算均值和方差
        mean = total_sum / total_count
        std = np.sqrt((total_sum_sq / total_count) - mean ** 2)
        print("Mean:", mean)
        print("Std:", std)

    # --------------------- 测试加载数据集 ---------------------
    elif (choose_run == 2):
        batch_size = 8
        shuffle = True
        # 设定训练数据集路径
        train_dataset = AIM500Dataset('train', root_dir='/workspace/ViTMatting/data/AIM500')
        # 设置批量大小和是否随机打乱数据
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4,  pin_memory=True, drop_last=True, shuffle=shuffle)
        # 检查数据加载是否正常工作
        for images, labels in train_loader:
            print(images.shape, labels.shape)
        print("-" * 40)
        # 设定训练数据集路径
        test_dataset = AIM500Dataset('test', root_dir='/workspace/ViTMatting/data/AIM500')
        # 设置批量大小和是否随机打乱数据
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4,  pin_memory=True, drop_last=True, shuffle=shuffle)
        # 检查数据加载是否正常工作
        for images, labels in test_loader:
            print(images.shape, labels.shape)
