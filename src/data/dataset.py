import cv2
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict
from PIL import Image
from torchvision import transforms
from .augmentation import PatternAugmentor

class CarpetDataset(Dataset):
    def __init__(self, config: Dict, mode: str = 'train'):
        super().__init__()
        self.config = config
        self.mode = mode
        self.data_dir = Path(config['data']['raw_dir'])
        
        # 获取所有数据目录
        self.data_folders = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        print(f"找到数据集目录：{len(self.data_folders)}个")
        
        # 划分训练集和验证集
        train_size = int(len(self.data_folders) * config['data']['train_ratio'])
        if mode == 'train':
            self.data_folders = self.data_folders[:train_size]
            print(f"train模式使用{len(self.data_folders)}个数据集")
        else:  # val模式
            self.data_folders = self.data_folders[train_size:]
            print(f"val模式使用{len(self.data_folders)}个数据集")
    
        print(f"有效数据集数量：{len(self.data_folders)}")
        
        # 只在训练模式下初始化数据增强器
        self.augmentor = PatternAugmentor(config) if mode == 'train' else None
    
    def __len__(self):
        return len(self.data_folders) * 2  # 每个文件夹有2张图片
    
    def __getitem__(self, idx):
        folder_idx = idx // 2
        img_idx = idx % 2
        
        folder = self.data_folders[folder_idx]
        images = sorted(list(folder.glob('*.jpg')))
        
        # 读取图像
        img = Image.open(images[img_idx])
        
        # 调整图像大小
        input_size = self.config['data']['input_size']
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        img = img.resize(input_size)
        
        # 转换为张量
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        img_tensor = transform(img)
        
        # 应用数据增强
        if self.augmentor is not None:
            img_tensor = self.augmentor(img_tensor)
        
        # 第一张图作为输入，第二张图作为目标
        if img_idx == 0:
            return {'input': img_tensor, 'target': img_tensor}
        else:
            return {'input': img_tensor, 'target': img_tensor} 