import os
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class CarpetDataset(Dataset):
    """地毯图案数据集"""
    def __init__(self, config, mode='train'):
        self.config = config
        self.mode = mode
        self.data_dir = Path(config['data']['split_dir']) / mode
        
        # 获取所有数据集目录
        self.sample_dirs = sorted([d for d in self.data_dir.glob('set_*') if d.is_dir()])
        
        # 设置图像大小
        self.img_size = tuple(config['data']['input_size'])
        
        # 创建数据增强
        if mode == 'train' and config['data']['augmentation']['enabled']:
            self.aug = A.Compose([
                A.Resize(height=self.img_size[0], width=self.img_size[1]),
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.OneOf([
                    A.GaussNoise(p=1),
                    A.GaussianBlur(p=1),
                    A.MotionBlur(p=1),
                ], p=0.3),
                A.OneOf([
                    A.RandomBrightnessContrast(p=1),
                    A.HueSaturationValue(p=1),
                ], p=0.3)
            ], additional_targets={'mask': 'image'})
        else:
            self.aug = A.Compose([
                A.Resize(height=self.img_size[0], width=self.img_size[1])
            ], additional_targets={'mask': 'image'})
    
    def __len__(self):
        return len(self.sample_dirs)
    
    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]
        
        # 读取输入和目标图像
        input_path = sample_dir / 'input.jpg'
        target_path = sample_dir / 'target.jpg'
        
        input_img = Image.open(input_path).convert('RGB')
        target_img = Image.open(target_path).convert('RGB')
        
        # 转换为numpy数组
        input_np = np.array(input_img)
        target_np = np.array(target_img)
        
        # 应用数据增强/调整大小
        if self.mode == 'train' and self.config['data']['augmentation']['enabled']:
            # 对输入和目标应用相同的随机变换
            transformed = self.aug(image=input_np, mask=target_np)
            input_np = transformed['image']
            target_np = transformed['mask']
        else:
            # 只调整大小
            input_np = self.aug(image=input_np)['image']
            target_np = self.aug(image=target_np)['image']
        
        # 转换为张量并标准化
        input_tensor = torch.from_numpy(input_np).permute(2, 0, 1).float() / 255.0
        target_tensor = torch.from_numpy(target_np).permute(2, 0, 1).float() / 255.0
        
        # 标准化到[-1, 1]
        input_tensor = input_tensor * 2 - 1
        target_tensor = target_tensor * 2 - 1
        
        return {
            'input': input_tensor,
            'target': target_tensor,
            'path': str(sample_dir)
        } 