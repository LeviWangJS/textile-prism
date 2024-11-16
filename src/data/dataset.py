import cv2
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict

class CarpetDataset(Dataset):
    def __init__(self, config: Dict, mode: str = 'train'):
        self.config = config
        self.mode = mode
        self.data_dir = Path(config['data']['raw_dir'])
        
        # 打印调试信息
        print(f"初始化数据集：{mode}")
        print(f"数据目录：{self.data_dir}")
        
        # 获取所有数据集目录
        self.sample_dirs = sorted(list(self.data_dir.glob("set_*")))
        print(f"找到数据集目录：{len(self.sample_dirs)}个")
        
        if not self.sample_dirs:
            raise ValueError(f"未找到数据集！请检查目录：{self.data_dir}")
        
        # 数据集划分
        if mode == 'train':
            self.sample_dirs = self.sample_dirs[:int(len(self.sample_dirs) * 0.8)]
        else:
            self.sample_dirs = self.sample_dirs[int(len(self.sample_dirs) * 0.8):]
        
        print(f"{mode}模式使用{len(self.sample_dirs)}个数据集")
        
        # 验证每个数据集的完整性
        valid_dirs = []
        for dir_path in self.sample_dirs:
            input_path = dir_path / "input.jpg"
            target_path = dir_path / "target.jpg"
            if input_path.exists() and target_path.exists():
                valid_dirs.append(dir_path)
            else:
                print(f"警告：数据集{dir_path}不完整，已跳过")
        
        self.sample_dirs = valid_dirs
        print(f"有效数据集数量：{len(self.sample_dirs)}")
    
    def __len__(self):
        return len(self.sample_dirs)
    
    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]
        
        # 读取图像
        input_path = sample_dir / "input.jpg"
        target_path = sample_dir / "target.jpg"
        
        # 读取并预处理图像
        input_img = cv2.imread(str(input_path))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        
        target_img = cv2.imread(str(target_path))
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
        
        # 调整大小
        input_size = self.config['data']['input_size']
        input_img = cv2.resize(input_img, (input_size[0], input_size[1]))
        target_img = cv2.resize(target_img, (input_size[0], input_size[1]))
        
        # 转换为张量
        input_tensor = torch.from_numpy(input_img.transpose(2, 0, 1)).float() / 255.0
        target_tensor = torch.from_numpy(target_img.transpose(2, 0, 1)).float() / 255.0
        
        return {
            'input': input_tensor,
            'target': target_tensor,
            'path': str(sample_dir)
        } 