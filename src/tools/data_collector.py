import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

class DataCollector:
    def __init__(self, root_dir: str = "data/raw"):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        
        # 更新分辨率要求
        self.min_resolution = (800, 800)    # 最小分辨率
        self.max_resolution = (2048, 2048)  # 最大分辨率
    
    def _validate_images(self, input_img: np.ndarray, target_img: np.ndarray) -> bool:
        """验证图像质量"""
        # 检查input图像最小分辨率
        if (input_img.shape[0] < self.min_resolution[1] or 
            input_img.shape[1] < self.min_resolution[0]):
            print(f"输入图像分辨率不足，最小要求：{self.min_resolution}")
            return False
            
        # 检查target图像最小分辨率    
        if (target_img.shape[0] < self.min_resolution[1] or
            target_img.shape[1] < self.min_resolution[0]):
            print(f"目标图像分辨率不足，最小要求：{self.min_resolution}")
            return False
            
        # 检查input图像最大分辨率
        if (input_img.shape[0] > self.max_resolution[1] or 
            input_img.shape[1] > self.max_resolution[0]):
            print(f"输入图像分辨率过大，最大限制：{self.max_resolution}")
            return False
            
        # 检查target图像最大分辨率
        if (target_img.shape[0] > self.max_resolution[1] or
            target_img.shape[1] > self.max_resolution[0]):
            print(f"目标图像分辨率过大，最大限制：{self.max_resolution}")
            return False
            
        # 检查图像清晰度
        input_blur = cv2.Laplacian(cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        target_blur = cv2.Laplacian(cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        
        if input_blur < 50:  # 检查输入图像清晰度
            print(f"输入图像清晰度不足: {input_blur:.2f}")
            return False
            
        if target_blur < 50:  # 检查目标图像清晰度
            print(f"目标图像清晰度不足: {target_blur:.2f}")
            return False
            
        return True
    
    def get_next_set_number(self) -> int:
        """获取下一个可用的数据集编号"""
        existing_sets = [d for d in self.root_dir.glob("set_*")]
        if not existing_sets:
            return 1
        numbers = [int(d.name.split("_")[1]) for d in existing_sets]
        return max(numbers) + 1
    
    def add_image_pair(self, input_path: str, target_path: str) -> bool:
        """添加一对新的图像"""
        # 验证文件存在
        if not os.path.exists(input_path) or not os.path.exists(target_path):
            print("输入文件不存在")
            return False
            
        # 读取图像
        input_img = cv2.imread(input_path)
        target_img = cv2.imread(target_path)
        
        if input_img is None or target_img is None:
            print("无法读取图像文件")
            return False
        
        # 如果图像太大，自动调整大小
        if (input_img.shape[0] > self.max_resolution[1] or 
            input_img.shape[1] > self.max_resolution[0]):
            scale = min(self.max_resolution[0] / input_img.shape[1],
                       self.max_resolution[1] / input_img.shape[0])
            new_size = (int(input_img.shape[1] * scale),
                       int(input_img.shape[0] * scale))
            input_img = cv2.resize(input_img, new_size)
            
        if (target_img.shape[0] > self.max_resolution[1] or 
            target_img.shape[1] > self.max_resolution[0]):
            scale = min(self.max_resolution[0] / target_img.shape[1],
                       self.max_resolution[1] / target_img.shape[0])
            new_size = (int(target_img.shape[1] * scale),
                       int(target_img.shape[0] * scale))
            target_img = cv2.resize(target_img, new_size)
            
        # 验证图像质量
        if not self._validate_images(input_img, target_img):
            return False
            
        # 创建新的数据集目录
        set_num = self.get_next_set_number()
        set_dir = self.root_dir / f"set_{set_num:03d}"
        set_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存处理后的图像
        cv2.imwrite(str(set_dir / "input.jpg"), input_img)
        cv2.imwrite(str(set_dir / "target.jpg"), target_img)
        
        print(f"成功添加数据集: set_{set_num:03d}")
        return True 