import torch
import torchvision.transforms.functional as TF
import random
from typing import Dict, List, Optional
import numpy as np
import logging

class PatternAugmentor:
    def __init__(self, config: Dict):
        self.config = config
        self.augmentations = []
        
        # 根据配置初始化增强方法
        if config.get('augmentation', {}).get('enabled', False):
            self._setup_augmentations()
        
        self.logger = logging.getLogger(__name__)
    
    def _setup_augmentations(self):
        """设置数据增强管道"""
        aug_config = self.config['augmentation']
        
        # 几何变换
        if aug_config.get('geometric', {}).get('enabled', False):
            self.augmentations.extend([
                self.random_rotate,
                self.random_flip,
                self.random_scale
            ])
            
        # 光照/颜色变换
        if aug_config.get('color', {}).get('enabled', False):
            self.augmentations.extend([
                self.adjust_brightness,
                self.adjust_contrast,
                self.adjust_saturation
            ])
            
        # 纹理增强
        if aug_config.get('texture', {}).get('enabled', False):
            self.augmentations.extend([
                self.add_noise,
                self.elastic_transform
            ])
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """应用数据增强"""
        if not self.config.get('augmentation', {}).get('enabled', False):
            return image
            
        applied_augs = []
        for aug in self.augmentations:
            if random.random() < self.config['augmentation']['probability']:
                image = aug(image)
                applied_augs.append(aug.__name__)
                
        if applied_augs:
            self.logger.debug(f"Applied augmentations: {', '.join(applied_augs)}")
            
        return image
    
    # 几何变换方法
    def random_rotate(self, image: torch.Tensor) -> torch.Tensor:
        angles = self.config['augmentation']['geometric'].get('rotate_angles', [-30, -15, 0, 15, 30])
        angle = random.choice(angles)
        return TF.rotate(image, angle)
    
    def random_flip(self, image: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.5:
            image = TF.hflip(image)
        if random.random() < 0.5:
            image = TF.vflip(image)
        return image
        
    def random_scale(self, image: torch.Tensor) -> torch.Tensor:
        scales = self.config['augmentation']['geometric'].get('scale_factors', [0.8, 1.0, 1.2])
        scale = random.choice(scales)
        return TF.affine(image, angle=0, translate=(0,0), scale=scale, shear=0)
    
    # 光照/颜色变换方法
    def adjust_brightness(self, image: torch.Tensor) -> torch.Tensor:
        factors = self.config['augmentation']['color'].get('brightness_levels', [0.8, 1.0, 1.2])
        factor = random.choice(factors)
        return TF.adjust_brightness(image, factor)
        
    def adjust_contrast(self, image: torch.Tensor) -> torch.Tensor:
        factors = self.config['augmentation']['color'].get('contrast_levels', [0.8, 1.0, 1.2])
        factor = random.choice(factors)
        return TF.adjust_contrast(image, factor)
        
    def adjust_saturation(self, image: torch.Tensor) -> torch.Tensor:
        factors = self.config['augmentation']['color'].get('saturation_levels', [0.8, 1.0, 1.2])
        factor = random.choice(factors)
        return TF.adjust_saturation(image, factor)
    
    # 纹理增强方法
    def add_noise(self, image: torch.Tensor) -> torch.Tensor:
        noise_std = self.config['augmentation']['texture'].get('noise_std', 0.05)
        noise = torch.randn_like(image) * noise_std
        return torch.clamp(image + noise, 0, 1)
        
    def elastic_transform(self, image: torch.Tensor) -> torch.Tensor:
        """弹性变换，用于模拟纹理变形"""
        # 暂时返回原图，后续可以实现更复杂的弹性变换
        return image 