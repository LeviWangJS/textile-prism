import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from typing import Dict, Optional, Tuple
import logging
from pathlib import Path
import numpy as np
from PIL import Image
import os

class CLIPFeatureExtractor(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 配置参数
        self.cache_dir = Path(config.get('cache_dir', 'cache/clip_features'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = config.get('use_feature_cache', True)
        
        # 加载CLIP模型
        model_name = config.get('clip_model', 'openai/clip-vit-base-patch32')
        self.logger.info(f"Loading CLIP model: {model_name}")
        self.clip = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # 将模型移到设备
        self.clip = self.clip.to(self.device)
        
        # 冻结CLIP参数
        for param in self.clip.parameters():
            param.requires_grad = False
            
    def _get_cache_path(self, image_hash: str) -> Path:
        """获取特征缓存路径"""
        return self.cache_dir / f"{image_hash}.npy"
        
    def _compute_image_hash(self, image: torch.Tensor) -> str:
        """计算图像哈希值用于缓存"""
        return str(hash(image.cpu().numpy().tobytes()))
        
    def _load_cached_features(self, image_hash: str) -> Optional[torch.Tensor]:
        """从缓存加载特征"""
        cache_path = self._get_cache_path(image_hash)
        if cache_path.exists():
            try:
                features = torch.from_numpy(np.load(cache_path))
                return features.to(self.device)
            except Exception as e:
                self.logger.warning(f"Failed to load cached features: {e}")
        return None
        
    def _save_features_to_cache(self, features: torch.Tensor, image_hash: str):
        """保存特征到缓存"""
        try:
            cache_path = self._get_cache_path(image_hash)
            np.save(cache_path, features.cpu().numpy())
        except Exception as e:
            self.logger.warning(f"Failed to cache features: {e}")
            
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """提取CLIP特征"""
        batch_size = images.size(0)
        features_list = []
        
        for i in range(batch_size):
            image = images[i]
            image_hash = self._compute_image_hash(image)
            
            # 尝试从缓存加载
            if self.use_cache:
                cached_features = self._load_cached_features(image_hash)
                if cached_features is not None:
                    features_list.append(cached_features)
                    continue
            
            # 转换为PIL图像
            if image.dim() == 3:
                image = image.unsqueeze(0)
            image_np = (image.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np[0])
            
            # 处理图像
            inputs = self.processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 提取特征
            with torch.no_grad():
                features = self.clip.get_image_features(**inputs)
                
            # 缓存特征
            if self.use_cache:
                self._save_features_to_cache(features, image_hash)
                
            features_list.append(features)
            
        # 合并批次特征
        batch_features = torch.cat(features_list, dim=0)
        return batch_features

class FeatureFusion(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        clip_dim = config.get('clip_feature_dim', 512)
        original_dim = config.get('original_feature_dim', 512)
        fusion_dim = config.get('fusion_dim', 512)
        dropout_rate = config.get('dropout_rate', 0.1)
        
        self.fusion = nn.Sequential(
            nn.Linear(clip_dim + original_dim, fusion_dim * 2),
            nn.LayerNorm(fusion_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
    def forward(self, clip_features: torch.Tensor, original_features: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([clip_features, original_features], dim=1)
        return self.fusion(combined) 