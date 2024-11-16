import cv2
import numpy as np
from typing import Tuple, Optional

class ImageProcessor:
    @staticmethod
    def load_image(path: str) -> np.ndarray:
        """加载图像"""
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        return img
        
    @staticmethod
    def resize_image(
        image: np.ndarray,
        target_size: Tuple[int, int],
        keep_ratio: bool = True
    ) -> np.ndarray:
        """调整图像大小"""
        if keep_ratio:
            # 保持宽高比的resize逻辑
            pass
        return cv2.resize(image, target_size)
        
    @staticmethod
    def normalize_image(
        image: np.ndarray,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None
    ) -> np.ndarray:
        """标准化图像"""
        if mean is None:
            mean = (0.485, 0.456, 0.406)
        if std is None:
            std = (0.229, 0.224, 0.225)
            
        image = image.astype(np.float32) / 255.0
        image = (image - mean) / std
        return image
