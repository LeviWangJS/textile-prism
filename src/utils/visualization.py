import torch
from torchvision.utils import save_image as torch_save_image
from pathlib import Path

def save_image(tensor, save_path):
    """保存图像张量为PNG文件"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 确保张量在[0,1]范围内
    if tensor.min() < 0 or tensor.max() > 1:
        tensor = (tensor + 1) / 2  # 从[-1,1]转换到[0,1]
    
    torch_save_image(tensor, save_path)
