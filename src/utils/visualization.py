import torch
import torchvision.utils as vutils
from pathlib import Path
from typing import Union

def save_image(
    tensor: torch.Tensor,
    filepath: Union[str, Path],
    normalize: bool = True
):
    """保存张量为图像"""
    # 确保输入是CPU张量
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # 如果是批次的第一张图片，去掉批次维度
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    # 保存图像
    vutils.save_image(
        tensor,
        filepath,
        normalize=normalize,
        padding=0
    )
