import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Tuple

class SpatialTransformer(nn.Module):
    """空间变换网络"""
    def __init__(self, input_channels: int):
        super().__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(output_size=(4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 16, 32),
            nn.ReLU(True),
            nn.Linear(32, 6)
        )
        
        # 初始化为恒等变换
        self.localization[-1].weight.data.zero_()
        self.localization[-1].bias.data.copy_(torch.tensor(
            [1, 0, 0, 0, 1, 0], dtype=torch.float))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        theta = self.localization(x)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        transformed = F.grid_sample(x, grid, align_corners=True)
        return transformed

class PatternNet(nn.Module):
    """轻量级模型架构"""
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # 使用ResNet18作为编码器
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.ModuleList([
            nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool),
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        ])
        
        # 简化的特征处理
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(512, 256, 1),
            nn.Conv2d(256, 256, 1),
            nn.Conv2d(128, 256, 1),
            nn.Conv2d(64, 256, 1)
        ])
        
        # 简化的空间变换器
        self.transformer = SpatialTransformer(256)
        
        # 修改解码器以匹配输入尺寸
        self.decoder = nn.Sequential(
            # 从 20x20 开始 (输入 640x640 经过编码器后的尺寸)
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),  # 40x40
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 80x80
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 160x160
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 320x320
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),    # 640x640
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # 编码
        features = []
        for encoder_block in self.encoder:
            x = encoder_block(x)
            features.append(x)
        
        # 特征处理
        x = self.lateral_convs[0](features[-1])
        
        # 空间变换
        x = self.transformer(x)
        
        # 解码
        output = self.decoder(x)
        
        return output, {
            'features': features,
            'transformed': x
        }

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        criterion: nn.Module
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        inputs = batch['input']
        targets = batch['target']
        
        outputs, aux_outputs = self(inputs)
        losses = criterion(outputs, targets)
        
        return losses, {
            'outputs': outputs,
            **aux_outputs
        }