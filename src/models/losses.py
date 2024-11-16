import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features
        self.blocks = nn.ModuleList([
            vgg[:4],   # conv1_2
            vgg[4:9],  # conv2_2
            vgg[9:16], # conv3_3
            vgg[16:23] # conv4_3
        ])
        
        for bl in self.blocks:
            for p in bl.parameters():
                p.requires_grad = False
                
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != 3:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
            
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        y = (y - self.mean.to(y.device)) / self.std.to(y.device)
        
        loss = 0.0
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += F.l1_loss(x, y)
            
        return loss

class PatternLoss(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = VGGPerceptualLoss()
        
        self.lambda_dict = config['train']['loss_weights']
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 计算各个损失
        l1 = self.l1_loss(pred, target)
        perceptual = self.perceptual_loss(pred, target)
        
        # 总损失
        total = (
            self.lambda_dict['l1'] * l1 +
            self.lambda_dict['perceptual'] * perceptual
        )
        
        return {
            'total': total,
            'l1': l1,
            'perceptual': perceptual
        } 