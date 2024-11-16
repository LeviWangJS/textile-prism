import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SpatialTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(output_size=(32, 32)),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(256 * 32 * 32, 32),
            nn.ReLU(True),
            nn.Linear(32, 6)
        )
        
        # 初始化为恒等变换
        self.localization[-1].weight.data.zero_()
        self.localization[-1].bias.data.copy_(torch.tensor(
            [1, 0, 0, 0, 1, 0], dtype=torch.float))
    
    def forward(self, x):
        theta = self.localization(x)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        return F.grid_sample(x, grid)

class PatternTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 编码器
        backbone = models.resnet50(pretrained=config.model.pretrained)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])
        
        # 空间变换
        self.transformer = SpatialTransformer()
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        features = self.encoder(x)
        transformed = self.transformer(features)
        output = self.decoder(transformed)
        return output 