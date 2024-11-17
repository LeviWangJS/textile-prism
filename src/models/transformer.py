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
        
        # 加载预训练的ResNet作为特征提取器
        backbone = models.resnet50(pretrained=config['model']['pretrained'])
        
        # 移除最后的全连接层
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])
        
        # 添加投影层，将特征维度从2048降到config指定的维度
        self.projection = nn.Conv2d(2048, config['model']['feature_dim'], kernel_size=1)
        
        # 添加transformer层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['model']['feature_dim'],
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # 解码器网络
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(config['model']['feature_dim'], 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # 编码
        features = self.encoder(x)
        
        # 投影到较低维度
        features = self.projection(features)
        
        # 重塑特征图为序列
        batch_size, channels, height, width = features.shape
        features = features.flatten(2).permute(2, 0, 1)  # [H*W, B, C]
        
        # 通过transformer
        transformed = self.transformer(features)
        
        # 重塑回特征图
        transformed = transformed.permute(1, 2, 0).view(batch_size, channels, height, width)
        
        # 解码
        output = self.decoder(transformed)
        return output 