import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict
import math

class DepthFeatureExtractor(nn.Module):
    """增强的深度特征提取模块"""
    def __init__(self, config):
        super().__init__()
        self.encoder = models.resnet18(pretrained=True)
        
        # 修改第一层以适应输入通道
        in_channels = config.get('input_channels', 3)
        self.encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 冻结部分预训练权重
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # 只训练最后几层
        for param in self.encoder.layer3.parameters():
            param.requires_grad = True
        for param in self.encoder.layer4.parameters():
            param.requires_grad = True
        
        # 添加特征增强层
        self.enhance_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            )
        ])
    
    def forward(self, x):
        # 保存中间特征
        features = []
        
        # 提取特征
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        
        # layer1 输出 64 通道
        x = self.encoder.layer1(x)
        features.append(self.enhance_layers[0](x))
        
        # layer2 输出 128 通道
        x = self.encoder.layer2(x)
        features.append(self.enhance_layers[1](x))
        
        # layer3 输出 256 通道
        x = self.encoder.layer3(x)
        features.append(self.enhance_layers[2](x))
        
        # layer4 输出 512 通道
        x = self.encoder.layer4(x)
        features.append(self.enhance_layers[3](x))
        
        return features[-1]  # 返回最后一层特征

class SelfAttention(nn.Module):
    """自注意力模块"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = 4
        self.head_dim = hidden_dim // self.num_heads
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # 将特征图重塑为序列
        x = x.view(batch_size, channels, -1).permute(0, 2, 1)
        
        # 计算注意力
        q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        x = torch.matmul(attn, v)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        
        # 投影并重塑回特征图
        x = self.proj(x)
        x = x.permute(0, 2, 1).view(batch_size, channels, height, width)
        
        return x, attn

class EnhancedSpatialTransformer(nn.Module):
    """增强的空间变换器模块"""
    def __init__(self, config):
        super().__init__()
        self.input_channels = config.get('input_channels', 512)
        self.num_scales = config.get('num_scales', 4)
        
        # 定位网络
        self.localization = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.input_channels, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.MaxPool2d(2),
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(True)
            ) for _ in range(self.num_scales)
        ])
        
        # 将adaptive_avg_pool和全连接层分开
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.ModuleList([
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 16, 32),
                nn.ReLU(True),
                nn.Linear(32, 6)
            ) for _ in range(self.num_scales)
        ])
        
        # 初始化全连接层的最后一层
        for net in self.fc:
            net[-1].weight.data.zero_()
            net[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
    
    def forward(self, x, level):
        # 获取空间变换参数
        device = x.device
        
        # 在原设备上执行卷积操作
        x = self.localization[level](x)
        
        # 将数据临时转移到CPU执行adaptive_avg_pool2d
        x_cpu = x.cpu()
        x_cpu = self.pool(x_cpu)
        
        # 将数据转回原设备执行全连接层
        x = x_cpu.to(device)
        theta = self.fc[level](x)
        
        theta = theta.view(-1, 2, 3)
        
        # 创建采样网格
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        
        # 应用空间变换
        x = F.grid_sample(x, grid, align_corners=True)
        
        return x

class FeaturePyramidNetwork(nn.Module):
    """特征金字塔网络"""
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        
        # 横向连接
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for in_channels in in_channels_list
        ])
        
        # 特征融合
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(len(in_channels_list))
        ])
    
    def forward(self, features):
        # 自顶向下的路径
        laterals = [conv(f) for f, conv in zip(features, self.lateral_convs)]
        
        # 特征融合
        for i in range(len(laterals)-1, 0, -1):
            laterals[i-1] = laterals[i-1] + F.interpolate(
                laterals[i], size=laterals[i-1].shape[2:],
                mode='bilinear', align_corners=True
            )
        
        # 特征增强
        outs = [conv(lat) for lat, conv in zip(laterals, self.fpn_convs)]
        return outs

class AttentionGate(nn.Module):
    """注意力门模块"""
    
    def __init__(self, F_g, F_l):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_l, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_l)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_l, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_l)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_l, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class Decoder(nn.Module):
    """解码器模块"""
    def __init__(self, config):
        super().__init__()
        self.decoder = nn.Sequential(
            # 从128通道开始
            nn.ConvTranspose2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.decoder(x)

class PatternTransformer(nn.Module):
    """主模型类"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 特征提取器
        self.feature_extractor = DepthFeatureExtractor(config)
        
        # 空间变换器
        self.spatial_transformer = EnhancedSpatialTransformer(config)
        
        # 解码器
        self.decoder = Decoder(config)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
        # 上采样层
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        # 提取特征
        features = self.feature_extractor(x)  # [B, 512, H/32, W/32]
        
        # 空间变换
        transformed = self.spatial_transformer(features, 0)  # [B, 128, H/32, W/32]
        
        # 解码
        decoded = self.decoder(transformed)  # [B, 3, H, W]
        
        # 确保输出尺寸正确
        if decoded.shape[2:] != x.shape[2:]:
            decoded = F.interpolate(decoded, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        return decoded