import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict

class PerceptualLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # 获取VGG配置
        if 'loss' not in config:
            config['loss'] = {}
            
        vgg_layers = config['loss'].get('vgg_layers', [3, 8, 15, 22])
        self.layers = [str(layer) for layer in vgg_layers]
        
        # 加载VGG模型
        try:
            self.vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.to(self.device)
        except Exception as e:
            print(f"Warning: Failed to load VGG with weights, using pretrained=True: {e}")
            self.vgg = models.vgg16(pretrained=True).features.to(self.device)
        
        self.vgg.eval()
        
        # 冻结参数
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        # 特征权重
        self.feature_weights = config['loss'].get('perceptual_weights', 
                                                [1.0] * len(self.layers))
        
        self.debug = False
        
        # 打印配置信息
        print(f"VGG layers: {self.layers}")
        print(f"Feature weights: {self.feature_weights}")
    
    def forward(self, x, y):
        if self.debug:
            print(f"Input device: {x.device}, VGG device: {next(self.vgg.parameters()).device}")
        
        # 确保输入在正确的设备上
        x = x.to(self.device)
        y = y.to(self.device)
        
        x_features = []
        y_features = []
        
        for name, block in self.vgg.named_children():
            x = block(x)
            y = block(y)
            if name in self.layers:
                if self.debug:
                    print(f"Layer {name} output shape: {x.shape}")
                x_features.append(x)
                y_features.append(y)
        
        # 计算特征损失
        loss = 0
        for x_feat, y_feat in zip(x_features, y_features):
            loss += self.criterion(x_feat, y_feat)
        
        return loss

class PatternLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # 初始化损失组件
        self.l1_loss = nn.L1Loss().to(self.device)
        self.perceptual_loss = PerceptualLoss(config).to(self.device)
        
        # 获取损失权重，使用默认值
        if 'loss' not in config:
            print("Warning: 'loss' configuration not found, using default values")
            config['loss'] = {}
            
        self.lambda_l1 = config['loss'].get('lambda_l1', 1.0)
        self.lambda_perceptual = config['loss'].get('lambda_perceptual', 0.1)
        
        # 打印配置信息
        print(f"Loss weights: L1={self.lambda_l1}, Perceptual={self.lambda_perceptual}")
        
        self.debug = False
    
    def forward(self, pred, target):
        if self.debug:
            print(f"Pred device: {pred.device}, Target device: {target.device}")
        
        # 确保输入在正确的设备上
        pred = pred.to(self.device)
        target = target.to(self.device)
        
        # 计算各个损失
        l1 = self.l1_loss(pred, target)
        perceptual = self.perceptual_loss(pred, target)
        
        # 组合损失
        total_loss = self.lambda_l1 * l1 + self.lambda_perceptual * perceptual
        
        return {
            'total': total_loss,
            'l1': l1,
            'perceptual': perceptual
        } 