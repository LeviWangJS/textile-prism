import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Tuple

class SpatialTransformer(nn.Module):
    """空间变换网络"""
    def __init__(self, config):
        super().__init__()
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # 添加设备信息
        if self.device.type == "mps":
            print("SpatialTransformer: 使用MPS设备 (已启用CPU回退)")
        
        # 计算最终特征图大小
        input_size = 20  # 初始输入大小
        conv_layers = 3   # 卷积层数量
        final_size = input_size // (2 ** conv_layers)  # 每次MaxPool后大小减半
        
        # 计算展平后的特征数量
        self.flattened_features = 32 * final_size * final_size
        
        # 修改定位网络结构
        self.localization = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # 20 -> 10
            
            # 第二个卷积块
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # 10 -> 5
            
            # 第三个卷积块
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # 5 -> 3
            
            # 展平层
            nn.Flatten(),  # 32 * 3 * 3 = 288
            
            # 全连接层
            nn.Linear(self.flattened_features, 32),
            nn.ReLU(True),
            nn.Linear(32, 6)
        )
        
        # 初始化变换参数
        self.localization[-1].weight.data.zero_()
        self.localization[-1].bias.data.copy_(torch.tensor(
            [1, 0, 0, 0, 1, 0], dtype=torch.float))
        
        self.debug = True
    
    def forward(self, x):
        if self.debug:
            print(f"Transformer input shape: {x.shape}")
        
        try:
            features = x
            for i, layer in enumerate(self.localization):
                features = layer(features)
                if self.debug:
                    if isinstance(layer, (nn.Conv2d, nn.Linear, nn.Flatten)):
                        print(f"After {type(layer).__name__}: {features.shape}")
            
            theta = features
            theta = theta.view(-1, 2, 3)
            
            if self.debug:
                print(f"Theta shape: {theta.shape}")
            
            # 使用原始输入x的尺寸
            grid = F.affine_grid(theta, x.size(), align_corners=True)
            transformed = F.grid_sample(x, grid, align_corners=True)
            
            if self.debug:
                print(f"Transformed shape: {transformed.shape}")
            
            return transformed
            
        except Exception as e:
            print(f"Transformer error: {str(e)}")
            if "MPS" in str(e):
                print("使用CPU回退...")
                return self._forward_cpu(x)
            raise e
    
    def _forward_cpu(self, x):
        """CPU回退实现"""
        x_cpu = x.cpu()
        self.to('cpu')
        
        try:
            # 在CPU上执行变换
            features = self.localization(x_cpu)
            theta = features.view(-1, 2, 3)
            grid = F.affine_grid(theta, x_cpu.size(), align_corners=True)
            transformed = F.grid_sample(x_cpu, grid, align_corners=True)
            
            # 移回原设备
            self.to(self.device)
            return transformed.to(self.device)
            
        finally:
            self.to(self.device)

    def _calculate_output_size(self, size):
        """计算每层后的输出大小"""
        for layer in self.localization:
            if isinstance(layer, nn.Conv2d):
                # 考虑padding和stride
                padding = layer.padding[0] if isinstance(layer.padding, tuple) else layer.padding
                size = (size + 2 * padding - layer.kernel_size[0]) // layer.stride[0] + 1
            elif isinstance(layer, nn.MaxPool2d):
                size = size // layer.stride
        return size

class PatternNet(nn.Module):
    """轻量级模型架构"""
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # 添加维度检查
        self.debug = True
        
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
            nn.Conv2d(512, 256, kernel_size=1)  # 确保输出通道数为256
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
        if self.debug:
            print(f"\nInput shape: {x.shape}")
        
        try:
            # 编码
            features = []
            for i, encoder_block in enumerate(self.encoder):
                x = encoder_block(x)
                if self.debug:
                    print(f"After encoder block {i}: {x.shape}")
                features.append(x)
            
            # 特征处理
            x = self.lateral_convs[0](features[-1])
            if self.debug:
                print(f"After lateral conv: {x.shape}")
            
            # 空间变换
            x = self.transformer(x)
            if self.debug:
                print(f"After transformer: {x.shape}")
            
            # 解码
            output = self.decoder(x)
            if self.debug:
                print(f"Final output: {output.shape}")
            
            return output, {
                'features': features,
                'transformed': x
            }
            
        except RuntimeError as e:
            print(f"Error in forward pass: {str(e)}")
            if "MPS" in str(e):
                return self._forward_cpu(x)
            raise e
    
    def _forward_cpu(self, x):
        """CPU回退实现"""
        x = x.cpu()
        self.to('cpu')
        
        try:
            output, aux_outputs = self.forward(x)
            
            # 移回原设备
            self.to(self.device)
            output = output.to(self.device)
            aux_outputs['features'] = [f.to(self.device) for f in aux_outputs['features']]
            aux_outputs['transformed'] = aux_outputs['transformed'].to(self.device)
            
            return output, aux_outputs
            
        finally:
            self.to(self.device)

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