from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用新的权重API
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # 冻结参数
        for param in self.model.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        # 保持原有的forward逻辑不变
        features = []
        for name, module in self.model.named_children():
            x = module(x)
            if name in self.target_layers:
                features.append(x)
        return features 