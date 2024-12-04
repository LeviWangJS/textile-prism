import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from math import exp

class SSIM(nn.Module):
    """结构相似性损失"""
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = self._create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self._create_window(self.window_size).to(img1.device)
            window = window.type_as(img1)
            self.window = window
            self.channel = channel

        return 1 - self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

    def _gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def _create_window(self, window_size):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(3, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        ssim_value = 0.0
        for i in range(channel):
            mu1 = F.conv2d(img1[:, i:i+1], window[i:i+1], padding=window_size//2)
            mu2 = F.conv2d(img2[:, i:i+1], window[i:i+1], padding=window_size//2)

            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2

            sigma1_sq = F.conv2d(img1[:, i:i+1] * img1[:, i:i+1], window[i:i+1], padding=window_size//2) - mu1_sq
            sigma2_sq = F.conv2d(img2[:, i:i+1] * img2[:, i:i+1], window[i:i+1], padding=window_size//2) - mu2_sq
            sigma12 = F.conv2d(img1[:, i:i+1] * img2[:, i:i+1], window[i:i+1], padding=window_size//2) - mu1_mu2

            C1 = 0.01 ** 2
            C2 = 0.03 ** 2

            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            ssim_value += ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)

        return ssim_value / channel

class PatternLoss(nn.Module):
    """图案转换模型的损失函数"""
    
    def __init__(self, lambda_l1=5.0, lambda_perceptual=2.0, lambda_style=0.5, lambda_ssim=1.0):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.lambda_style = lambda_style
        self.lambda_ssim = lambda_ssim
        
        # 加载预训练的VGG16用于感知损失
        vgg16 = models.vgg16(pretrained=True)
        self.vgg_features = nn.Sequential(*list(vgg16.features.children())[:23])
        for param in self.vgg_features.parameters():
            param.requires_grad = False
        self.vgg_features.eval()
        
        # 初始化SSIM损失
        self.ssim = SSIM()
    
    def forward(self, pred, target):
        # L1损失
        l1_loss = F.l1_loss(pred, target)
        
        # 感知损失
        pred_features = self.vgg_features(pred)
        target_features = self.vgg_features(target)
        perceptual_loss = F.mse_loss(pred_features, target_features)
        
        # 风格损失
        pred_gram = self._gram_matrix(pred_features)
        target_gram = self._gram_matrix(target_features)
        style_loss = F.mse_loss(pred_gram, target_gram)
        
        # SSIM损失
        ssim_loss = self.ssim(pred, target)
        
        # 总损失
        total_loss = (
            self.lambda_l1 * l1_loss +
            self.lambda_perceptual * perceptual_loss +
            self.lambda_style * style_loss +
            self.lambda_ssim * ssim_loss
        )
        
        # 返回损失字典
        return {
            'total': total_loss,
            'l1': l1_loss.item(),
            'perceptual': perceptual_loss.item(),
            'style': style_loss.item(),
            'ssim': ssim_loss.item()
        }
    
    def _gram_matrix(self, x):
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram

class DepthConsistencyLoss(nn.Module):
    """深度一致性损失"""
    
    def __init__(self):
        super().__init__()
        # 使用预训练的ResNet18提取深度特征
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()
    
    def forward(self, pred_depth, target_depth):
        # 提取深度特征
        pred_features = self.feature_extractor(pred_depth)
        target_features = self.feature_extractor(target_depth)
        
        # 计算特征差异
        return F.mse_loss(pred_features, target_features)

class PatternConsistencyLoss(nn.Module):
    """图案一致性损失"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        # 计算每个通道的自相关性并取平均
        pred_corr = torch.stack([self._autocorr2d(pred[:, i]) for i in range(pred.size(1))]).mean(0)
        target_corr = torch.stack([self._autocorr2d(target[:, i]) for i in range(target.size(1))]).mean(0)
        
        # 归一化相关性矩阵
        pred_corr = pred_corr / (pred_corr.max() + 1e-8)
        target_corr = target_corr / (target_corr.max() + 1e-8)
        
        # 使用L1损失而不是MSE，对异常值更不敏感
        return F.l1_loss(pred_corr, target_corr)
    
    def _autocorr2d(self, x):
        # 计算2D自相关
        # x shape: [batch_size, height, width]
        x = x - x.mean(dim=(-2, -1), keepdim=True)
        x = x / (x.std(dim=(-2, -1), keepdim=True) + 1e-8)  # 归一化
        batch_size = x.size(0)
        
        # 将输入重塑为卷积所需的形状
        x_reshaped = x.view(1, batch_size, *x.shape[1:])
        kernel = x.view(batch_size, 1, *x.shape[1:])
        
        # 使用更小的padding来减少边缘效应
        padding = (x.size(-2)//4, x.size(-1)//4)
        
        corr = F.conv2d(
            x_reshaped,
            kernel,
            padding=padding,
            groups=batch_size
        )
        
        # 归一化相关性结果
        corr = corr / (torch.sqrt(torch.sum(x ** 2, dim=(-2, -1))).view(batch_size, 1, 1, 1) + 1e-8)
        return corr.squeeze(0)

class EdgeAwareLoss(nn.Module):
    """边缘感知损失"""
    
    def __init__(self):
        super().__init__()
        self.sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
    
    def forward(self, pred, target):
        # 转换为灰度图
        pred_gray = 0.299 * pred[:, 0] + 0.587 * pred[:, 1] + 0.114 * pred[:, 2]
        target_gray = 0.299 * target[:, 0] + 0.587 * target[:, 1] + 0.114 * target[:, 2]
        
        pred_gray = pred_gray.unsqueeze(1)
        target_gray = target_gray.unsqueeze(1)
        
        # 计算边缘
        sobel_x = self.sobel_x.to(pred.device)
        sobel_y = self.sobel_y.to(pred.device)
        
        pred_edges_x = F.conv2d(pred_gray, sobel_x, padding=1)
        pred_edges_y = F.conv2d(pred_gray, sobel_y, padding=1)
        pred_edges = torch.sqrt(pred_edges_x ** 2 + pred_edges_y ** 2)
        
        target_edges_x = F.conv2d(target_gray, sobel_x, padding=1)
        target_edges_y = F.conv2d(target_gray, sobel_y, padding=1)
        target_edges = torch.sqrt(target_edges_x ** 2 + target_edges_y ** 2)
        
        return F.mse_loss(pred_edges, target_edges)

class Discriminator(nn.Module):
    """图案判别器"""
    def __init__(self, input_channels=3):
        super().__init__()
        
        def discriminator_block(in_filters, out_filters, normalize=True):
            """构建判别器基本块"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(input_channels, 64, normalize=False),  # (b, 64, 128, 128)
            *discriminator_block(64, 128),                             # (b, 128, 64, 64)
            *discriminator_block(128, 256),                           # (b, 256, 32, 32)
            *discriminator_block(256, 512),                           # (b, 512, 16, 16)
            nn.ZeroPad2d((1, 0, 1, 0)),                              # (b, 512, 17, 17)
            nn.Conv2d(512, 1, 4, padding=1)                          # (b, 1, 16, 16)
        )
    
    def forward(self, img):
        return self.model(img)

class AdversarialLoss(nn.Module):
    """对抗损失"""
    def __init__(self, device='cpu'):
        super().__init__()
        self.discriminator = Discriminator().to(device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.device = device
        
        # 优化器 - 增加学习率以加强判别器训练
        self.optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=0.0003,  # 增加学习率
            betas=(0.5, 0.999)
        )
        
        # 真假标签
        self.real_label = 1.0
        self.fake_label = 0.0
        
        # 标签平滑化
        self.label_smoothing = 0.1  # 添加标签平滑化
    
    def train_discriminator(self, real_images, fake_images):
        """训练判别器"""
        self.optimizer.zero_grad()
        
        # 真实图像的损失（使用标签平滑化）
        real_target = torch.full(
            (real_images.size(0), 1, 16, 16),
            self.real_label - self.label_smoothing,  # 标签平滑化
            device=self.device
        )
        real_pred = self.discriminator(real_images)
        real_loss = self.criterion(real_pred, real_target)
        
        # 生成图像的损失
        fake_target = torch.full(
            (fake_images.size(0), 1, 16, 16),
            self.fake_label,
            device=self.device
        )
        fake_pred = self.discriminator(fake_images.detach())
        fake_loss = self.criterion(fake_pred, fake_target)
        
        # 总损失
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        self.optimizer.step()
        
        return d_loss.item()
    
    def generator_loss(self, fake_images):
        """计算生成器的对抗损失"""
        # 使用标签平滑化的真实标签
        fake_target = torch.full(
            (fake_images.size(0), 1, 16, 16),
            self.real_label - self.label_smoothing,  # 标签平滑化
            device=self.device
        )
        fake_pred = self.discriminator(fake_images)
        
        # 特征匹配损失
        with torch.no_grad():
            real_features = self.discriminator.model[:-1](fake_images)  # 获取判别器中间特征
        fake_features = self.discriminator.model[:-1](fake_images)
        feature_matching_loss = F.mse_loss(fake_features, real_features.detach())
        
        # 组合损失
        g_loss = self.criterion(fake_pred, fake_target) + 0.1 * feature_matching_loss
        
        return g_loss

class CompositeLoss(nn.Module):
    """组合损失函数"""
    
    def __init__(self, weights=None, device='cpu'):
        super().__init__()
        self.pattern_loss = PatternLoss()
        self.pattern_consistency = PatternConsistencyLoss()
        self.edge_loss = EdgeAwareLoss()
        self.adversarial_loss = AdversarialLoss(device)
        
        # 调整损失权重以平衡各个损失项
        self.weights = weights or {
            'pattern': 1.0,         # 主要的图案损失
            'consistency': 0.0005,  # 一致性损失（保持较小以避免过度约束）
            'edge': 0.05,          # 降低边缘损失权重，让对抗损失来处理细节
            'adversarial': 0.05    # 增加对抗损失权重，但保持适度
        }
    
    def forward(self, pred, target, train_discriminator=True):
        # 基础损失
        losses = {
            'pattern': self.pattern_loss(pred, target),
            'consistency': self.pattern_consistency(pred, target),
            'edge': self.edge_loss(pred, target)
        }
        
        # 对抗损失
        if train_discriminator:
            d_loss = self.adversarial_loss.train_discriminator(target, pred)
            losses['discriminator'] = d_loss
        
        g_loss = self.adversarial_loss.generator_loss(pred)
        losses['adversarial'] = g_loss
        
        # 计算总损失
        total_loss = sum(
            self.weights[k] * v['total'] if isinstance(v, dict) else self.weights[k] * v
            for k, v in losses.items()
            if k in self.weights
        )
        
        # 构建返回字典
        result = {
            'total': total_loss,
            'pattern_total': losses['pattern']['total'],
            'pattern_l1': losses['pattern']['l1'],
            'pattern_perceptual': losses['pattern']['perceptual'],
            'pattern_style': losses['pattern']['style'],
            'pattern_ssim': losses['pattern']['ssim'],
            'consistency': losses['consistency'],
            'edge': losses['edge'],
            'adversarial': losses['adversarial']
        }
        
        if train_discriminator:
            result['discriminator'] = losses['discriminator']
        
        return result 