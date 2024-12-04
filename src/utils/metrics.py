import torch
import torch.nn.functional as F
import numpy as np
from torchvision.transforms.functional import rgb_to_grayscale
import lpips
from scipy import ndimage

class MetricCalculator:
    """评估指标计算器"""
    
    def __init__(self, config, device='cuda'):
        self.device = device
        self.config = config
        self.enabled_metrics = config.get('evaluation', {}).get('enabled_metrics', {
            'psnr': True,
            'ssim': True,
            'lpips': False,
            'edge_accuracy': False,
            'pattern_consistency': False
        })
        
        # 只在需要时初始化LPIPS模型
        if self.enabled_metrics.get('lpips', False):
            self.lpips_model = lpips.LPIPS(net='alex').to(device)
    
    def calculate_psnr(self, pred, target, max_val=1.0):
        """计算PSNR (Peak Signal-to-Noise Ratio)"""
        mse = torch.mean((pred - target) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * torch.log10(max_val / torch.sqrt(mse))
    
    def calculate_ssim(self, pred, target, window_size=11):
        """计算SSIM (Structural Similarity Index)"""
        # 转换为灰度图
        if pred.shape[1] == 3:
            pred = rgb_to_grayscale(pred)
            target = rgb_to_grayscale(target)
        
        # 创建高斯窗口
        sigma = 1.5
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) 
                            for x in range(window_size)])
        gauss = gauss/gauss.sum()
        
        # 创建1D核并扩展为2D
        kernel = gauss.unsqueeze(1) * gauss.unsqueeze(0)
        kernel = kernel.unsqueeze(0).unsqueeze(0).to(pred.device)
        kernel = kernel.repeat(pred.shape[1], 1, 1, 1)
        
        # 计算均值和方差
        mu1 = F.conv2d(pred, kernel, padding=window_size//2, groups=pred.shape[1])
        mu2 = F.conv2d(target, kernel, padding=window_size//2, groups=target.shape[1])
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(pred * pred, kernel, padding=window_size//2, groups=pred.shape[1]) - mu1_sq
        sigma2_sq = F.conv2d(target * target, kernel, padding=window_size//2, groups=target.shape[1]) - mu2_sq
        sigma12 = F.conv2d(pred * target, kernel, padding=window_size//2, groups=pred.shape[1]) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean()
    
    def calculate_lpips(self, pred, target):
        """计算LPIPS (Learned Perceptual Image Patch Similarity)"""
        return self.lpips_model(pred, target).mean()
    
    def calculate_edge_accuracy(self, pred, target, threshold=0.5):
        """计算边缘准确率"""
        # 转换为灰度图
        if pred.shape[1] == 3:
            pred = rgb_to_grayscale(pred)
            target = rgb_to_grayscale(target)
        
        # 使用Sobel算子检测边缘
        def detect_edges(img):
            img = img.squeeze().cpu().numpy()
            dx = ndimage.sobel(img, axis=0)
            dy = ndimage.sobel(img, axis=1)
            return np.hypot(dx, dy) > threshold
        
        # 批量处理
        pred_edges = torch.stack([torch.from_numpy(detect_edges(p)) for p in pred]).float()
        target_edges = torch.stack([torch.from_numpy(detect_edges(t)) for t in target]).float()
        
        # 计算IoU
        intersection = (pred_edges * target_edges).sum()
        union = (pred_edges + target_edges).gt(0).sum()
        
        return (intersection / union).item() if union > 0 else 0.0
    
    def calculate_pattern_consistency(self, pred, target):
        """计算图案一致性"""
        # 转换为灰度图
        if pred.shape[1] == 3:
            pred = rgb_to_grayscale(pred)
            target = rgb_to_grayscale(target)
        
        # 计算自相关性
        def autocorr2d(x):
            # 确保输入是4D张量 [batch_size, channels, height, width]
            if x.dim() == 5:  # [batch_size, batch_size, channels, height, width]
                x = x.squeeze(1)
            
            # 计算每个通道的自相关性
            batch_size, channels, height, width = x.size()
            
            # 减去均值
            x = x - x.mean(dim=(2, 3), keepdim=True)
            
            # 将所有批次和通道合并为一个批次
            x = x.view(-1, 1, height, width)
            
            # 计算自相关
            kernel = x.clone()
            corr = F.conv2d(x, kernel, padding=(height//2, width//2))
            
            # 重塑回原始维度
            corr = corr.view(batch_size, channels, height, width)
            return corr
        
        pred_corr = autocorr2d(pred)
        target_corr = autocorr2d(target)
        
        return F.mse_loss(pred_corr, target_corr)
    
    def calculate_all_metrics(self, pred, target):
        """计算所有启用的指标"""
        with torch.no_grad():
            metrics = {}
            
            if self.enabled_metrics.get('psnr', True):
                metrics['psnr'] = self.calculate_psnr(pred, target).item()
            
            if self.enabled_metrics.get('ssim', True):
                metrics['ssim'] = self.calculate_ssim(pred, target).item()
            
            if self.enabled_metrics.get('lpips', False):
                metrics['lpips'] = self.calculate_lpips(pred, target).item()
            
            if self.enabled_metrics.get('edge_accuracy', False):
                metrics['edge_acc'] = self.calculate_edge_accuracy(pred, target)
            
            if self.enabled_metrics.get('pattern_consistency', False):
                metrics['pattern_consistency'] = self.calculate_pattern_consistency(pred, target)
        
        return metrics
    
    def update_average_metrics(self, avg_metrics, batch_metrics, batch_size):
        """更新平均指标"""
        if not avg_metrics:
            avg_metrics = {k: 0.0 for k in batch_metrics.keys()}
        
        for k, v in batch_metrics.items():
            avg_metrics[k] = (avg_metrics[k] * batch_size + v) / (batch_size + 1)
        
        return avg_metrics 