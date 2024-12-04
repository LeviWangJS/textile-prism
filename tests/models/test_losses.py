import torch
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from src.models.losses import (
    PatternLoss,
    DepthConsistencyLoss,
    PatternConsistencyLoss,
    EdgeAwareLoss,
    CompositeLoss
)

def test_frequency_loss():
    print("\n=== 测试频率域损失 ===")
    
    # 创建测试数据
    batch_size = 2
    channels = 3
    size = 64
    pred = torch.randn(batch_size, channels, size, size, requires_grad=True)
    target = torch.randn(batch_size, channels, size, size)
    
    # 初始化损失函数
    loss_fn = PatternLoss()
    
    try:
        # 计算损失
        loss = loss_fn(pred, target)
        print(f"频率域损失: {loss.item():.4f}")
        
        # 测试梯度
        loss.backward()
        print(f"预测梯度范围: [{pred.grad.min():.4f}, {pred.grad.max():.4f}]")
        
        # 可视化频谱
        save_dir = Path('test_results/losses')
        save_dir.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(15, 5))
        
        # 显示原始图像
        plt.subplot(131)
        plt.imshow(pred[0].mean(0).detach().numpy(), cmap='gray')
        plt.title('预测图像')
        
        # 显示预测的频谱
        pred_freq = torch.fft.fft2(pred[0].mean(0))
        pred_magnitude = torch.log(torch.abs(pred_freq) + 1)
        plt.subplot(132)
        plt.imshow(pred_magnitude.detach().numpy(), cmap='viridis')
        plt.title('预测频谱')
        
        # 显示目标的频谱
        target_freq = torch.fft.fft2(target[0].mean(0))
        target_magnitude = torch.log(torch.abs(target_freq) + 1)
        plt.subplot(133)
        plt.imshow(target_magnitude.detach().numpy(), cmap='viridis')
        plt.title('目标频谱')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'frequency_loss.png')
        plt.close()
        
        print(f"已保存频率域损失可视化到: {save_dir}")
        
        return True
        
    except Exception as e:
        print(f"频率域损失测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_pattern_loss():
    print("\n=== 测试增强的图案损失 ===")
    
    # 创建测试数据
    batch_size = 2
    channels = 3
    size = 64
    pred = torch.randn(batch_size, channels, size, size, requires_grad=True)
    target = torch.randn(batch_size, channels, size, size)
    
    # 加载配置
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 初始化损失函数
    loss_fn = CompositeLoss()
    
    try:
        # 计算损失
        total_loss, loss_dict = loss_fn(pred, target)
        
        print("各项损失值:")
        for name, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                print(f"{name}: {value.item():.4f}")
            else:
                print(f"{name}: {value:.4f}")
        
        # 测试梯度
        total_loss.backward()
        print(f"\n预测梯度范围: [{pred.grad.min():.4f}, {pred.grad.max():.4f}]")
        
        print("\n增强的图案损失测试通过！")
        return True
        
    except Exception as e:
        print(f"增强的图案损失测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_losses(config=None):
    print("开始测试损失函数...")
    
    # 测试频率域损失
    if not test_frequency_loss():
        return False
    
    # 测试增强的图案损失
    if not test_enhanced_pattern_loss():
        return False
    
    print("\n所有损失函数测试完成")
    return True

if __name__ == '__main__':
    # 加载配置
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置随机种子
    torch.manual_seed(config['system']['seed'])
    
    # 运行测试
    test_losses() 