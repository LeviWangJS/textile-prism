import os
import sys
import torch
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.models.pattern_transformer import (
    DepthFeatureExtractor,
    SelfAttention,
    EnhancedSpatialTransformer,
    FeaturePyramidNetwork
)
from src.utils.visualizer import Visualizer

def save_image(tensor, path):
    """保存图像张量为PNG文件"""
    # 确保输入是CPU张量
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # 转换为numpy数组
    img = tensor.detach().numpy()
    
    # 调整维度顺序从[C,H,W]到[H,W,C]
    if img.shape[0] in [1, 3]:
        img = np.transpose(img, (1, 2, 0))
    
    # 归一化到[0,1]
    img = (img + 1) / 2
    img = np.clip(img, 0, 1)
    
    # 保存图像
    plt.imsave(path, img)

def visualize_depth(depth_tensor, path):
    """可视化深度图"""
    # 确保输入是CPU张量
    if depth_tensor.is_cuda:
        depth_tensor = depth_tensor.cpu()
    
    # 转换为numpy数组并压缩到2D
    depth = depth_tensor.detach().numpy()
    if len(depth.shape) > 2:
        depth = depth.squeeze()
    
    # 创建图像
    plt.figure(figsize=(10, 10))
    plt.imshow(depth, cmap='viridis')
    plt.colorbar(label='Depth')
    plt.title('Depth Map')
    
    # 保存图像
    plt.savefig(path)
    plt.close()

def visualize_attention_maps(attention_maps, path):
    """可视化注意力图"""
    # 确保输入是CPU张量
    if attention_maps.is_cuda:
        attention_maps = attention_maps.cpu()
    
    attention = attention_maps.detach().numpy()
    
    # 创建单个图像
    plt.figure(figsize=(10, 10))
    plt.imshow(attention.mean(axis=0), cmap='viridis')  # 对通道维度取平均
    plt.colorbar(label='Attention')
    plt.title('Attention Map')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def test_depth_module(config):
    """测试深度特征提取模块"""
    print("\n=== 测试深度特征提取模块 ===")
    
    try:
        # 创建测试数据
        batch_size = 2
        channels = 3
        size = 256
        x = torch.randn(batch_size, channels, size, size)
        
        # 初始化模块
        depth_module = DepthFeatureExtractor(config)
        
        # 测试前向传播
        features = depth_module(x)
        
        print(f"输入形状: {x.shape}")
        print(f"输出形状: {features.shape}")
        
        # 保存可视化结果
        save_dir = Path('test_results/depth_module')
        save_dir.mkdir(parents=True, exist_ok=True)
        
        visualizer = Visualizer(config)
        visualizer.save_image(x[0], save_dir / 'input.png')
        visualizer.save_image(features[0].mean(dim=0, keepdim=True).repeat(3,1,1), 
                            save_dir / 'features.png')
        
        print("\n深度特征提取模块测试通过！")
        return True
        
    except Exception as e:
        print(f"深度特征提取模块测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_spatial_transformer(config):
    """测试空间变换器模块"""
    print("\n=== 测试空间变换器模块 ===")
    
    try:
        # 创建测试数据
        batch_size = 2
        channels = 512
        size = 64
        x = torch.randn(batch_size, channels, size, size)
        
        # 初始化模块
        stn = EnhancedSpatialTransformer(config)
        
        # 测试前向传播
        output = stn(x, level=0)
        
        print(f"输入形状: {x.shape}")
        print(f"输出形状: {output.shape}")
        
        # 保存可视化结果
        save_dir = Path('test_results/spatial_transformer')
        save_dir.mkdir(parents=True, exist_ok=True)
        
        visualizer = Visualizer(config)
        visualizer.save_image(x[0].mean(dim=0, keepdim=True).repeat(3,1,1), 
                            save_dir / 'input.png')
        visualizer.save_image(output[0].mean(dim=0, keepdim=True).repeat(3,1,1), 
                            save_dir / 'output.png')
        
        print("\n空间变换器模块测试通过！")
        return True
        
    except Exception as e:
        print(f"空间变换器模块测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_full_model(config):
    """测试完整模型"""
    print("\n=== 测试完整模型 ===")
    
    try:
        # 创建测试数据
        batch_size = 2
        channels = 3
        size = 256
        x = torch.randn(batch_size, channels, size, size)
        
        # 初始化模型
        model = DepthFeatureExtractor(config)
        
        # 测试前向传播
        output = model(x)
        
        print(f"输入形状: {x.shape}")
        print(f"输出形状: {output.shape}")
        
        # 保存可视化结果
        save_dir = Path('test_results/full_model')
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建测试输入
        test_input = torch.ones(1, 3, size, size) * 0.5  # 灰色图像
        test_input[:, :, size//4:size//2, size//4:size//2] = 1.0  # 添加白色方块
        
        # 生成输出
        test_output = model(test_input)
        
        # 将特征图转换为RGB图像格式
        feature_vis = test_output[0].mean(dim=0, keepdim=True).repeat(3, 1, 1)
        feature_vis = (feature_vis - feature_vis.min()) / (feature_vis.max() - feature_vis.min())
        
        visualizer = Visualizer(config)
        visualizer.save_image(test_input[0], save_dir / 'test_input.png')
        visualizer.save_image(feature_vis, save_dir / 'test_output.png')
        
        print("\n完整模型测试通过！")
        return True
        
    except Exception as e:
        print(f"完整模型测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_model_components(config):
    print("开始测试模型组件...")
    
    # 测试深度感知模块
    if not test_depth_module(config):
        return False
    
    # 测试空间变换网络
    if not test_spatial_transformer(config):
        return False
    
    # 测试完整模型
    if not test_full_model(config):
        return False
    
    print("\n所有模型组件测试完成")
    return True

if __name__ == '__main__':
    # 加载配置
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置随机种子
    torch.manual_seed(config['system']['seed'])
    
    # 设置设备
    device = torch.device(config['system']['device'])
    print(f"使用设备: {device}")
    
    # 运行测试
    test_model_components(config) 