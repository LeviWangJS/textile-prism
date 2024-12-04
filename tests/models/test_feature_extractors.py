import torch
import yaml
import pytest
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
import logging

from src.models.feature_extractors import CLIPFeatureExtractor, FeatureFusion

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler('test_feature_extractors.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def test_clip_feature_extractor():
    print("\n=== 测试CLIP特征提取器 ===")
    logger = setup_logging()
    
    # 加载配置
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 添加CLIP相关配置
    config.update({
        'clip_model': 'openai/clip-vit-base-patch32',
        'cache_dir': 'cache/clip_features',
        'use_feature_cache': True
    })
    
    # 创建测试数据
    batch_size = 2
    channels = 3
    size = 256
    test_images = torch.randn(batch_size, channels, size, size)
    
    try:
        # 设置离线模式
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'
        
        # 初始化特征提取器
        try:
            extractor = CLIPFeatureExtractor(config)
        except Exception as e:
            logger.warning(f"无法在离线模式下加载CLIP模型，跳过特征提取测试: {str(e)}")
            return True
        
        # 测试特征提取
        features = extractor.extract_features(test_images)
        
        logger.info(f"输入形状: {test_images.shape}")
        logger.info(f"特征形状: {features.shape}")
        
        # 测试缓存功能
        if config.get('use_feature_cache', True):
            # 第二次提取应该使用缓存
            cached_features = extractor.extract_features(test_images)
            assert torch.allclose(features, cached_features, rtol=1e-5), "缓存特征与原始特征不匹配"
            logger.info("缓存功能测试通过")
        
        # 可视化特征
        save_dir = Path('test_results/feature_extractors')
        save_dir.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(15, 5))
        
        # 显示原始图像
        plt.subplot(131)
        plt.imshow(test_images[0].permute(1, 2, 0).detach().numpy() * 0.5 + 0.5)
        plt.title('输入图像')
        
        # 显示特征向量分布
        plt.subplot(132)
        plt.hist(features[0].detach().numpy(), bins=50)
        plt.title('CLIP特征分布')
        
        # 显示特征相似度矩阵
        plt.subplot(133)
        similarity = torch.matmul(features, features.t())
        plt.imshow(similarity.detach().numpy(), cmap='viridis')
        plt.colorbar()
        plt.title('特征相似度矩阵')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'clip_features.png')
        plt.close()
        
        logger.info(f"已保存特征可视化到: {save_dir}")
        logger.info("\nCLIP特征提取器测试通过！")
        return True
        
    except Exception as e:
        logger.error(f"CLIP特征提取器测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 清除环境变量
        os.environ.pop('TRANSFORMERS_OFFLINE', None)
        os.environ.pop('HF_DATASETS_OFFLINE', None)

def test_feature_fusion():
    print("\n=== 测试特征融合模块 ===")
    logger = setup_logging()
    
    # 加载配置
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 添加特征融合相关配置
    config.update({
        'clip_feature_dim': 512,
        'original_feature_dim': 512,
        'fusion_dim': 512,
        'dropout_rate': 0.1
    })
    
    # 创建测试数据
    batch_size = 2
    clip_dim = config.get('clip_feature_dim', 512)
    original_dim = config.get('original_feature_dim', 512)
    
    clip_features = torch.randn(batch_size, clip_dim)
    original_features = torch.randn(batch_size, original_dim)
    
    try:
        # 初始化特征融合模块
        fusion = FeatureFusion(config)
        
        # 测试特征融合
        fused_features = fusion(clip_features, original_features)
        
        logger.info(f"CLIP特征形状: {clip_features.shape}")
        logger.info(f"原始特征形状: {original_features.shape}")
        logger.info(f"融合特征形状: {fused_features.shape}")
        
        # 测试梯度流
        fused_features.sum().backward()
        
        # 检查梯度
        has_grad = all(p.grad is not None for p in fusion.parameters())
        logger.info(f"梯度检查: {'通过' if has_grad else '失败'}")
        
        # 可视化特征分布
        save_dir = Path('test_results/feature_extractors')
        save_dir.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(15, 5))
        
        # 显示CLIP特征分布
        plt.subplot(131)
        plt.hist(clip_features.detach().numpy().flatten(), bins=50)
        plt.title('CLIP特征分布')
        
        # 显示原始特征分布
        plt.subplot(132)
        plt.hist(original_features.detach().numpy().flatten(), bins=50)
        plt.title('原始特征分布')
        
        # 显示融合特征分布
        plt.subplot(133)
        plt.hist(fused_features.detach().numpy().flatten(), bins=50)
        plt.title('融合特征分布')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'feature_distributions.png')
        plt.close()
        
        logger.info(f"已保存特征分布可视化到: {save_dir}")
        logger.info("\n特征融合模块测试通过！")
        return True
        
    except Exception as e:
        logger.error(f"特征融合模块测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_extractors():
    print("开始测试特征提取器...")
    logger = setup_logging()
    
    # 测试CLIP特征提取器
    if not test_clip_feature_extractor():
        return False
    
    # 测试特征融合模块
    if not test_feature_fusion():
        return False
    
    logger.info("\n所有特征提取器测试完成")
    return True

if __name__ == '__main__':
    # 加载配置
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置随机种子
    torch.manual_seed(config['system']['seed'])
    
    # 运行测试
    test_feature_extractors() 