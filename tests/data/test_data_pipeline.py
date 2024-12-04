import torch
import yaml
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.data.dataset import CarpetDataset
from src.utils.visualizer import Visualizer

def test_data_pipeline(config):
    print("=== 开始测试数据流程 ===")
    
    # 创建可视化器
    visualizer = Visualizer(config)
    
    # 1. 测试数据集加载
    print("\n1. 测试数据集加载...")
    try:
        train_dataset = CarpetDataset(config, mode='train')
        val_dataset = CarpetDataset(config, mode='val')
        print(f"训练集大小: {len(train_dataset)}")
        print(f"验证集大小: {len(val_dataset)}")
    except Exception as e:
        print(f"数据集加载失败: {str(e)}")
        return False
    
    # 2. 测试数据批处理
    print("\n2. 测试数据批处理...")
    batch_size = config['train']['batch_size']
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config['system']['num_workers']
    )
    
    # 3. 检查数据格式
    print("\n3. 检查数据格式...")
    batch = next(iter(train_loader))
    print(f"输入张量形状: {batch['input'].shape}")
    print(f"目标张量形状: {batch['target'].shape}")
    print(f"输入值范围: [{batch['input'].min():.2f}, {batch['input'].max():.2f}]")
    print(f"目标值范围: [{batch['target'].min():.2f}, {batch['target'].max():.2f}]")
    
    # 4. 测试数据增强
    print("\n4. 测试数据增强...")
    if config['data']['augmentation']['enabled']:
        # 保存原始图像和增强后的图像进行对比
        save_dir = Path('test_results/data_augmentation')
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(min(5, len(train_dataset))):
            sample = train_dataset[i]
            visualizer.save_image(
                sample['input'],
                save_dir / f'sample_{i}_input.png'
            )
            visualizer.save_image(
                sample['target'],
                save_dir / f'sample_{i}_target.png'
            )
        print(f"已保存增强样本到: {save_dir}")
    else:
        print("数据增强未启用")
    
    # 5. 测试内存占用
    print("\n5. 测试内存占用...")
    try:
        total_memory = 0
        for batch in tqdm(train_loader, desc="检查内存占用"):
            batch_memory = (batch['input'].element_size() * batch['input'].nelement() +
                          batch['target'].element_size() * batch['target'].nelement())
            total_memory += batch_memory
        print(f"估计总内存占用: {total_memory / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"内存测试失败: {str(e)}")
        return False
    
    # 6. 测试数据加载速度
    print("\n6. 测试数据加载速度...")
    import time
    start_time = time.time()
    for _ in tqdm(train_loader, desc="测试加载速度"):
        pass
    end_time = time.time()
    print(f"数据加载速度: {len(train_loader)/(end_time-start_time):.2f} 批次/秒")
    
    print("\n=== 数据流程测试完成 ===")
    return True

if __name__ == '__main__':
    # 加载配置
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置随机种子
    torch.manual_seed(config['system']['seed'])
    
    # 运行测试
    test_data_pipeline(config) 