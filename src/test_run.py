import os
import time
import psutil
import numpy as np
from contextlib import contextmanager

# 设置环境变量启用MPS回退
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import sys
import torch
import torch.nn as nn
import yaml
from pathlib import Path

# 添加src目录到Python路径
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from models.pattern_net import PatternNet
from data.dataloader import create_dataloader
from models.losses import PatternLoss
from utils.logger import setup_logger
from utils.visualization import save_image

@contextmanager
def timer(name):
    """计时器上下文管理器"""
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"{name}: {end - start:.4f} 秒")

def get_memory_usage():
    """获取当前内存使用"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # 转换为MB

def benchmark_device(device_type, model, criterion, batch, num_iterations=100):
    """对指定设备进行基准测试"""
    print(f"\n=== 在 {device_type} 上进行性能测试 ===")
    
    # 设置设备
    if device_type == "mps":
        device = torch.device("mps")
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    else:
        device = torch.device("cpu")
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
    
    # 移动模型到指定设备
    model = model.to(device)
    
    # 准备数据
    inputs = batch['input'].to(device)
    targets = batch['target'].to(device)
    
    # 初始化优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 记录时间
    forward_times = []
    backward_times = []
    total_times = []
    
    # 预热
    print("预热中...")
    for _ in range(10):
        outputs, _ = model(inputs)
        loss = criterion(outputs, targets)['total']
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # 主测试循环
    print(f"开始测试 {num_iterations} 次迭代...")
    for i in range(num_iterations):
        # 记录总时间开始
        start_total = time.perf_counter()
        
        # 前向传播
        start_forward = time.perf_counter()
        outputs, _ = model(inputs)
        loss = criterion(outputs, targets)['total']
        forward_time = time.perf_counter() - start_forward
        forward_times.append(forward_time)
        
        # 反向传播
        start_backward = time.perf_counter()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        backward_time = time.perf_counter() - start_backward
        backward_times.append(backward_time)
        
        # 记录总时间
        total_time = time.perf_counter() - start_total
        total_times.append(total_time)
        
        if (i + 1) % 10 == 0:
            print(f"完成 {i + 1}/{num_iterations} 次迭代")
            print(f"当前迭代 - 前向: {forward_time:.4f}秒, 反向: {backward_time:.4f}秒, 总计: {total_time:.4f}秒")
    
    # 计算统计数据
    stats = {
        "forward_mean": np.mean(forward_times),
        "forward_std": np.std(forward_times),
        "backward_mean": np.mean(backward_times),
        "backward_std": np.std(backward_times),
        "total_mean": np.mean(total_times),
        "total_std": np.std(total_times)
    }
    
    # 打印结果
    print(f"\n{device_type} 性能统计:")
    print(f"前向传播: {stats['forward_mean']:.4f} ± {stats['forward_std']:.4f} 秒")
    print(f"反向传播: {stats['backward_mean']:.4f} ± {stats['backward_std']:.4f} 秒")
    print(f"总时间: {stats['total_mean']:.4f} ± {stats['total_std']:.4f} 秒")
    
    return stats

def compare_performance(config, model, batch):
    """比较CPU和MPS性能"""
    print("\n=== 性能对比测试 ===")
    
    # 创建损失函数
    criterion = PatternLoss(config)
    
    # 测试CPU
    cpu_stats = benchmark_device("cpu", model, criterion, batch)
    
    # 测试MPS
    mps_stats = benchmark_device("mps", model, criterion, batch)
    
    # 计算性能比
    speedup = cpu_stats['total_mean'] / mps_stats['total_mean']
    
    print("\n=== 性能对比总结 ===")
    print(f"速度提升: {speedup:.2f}x (MPS vs CPU)")
    
    if speedup < 1:
        print("\n建议: 由于MPS回退的开销，建议使用CPU进行训练")
    else:
        print("\n建议: MPS性能更好，可以继续使用")

def get_device():
    """获取计算设备"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("使用 MPS 设备 (已启用CPU回退)")
        print("警告: 某些操作将回退到CPU执行，可能影响性能")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("使用 CUDA 设备")
    else:
        device = torch.device("cpu")
        print("使用 CPU 设备")
    return device

def test_environment():
    """测试环境配置"""
    print("\n=== 测试环境 ===")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"MPS可用: {torch.backends.mps.is_available()}")
    print(f"MPS已构建: {torch.backends.mps.is_built()}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    device = get_device()
    print(f"使用设备: {device}")
    
    # 测试数据目录
    data_dir = Path("data/raw")
    if not data_dir.exists():
        raise RuntimeError(f"数据目录不存在: {data_dir}")
    print(f"数据目录检查通过: {data_dir}")
    
    return device

def test_data_loading(config):
    """测试数据加载"""
    print("\n=== 测试数据加载 ===")
    loader = create_dataloader(config, mode='train')
    batch = next(iter(loader))
    
    print(f"批次大小: {batch['input'].shape}")
    print(f"数据类型: {batch['input'].dtype}")
    print(f"值范围: [{batch['input'].min():.2f}, {batch['input'].max():.2f}]")
    
    return batch

def test_model(config, device, batch):
    print("\n=== 测试模型 ===")
    
    # 创建模型
    model = PatternNet(config).to(device)
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    print()
    
    # 测试前向传播
    inputs = batch['input'].to(device)  # 从字典中获取输入
    outputs, aux_outputs = model(inputs)
    
    # 打印输出信息
    print(f"输出形状: {outputs.shape}")
    print(f"输出值范围: [{outputs.min():.2f}, {outputs.max():.2f}]")
    
    return model

def test_loss(config, device, model, batch):
    print("\n=== 测试损失函数 ===")
    
    # 确保配置存在
    if 'loss' not in config:
        print("添加默认损失函数配置")
        config['loss'] = {
            'lambda_l1': 1.0,
            'lambda_perceptual': 0.1,
            'vgg_layers': [3, 8, 15, 22],
            'perceptual_weights': [1.0, 1.0, 1.0, 1.0]
        }
    
    # 初始化损失函数
    criterion = PatternLoss(config)
    criterion.debug = True
    
    # 从batch字典中获取数据
    if isinstance(batch, dict):
        inputs = batch.get('input')
        targets = batch.get('target', inputs)  # 如果没有target，使用input
    else:
        raise TypeError(f"Expected dict for batch, got {type(batch)}")
    
    # 确保数据类型正确
    if not isinstance(inputs, torch.Tensor):
        raise TypeError(f"Expected tensor for inputs, got {type(inputs)}")
    
    # 移动到设备
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    print(f"Input shape: {inputs.shape}, device: {inputs.device}")
    print(f"Target shape: {targets.shape}, device: {targets.device}")
    
    # 前向传播
    outputs, _ = model(inputs)
    
    # 计算损失
    try:
        losses = criterion(outputs, targets)
        print("\n损失值:")
        for name, value in losses.items():
            print(f"{name}: {value.item():.4f}")
    except Exception as e:
        print(f"计算损失时出错: {str(e)}")
        raise

def test_training_step(config, device, model, batch):
    print("\n=== 测试训练步骤 ===")
    
    # 初始化优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 初始化损失函数
    criterion = PatternLoss(config)
    
    try:
        # 获取输入和目标
        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs, _ = model(inputs)
        
        # 计算损失
        losses = criterion(outputs, targets)
        total_loss = losses['total']
        
        # 反向传播
        print("\n开始反向传播...")
        total_loss.backward()
        
        # 更新参数
        optimizer.step()
        
        print("\n训练步骤完成")
        print(f"总损失: {total_loss.item():.4f}")
        
    except Exception as e:
        print(f"\n训练步骤出错: {str(e)}")
        if "MPS" in str(e):
            print("\n提示: MPS设备不支持某些操作，已启用CPU回退")
        raise

def main():
    # 加载配置
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # 创建必要的目录
    for dir_name in ['outputs', 'checkpoints', 'logs']:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
    
    # 测试环境
    device = test_environment()
    
    # 测试数据加载
    batch = test_data_loading(config)
    
    # 测试模型
    model = test_model(config, device, batch)
    
    # 测试损失函数
    test_loss(config, device, model, batch)
    
    # 测试训练步骤
    test_training_step(config, device, model, batch)
    
    # 添加性能对比测试
    compare_performance(config, model, batch)

if __name__ == "__main__":
    main() 