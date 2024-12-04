import os
import sys
import torch
import yaml
from pathlib import Path
import traceback
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.data.dataset import CarpetDataset
from src.models.pattern_transformer import (
    DepthFeatureExtractor,
    PatternTransformer
)

__all__ = ['test_training', 'test_model_architecture', 'test_data_loader', 'test_loss_functions', 'test_optimizer', 'test_training_loop']

def test_model_architecture(config):
    """测试模型架构"""
    print("\n=== 测试模型架构 ===")
    
    # 创建测试输入
    batch_size = 2
    input_size = config['data']['input_size'][0]
    input_images = torch.randn(batch_size, 3, input_size, input_size)
    
    # 将数据移动到正确的设备
    device = torch.device(config['system']['device'])
    input_images = input_images.to(device)
    
    # 创建模型
    model = DepthFeatureExtractor(config).to(device)
    model.train()  # 设置为训练模式
    
    try:
        # 1. 测试前向传播
        print("1. 测试前向传播...")
        outputs = model(input_images)
        print(f"输入形状: {input_images.shape}")
        print(f"输出形状: {outputs.shape}")
        
        # 2. 测试梯度流
        print("\n2. 测试梯度流...")
        loss = outputs.mean()
        loss.backward()
        
        # 检查梯度
        has_grad = False
        for param in model.parameters():
            if param.grad is not None:
                has_grad = True
                print(f"参数梯度范数: {param.grad.norm().item():.4f}")
        
        if not has_grad:
            print("警告：没有参数具有梯度")
        
        # 3. 测试推理模式
        print("\n3. 测试推理模式...")
        model.eval()
        with torch.no_grad():
            eval_outputs = model(input_images)
        print(f"推理输出形状: {eval_outputs.shape}")
        
        print("\n模型架构测试通过！")
        return True
        
    except Exception as e:
        print(f"模型架构测试失败: {str(e)}")
        traceback.print_exc()
        return False

def test_loss_functions(config):
    """测试损失函数"""
    print("\n=== 测试损失函数 ===\n")
    
    try:
        # 创建测试数据
        batch_size = 2
        input_size = config['data']['input_size'][0]
        device = torch.device(config['system']['device'])
        
        # 创建需要梯度的预测和目标
        pred = torch.randn(batch_size, 3, input_size, input_size, requires_grad=True, device=device)
        target = torch.randn(batch_size, 3, input_size, input_size, device=device)
        
        # 1. 测试L1损失
        print("1. 测试L1损失...")
        l1_loss = torch.nn.L1Loss()(pred, target)
        l1_loss.backward(retain_graph=True)
        print(f"L1损失值: {l1_loss.item():.4f}")
        print(f"L1损失梯度范数: {torch.norm(pred.grad).item():.4f}")
        pred.grad.zero_()
        
        # 2. 测试MSE损失
        print("\n2. 测试MSE损失...")
        mse_loss = torch.nn.MSELoss()(pred, target)
        mse_loss.backward(retain_graph=True)
        print(f"MSE损失值: {mse_loss.item():.4f}")
        print(f"MSE损失梯度范数: {torch.norm(pred.grad).item():.4f}")
        pred.grad.zero_()
        
        # 3. 测试SSIM损失
        print("\n3. 测试SSIM损失...")
        def ssim_loss(pred, target, window_size=11):
            C1 = (0.01 * 2) ** 2
            C2 = (0.03 * 2) ** 2
            
            mu1 = F.avg_pool2d(pred, window_size, stride=1, padding=window_size//2)
            mu2 = F.avg_pool2d(target, window_size, stride=1, padding=window_size//2)
            
            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2
            
            sigma1_sq = F.avg_pool2d(pred * pred, window_size, stride=1, padding=window_size//2) - mu1_sq
            sigma2_sq = F.avg_pool2d(target * target, window_size, stride=1, padding=window_size//2) - mu2_sq
            sigma12 = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size//2) - mu1_mu2
            
            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            return 1 - ssim_map.mean()
        
        ssim = ssim_loss(pred, target)
        ssim.backward(retain_graph=True)
        print(f"SSIM损失值: {ssim.item():.4f}")
        print(f"SSIM损失梯度范数: {torch.norm(pred.grad).item():.4f}")
        pred.grad.zero_()
        
        # 4. 测试总变差损失
        print("\n4. 测试总变差损失...")
        def total_variation_loss(x):
            diff_h = x[:, :, 1:, :] - x[:, :, :-1, :]
            diff_w = x[:, :, :, 1:] - x[:, :, :, :-1]
            return (diff_h.abs().mean() + diff_w.abs().mean()) / 2
        
        tv_loss = total_variation_loss(pred)
        tv_loss.backward(retain_graph=True)
        print(f"总变差损失值: {tv_loss.item():.4f}")
        print(f"总变差损失梯度范数: {torch.norm(pred.grad).item():.4f}")
        pred.grad.zero_()
        
        # 5. 测试组合损失
        print("\n5. 测试组合损失...")
        total_loss = l1_loss + mse_loss + ssim + tv_loss
        total_loss.backward()
        print(f"组合损失值: {total_loss.item():.4f}")
        print(f"组合损失梯度范数: {torch.norm(pred.grad).item():.4f}")
        
        print("\n损失函数测试通过！")
        return True
        
    except Exception as e:
        print(f"损失函数测试失败: {str(e)}")
        traceback.print_exc()
        return False

def test_optimizer(config):
    """测试优化器"""
    print("\n=== 测试优化器 ===\n")
    
    try:
        # 创建模型
        device = torch.device(config['system']['device'])
        model = DepthFeatureExtractor(config).to(device)
        
        # 1. 测试优化器创建
        print("1. 测试优化器创建...")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['train']['learning_rate'],
            weight_decay=config['train']['optimizer']['weight_decay']
        )
        print("优化器创建成功")
        
        # 2. 测试学习率调度器
        print("\n2. 测试学习率调度器...")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['train']['epochs'],
            eta_min=1e-6
        )
        print("学习率调度器创建成功")
        
        # 3. 测试参数更新
        print("\n3. 测试参数更新...")
        # 创建测试数据
        batch_size = 2
        input_size = config['data']['input_size'][0]
        test_input = torch.randn(batch_size, 3, input_size, input_size).to(device)
        
        # 前向传播
        output = model(test_input)
        loss = output.mean()
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 记录更新前的参数
        params_before = []
        for param in model.parameters():
            if param.grad is not None:
                params_before.append(param.clone().detach())
        
        # 更新参数
        optimizer.step()
        
        # 检查参数是否更新
        params_updated = False
        for param, param_before in zip(model.parameters(), params_before):
            if param.grad is not None and not torch.equal(param, param_before):
                params_updated = True
                break
        
        if params_updated:
            print("参数成功更新")
        else:
            print("警告：参数未更新")
        
        # 4. 测试学习率调度
        print("\n4. 测试学习率调度...")
        initial_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        updated_lr = optimizer.param_groups[0]['lr']
        print(f"初始学习率: {initial_lr:.6f}")
        print(f"更新后学习率: {updated_lr:.6f}")
        
        # 5. 测试梯度裁剪
        print("\n5. 测试梯度裁剪...")
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        grad_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm().item()
        print(f"裁剪后梯度范数: {grad_norm:.4f}")
        
        print("\n优化器测试通过！")
        return True
        
    except Exception as e:
        print(f"优化器测试失败: {str(e)}")
        traceback.print_exc()
        return False

def test_training_loop(config):
    """测试训练循环"""
    print("\n1. 测试单个epoch...")
    
    try:
        # 创建测试数据
        batch_size = 1
        input_size = config['data']['input_size'][0]
        inputs = torch.randn(batch_size, 3, input_size, input_size)
        targets = torch.randn(batch_size, 3, input_size, input_size)
        
        # 将数据移动到正确的设备
        device = torch.device(config['system']['device'])
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # 创建模型和优化器
        model = PatternTransformer(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # 训练一个批次
        model.train()
        optimizer.zero_grad()
        
        outputs = model(inputs)
        
        # 确保输出和目标尺寸相同
        if outputs.shape != targets.shape:
            outputs = F.interpolate(outputs, size=targets.shape[2:], mode='bilinear', align_corners=True)
        
        loss = F.l1_loss(outputs, targets)
        loss.backward()
        optimizer.step()
        
        print(f"训练损失: {loss.item():.4f}")
        print("\n训练循环测试通过！")
        return True
        
    except Exception as e:
        print(f"训练循环测试失败: {str(e)}")
        traceback.print_exc()
        return False

def test_data_loader(config):
    """测试数据加载器"""
    print("\n=== 测试数据加载器 ===\n")
    
    try:
        # 1. 测试数据集创建
        print("1. 测试数据集创建...")
        train_dataset = CarpetDataset(config, mode='train')
        val_dataset = CarpetDataset(config, mode='val')
        print(f"训练集大小: {len(train_dataset)}")
        print(f"验证集大小: {len(val_dataset)}")
        
        # 2. 测试数据加载器创建
        print("\n2. 测试数据加载器创建...")
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['train']['batch_size'],
            shuffle=True,
            num_workers=config['system']['num_workers']
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['train']['batch_size'],
            shuffle=False,
            num_workers=config['system']['num_workers']
        )
        print(f"训练批次数: {len(train_loader)}")
        print(f"验证批次数: {len(val_loader)}")
        
        # 3. 测试数据批次加载
        print("\n3. 测试数据批次加载...")
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        print(f"训练批次形状:")
        print(f"- 输入图像: {train_batch['input'].shape}")
        print(f"- 目标图像: {train_batch['target'].shape}")
        print(f"验证批次形状:")
        print(f"- 输入图像: {val_batch['input'].shape}")
        print(f"- 目标图像: {val_batch['target'].shape}")
        
        # 4. 测试数据范围
        print("\n4. 测试数据范围...")
        print(f"训练输入范围: [{train_batch['input'].min():.2f}, {train_batch['input'].max():.2f}]")
        print(f"训练目标范围: [{train_batch['target'].min():.2f}, {train_batch['target'].max():.2f}]")
        print(f"验证输入范围: [{val_batch['input'].min():.2f}, {val_batch['input'].max():.2f}]")
        print(f"验证目标范围: [{val_batch['target'].min():.2f}, {val_batch['target'].max():.2f}]")
        
        # 5. 测试数据加载性
        print("\n5. 测试数据加载性能...")
        import time
        start_time = time.time()
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx == 10:  # 只测试前10个批次
                break
        end_time = time.time()
        print(f"加载10个批次用时: {end_time - start_time:.2f}秒")
        print(f"平均每批次用时: {(end_time - start_time) / 10:.2f}秒")
        
        print("\n数据加载器测试通过！")
        return True
        
    except Exception as e:
        print(f"数据加载器测试失败: {str(e)}")
        traceback.print_exc()
        return False

def test_training(config):
    """运行所有训练相关的测试"""
    print("\n=== 开始训练测试 ===")
    
    # 运行各个测试函数
    tests = [
        test_model_architecture,
        test_loss_functions,
        test_optimizer,
        test_training_loop
    ]
    
    success = True
    for test_func in tests:
        try:
            if not test_func(config):
                success = False
        except Exception as e:
            print(f"\n{test_func.__name__} 失败: {str(e)}")
            traceback.print_exc()
            success = False
    
    return success

if __name__ == '__main__':
    test_training()