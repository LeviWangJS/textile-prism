import warnings
import gc
import logging
import sys
import io
import os
from datetime import datetime
from typing import Dict

# 配置logger
logger = logging.getLogger('pattern_transform')
logger.setLevel(logging.INFO)

# 如果还没有处理器，添加一个
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# 使用更通用的方式过滤警告
warnings.filterwarnings('ignore', message='.*OpenSSL.*')
warnings.filterwarnings('ignore', message='.*pretrained.*')
warnings.filterwarnings('ignore', message='.*weights.*')
warnings.filterwarnings('ignore', message='.*batch_first.*')
warnings.filterwarnings('ignore', message='.*verbose.*')
warnings.filterwarnings('ignore', message='.*norm_first.*')

# 其他导入
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
import os
from pathlib import Path

from data.dataset import CarpetDataset
from models.transformer import PatternTransformer
from models.losses import PatternLoss
from utils.logger import setup_logger
from utils.visualizer import save_image, TrainingVisualizer
from utils.config import load_config, validate_scheduler_config
from utils.early_stopping import EarlyStopping
from utils.lr_scheduler import LRSchedulerWrapper
# from utils.openai_helper import OpenAIHelper
from utils.smart_monitor import SmartTrainingMonitor
from utils.feature_visualizer import FeatureVisualizer
from optimization.auto_tuner import AutoTuner, OptimizationConfig, OptimizationStrategyGenerator
from optimization.dynamic_tuner import DynamicTuner
from utils.visualization import ParameterVisualizer, visualize_augmentations

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def create_optimizer(model, config):
    """创建优化器"""
    lr = config['train']['optimizer']['lr']
    weight_decay = config['train']['optimizer'].get('weight_decay', 0)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,  # 这个将作为initial_lr
        weight_decay=weight_decay
    )
    
    return optimizer

def train_one_epoch(model, train_loader, criterion, optimizer, device, max_grad_norm=1.0, loss_weights=None, logger=None):
    """单个epoch的训练循环"""
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    
    for batch_idx, batch in enumerate(train_loader):
        outputs = model(batch['input'].to(device))
        loss_dict = criterion(outputs, batch['target'].to(device))
        
        # 计算总损失
        if isinstance(loss_dict, dict):
            weighted_loss = 0
            loss_info = {}
            
            for loss_name, loss_value in loss_dict.items():
                if not isinstance(loss_value, torch.Tensor):
                    loss_value = torch.tensor(loss_value, device=device)
                    
                weight = loss_weights.get(loss_name, 1.0) if loss_weights else 1.0
                weighted_loss += weight * loss_value
                loss_info[loss_name] = loss_value.item()
                
            total_loss_tensor = weighted_loss
        else:
            total_loss_tensor = loss_dict
            loss_info = {'total_loss': loss_dict.item()}
        
        # 反向传播
        optimizer.zero_grad()
        total_loss_tensor.backward()
        
        # 梯度裁剪
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            max_norm=max_grad_norm
        )
        
        # 优化器步进
        optimizer.step()
        
        # 更新总损失
        total_loss += total_loss_tensor.item()
        
        # 记录详细信息
        if batch_idx % 10 == 0 and logger is not None:
            log_msg = f"Batch [{batch_idx}/{num_batches}] "
            for name, value in loss_info.items():
                log_msg += f"{name}: {value:.4f} "
            log_msg += f"Grad Norm: {grad_norm:.4f}"
            logger.info(log_msg)
    
    return total_loss / num_batches

def validate_config(config: Dict) -> None:
    """验证配置参数的类型和值"""
    lr_config = config.get('train', {}).get('dynamic_tuning', {}).get('lr_adjust', {})
    
    # 验证学习率相关配置
    required_float_params = ['min_lr', 'max_lr', 'adjust_threshold']
    for param in required_float_params:
        try:
            lr_config[param] = float(lr_config[param])
        except (ValueError, TypeError, KeyError):
            raise ValueError(
                f"Invalid {param} in config. Please ensure it's a valid number."
            )
    
    # 验证值的合理性
    if lr_config['min_lr'] >= lr_config['max_lr']:
        raise ValueError("min_lr must be less than max_lr")
    if lr_config['adjust_threshold'] <= 0:
        raise ValueError("adjust_threshold must be positive")

def check_augmentations(config: Dict):
    """检查数据增强效果"""
    dataset = CarpetDataset(config, mode='train')
    
    # 保存增强效果图
    save_dir = 'visualization/augmentations'
    os.makedirs(save_dir, exist_ok=True)
    
    # 检查多个样本
    for i in range(min(5, len(dataset))):
        save_path = os.path.join(save_dir, f'augmentation_sample_{i}.png')
        visualize_augmentations(dataset, i, n_samples=5, save_path=save_path)
        
    logger.info(f"Augmentation visualization saved to {save_dir}")

def train(config: Dict):
    # 在训练开始前检查数据增强
    if config.get('debug', {}).get('check_augmentation', False):
        check_augmentations(config)
    
    # 在训练开始前验证配置
    validate_config(config)
    
    # 验证配置
    validate_scheduler_config(config)
    
    # 设置日志和设备
    logger = setup_logger(config)
    device = get_device()
    
    # 创建保存目录
    Path(config['output']['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['output']['log_dir']).mkdir(parents=True, exist_ok=True)
    
    # 初始化数据集和加载器
    train_dataset = CarpetDataset(config, mode='train')
    val_dataset = CarpetDataset(config, mode='val')
    
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
    
    # 初始化模型、损失函数和优化器
    model = PatternTransformer(config).to(device)
    criterion = PatternLoss(config)
    optimizer = create_optimizer(model, config)
    
    # 初始化早停
    early_stopping = EarlyStopping(
        patience=config['train']['early_stopping']['patience'],
        min_delta=config['train']['early_stopping']['min_delta'],
        verbose=config['train']['early_stopping']['verbose']
    )
    
    # 初始化可视化器
    visualizer = TrainingVisualizer(save_dir='logs')
    feature_visualizer = FeatureVisualizer(save_dir=os.path.join('logs', 'feature_distributions'))
    
    # 创建训练进度条
    progress_bar = visualizer.create_progress_bar(config['train']['epochs'])
    
    # 从配置中获取梯度裁剪参数
    max_grad_norm = config['train'].get('max_grad_norm', 1.0)
    
    # 获取损失权重配置
    loss_weights = config['train'].get('loss_weights', None)
    
    best_val_loss = float('inf')
    scheduler = LRSchedulerWrapper(optimizer, config)
    
    # 初始化 TensorBoard
    writer = SummaryWriter(log_dir=config['visualization']['save_dir'])
    
    # 初始化OpenAI助手和监控器
    # openai_helper = OpenAIHelper(config['openai']['api_key'], config)
    smart_monitor = SmartTrainingMonitor(config, openai_helper)
    
    # 初始化动态调整器
    if config['train']['dynamic_tuning']['enabled']:
        dynamic_tuner = DynamicTuner(config)
        param_visualizer = ParameterVisualizer(save_dir=config['visualization']['save_dir'])
    else:
        dynamic_tuner = None
        
    # 添加epoch级别的进度条
    with tqdm(range(config['train']['epochs']), desc='Epochs') as epoch_pbar:
        for epoch in epoch_pbar:
            model.train()
            train_loss = 0
            
            # batch级别的进度条
            with tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{config["train"]["epochs"]}', 
                     leave=False) as pbar:  # leave=False 让batch进度条在完成后消失
                for batch_idx, batch in enumerate(pbar):
                    # 获取数据
                    inputs = batch['input'].to(device)
                    targets = batch['target'].to(device)
                    
                    # 清零梯度
                    optimizer.zero_grad()
                    
                    # 前向传播
                    outputs = model(inputs)
                    
                    # 计算损失
                    losses = criterion(outputs, targets)
                    total_loss = losses['total']
                    
                    # 反向传播
                    total_loss.backward()
                    
                    # 计算梯度范数
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # 更新参数
                    optimizer.step()
                    
                    # 更新batch进度条
                    train_loss += total_loss.item()
                    current_lr = optimizer.param_groups[0]['lr']
                    pbar.set_postfix({
                        'train_loss': f'{train_loss/(batch_idx+1):.4f}',
                        'lr': f'{current_lr:.6f}'
                    })
                    
                    # 记录训练信息
                    logger.info(f"Batch [{batch_idx}/{len(train_loader)}] "
                              f"total: {total_loss.item():.4f} "
                              f"l1: {losses['l1'].item():.4f} "
                              f"perceptual: {losses['perceptual'].item():.4f} "
                              f"Grad Norm: {grad_norm:.4f}")
                    
                    # 智能分析
                    metrics = {
                        'train_loss': train_loss/(batch_idx+1),
                        'current_lr': current_lr,
                        'grad_norm': grad_norm
                    }
                    metrics = smart_monitor.analyze_batch(
                        epoch, batch_idx, outputs, targets, metrics
                    )
                    
                    # 更新进度条
                    pbar.set_postfix(metrics)
            
            # 计算平均训练损失
            train_loss = train_loss / len(train_loader)
            
            # 验证
            val_loss = validate(model, val_loader, criterion, device)
            
            # 更新epoch进度条
            epoch_pbar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'val_loss': f'{val_loss:.4f}',
                'lr': f'{current_lr:.6f}'
            })
            
            # 更新学习率调度器
            scheduler.step(val_loss)
            
            # 记录指标
            writer.add_scalar('Loss/train_epoch', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('LR/epoch', current_lr, epoch)
            
            # 更新可视化器
            visualizer.update_history(epoch, train_loss, val_loss, current_lr)
            visualizer.plot_losses(save=True)
            
            # 保存检查点和最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, epoch, 'best_model.pth')
                logger.info(f"Saved best model with validation loss: {val_loss:.4f}")
            
            # 早停检查
            if early_stopping(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
            
            # 获取epoch级别的建议
            epoch_metrics = {
                'train_loss': train_loss/len(train_loader),
                'val_loss': val_loss,
                'quality_score': metrics.get('quality_score', 0)
            }
            suggestions = smart_monitor.get_epoch_suggestions(epoch_metrics)
            if suggestions:
                logger.info(f"Training Suggestions: {suggestions}")
            
            # 批次级别的动态调整
            if dynamic_tuner and batch_idx % config['train']['dynamic_tuning'].get('batch_adjust_freq', 10) == 0:
                metrics = {
                    'train_loss': train_loss/(batch_idx+1),
                    'batch_idx': batch_idx,
                    'epoch': epoch
                }
                
                # 调整批次大小
                if config['train']['dynamic_tuning']['batch_size_adjust']['enabled']:
                    new_batch_size = dynamic_tuner.update_batch_size(train_loader, metrics)
                    if new_batch_size != train_loader.batch_size:
                        train_loader = update_dataloader(train_dataset, new_batch_size)
                        
            # epoch级别的动态调整
            if dynamic_tuner:
                metrics = {
                    'train_loss': train_loss/len(train_loader),
                    'val_loss': val_loss,
                    'epoch': epoch
                }
                
                # 调整学习率
                if config['train']['dynamic_tuning']['lr_adjust']['enabled']:
                    new_lr = dynamic_tuner.update_learning_rate(optimizer, metrics, epoch)
                    
                # 更新优化器的学习率
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                    
                # 记录学习率变化
                logger.info(f"Learning rate adjusted to: {new_lr}")
                
                # 如果使用wandb，也记录到wandb
                if config.get('logging', {}).get('use_wandb', False):
                    wandb.log({'learning_rate': new_lr}, step=epoch)
                
                # 调整模型参数
                if config['train']['dynamic_tuning']['dropout_adjust']['enabled']:
                    dynamic_tuner.update_model_params(model, metrics)
                
                # 可视化参数调整
                if config['train']['dynamic_tuning']['logging']['visualization']:
                    param_visualizer.log_parameters({
                        'learning_rate': optimizer.param_groups[0]['lr'],
                        'batch_size': train_loader.batch_size,
                        'dropout': model.dropout.p if hasattr(model, 'dropout') else None,
                        'train_loss': train_loss,
                        'val_loss': val_loss
                    }, epoch)
    
    # 训练结束，关闭writer
    writer.close()
    
    # 训练结束，生成参数调整报告
    if dynamic_tuner and config['train']['dynamic_tuning']['logging']['visualization']:
        param_visualizer.generate_adjustment_report()
        
    return {
        'best_val_loss': best_val_loss,
        'epochs_trained': epoch + 1
    }

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    num_batches = len(val_loader)
    
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            outputs = model(inputs)
            losses = criterion(outputs, targets)
            total_loss += losses['total'].item()
    
    return total_loss / num_batches

def visualize_results(model, val_loader, epoch, config, device):
    """保存一些验证集的可视化结果"""
    model.eval()
    save_dir = Path(config['output']['log_dir']) / f'epoch_{epoch}'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 5:  # 只保存前5个样本
                break
                
            inputs = batch['input'].to(device)
            outputs = model(inputs)
            
            # 保存输入、输出和目标图像
            save_image(inputs[0], save_dir / f'sample_{i}_input.png')
            save_image(outputs[0], save_dir / f'sample_{i}_output.png')
            save_image(batch['target'][0], save_dir / f'sample_{i}_target.png')

def save_checkpoint(model, optimizer, epoch, filename):
    """保存检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前epoch
        filename: 保存的文件名
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    # 确保checkpoints目录存在
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
        
    # 保存检查点
    checkpoint_path = os.path.join('checkpoints', filename)
    torch.save(checkpoint, checkpoint_path)
    logger.info(f'Saved checkpoint to {checkpoint_path}')

def setup_logging():
    """设置日志配置"""
    logger = logging.getLogger('pattern_transform')
    if not logger.handlers:  # 防止重复添加handler
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 文件处理器
        os.makedirs('logs', exist_ok=True)
        file_handler = logging.FileHandler(f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def main():
    # 加载配置
    config = load_config()
    
    # 是否使用自动调优
    if config.get('use_auto_tuning', False):  # 从配置中读取是否使用自动调优
        # 创建优化配置
        opt_config = OptimizationConfig(
            n_trials=100,
            timeout=3600 * 24,  # 24小时
            n_jobs=2
        )
        
        # 初始化自动调器
        auto_tuner = AutoTuner(
            base_config=config,
            optimization_config=opt_config,
            study_name="pattern_transformer_optimization"
        )
        
        try:
            # 执行优化
            results = auto_tuner.optimize()
            
            # 获取优化策略
            strategy_generator = OptimizationStrategyGenerator(auto_tuner.study)
            strategy = strategy_generator.generate_strategy()
            
            # 打印优化结果
            print("Best configuration:", results['best_config'])
            print("\nOptimization suggestions:")
            for suggestion in strategy['suggestions']:
                print(f"- {suggestion}")
            
            # 使用优化后的配置进行训练
            optimized_config = results['best_config']
            train(optimized_config)
            
        except Exception as e:
            print(f"Optimization failed: {str(e)}")
            print("Falling back to default configuration...")
            train(config)  # 如果优化失败，使用默认配置
    else:
        # 直接使用默认配置进行训练
        train(config)

if __name__ == "__main__":
    main() 