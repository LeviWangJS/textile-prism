import warnings
import gc
import logging
import sys
import io
import os
from datetime import datetime

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
from utils.config import load_config
from utils.early_stopping import EarlyStopping
from utils.lr_scheduler import CosineAnnealingWarmup, ReduceLROnPlateauWithWarmup

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

def train(config):
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
    
    # 创建训练进度条
    progress_bar = visualizer.create_progress_bar(config['train']['epochs'])
    
    # 从配置中获取梯度裁剪参数
    max_grad_norm = config['train'].get('max_grad_norm', 1.0)
    
    # 获取损失权重配置
    loss_weights = config['train'].get('loss_weights', None)
    
    best_val_loss = float('inf')
    scheduler = ReduceLROnPlateauWithWarmup(
        optimizer,
        warmup_epochs=config['train']['scheduler'].get('warmup_epochs', 5),
        warmup_lr_init=config['train']['scheduler'].get('warmup_lr_init', 1e-6),
        factor=config['train']['scheduler'].get('factor', 0.5),
        patience=config['train']['scheduler'].get('patience', 5),
        min_lr=config['train']['scheduler'].get('min_lr', 1e-6),
        mode=config['train']['scheduler'].get('mode', 'min')
    )
    
    # 训练循环
    for epoch in range(config['train']['epochs']):
        train_loss = train_one_epoch(
            model, 
            train_loader, 
            criterion, 
            optimizer, 
            device,
            max_grad_norm=max_grad_norm,
            loss_weights=loss_weights,
            logger=logger
        )
        
        # 验证
        val_loss = validate(model, val_loader, criterion, device)
        logger.info(f"Validation Loss: {val_loss:.4f}")
        
        # 更新学习率
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 更新可视化
        visualizer.update_history(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            lr=current_lr
        )
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, 'best_model.pth')
            logger.info(f"Saved best model with validation loss: {val_loss:.4f}")
        
        # 早停检查
        if early_stopping(val_loss):
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break
            
        # 更新进度条
        progress_bar.update(1)
        progress_bar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'lr': f'{current_lr:.6f}'
        })
    
    # 训练结束，创建GIF和清理
    if config['visualization'].get('save_gifs', True):
        visualizer.create_training_gif()
    visualizer.cleanup()
    
    # 关闭进度条
    progress_bar.close()

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

if __name__ == "__main__":
    config = load_config()
    train(config) 