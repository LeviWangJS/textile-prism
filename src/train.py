from pathlib import Path
import torch
from torch.utils.data import DataLoader
import gc
from tqdm import tqdm
from data.dataset import CarpetDataset
from models.transformer import PatternTransformer
from models.losses import PatternLoss
from utils.logger import setup_logger
from utils.visualization import save_image
from utils.config import load_config

def train(config):
    # 设置日志和设备
    logger = setup_logger(config)
    device = torch.device('cpu')
    
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
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['train']['learning_rate']
    )
    
    # 梯度累积步数
    accumulation_steps = config['train'].get('gradient_accumulation', 1)
    
    # 训练循环
    for epoch in range(config['train']['epochs']):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        # 使用tqdm显示进度条
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["train"]["epochs"]}') as pbar:
            for batch_idx, batch in enumerate(pbar):
                # 定期清理内存
                gc.collect()
                
                inputs = batch['input'].to(device)
                targets = batch['target'].to(device)
                
                # 前向传播
                outputs = model(inputs)
                losses = criterion(outputs, targets)
                loss = losses['total']
                
                # 梯度累积
                loss = loss / accumulation_steps
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # 更新进度条
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                # 累积足够步数后更新参数
                if (batch_idx + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                total_loss += loss.item()
                
                # 记录日志
                if batch_idx % 10 == 0:
                    logger.info(
                        f"Epoch [{epoch+1}/{config['train']['epochs']}] "
                        f"Batch [{batch_idx}/{len(train_loader)}] "
                        f"Total Loss: {loss.item():.4f}"
                    )
        
        # 每个epoch结束后验证和保存
        if (epoch + 1) % config['train']['save_interval'] == 0:
            val_loss = validate(model, val_loader, criterion, device)
            logger.info(f"Validation Loss: {val_loss:.4f}")
            
            # 保存检查点
            checkpoint_path = Path(config['output']['checkpoint_dir']) / f"model_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': total_loss / len(train_loader),
                'val_loss': val_loss
            }, str(checkpoint_path))
            
            # 可视化一些结果
            visualize_results(model, val_loader, epoch + 1, config)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            outputs = model(inputs)
            losses = criterion(outputs, targets)
            total_loss += losses['total'].item()
    
    return total_loss / len(val_loader)

def visualize_results(model, val_loader, epoch, config):
    """保存一些验证集的可视化结果"""
    model.eval()
    save_dir = Path(config['output']['log_dir']) / f'epoch_{epoch}'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 5:  # 只保存前5个样本
                break
                
            inputs = batch['input'].to(model.device)
            outputs = model(inputs)
            
            # 保存输入、输出和目标图像
            save_image(inputs[0], save_dir / f'sample_{i}_input.png')
            save_image(outputs[0], save_dir / f'sample_{i}_output.png')
            save_image(batch['target'][0], save_dir / f'sample_{i}_target.png')

if __name__ == "__main__":
    config = load_config()
    train(config) 