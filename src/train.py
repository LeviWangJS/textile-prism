import torch
from torch.utils.data import DataLoader
from pathlib import Path
from .data.dataset import CarpetDataset
from .models.transformer import PatternTransformer
from .models.losses import PatternLoss
from .utils.logger import setup_logger

def train(config):
    # 设置日志和设备
    logger = setup_logger(config)
    device = torch.device(config.system.device)
    
    # 创建保存目录
    Path(config.output.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # 初始化数据集和加载器
    dataset = CarpetDataset(config, mode='train')
    dataloader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.system.num_workers
    )
    
    # 初始化模型、损失函数和优化器
    model = PatternTransformer(config).to(device)
    criterion = PatternLoss(config)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.train.learning_rate
    )
    
    # 训练循环
    for epoch in range(config.train.epochs):
        model.train()
        for batch_idx, batch in enumerate(dataloader):
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            losses = criterion(outputs, targets)
            total_loss = losses['total']
            
            total_loss.backward()
            optimizer.step()
            
            # 记录日志
            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch [{epoch+1}/{config.train.epochs}] "
                    f"Batch [{batch_idx}/{len(dataloader)}] "
                    f"Total Loss: {total_loss.item():.4f}"
                )
        
        # 保存检查点
        if (epoch + 1) % config.train.save_interval == 0:
            checkpoint_path = Path(config.output.checkpoint_dir) / f"model_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss.item(),
            }, str(checkpoint_path))
            
            logger.info(f"Saved checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    from utils.config import load_config
    config = load_config()
    train(config) 