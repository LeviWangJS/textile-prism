import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional
from ..models.pattern_net import PatternNet
from ..models.losses import PatternLoss
from ..data.dataloader import create_dataloader
from ..utils.logger import setup_logger
from ..utils.visualization import save_image

class Trainer:
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config['system']['device'])
        self.logger = setup_logger(config)
        
        # 创建模型
        self.model = PatternNet(config).to(self.device)
        
        # 创建损失函数
        self.criterion = PatternLoss(config)
        
        # 创建优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config['train']['learning_rate']
        )
        
        # 创建数据加载器
        self.train_loader = create_dataloader(config, mode='train')
        
        # 创建输出目录
        self.output_dir = Path(config['output']['save_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = Path(config['output']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, epoch: int) -> Dict:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        with tqdm(self.train_loader, desc=f'Epoch {epoch}') as pbar:
            for batch_idx, batch in enumerate(pbar):
                # 将数据移到设备上
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                
                # 训练步骤
                self.optimizer.zero_grad()
                losses, outputs = self.model.train_step(batch, self.criterion)
                
                # 反向传播
                total_loss = losses['total']
                total_loss.backward()
                self.optimizer.step()
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f'{total_loss.item():.4f}',
                    'l1': f'{losses["l1"].item():.4f}',
                    'perceptual': f'{losses["perceptual"].item():.4f}'
                })
                
                # 保存中间结果
                if batch_idx % 10 == 0:
                    save_image(
                        outputs['outputs'][0],
                        self.output_dir / f'epoch_{epoch}_batch_{batch_idx}.png'
                    )
        
        return {k: v.item() for k, v in losses.items()}
    
    def save_checkpoint(self, epoch: int, losses: Dict):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': losses
        }
        
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f'Saved checkpoint to {checkpoint_path}')
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']
    
    def train(self, resume_from: Optional[str] = None):
        """训练模型"""
        start_epoch = 0
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from)
            self.logger.info(f'Resumed from epoch {start_epoch}')
        
        for epoch in range(start_epoch, self.config['train']['epochs']):
            self.logger.info(f'Starting epoch {epoch}')
            
            # 训练一个epoch
            losses = self.train_epoch(epoch)
            
            # 记录损失
            self.logger.info(
                f'Epoch {epoch} losses: ' +
                ', '.join(f'{k}: {v:.4f}' for k, v in losses.items())
            )
            
            # 保存检查点
            if (epoch + 1) % self.config['train']['save_interval'] == 0:
                self.save_checkpoint(epoch, losses) 