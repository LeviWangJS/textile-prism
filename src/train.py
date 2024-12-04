import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import yaml
import logging
from tqdm import tqdm
import time
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.pattern_transformer import PatternTransformer
from models.losses import CompositeLoss
from data.dataset import CarpetDataset
from utils.training_monitor import TrainingMonitor

def plot_loss_curves(train_losses, val_losses, save_path):
    """绘制训练和验证损失曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_metrics(metrics, save_path):
    """绘制评估指标曲线"""
    plt.figure(figsize=(12, 8))
    for metric_name, values in metrics.items():
        plt.plot(values, label=metric_name)
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Metrics')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

class Trainer:
    def __init__(self, config):
        """初始化训练器"""
        self.config = config
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        
        # 创建必要的目录
        self.checkpoint_dir = Path('checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化数据加载器
        self.train_loader, self.val_loader = self._init_data_loaders()
        
        # 初始化模型
        self.model = self._init_model()
        
        # 初始化损失函数和优化器
        self.criterion = self._init_criterion()
        self.optimizer = self._init_optimizer()
        
        # 初始化学习率调度器
        self.scheduler = self._init_scheduler()
        
        # 初始化TensorBoard writer
        self.writer = SummaryWriter(log_dir='runs/textile_prism')
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_model_path = self.checkpoint_dir / 'best_model.pth'
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # 创建保存目录
        self.checkpoint_dir = Path(config['training']['checkpoint']['save_dir'])
        self.viz_dir = Path(config['visualization']['save_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # 记录训练历史
        self.train_losses = []
        self.val_losses = []
        self.metrics = {}
    
    def _init_data_loaders(self):
        """初始化数据加载器"""
        # 创建数据加载器
        train_loader = DataLoader(
            CarpetDataset(self.config, mode='train'),
            batch_size=self.config['train']['batch_size'],
            shuffle=True,
            num_workers=self.config['system']['num_workers']
        )
        
        val_loader = DataLoader(
            CarpetDataset(self.config, mode='val'),
            batch_size=self.config['train']['batch_size'],
            shuffle=False,
            num_workers=self.config['system']['num_workers']
        )
        
        return train_loader, val_loader
    
    def _init_model(self):
        """初始化模型"""
        # 创建模型
        return PatternTransformer(self.config).to(self.device)
    
    def _init_criterion(self):
        """初始化损失函数"""
        # 创建损失函数
        return CompositeLoss(device=self.device).to(self.device)
    
    def _init_optimizer(self):
        """初始化优化器"""
        # 创建优化器
        return optim.Adam(
            self.model.parameters(),
            lr=self.config['train']['learning_rate']
        )
    
    def _init_scheduler(self):
        """初始化学习率调度器"""
        # 创建学习率调度器
        return optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['train']['scheduler']['params']['T_max'],
            eta_min=self.config['train']['scheduler']['params']['eta_min']
        )
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        # 创建进度条
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}/{self.config["train"]["epochs"]}')
        
        # 创建保存图像的目录
        save_dir = Path('outputs/images') / f'epoch_{self.current_epoch}'
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for batch_idx, batch in enumerate(pbar):
            # 获取数据
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # 计算损失
            loss_dict = self.criterion(outputs, targets, train_discriminator=True)
            loss = loss_dict['total']
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 更新参数
            self.optimizer.step()
            
            # 更新进度条
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'pattern': f'{loss_dict["pattern_total"]:.4f}',
                'l1': f'{loss_dict["pattern_l1"]:.4f}',
                'consist': f'{loss_dict["consistency"]:.4f}',
                'adv': f'{loss_dict["adversarial"]:.4f}',
                'd_loss': f'{loss_dict.get("discriminator", 0):.4f}'
            })
            
            # 记录到TensorBoard
            if self.writer is not None:
                step = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Loss/train/total', loss.item(), step)
                self.writer.add_scalar('Loss/train/pattern_total', loss_dict['pattern_total'], step)
                self.writer.add_scalar('Loss/train/pattern_l1', loss_dict['pattern_l1'], step)
                self.writer.add_scalar('Loss/train/pattern_perceptual', loss_dict['pattern_perceptual'], step)
                self.writer.add_scalar('Loss/train/pattern_style', loss_dict['pattern_style'], step)
                self.writer.add_scalar('Loss/train/pattern_ssim', loss_dict['pattern_ssim'], step)
                self.writer.add_scalar('Loss/train/consistency', loss_dict['consistency'], step)
                self.writer.add_scalar('Loss/train/adversarial', loss_dict['adversarial'], step)
                if 'discriminator' in loss_dict:
                    self.writer.add_scalar('Loss/train/discriminator', loss_dict['discriminator'], step)
                
                # 每隔一定步数保存图像
                if batch_idx % self.config['train']['monitor']['log_interval'] == 0:
                    # 将图像从[-1,1]转换到[0,1]
                    def denormalize(x):
                        return (x + 1) / 2
                    
                    # 保存��入图像
                    save_image(
                        denormalize(inputs),
                        save_dir / f'batch_{batch_idx}_input.png',
                        normalize=False
                    )
                    
                    # 保存输出图像
                    save_image(
                        denormalize(outputs),
                        save_dir / f'batch_{batch_idx}_output.png',
                        normalize=False
                    )
                    
                    # 保存目标图像
                    save_image(
                        denormalize(targets),
                        save_dir / f'batch_{batch_idx}_target.png',
                        normalize=False
                    )
                    
                    # 保存对比图
                    comparison = torch.cat([
                        denormalize(inputs),
                        denormalize(outputs),
                        denormalize(targets)
                    ], dim=0)
                    save_image(
                        comparison,
                        save_dir / f'batch_{batch_idx}_comparison.png',
                        nrow=inputs.size(0),
                        normalize=False
                    )
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # 获取数据
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                
                # 前向传播
                outputs = self.model(inputs)
                
                # 计算损失（验证时不训练判别器）
                loss_dict = self.criterion(outputs, targets, train_discriminator=False)
                loss = loss_dict['total']
                
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def save_checkpoint(self, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        
        # 保存最新的检查点
        latest_checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, str(latest_checkpoint_path))
        
        # 如果是最佳模型，保存一个副本
        if is_best:
            best_model_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, str(best_model_path))
            print(f"保存最佳模型到 {best_model_path}")
    
    def save_test_visualization(self):
        """保存测试数据可视化结果"""
        self.model.eval()
        viz_dir = self.viz_dir / f'best_model_epoch_{self.current_epoch}'
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        with torch.no_grad():
            # 获取一个随机测试批次
            test_batch = next(iter(self.test_loader))
            inputs = test_batch['input'].to(self.device)
            targets = test_batch['target'].to(self.device)
            
            # 生成预测结果
            outputs = self.model(inputs)
            
            # 保存可视化结果
            for i in range(min(4, len(inputs))):  # 最多保存4个样本
                # 保存输入图像
                save_image(
                    inputs[i],
                    viz_dir / f'sample_{i}_input.png'
                )
                # 保存预测输出
                save_image(
                    outputs[i],
                    viz_dir / f'sample_{i}_pred.png'
                )
                # 保存真实目标
                save_image(
                    targets[i],
                    viz_dir / f'sample_{i}_target.png'
                )
            
            self.logger.info(f"已保存最佳模型(epoch {self.current_epoch})的测试数据可视化到: {viz_dir}")
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path)
        
        self.current_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.metrics = checkpoint['metrics']
        
        self.logger.info(f"加载检查点: epoch {self.current_epoch}")
    
    def save_visualizations(self, inputs, outputs, targets, batch_idx):
        """保存可视化结果"""
        viz_dir = self.viz_dir / f'epoch_{self.current_epoch}'
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存第一个样本
        save_image(
            inputs[0],
            viz_dir / f'batch_{batch_idx}_input.png'
        )
        save_image(
            outputs[0],
            viz_dir / f'batch_{batch_idx}_output.png'
        )
        save_image(
            targets[0],
            viz_dir / f'batch_{batch_idx}_target.png'
        )
    
    def train(self):
        """训练模型"""
        print(f"开始训练...")
        
        for epoch in range(self.config['train']['epochs']):
            self.current_epoch = epoch
            
            # 训练一个epoch
            train_loss = self.train_epoch()
            
            # 验证
            if (epoch + 1) % self.config['train']['val_interval'] == 0:
                val_loss = self.validate()
                
                # 检查是否是最佳模型
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
            else:
                is_best = False
            
            # 保存检查点
            self.save_checkpoint(is_best)
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录学习率
            if self.writer is not None:
                self.writer.add_scalar(
                    'Learning_Rate',
                    self.optimizer.param_groups[0]['lr'],
                    self.current_epoch
                )
        
        print("训练完成！")

def main():
    # 加载配置
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置随机种子
    torch.manual_seed(config['system']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['system']['seed'])
    
    # 创建训练器
    trainer = Trainer(config)
    
    # 开始训练
    trainer.train()

if __name__ == '__main__':
    main() 