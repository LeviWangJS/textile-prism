import os
import time
import logging
from pathlib import Path
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import yaml
from datetime import datetime
import json
from collections import defaultdict
import cv2

try:
    import wandb
except ImportError:
    wandb = None

class MetricTracker:
    """指标跟踪器"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.metrics = defaultdict(list)
        self.current_epoch = 0
        self.best_values = {}
        self.best_epochs = {}
    
    def update(self, metrics, epoch=None):
        if epoch is not None:
            self.current_epoch = epoch
        
        for name, value in metrics.items():
            self.metrics[name].append(float(value))
            
            # 更新最佳值
            if name not in self.best_values or value < self.best_values[name]:
                self.best_values[name] = value
                self.best_epochs[name] = self.current_epoch
    
    def get_latest(self, name):
        return self.metrics[name][-1] if self.metrics[name] else None
    
    def get_best(self, name):
        return self.best_values.get(name, None)
    
    def get_history(self, name):
        return self.metrics[name]

class TrainingMonitor:
    """训练监控器"""
    
    def __init__(self, config):
        """初始化训练监控器"""
        self.config = config
        self.log_dir = Path(config['output']['log_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.setup_logging()
        
        # 初始化指标跟踪器
        self.train_metrics = MetricTracker()
        self.val_metrics = MetricTracker()
        
        # 初始化TensorBoard
        self.writer = SummaryWriter(self.log_dir / 'tensorboard')
        
        # 性能监控
        self.batch_times = []
        self.data_times = []
        self.start_time = time.time()
    
    def setup_logging(self):
        """设置日志记录"""
        log_file = self.log_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def on_epoch_start(self, epoch):
        """epoch开始时的处理"""
        self.epoch_start_time = time.time()
        self.batch_times = []
        self.data_times = []
        self.logger.info(f"\nEpoch {epoch} 开始")
    
    def on_epoch_end(self, epoch, train_metrics, val_metrics=None):
        """epoch结束时的处理"""
        epoch_time = time.time() - self.epoch_start_time
        
        # 更新指标
        self.train_metrics.update(train_metrics, epoch)
        if val_metrics:
            self.val_metrics.update(val_metrics, epoch)
        
        # 记录到TensorBoard
        if self.writer:
            for name, value in train_metrics.items():
                self.writer.add_scalar(f'train/{name}', value, epoch)
            if val_metrics:
                for name, value in val_metrics.items():
                    self.writer.add_scalar(f'val/{name}', value, epoch)
        
        # 记录到W&B
        if wandb is not None and wandb.run is not None:
            wandb.log({
                **{f'train/{k}': v for k, v in train_metrics.items()},
                **(({f'val/{k}': v for k, v in val_metrics.items()}) if val_metrics else {}),
                'epoch': epoch
            })
        
        # 计算和记录性能指标
        avg_batch_time = np.mean(self.batch_times) if self.batch_times else 0
        avg_data_time = np.mean(self.data_times) if self.data_times else 0
        
        # 输出日志
        self.logger.info(f"Epoch {epoch} 完成 (用时: {epoch_time:.2f}s)")
        self.logger.info(f"训练指标: {train_metrics}")
        if val_metrics:
            self.logger.info(f"验证指标: {val_metrics}")
        self.logger.info(f"平均批次时间: {avg_batch_time:.4f}s (数据加载: {avg_data_time:.4f}s)")
    
    def on_batch_end(self, batch_time, data_time):
        """批次结束时的处理"""
        self.batch_times.append(batch_time)
        self.data_times.append(data_time)
    
    def save_visualization(self, epoch, images, output_dir):
        """保存可视化结果"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建网格图像
        grid = make_grid(images, nrow=2, normalize=True)
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        
        # 保存图像
        plt.figure(figsize=(12, 8))
        plt.imshow(grid_np)
        plt.axis('off')
        plt.savefig(output_dir / f'epoch_{epoch:03d}.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # 添加到TensorBoard
        if self.writer:
            self.writer.add_image('samples', grid, epoch)
        
        # 添加到W&B
        if wandb is not None and wandb.run is not None:
            wandb.log({
                'samples': wandb.Image(grid_np),
                'epoch': epoch
            })
    
    def save_checkpoint(self, state, is_best, filename):
        """保存检查点"""
        checkpoint_dir = Path(self.config['output']['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存当前检查点
        torch.save(state, checkpoint_dir / filename)
        
        # 如果是最佳模型，创建一个副本
        if is_best:
            best_filename = filename.replace('.pth', '_best.pth')
            torch.save(state, checkpoint_dir / best_filename)
            self.logger.info(f"保存最佳模型检查点: {best_filename}")
        
        # 清理旧的检查点
        keep_last_k = 3  # 保留最后k个检查点
        checkpoints = sorted(checkpoint_dir.glob('epoch_*.pth'))
        if len(checkpoints) > keep_last_k:
            for checkpoint in checkpoints[:-keep_last_k]:
                checkpoint.unlink()
    
    def should_stop_early(self):
        """检查是否应该早停"""
        if not self.config['train']['early_stopping']['enabled']:
            return False
        
        patience = self.config['train']['early_stopping']['patience']
        min_delta = self.config['train']['early_stopping']['min_delta']
        
        # 获取验证损失历史
        val_losses = self.val_metrics.get_history('loss')
        if len(val_losses) <= patience:
            return False
        
        # 检查最近patience轮的验证损失是否有改善
        best_loss = min(val_losses[:-patience])
        current_loss = val_losses[-1]
        
        return current_loss > best_loss - min_delta
    
    def get_learning_rate(self, epoch):
        """获取当前学习率"""
        initial_lr = self.config['train']['learning_rate']
        total_epochs = self.config['train']['epochs']
        
        # Cosine衰减
        if self.config['train']['scheduler']['type'] == 'cosine':
            progress = epoch / total_epochs
            min_lr = self.config['train']['scheduler']['params']['min_lr']
            return min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(np.pi * progress))
        
        return initial_lr
    
    def close(self):
        """关闭监控器"""
        if self.writer:
            self.writer.close()
        
        if wandb is not None and wandb.run is not None:
            wandb.finish()
        
        # 保存训练历史
        history = {
            'train_metrics': self.train_metrics.metrics,
            'val_metrics': self.val_metrics.metrics,
            'best_values': {
                'train': self.train_metrics.best_values,
                'val': self.val_metrics.best_values
            },
            'best_epochs': {
                'train': self.train_metrics.best_epochs,
                'val': self.val_metrics.best_epochs
            },
            'total_time': time.time() - self.start_time
        }
        
        with open(self.log_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=4) 