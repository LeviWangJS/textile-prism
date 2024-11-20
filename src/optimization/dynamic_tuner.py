from typing import Dict, Optional, Any
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import psutil
from collections import defaultdict
import time

class DynamicTuner:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.last_adjustment_epoch = -1
        self.adjustments_this_epoch = 0
        
    def log_adjustment(self, param_name: str, old_value: float, new_value: float, metrics: Dict):
        """记录参数调整信息"""
        self.logger.info(
            f"Adjusting {param_name}: {old_value:.6f} -> {new_value:.6f}\n"
            f"Current metrics: train_loss={metrics.get('train_loss', 'N/A'):.4f}, "
            f"val_loss={metrics.get('val_loss', 'N/A'):.4f}"
        )
        
        # 如果使用wandb，也记录到wandb
        if self.config.get('logging', {}).get('use_wandb', False):
            import wandb
            wandb.log({
                f'{param_name}_adjustment': new_value,
                'adjustment_epoch': metrics.get('epoch', 0)
            })
    
    def update_learning_rate(self, optimizer, metrics: Dict, epoch: int) -> float:
        """动态调整学习率"""
        current_lr = optimizer.param_groups[0]['lr']
        config = self.config['train']['dynamic_tuning']['lr_adjust']
        
        # 确保配置值为float类型
        min_lr = float(config['min_lr'])
        max_lr = float(config['max_lr'])
        adjust_threshold = float(config['adjust_threshold'])
        
        # 检查冷却期
        if epoch - self.last_adjustment_epoch < config.get('adjustment_cooldown', 3):
            return current_lr
            
        # 检查当前epoch的调整次数
        if epoch != self.last_adjustment_epoch:
            self.adjustments_this_epoch = 0
        if self.adjustments_this_epoch >= config.get('max_adjustments_per_epoch', 2):
            return current_lr
        
        # 获取损失指标
        val_loss = metrics.get('val_loss', float('inf'))
        train_loss = metrics.get('train_loss', float('inf'))
        loss_diff = abs(val_loss - train_loss)
        
        # 判断是否需要调整学习率
        if loss_diff > adjust_threshold:
            if val_loss > train_loss:
                new_lr = max(current_lr * 0.5, min_lr)
            else:
                new_lr = min(current_lr * 1.5, max_lr)
                
            # 更新调整记录
            if new_lr != current_lr:
                self.last_adjustment_epoch = epoch
                self.adjustments_this_epoch += 1
                self.log_adjustment('learning_rate', current_lr, new_lr, metrics)
        else:
            new_lr = current_lr
            
        return new_lr
        
    def update_batch_size(self, 
                         loader: DataLoader,
                         metrics: Dict[str, float]) -> int:
        """动态调整批次大小"""
        current_batch_size = loader.batch_size
        memory_usage = psutil.virtual_memory().percent
        
        # 根据内存使用和性能指标调整批次大小
        if memory_usage > 90:  # 内存使用过高
            new_batch_size = max(4, current_batch_size // 2)
        elif memory_usage < 70 and metrics.get('gpu_util', 0) < 50:
            new_batch_size = min(256, current_batch_size * 2)
        else:
            new_batch_size = current_batch_size
            
        if new_batch_size != current_batch_size:
            self.logger.info(f"Adjusting batch size: {current_batch_size} -> {new_batch_size}")
            
        return new_batch_size
        
    def update_model_params(self, 
                          model: nn.Module,
                          metrics: Dict[str, float]) -> None:
        """动态调整模型参数"""
        # 根据验证指标调整dropout率
        if hasattr(model, 'dropout'):
            current_dropout = model.dropout.p
            val_loss = metrics.get('val_loss', float('inf'))
            
            if len(self.history['val_loss']) > 1:
                if val_loss > self.history['val_loss'][-2]:  # 过拟合趋势
                    new_dropout = min(0.5, current_dropout + 0.1)
                else:  # 欠拟合趋势
                    new_dropout = max(0.1, current_dropout - 0.1)
                    
                if new_dropout != current_dropout:
                    model.dropout.p = new_dropout
                    self.logger.info(f"Adjusting dropout: {current_dropout} -> {new_dropout}")