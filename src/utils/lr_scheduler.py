import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import lr_scheduler
from typing import Optional, Union, Dict

class WarmupLRScheduler(_LRScheduler):
    """带预热的学习率调度器基类"""
    def __init__(self, optimizer, warmup_epochs=5, warmup_lr_init=1e-6, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.warmup_lr_init = warmup_lr_init
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """重写父类方法"""
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / self.warmup_epochs
            return [self.warmup_lr_init + (base_lr - self.warmup_lr_init) * alpha
                    for base_lr in self.base_lrs]
        return [self.min_lr for _ in self.base_lrs]

class CosineAnnealingWarmup(WarmupLRScheduler):
    """带预热的余弦退火调度器"""
    def __init__(self, optimizer, total_epochs, warmup_epochs=5, warmup_lr_init=1e-6, min_lr=1e-6):
        self.total_epochs = total_epochs
        super().__init__(optimizer, warmup_epochs, warmup_lr_init, min_lr)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return self.get_warmup_lr(self.last_epoch)
        
        # 余弦退火计算
        progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
        return [self.min_lr + (base_lr - self.min_lr) * 
                (1 + math.cos(math.pi * progress)) / 2
                for base_lr in self.base_lrs]

class ReduceLROnPlateauWithWarmup:
    """带预热的ReduceLROnPlateau"""
    def __init__(self, optimizer, warmup_epochs=5, warmup_lr_init=1e-6,
                 factor=0.5, patience=5, min_lr=1e-6, mode='min'):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_lr_init = warmup_lr_init
        self.min_lr = min_lr
        self.current_epoch = 0
        
        # 保存初始学习率
        self.initial_lr = optimizer.param_groups[0]['lr']
        
        # 设置warmup起始学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = warmup_lr_init
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            min_lr=min_lr
        )
    
    def step(self, metrics=None):
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            # warmup阶段
            progress = self.current_epoch / self.warmup_epochs
            lr = self.warmup_lr_init + (self.initial_lr - self.warmup_lr_init) * progress
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # 正常调度阶段
            self.scheduler.step(metrics)
    
    def get_last_lr(self):
        """返回当前学习率"""
        return [group['lr'] for group in self.optimizer.param_groups] 

class LRSchedulerWrapper:
    def __init__(self, optimizer, config: Dict):
        self.optimizer = optimizer
        self.scheduler_config = config['train']['scheduler']
        self.scheduler_type = self.scheduler_config['type']
        
        # 确保数值参数为float类型
        params = self.scheduler_config['params']
        if self.scheduler_type == 'step':
            self.scheduler = lr_scheduler.StepLR(
                optimizer,
                step_size=int(params['step_size']),
                gamma=float(params['gamma'])
            )
        elif self.scheduler_type == 'cosine':
            self.scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(params['T_max']),
                eta_min=float(params['eta_min'])
            )
        elif self.scheduler_type == 'reduce_plateau':
            self.scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=params['mode'],
                factor=float(params['factor']),
                patience=int(params['patience']),
                min_lr=float(params['min_lr'])
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {self.scheduler_type}")
    
    def step(self, metrics: Optional[float] = None):
        """统一的step接口"""
        if self.scheduler_type == 'reduce_plateau':
            if metrics is None:
                raise ValueError("ReduceLROnPlateau requires metrics for step()")
            self.scheduler.step(metrics)
        else:
            self.scheduler.step()
    
    def get_last_lr(self) -> float:
        """获取当前学习率"""
        if self.scheduler_type == 'reduce_plateau':
            return self.optimizer.param_groups[0]['lr']
        return self.scheduler.get_last_lr()[0] 