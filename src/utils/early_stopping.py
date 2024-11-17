import logging
import numpy as np

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=False):
        """
        早停机制
        
        Args:
            patience (int): 容忍的epoch数量
            min_delta (float): 最小改善量
            verbose (bool): 是否打印信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.logger = logging.getLogger(__name__)

    def __call__(self, val_loss):
        """
        检查是否应该早停
        
        Args:
            val_loss (float): 当前的验证损失
            
        Returns:
            bool: 是否应该停止训练
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        # 检查是否有改善
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                self.logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f})')
            self.val_loss_min = val_loss
        else:
            self.counter += 1
            if self.verbose:
                self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False
    
    def restore_best_weights(self, model):
        """恢复最佳权重"""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            if self.verbose:
                print('Restoring model weights from the end of the best epoch') 