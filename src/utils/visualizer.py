import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class Visualizer:
    """训练可视化器"""
    def __init__(self, config):
        self.config = config
        self.save_dir = Path(config['visualization']['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.train_dir = self.save_dir / 'train'
        self.val_dir = self.save_dir / 'val'
        self.test_dir = self.save_dir / 'test'
        
        for dir_path in [self.train_dir, self.val_dir, self.test_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def save_image(self, tensor, path):
        """保存图像张量为PNG文件"""
        # 确保输入是CPU张量
        if tensor.is_cuda:
            tensor = tensor.cpu()
        
        # 转换为numpy数组
        img = tensor.detach().numpy()
        
        # 调整维度顺序从[C,H,W]到[H,W,C]
        if img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))
        
        # 归一化到[0,1]
        img = (img + 1) / 2
        img = np.clip(img, 0, 1)
        
        # 保存图像
        plt.imsave(path, img)
    
    def plot_loss_curves(self, train_losses, val_losses=None, epoch=None):
        """绘制损失曲线"""
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label="Train Loss")
        if val_losses is not None:
            plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        
        # 保存图像
        save_path = self.save_dir / f'loss_curves_epoch_{epoch}.png' if epoch is not None else self.save_dir / 'loss_curves.png'
        plt.savefig(save_path)
        plt.close()
    
    def plot_metrics(self, metrics_dict, epoch=None):
        """绘制评估指标"""
        num_metrics = len(metrics_dict)
        fig, axes = plt.subplots(1, num_metrics, figsize=(5*num_metrics, 5))
        
        if num_metrics == 1:
            axes = [axes]
        
        for ax, (metric_name, values) in zip(axes, metrics_dict.items()):
            ax.plot(values)
            ax.set_title(metric_name)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Value")
            ax.grid(True)
        
        plt.tight_layout()
        
        # 保存图像
        save_path = self.save_dir / f'metrics_epoch_{epoch}.png' if epoch is not None else self.save_dir / 'metrics.png'
        plt.savefig(save_path)
        plt.close()
    
    def save_batch_results(self, inputs, outputs, targets, epoch, batch_idx, mode='train'):
        """保存一个批次的结果"""
        save_dir = getattr(self, f'{mode}_dir') / f'epoch_{epoch}'
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(len(inputs)):
            # 保存输入图像
            self.save_image(
                inputs[i],
                save_dir / f'batch_{batch_idx}_sample_{i}_input.png'
            )
            
            # 保存输出图像
            self.save_image(
                outputs[i],
                save_dir / f'batch_{batch_idx}_sample_{i}_output.png'
            )
            
            # 保存目标图像
            self.save_image(
                targets[i],
                save_dir / f'batch_{batch_idx}_sample_{i}_target.png'
            )
    
    def save_attention_maps(self, attention_maps, epoch, batch_idx, mode='train'):
        """保存注意力图"""
        if not self.config['visualization'].get('save_attention_maps', False):
            return
        
        save_dir = getattr(self, f'{mode}_dir') / f'epoch_{epoch}' / 'attention_maps'
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for i, attn_map in enumerate(attention_maps):
            plt.figure(figsize=(10, 10))
            plt.imshow(attn_map.cpu().numpy(), cmap='viridis')
            plt.colorbar()
            plt.title(f'Attention Map {i+1}')
            plt.savefig(save_dir / f'batch_{batch_idx}_attention_{i+1}.png')
            plt.close()