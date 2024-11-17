import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import wandb
from PIL import Image
import io
import imageio
from torchvision.utils import save_image as save_torch_image

def save_image(tensor, filename):
    """保存图像张量到文件"""
    save_torch_image(tensor, filename)

class TrainingVisualizer:
    def __init__(self, save_dir='runs'):
        """
        初始化可视化工具
        
        Args:
            save_dir: 保存可视化结果的目录
        """
        self.save_dir = os.path.join(save_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 初始化图像缓存
        self.image_buffer = []
        self.temp_image_paths = []
        
        # 使用更现代的样式
        plt.style.use('ggplot')
        
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 初始化历史记录
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'epoch': []
        }
    
    def update_history(self, epoch, train_loss, val_loss, lr):
        """更新训练历史"""
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['learning_rates'].append(lr)
    
    def plot_losses(self, save=True):
        """绘制损失曲线
        
        Args:
            save (bool): 是否保存图像。如果为False，则使用非阻塞式显示
        """
        plt.figure(figsize=(10, 5))
        
        # 绘制训练损失
        plt.subplot(1, 2, 1)
        plt.plot(self.history['epoch'], self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['epoch'], self.history['val_loss'], label='Val Loss')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 绘制学习率
        plt.subplot(1, 2, 2)
        plt.plot(self.history['epoch'], self.history['learning_rates'], label='Learning Rate')
        plt.title('Learning Rate Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        if save:
            # 确保保存目录存在
            os.makedirs(self.save_dir, exist_ok=True)
            save_path = os.path.join(self.save_dir, f'loss_plot_epoch_{len(self.history["epoch"])}.png')
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show(block=False)
            plt.pause(0.1)
            plt.close()
    
    def plot_lr(self, save=True):
        """绘制学习率变化曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['epoch'], self.history['learning_rates'], 
                marker='o')
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True)
        
        if save:
            plt.savefig(os.path.join(self.save_dir, 'lr_curve.png'))
            plt.close()
        else:
            plt.show()
    
    def plot_prediction(self, input_img, output_img, target_img, epoch, save=True):
        """绘制预测结果对比"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 转换张量到numpy并调整范围到[0,1]
        def tensor_to_img(tensor):
            if isinstance(tensor, torch.Tensor):
                img = tensor.cpu().detach().numpy()
                if img.shape[0] == 3:  # CHW to HWC
                    img = np.transpose(img, (1, 2, 0))
                img = (img + 1) / 2  # [-1,1] to [0,1]
                return np.clip(img, 0, 1)
            return tensor
        
        # 绘制三张图
        axes[0].imshow(tensor_to_img(input_img))
        axes[0].set_title('Input')
        axes[1].imshow(tensor_to_img(output_img))
        axes[1].set_title('Prediction')
        axes[2].imshow(tensor_to_img(target_img))
        axes[2].set_title('Target')
        
        for ax in axes:
            ax.axis('off')
        
        plt.suptitle(f'Epoch {epoch}')
        
        if save:
            plt.savefig(os.path.join(self.save_dir, f'pred_epoch_{epoch}.png'))
            plt.close()
        else:
            plt.show()
    
    def plot_attention_maps(self, attention_maps, epoch, save=True):
        """绘制注意力图"""
        num_maps = len(attention_maps)
        fig, axes = plt.subplots(1, num_maps, figsize=(5*num_maps, 5))
        if num_maps == 1:
            axes = [axes]
        
        for idx, (name, attn) in enumerate(attention_maps.items()):
            if isinstance(attn, torch.Tensor):
                attn = attn.cpu().detach().numpy()
            sns.heatmap(attn, ax=axes[idx], cmap='viridis')
            axes[idx].set_title(name)
        
        plt.suptitle(f'Attention Maps - Epoch {epoch}')
        
        if save:
            plt.savefig(os.path.join(self.save_dir, f'attention_epoch_{epoch}.png'))
            plt.close()
        else:
            plt.show()
    
    def create_progress_bar(self, total, desc='Training'):
        """创建进度条"""
        return tqdm(total=total, desc=desc, ncols=100)
    
    def save_training_image(self, image_tensor, epoch, batch_idx):
        """保存训练过程中的图像
        
        Args:
            image_tensor: 图像张量 (C,H,W)
            epoch: 当前epoch
            batch_idx: 当前batch索引
        """
        # 确保图像是numpy数组
        if torch.is_tensor(image_tensor):
            image = image_tensor.detach().cpu().numpy()
        else:
            image = np.array(image_tensor)
            
        # 转换为PIL图像
        if image.shape[0] in [1, 3, 4]:  # 如果通道在前面
            image = np.transpose(image, (1, 2, 0))
        
        if image.shape[-1] == 1:  # 如果是单通道
            image = np.squeeze(image)
            
        # 归一化到0-255
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
            
        # 保存图像
        save_path = os.path.join(
            self.save_dir, 
            f'train_progress_e{epoch:03d}_b{batch_idx:04d}.png'
        )
        Image.fromarray(image).save(save_path)
        self.temp_image_paths.append(save_path)
    
    def create_training_gif(self, output_path=None, fps=10, quality=95, cleanup=True):
        """创建训练过程的GIF动画
        
        Args:
            output_path: GIF输出路径，默认为save_dir下的training_progress.gif
            fps: 帧率，默认10
            quality: 图像质量(0-100)，默认95
            cleanup: 是否清理临时图像文件，默认True
        
        Returns:
            str: 生成的GIF文件路径
        """
        if not self.temp_image_paths:
            print("没有可用的训练图像")
            return None
            
        if output_path is None:
            output_path = os.path.join(self.save_dir, 'training_progress.gif')
            
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 读取所有图像
        images = []
        for img_path in tqdm(sorted(self.temp_image_paths), desc="Creating GIF"):
            try:
                img = Image.open(img_path)
                images.append(img)
            except Exception as e:
                print(f"Warning: Failed to load image {img_path}: {e}")
                
        if not images:
            print("No valid images found")
            return None
            
        # 创建GIF
        try:
            # 计算持续时间
            duration = 1000 / fps  # 转换为毫秒
            
            # 保存GIF
            images[0].save(
                output_path,
                save_all=True,
                append_images=images[1:],
                duration=duration,
                loop=0,
                quality=quality,
                optimize=True
            )
            
            print(f"GIF created successfully: {output_path}")
            
            # 清理临时文件
            if cleanup:
                for img_path in self.temp_image_paths:
                    try:
                        os.remove(img_path)
                    except Exception as e:
                        print(f"Warning: Failed to remove temp file {img_path}: {e}")
                self.temp_image_paths = []
                
            return output_path
            
        except Exception as e:
            print(f"Error creating GIF: {e}")
            return None
    
    def cleanup(self):
        """清理所有临时文件"""
        for img_path in self.temp_image_paths:
            try:
                if os.path.exists(img_path):
                    os.remove(img_path)
            except Exception as e:
                print(f"Warning: Failed to remove temp file {img_path}: {e}")
        self.temp_image_paths = []