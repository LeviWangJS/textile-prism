from typing import Dict, Any, Optional
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import json

class ParameterVisualizer:
    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.history = defaultdict(list)
        
    def log_parameters(self, params: Dict[str, Any], epoch: int):
        """记录参数变化"""
        for name, value in params.items():
            if value is not None:
                self.history[name].append((epoch, value))
                
    def generate_adjustment_report(self):
        """生成参数调整可视化报告"""
        plt.figure(figsize=(12, 8))
        
        # 创建子图
        n_params = len(self.history)
        fig, axes = plt.subplots(n_params, 1, figsize=(12, 4*n_params))
        if n_params == 1:
            axes = [axes]
            
        for ax, (param_name, history) in zip(axes, self.history.items()):
            epochs, values = zip(*history)
            ax.plot(epochs, values, marker='o')
            ax.set_title(f'{param_name} Adjustment History')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(param_name)
            ax.grid(True)
            
        plt.tight_layout()
        plt.savefig(self.save_dir / 'parameter_adjustments.png')
        plt.close()
        
        # 生成统计报告
        stats = {}
        for param_name, history in self.history.items():
            _, values = zip(*history)
            stats[param_name] = {
                'initial': values[0],
                'final': values[-1],
                'min': min(values),
                'max': max(values),
                'mean': sum(values) / len(values),
                'changes': len(set(values)) - 1
            }
            
        # 保存统计报告
        with open(self.save_dir / 'adjustment_stats.json', 'w') as f:
            json.dump(stats, f, indent=4) 

def visualize_augmentations(dataset, index: int, n_samples: int = 5, save_path: Optional[str] = None):
    """可视化数据增强效果"""
    # 获取原始图像
    original_data = dataset[index]
    original_image = original_data['input']
    
    # 创建多个增强版本
    augmented_images = []
    for _ in range(n_samples):
        data = dataset[index]
        augmented_images.append(data['input'])
    
    def normalize_for_display(img):
        """将图像归一化到[0,1]范围"""
        img = img.clone()
        img = (img + 1) / 2  # 从[-1,1]转换到[0,1]
        return img.clamp(0, 1)
    
    # 显示结果
    fig, axes = plt.subplots(1, n_samples + 1, figsize=(3*(n_samples + 1), 3))
    
    # 显示原图
    norm_original = normalize_for_display(original_image)
    if original_image.shape[0] == 1:  # 如果是灰度图
        axes[0].imshow(norm_original[0].cpu().numpy(), cmap='gray')
    else:  # RGB图
        axes[0].imshow(norm_original.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # 显示增强后的图像
    for i, aug_img in enumerate(augmented_images, 1):
        norm_aug = normalize_for_display(aug_img)
        if aug_img.shape[0] == 1:  # 灰度图
            axes[i].imshow(norm_aug[0].cpu().numpy(), cmap='gray')
        else:  # RGB图
            axes[i].imshow(norm_aug.permute(1, 2, 0).cpu().numpy())
        axes[i].set_title(f'Augmented {i}')
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show() 