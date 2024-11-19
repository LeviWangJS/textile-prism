import torch
import matplotlib.pyplot as plt
from pathlib import Path

class FeatureVisualizer:
    def __init__(self, save_dir: str = 'visualizations'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def visualize_features(self, 
                          clip_features: torch.Tensor,
                          original_features: torch.Tensor,
                          fused_features: torch.Tensor,
                          save_name: str):
        """可视化不同特征的分布"""
        plt.figure(figsize=(15, 5))
        
        # CLIP特征分布
        plt.subplot(131)
        self._plot_feature_distribution(clip_features, "CLIP Features")
        
        # 原始特征分布
        plt.subplot(132)
        self._plot_feature_distribution(original_features, "Original Features")
        
        # 融合特征分布
        plt.subplot(133)
        self._plot_feature_distribution(fused_features, "Fused Features")
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f"{save_name}.png")
        plt.close()
        
    def _plot_feature_distribution(self, features: torch.Tensor, title: str):
        features_np = features.detach().cpu().numpy()
        plt.hist(features_np.flatten(), bins=50)
        plt.title(title)
        plt.xlabel("Feature Value")
        plt.ylabel("Frequency") 