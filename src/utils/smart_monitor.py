from utils.openai_helper import OpenAIHelper
import torch
from typing import Dict, Optional
import logging
import time
from PIL import Image
import numpy as np
from pathlib import Path

class SmartTrainingMonitor:
    def __init__(self, config: Dict, openai_helper: OpenAIHelper):
        self.config = config
        self.openai_helper = openai_helper
        self.logger = logging.getLogger(__name__)
        self.analysis_frequency = config['training'].get('analysis_frequency', 200)
        self.min_analysis_interval = 60
        self.last_analysis_time = 0
        self.epoch_metrics = []
        
        # 创建保存图像的目录
        self.output_dir = Path(config.get('paths', {}).get('output_dir', 'outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def analyze_batch(self, epoch: int, batch_idx: int, 
                     outputs: torch.Tensor, targets: torch.Tensor,
                     metrics: Dict) -> Dict:
        """分析训练批次结果"""
        current_time = time.time()
        
        if (self.analysis_frequency <= 0 or 
            (epoch * batch_idx) % self.analysis_frequency != 0 or
            current_time - self.last_analysis_time < self.min_analysis_interval):
            return metrics
            
        self.last_analysis_time = current_time
        
        try:
            # 保存生成的图像用于调试
            if self.config.get('debug', {}).get('save_images', False):
                output_image = self._tensor_to_pil(outputs[0])
                save_path = self.output_dir / f"epoch_{epoch}_batch_{batch_idx}.png"
                output_image.save(str(save_path))
                self.logger.debug(f"保存生成图像: {save_path}")
            
            # 分析训练指标
            analysis = self.openai_helper.analyze_training(metrics)
            if analysis:
                self.logger.info(f"训练分析: {analysis}")
                metrics['training_analysis'] = analysis
                
                if analysis.get('risk_level') == 'high':
                    self.logger.warning("训练风险警告！请查看分析建议。")
                    
        except Exception as e:
            self.logger.error(f"分析过程出错: {str(e)}")
            
        return metrics
        
    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """将tensor转换为PIL图像（仅用于调试）"""
        if tensor.is_cuda:
            tensor = tensor.cpu()
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        if tensor.dim() == 3:
            if tensor.size(0) == 1:
                tensor = tensor.squeeze(0)
            else:
                tensor = tensor.permute(1, 2, 0)
                
        numpy_image = tensor.detach().numpy()
        numpy_image = np.clip(numpy_image * 255, 0, 255).astype(np.uint8)
        
        return Image.fromarray(numpy_image)
        
    def get_epoch_suggestions(self, epoch_metrics: Dict) -> Optional[Dict]:
        """分析epoch级别的训练指标并提供建议"""
        try:
            self.epoch_metrics.append(epoch_metrics)
            if len(self.epoch_metrics) > 5:  # 保留最近5个epoch的数据
                self.epoch_metrics.pop(0)
                
            metrics_summary = {
                'current': epoch_metrics,
                'trend': self._calculate_epoch_trends(),
                'train_loss': epoch_metrics.get('train_loss'),
                'val_loss': epoch_metrics.get('val_loss'),
                'learning_rate': epoch_metrics.get('lr')
            }
            
            response = self.openai_helper.analyze_training(metrics_summary)
            
            if response:
                self.logger.info(f"Epoch分析: {response}")
                return response
            return None
            
        except Exception as e:
            self.logger.error(f"获取epoch建议失败: {str(e)}")
            return None
            
    def _calculate_epoch_trends(self) -> Dict:
        """计算训练趋势"""
        if len(self.epoch_metrics) < 2:
            return {}
            
        current = self.epoch_metrics[-1]
        previous = self.epoch_metrics[-2]
        
        return {
            'train_loss_change': current.get('train_loss', 0) - previous.get('train_loss', 0),
            'val_loss_change': current.get('val_loss', 0) - previous.get('val_loss', 0),
            'loss_gap': current.get('val_loss', 0) - current.get('train_loss', 0)
        }