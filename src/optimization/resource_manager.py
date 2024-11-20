import contextlib
from typing import Generator, Optional
import psutil
import threading
import time
import torch
import logging
from dataclasses import dataclass

@dataclass
class MPSResourceConfig:
    """MPS资源配置"""
    memory_threshold: float = 0.8  # 内存使用阈值
    check_interval: float = 1.0    # 资源检查间隔(秒)
    max_trials: int = 1            # 最大并行试验数
    min_free_memory: float = 2.0   # 最小可用内存(GB)

class MPSResourceManager:
    def __init__(self, config: MPSResourceConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.lock = threading.Lock()
        self.active_trials = set()
        
        # 验证MPS可用性
        self.device_available = torch.backends.mps.is_available()
        if not self.device_available:
            self.logger.warning("MPS device not available, falling back to CPU")
            
    def get_device(self) -> torch.device:
        """获取计算设备"""
        if self.device_available:
            return torch.device("mps")
        return torch.device("cpu")
        
    def get_memory_info(self) -> dict:
        """获取系统内存信息"""
        vm = psutil.virtual_memory()
        return {
            'total': vm.total / (1024**3),  # GB
            'available': vm.available / (1024**3),  # GB
            'used': vm.used / (1024**3),  # GB
            'percent': vm.percent
        }
        
    def _can_allocate_trial(self) -> bool:
        """检查是否可以分配新试验"""
        # 检查活跃试验数
        if len(self.active_trials) >= self.config.max_trials:
            return False
            
        # 检查可用内存
        mem_info = self.get_memory_info()
        if mem_info['available'] < self.config.min_free_memory:
            return False
            
        return True
        
    @contextlib.contextmanager
    def allocate_resource(self, trial_id: int) -> Generator:
        """分配计算资源"""
        try:
            # 等待资源可用
            while not self._can_allocate_trial():
                time.sleep(self.config.check_interval)
                
            with self.lock:
                self.active_trials.add(trial_id)
                self.logger.info(f"Allocated resources for trial {trial_id}")
                
                # 记录初始内存状态
                initial_mem = self.get_memory_info()
                self.logger.debug(f"Initial memory state: {initial_mem}")
            
            yield
            
        finally:
            with self.lock:
                self.active_trials.remove(trial_id)
                
                # 记录最终内存状态
                final_mem = self.get_memory_info()
                self.logger.debug(f"Final memory state: {final_mem}")
                
                self.logger.info(f"Released resources for trial {trial_id}")
                
    def monitor_performance(self) -> dict:
        """监控系统性能"""
        return {
            'memory': self.get_memory_info(),
            'cpu_percent': psutil.cpu_percent(),
            'active_trials': len(self.active_trials)
        }

class MemoryTracker:
    """内存使用跟踪器"""
    def __init__(self):
        self.memory_logs = []
        
    def log_memory(self, trial_id: int, stage: str):
        """记录内存使用情况"""
        if torch.backends.mps.is_available():
            # 获取系统内存信息
            mem_info = psutil.virtual_memory()
            
            self.memory_logs.append({
                'trial_id': trial_id,
                'stage': stage,
                'timestamp': time.time(),
                'system_memory': mem_info.percent,
                'available_memory': mem_info.available / (1024**3)  # GB
            })
            
    def get_memory_summary(self) -> dict:
        """获取内存使用摘要"""
        if not self.memory_logs:
            return {}
            
        df = pd.DataFrame(self.memory_logs)
        return {
            'peak_memory_usage': df['system_memory'].max(),
            'avg_memory_usage': df['system_memory'].mean(),
            'min_available_memory': df['available_memory'].min(),
            'memory_usage_by_stage': df.groupby('stage')['system_memory'].mean().to_dict()
        }