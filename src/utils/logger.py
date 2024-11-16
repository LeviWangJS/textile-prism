import logging
from pathlib import Path
from typing import Dict

def setup_logger(config: Dict) -> logging.Logger:
    """设置日志器"""
    # 创建日志目录
    log_dir = Path(config['output']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建日志器
    logger = logging.getLogger('pattern_transform')
    logger.setLevel(logging.INFO)
    
    # 文件处理器
    fh = logging.FileHandler(log_dir / 'train.log')
    fh.setLevel(logging.INFO)
    
    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger 