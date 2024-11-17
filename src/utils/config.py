import yaml
from pathlib import Path

def load_config():
    """加载配置文件"""
    config_path = Path('configs/config.yaml')
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    return config 