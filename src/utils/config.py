import yaml
from pathlib import Path
from typing import Dict

def load_config():
    """加载配置文件"""
    config_path = Path('configs/config.yaml')
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    return config 

def validate_scheduler_config(config: Dict) -> None:
    """验证scheduler配置的正确性"""
    scheduler_config = config['train']['scheduler']
    scheduler_type = scheduler_config['type']
    params = scheduler_config['params']
    
    try:
        if scheduler_type == 'step':
            int(params['step_size'])
            float(params['gamma'])
        elif scheduler_type == 'cosine':
            int(params['T_max'])
            float(params['eta_min'])
        elif scheduler_type == 'reduce_plateau':
            float(params['factor'])
            int(params['patience'])
            float(params['min_lr'])
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    except (ValueError, KeyError) as e:
        raise ValueError(f"Invalid scheduler configuration: {str(e)}") 