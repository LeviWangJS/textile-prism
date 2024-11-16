from pathlib import Path
import yaml
from trainer.trainer import Trainer

def main():
    # 加载配置
    config_path = Path('configs/config.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # 创建训练器
    trainer = Trainer(config)
    
    # 开始训练
    trainer.train()

if __name__ == '__main__':
    main()
