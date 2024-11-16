from torch.utils.data import DataLoader
from .dataset import CarpetDataset

def create_dataloader(config, mode='train'):
    # 创建数据集
    dataset = CarpetDataset(config, mode=mode)
    
    # 创建数据加载器
    loader = DataLoader(
        dataset,
        batch_size=config['data']['batch_size'],
        shuffle=(mode == 'train'),
        num_workers=config['data']['num_workers'],
        pin_memory=False
    )
    
    return loader 