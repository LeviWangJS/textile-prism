import os
import yaml
import random
import shutil
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler('dataset_split.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def split_dataset(processed_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """划分数据集为训练集、验证集和测试集"""
    logger = setup_logging()
    
    # 设置随机种子
    random.seed(seed)
    
    # 创建输出目录
    output_dir = Path(output_dir)
    train_dir = output_dir / 'train'
    val_dir = output_dir / 'val'
    test_dir = output_dir / 'test'
    
    for dir_path in [train_dir, val_dir, test_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有数据集目录
    processed_dir = Path(processed_dir)
    all_sets = sorted([d for d in processed_dir.glob('set_*') if d.is_dir()])
    
    if not all_sets:
        logger.error(f"在 {processed_dir} 中没有找到数据集")
        return
    
    # 计算划分数量
    total_sets = len(all_sets)
    logger.info(f"总数据集数量: {total_sets}")
    
    # 使用sklearn进行划分
    train_val_sets, test_sets = train_test_split(
        all_sets,
        test_size=test_ratio,
        random_state=seed
    )
    
    train_sets, val_sets = train_test_split(
        train_val_sets,
        test_size=val_ratio/(train_ratio + val_ratio),
        random_state=seed
    )
    
    # 复制文件到相应目录
    def copy_sets(sets, target_dir, split_name):
        logger.info(f"\n复制{split_name}集...")
        for set_dir in sets:
            target_set_dir = target_dir / set_dir.name
            target_set_dir.mkdir(parents=True, exist_ok=True)
            
            # 复制输入和目标图像
            for img_name in ['input.jpg', 'target.jpg']:
                src_path = set_dir / img_name
                if src_path.exists():
                    shutil.copy2(src_path, target_set_dir / img_name)
        
        logger.info(f"{split_name}集数量: {len(sets)}")
    
    # 执行复制
    copy_sets(train_sets, train_dir, "训练")
    copy_sets(val_sets, val_dir, "验证")
    copy_sets(test_sets, test_dir, "测试")
    
    # 保存划分信息
    split_info = {
        'total_sets': total_sets,
        'train_sets': [s.name for s in train_sets],
        'val_sets': [s.name for s in val_sets],
        'test_sets': [s.name for s in test_sets],
        'split_ratios': {
            'train': train_ratio,
            'val': val_ratio,
            'test': test_ratio
        }
    }
    
    split_info_file = output_dir / 'split_info.yaml'
    with open(split_info_file, 'w') as f:
        yaml.dump(split_info, f)
    
    logger.info(f"\n数据集划分信息已保存到: {split_info_file}")
    
    return split_info

def main():
    # 加载配置
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 执行数据集划分
    split_dataset(
        processed_dir='data/processed',
        output_dir='data/split',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )

if __name__ == '__main__':
    main() 