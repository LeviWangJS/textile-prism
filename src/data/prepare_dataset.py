import os
import yaml
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
from PIL import Image
import torch
import torchvision.transforms as T
from concurrent.futures import ThreadPoolExecutor
import logging

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler('data_preparation.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def validate_image(image_path):
    """验证图像是否有效"""
    try:
        img = Image.open(image_path)
        img.verify()
        img = Image.open(image_path)
        img.load()
        
        # 检查图像尺寸
        if img.size[0] < 100 or img.size[1] < 100:
            return False, f"图像尺寸过小: {img.size}"
        
        # 检查图像模式
        if img.mode not in ['RGB', 'RGBA']:
            return False, f"不支持的图像模式: {img.mode}"
        
        # 转换为numpy数组进行进��步检查
        img_array = np.array(img)
        
        # 检查是否全黑或全白
        if np.mean(img_array) < 5 or np.mean(img_array) > 250:
            return False, "图像可能是全黑或全白"
        
        # 检查对比度
        contrast = np.std(img_array)
        if contrast < 10:
            return False, f"图像对比度过低: {contrast:.2f}"
        
        return True, "图像有效"
    except Exception as e:
        return False, str(e)

def process_image_pair(input_path, target_path, output_dir, size=(640, 640)):
    """处理单对图像"""
    # 验证图像
    input_valid, input_msg = validate_image(input_path)
    target_valid, target_msg = validate_image(target_path)
    
    if not (input_valid and target_valid):
        return False, f"输入图像: {input_msg}, 目标图像: {target_msg}"
    
    try:
        # 读取图像
        input_img = Image.open(input_path).convert('RGB')
        target_img = Image.open(target_path).convert('RGB')
        
        # 调整尺寸
        input_img = input_img.resize(size, Image.LANCZOS)
        target_img = target_img.resize(size, Image.LANCZOS)
        
        # 保存处理后的图像
        output_pair_dir = output_dir / input_path.parent.name
        output_pair_dir.mkdir(parents=True, exist_ok=True)
        
        input_img.save(output_pair_dir / 'input.jpg', quality=95)
        target_img.save(output_pair_dir / 'target.jpg', quality=95)
        
        return True, "处理成功"
    except Exception as e:
        return False, str(e)

def analyze_dataset(data_dir):
    """分析数据集统计信息"""
    stats = {
        'total_pairs': 0,
        'valid_pairs': 0,
        'invalid_pairs': 0,
        'input_sizes': [],
        'target_sizes': [],
        'input_channels': [],
        'target_channels': [],
        'input_means': [],
        'input_stds': [],
        'target_means': [],
        'target_stds': []
    }
    
    for set_dir in tqdm(sorted(Path(data_dir).glob('set_*'))):
        input_path = set_dir / 'input.jpg'
        target_path = set_dir / 'target.jpg'
        
        if not (input_path.exists() and target_path.exists()):
            stats['invalid_pairs'] += 1
            continue
        
        try:
            input_img = cv2.imread(str(input_path))
            target_img = cv2.imread(str(target_path))
            
            if input_img is None or target_img is None:
                stats['invalid_pairs'] += 1
                continue
            
            stats['valid_pairs'] += 1
            stats['input_sizes'].append(input_img.shape[:2])
            stats['target_sizes'].append(target_img.shape[:2])
            stats['input_channels'].append(input_img.shape[2])
            stats['target_channels'].append(target_img.shape[2])
            
            # 计算均值和标准差
            stats['input_means'].append(np.mean(input_img, axis=(0,1)))
            stats['input_stds'].append(np.std(input_img, axis=(0,1)))
            stats['target_means'].append(np.mean(target_img, axis=(0,1)))
            stats['target_stds'].append(np.std(target_img, axis=(0,1)))
            
        except Exception as e:
            stats['invalid_pairs'] += 1
            continue
    
    stats['total_pairs'] = stats['valid_pairs'] + stats['invalid_pairs']
    
    # 计算整体统计信息
    if stats['valid_pairs'] > 0:
        stats['mean_input_size'] = np.mean(stats['input_sizes'], axis=0)
        stats['mean_target_size'] = np.mean(stats['target_sizes'], axis=0)
        stats['mean_input_channels'] = np.mean(stats['input_channels'])
        stats['mean_target_channels'] = np.mean(stats['target_channels'])
        stats['global_input_mean'] = np.mean(stats['input_means'], axis=0)
        stats['global_input_std'] = np.mean(stats['input_stds'], axis=0)
        stats['global_target_mean'] = np.mean(stats['target_means'], axis=0)
        stats['global_target_std'] = np.mean(stats['target_stds'], axis=0)
    
    return stats

def prepare_dataset(config):
    """准备数据集"""
    logger = setup_logging()
    
    # 创建输出目录
    raw_dir = Path(config['data']['raw_dir'])
    processed_dir = Path('data/processed')
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # 分析数据集
    logger.info("开始分析数据集...")
    stats = analyze_dataset(raw_dir)
    
    logger.info(f"数据集统计信息:")
    logger.info(f"总样本对数: {stats['total_pairs']}")
    logger.info(f"有效样本对数: {stats['valid_pairs']}")
    logger.info(f"无效样本对数: {stats['invalid_pairs']}")
    if stats['valid_pairs'] > 0:
        logger.info(f"平均输入尺寸: {stats['mean_input_size']}")
        logger.info(f"平均目标尺寸: {stats['mean_target_size']}")
        logger.info(f"输入图像全局均值: {stats['global_input_mean']}")
        logger.info(f"输入图像全局标准差: {stats['global_input_std']}")
    
    # 处理数据集
    logger.info("\n开始处理数据集...")
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for set_dir in sorted(raw_dir.glob('set_*')):
            input_path = set_dir / 'input.jpg'
            target_path = set_dir / 'target.jpg'
            
            if input_path.exists() and target_path.exists():
                future = executor.submit(
                    process_image_pair,
                    input_path,
                    target_path,
                    processed_dir,
                    (config['data']['input_size'][0], config['data']['input_size'][1])
                )
                futures.append((set_dir.name, future))
        
        # 收集处理结果
        success_count = 0
        for set_name, future in tqdm(futures):
            success, msg = future.result()
            if success:
                success_count += 1
            else:
                logger.warning(f"{set_name}: {msg}")
    
    logger.info(f"\n数据集处理完成:")
    logger.info(f"成功处理: {success_count}/{len(futures)} 对图像")
    
    # 保存数据集统计信息
    stats_file = processed_dir / 'dataset_stats.yaml'
    with open(stats_file, 'w') as f:
        yaml.dump(stats, f)
    
    logger.info(f"数据集统计信息已保存到: {stats_file}")
    
    return stats

def main():
    # 加载配置
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 准备数据集
    prepare_dataset(config)

if __name__ == '__main__':
    main() 