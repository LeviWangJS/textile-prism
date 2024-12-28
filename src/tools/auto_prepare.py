"""
自动化数据集准备脚本
包括：
1. 定时下载图片
2. 自动执行PS处理
3. 收集数据到data/raw
4. 预处理数据集
5. 分割数据集
"""
import os
import time
import logging
import asyncio
import schedule
from pathlib import Path
from datetime import datetime
from typing import Optional

# 导入所需模块
from image_downloader.unsplash_downloader import UnsplashDownloader
from photoshop_processor.processor import PhotoshopProcessor
from data_collector import DataCollector
from ..data.prepare_dataset import prepare_dataset
from ..data.split_dataset import split_dataset

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('auto_prepare.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutoPrepare:
    """自动化数据集准备"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent.parent
        self.downloader = UnsplashDownloader()
        self.processor = PhotoshopProcessor()
        self.collector = DataCollector()
        
        # 创建必要的目录
        self.raw_images_dir = self.base_dir / 'data/raw_images'
        self.raw_dir = self.base_dir / 'data/raw'
        self.processed_dir = self.base_dir / 'data/processed'
        self.split_dir = self.base_dir / 'data/split'
        
        for dir_path in [self.raw_images_dir, self.raw_dir, 
                        self.processed_dir, self.split_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    async def download_images(self):
        """下载图片"""
        logger.info("开始下载图片...")
        try:
            await self.downloader.run()
        except Exception as e:
            logger.error(f"下载图片时出错: {str(e)}")
    
    def process_images(self):
        """处理图片"""
        logger.info("开始处理图片...")
        try:
            self.processor.process_all()
        except Exception as e:
            logger.error(f"处理图片时出错: {str(e)}")
    
    def collect_data(self):
        """收集数据"""
        logger.info("开始收集数据...")
        try:
            # 获取所有已下载但未处理的图片
            downloaded_images = sorted(self.raw_images_dir.glob('*.jpg'))
            processed_images = sorted(self.processor.processed_files)
            
            for img_path in downloaded_images:
                if str(img_path) not in processed_images:
                    continue
                
                # 获取对应的处理后图片
                img_id = img_path.stem.split('_')[0]  # 提取Unsplash ID
                processed_path = next(
                    (p for p in processed_images if img_id in p),
                    None
                )
                
                if processed_path:
                    # 添加到数据集
                    self.collector.add_image_pair(
                        str(img_path),
                        processed_path
                    )
        except Exception as e:
            logger.error(f"收集数据时出错: {str(e)}")
    
    def prepare_dataset(self):
        """预处理数据集"""
        logger.info("开始预处理数据集...")
        try:
            # 加载配置
            with open(self.base_dir / 'configs/config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            
            # 执行预处理
            prepare_dataset(config)
        except Exception as e:
            logger.error(f"预处理数据集时出错: {str(e)}")
    
    def split_dataset(self):
        """分割数据集"""
        logger.info("开始分割数据集...")
        try:
            # 加载配置
            with open(self.base_dir / 'configs/config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            
            # 执行分割
            split_dataset(
                processed_dir=config['data']['raw_dir'],
                output_dir=config['data']['split_dir'],
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15
            )
        except Exception as e:
            logger.error(f"分割数据集时出错: {str(e)}")
    
    async def run_pipeline(self):
        """运行完整流程"""
        logger.info("开始运行数据集准备流程...")
        
        try:
            # 1. 下载图片
            await self.download_images()
            
            # 2. PS处理
            self.process_images()
            
            # 3. 收集数据
            self.collect_data()
            
            # 4. 预处理数据集
            self.prepare_dataset()
            
            # 5. 分割数据集
            self.split_dataset()
            
            logger.info("数据集准备流程完成")
            
        except Exception as e:
            logger.error(f"数据集准备流程出错: {str(e)}")
    
    def schedule_pipeline(self, interval_hours: int = 12):
        """定时运行流程"""
        logger.info(f"设置定时任务，每{interval_hours}小时运行一次")
        
        def run_job():
            asyncio.run(self.run_pipeline())
        
        # 设置定时任务
        schedule.every(interval_hours).hours.do(run_job)
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # 每分钟检查一次

def main():
    """主函数"""
    try:
        auto_prepare = AutoPrepare()
        
        # 解析命令行参数
        import argparse
        parser = argparse.ArgumentParser(description='自动化数据集准备')
        parser.add_argument(
            '--schedule',
            type=int,
            help='定时运行间隔（小时）',
            default=0
        )
        args = parser.parse_args()
        
        if args.schedule > 0:
            # 定时运行
            auto_prepare.schedule_pipeline(args.schedule)
        else:
            # 立即运行一次
            asyncio.run(auto_prepare.run_pipeline())
            
    except KeyboardInterrupt:
        logger.info("用户中断运行")
    except Exception as e:
        logger.error(f"运行出错: {str(e)}")
    finally:
        logger.info("程序已停止")

if __name__ == '__main__':
    main() 