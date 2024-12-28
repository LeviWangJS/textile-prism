"""
测试样机处理模块
"""
import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.tools.photoshop_processor import PhotoshopProcessor, config

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def test_single_image():
    """测试单张图片处理"""
    processor = PhotoshopProcessor()
    
    # 检查样机文件
    mockup_path = config.MOCKUP_DIR / config.MOCKUP_TEMPLATES['carpet']['file']
    logger.info(f"样机文件路径: {mockup_path}")
    if not mockup_path.exists():
        logger.error(f"样机文件不存在: {mockup_path}")
        return False
    
    # 获取第一张测试图片
    raw_images_dir = config.RAW_IMAGES_DIR
    logger.info(f"搜索测试图片目录: {raw_images_dir}")
    logger.info(f"目录是否存在: {raw_images_dir.exists()}")
    
    test_images = list(raw_images_dir.glob('*.jpg'))
    logger.info(f"找到的所有图片:")
    for img in test_images:
        logger.info(f"- {img}")
    
    if not test_images:
        logger.error(f"未找到测试图片: {raw_images_dir}")
        return False
        
    test_image = test_images[0]
    logger.info(f"选择的测试图片: {test_image}")
    logger.info(f"测试图片是否存在: {test_image.exists()}")
    
    # 先打开样机文件
    if not processor.open_mockup():
        return False
    
    # 处理图片
    logger.info(f"开始处理图片...")
    logger.info(f"测试图片路径: {str(test_image)}")
    output_path = processor.process_image(str(test_image))
    
    # if output_path:
    #     logger.info(f"测试成功，输出文件: {output_path}")
    #     return True
    # else:
    #     logger.error("测试失败")
    #     return False

def test_batch_processing():
    """测试批量处理"""
    processor = PhotoshopProcessor()
    
    # 检查输入目录
    raw_images_dir = config.RAW_IMAGES_DIR
    logger.info(f"检查输入目录: {raw_images_dir}")
    if not raw_images_dir.exists():
        logger.error(f"输入目录不存在: {raw_images_dir}")
        return False
        
    # 检查是否有测试图片
    test_images = list(raw_images_dir.glob('*.jpg'))
    logger.info(f"找到的所有图片:")
    for img in test_images:
        logger.info(f"- {img}")
    
    if not test_images:
        logger.error(f"未找到测试图片: {raw_images_dir}")
        return False
        
    logger.info(f"找到 {len(test_images)} 张测试图片")
    
    # 处理所有图片
    processor.process_all()
    
    logger.info("批量处理测试完成")
    return True

def main():
    """运行测试"""
    logger.info("开始测试样机处理模块...")
    
    # 1. 检查目录结构
    logger.info("\n1. 检查目录结构...")
    for dir_path in [config.RAW_IMAGES_DIR, config.MOCKUP_DIR, config.TEST_OUTPUT_DIR]:
        logger.info(f"检查目录: {dir_path}")
        logger.info(f"目录是否存在: {dir_path.exists()}")
        if not dir_path.exists():
            logger.error(f"目录不存在: {dir_path}")
            return
        logger.info(f"目录存在: {dir_path}")
    
    # 2. 测试单张图片处理
    logger.info("\n2. 测试单张图片处理...")
    if not test_single_image():
        logger.error("单张图片处理测试失败")
        return
        
    # 3. 测试批量处理
    logger.info("\n3. 测试批量处理...")
    if not test_batch_processing():
        logger.error("批量处理测试失败")
        return
        
    logger.info("\n所有测试完成!")

if __name__ == '__main__':
    main() 