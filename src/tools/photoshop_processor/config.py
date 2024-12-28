"""
Photoshop处理器配置
"""
import os
from pathlib import Path
from types import SimpleNamespace

# 基础路径配置
BASE_DIR = Path(__file__).parent.parent.parent.parent
DATA_DIR = BASE_DIR / 'data'

# 输入输出目录
RAW_IMAGES_DIR = DATA_DIR / 'raw_images'  # 原始图片目录
RAW_DIR = DATA_DIR / 'raw'  # 数据集目录
TEST_OUTPUT_DIR = DATA_DIR / 'test_output'  # 测试输出目录
HISTORY_FILE = DATA_DIR / 'photoshop_history.json'

# 样机配置
MOCKUP_DIR = DATA_DIR / 'mockups'  # 样机文件目录
MOCKUP_TEMPLATES = {
    'carpet': {
        'file': 'fabricmockup.psd',
        'smart_object': 'Design',  # 智能对象图层名
    }
}

# 输出配置
OUTPUT_QUALITY = 12    # JPEG输出质量(1-12)

# 创建必要的目录
RAW_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)
MOCKUP_DIR.mkdir(parents=True, exist_ok=True)
TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 日志配置
LOG_DIR = DATA_DIR / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / 'photoshop_processor.log'
LOG_FORMAT = '%(asctime)s [%(levelname)s] %(message)s'

# 创建配置对象
config = SimpleNamespace(
    BASE_DIR=BASE_DIR,
    DATA_DIR=DATA_DIR,
    RAW_IMAGES_DIR=RAW_IMAGES_DIR,
    RAW_DIR=RAW_DIR,
    TEST_OUTPUT_DIR=TEST_OUTPUT_DIR,
    HISTORY_FILE=HISTORY_FILE,
    MOCKUP_DIR=MOCKUP_DIR,
    MOCKUP_TEMPLATES=MOCKUP_TEMPLATES,
    OUTPUT_QUALITY=OUTPUT_QUALITY,
    LOG_DIR=LOG_DIR,
    LOG_FILE=LOG_FILE,
    LOG_FORMAT=LOG_FORMAT
)