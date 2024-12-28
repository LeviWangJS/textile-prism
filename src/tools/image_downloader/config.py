"""
Unsplash图片下载器配置
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
BASE_DIR = Path(__file__).parent.parent.parent.parent
ENV_FILE = BASE_DIR / '.env'
load_dotenv(ENV_FILE)

# Unsplash API配置
UNSPLASH_ACCESS_KEY = os.getenv('UNSPLASH_ACCESS_KEY', '')  # 需要在环境变量中设置
UNSPLASH_SECRET_KEY = os.getenv('UNSPLASH_SECRET_KEY', '')  # 需要在环境变量中设置

# API限制配置
HOURLY_RATE_LIMIT = 50  # Demo阶段每小时请求限制
RATE_LIMIT_BUFFER = 5   # 保留一些配额用于其他操作

# 下载配置
MAX_IMAGES = 1000  # 计划下载的总图片数
BATCH_SIZE = 10    # 每批次下载的图片数(为了让配额更持久)
CONCURRENT_DOWNLOADS = 5  # 降低并发数以避免触发限制
TIMEOUT = 30  # 下载超时时间(秒)
DELAY_BETWEEN_BATCHES = 300  # 每批次间隔时间(秒)

# 搜索关键词 - 专注于纺织品材质和细节
SEARCH_KEYWORDS = [
    'fabric texture detail',
    'textile material natural',
    'fabric pattern close up',
    'woven textile detail',
    'knitted fabric texture',
    'silk fabric drape',
    'cotton fabric texture',
    'linen textile natural',
    'wool fabric detail',
    'denim texture close up',
    'velvet fabric detail',
    'satin fabric shine',
    'tweed fabric pattern',
    'chiffon fabric flow',
    'jersey fabric stretch'
]

# 图片质量和尺寸要求
MIN_WIDTH = 1000
MIN_HEIGHT = 1000
QUALITY = 'regular'  # regular, full, raw

# 保存路径配置
DATA_DIR = BASE_DIR / 'data'
DOWNLOAD_DIR = DATA_DIR / 'raw_images'
LOG_DIR = DATA_DIR / 'logs'

# 创建必要的目录
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 日志配置
LOG_FILE = LOG_DIR / 'download.log'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# 进度保存配置
PROGRESS_FILE = DATA_DIR / 'download_progress.json' 