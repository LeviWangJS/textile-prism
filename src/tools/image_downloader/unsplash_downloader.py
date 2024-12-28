"""
Unsplash图片下载器
使用异步方式批量下载Unsplash图片,支持限速和进度保存
"""
import asyncio
import aiohttp
import logging
from pathlib import Path
from typing import List, Dict, Optional, Set
import random
import time
import json
from datetime import datetime
from . import config

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class UnsplashDownloader:
    """Unsplash图片异步下载器"""
    
    def __init__(self):
        self.access_key = config.UNSPLASH_ACCESS_KEY
        if not self.access_key:
            raise ValueError("请设置UNSPLASH_ACCESS_KEY环境变量")
            
        self.headers = {
            'Authorization': f'Client-ID {self.access_key}',
            'Accept-Version': 'v1'
        }
        self.download_dir = config.DOWNLOAD_DIR
        self.downloaded_count = 0
        self.failed_downloads = []
        self.downloaded_ids: Set[str] = set()  # 记录已下载的图片ID
        self.api_calls_remaining = config.HOURLY_RATE_LIMIT
        self.last_reset_time = time.time()
        
        # 加载已有进度
        self._load_progress()
        
    def _load_progress(self):
        """加载下载进度"""
        if config.PROGRESS_FILE.exists():
            try:
                with open(config.PROGRESS_FILE) as f:
                    data = json.load(f)
                    self.downloaded_ids = set(data.get('downloaded_ids', []))
                    self.downloaded_count = len(self.downloaded_ids)
                    logger.info(f"加载已有进度: {self.downloaded_count}张图片")
            except Exception as e:
                logger.error(f"加载进度文件失败: {str(e)}")
                
    def _save_progress(self):
        """保存下载进度"""
        try:
            with open(config.PROGRESS_FILE, 'w') as f:
                json.dump({
                    'downloaded_ids': list(self.downloaded_ids),
                    'last_update': datetime.now().isoformat()
                }, f)
        except Exception as e:
            logger.error(f"保存进度失败: {str(e)}")
            
    async def check_rate_limit(self, response: aiohttp.ClientResponse):
        """检查并更新API限制"""
        remaining = int(response.headers.get('X-Ratelimit-Remaining', 0))
        self.api_calls_remaining = remaining
        
        if remaining < config.RATE_LIMIT_BUFFER:
            reset_time = int(response.headers.get('X-Ratelimit-Reset', 0))
            current_time = time.time()
            sleep_time = reset_time - current_time
            if sleep_time > 0:
                logger.info(f"API配额即将用完,等待{sleep_time:.0f}秒后继续")
                await asyncio.sleep(sleep_time)
                self.api_calls_remaining = config.HOURLY_RATE_LIMIT
                
    async def search_photos(self, session: aiohttp.ClientSession, 
                          query: str, page: int = 1) -> List[Dict]:
        """搜索图片"""
        if self.api_calls_remaining < config.RATE_LIMIT_BUFFER:
            logger.warning("API配额不足,跳过本次搜索")
            return []
            
        url = 'https://api.unsplash.com/search/photos'
        params = {
            'query': query,
            'page': page,
            'per_page': config.BATCH_SIZE,
            'orientation': 'squarish',
            'content_filter': 'high',
        }
        
        try:
            async with session.get(url, params=params, headers=self.headers) as response:
                await self.check_rate_limit(response)
                if response.status == 200:
                    data = await response.json()
                    return data.get('results', [])
                else:
                    logger.error(f"搜索请求失败: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"搜索异常: {str(e)}")
            return []
            
    async def download_photo(self, session: aiohttp.ClientSession, 
                           photo: Dict, semaphore: asyncio.Semaphore) -> Optional[Path]:
        """下载单张图片"""
        photo_id = photo.get('id')
        if photo_id in self.downloaded_ids:
            logger.debug(f"图片已下载过: {photo_id}")
            return None
            
        if not self._validate_photo(photo):
            return None
            
        url = photo['urls'][config.QUALITY]
        file_name = f"{photo_id}_{int(time.time())}.jpg"
        file_path = self.download_dir / file_name
        
        try:
            async with semaphore:
                async with session.get(url, timeout=config.TIMEOUT) as response:
                    if response.status == 200:
                        content = await response.read()
                        file_path.write_bytes(content)
                        self.downloaded_count += 1
                        self.downloaded_ids.add(photo_id)
                        logger.info(f"下载成功: {file_name} ({self.downloaded_count}/{config.MAX_IMAGES})")
                        return file_path
                    else:
                        logger.error(f"下载失败: {url} - 状态码: {response.status}")
                        self.failed_downloads.append(url)
                        return None
        except Exception as e:
            logger.error(f"下载异常: {url} - {str(e)}")
            self.failed_downloads.append(url)
            return None
            
    def _validate_photo(self, photo: Dict) -> bool:
        """验证图片是否满足要求"""
        try:
            width = photo['width']
            height = photo['height']
            return width >= config.MIN_WIDTH and height >= config.MIN_HEIGHT
        except KeyError:
            return False
            
    async def download_batch(self, keyword: str, page: int):
        """下载一批图片"""
        async with aiohttp.ClientSession() as session:
            photos = await self.search_photos(session, keyword, page)
            if not photos:
                return
                
            semaphore = asyncio.Semaphore(config.CONCURRENT_DOWNLOADS)
            tasks = [
                self.download_photo(session, photo, semaphore)
                for photo in photos
            ]
            await asyncio.gather(*tasks)
            
            # 每批次后保存进度
            self._save_progress()
            
            # 批次间延迟
            if config.DELAY_BETWEEN_BATCHES > 0:
                logger.info(f"等待{config.DELAY_BETWEEN_BATCHES}秒后继续下一批")
                await asyncio.sleep(config.DELAY_BETWEEN_BATCHES)
            
    async def run(self):
        """运行下载器"""
        start_time = datetime.now()
        logger.info(f"开始下载 - 目标数量: {config.MAX_IMAGES}")
        logger.info(f"已有进度: {self.downloaded_count}张")
        
        page = 1
        while self.downloaded_count < config.MAX_IMAGES:
            if self.api_calls_remaining < config.RATE_LIMIT_BUFFER:
                logger.warning("API配额不足,等待下一个小时")
                await asyncio.sleep(3600)  # 等待一小时
                self.api_calls_remaining = config.HOURLY_RATE_LIMIT
                
            keyword = random.choice(config.SEARCH_KEYWORDS)
            await self.download_batch(keyword, page)
            
            if not self.downloaded_count % 50:  # 每50张图片记录一次进度
                logger.info(f"已下载: {self.downloaded_count}")
                logger.info(f"API调用配额剩余: {self.api_calls_remaining}")
                
            page += 1
            
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"下载完成 - 总数: {self.downloaded_count}")
        logger.info(f"耗时: {duration}")
        logger.info(f"失败数: {len(self.failed_downloads)}")
        
def main():
    """主函数"""
    try:
        downloader = UnsplashDownloader()
        asyncio.run(downloader.run())
    except KeyboardInterrupt:
        logger.info("用户中断下载")
    except Exception as e:
        logger.error(f"下载器异常: {str(e)}")
    finally:
        logger.info("下载器已停止")
    
if __name__ == '__main__':
    main() 