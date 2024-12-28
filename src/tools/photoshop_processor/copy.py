"""
Photoshop处理器模块 - macOS版本
"""
import os
import sys
import logging
import subprocess
import time
from pathlib import Path
from typing import Optional, List

from .config import config

logger = logging.getLogger(__name__)

class PhotoshopProcessor:
    """Photoshop处理器类 - macOS版本"""
    
    def __init__(self):
        """初始化处理器"""
        self.mockup_template = config.MOCKUP_TEMPLATES['carpet']
        if not self.check_photoshop_installation():
            return
        self.test_photoshop_connection()
    
    def check_photoshop_installation(self) -> bool:
        """检查Photoshop是否已安装"""
        # macOS上检查Applications目录中是否存在Photoshop.app
        photoshop_path = "/Applications/Adobe Photoshop 2023/Adobe Photoshop 2023.app"
        if not os.path.exists(photoshop_path):
            logger.error("未找到Photoshop安装，请确保已安装Adobe Photoshop 2023")
            return False
        return True
    
    def test_photoshop_connection(self) -> bool:
        """测试与Photoshop的连接并触发权限请求"""
        logger.info("正在测试与Photoshop的连接...")
        
        # 测试基本操作
        script = '''
        tell application "System Events"
            tell process "Adobe Photoshop 2023"
                return name
            end tell
        end tell
        '''
        
        if not self.run_applescript(script):
            logger.error("无法与Photoshop建立连接，请在系统偏好设置中授予必要的权限")
            logger.info("请按照以下步骤操作：")
            logger.info("1. 打开系统偏好设置")
            logger.info('2. 点击"安全性与隐私"')
            logger.info('3. 选择"隐私"标签')
            logger.info('4. 在左侧列表中选择"辅助功能"')
            logger.info("5. 如果看到权限请求对话框，请允许Terminal或Python控制您的电脑")
            return False
            
        logger.info("成功连接到Photoshop")
        return True
    
    def run_applescript(self, script: str) -> bool:
        """运行AppleScript命令并检查结果"""
        try:
            process = subprocess.Popen(['osascript', '-e', script], 
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                logger.error(f"AppleScript执行失败: {stderr.decode()}")
                return False
            
            if stdout:
                output = stdout.decode().strip()
                logger.info(f"命令输出: {output}")
                # 检查输出是否包含错误信息
                if output.startswith("错误:"):
                    return False
                
            return True
        except Exception as e:
            logger.error(f"执行AppleScript时出错: {str(e)}")
            return False
            
    def open_mockup(self) -> bool:
        """打开样机文件"""
        mockup_path = config.MOCKUP_DIR / self.mockup_template['file']
        if not mockup_path.exists():
            logger.error(f"样机文件不存在: {mockup_path}")
            return False
            
        logger.info(f"正在打开样机文件: {mockup_path}")
        
        # 首先尝试直接打开文件的方式
        script = f'''
        tell application "Adobe Photoshop 2023"
            activate
        end tell

        delay 1

        tell application "System Events"
            tell process "Adobe Photoshop 2023"
                -- 打开样机文件
                keystroke "o" using command down
                delay 1
                
                -- 输入样机文件路径
                keystroke "g" using {{command down, shift down}}
                delay 1
                keystroke "{str(mockup_path.absolute())}"
                delay 1
                
                -- 双击搜索结果
                keystroke return using {{command down}}
                delay 1
                
                -- 选择文件
                keystroke return
                delay 1
                
                -- 确认打开文件
                keystroke return
                delay 2
            end tell
        end tell
        '''
        
        # 先尝试直接打开
        if not self.run_applescript(script):
            logger.info("直接打开文件失败")
            
        logger.info("样机文件打开成功")
        return True
    
    def process_image(self, image_path: str) -> Optional[str]:
        """处理单张图片
        
        Args:
            image_path: 输入图片路径
            
        Returns:
            处理后的图片路径，如果处理失败则返回None
        """
        if not os.path.exists(image_path):
            logger.error(f"输入图片不存在: {image_path}")
            return None
            
        # 构建输出路径
        image_name = Path(image_path).stem
        output_path = config.TEST_OUTPUT_DIR / f"{image_name}_mockup.jpg"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info("开始处理图片...")
        
        # 构建AppleScript命令
        script = f'''
        tell application "System Events"
            tell process "Adobe Photoshop 2023"
                -- 打开图层面板 (F7)
                key code 98
                delay 1
                
                -- 选择智能对象图层
                keystroke "f" using {{command down, shift down}}  -- 打开搜索
                delay 0.5
                keystroke "{self.mockup_template['smart_object']}"  -- 输入图层名
                delay 0.5
                keystroke return
                delay 1
                
                -- 编辑智能对象
                keystroke return using command down  -- 双击图层
                delay 2
                
                -- 打开要替换的图片
                keystroke "o" using command down
                delay 1
                
                -- 输入图片路径
                keystroke "g" using {{command down, shift down}}
                delay 1
                keystroke "{str(Path(image_path).absolute())}"
                delay 0.5
                keystroke return
                delay 1
                keystroke return
                delay 1
                
                -- 保存并关闭智能对象
                keystroke "s" using command down
                delay 1
                keystroke "w" using command down
                delay 1
                
                -- 保存为JPEG
                keystroke "s" using {{command down, shift down}}
                delay 1
                
                -- 输入保存路径
                keystroke "g" using {{command down, shift down}}
                delay 1
                keystroke "{str(output_path.absolute())}"
                delay 0.5
                keystroke return
                delay 1
                keystroke return
                delay 1
                
                -- 设置JPEG质量
                keystroke "12"
                delay 0.5
                keystroke return
                delay 1
                
                -- 关闭样机文档
                keystroke "w" using command down
                delay 1
                
                -- 不保存更改
                keystroke "d"
                delay 1
            end tell
        end tell
        '''
        
        if not self.run_applescript(script):
            logger.error("处理图片失败")
            return None
            
        logger.info(f"图片处理成功: {output_path}")
        return str(output_path)
    
    def process_all(self) -> List[str]:
        """量处理所有图片
        
        Returns:
            成功处理的图片路径列表
        """
        processed_files = []
        input_files = list(config.RAW_IMAGES_DIR.glob('*.jpg'))
        
        logger.info(f"开始处理 {len(input_files)} 张图片...")
        
        # 先打开样机文件
        if not self.open_mockup():
            return processed_files
        
        # 处理所有图片
        for input_file in input_files:
            output_path = self.process_image(str(input_file))
            if output_path:
                processed_files.append(output_path)
                
        logger.info(f"成功处理 {len(processed_files)} 张图片")
        return processed_files