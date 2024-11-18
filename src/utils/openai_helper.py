from openai import OpenAI
import base64
import io
import time
import logging
from PIL import Image
from typing import Dict, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
import json

class OpenAIHelper:
    def __init__(self, api_key: str, config: Dict):
        self.client = OpenAI(api_key=api_key)
        self.config = config
        self.rate_limit_delay = 5.0
        self.logger = logging.getLogger(__name__)
        self.metrics_history = []
        
    def _parse_analysis(self, content: str) -> Optional[Dict]:
        """解析GPT响应内容"""
        try:
            if not content or not content.strip():
                self.logger.error("响应内容为空")
                return None
                
            import re
            json_pattern = r'\{[\s\S]*\}'
            json_match = re.search(json_pattern, content)
            
            if json_match:
                return json.loads(json_match.group())
            else:
                self.logger.error("未找到JSON格式内容")
                return None
                
        except Exception as e:
            self.logger.error(f"分析结果解析失败: {str(e)}")
            return None
            
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=20),
        reraise=True
    )
    def analyze_training(self, metrics: Dict) -> Optional[Dict]:
        """分析训练指标"""
        try:
            time.sleep(self.rate_limit_delay)
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # 使用gpt-4
                messages=[
                    {
                        "role": "system",
                        "content": """你是一个深度学习专家，专注于训练优化。
                        分析训练指标并提供具体的优化建议。
                        必须以JSON格式返回分析结果。"""
                    },
                    {
                        "role": "user",
                        "content": f"""分析以下训练指标并给出优化建议：

当前指标：
训练损失: {metrics.get('train_loss', 'N/A')}
验证损失: {metrics.get('val_loss', 'N/A')}
学习率: {metrics.get('learning_rate', 'N/A')}

趋势分析：
{metrics.get('trend', {})}

请严格按照以下JSON格式返回：
{{
    "analysis": {{
        "convergence": "收敛状态分析",
        "overfitting": "过拟合风险分析",
        "learning_rate": "学习率分析"
    }},
    "suggestions": [
        "具体的优化建议1",
        "具体的优化建议2"
    ],
    "risk_level": "low/medium/high"
}}"""
                    }
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            return self._parse_analysis(response.choices[0].message.content)
            
        except Exception as e:
            self.logger.error(f"训练分析失败: {str(e)}")
            return None
            
    def _calculate_trends(self) -> str:
        """计算训练指标趋势"""
        if len(self.metrics_history) < 2:
            return "数据不足以分析趋势"
            
        try:
            latest = self.metrics_history[-1]
            previous = self.metrics_history[0]
            
            total_change = latest.get('total', 0) - previous.get('total', 0)
            l1_change = latest.get('l1', 0) - previous.get('l1', 0)
            perceptual_change = latest.get('perceptual', 0) - previous.get('perceptual', 0)
            
            return f"""
            - 总损失变化: {total_change:.4f}
            - L1损失变化: {l1_change:.4f}
            - 感知损失变化: {perceptual_change:.4f}
            """
            
        except Exception as e:
            self.logger.error(f"趋势计算失败: {str(e)}")
            return "趋势计算失败"
            
    def get_training_suggestions(self, metrics: Dict) -> Dict:
        """获取训练优化建议"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个专注于图像生成模型训练的深度学习专家。"
                    },
                    {
                        "role": "user",
                        "content": f"""基于以下训练指标，请提供具体的优化建议：

                        训练损失: {metrics.get('train_loss', 'N/A')}
                        验证损失: {metrics.get('val_loss', 'N/A')}
                        图像质量分数: {metrics.get('quality_score', 'N/A')}
                        
                        请分析：
                        1. 当前训练状态
                        2. 潜在的问题
                        3. 具体的优化建议
                        
                        以JSON格式返回：
                        {{
                            "status": "当前训练状态分析",
                            "issues": ["潜在问题1", "潜在问题2"],
                            "suggestions": [
                                "具体的优化建议1",
                                "具体的优化建议2"
                            ]
                        }}"""
                    }
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            self.logger.error(f"获取训练建议失败: {str(e)}")
            return {
                'status': '无法获取分析',
                'issues': [],
                'suggestions': []
            }
            
    def generate_reference_pattern(self, prompt: str) -> Optional[Image.Image]:
        """使用DALL-E 3生成参考图案"""
        try:
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=f"高质量的纺织图案，{prompt}",
                size="1024x1024",
                quality="hd",
                n=1
            )
            
            time.sleep(self.rate_limit_delay)
            
            # 下载生成的图像
            image_url = response.data[0].url
            return self._download_image(image_url)
            
        except Exception as e:
            print(f"DALL-E 3生成失败: {str(e)}")
            return None
            
    def _download_image(self, url: str) -> Optional[Image.Image]:
        """下载图像"""
        try:
            import requests
            response = requests.get(url)
            return Image.open(io.BytesIO(response.content))
        except Exception as e:
            print(f"图像下载失败: {str(e)}")
            return None 