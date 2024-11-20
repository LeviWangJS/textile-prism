import os
import sys
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from models.transformer import PatternTransformer
from utils.visualizer import save_image
import logging
import random
import matplotlib.pyplot as plt

def setup_logger():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)

def load_model(config, checkpoint_path):
    """加载训练好的模型"""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = PatternTransformer(config).to(device)
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, device

def process_image(image_path, config):
    """处理输入图像"""
    # 读取图像
    image = Image.open(image_path).convert('RGB')
    
    # 定义转换
    transform = transforms.Compose([
        transforms.Resize(config['data']['input_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 转换图像
    image = transform(image)
    return image.unsqueeze(0)  # 添加batch维度

def show_comparison(input_path, output_path):
    """显示输入和输出图像的对比"""
    input_img = Image.open(input_path)
    output_img = Image.open(output_path)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(input_img)
    plt.title('Input')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(output_img)
    plt.title('Output')
    plt.axis('off')
    
    plt.show()

def main():
    logger = setup_logger()
    
    # 加载配置
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建输出目录
    output_dir = Path('inference_results')
    output_dir.mkdir(exist_ok=True)
    
    # 加载模型
    model, device = load_model(config, 'checkpoints/best_model.pth')
    logger.info("Model loaded successfully")
    
    # 获取输入图像路径
    input_dir = Path(config['data']['raw_dir'])
    image_paths = list(input_dir.glob('**/input.*'))
    
    if not image_paths:
        logger.error("No input images found!")
        return
    
    # 随机选择5个图像进行推理
    selected_images = random.sample(image_paths, min(5, len(image_paths)))
    logger.info(f"Randomly selected {len(selected_images)} images for inference")
    
    # 处理选中的图像
    for img_path in selected_images:
        logger.info(f"Processing {img_path}")
        
        # 准备输入
        input_tensor = process_image(img_path, config)
        input_tensor = input_tensor.to(device)
        
        # 推理
        with torch.no_grad():
            output = model(input_tensor)
        
        # 保存结果
        output_path = output_dir / f"{img_path.stem}_transformed.png"
        save_image(output[0], output_path)
        logger.info(f"Saved result to {output_path}")
        
        # 保存对比图
        input_save_path = output_dir / f"{img_path.stem}_input.png"
        save_image(input_tensor[0], input_save_path)
        
        # 显示对比图
        show_comparison(input_save_path, output_path)

if __name__ == '__main__':
    main() 