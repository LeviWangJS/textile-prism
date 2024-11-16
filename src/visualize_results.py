import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import numpy as np

def visualize_results():
    # 加载原始图像
    data_dir = Path("data/raw")
    input_images = list(data_dir.glob("**/input.*"))
    if not input_images:
        print("未找到输入图像！")
        return
    
    input_path = input_images[0]
    input_img = cv2.imread(str(input_path))
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    
    # 加载生成的图像
    output_path = Path("outputs/test_output.png")
    if not output_path.exists():
        print("未找到输出图像！")
        return
    
    output_img = cv2.imread(str(output_path))
    output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    
    # 显示对比
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(input_img)
    plt.title("输入图像")
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(output_img)
    plt.title("模型输出")
    plt.axis('off')
    
    # 计算差异图
    if input_img.shape == output_img.shape:
        diff = np.abs(input_img.astype(float) - output_img.astype(float))
        diff = diff / diff.max() * 255  # 归一化以便更好地可视化
        
        plt.subplot(133)
        plt.imshow(diff.astype(np.uint8))
        plt.title("差异图")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_results() 