import torch
import cv2
import numpy as np

# 测试基本功能
def test_environment():
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS可用: {torch.backends.mps.is_available()}")
    print(f"MPS已构建: {torch.backends.mps.is_built()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"Numpy version: {np.__version__}")
    
    # 测试 GPU 可用性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

# 运行测试
test_environment()