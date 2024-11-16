import os
import sys
import torch
import yaml
from pathlib import Path

# 添加src目录到Python路径
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from models.pattern_net import PatternNet
from data.dataloader import create_dataloader
from models.losses import PatternLoss
from utils.logger import setup_logger
from utils.visualization import save_image

def test_environment():
    """测试环境配置"""
    print("\n=== 测试环境 ===")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 测试数据目录
    data_dir = Path("data/raw")
    if not data_dir.exists():
        raise RuntimeError(f"数据目录不存在: {data_dir}")
    print(f"数据目录检查通过: {data_dir}")
    
    return device

def test_data_loading(config):
    """测试数据加载"""
    print("\n=== 测试数据加载 ===")
    loader = create_dataloader(config, mode='train')
    batch = next(iter(loader))
    
    print(f"批次大小: {batch['input'].shape}")
    print(f"数据类型: {batch['input'].dtype}")
    print(f"值范围: [{batch['input'].min():.2f}, {batch['input'].max():.2f}]")
    
    return batch

def test_model(config, device, batch):
    """测试模型"""
    print("\n=== 测试模型 ===")
    model = PatternNet(config).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试前向传播
    with torch.no_grad():
        inputs = batch['input'].to(device)
        outputs, aux_outputs = model(inputs)
    
    print(f"输出形状: {outputs.shape}")
    print(f"输出值范围: [{outputs.min():.2f}, {outputs.max():.2f}]")
    
    return model

def test_loss(config, device, model, batch):
    """测试损失函数"""
    print("\n=== 测试损失函数 ===")
    criterion = PatternLoss(config)
    
    # 计算损失
    inputs = batch['input'].to(device)
    targets = batch['target'].to(device)
    
    with torch.no_grad():
        outputs, _ = model(inputs)
        losses = criterion(outputs, targets)
    
    for name, value in losses.items():
        print(f"{name} loss: {value.item():.4f}")

def test_training_step(config, device, model, batch):
    """测试训练步骤"""
    print("\n=== 测试训练步骤 ===")
    criterion = PatternLoss(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['learning_rate'])
    
    # 移动数据到设备
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}
    
    # 训练步骤
    model.train()
    optimizer.zero_grad()
    losses, outputs = model.train_step(batch, criterion)
    
    total_loss = losses['total']
    total_loss.backward()
    optimizer.step()
    
    print(f"训练步骤完成，总损失: {total_loss.item():.4f}")
    
    # 保存测试输出
    save_image(
        outputs['outputs'][0],
        Path("outputs") / "test_output.png"
    )
    print("测试输出已保存到 outputs/test_output.png")

def main():
    # 加载配置
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # 创建必要的目录
    for dir_name in ['outputs', 'checkpoints', 'logs']:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
    
    # 测试环境
    device = test_environment()
    
    # 测试数据加载
    batch = test_data_loading(config)
    
    # 测试模型
    model = test_model(config, device, batch)
    
    # 测试损失函数
    test_loss(config, device, model, batch)
    
    # 测试训练步骤
    test_training_step(config, device, model, batch)

if __name__ == "__main__":
    main() 