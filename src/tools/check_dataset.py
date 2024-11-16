#!/usr/bin/env python3
import yaml
from pathlib import Path

def check_dataset():
    # 加载配置
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 检查数据目录
    data_dir = Path(config['data']['raw_dir'])
    if not data_dir.exists():
        print(f"❌ 数据目录不存在：{data_dir}")
        return False
    
    # 检查数据集
    sample_dirs = sorted(data_dir.glob("set_*"))
    if not sample_dirs:
        print(f"❌ 未找到数据集！请检查目录：{data_dir}")
        return False
    
    print(f"\n✅ 找到 {len(sample_dirs)} 个数据集")
    
    # 检查每个数据集的文件
    valid_count = 0
    for sample_dir in sample_dirs:
        input_path = sample_dir / "input.jpg"
        target_path = sample_dir / "target.jpg"
        
        if input_path.exists() and target_path.exists():
            valid_count += 1
        else:
            print(f"❌ 数据集 {sample_dir.name} 文件不完整")
    
    print(f"✅ 有效数据集：{valid_count}/{len(sample_dirs)}")
    return valid_count > 0

if __name__ == "__main__":
    check_dataset() 