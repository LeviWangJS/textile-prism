from pathlib import Path
import sys

def check_data_structure():
    data_dir = Path("data/raw")
    
    # 检查根目录
    if not data_dir.exists():
        print(f"❌ 数据根目录不存在: {data_dir}")
        return
        
    # 列出所有文件和目录
    print("\n📁 数据目录结构:")
    def print_tree(directory, prefix=""):
        print(f"{prefix}└── {directory.name}/")
        for path in sorted(directory.iterdir()):
            if path.is_dir():
                print_tree(path, prefix + "    ")
            else:
                print(f"{prefix}    └── {path.name}")
    
    print_tree(data_dir)
    
    # 检查数据集
    sets = list(data_dir.glob("set_*"))
    if not sets:
        print("\n❌ 未找到任何数据集目录 (set_*)")
    else:
        print(f"\n✅ 找到 {len(sets)} 个数据集目录")
        
        # 检查每个数据集
        for set_dir in sets:
            input_file = set_dir / "input.jpg"
            target_file = set_dir / "target.jpg"
            
            if not input_file.exists():
                print(f"❌ 缺少输入文件: {input_file}")
            if not target_file.exists():
                print(f"❌ 缺少目标文件: {target_file}")

if __name__ == "__main__":
    check_data_structure() 