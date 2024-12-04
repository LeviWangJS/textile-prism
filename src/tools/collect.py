#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from data_collector import DataCollector

def print_tree(directory, prefix=""):
    """打印目录树结构"""
    print(f"{prefix}└── {directory.name}/")
    for path in sorted(directory.iterdir()):
        if path.is_dir():
            print_tree(path, prefix + "    ")
        else:
            print(f"{prefix}    └── {path.name}")

def main():
    parser = argparse.ArgumentParser(description='数据收集工具')
    
    # 添加子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # add 命令
    add_parser = subparsers.add_parser('add', help='添加一对图像')
    add_parser.add_argument('-i', '--input', required=True, help='输入图像路径（斜视角）')
    add_parser.add_argument('-t', '--target', required=True, help='目标图像路径（正视图）')
    
    # batch 命令
    batch_parser = subparsers.add_parser('batch', help='批量添加图像')
    batch_parser.add_argument('-d', '--dir', required=True, help='包含图像的目录')
    batch_parser.add_argument('--input-pattern', default='*_input.*', help='输入图像匹配模式')
    batch_parser.add_argument('--target-pattern', default='*_target.*', help='目标图像匹配模式')
    
    # status 命令
    status_parser = subparsers.add_parser('status', help='显示数据集状态')
    status_parser.add_argument('--tree', action='store_true', help='显示目录树结构')
    
    # validate 命令
    validate_parser = subparsers.add_parser('validate', help='验证数据集')
    
    args = parser.parse_args()
    
    # 初始化收集器
    collector = DataCollector()
    
    if args.command == 'add':
        success = collector.add_image_pair(args.input, args.target)
        if success:
            print("✅ 成功添加图像对")
        else:
            print("❌ 添加失败")
            sys.exit(1)
            
    elif args.command == 'batch':
        input_dir = Path(args.dir)
        input_files = sorted(input_dir.glob(args.input_pattern))
        target_files = sorted(input_dir.glob(args.target_pattern))
        
        if len(input_files) != len(target_files):
            print("❌ 输入图像和目标图像数量不匹配")
            sys.exit(1)
            
        success_count = 0
        for input_path, target_path in zip(input_files, target_files):
            if collector.add_image_pair(str(input_path), str(target_path)):
                success_count += 1
                
        print(f"✅ 成功添加 {success_count}/{len(input_files)} 对图像")
        
    elif args.command == 'status':
        data_dir = Path(collector.root_dir)
        sets = list(data_dir.glob("set_*"))
        
        print(f"\n📊 数据集状态:")
        print(f"总数据集: {len(sets)} 对")
        
        if args.tree:
            print("\n📁 数据目录结构:")
            print_tree(data_dir)
        else:
            print("\n最近添加的数据集:")
            for set_dir in sorted(sets)[-5:]:
                print(f"- {set_dir.name}")
                
        # 检查数据集完整性
        incomplete = []
        for set_dir in sets:
            input_path = set_dir / "input.jpg"
            target_path = set_dir / "target.jpg"
            if not input_path.exists() or not target_path.exists():
                incomplete.append(set_dir.name)
        
        if incomplete:
            print(f"\n⚠️  发现 {len(incomplete)} 个不完整的数据集:")
            for name in incomplete:
                print(f"- {name}")
            
    elif args.command == 'validate':
        data_dir = Path(collector.root_dir)
        sets = list(data_dir.glob("set_*"))
        valid_count = 0
        
        print("\n🔍 验证数据集...")
        for set_dir in sets:
            input_path = set_dir / "input.jpg"
            target_path = set_dir / "target.jpg"
            
            issues = []
            if not input_path.exists():
                issues.append("缺少输入图像")
            if not target_path.exists():
                issues.append("缺少目标图像")
                
            if not issues:
                valid_count += 1
            else:
                print(f"❌ {set_dir.name}: {', '.join(issues)}")
                
        print(f"\n✅ 有效数据集: {valid_count}/{len(sets)}")
        
        if valid_count < len(sets):
            print("\n💡 提示: 使用 status --tree 命令查看完整的目录结构")

if __name__ == "__main__":
    main() 