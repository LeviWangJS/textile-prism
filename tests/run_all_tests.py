import os
import sys
import yaml
import torch
from pathlib import Path
from termcolor import colored
import time

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 导入所有测试模块
from tests.models.test_losses import test_losses
from tests.models.test_model_components import test_model_components
from tests.models.test_feature_extractors import test_feature_extractors
from tests.data.test_data_pipeline import test_data_pipeline
from tests.training.test_training import test_training

def print_header(text):
    """打印带颜色的标题"""
    print("\n" + "="*80)
    print(colored(f" {text} ", "blue", attrs=["bold"]))
    print("="*80)

def print_result(name, success, time_taken):
    """打印测试结果"""
    status = colored("✓ PASS", "green") if success else colored("✗ FAIL", "red")
    print(f"{status} {name:<50} {time_taken:.2f}s")

def run_all_tests():
    """运行所有测试"""
    # 加载配置
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置随机种子
    torch.manual_seed(config['system']['seed'])
    
    # 创建测试结果目录
    Path('test_results').mkdir(parents=True, exist_ok=True)
    
    print_header("运行所有测试")
    
    test_cases = [
        ("损失函数测试", test_losses),
        ("模型组件测试", lambda: test_model_components(config)),
        ("特征提取器测试", test_feature_extractors),
        ("数据流水线测试", lambda: test_data_pipeline(config)),
        ("训练流程测试", lambda: test_training(config))
    ]
    
    results = []
    total_time = 0
    
    for name, test_func in test_cases:
        try:
            start_time = time.time()
            success = test_func()
            time_taken = time.time() - start_time
            total_time += time_taken
            results.append((name, success, time_taken))
        except Exception as e:
            print(f"\n{colored('错误', 'red')}: 在运行 {name} 时发生异常:")
            print(str(e))
            results.append((name, False, 0))
    
    # 打印总结
    print_header("测试结果总结")
    
    for name, success, time_taken in results:
        print_result(name, success, time_taken)
    
    # 计算统计信息
    total_tests = len(results)
    passed_tests = sum(1 for _, success, _ in results if success)
    failed_tests = total_tests - passed_tests
    
    print("\n" + "="*80)
    print(f"总测试数: {total_tests}")
    print(colored(f"通过: {passed_tests}", "green"))
    if failed_tests > 0:
        print(colored(f"失败: {failed_tests}", "red"))
    print(f"总耗时: {total_time:.2f}s")
    print("="*80)
    
    # 返回是否所有测试都通过
    return all(success for _, success, _ in results)

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1) 