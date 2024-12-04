import os
import sys
import coverage
import yaml
from pathlib import Path

def run_tests_with_coverage():
    # 启动覆盖率收集
    cov = coverage.Coverage()
    cov.start()
    
    try:
        # 运行测试
        import tests.run_all_tests
        success = tests.run_all_tests.run_all_tests()
        
        # 停止覆盖率收集
        cov.stop()
        
        # 生成报告
        print("\n" + "="*80)
        print(" 代码覆盖率报告 ")
        print("="*80 + "\n")
        
        # 生成控制台报告
        cov.report()
        
        # 生成HTML报告
        cov.html_report()
        
        print(f"\nHTML报告已生成到: {os.path.abspath('coverage_html_report')}")
        
        return success
        
    except Exception as e:
        print(f"运行测试时发生错误: {str(e)}")
        return False

if __name__ == '__main__':
    success = run_tests_with_coverage()
    sys.exit(0 if success else 1) 