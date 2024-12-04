# Textile Prism

一个优雅的深度学习项目，致力于将各类纺织品商品效果图片转换为平面原图。

## 特点
- 支持多种织物类型（地毯、窗帘、毯子等）
- 智能视角转换
- 简单易用的数据收集工具
- 高效的训练流程
- 完善的可视化支持

## 项目结构

```bash
textile-prism/
├── configs/          # 配置文件
├── data/            # 数据集
│   ├── raw/        # 原始数据
│   ├── processed/  # 处理后的数据
│   └── split/      # 训练验证测试集
├── src/            # 源代码
│   ├── models/     # 模型定义
│   ├── data/       # 数据处理
│   └── utils/      # 通用工具
├── logs/           # 训练日志
├── checkpoints/    # 模型检查点
├── visualizations/ # 可视化结果
└── docs/           # 文档
```

## 环境配置

```bash
# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# 安装依赖
pip3 install -r requirements.txt
```

## 数据准备

1. 准备数据集：
```bash
# 处理原始数据
python src/data/prepare_dataset.py

# 划分数据集
python src/data/split_dataset.py
```

2. 测试数据流程：
```bash
python -m tests.data.test_data_pipeline
```

## 训练模型

1. 测试模型组件：
```bash
python3 -m tests.models.test_model_components
```

2. 开始训练：
```bash
python3 src/train.py
```

3. 监控训练：
```bash
# 查看训练日志
tail -f logs/training.log

# 查看可视化结果
tensorboard --logdir=runs
```

## 模型评估

```bash
# 运行测试
python -m tests.models.test_model_components

# 查看评估报告
cat logs/evaluation_report.txt
```

## License

MIT