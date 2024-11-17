# Textile Prism（目前还在训练中）

一个优雅的深度学习项目，致力于将各类纺织品商品效果图片转换为平面原图。

## 特点
- 支持多种织物类型（地毯、窗帘、毯子等）
- 智能视角转换
- 简单易用的数据收集工具

## 项目结构

```bash
pattern-extract/
├── configs/        # 配置文件
├── src/           # 源代码
│   ├── models/    # 模型定义
│   ├── tools/     # 工具脚本
│   ├── trainer/   # 训练相关
│   └── utils/     # 通用工具
└── tests/         # 测试代码
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

## 数据集结构

### 示例数据
项目包含两组示例数据：
1. 第一组：斜视角地毯照片及其对应的平面图
2. 第二组：不同角度的地毯照片及其平面效果图

### 添加新数据
使用数据收集工具添加新的训练数据：

```bash
# 添加单对图像
collect add -i path/to/input.jpg -t path/to/target.jpg

# 查看数据集状态
collect status
```

## 训练模型

```bash
# 测试环境
python3 src/test_run.py

# 开始训练
python3 src/train.py
```

## License

MIT