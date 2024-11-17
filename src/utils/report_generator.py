import io
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import base64
from jinja2 import Template
import markdown
import numpy as np
from pathlib import Path
import weasyprint
from jinja2 import Environment, FileSystemLoader

class ReportGenerator:
    def __init__(self, config, save_dir):
        self.config = config
        self.save_dir = save_dir
        self.report_dir = os.path.join(save_dir, 'reports')
        os.makedirs(self.report_dir, exist_ok=True)
        
        # 初始化数据收集器
        self.training_data = {
            'config': config,
            'metrics': {},
            'analysis': {},
            'visualizations': [],
            'recommendations': []
        }
        
        # 加载报告模板
        self.templates = self._load_templates()
    
    def _load_templates(self):
        """加载HTML和Markdown模板"""
        templates_dir = Path(__file__).parent / 'templates'
        
        with open(templates_dir / 'report_template.html', 'r', encoding='utf-8') as f:
            html_template = Template(f.read())
            
        with open(templates_dir / 'report_template.md', 'r', encoding='utf-8') as f:
            md_template = Template(f.read())
            
        return {
            'html': html_template,
            'markdown': md_template
        }
    
    def add_metric(self, name, values, epoch):
        """添加训练指标"""
        if name not in self.training_data['metrics']:
            self.training_data['metrics'][name] = []
        self.training_data['metrics'][name].append({
            'epoch': epoch,
            'value': values
        })
    
    def add_visualization(self, fig, title, description):
        """添加可视化图表"""
        # 将matplotlib图表转换为base64字符串
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        self.training_data['visualizations'].append({
            'title': title,
            'description': description,
            'image': image_base64
        })
    
    def add_analysis(self, name, content):
        """添加分析结果"""
        self.training_data['analysis'][name] = content
    
    def add_recommendation(self, recommendation):
        """添加建议"""
        self.training_data['recommendations'].append(recommendation)
    
    def analyze_training_metrics(self):
        """分析训练指标"""
        metrics = self.training_data['metrics']
        
        for name, values in metrics.items():
            df = pd.DataFrame(values)
            
            analysis = {
                'mean': df['value'].mean(),
                'std': df['value'].std(),
                'min': df['value'].min(),
                'max': df['value'].max(),
                'improvement': df['value'].iloc[-1] - df['value'].iloc[0],
                'trend': 'improving' if df['value'].iloc[-1] < df['value'].iloc[0] else 'worsening'
            }
            
            self.add_analysis(f'{name}_stats', analysis)
            
            # 添加建议
            if name == 'val_loss' and analysis['trend'] == 'worsening':
                self.add_recommendation(
                    f"验证损失显示模型可能过拟合。建议增加正则化或减少模型复杂度。"
                )
    
    def generate_training_summary(self):
        """生成训练总结"""
        metrics = self.training_data['metrics']
        
        # 计算训练时间
        total_epochs = len(metrics.get('train_loss', []))
        
        summary = {
            'total_epochs': total_epochs,
            'best_val_loss': min([x['value'] for x in metrics.get('val_loss', [])]),
            'final_train_loss': metrics.get('train_loss', [])[-1]['value'],
            'learning_rate_range': [
                min([x['value'] for x in metrics.get('learning_rate', [])]),
                max([x['value'] for x in metrics.get('learning_rate', [])])
            ]
        }
        
        self.add_analysis('training_summary', summary)
    
    def generate_html_report(self):
        """生成HTML报告"""
        # 生成分析
        self.analyze_training_metrics()
        self.generate_training_summary()
        
        # 渲染HTML
        html_content = self.templates['html'].render(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            data=self.training_data
        )
        
        # 保存HTML报告
        report_path = os.path.join(self.report_dir, 'training_report.html')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return report_path
    
    def generate_markdown_report(self):
        """生成Markdown报告"""
        # 生成分析
        self.analyze_training_metrics()
        self.generate_training_summary()
        
        # 渲染Markdown
        md_content = self.templates['markdown'].render(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            data=self.training_data
        )
        
        # 保存Markdown报告
        report_path = os.path.join(self.report_dir, 'training_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        return report_path
    
    def generate_interactive_notebook(self):
        """生成交互式Jupyter notebook"""
        import nbformat as nbf
        
        nb = nbf.v4.new_notebook()
        
        # 添加导入单元格
        imports = """\
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import json
        """
        nb.cells.append(nbf.v4.new_code_cell(imports))
        
        # 添加数据加载单元格
        data_path = os.path.join(self.report_dir, 'training_data.json')
        with open(data_path, 'w') as f:
            json.dump(self.training_data, f)
            
        load_data = f"""\
        with open('{data_path}') as f:
            training_data = json.load(f)
        """
        nb.cells.append(nbf.v4.new_code_cell(load_data))
        
        # 添加分析单元格
        analysis_cells = [
            "# 损失曲线分析",
            "plt.figure(figsize=(10, 6))\nplt.plot(training_data['metrics']['train_loss'])\nplt.plot(training_data['metrics']['val_loss'])\nplt.title('Training and Validation Loss')\nplt.show()",
            
            "# 学习率分析",
            "plt.figure(figsize=(10, 6))\nplt.plot(training_data['metrics']['learning_rate'])\nplt.yscale('log')\nplt.title('Learning Rate Schedule')\nplt.show()",
            
            "# 性能指标统计",
            "pd.DataFrame(training_data['analysis']).T"
        ]
        
        for cell in analysis_cells:
            if cell.startswith('#'):
                nb.cells.append(nbf.v4.new_markdown_cell(cell))
            else:
                nb.cells.append(nbf.v4.new_code_cell(cell))
        
        # 保存notebook
        notebook_path = os.path.join(self.report_dir, 'training_analysis.ipynb')
        with open(notebook_path, 'w') as f:
            nbf.write(nb, f)
        
        return notebook_path 