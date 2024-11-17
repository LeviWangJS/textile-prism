# 训练报告

## 实验时间
{{ timestamp }}

## 配置信息
{{ data.config }}

## 训练指标
{% for metric_name, values in data.metrics.items() %}
### {{ metric_name }}
{% for record in values %}
- {{ record }}
{% endfor %}
{% endfor %}

## 训练分析
{% for key, value in data.analysis.items() %}
### {{ key }}
{{ value }}
{% endfor %}

## 可视化结果
{% for viz in data.visualizations %}
![{{ viz.title }}]({{ viz.path }})
{% endfor %}

## 优化建议
{% for recommendation in data.recommendations %}
- {{ recommendation }}
{% endfor %}