class TrainingAnalyzer:
    def __init__(self, config):
        self.config = config
        self.metrics = {}
        self.visualizer = TrainingVisualizer(config['visualization']['save_dir'])
        self.report_gen = ReportGenerator()
        
    def update_metrics(self, epoch, losses, lr):
        for name, value in losses.items():
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append({
                'epoch': epoch,
                'value': value.item() if torch.is_tensor(value) else value
            })
            
    def analyze_training(self):
        # 分析损失趋势
        loss_analysis = self._analyze_loss_trends()
        
        # 生成训练报告
        self._generate_report(loss_analysis)
        
        # 保存可视化结果
        self._save_visualizations()
        
    def _analyze_loss_trends(self):
        analysis = {}
        for metric_name, values in self.metrics.items():
            df = pd.DataFrame(values)
            
            # 计算统计指标
            stats = {
                'mean': df['value'].mean(),
                'std': df['value'].std(),
                'min': df['value'].min(),
                'max': df['value'].max(),
                'final': df['value'].iloc[-1],
                'improvement': df['value'].iloc[0] - df['value'].iloc[-1]
            }
            
            # 判断趋势
            stats['trend'] = self._analyze_trend(df['value'])
            
            analysis[metric_name] = stats
            
        return analysis
    
    def _analyze_trend(self, values):
        # 使用线性回归分析趋势
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        
        if abs(slope) < 0.001:
            return 'stable'
        return 'improving' if slope < 0 else 'worsening' 