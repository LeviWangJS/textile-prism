class OptimizationStrategyGenerator:
    def __init__(self, study: optuna.Study):
        self.study = study
        
    def generate_strategy(self) -> Dict:
        """生成优化策略建议"""
        trials_df = self.study.trials_dataframe()
        
        # 分析参数重要性
        param_importance = optuna.importance.get_param_importances(self.study)
        
        # 分析参数相关性
        param_correlations = self._analyze_parameter_correlations(trials_df)
        
        # 生成优化建议
        suggestions = self._generate_suggestions(
            param_importance,
            param_correlations,
            trials_df
        )
        
        return {
            'param_importance': param_importance,
            'param_correlations': param_correlations,
            'suggestions': suggestions
        }
        
    def _analyze_parameter_correlations(self, trials_df):
        """分析参数相关性"""
        param_columns = [c for c in trials_df.columns if c.startswith('params_')]
        return trials_df[param_columns].corr()
        
    def _generate_suggestions(self, importance, correlations, trials_df):
        """生成优化建议"""
        suggestions = []
        
        # 基于重要参数
        for param, importance_score in importance.items():
            if importance_score > 0.1:
                best_value = self.study.best_params[param]
                suggestions.append(f"Parameter {param} is important (score: {importance_score:.3f})")
                suggestions.append(f"Consider fine-tuning around value: {best_value}")
                
        # 基于相关性
        for param1 in correlations.index:
            for param2 in correlations.columns:
                if param1 < param2:
                    corr = correlations.loc[param1, param2]
                    if abs(corr) > 0.7:
                        suggestions.append(
                            f"Strong correlation ({corr:.2f}) between {param1} and {param2}"
                        )
                        
        return suggestions 