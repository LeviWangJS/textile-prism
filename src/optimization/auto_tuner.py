from typing import Dict, List, Optional
import optuna
from dataclasses import dataclass
import logging
import numpy as np
from pathlib import Path
import pandas as pd
import copy

@dataclass
class OptimizationConfig:
    """优化配置类"""
    n_trials: int = 100  # 优化试验次数
    timeout: int = 3600 * 24  # 超时时间(秒)
    n_jobs: int = 1  # 并行任务数
    
class OptimizationStrategyGenerator:
    """优化策略生成器"""
    def __init__(self, study: optuna.Study):
        self.study = study
        self.logger = logging.getLogger(__name__)
        
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
        
    def _analyze_parameter_correlations(self, trials_df: pd.DataFrame) -> pd.DataFrame:
        """分析���数相关性"""
        param_columns = [c for c in trials_df.columns if c.startswith('params_')]
        if not param_columns:
            return pd.DataFrame()
        return trials_df[param_columns].corr()
        
    def _generate_suggestions(self, 
                            importance: Dict, 
                            correlations: pd.DataFrame,
                            trials_df: pd.DataFrame) -> List[str]:
        """生成优化建议"""
        suggestions = []
        
        # 基于重要参数
        for param, importance_score in importance.items():
            if importance_score > 0.1:
                best_value = self.study.best_params[param]
                suggestions.append(f"Parameter {param} is important (score: {importance_score:.3f})")
                suggestions.append(f"Consider fine-tuning around value: {best_value}")
                
        # 基于相关性
        if not correlations.empty:
            for param1 in correlations.index:
                for param2 in correlations.columns:
                    if param1 < param2:
                        corr = correlations.loc[param1, param2]
                        if abs(corr) > 0.7:
                            suggestions.append(
                                f"Strong correlation ({corr:.2f}) between {param1} and {param2}"
                            )
                            
        # 基于性能趋势
        if 'value' in trials_df.columns:
            recent_trials = trials_df.tail(10)
            if recent_trials['value'].is_monotonic_decreasing:
                suggestions.append("Performance is still improving, consider more trials")
            elif recent_trials['value'].std() < 0.01:
                suggestions.append("Performance has stabilized, current parameters may be optimal")
                
        return suggestions

class AutoTuner:
    def __init__(self, base_config: Dict, optimization_config: OptimizationConfig, study_name: str):
        self.base_config = base_config
        self.opt_config = optimization_config
        self.study = optuna.create_study(
            study_name=study_name,
            direction="minimize"
        )
        self.logger = logging.getLogger(__name__)
        
    def define_search_space(self, trial: optuna.Trial) -> Dict:
        """定义搜索空间"""
        config = copy.deepcopy(self.base_config)
        
        # 使用suggest_float替代suggest_loguniform
        config['training']['learning_rate'] = trial.suggest_float(
            'learning_rate', 
            1e-5, 
            1e-2, 
            log=True
        )
        
        config['training']['weight_decay'] = trial.suggest_float(
            'weight_decay', 
            1e-6, 
            1e-3, 
            log=True
        )
        
        return config
        
    def objective(self, trial: optuna.Trial) -> float:
        """优化目标函数"""
        # 获取配置
        config = self.define_search_space(trial)
        
        try:
            # 直接使用train函数而不是Trainer类
            from train import train  # 导入train函数
            results = train(config)
            
            # 记录中间结果
            trial.report(results['best_val_loss'], step=results['epochs_trained'])
            
            # 提前停止检查
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return results['best_val_loss']
            
        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {str(e)}")
            raise optuna.TrialPruned() 