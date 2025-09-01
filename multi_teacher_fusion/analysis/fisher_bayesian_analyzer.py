"""
Fisher Information Matrix & Bayesian Analysis Tools for Recommender Systems
深度分析推荐系统内部机制的工具集
"""

import numpy as np
import pandas as pd
from scipy.linalg import inv, eigh
from scipy.stats import multivariate_normal, beta, gamma
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class FisherInformationAnalyzer:
    """
    Fisher信息矩阵分析器
    用于分析模型参数的信息量和不确定性
    """
    
    def __init__(self, model_predictions, true_ratings, user_features=None, item_features=None):
        self.predictions = np.array(model_predictions)
        self.true_ratings = np.array(true_ratings)
        self.user_features = user_features
        self.item_features = item_features
        self.fisher_matrix = None
        self.parameter_uncertainty = None
        
    def compute_fisher_information(self, model_params=None):
        """
        计算Fisher信息矩阵
        """
        if model_params is None:
            # 使用预测误差作为参数估计
            residuals = self.true_ratings - self.predictions
            model_params = np.array([np.mean(residuals), np.std(residuals)])
        
        n_params = len(model_params)
        fisher_matrix = np.zeros((n_params, n_params))
        
        # 计算二阶导数（Fisher信息）
        for i in range(n_params):
            for j in range(n_params):
                # 使用数值方法计算Fisher信息
                if i == j:
                    fisher_matrix[i, j] = self._compute_diagonal_fisher(i, model_params)
                else:
                    fisher_matrix[i, j] = self._compute_offdiagonal_fisher(i, j, model_params)
        
        self.fisher_matrix = fisher_matrix
        return fisher_matrix
    
    def _compute_diagonal_fisher(self, param_idx, params):
        """计算Fisher矩阵对角元素"""
        epsilon = 1e-6
        n_samples = len(self.predictions)
        
        # 计算对数似然的二阶导数
        if param_idx == 0:  # 均值参数
            return n_samples / (params[1] ** 2)
        else:  # 方差参数
            residuals = self.true_ratings - self.predictions
            return n_samples / (2 * params[1] ** 2) + np.sum(residuals ** 2) / (params[1] ** 4)
    
    def _compute_offdiagonal_fisher(self, i, j, params):
        """计算Fisher矩阵非对角元素"""
        # 简化计算，假设参数独立
        return 0.0
    
    def compute_parameter_uncertainty(self):
        """
        计算参数不确定性（Fisher信息矩阵的逆）
        """
        if self.fisher_matrix is None:
            self.compute_fisher_information()
        
        try:
            self.parameter_uncertainty = inv(self.fisher_matrix)
        except np.linalg.LinAlgError:
            # 如果矩阵奇异，使用伪逆
            self.parameter_uncertainty = np.linalg.pinv(self.fisher_matrix)
        
        return self.parameter_uncertainty
    
    def analyze_information_content(self):
        """
        分析信息内容
        """
        if self.fisher_matrix is None:
            self.compute_fisher_information()
        
        # 计算特征值和特征向量
        eigenvals, eigenvecs = eigh(self.fisher_matrix)
        
        # 计算信息量指标
        info_content = {
            'determinant': np.linalg.det(self.fisher_matrix),
            'trace': np.trace(self.fisher_matrix),
            'condition_number': np.max(eigenvals) / np.max(eigenvals[eigenvals > 1e-10]),
            'effective_parameters': np.sum(eigenvals > 1e-6),
            'eigenvalues': eigenvals,
            'eigenvectors': eigenvecs
        }
        
        return info_content
    
    def plot_fisher_analysis(self):
        """
        可视化Fisher信息分析
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Fisher信息矩阵热图
        if self.fisher_matrix is not None:
            sns.heatmap(self.fisher_matrix, annot=True, cmap='viridis', ax=axes[0,0])
            axes[0,0].set_title('Fisher Information Matrix')
        
        # 2. 参数不确定性热图
        if self.parameter_uncertainty is not None:
            sns.heatmap(self.parameter_uncertainty, annot=True, cmap='plasma', ax=axes[0,1])
            axes[0,1].set_title('Parameter Uncertainty (Inverse Fisher)')
        
        # 3. 预测vs真实值散点图
        axes[1,0].scatter(self.true_ratings, self.predictions, alpha=0.6)
        axes[1,0].plot([min(self.true_ratings), max(self.true_ratings)], 
                      [min(self.true_ratings), max(self.true_ratings)], 'r--')
        axes[1,0].set_xlabel('True Ratings')
        axes[1,0].set_ylabel('Predicted Ratings')
        axes[1,0].set_title('Prediction vs True Ratings')
        
        # 4. 残差分布
        residuals = self.true_ratings - self.predictions
        axes[1,1].hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1,1].set_xlabel('Residuals')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_title('Residual Distribution')
        
        plt.tight_layout()
        return fig


class BayesianUncertaintyQuantifier:
    """
    贝叶斯不确定性量化器
    用于量化推荐结果的不确定性
    """
    
    def __init__(self, ensemble_predictions, llm_predictions, fusion_predictions, true_ratings):
        self.ensemble_preds = np.array(ensemble_predictions)
        self.llm_preds = np.array(llm_predictions)
        self.fusion_preds = np.array(fusion_predictions)
        self.true_ratings = np.array(true_ratings)
        
        # 贝叶斯先验参数
        self.prior_alpha = 1.0  # Beta分布参数
        self.prior_beta = 1.0
        self.prior_shape = 1.0  # Gamma分布参数
        self.prior_rate = 1.0
    
    def compute_prediction_uncertainty(self):
        """
        计算预测不确定性
        """
        uncertainties = {}
        
        for name, predictions in [('Ensemble', self.ensemble_preds), 
                                ('LLM', self.llm_preds), 
                                ('Fusion', self.fusion_preds)]:
            
            # 计算预测方差（不确定性度量）
            pred_variance = np.var(predictions)
            
            # 计算预测误差的不确定性
            errors = np.abs(predictions - self.true_ratings)
            error_uncertainty = np.var(errors)
            
            # 贝叶斯置信区间
            conf_intervals = self._compute_bayesian_intervals(predictions)
            
            uncertainties[name] = {
                'prediction_variance': pred_variance,
                'error_uncertainty': error_uncertainty,
                'mean_absolute_error': np.mean(errors),
                'confidence_intervals': conf_intervals,
                'prediction_entropy': self._compute_prediction_entropy(predictions)
            }
        
        return uncertainties
    
    def _compute_bayesian_intervals(self, predictions, confidence=0.95):
        """
        计算贝叶斯置信区间
        """
        n = len(predictions)
        
        # 使用正态-逆Gamma先验的后验分布
        posterior_mean = np.mean(predictions)
        posterior_var = np.var(predictions)
        
        # 计算置信区间
        alpha = 1 - confidence
        lower_bound = posterior_mean - 1.96 * np.sqrt(posterior_var / n)
        upper_bound = posterior_mean + 1.96 * np.sqrt(posterior_var / n)
        
        return (lower_bound, upper_bound)
    
    def _compute_prediction_entropy(self, predictions):
        """
        计算预测熵（不确定性度量）
        """
        # 将预测值离散化
        hist, bin_edges = np.histogram(predictions, bins=10, density=True)
        hist = hist + 1e-10  # 避免log(0)
        
        # 计算熵
        entropy = -np.sum(hist * np.log(hist) * np.diff(bin_edges))
        return entropy
    
    def model_comparison_analysis(self):
        """
        模型比较分析
        """
        models = {
            'Ensemble': self.ensemble_preds,
            'LLM': self.llm_preds,
            'Fusion': self.fusion_preds
        }
        
        comparison = {}
        
        for name, predictions in models.items():
            mse = np.mean((predictions - self.true_ratings) ** 2)
            mae = np.mean(np.abs(predictions - self.true_ratings))
            correlation = np.corrcoef(predictions, self.true_ratings)[0, 1]
            
            # 贝叶斯模型证据（近似）
            log_likelihood = self._compute_log_likelihood(predictions)
            model_complexity = self._estimate_model_complexity(predictions)
            bayesian_ic = -2 * log_likelihood + model_complexity
            
            comparison[name] = {
                'mse': mse,
                'mae': mae,
                'correlation': correlation,
                'log_likelihood': log_likelihood,
                'bayesian_ic': bayesian_ic,
                'model_evidence': np.exp(-bayesian_ic / 2)
            }
        
        return comparison
    
    def _compute_log_likelihood(self, predictions):
        """
        计算对数似然
        """
        residuals = self.true_ratings - predictions
        sigma = np.std(residuals)
        n = len(predictions)
        
        log_likelihood = -n/2 * np.log(2 * np.pi * sigma**2) - np.sum(residuals**2) / (2 * sigma**2)
        return log_likelihood
    
    def _estimate_model_complexity(self, predictions):
        """
        估计模型复杂度
        """
        # 简化的复杂度估计（基于预测多样性）
        unique_predictions = len(np.unique(np.round(predictions, 2)))
        complexity = np.log(unique_predictions)
        return complexity
    
    def plot_uncertainty_analysis(self):
        """
        可视化不确定性分析
        """
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        
        models = {
            'Ensemble': self.ensemble_preds,
            'LLM': self.llm_preds,
            'Fusion': self.fusion_preds
        }
        
        colors = ['blue', 'green', 'red']
        
        # 1. 预测分布比较
        for i, (name, predictions) in enumerate(models.items()):
            axes[0, 0].hist(predictions, bins=20, alpha=0.6, label=name, color=colors[i])
        axes[0, 0].set_xlabel('Predicted Ratings')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Prediction Distributions')
        axes[0, 0].legend()
        
        # 2. 误差分布比较
        for i, (name, predictions) in enumerate(models.items()):
            errors = np.abs(predictions - self.true_ratings)
            axes[0, 1].hist(errors, bins=20, alpha=0.6, label=name, color=colors[i])
        axes[0, 1].set_xlabel('Absolute Errors')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Error Distributions')
        axes[0, 1].legend()
        
        # 3. 不确定性度量
        uncertainties = self.compute_prediction_uncertainty()
        model_names = list(uncertainties.keys())
        pred_vars = [uncertainties[name]['prediction_variance'] for name in model_names]
        error_uncs = [uncertainties[name]['error_uncertainty'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, pred_vars, width, label='Prediction Variance', color='skyblue')
        axes[1, 0].bar(x + width/2, error_uncs, width, label='Error Uncertainty', color='lightcoral')
        axes[1, 0].set_xlabel('Models')
        axes[1, 0].set_ylabel('Uncertainty')
        axes[1, 0].set_title('Uncertainty Comparison')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(model_names)
        axes[1, 0].legend()
        
        # 4. 贝叶斯模型比较
        comparison = self.model_comparison_analysis()
        model_evidence = [comparison[name]['model_evidence'] for name in model_names]
        
        axes[1, 1].bar(model_names, model_evidence, color=['blue', 'green', 'red'], alpha=0.7)
        axes[1, 1].set_xlabel('Models')
        axes[1, 1].set_ylabel('Model Evidence')
        axes[1, 1].set_title('Bayesian Model Comparison')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 5. 置信区间可视化
        sample_indices = np.random.choice(len(self.true_ratings), 50, replace=False)
        sample_true = self.true_ratings[sample_indices]
        
        for i, (name, predictions) in enumerate(models.items()):
            sample_pred = predictions[sample_indices]
            ci_lower, ci_upper = uncertainties[name]['confidence_intervals']
            
            axes[2, 0].errorbar(sample_true, sample_pred, 
                              yerr=[[sample_pred - ci_lower] * len(sample_pred), 
                                   [ci_upper - sample_pred] * len(sample_pred)],
                              fmt='o', alpha=0.6, label=name, color=colors[i])
        
        axes[2, 0].plot([min(sample_true), max(sample_true)], 
                       [min(sample_true), max(sample_true)], 'k--')
        axes[2, 0].set_xlabel('True Ratings')
        axes[2, 0].set_ylabel('Predicted Ratings')
        axes[2, 0].set_title('Predictions with Confidence Intervals')
        axes[2, 0].legend()
        
        # 6. 模型性能指标
        metrics = ['mse', 'mae', 'correlation']
        metric_values = {metric: [comparison[name][metric] for name in model_names] 
                        for metric in metrics}
        
        x = np.arange(len(model_names))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            axes[2, 1].bar(x + i*width, metric_values[metric], width, 
                          label=metric.upper(), alpha=0.8)
        
        axes[2, 1].set_xlabel('Models')
        axes[2, 1].set_ylabel('Metric Value')
        axes[2, 1].set_title('Performance Metrics Comparison')
        axes[2, 1].set_xticks(x + width)
        axes[2, 1].set_xticklabels(model_names)
        axes[2, 1].legend()
        
        plt.tight_layout()
        return fig


class RecommenderSystemAnalyzer:
    """
    推荐系统综合分析器
    整合Fisher信息和贝叶斯分析
    """
    
    def __init__(self, predictions_dict, true_ratings, user_item_matrix=None):
        """
        predictions_dict: {'ensemble': [...], 'llm': [...], 'fusion': [...]}
        """
        self.predictions = predictions_dict
        self.true_ratings = true_ratings
        self.user_item_matrix = user_item_matrix
        
        # 初始化分析器
        self.fisher_analyzer = None
        self.bayesian_analyzer = None
        
    def initialize_analyzers(self):
        """
        初始化分析器
        """
        ensemble_preds = self.predictions.get('ensemble', [])
        llm_preds = self.predictions.get('llm', [])
        fusion_preds = self.predictions.get('fusion', [])
        
        # Fisher信息分析器（使用fusion预测）
        if len(fusion_preds) > 0:
            self.fisher_analyzer = FisherInformationAnalyzer(
                fusion_preds, self.true_ratings
            )
        
        # 贝叶斯不确定性分析器
        if len(ensemble_preds) > 0 and len(llm_preds) > 0 and len(fusion_preds) > 0:
            self.bayesian_analyzer = BayesianUncertaintyQuantifier(
                ensemble_preds, llm_preds, fusion_preds, self.true_ratings
            )
    
    def run_comprehensive_analysis(self):
        """
        运行综合分析
        """
        self.initialize_analyzers()
        
        results = {}
        
        # Fisher信息分析
        if self.fisher_analyzer:
            fisher_info = self.fisher_analyzer.compute_fisher_information()
            param_uncertainty = self.fisher_analyzer.compute_parameter_uncertainty()
            info_content = self.fisher_analyzer.analyze_information_content()
            
            results['fisher_analysis'] = {
                'fisher_matrix': fisher_info,
                'parameter_uncertainty': param_uncertainty,
                'information_content': info_content
            }
        
        # 贝叶斯不确定性分析
        if self.bayesian_analyzer:
            prediction_uncertainty = self.bayesian_analyzer.compute_prediction_uncertainty()
            model_comparison = self.bayesian_analyzer.model_comparison_analysis()
            
            results['bayesian_analysis'] = {
                'prediction_uncertainty': prediction_uncertainty,
                'model_comparison': model_comparison
            }
        
        return results
    
    def generate_analysis_report(self, save_path=None):
        """
        生成分析报告
        """
        results = self.run_comprehensive_analysis()
        
        report = []
        report.append("# 推荐系统深度分析报告")
        report.append("=" * 50)
        report.append("")
        
        # Fisher信息分析结果
        if 'fisher_analysis' in results:
            fisher_results = results['fisher_analysis']
            report.append("## Fisher信息分析")
            report.append("-" * 30)
            
            info_content = fisher_results['information_content']
            report.append(f"信息量行列式: {info_content['determinant']:.6f}")
            report.append(f"信息量迹: {info_content['trace']:.6f}")
            report.append(f"条件数: {info_content['condition_number']:.6f}")
            report.append(f"有效参数数: {info_content['effective_parameters']}")
            report.append("")
        
        # 贝叶斯分析结果
        if 'bayesian_analysis' in results:
            bayesian_results = results['bayesian_analysis']
            report.append("## 贝叶斯不确定性分析")
            report.append("-" * 30)
            
            # 模型比较
            model_comp = bayesian_results['model_comparison']
            for model_name, metrics in model_comp.items():
                report.append(f"### {model_name} 模型")
                report.append(f"  MSE: {metrics['mse']:.6f}")
                report.append(f"  MAE: {metrics['mae']:.6f}")
                report.append(f"  相关系数: {metrics['correlation']:.6f}")
                report.append(f"  对数似然: {metrics['log_likelihood']:.6f}")
                report.append(f"  贝叶斯IC: {metrics['bayesian_ic']:.6f}")
                report.append(f"  模型证据: {metrics['model_evidence']:.6f}")
                report.append("")
            
            # 不确定性分析
            pred_uncertainty = bayesian_results['prediction_uncertainty']
            report.append("### 预测不确定性")
            for model_name, uncertainty in pred_uncertainty.items():
                report.append(f"**{model_name}:**")
                report.append(f"  预测方差: {uncertainty['prediction_variance']:.6f}")
                report.append(f"  误差不确定性: {uncertainty['error_uncertainty']:.6f}")
                report.append(f"  预测熵: {uncertainty['prediction_entropy']:.6f}")
                report.append("")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
        
        return report_text
    
    def plot_comprehensive_analysis(self, save_path=None):
        """
        生成综合分析图表
        """
        fig = plt.figure(figsize=(20, 15))
        
        # Fisher分析图表
        if self.fisher_analyzer:
            plt.subplot(2, 3, (1, 2))
            self.fisher_analyzer.plot_fisher_analysis()
        
        # 贝叶斯分析图表
        if self.bayesian_analyzer:
            plt.subplot(2, 3, (3, 6))
            self.bayesian_analyzer.plot_uncertainty_analysis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


if __name__ == "__main__":
    # 示例使用
    print("Fisher信息矩阵 & 贝叶斯分析工具已加载")
    print("使用方法:")
    print("1. 初始化 RecommenderSystemAnalyzer")
    print("2. 调用 run_comprehensive_analysis()")
    print("3. 生成报告和图表")
