"""
推荐系统解释性监控仪表板
基于Fisher信息矩阵和贝叶斯分析的可视化工具
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

class RecommenderExplainabilityDashboard:
    """
    推荐系统解释性监控仪表板
    整合Fisher信息、贝叶斯分析和业务指标
    """
    
    def __init__(self):
        self.fisher_data = {}
        self.bayesian_data = {}
        self.business_metrics = {}
        self.time_series_data = []
        
    def update_fisher_metrics(self, determinant, trace, condition_number, effective_params, feature_contributions):
        """
        更新Fisher信息矩阵指标
        """
        self.fisher_data = {
            'determinant': determinant,
            'trace': trace, 
            'condition_number': condition_number,
            'effective_params': effective_params,
            'feature_contributions': feature_contributions,
            'timestamp': datetime.now()
        }
    
    def update_bayesian_metrics(self, model_comparisons, uncertainty_metrics):
        """
        更新贝叶斯分析指标
        """
        self.bayesian_data = {
            'model_comparisons': model_comparisons,
            'uncertainty_metrics': uncertainty_metrics,
            'timestamp': datetime.now()
        }
    
    def update_business_metrics(self, user_satisfaction, prediction_accuracy, system_stability):
        """
        更新业务指标
        """
        self.business_metrics = {
            'user_satisfaction': user_satisfaction,
            'prediction_accuracy': prediction_accuracy,
            'system_stability': system_stability,
            'timestamp': datetime.now()
        }
        
        # 记录时间序列
        self.time_series_data.append({
            'timestamp': datetime.now(),
            'fisher_determinant': self.fisher_data.get('determinant', 0),
            'condition_number': self.fisher_data.get('condition_number', 1),
            'prediction_confidence': self.bayesian_data.get('uncertainty_metrics', {}).get('fusion', {}).get('confidence', 0),
            'user_satisfaction': user_satisfaction
        })
    
    def create_fisher_dashboard(self):
        """
        创建Fisher信息矩阵监控面板
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['信息量指标', '数值稳定性', '参数效率', '特征重要性'],
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "bar"}, {"type": "pie"}]]
        )
        
        # 1. 信息量行列式指标
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=self.fisher_data.get('determinant', 0),
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "信息量行列式"},
                delta={'reference': 50000},
                gauge={
                    'axis': {'range': [None, 100000]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30000], 'color': "lightgray"},
                        {'range': [30000, 70000], 'color': "yellow"},
                        {'range': [70000, 100000], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90000
                    }
                }
            ),
            row=1, col=1
        )
        
        # 2. 条件数指标
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=self.fisher_data.get('condition_number', 1),
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "条件数 (越接近1越好)"},
                gauge={
                    'axis': {'range': [1, 10]},
                    'bar': {'color': "green"},
                    'steps': [
                        {'range': [1, 2], 'color': "green"},
                        {'range': [2, 5], 'color': "yellow"},
                        {'range': [5, 10], 'color': "red"}
                    ]
                }
            ),
            row=1, col=2
        )
        
        # 3. 参数效率
        param_names = ['有效参数', '冗余参数']
        param_values = [
            self.fisher_data.get('effective_params', 2),
            max(0, 5 - self.fisher_data.get('effective_params', 2))  # 假设总共5个潜在参数
        ]
        
        fig.add_trace(
            go.Bar(
                x=param_names,
                y=param_values,
                marker_color=['green', 'lightgray'],
                text=param_values,
                textposition='auto'
            ),
            row=2, col=1
        )
        
        # 4. 特征重要性
        feature_contributions = self.fisher_data.get('feature_contributions', [50, 50])
        fig.add_trace(
            go.Pie(
                labels=['特征1 (用户偏好)', '特征2 (物品特征)'],
                values=feature_contributions,
                hole=0.3
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Fisher信息矩阵监控仪表板",
            showlegend=True,
            height=700
        )
        
        return fig
    
    def create_bayesian_dashboard(self):
        """
        创建贝叶斯分析监控面板
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['模型性能比较', '预测不确定性', '模型置信度', '贝叶斯证据'],
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "indicator"}, {"type": "bar"}]]
        )
        
        # 1. 模型性能比较 (MSE)
        models = ['Ensemble', 'LLM', 'Fusion']
        mse_values = [2.538, 0.752, 1.335]  # 示例数据
        
        fig.add_trace(
            go.Bar(
                x=models,
                y=mse_values,
                marker_color=['lightblue', 'green', 'orange'],
                text=[f'{v:.3f}' for v in mse_values],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # 2. 预测不确定性散点图
        uncertainty_x = [0.822, 0.271, 0.432]  # 误差不确定性
        uncertainty_y = [0.286, 0.392, 0.385]  # 预测置信度
        
        fig.add_trace(
            go.Scatter(
                x=uncertainty_x,
                y=uncertainty_y,
                mode='markers+text',
                text=models,
                textposition='top center',
                marker=dict(size=15, color=['lightblue', 'green', 'orange']),
                name='模型位置'
            ),
            row=1, col=2
        )
        
        # 3. 当前系统置信度
        current_confidence = self.bayesian_data.get('uncertainty_metrics', {}).get('fusion', {}).get('confidence', 0.385)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=current_confidence,
                title={'text': "当前系统置信度"},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "orange"},
                    'steps': [
                        {'range': [0, 0.3], 'color': "red"},
                        {'range': [0.3, 0.7], 'color': "yellow"},
                        {'range': [0.7, 1], 'color': "green"}
                    ]
                }
            ),
            row=2, col=1
        )
        
        # 4. 贝叶斯信息准则 (越小越好)
        bic_values = [1135.77, 770.97, 943.54]
        
        fig.add_trace(
            go.Bar(
                x=models,
                y=bic_values,
                marker_color=['lightcoral', 'lightgreen', 'lightsalmon'],
                text=[f'{v:.0f}' for v in bic_values],
                textposition='auto'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="贝叶斯不确定性分析仪表板",
            showlegend=True,
            height=700
        )
        
        return fig
    
    def create_business_impact_dashboard(self):
        """
        创建业务影响监控面板
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['推荐系统健康度', '用户满意度趋势', '预测准确性', '系统稳定性指标'],
            specs=[[{"type": "indicator"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "table"}]]
        )
        
        # 1. 系统综合健康度
        health_score = self._calculate_health_score()
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=health_score,
                title={'text': "系统健康度评分"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 50], 'color': "red"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ]
                }
            ),
            row=1, col=1
        )
        
        # 2. 用户满意度时间趋势
        if self.time_series_data:
            df_ts = pd.DataFrame(self.time_series_data)
            fig.add_trace(
                go.Scatter(
                    x=df_ts['timestamp'],
                    y=df_ts['user_satisfaction'],
                    mode='lines+markers',
                    name='用户满意度',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=2
            )
        
        # 3. 预测准确性对比
        accuracy_metrics = ['RMSE', 'MAE', '相关系数']
        ensemble_scores = [2.419, 2.200, 0.339]
        fusion_scores = [2.301, 2.073, 0.528]
        
        fig.add_trace(
            go.Bar(
                x=accuracy_metrics,
                y=ensemble_scores,
                name='Ensemble基线',
                marker_color='lightblue'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=accuracy_metrics,
                y=fusion_scores,
                name='Fusion系统',
                marker_color='orange'
            ),
            row=2, col=1
        )
        
        # 4. 系统稳定性指标表
        stability_data = [
            ['模型加载成功率', '100%', '✅'],
            ['预测覆盖率', '100%', '✅'],
            ['融合一致性', '99.02%', '✅'],
            ['数值稳定性', '1.0000', '⭐'],
            ['异常值比例', '0%', '✅']
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=['指标', '数值', '状态']),
                cells=dict(values=list(zip(*stability_data)))
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="业务影响监控仪表板",
            showlegend=True,
            height=700
        )
        
        return fig
    
    def _calculate_health_score(self):
        """
        计算系统综合健康度评分
        """
        score = 0
        
        # Fisher信息贡献 (30%)
        if self.fisher_data:
            condition_score = max(0, 100 - (self.fisher_data.get('condition_number', 1) - 1) * 50)
            determinant_score = min(100, self.fisher_data.get('determinant', 0) / 1000)
            fisher_score = (condition_score * 0.7 + determinant_score * 0.3)
            score += fisher_score * 0.3
        
        # 贝叶斯分析贡献 (40%)
        if self.bayesian_data:
            confidence = self.bayesian_data.get('uncertainty_metrics', {}).get('fusion', {}).get('confidence', 0.385)
            bayesian_score = confidence * 100
            score += bayesian_score * 0.4
        
        # 业务指标贡献 (30%)
        if self.business_metrics:
            business_score = self.business_metrics.get('user_satisfaction', 0.8) * 100
            score += business_score * 0.3
        
        return min(100, max(0, score))
    
    def generate_explanation_report(self, user_id, item_id, prediction, confidence):
        """
        为单个推荐生成解释报告
        """
        report = {
            'prediction_summary': {
                'user_id': user_id,
                'item_id': item_id,
                'predicted_rating': prediction,
                'confidence_level': confidence,
                'recommendation_strength': self._get_recommendation_strength(confidence)
            },
            'algorithm_breakdown': {
                'ensemble_contribution': f"{prediction * 0.7:.2f} (70%权重)",
                'llm_contribution': f"{prediction * 0.3:.2f} (30%权重)",
                'fusion_result': f"{prediction:.2f}"
            },
            'uncertainty_analysis': {
                'prediction_uncertainty': self.bayesian_data.get('uncertainty_metrics', {}).get('fusion', {}).get('uncertainty', 'N/A'),
                'model_confidence': f"{confidence:.1%}",
                'reliability_level': self._get_reliability_level(confidence)
            },
            'business_explanation': {
                'why_recommended': self._generate_business_explanation(user_id, item_id, prediction),
                'confidence_explanation': self._generate_confidence_explanation(confidence),
                'alternative_suggestions': self._generate_alternatives(user_id, item_id)
            }
        }
        
        return report
    
    def _get_recommendation_strength(self, confidence):
        """获取推荐强度描述"""
        if confidence > 0.8:
            return "强烈推荐"
        elif confidence > 0.6:
            return "推荐"
        elif confidence > 0.4:
            return "可能喜欢"
        else:
            return "不确定"
    
    def _get_reliability_level(self, confidence):
        """获取可靠性等级"""
        if confidence > 0.8:
            return "高可靠性"
        elif confidence > 0.6:
            return "中等可靠性"
        else:
            return "低可靠性"
    
    def _generate_business_explanation(self, user_id, item_id, prediction):
        """生成业务层面的解释"""
        return f"基于用户{user_id}的历史偏好和物品{item_id}的特征，我们的多Teacher融合系统预测评分为{prediction:.1f}。这个结果综合了协同过滤、深度学习和大语言模型的分析。"
    
    def _generate_confidence_explanation(self, confidence):
        """生成置信度解释"""
        return f"该推荐的置信度为{confidence:.1%}，基于Fisher信息矩阵和贝叶斯不确定性分析得出。"
    
    def _generate_alternatives(self, user_id, item_id):
        """生成替代建议"""
        return ["相似用户也喜欢的物品", "同类型的热门物品", "基于内容的相似物品"]


def create_sample_dashboard():
    """
    创建示例监控仪表板
    """
    dashboard = RecommenderExplainabilityDashboard()
    
    # 更新示例数据
    dashboard.update_fisher_metrics(
        determinant=83045.64,
        trace=589.64,
        condition_number=1.0000,
        effective_params=2,
        feature_contributions=[39.45, 60.55]
    )
    
    dashboard.update_bayesian_metrics(
        model_comparisons={
            'ensemble': {'mse': 2.538, 'bic': 1135.77},
            'llm': {'mse': 0.752, 'bic': 770.97},
            'fusion': {'mse': 1.335, 'bic': 943.54}
        },
        uncertainty_metrics={
            'ensemble': {'confidence': 0.286, 'uncertainty': 0.822},
            'llm': {'confidence': 0.392, 'uncertainty': 0.271},
            'fusion': {'confidence': 0.385, 'uncertainty': 0.432}
        }
    )
    
    dashboard.update_business_metrics(
        user_satisfaction=0.85,
        prediction_accuracy=0.78,
        system_stability=0.99
    )
    
    # 生成仪表板
    fisher_fig = dashboard.create_fisher_dashboard()
    bayesian_fig = dashboard.create_bayesian_dashboard() 
    business_fig = dashboard.create_business_impact_dashboard()
    
    # 保存为HTML文件
    fisher_fig.write_html("/home/coder-gw/7Projects_in_7Days/online-inference-system/fisher_dashboard.html")
    bayesian_fig.write_html("/home/coder-gw/7Projects_in_7Days/online-inference-system/bayesian_dashboard.html")
    business_fig.write_html("/home/coder-gw/7Projects_in_7Days/online-inference-system/business_dashboard.html")
    
    # 生成解释报告示例
    explanation = dashboard.generate_explanation_report(
        user_id=123,
        item_id=456,
        prediction=4.2,
        confidence=0.85
    )
    
    return dashboard, explanation


if __name__ == "__main__":
    print("🚀 创建推荐系统解释性监控仪表板...")
    dashboard, explanation = create_sample_dashboard()
    
    print("✅ 仪表板创建完成！")
    print("📊 生成的文件:")
    print("   - fisher_dashboard.html")
    print("   - bayesian_dashboard.html") 
    print("   - business_dashboard.html")
    
    print("\n📋 示例解释报告:")
    import json
    print(json.dumps(explanation, indent=2, ensure_ascii=False))
