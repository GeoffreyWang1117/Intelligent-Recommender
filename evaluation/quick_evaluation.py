"""
快速评估已训练的SOTA推荐模型
使用简化的评估流程，生成核心指标和可视化图表
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
import time
import logging
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QuickEvaluator:
    """快速评估器"""
    
    def __init__(self, models_dir: str = "models/saved"):
        self.models_dir = models_dir
        self.results = {}
        
    def load_training_results(self) -> Dict[str, Dict[str, float]]:
        """加载训练时的评估结果"""
        results_path = os.path.join(self.models_dir, "real_data_training_results.json")
        
        if not os.path.exists(results_path):
            logger.error(f"训练结果文件不存在: {results_path}")
            return {}
        
        try:
            with open(results_path, 'r', encoding='utf-8') as f:
                training_results = json.load(f)
            
            # 提取评估指标
            evaluation_results = {}
            for model_name, data in training_results.items():
                if 'evaluation' in data:
                    eval_data = data['evaluation']
                    evaluation_results[model_name] = {
                        'RMSE': eval_data.get('rmse', 0.0),
                        'MAE': eval_data.get('mae', 0.0),
                        'Coverage': eval_data.get('coverage', 0.0),
                        'Training_Time': eval_data.get('training_time', 0.0),
                        'Sample_Size': eval_data.get('sample_size', 0),
                        'Error_Count': eval_data.get('error_count', 0)
                    }
            
            logger.info(f"成功加载 {len(evaluation_results)} 个模型的训练评估结果")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"加载训练结果失败: {e}")
            return {}
    
    def calculate_derived_metrics(self, results):
        """计算衍生指标"""
        enhanced_results = {}
        
        for model_name, metrics in results.items():
            enhanced_metrics = metrics.copy()
            
            # 性能评级 (基于RMSE，越低越好)
            rmse = metrics.get('RMSE', 10.0)
            if rmse < 1.0:
                performance_grade = 'A'
                performance_score = 95
            elif rmse < 1.5:
                performance_grade = 'B'
                performance_score = 85
            elif rmse < 2.0:
                performance_grade = 'C'
                performance_score = 75
            elif rmse < 3.0:
                performance_grade = 'D'
                performance_score = 65
            else:
                performance_grade = 'F'
                performance_score = 50
            
            enhanced_metrics['Performance_Grade'] = performance_grade
            enhanced_metrics['Performance_Score'] = performance_score
            
            # 效率评级 (基于训练时间)
            training_time = metrics.get('Training_Time', 0)
            if training_time < 30:
                efficiency_grade = 'A'
            elif training_time < 60:
                efficiency_grade = 'B'
            elif training_time < 120:
                efficiency_grade = 'C'
            elif training_time < 300:
                efficiency_grade = 'D'
            else:
                efficiency_grade = 'F'
            
            enhanced_metrics['Efficiency_Grade'] = efficiency_grade
            
            # 预测准确率 (基于错误数量)
            error_count = metrics.get('Error_Count', 0)
            sample_size = metrics.get('Sample_Size', 1)
            accuracy = max(0, (sample_size - error_count) / sample_size * 100)
            enhanced_metrics['Prediction_Accuracy'] = accuracy
            
            enhanced_results[model_name] = enhanced_metrics
        
        return enhanced_results
    
    def create_performance_comparison_chart(self, results, output_dir: str = "evaluation_results"):
        """创建性能对比图"""
        os.makedirs(output_dir, exist_ok=True)
        
        models = list(results.keys())
        rmse_values = [results[model].get('RMSE', 0) for model in models]
        mae_values = [results[model].get('MAE', 0) for model in models]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # RMSE对比
        bars1 = ax1.bar(models, rmse_values, alpha=0.7, color='skyblue', edgecolor='navy')
        ax1.set_title('RMSE Performance Comparison', fontweight='bold', fontsize=14)
        ax1.set_ylabel('RMSE (Lower is Better)')
        ax1.set_xlabel('Models')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars1, rmse_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # MAE对比
        bars2 = ax2.bar(models, mae_values, alpha=0.7, color='lightcoral', edgecolor='darkred')
        ax2.set_title('MAE Performance Comparison', fontweight='bold', fontsize=14)
        ax2.set_ylabel('MAE (Lower is Better)')
        ax2.set_xlabel('Models')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars2, mae_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("性能对比图已保存")
    
    def create_training_efficiency_chart(self, results, output_dir: str = "evaluation_results"):
        """创建训练效率图"""
        models = list(results.keys())
        training_times = [results[model].get('Training_Time', 0) for model in models]
        performance_scores = [results[model].get('Performance_Score', 0) for model in models]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 训练时间对比
        bars1 = ax1.bar(models, training_times, alpha=0.7, color='lightgreen', edgecolor='darkgreen')
        ax1.set_title('Training Time Comparison', fontweight='bold', fontsize=14)
        ax1.set_ylabel('Training Time (seconds)')
        ax1.set_xlabel('Models')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars1, training_times):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        # 效率vs性能散点图
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink'][:len(models)]
        scatter = ax2.scatter(training_times, performance_scores, 
                            c=colors, s=100, alpha=0.7, edgecolors='black')
        
        for i, model in enumerate(models):
            ax2.annotate(model, (training_times[i], performance_scores[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax2.set_title('Training Efficiency vs Performance', fontweight='bold', fontsize=14)
        ax2.set_xlabel('Training Time (seconds)')
        ax2.set_ylabel('Performance Score')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_efficiency.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("训练效率图已保存")
    
    def create_model_ranking_chart(self, results, output_dir: str = "evaluation_results"):
        """创建模型排名图"""
        # 按性能评分排序
        sorted_models = sorted(results.items(), 
                             key=lambda x: x[1].get('Performance_Score', 0), 
                             reverse=True)
        
        models = [item[0] for item in sorted_models]
        performance_scores = [item[1].get('Performance_Score', 0) for item in sorted_models]
        rmse_values = [item[1].get('RMSE', 0) for item in sorted_models]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 创建双Y轴图
        ax2 = ax.twinx()
        
        # 性能评分条形图
        bars = ax.bar(models, performance_scores, alpha=0.7, color='lightblue', 
                     label='Performance Score', edgecolor='navy')
        
        # RMSE折线图
        line = ax2.plot(models, rmse_values, color='red', marker='o', linewidth=2, 
                       markersize=8, label='RMSE')
        
        # 设置标签和标题
        ax.set_title('Model Performance Ranking', fontweight='bold', fontsize=16)
        ax.set_xlabel('Models (Ranked by Performance)', fontsize=12)
        ax.set_ylabel('Performance Score', color='blue', fontsize=12)
        ax2.set_ylabel('RMSE (Lower is Better)', color='red', fontsize=12)
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        # 旋转X轴标签
        ax.tick_params(axis='x', rotation=45)
        
        # 添加图例
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='center right')
        
        # 添加排名标注
        for i, (model, score) in enumerate(zip(models, performance_scores)):
            ax.text(i, score + 1, f'#{i+1}', ha='center', va='bottom', 
                   fontweight='bold', fontsize=12, color='darkblue')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_ranking.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("模型排名图已保存")
    
    def create_comprehensive_summary_table(self, results, output_dir: str = "evaluation_results"):
        """创建综合总结表"""
        # 准备表格数据
        table_data = []
        for model, metrics in results.items():
            table_data.append({
                'Model': model,
                'Performance_Grade': metrics.get('Performance_Grade', 'N/A'),
                'RMSE': f"{metrics.get('RMSE', 0):.4f}",
                'MAE': f"{metrics.get('MAE', 0):.4f}",
                'Training_Time': f"{metrics.get('Training_Time', 0):.1f}s",
                'Efficiency_Grade': metrics.get('Efficiency_Grade', 'N/A'),
                'Prediction_Accuracy': f"{metrics.get('Prediction_Accuracy', 0):.1f}%",
                'Coverage': f"{metrics.get('Coverage', 0):.2f}"
            })
        
        # 按性能评分排序
        table_data.sort(key=lambda x: results[x['Model']].get('Performance_Score', 0), reverse=True)
        
        # 保存为CSV
        df = pd.DataFrame(table_data)
        csv_path = os.path.join(output_dir, 'model_evaluation_summary.csv')
        df.to_csv(csv_path, index=False)
        
        # 保存为JSON
        json_path = os.path.join(output_dir, 'model_evaluation_summary.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 创建HTML表格
        html_path = os.path.join(output_dir, 'model_evaluation_summary.html')
        html_content = df.to_html(index=False, table_id='evaluation_table', 
                                 classes='table table-striped table-hover')
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(f"""
<!DOCTYPE html>
<html>
<head>
    <title>SOTA推荐模型评估报告</title>
    <meta charset="utf-8">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .container {{ margin-top: 20px; }}
        .grade-A {{ color: #28a745; font-weight: bold; }}
        .grade-B {{ color: #17a2b8; font-weight: bold; }}
        .grade-C {{ color: #ffc107; font-weight: bold; }}
        .grade-D {{ color: #fd7e14; font-weight: bold; }}
        .grade-F {{ color: #dc3545; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1 class="text-center mb-4">SOTA推荐模型评估报告</h1>
        <p class="text-muted text-center">基于真实MovieLens数据的7个SOTA模型性能评估</p>
        {html_content}
        <div class="mt-4">
            <h5>评级说明：</h5>
            <ul>
                <li><span class="grade-A">A级</span>: 优秀 (RMSE < 1.0)</li>
                <li><span class="grade-B">B级</span>: 良好 (RMSE < 1.5)</li>
                <li><span class="grade-C">C级</span>: 中等 (RMSE < 2.0)</li>
                <li><span class="grade-D">D级</span>: 一般 (RMSE < 3.0)</li>
                <li><span class="grade-F">F级</span>: 较差 (RMSE ≥ 3.0)</li>
            </ul>
        </div>
    </div>
</body>
</html>
            """)
        
        logger.info(f"综合总结已保存: {csv_path}, {json_path}, {html_path}")
    
    def generate_quick_report(self, output_dir: str = "evaluation_results"):
        """生成快速评估报告"""
        logger.info("开始生成快速评估报告...")
        
        # 加载训练结果
        results = self.load_training_results()
        if not results:
            logger.error("无法加载训练结果，无法生成报告")
            return
        
        # 计算衍生指标
        enhanced_results = self.calculate_derived_metrics(results)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成各种图表
        self.create_performance_comparison_chart(enhanced_results, output_dir)
        self.create_training_efficiency_chart(enhanced_results, output_dir)
        self.create_model_ranking_chart(enhanced_results, output_dir)
        self.create_comprehensive_summary_table(enhanced_results, output_dir)
        
        # 输出总结
        logger.info("\n" + "="*80)
        logger.info("快速评估报告总结")
        logger.info("="*80)
        
        # 按性能排序
        sorted_models = sorted(enhanced_results.items(), 
                             key=lambda x: x[1].get('Performance_Score', 0), 
                             reverse=True)
        
        for i, (model, metrics) in enumerate(sorted_models):
            logger.info(f"#{i+1} {model}:")
            logger.info(f"  性能等级: {metrics.get('Performance_Grade', 'N/A')}")
            logger.info(f"  RMSE: {metrics.get('RMSE', 0):.4f}")
            logger.info(f"  MAE: {metrics.get('MAE', 0):.4f}")
            logger.info(f"  训练时间: {metrics.get('Training_Time', 0):.1f}秒")
            logger.info(f"  效率等级: {metrics.get('Efficiency_Grade', 'N/A')}")
            logger.info(f"  预测准确率: {metrics.get('Prediction_Accuracy', 0):.1f}%")
        
        logger.info(f"\n✅ 快速评估报告生成完成!")
        logger.info(f"📊 报告文件已保存到: {output_dir}")
        logger.info(f"📈 包含性能对比图、效率分析图、排名图和详细数据表")


def main():
    """主函数"""
    logger.info("开始快速评估SOTA推荐模型...")
    
    evaluator = QuickEvaluator()
    evaluator.generate_quick_report()


if __name__ == "__main__":
    main()
