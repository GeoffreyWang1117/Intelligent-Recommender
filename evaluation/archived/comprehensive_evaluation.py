"""
全面评估已训练的SOTA推荐模型
生成详细的评估报告和可视化图表
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
from typing import Dict, List, Tuple, Any
import time
import logging
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from utils.data_loader import MovieLensLoader
from evaluation.metrics import RecommendationMetrics

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ComprehensiveEvaluator:
    """全面评估器"""
    
    def __init__(self, models_dir: str = "models/saved"):
        self.models_dir = models_dir
        self.metrics_calculator = RecommendationMetrics()
        self.results = {}
        
    def load_trained_models(self) -> Dict[str, Any]:
        """加载已训练的模型"""
        models = {}
        model_files = [
            "SVD_real_movielens.pkl",
            "DeepFM_real_movielens.pkl", 
            "DCNv2_real_movielens.pkl",
            "AutoInt_real_movielens.pkl",
            "xDeepFM_real_movielens.pkl",
            "Transformer4Rec_real_movielens.pkl",
            "DIN_real_movielens.pkl"
        ]
        
        for model_file in model_files:
            model_path = os.path.join(self.models_dir, model_file)
            if os.path.exists(model_path):
                try:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    model_name = model_file.replace('_real_movielens.pkl', '')
                    models[model_name] = model
                    logger.info(f"✓ 加载模型: {model_name}")
                except Exception as e:
                    logger.error(f"✗ 加载模型失败 {model_file}: {e}")
            else:
                logger.warning(f"模型文件不存在: {model_path}")
        
        return models
    
    def generate_recommendations(self, model, model_name: str, test_users: List[int], 
                               top_k: int = 20) -> Dict[int, List[int]]:
        """为测试用户生成推荐"""
        recommendations = {}
        successful_users = 0
        
        logger.info(f"为 {len(test_users)} 个用户生成推荐...")
        
        for i, user_id in enumerate(test_users):
            if i % 50 == 0:
                logger.info(f"  进度: {i}/{len(test_users)} ({i/len(test_users)*100:.1f}%)")
            
            try:
                # 设置超时机制
                import signal
                def timeout_handler(signum, frame):
                    raise TimeoutError("推荐生成超时")
                
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(10)  # 10秒超时
                
                try:
                    user_recs = model.get_user_recommendations(user_id, top_k=top_k)
                    signal.alarm(0)  # 取消超时
                    
                    if user_recs:
                        if isinstance(user_recs[0], tuple):
                            # (item_id, score) 格式
                            rec_items = [item for item, _ in user_recs]
                        elif isinstance(user_recs[0], dict):
                            # 字典格式
                            rec_items = [rec.get('item_id', rec.get('movieId', 0)) for rec in user_recs]
                        else:
                            # 直接是item_id列表
                            rec_items = user_recs
                        
                        # 过滤无效推荐
                        rec_items = [item for item in rec_items if item is not None and item > 0]
                        recommendations[user_id] = rec_items[:top_k]
                        successful_users += 1
                    else:
                        recommendations[user_id] = []
                        
                except TimeoutError:
                    recommendations[user_id] = []
                    signal.alarm(0)
                    
            except Exception as e:
                recommendations[user_id] = []
        
        success_rate = successful_users / len(test_users) * 100
        logger.info(f"推荐生成完成: {successful_users}/{len(test_users)} ({success_rate:.1f}%) 成功")
        
        return recommendations
    
    def calculate_comprehensive_metrics(self, recommendations: Dict[int, List[int]], 
                                      model_name: str) -> Dict[str, float]:
        """计算全面的评估指标"""
        logger.info(f"计算 {model_name} 的评估指标...")
        
        metrics = {}
        k_values = [5, 10, 20]
        
        # 基础排序指标
        for k in k_values:
            try:
                recall = self.metrics_calculator.recall_at_k(recommendations, k)
                precision = self.metrics_calculator.precision_at_k(recommendations, k)
                ndcg = self.metrics_calculator.ndcg_at_k(recommendations, k=k)
                hit_rate = self.metrics_calculator.hit_rate_at_k(recommendations, k)
                
                metrics[f'Recall@{k}'] = recall
                metrics[f'Precision@{k}'] = precision
                metrics[f'NDCG@{k}'] = ndcg
                metrics[f'HitRate@{k}'] = hit_rate
                
                # F1 Score
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                    metrics[f'F1@{k}'] = f1
                else:
                    metrics[f'F1@{k}'] = 0.0
                    
            except Exception as e:
                logger.warning(f"计算K={k}指标时出错: {e}")
                for metric_name in [f'Recall@{k}', f'Precision@{k}', f'NDCG@{k}', f'HitRate@{k}', f'F1@{k}']:
                    metrics[metric_name] = 0.0
        
        # 高级指标
        try:
            metrics['MAP'] = self.metrics_calculator.mean_average_precision(recommendations)
            metrics['MRR'] = self.metrics_calculator.mean_reciprocal_rank(recommendations)
            metrics['Coverage'] = self.metrics_calculator.coverage(recommendations)
            # 简化多样性计算
            all_items = set()
            for recs in recommendations.values():
                all_items.update(recs[:10])  # 只考虑前10个推荐
            metrics['Diversity'] = len(all_items) / max(len(recommendations), 1)
            metrics['Novelty'] = self.metrics_calculator.novelty(recommendations)
        except Exception as e:
            logger.warning(f"计算高级指标时出错: {e}")
            for metric_name in ['MAP', 'MRR', 'Coverage', 'Diversity', 'Novelty']:
                metrics[metric_name] = 0.0
        
        # 推荐覆盖统计
        total_users = len(recommendations)
        users_with_recs = sum(1 for recs in recommendations.values() if len(recs) > 0)
        metrics['User_Coverage'] = users_with_recs / total_users if total_users > 0 else 0.0
        
        avg_rec_length = np.mean([len(recs) for recs in recommendations.values()])
        metrics['Avg_Rec_Length'] = avg_rec_length
        
        return metrics
    
    def evaluate_all_models(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """评估所有模型"""
        logger.info("开始全面评估所有已训练的模型...")
        
        # 准备评估数据
        self.metrics_calculator.prepare_data(train_data, test_data)
        
        # 加载模型
        models = self.load_trained_models()
        if not models:
            logger.error("没有找到可用的训练模型")
            return {}
        
        # 选择测试用户（限制数量以提高效率）
        test_users = list(test_data['user_id'].unique()[:200])  # 评估200个用户
        logger.info(f"选择 {len(test_users)} 个测试用户进行评估")
        
        results = {}
        
        for model_name, model in models.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"评估模型: {model_name}")
            logger.info(f"{'='*60}")
            
            try:
                start_time = time.time()
                
                # 生成推荐
                recommendations = self.generate_recommendations(model, model_name, test_users)
                
                # 计算指标
                metrics = self.calculate_comprehensive_metrics(recommendations, model_name)
                metrics['Evaluation_Time'] = time.time() - start_time
                
                results[model_name] = metrics
                
                logger.info(f"✓ {model_name} 评估完成")
                logger.info(f"  主要指标: Recall@10={metrics.get('Recall@10', 0):.4f}, "
                          f"Precision@10={metrics.get('Precision@10', 0):.4f}, "
                          f"NDCG@10={metrics.get('NDCG@10', 0):.4f}")
                
            except Exception as e:
                logger.error(f"✗ {model_name} 评估失败: {e}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def create_visualization_reports(self, results: Dict[str, Dict[str, float]], 
                                   output_dir: str = "evaluation_results"):
        """创建可视化报告"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 准备数据
        models = list(results.keys())
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not valid_results:
            logger.error("没有有效的评估结果可用于可视化")
            return
        
        # 1. 主要指标对比图
        self._create_main_metrics_chart(valid_results, output_dir)
        
        # 2. Recall@K 和 Precision@K 对比
        self._create_recall_precision_chart(valid_results, output_dir)
        
        # 3. 排序质量指标对比
        self._create_ranking_quality_chart(valid_results, output_dir)
        
        # 4. 覆盖率和多样性对比
        self._create_coverage_diversity_chart(valid_results, output_dir)
        
        # 5. 综合性能雷达图
        self._create_radar_chart(valid_results, output_dir)
        
        # 6. 详细数据表格
        self._create_detailed_table(valid_results, output_dir)
        
        logger.info(f"可视化报告已保存到: {output_dir}")
    
    def _create_main_metrics_chart(self, results: Dict[str, Dict[str, float]], output_dir: str):
        """创建主要指标对比图"""
        models = list(results.keys())
        metrics = ['Recall@10', 'Precision@10', 'NDCG@10', 'F1@10']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(models))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = [results[model].get(metric, 0) for model in models]
            ax.bar(x + i * width, values, width, label=metric, alpha=0.8)
        
        ax.set_xlabel('模型')
        ax.set_ylabel('指标值')
        ax.set_title('主要推荐指标对比 (K=10)', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'main_metrics_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_recall_precision_chart(self, results: Dict[str, Dict[str, float]], output_dir: str):
        """创建Recall@K和Precision@K对比图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        models = list(results.keys())
        k_values = [5, 10, 20]
        
        # Recall@K
        for model in models:
            recall_values = [results[model].get(f'Recall@{k}', 0) for k in k_values]
            ax1.plot(k_values, recall_values, marker='o', linewidth=2, label=model)
        
        ax1.set_xlabel('K值')
        ax1.set_ylabel('Recall@K')
        ax1.set_title('Recall@K 对比', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Precision@K
        for model in models:
            precision_values = [results[model].get(f'Precision@{k}', 0) for k in k_values]
            ax2.plot(k_values, precision_values, marker='s', linewidth=2, label=model)
        
        ax2.set_xlabel('K值')
        ax2.set_ylabel('Precision@K')
        ax2.set_title('Precision@K 对比', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'recall_precision_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_ranking_quality_chart(self, results: Dict[str, Dict[str, float]], output_dir: str):
        """创建排序质量指标对比图"""
        models = list(results.keys())
        metrics = ['NDCG@10', 'MAP', 'MRR']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(models))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = [results[model].get(metric, 0) for model in models]
            ax.bar(x + i * width, values, width, label=metric, alpha=0.8)
        
        ax.set_xlabel('模型')
        ax.set_ylabel('指标值')
        ax.set_title('排序质量指标对比', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ranking_quality_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_coverage_diversity_chart(self, results: Dict[str, Dict[str, float]], output_dir: str):
        """创建覆盖率和多样性对比图"""
        models = list(results.keys())
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 覆盖率指标
        coverage_metrics = ['Coverage', 'User_Coverage']
        x = np.arange(len(models))
        width = 0.35
        
        for i, metric in enumerate(coverage_metrics):
            values = [results[model].get(metric, 0) for model in models]
            ax1.bar(x + i * width, values, width, label=metric, alpha=0.8)
        
        ax1.set_xlabel('模型')
        ax1.set_ylabel('覆盖率')
        ax1.set_title('覆盖率指标对比', fontweight='bold')
        ax1.set_xticks(x + width / 2)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 多样性和新颖性
        diversity_values = [results[model].get('Diversity', 0) for model in models]
        novelty_values = [results[model].get('Novelty', 0) for model in models]
        
        ax2_twin = ax2.twinx()
        
        bars1 = ax2.bar(x - width/2, diversity_values, width, label='Diversity', alpha=0.8, color='skyblue')
        bars2 = ax2_twin.bar(x + width/2, novelty_values, width, label='Novelty', alpha=0.8, color='lightcoral')
        
        ax2.set_xlabel('模型')
        ax2.set_ylabel('多样性', color='skyblue')
        ax2_twin.set_ylabel('新颖性', color='lightcoral')
        ax2.set_title('多样性和新颖性对比', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=45, ha='right')
        
        # 添加图例
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'coverage_diversity_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_radar_chart(self, results: Dict[str, Dict[str, float]], output_dir: str):
        """创建综合性能雷达图"""
        models = list(results.keys())
        metrics = ['Recall@10', 'Precision@10', 'NDCG@10', 'Coverage', 'Diversity', 'Novelty']
        
        # 归一化指标值到0-1范围
        normalized_results = {}
        for metric in metrics:
            values = [results[model].get(metric, 0) for model in models]
            max_val = max(values) if max(values) > 0 else 1
            for i, model in enumerate(models):
                if model not in normalized_results:
                    normalized_results[model] = {}
                normalized_results[model][metric] = values[i] / max_val
        
        # 创建雷达图
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # 创建颜色列表
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink'][:len(models)]
        
        for i, model in enumerate(models):
            values = [normalized_results[model].get(metric, 0) for metric in metrics]
            values += values[:1]  # 闭合图形
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('模型综合性能雷达图', size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comprehensive_radar_chart.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_detailed_table(self, results: Dict[str, Dict[str, float]], output_dir: str):
        """创建详细数据表格"""
        # 创建DataFrame
        df_data = []
        for model, metrics in results.items():
            row = {'Model': model}
            for key, value in metrics.items():
                row[key] = value  # type: ignore
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # 保存为CSV
        csv_path = os.path.join(output_dir, 'detailed_evaluation_results.csv')
        df.to_csv(csv_path, index=False, float_format='%.4f')
        
        # 保存为JSON
        json_path = os.path.join(output_dir, 'detailed_evaluation_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"详细结果已保存: {csv_path}, {json_path}")


def main():
    """主函数"""
    logger.info("开始全面评估SOTA推荐模型...")
    
    # 1. 加载数据
    logger.info("加载MovieLens数据...")
    try:
        # 直接读取CSV格式的MovieLens small数据集
        ratings_path = "data/movielens/small/ratings.csv"
        movies_path = "data/movielens/small/movies.csv"
        
        ratings_df = pd.read_csv(ratings_path)
        movies_df = pd.read_csv(movies_path)
        
        # 重命名列以匹配我们的格式
        if 'movieId' in ratings_df.columns:
            ratings_df = ratings_df.rename(columns={'movieId': 'item_id', 'userId': 'user_id'})
        
        # 简单的训练测试分割
        from sklearn.model_selection import train_test_split
        train_data, test_data = train_test_split(ratings_df, test_size=0.2, random_state=42)
        
        logger.info(f"数据加载完成: 训练集 {len(train_data)} 条, 测试集 {len(test_data)} 条")
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        return
    
    # 2. 创建评估器
    evaluator = ComprehensiveEvaluator()
    
    # 3. 执行评估
    logger.info("开始模型评估...")
    results = evaluator.evaluate_all_models(train_data, test_data)
    
    if not results:
        logger.error("评估失败，没有获得任何结果")
        return
    
    # 4. 生成报告和图表
    logger.info("生成评估报告和可视化图表...")
    evaluator.create_visualization_reports(results)
    
    # 5. 输出总结
    logger.info("\n" + "="*80)
    logger.info("评估总结")
    logger.info("="*80)
    
    for model, metrics in results.items():
        if 'error' in metrics:
            logger.error(f"{model}: 评估失败 - {metrics['error']}")
        else:
            logger.info(f"{model}:")
            logger.info(f"  Recall@10: {metrics.get('Recall@10', 0):.4f}")
            logger.info(f"  Precision@10: {metrics.get('Precision@10', 0):.4f}")
            logger.info(f"  NDCG@10: {metrics.get('NDCG@10', 0):.4f}")
            logger.info(f"  覆盖率: {metrics.get('Coverage', 0):.4f}")
            logger.info(f"  用户覆盖率: {metrics.get('User_Coverage', 0):.4f}")
    
    logger.info("\n✅ 全面评估完成！")
    logger.info("📊 评估报告和图表已生成在 evaluation_results/ 目录")


if __name__ == "__main__":
    main()
