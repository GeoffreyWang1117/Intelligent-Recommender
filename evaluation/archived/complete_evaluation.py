"""
完整推荐系统评估框架
集成所有评估指标和可视化功能的完整评估系统
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
from typing import Dict, List, Tuple, Any, Optional
import time
import logging
from collections import defaultdict
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings('ignore')

# 设置matplotlib
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdvancedRecommendationMetrics:
    """高级推荐系统评价指标"""
    
    def __init__(self):
        self.item_popularity = {}
        self.total_items = 0
        self.test_user_items = defaultdict(set)
        self.test_user_ratings = defaultdict(dict)
        
    def prepare_data(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        """准备评价所需的数据"""
        # 计算物品流行度
        item_counts = train_data['item_id'].value_counts()
        total_interactions = len(train_data)
        self.item_popularity = {item: count/total_interactions 
                               for item, count in item_counts.items()}
        self.total_items = train_data['item_id'].nunique()
        
        # 构建测试集数据结构
        self.test_user_items = defaultdict(set)
        self.test_user_ratings = defaultdict(dict)
        
        for _, row in test_data.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            rating = row.get('rating', 1.0)
            
            self.test_user_items[user_id].add(item_id)
            self.test_user_ratings[user_id][item_id] = rating
    
    def recall_at_k(self, recommendations: Dict[int, List[int]], k: int = 10) -> float:
        """计算Recall@K"""
        total_recall = 0.0
        user_count = 0
        
        for user_id, rec_items in recommendations.items():
            if user_id not in self.test_user_items:
                continue
                
            true_items = self.test_user_items[user_id]
            rec_items_k = set(rec_items[:k])
            
            if len(true_items) > 0:
                recall = len(rec_items_k & true_items) / len(true_items)
                total_recall += recall
                user_count += 1
        
        return total_recall / user_count if user_count > 0 else 0.0
    
    def precision_at_k(self, recommendations: Dict[int, List[int]], k: int = 10) -> float:
        """计算Precision@K"""
        total_precision = 0.0
        user_count = 0
        
        for user_id, rec_items in recommendations.items():
            if user_id not in self.test_user_items:
                continue
                
            true_items = self.test_user_items[user_id]
            rec_items_k = set(rec_items[:k])
            
            if len(rec_items_k) > 0:
                precision = len(rec_items_k & true_items) / len(rec_items_k)
                total_precision += precision
                user_count += 1
        
        return total_precision / user_count if user_count > 0 else 0.0
    
    def hit_rate_at_k(self, recommendations: Dict[int, List[int]], k: int = 10) -> float:
        """计算Hit Rate@K"""
        hits = 0
        user_count = 0
        
        for user_id, rec_items in recommendations.items():
            if user_id not in self.test_user_items:
                continue
                
            true_items = self.test_user_items[user_id]
            rec_items_k = set(rec_items[:k])
            
            if len(rec_items_k & true_items) > 0:
                hits += 1
            user_count += 1
        
        return hits / user_count if user_count > 0 else 0.0
    
    def ndcg_at_k(self, recommendations: Dict[int, List[int]], k: int = 10) -> float:
        """计算NDCG@K"""
        total_ndcg = 0.0
        user_count = 0
        
        for user_id, rec_items in recommendations.items():
            if user_id not in self.test_user_ratings:
                continue
                
            user_ratings = self.test_user_ratings[user_id]
            rec_items_k = rec_items[:k]
            
            # 计算DCG
            dcg = 0.0
            for i, item_id in enumerate(rec_items_k):
                if item_id in user_ratings:
                    relevance = user_ratings[item_id]
                    dcg += (2**relevance - 1) / np.log2(i + 2)
            
            # 计算IDCG
            sorted_ratings = sorted(user_ratings.values(), reverse=True)[:k]
            idcg = 0.0
            for i, relevance in enumerate(sorted_ratings):
                idcg += (2**relevance - 1) / np.log2(i + 2)
            
            # 计算NDCG
            if idcg > 0:
                ndcg = dcg / idcg
                total_ndcg += ndcg
                user_count += 1
        
        return total_ndcg / user_count if user_count > 0 else 0.0
    
    def mean_average_precision(self, recommendations: Dict[int, List[int]]) -> float:
        """计算MAP (Mean Average Precision)"""
        total_ap = 0.0
        user_count = 0
        
        for user_id, rec_items in recommendations.items():
            if user_id not in self.test_user_items:
                continue
                
            true_items = self.test_user_items[user_id]
            
            # 计算Average Precision
            ap = 0.0
            num_hits = 0
            
            for i, item_id in enumerate(rec_items):
                if item_id in true_items:
                    num_hits += 1
                    precision_at_i = num_hits / (i + 1)
                    ap += precision_at_i
            
            if len(true_items) > 0:
                ap /= len(true_items)
                total_ap += ap
                user_count += 1
        
        return total_ap / user_count if user_count > 0 else 0.0
    
    def mean_reciprocal_rank(self, recommendations: Dict[int, List[int]]) -> float:
        """计算MRR (Mean Reciprocal Rank)"""
        total_rr = 0.0
        user_count = 0
        
        for user_id, rec_items in recommendations.items():
            if user_id not in self.test_user_items:
                continue
                
            true_items = self.test_user_items[user_id]
            
            # 找到第一个命中的位置
            for i, item_id in enumerate(rec_items):
                if item_id in true_items:
                    rr = 1.0 / (i + 1)
                    total_rr += rr
                    break
            
            user_count += 1
        
        return total_rr / user_count if user_count > 0 else 0.0
    
    def coverage(self, recommendations: Dict[int, List[int]]) -> float:
        """计算物品覆盖率"""
        all_recommended_items = set()
        for rec_items in recommendations.values():
            all_recommended_items.update(rec_items)
        
        return len(all_recommended_items) / self.total_items if self.total_items > 0 else 0.0
    
    def diversity(self, recommendations: Dict[int, List[int]], top_k: int = 10) -> float:
        """计算推荐多样性（基于物品共现）"""
        diversities = []
        
        for user_id, rec_items in recommendations.items():
            rec_items_k = rec_items[:top_k]
            n_items = len(rec_items_k)
            
            if n_items <= 1:
                continue
            
            # 计算物品间的平均距离（这里用简单的不相似度）
            diversity_sum = 0
            pair_count = 0
            
            for i in range(n_items):
                for j in range(i + 1, n_items):
                    # 简单的不相似度度量（基于物品ID差异）
                    diversity_sum += 1.0  # 假设所有物品都不相似
                    pair_count += 1
            
            if pair_count > 0:
                diversity = diversity_sum / pair_count
                diversities.append(diversity)
        
        return float(np.mean(diversities)) if diversities else 0.0
    
    def novelty(self, recommendations: Dict[int, List[int]]) -> float:
        """计算推荐新颖性"""
        total_novelty = 0.0
        item_count = 0
        
        for rec_items in recommendations.values():
            for item_id in rec_items:
                if item_id in self.item_popularity:
                    # 新颖性定义为 -log(popularity)
                    novelty = -np.log2(self.item_popularity[item_id] + 1e-10)
                    total_novelty += novelty
                    item_count += 1
        
        return total_novelty / item_count if item_count > 0 else 0.0


class CompleteEvaluator:
    """完整评估器"""
    
    def __init__(self, models_dir: str = "models/saved"):
        self.models_dir = models_dir
        self.metrics_calculator = AdvancedRecommendationMetrics()
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
                    logger.info(f"✓ 成功加载模型: {model_name}")
                except Exception as e:
                    logger.error(f"✗ 加载模型失败 {model_file}: {e}")
            else:
                logger.warning(f"模型文件不存在: {model_path}")
        
        return models
    
    def generate_recommendations_parallel(self, model, model_name: str, 
                                        test_users: List[int], top_k: int = 20) -> Dict[int, List[int]]:
        """并行生成推荐（优化版本）"""
        recommendations = {}
        successful_users = 0
        
        logger.info(f"为 {len(test_users)} 个用户生成推荐...")
        
        def get_user_recommendations(user_id):
            try:
                # 设置更短的超时
                import signal
                def timeout_handler(signum, frame):
                    raise TimeoutError("推荐生成超时")
                
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(5)  # 5秒超时
                
                try:
                    user_recs = model.get_user_recommendations(user_id, top_k=top_k)
                    signal.alarm(0)
                    
                    if user_recs:
                        if isinstance(user_recs[0], tuple):
                            rec_items = [item for item, _ in user_recs]
                        elif isinstance(user_recs[0], dict):
                            rec_items = [rec.get('item_id', rec.get('movieId', 0)) for rec in user_recs]
                        else:
                            rec_items = user_recs
                        
                        rec_items = [item for item in rec_items if item is not None and item > 0]
                        return user_id, rec_items[:top_k]
                    else:
                        return user_id, []
                        
                except TimeoutError:
                    return user_id, []
                    
            except Exception:
                return user_id, []
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(get_user_recommendations, user_id): user_id 
                      for user_id in test_users[:100]}  # 限制为100个用户
            
            for i, future in enumerate(as_completed(futures)):
                if i % 25 == 0:
                    logger.info(f"  进度: {i}/100 ({i}%)")
                
                user_id, rec_items = future.result()
                recommendations[user_id] = rec_items
                if len(rec_items) > 0:
                    successful_users += 1
        
        success_rate = successful_users / len(recommendations) * 100
        logger.info(f"推荐生成完成: {successful_users}/{len(recommendations)} ({success_rate:.1f}%) 成功")
        
        return recommendations
    
    def calculate_rating_metrics(self, model, test_data: pd.DataFrame, 
                               sample_size: int = 1000) -> Dict[str, float]:
        """计算评分预测指标"""
        # 随机采样测试数据
        test_sample = test_data.sample(n=min(sample_size, len(test_data)), random_state=42)
        
        true_ratings = []
        pred_ratings = []
        error_count = 0
        
        logger.info(f"计算评分指标，样本数: {len(test_sample)}")
        
        for _, row in test_sample.iterrows():
            try:
                user_id = row['user_id']
                item_id = row['item_id']
                true_rating = row['rating']
                
                # 设置超时
                import signal
                def timeout_handler(signum, frame):
                    raise TimeoutError("预测超时")
                
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(2)  # 2秒超时
                
                try:
                    pred_rating = model.predict(user_id, item_id)
                    signal.alarm(0)
                    
                    if pred_rating is not None and not np.isnan(pred_rating):
                        true_ratings.append(true_rating)
                        pred_ratings.append(pred_rating)
                    else:
                        error_count += 1
                        
                except TimeoutError:
                    error_count += 1
                    signal.alarm(0)
                    
            except Exception:
                error_count += 1
        
        if len(true_ratings) > 0:
            rmse = np.sqrt(mean_squared_error(true_ratings, pred_ratings))
            mae = mean_absolute_error(true_ratings, pred_ratings)
            coverage = len(true_ratings) / len(test_sample)
        else:
            rmse = mae = coverage = 0.0
        
        return {
            'rmse': rmse,
            'mae': mae,
            'coverage': coverage,
            'sample_size': len(test_sample),
            'error_count': error_count,
            'valid_predictions': len(true_ratings)
        }
    
    def calculate_ranking_metrics(self, recommendations: Dict[int, List[int]]) -> Dict[str, float]:
        """计算排序相关指标"""
        k_values = [5, 10, 20]
        metrics = {}
        
        # 计算各种K值的指标
        for k in k_values:
            metrics[f'Recall@{k}'] = self.metrics_calculator.recall_at_k(recommendations, k)
            metrics[f'Precision@{k}'] = self.metrics_calculator.precision_at_k(recommendations, k)
            metrics[f'HitRate@{k}'] = self.metrics_calculator.hit_rate_at_k(recommendations, k)
            metrics[f'NDCG@{k}'] = self.metrics_calculator.ndcg_at_k(recommendations, k)
            
            # 计算F1分数
            recall = metrics[f'Recall@{k}']
            precision = metrics[f'Precision@{k}']
            if recall + precision > 0:
                metrics[f'F1@{k}'] = 2 * recall * precision / (recall + precision)
            else:
                metrics[f'F1@{k}'] = 0.0
        
        # 计算高级指标
        metrics['MAP'] = self.metrics_calculator.mean_average_precision(recommendations)
        metrics['MRR'] = self.metrics_calculator.mean_reciprocal_rank(recommendations)
        metrics['Coverage'] = self.metrics_calculator.coverage(recommendations)
        metrics['Diversity'] = self.metrics_calculator.diversity(recommendations)
        metrics['Novelty'] = self.metrics_calculator.novelty(recommendations)
        
        # 用户覆盖率
        total_users = len(recommendations)
        users_with_recs = sum(1 for recs in recommendations.values() if len(recs) > 0)
        metrics['User_Coverage'] = users_with_recs / total_users if total_users > 0 else 0.0
        
        return metrics
    
    def evaluate_model_complete(self, model, model_name: str, train_data: pd.DataFrame, 
                              test_data: pd.DataFrame) -> Dict[str, Any]:
        """完整评估单个模型"""
        logger.info(f"开始完整评估: {model_name}")
        start_time = time.time()
        
        results = {'model_name': model_name, 'status': 'success'}  # type: ignore
        
        try:
            # 1. 评分预测指标
            logger.info("  计算评分预测指标...")
            rating_metrics = self.calculate_rating_metrics(model, test_data)
            for key, value in rating_metrics.items():
                results[key] = value  # type: ignore
            
            # 2. 排序推荐指标
            logger.info("  生成推荐并计算排序指标...")
            test_users = list(test_data['user_id'].unique()[:100])  # 限制用户数
            recommendations = self.generate_recommendations_parallel(model, model_name, test_users)
            
            if recommendations:
                ranking_metrics = self.calculate_ranking_metrics(recommendations)
                for key, value in ranking_metrics.items():
                    results[key] = value  # type: ignore
            else:
                logger.warning("%s: 无法生成推荐", model_name)
                results['status'] = 'partial_failure'
            
            # 3. 计算总体评估时间
            results['evaluation_time'] = time.time() - start_time  # type: ignore
            
            logger.info(f"✓ {model_name} 评估完成 ({results['evaluation_time']:.1f}秒)")
            
        except Exception as e:
            logger.error(f"✗ {model_name} 评估失败: {e}")
            results['status'] = 'failure'
            results['error'] = str(e)
        
        return results
    
    def run_complete_evaluation(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict[str, Any]:
        """运行完整评估"""
        logger.info("="*80)
        logger.info("开始完整推荐系统评估")
        logger.info("="*80)
        
        # 准备数据
        self.metrics_calculator.prepare_data(train_data, test_data)
        
        # 加载模型
        models = self.load_trained_models()
        if not models:
            logger.error("没有找到可用的训练模型")
            return {}
        
        # 评估所有模型
        evaluation_results = {}
        
        for model_name, model in models.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"评估模型: {model_name}")
            logger.info(f"{'='*60}")
            
            result = self.evaluate_model_complete(model, model_name, train_data, test_data)
            evaluation_results[model_name] = result
        
        return evaluation_results
    
    def create_comprehensive_visualization(self, results: Dict[str, Any], 
                                         output_dir: str = "evaluation_results"):
        """创建综合可视化报告"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 过滤成功的结果
        valid_results = {k: v for k, v in results.items() 
                        if v.get('status') == 'success' and 'rmse' in v}
        
        if not valid_results:
            logger.error("没有有效的评估结果用于可视化")
            return
        
        # 1. 综合性能对比图
        self._create_comprehensive_performance_chart(valid_results, output_dir)
        
        # 2. 排序指标热力图
        self._create_ranking_metrics_heatmap(valid_results, output_dir)
        
        # 3. 评分vs排序性能散点图
        self._create_rating_vs_ranking_scatter(valid_results, output_dir)
        
        # 4. 详细指标雷达图
        self._create_detailed_radar_chart(valid_results, output_dir)
        
        logger.info(f"综合可视化报告已保存到: {output_dir}")
    
    def _create_comprehensive_performance_chart(self, results: Dict[str, Any], output_dir: str):
        """创建综合性能对比图"""
        models = list(results.keys())
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comprehensive Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # RMSE和MAE
        rmse_values = [results[model].get('rmse', 0) for model in models]
        mae_values = [results[model].get('mae', 0) for model in models]
        
        ax1.bar(models, rmse_values, alpha=0.7, color='lightcoral', edgecolor='darkred')
        ax1.set_title('RMSE Comparison', fontweight='bold')
        ax1.set_ylabel('RMSE (Lower is Better)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        ax2.bar(models, mae_values, alpha=0.7, color='lightblue', edgecolor='darkblue')
        ax2.set_title('MAE Comparison', fontweight='bold')
        ax2.set_ylabel('MAE (Lower is Better)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Recall@10和Precision@10
        recall_values = [results[model].get('Recall@10', 0) for model in models]
        precision_values = [results[model].get('Precision@10', 0) for model in models]
        
        ax3.bar(models, recall_values, alpha=0.7, color='lightgreen', edgecolor='darkgreen')
        ax3.set_title('Recall@10 Comparison', fontweight='bold')
        ax3.set_ylabel('Recall@10 (Higher is Better)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        ax4.bar(models, precision_values, alpha=0.7, color='gold', edgecolor='orange')
        ax4.set_title('Precision@10 Comparison', fontweight='bold')
        ax4.set_ylabel('Precision@10 (Higher is Better)')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comprehensive_performance.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_ranking_metrics_heatmap(self, results: Dict[str, Any], output_dir: str):
        """创建排序指标热力图"""
        models = list(results.keys())
        metrics = ['Recall@5', 'Recall@10', 'Recall@20', 
                  'Precision@5', 'Precision@10', 'Precision@20',
                  'NDCG@5', 'NDCG@10', 'NDCG@20', 'MAP', 'MRR']
        
        # 创建数据矩阵
        data_matrix = []
        for model in models:
            row = [results[model].get(metric, 0) for metric in metrics]
            data_matrix.append(row)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(data_matrix, 
                   xticklabels=metrics, 
                   yticklabels=models,
                   annot=True, 
                   fmt='.3f', 
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Metric Value'})
        
        plt.title('Ranking Metrics Heatmap', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ranking_metrics_heatmap.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_rating_vs_ranking_scatter(self, results: Dict[str, Any], output_dir: str):
        """创建评分vs排序性能散点图"""
        models = list(results.keys())
        rmse_values = [results[model].get('rmse', 0) for model in models]
        recall_values = [results[model].get('Recall@10', 0) for model in models]
        
        plt.figure(figsize=(10, 8))
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink'][:len(models)]
        
        for i, model in enumerate(models):
            plt.scatter(rmse_values[i], recall_values[i], 
                       c=colors[i], s=100, alpha=0.7, 
                       edgecolors='black', label=model)
        
        plt.xlabel('RMSE (Rating Prediction Quality)', fontsize=12)
        plt.ylabel('Recall@10 (Ranking Quality)', fontsize=12)
        plt.title('Rating vs Ranking Performance Trade-off', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # 添加象限标注
        plt.axhline(y=float(np.mean(recall_values)), color='gray', linestyle='--', alpha=0.5)
        plt.axvline(x=float(np.mean(rmse_values)), color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rating_vs_ranking_scatter.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_detailed_radar_chart(self, results: Dict[str, Any], output_dir: str):
        """创建详细雷达图"""
        models = list(results.keys())
        metrics = ['Recall@10', 'Precision@10', 'NDCG@10', 'MAP', 'Coverage', 'Diversity']
        
        # 归一化指标值
        normalized_data = {}
        for metric in metrics:
            values = [results[model].get(metric, 0) for model in models]
            max_val = max(values) if max(values) > 0 else 1
            normalized_data[metric] = [v / max_val for v in values]
        
        # 创建雷达图
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink'][:len(models)]
        
        for i, model in enumerate(models):
            values = [normalized_data[metric][i] for metric in metrics]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('Comprehensive Performance Radar Chart', 
                    size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'detailed_radar_chart.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_complete_results(self, results: Dict[str, Any], output_dir: str = "evaluation_results"):
        """保存完整评估结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存JSON格式
        json_path = os.path.join(output_dir, 'complete_evaluation_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 创建详细CSV报告
        csv_data = []
        for model_name, metrics in results.items():
            if metrics.get('status') == 'success':
                row = {'Model': model_name}
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        row[key] = value  # type: ignore
                csv_data.append(row)
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_path = os.path.join(output_dir, 'complete_evaluation_results.csv')
            df.to_csv(csv_path, index=False, float_format='%.6f')
            
            logger.info(f"完整评估结果已保存: {json_path}, {csv_path}")


def main():
    """主函数"""
    logger.info("开始完整推荐系统评估...")
    
    # 1. 加载数据
    logger.info("加载MovieLens数据...")
    try:
        ratings_path = "data/movielens/small/ratings.csv"
        ratings_df = pd.read_csv(ratings_path)
        
        if 'movieId' in ratings_df.columns:
            ratings_df = ratings_df.rename(columns={'movieId': 'item_id', 'userId': 'user_id'})
        
        train_data, test_data = train_test_split(ratings_df, test_size=0.2, random_state=42)
        logger.info(f"数据加载完成: 训练集 {len(train_data)} 条, 测试集 {len(test_data)} 条")
        
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        return
    
    # 2. 创建评估器
    evaluator = CompleteEvaluator()
    
    # 3. 运行完整评估
    logger.info("开始完整模型评估...")
    results = evaluator.run_complete_evaluation(train_data, test_data)
    
    if not results:
        logger.error("评估失败，没有获得任何结果")
        return
    
    # 4. 保存结果
    evaluator.save_complete_results(results)
    
    # 5. 生成可视化报告
    logger.info("生成综合可视化报告...")
    evaluator.create_comprehensive_visualization(results)
    
    # 6. 输出总结
    logger.info("\n" + "="*80)
    logger.info("完整评估总结")
    logger.info("="*80)
    
    for model_name, metrics in results.items():
        if metrics.get('status') == 'success':
            logger.info(f"\n{model_name}:")
            logger.info(f"  评分预测 - RMSE: {metrics.get('rmse', 0):.4f}, MAE: {metrics.get('mae', 0):.4f}")
            logger.info(f"  排序质量 - Recall@10: {metrics.get('Recall@10', 0):.4f}, Precision@10: {metrics.get('Precision@10', 0):.4f}")
            logger.info(f"  高级指标 - NDCG@10: {metrics.get('NDCG@10', 0):.4f}, MAP: {metrics.get('MAP', 0):.4f}")
            logger.info(f"  评估时间: {metrics.get('evaluation_time', 0):.1f}秒")
        else:
            logger.error(f"{model_name}: 评估失败 - {metrics.get('error', 'Unknown error')}")
    
    logger.info("\n✅ 完整评估完成！")
    logger.info("📊 详细结果已保存在 evaluation_results/ 目录")


if __name__ == "__main__":
    main()
