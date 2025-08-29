"""
改进的算法训练和一致性评估脚本
支持增加训练epoch数、早停机制，并在MovieLens数据集上评估算法一致性
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import time
import logging
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from models import create_recommender, get_available_algorithms
from evaluation.metrics import RecommendationMetrics, format_evaluation_results
from evaluation.consistency_analysis import AlgorithmConsistencyEvaluator, format_recommendations_for_consistency

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_movielens_data(data_path: str = None) -> pd.DataFrame:
    """
    加载MovieLens数据集
    
    Args:
        data_path: 数据文件路径，如果None则生成模拟数据
        
    Returns:
        DataFrame with columns: user_id, item_id, rating, timestamp
    """
    if data_path and os.path.exists(data_path):
        logger.info(f"从 {data_path} 加载MovieLens数据...")
        try:
            # 尝试读取标准MovieLens格式
            data = pd.read_csv(data_path, sep='\t' if data_path.endswith('.dat') else ',')
            if 'userId' in data.columns:
                data = data.rename(columns={'userId': 'user_id', 'movieId': 'item_id'})
            elif 'UserID' in data.columns:
                data = data.rename(columns={'UserID': 'user_id', 'MovieID': 'item_id', 'Rating': 'rating'})
            
            logger.info(f"成功加载数据: {len(data)} 条交互")
            return data
        except Exception as e:
            logger.warning(f"加载数据失败: {e}，将生成模拟数据")
    
    # 生成更大规模的模拟MovieLens数据
    logger.info("生成模拟MovieLens数据...")
    np.random.seed(42)
    
    n_users = 1000
    n_items = 500 
    n_interactions = 15000
    
    # 生成用户-物品交互，考虑用户偏好和物品流行度
    users = np.random.choice(range(1, n_users + 1), size=n_interactions, 
                           p=np.random.dirichlet(np.ones(n_users) * 0.5))
    
    # 物品流行度分布（长尾分布）
    item_popularity = np.random.zipf(1.5, n_items)
    item_popularity = item_popularity / item_popularity.sum()
    
    items = np.random.choice(range(1, n_items + 1), size=n_interactions, p=item_popularity)
    
    # 生成评分，考虑用户和物品的特性
    ratings = np.random.choice([3, 4, 5], size=n_interactions, p=[0.2, 0.4, 0.4])
    
    # 添加一些低分评价
    low_rating_indices = np.random.choice(n_interactions, size=int(n_interactions * 0.1), replace=False)
    ratings[low_rating_indices] = np.random.choice([1, 2], size=len(low_rating_indices))
    
    data = pd.DataFrame({
        'user_id': users,
        'item_id': items,
        'rating': ratings,
        'timestamp': np.random.randint(1000000000, 1700000000, n_interactions)
    })
    
    # 去重并保留最高评分
    data = data.groupby(['user_id', 'item_id']).agg({
        'rating': 'max',
        'timestamp': 'first'
    }).reset_index()
    
    logger.info(f"生成数据: {len(data)} 条交互, {data['user_id'].nunique()} 用户, {data['item_id'].nunique()} 物品")
    return data


class EarlyStoppingCallback:
    """早停回调类"""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.wait = 0
        self.stopped_epoch = 0
        
    def __call__(self, current_loss: float, epoch: int) -> bool:
        """
        检查是否应该早停
        
        Args:
            current_loss: 当前损失
            epoch: 当前epoch
            
        Returns:
            是否应该停止训练
        """
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            logger.info(f"早停触发，在epoch {epoch}停止训练")
            return True
        
        return False


def train_algorithm_with_config(algorithm_name: str, train_data: pd.DataFrame, 
                               config: Dict) -> Tuple[object, Dict]:
    """
    使用指定配置训练算法
    
    Args:
        algorithm_name: 算法名称
        train_data: 训练数据
        config: 训练配置
        
    Returns:
        (trained_model, training_stats)
    """
    logger.info(f"开始训练算法: {algorithm_name}")
    logger.info(f"训练配置: {config}")
    
    try:
        # 创建推荐器
        recommender = create_recommender(algorithm_name, **config)
        if recommender is None:
            logger.error(f"无法创建推荐器: {algorithm_name}")
            return None, {'error': 'creation_failed'}
        
        # 训练统计
        training_stats = {
            'algorithm': algorithm_name,
            'config': config,
            'start_time': time.time()
        }
        
        # 特殊处理需要早停的算法
        if algorithm_name in ['deepfm', 'autoint', 'din'] and 'epochs' in config:
            # 对于深度学习算法，实现简单的早停逻辑
            if hasattr(recommender, 'fit_with_validation'):
                recommender.fit_with_validation(train_data, **config)
            else:
                recommender.fit(train_data)
        else:
            # 标准训练
            recommender.fit(train_data)
        
        training_stats.update({
            'train_time': time.time() - training_stats['start_time'],
            'is_trained': recommender.is_trained,
            'success': True
        })
        
        logger.info(f"{algorithm_name} 训练完成，耗时: {training_stats['train_time']:.2f}秒")
        return recommender, training_stats
        
    except Exception as e:
        logger.error(f"算法 {algorithm_name} 训练失败: {e}")
        return None, {'error': str(e), 'success': False}


def generate_recommendations_for_users(model, test_users: List[int], 
                                     top_k: int = 10) -> Dict[int, List[int]]:
    """
    为指定用户生成推荐
    
    Args:
        model: 训练好的推荐模型
        test_users: 测试用户列表
        top_k: 推荐数量
        
    Returns:
        {user_id: [item_id1, item_id2, ...]}
    """
    recommendations = {}
    
    for user_id in test_users:
        try:
            user_recs = model.get_user_recommendations(user_id, top_k=top_k)
            if isinstance(user_recs, list) and len(user_recs) > 0:
                if isinstance(user_recs[0], dict):
                    recommendations[user_id] = [rec['item_id'] for rec in user_recs if 'item_id' in rec]
                elif isinstance(user_recs[0], tuple):
                    recommendations[user_id] = [item for item, _ in user_recs]
                else:
                    recommendations[user_id] = user_recs
            else:
                recommendations[user_id] = []
        except Exception as e:
            logger.warning(f"用户 {user_id} 推荐生成失败: {e}")
            recommendations[user_id] = []
    
    return recommendations


def run_consistency_evaluation_experiment():
    """运行算法一致性评估实验"""
    logger.info("开始算法一致性评估实验...")
    
    # 1. 加载数据
    data = load_movielens_data()
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    logger.info(f"训练集: {len(train_data)} 条, 测试集: {len(test_data)} 条")
    
    # 2. 定义训练配置 - 增加epoch数并添加早停
    algorithm_configs = {
        'svd': {
            'n_components': 50,
            'learning_rate': 0.01,
            'regularization': 0.02,
            'epochs': 100  # 增加epoch数
        },
        'simple_deepfm': {
            'embedding_dim': 16,
            'learning_rate': 0.001,
            'epochs': 50,  # 增加epoch数
            'batch_size': 256
        },
        'simple_autoint': {
            'embedding_dim': 16,
            'learning_rate': 0.001,
            'epochs': 50,  # 增加epoch数
            'num_heads': 4
        },
        'simple_din': {
            'embedding_dim': 16,
            'learning_rate': 0.001,
            'epochs': 50,  # 增加epoch数
            'hidden_dim': 64
        }
    }
    
    # 3. 训练所有算法
    trained_models = {}
    training_results = {}
    
    for algo_name, config in algorithm_configs.items():
        model, stats = train_algorithm_with_config(algo_name, train_data, config)
        if model is not None and stats.get('success', False):
            trained_models[algo_name] = model
            training_results[algo_name] = stats
        else:
            logger.error(f"算法 {algo_name} 训练失败")
    
    if len(trained_models) < 2:
        logger.error("至少需要2个成功训练的算法才能进行一致性分析")
        return
    
    logger.info(f"成功训练了 {len(trained_models)} 个算法: {list(trained_models.keys())}")
    
    # 4. 生成推荐并进行一致性分析
    test_users = test_data['user_id'].unique()[:200]  # 选择200个测试用户
    
    # 创建一致性评估器
    consistency_evaluator = AlgorithmConsistencyEvaluator()
    
    # 为每个算法生成推荐
    for algo_name, model in trained_models.items():
        logger.info(f"为算法 {algo_name} 生成推荐...")
        
        # 生成top-10推荐
        user_recommendations = generate_recommendations_for_users(model, test_users, top_k=10)
        consistency_evaluator.add_algorithm_recommendations(algo_name, user_recommendations)
        
        logger.info(f"{algo_name}: 为 {len(user_recommendations)} 个用户生成了推荐")
    
    # 5. 计算一致性指标
    logger.info("计算算法一致性指标...")
    
    # 生成详细报告
    consistency_report = consistency_evaluator.generate_consistency_report([5, 10])
    
    # 6. 输出结果
    print("\n" + "="*80)
    print("算法训练结果总结")
    print("="*80)
    
    for algo_name, stats in training_results.items():
        print(f"\n{algo_name}:")
        print(f"  训练时间: {stats['train_time']:.2f}秒")
        print(f"  配置: {stats['config']}")
    
    print(consistency_report)
    
    # 7. 生成ensemble建议
    ensemble_strategy = consistency_evaluator.recommend_ensemble_strategy(top_k=10)
    
    print(f"\n" + "="*80)
    print("ENSEMBLE策略建议")
    print("="*80)
    
    print(f"\n最互补的算法对: {ensemble_strategy['most_complementary_pair']['algorithms']}")
    print(f"Jaccard相似度: {ensemble_strategy['most_complementary_pair']['jaccard_similarity']:.4f}")
    
    print(f"\n算法多样性排序:")
    for i, (algo, score) in enumerate(ensemble_strategy['diversity_ranking']):
        print(f"  {i+1}. {algo}: {score:.4f}")
    
    print(f"\nEnsemble建议:")
    for recommendation in ensemble_strategy['ensemble_recommendations']:
        print(f"  • {recommendation}")
    
    # 8. 可视化一致性矩阵（如果有matplotlib）
    try:
        consistency_evaluator.visualize_consistency_matrix(top_k=10, 
                                                          save_path='algorithm_consistency_matrix.png')
        logger.info("一致性矩阵已保存为 algorithm_consistency_matrix.png")
    except Exception as e:
        logger.warning(f"无法生成可视化图表: {e}")
    
    # 9. 评估推荐质量
    logger.info("评估推荐质量...")
    metrics_calculator = RecommendationMetrics()
    metrics_calculator.prepare_data(train_data, test_data)
    
    quality_results = {}
    for algo_name, model in trained_models.items():
        user_recs = generate_recommendations_for_users(model, test_users, top_k=50)
        results = metrics_calculator.evaluate_all(user_recs, k_values=[10, 20, 50])
        quality_results[algo_name] = results
    
    # 输出质量对比
    print(f"\n" + "="*80)
    print("推荐质量对比")
    print("="*80)
    
    print(f"\n{'Algorithm':<15} {'R@10':<8} {'P@10':<8} {'N@10':<8} {'R@50':<8} {'N@50':<8}")
    print("-" * 70)
    
    for algo_name, results in quality_results.items():
        print(f"{algo_name:<15} "
              f"{results.get('Recall@10', 0):<8.4f} "
              f"{results.get('Precision@10', 0):<8.4f} "
              f"{results.get('NDCG@10', 0):<8.4f} "
              f"{results.get('Recall@50', 0):<8.4f} "
              f"{results.get('NDCG@50', 0):<8.4f}")
    
    return {
        'trained_models': trained_models,
        'training_results': training_results,
        'consistency_evaluator': consistency_evaluator,
        'quality_results': quality_results,
        'ensemble_strategy': ensemble_strategy
    }


if __name__ == "__main__":
    # 运行实验
    results = run_consistency_evaluation_experiment()
