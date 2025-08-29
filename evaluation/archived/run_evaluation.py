"""
推荐系统算法评价测试脚本
用于测试各种推荐算法的性能指标
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from typing import Dict, List
import time
import logging
from sklearn.model_selection import train_test_split

from models import create_recommender, get_available_algorithms
from evaluation.metrics import RecommendationMetrics, format_evaluation_results

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_sample_data(n_users: int = 1000, n_items: int = 500, 
                        n_interactions: int = 10000) -> pd.DataFrame:
    """生成示例评测数据"""
    np.random.seed(42)
    
    # 生成用户-物品交互数据
    users = np.random.randint(1, n_users + 1, n_interactions)
    items = np.random.randint(1, n_items + 1, n_interactions)
    
    # 生成评分，偏向高分
    ratings = np.random.choice([3, 4, 5], n_interactions, p=[0.2, 0.3, 0.5])
    
    data = pd.DataFrame({
        'user_id': users,
        'item_id': items,
        'rating': ratings,
        'timestamp': np.random.randint(1000000000, 1700000000, n_interactions)
    })
    
    # 去重
    data = data.drop_duplicates(subset=['user_id', 'item_id']).reset_index(drop=True)
    
    logger.info(f"生成数据: {len(data)} 条交互, {data['user_id'].nunique()} 用户, {data['item_id'].nunique()} 物品")
    return data


def evaluate_algorithm(algorithm_name: str, train_data: pd.DataFrame, 
                      test_data: pd.DataFrame, metrics_calculator: RecommendationMetrics,
                      k_values: List[int] = [10, 20, 50]) -> Dict[str, float]:
    """评价单个算法"""
    try:
        logger.info(f"开始评价算法: {algorithm_name}")
        
        # 创建推荐器
        start_time = time.time()
        recommender = create_recommender(algorithm_name)
        if recommender is None:
            logger.error(f"无法创建推荐器: {algorithm_name}")
            return {}
        
        # 训练模型
        logger.info(f"训练模型: {algorithm_name}")
        train_start = time.time()
        
        try:
            recommender.fit(train_data)
            train_time = time.time() - train_start
            logger.info(f"训练完成，耗时: {train_time:.2f}秒")
        except Exception as e:
            logger.error(f"训练失败: {algorithm_name}, 错误: {e}")
            return {'train_error': str(e)}
        
        # 生成推荐
        logger.info(f"生成推荐: {algorithm_name}")
        recommendations = {}
        test_users = test_data['user_id'].unique()[:100]  # 限制测试用户数量
        
        pred_start = time.time()
        for user_id in test_users:
            try:
                user_recs = recommender.get_user_recommendations(user_id, top_k=max(k_values))
                if isinstance(user_recs, list) and len(user_recs) > 0:
                    if isinstance(user_recs[0], tuple):
                        # 如果返回的是(item_id, score)元组，提取item_id
                        recommendations[user_id] = [item for item, _ in user_recs]
                    elif isinstance(user_recs[0], dict):
                        # 如果返回的是字典格式，提取item_id
                        recommendations[user_id] = [rec['item_id'] for rec in user_recs if 'item_id' in rec]
                    else:
                        recommendations[user_id] = user_recs
                else:
                    recommendations[user_id] = []
            except Exception as e:
                logger.warning(f"用户 {user_id} 推荐生成失败: {e}")
                recommendations[user_id] = []
        
        pred_time = time.time() - pred_start
        logger.info(f"推荐生成完成，耗时: {pred_time:.2f}秒")
        
        # 计算评价指标
        if recommendations:
            results = metrics_calculator.evaluate_all(recommendations, k_values=k_values)
            results['train_time'] = train_time
            results['prediction_time'] = pred_time
            results['total_time'] = time.time() - start_time
            
            logger.info(f"算法 {algorithm_name} 评价完成")
            return results
        else:
            logger.error(f"算法 {algorithm_name} 未生成有效推荐")
            return {'error': 'no_recommendations'}
            
    except Exception as e:
        logger.error(f"算法 {algorithm_name} 评价失败: {e}")
        return {'error': str(e)}


def run_comprehensive_evaluation():
    """运行综合评价"""
    logger.info("开始推荐系统算法综合评价")
    
    # 生成测试数据
    logger.info("生成测试数据...")
    data = generate_sample_data(n_users=500, n_items=300, n_interactions=5000)
    
    # 分割训练集和测试集
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    logger.info(f"训练集: {len(train_data)} 条, 测试集: {len(test_data)} 条")
    
    # 初始化评价指标计算器
    metrics_calculator = RecommendationMetrics()
    metrics_calculator.prepare_data(train_data, test_data)
    
    # 获取所有可用算法
    algorithms = get_available_algorithms()
    logger.info(f"可用算法: {algorithms}")
    
    # 要测试的算法（选择一些关键算法）
    test_algorithms = ['svd', 'simple_deepfm', 'simple_autoint', 'simple_din']
    test_algorithms = [alg for alg in test_algorithms if alg in algorithms]
    
    logger.info(f"将测试算法: {test_algorithms}")
    
    # 评价结果
    all_results = {}
    k_values = [10, 20, 50]
    
    for algorithm in test_algorithms:
        logger.info(f"\n{'='*50}")
        logger.info(f"评价算法: {algorithm}")
        logger.info(f"{'='*50}")
        
        results = evaluate_algorithm(algorithm, train_data, test_data, 
                                   metrics_calculator, k_values)
        all_results[algorithm] = results
        
        if 'error' not in results:
            print(format_evaluation_results(results, algorithm))
        else:
            logger.error(f"算法 {algorithm} 评价失败: {results.get('error', 'unknown')}")
    
    # 生成汇总报告
    logger.info(f"\n{'='*60}")
    logger.info("评价汇总报告")
    logger.info(f"{'='*60}")
    
    # 创建汇总表格
    summary_metrics = ['Recall@10', 'Precision@10', 'NDCG@10', 'MAP@10', 
                      'HitRate@10', 'Recall@50', 'NDCG@50', 'train_time']
    
    print(f"\n{'Algorithm':<15}", end="")
    for metric in summary_metrics:
        print(f"{metric:<12}", end="")
    print()
    print("-" * (15 + 12 * len(summary_metrics)))
    
    for algorithm, results in all_results.items():
        if 'error' not in results:
            print(f"{algorithm:<15}", end="")
            for metric in summary_metrics:
                value = results.get(metric, 0.0)
                if 'time' in metric:
                    print(f"{value:<12.2f}", end="")
                else:
                    print(f"{value:<12.4f}", end="")
            print()
        else:
            print(f"{algorithm:<15}ERROR: {results.get('error', 'unknown')}")
    
    return all_results


if __name__ == "__main__":
    results = run_comprehensive_evaluation()
