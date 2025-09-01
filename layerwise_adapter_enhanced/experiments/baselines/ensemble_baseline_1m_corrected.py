#!/usr/bin/env python3
"""
基于ExplainableRecommender的Ensemble基准模型 - MovieLens 1M
这是正确的ensemble基准，基于项目中真正的ensemble_recommender.py实现
"""

import sys
import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = '/home/coder-gw/7Projects_in_7Days/online-inference-system'
sys.path.insert(0, project_root)

# 导入项目中的ExplainableRecommender
from models.ensemble_recommender import ExplainableRecommender

def load_movielens_1m():
    """加载MovieLens 1M数据集"""
    print("📊 加载MovieLens 1M数据集...")
    
    # 加载评分数据
    ratings_path = '/home/coder-gw/7Projects_in_7Days/online-inference-system/data/movielens/1m/ratings.csv'
    if not os.path.exists(ratings_path):
        # 尝试另一个路径格式
        ratings_path = '/home/coder-gw/7Projects_in_7Days/online-inference-system/data/movielens/1m/ratings.dat'
        if os.path.exists(ratings_path):
            # 读取dat格式文件
            ratings_df = pd.read_csv(ratings_path, sep='::', names=['user_id', 'item_id', 'rating', 'timestamp'], engine='python')
        else:
            raise FileNotFoundError("找不到MovieLens 1M ratings文件")
    else:
        ratings_df = pd.read_csv(ratings_path)
        # 确保列名正确
        if 'userId' in ratings_df.columns:
            ratings_df = ratings_df.rename(columns={'userId': 'user_id', 'movieId': 'item_id'})
    
    print(f"📊 加载了 {len(ratings_df)} 条评分记录")
    print(f"用户数量: {ratings_df['user_id'].nunique()}")
    print(f"物品数量: {ratings_df['item_id'].nunique()}")
    print(f"评分范围: {ratings_df['rating'].min()} - {ratings_df['rating'].max()}")
    
    return ratings_df

def prepare_data_splits(ratings_df, test_size=0.2, val_size=0.1):
    """准备训练/验证/测试数据分割"""
    print("🔄 准备数据分割...")
    
    # 按时间排序（如果有timestamp）
    if 'timestamp' in ratings_df.columns:
        ratings_df = ratings_df.sort_values('timestamp')
    
    # 先分出测试集
    train_val_data, test_data = train_test_split(
        ratings_df, test_size=test_size, random_state=42, stratify=ratings_df['user_id']
    )
    
    # 再从训练数据中分出验证集
    train_data, val_data = train_test_split(
        train_val_data, test_size=val_size/(1-test_size), random_state=42, stratify=train_val_data['user_id']
    )
    
    print(f"训练集: {len(train_data)} 条")
    print(f"验证集: {len(val_data)} 条")
    print(f"测试集: {len(test_data)} 条")
    
    return train_data, val_data, test_data

def evaluate_ensemble_model(model, test_data, user_sample_size=1000):
    """评估ensemble模型性能"""
    print("📊 评估模型性能...")
    
    # 随机采样用户进行评估（避免计算量过大）
    unique_users = test_data['user_id'].unique()
    if len(unique_users) > user_sample_size:
        sampled_users = np.random.choice(unique_users, user_sample_size, replace=False)
        test_subset = test_data[test_data['user_id'].isin(sampled_users)]
        print(f"🔍 从{len(unique_users)}个用户中采样{user_sample_size}个用户进行评估")
    else:
        test_subset = test_data
        sampled_users = unique_users
        print(f"🔍 对全部{len(unique_users)}个用户进行评估")
    
    print(f"📝 测试样本数: {len(test_subset)}")
    
    # 计算RMSE和MAE
    predictions = []
    targets = []
    
    print("🔍 计算评分预测精度...")
    total_samples = len(test_subset)
    processed = 0
    
    for idx, (_, row) in enumerate(test_subset.iterrows()):
        if idx % 1000 == 0:  # 每1000个样本显示一次进度
            progress = (idx / total_samples) * 100
            print(f"   进度: {progress:.1f}% ({idx}/{total_samples})")
        
        try:
            pred = model.predict(int(row['user_id']), int(row['item_id']))
            if pred > 0:  # 有效预测
                predictions.append(pred)
                targets.append(row['rating'])
            processed += 1
        except Exception as e:
            continue
    
    if len(predictions) == 0:
        print("⚠️ 没有有效预测结果")
        return {
            'rmse': float('inf'),
            'mae': float('inf'),
            'coverage': 0.0,
            'num_predictions': 0
        }
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # 计算指标
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    mae = np.mean(np.abs(predictions - targets))
    coverage = len(predictions) / len(test_subset)
    
    print(f"✅ 评分预测完成，有效预测: {len(predictions)}/{processed}")
    
    # 计算推荐质量指标
    precision_at_10_scores = []
    ndcg_at_10_scores = []
    
    # 随机选择一些用户评估推荐质量
    eval_users = np.random.choice(sampled_users, min(100, len(sampled_users)), replace=False)
    print(f"🎯 评估推荐质量，选择{len(eval_users)}个用户...")
    
    for idx, user_id in enumerate(eval_users):
        if idx % 20 == 0:  # 每20个用户显示一次进度
            progress = (idx / len(eval_users)) * 100
            print(f"   推荐质量评估进度: {progress:.1f}% ({idx}/{len(eval_users)})")
        
        try:
            # 获取用户的真实高评分物品（≥4分）
            user_test_items = test_data[test_data['user_id'] == user_id]
            high_rated_items = set(user_test_items[user_test_items['rating'] >= 4]['item_id'].tolist())
            
            if len(high_rated_items) == 0:
                continue
            
            # 获取推荐列表
            recommendations = model.get_user_recommendations(int(user_id), top_k=10, enable_explanation=False)
            recommended_items = [rec['item_id'] for rec in recommendations]
            
            if len(recommended_items) == 0:
                continue
            
            # 计算Precision@10
            relevant_recommended = len(set(recommended_items) & high_rated_items)
            precision_at_10 = relevant_recommended / min(len(recommended_items), 10)
            precision_at_10_scores.append(precision_at_10)
            
            # 计算NDCG@10
            dcg = 0
            for i, item_id in enumerate(recommended_items[:10]):
                if item_id in high_rated_items:
                    dcg += 1 / np.log2(i + 2)
            
            # 理想DCG
            idcg = sum(1 / np.log2(i + 2) for i in range(min(len(high_rated_items), 10)))
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_at_10_scores.append(ndcg)
            
        except Exception as e:
            continue
    
    avg_precision_at_10 = np.mean(precision_at_10_scores) if precision_at_10_scores else 0.0
    avg_ndcg_at_10 = np.mean(ndcg_at_10_scores) if ndcg_at_10_scores else 0.0
    
    # 计算多样性指标（类型多样性）
    diversity_scores = []
    for user_id in eval_users:
        try:
            recommendations = model.get_user_recommendations(int(user_id), top_k=10, enable_explanation=False)
            genres = []
            for rec in recommendations:
                if 'genres' in rec:
                    genres.extend(rec['genres'])
            
            if len(genres) > 0:
                unique_genres = len(set(genres))
                diversity = unique_genres / len(genres)
                diversity_scores.append(diversity)
        except Exception:
            continue
    
    avg_diversity = np.mean(diversity_scores) if diversity_scores else 0.0
    
    metrics = {
        'rmse': float(rmse),
        'mae': float(mae),
        'coverage': float(coverage),
        'precision_at_10': float(avg_precision_at_10),
        'ndcg_at_10': float(avg_ndcg_at_10),
        'diversity': float(avg_diversity),
        'num_predictions': len(predictions),
        'num_test_samples': len(test_subset)
    }
    
    return metrics

def main():
    """主函数"""
    print("=" * 80)
    print("基准Ensemble模型训练 - MovieLens 1M数据集")
    print("基于ExplainableRecommender实现")
    print("=" * 80)
    
    # 加载数据
    ratings_df = load_movielens_1m()
    train_data, val_data, test_data = prepare_data_splits(ratings_df)
    
    # 创建算法列表（与项目中使用的一致）
    algorithms = ['svd', 'deepfm', 'autoint']  # 根据项目实际可用算法
    
    print(f"\n🤖 创建Ensemble模型，算法: {algorithms}")
    
    # 创建ExplainableRecommender
    ensemble_model = ExplainableRecommender(
        algorithm_names=algorithms,
        diversity_weight=0.1,
        popularity_penalty=0.05
    )
    
    # 训练模型
    print("\n🚀 开始训练Ensemble模型...")
    print("📝 训练过程中会显示各算法的训练状态...")
    start_time = time.time()
    
    try:
        # 设置训练超时（30分钟）
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("训练超时（30分钟）")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(1800)  # 30分钟超时
        
        ensemble_model.fit(train_data)
        
        signal.alarm(0)  # 取消超时
        training_time = time.time() - start_time
        print(f"✅ 训练完成，耗时: {training_time:.2f}秒")
        
        # 检查成功训练的模型
        successful_models = list(ensemble_model.models.keys())
        print(f"成功训练的模型: {successful_models}")
        
        if len(successful_models) == 0:
            print("❌ 没有模型训练成功")
            return
        
    except TimeoutError as e:
        print(f"⏰ {e}")
        return
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        print("详细错误信息:")
        traceback.print_exc()
        return
    
    # 评估模型
    print("\n📊 评估模型性能...")
    test_metrics = evaluate_ensemble_model(ensemble_model, test_data)
    
    # 汇总结果
    final_results = {
        'model_type': 'ExplainableRecommender Ensemble',
        'dataset': 'MovieLens-1M',
        'algorithms_used': successful_models,
        'training_time_seconds': training_time,
        'test_metrics': test_metrics,
        'model_config': {
            'target_algorithms': algorithms,
            'diversity_weight': 0.1,
            'popularity_penalty': 0.05
        }
    }
    
    # 保存结果
    results_file = '/home/coder-gw/7Projects_in_7Days/online-inference-system/layerwise_adapter_enhanced/experiments/baselines/ensemble_baseline_1m_corrected_results.json'
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    # 打印最终结果
    print("\n" + "=" * 60)
    print("🎯 最终评估结果")
    print("=" * 60)
    print(f"数据集: MovieLens 1M")
    print(f"模型: ExplainableRecommender Ensemble")
    print(f"成功训练算法: {successful_models}")
    print(f"训练时间: {training_time:.2f}秒")
    
    print("\n📈 性能指标:")
    print(f"- RMSE: {test_metrics['rmse']:.4f}")
    print(f"- MAE: {test_metrics['mae']:.4f}")
    print(f"- Precision@10: {test_metrics['precision_at_10']:.4f}")
    print(f"- NDCG@10: {test_metrics['ndcg_at_10']:.4f}")
    print(f"- Coverage: {test_metrics['coverage']:.4f}")
    print(f"- Diversity: {test_metrics['diversity']:.4f}")
    print(f"- 有效预测数: {test_metrics['num_predictions']}")
    print(f"- 测试样本数: {test_metrics['num_test_samples']}")
    
    print(f"\n💾 结果已保存到: {results_file}")
    print("✅ ExplainableRecommender基准训练完成!")

if __name__ == "__main__":
    main()
