#!/usr/bin/env python3
"""
修正版本的Ensemble基准模型训练
解决数据列名不匹配问题，并简化评估流程
"""

import sys
import os
sys.path.append('/home/coder-gw/7Projects_in_7Days/online-inference-system')

import pandas as pd
import numpy as np
import json
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 导入推荐系统模型
from models.ensemble_recommender import ExplainableRecommender

def load_and_prepare_data():
    """加载并准备MovieLens 1M数据"""
    print("📁 加载MovieLens 1M数据...")
    
    ratings_file = '/home/coder-gw/7Projects_in_7Days/online-inference-system/data/movielens/1m/ratings.csv'
    
    if not os.path.exists(ratings_file):
        raise FileNotFoundError(f"数据文件不存在: {ratings_file}")
    
    # 加载数据
    df = pd.read_csv(ratings_file)
    print(f"✅ 原始数据: {len(df)} 条记录")
    
    # 🔧 关键修复：统一列名
    df = df.rename(columns={
        'userId': 'user_id',
        'movieId': 'item_id'
    })
    
    # 删除时间戳列（如果存在）
    if 'timestamp' in df.columns:
        df = df.drop('timestamp', axis=1)
    
    print(f"📊 数据统计:")
    print(f"   用户数量: {df['user_id'].nunique()}")
    print(f"   物品数量: {df['item_id'].nunique()}")
    print(f"   评分范围: {df['rating'].min()} - {df['rating'].max()}")
    print(f"   平均评分: {df['rating'].mean():.2f}")
    
    # 数据分割
    print("\n📊 数据分割...")
    train_data, temp_data = train_test_split(df, test_size=0.3, random_state=42, stratify=df['user_id'])
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    print(f"   训练集: {len(train_data)} 条")
    print(f"   验证集: {len(val_data)} 条") 
    print(f"   测试集: {len(test_data)} 条")
    
    return train_data, val_data, test_data

def train_ensemble_model(train_data):
    """训练Ensemble模型"""
    print("\n🤖 创建并训练Ensemble模型...")
    
    # 使用项目中已验证的算法
    algorithms = ['svd', 'autoint', 'deepfm']
    
    model = ExplainableRecommender(
        algorithm_names=algorithms,
        diversity_weight=0.1,
        popularity_penalty=0.05
    )
    
    print(f"🚀 开始训练，目标算法: {algorithms}")
    start_time = time.time()
    
    try:
        model.fit(train_data)
        training_time = time.time() - start_time
        
        successful_models = list(model.models.keys())
        print(f"✅ 训练完成! 耗时: {training_time:.2f}秒")
        print(f"   成功训练的模型: {successful_models}")
        
        if len(successful_models) == 0:
            raise RuntimeError("没有模型训练成功")
            
        return model, training_time, successful_models
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        raise

def evaluate_rating_prediction(model, test_data):
    """评估评分预测能力"""
    print("\n📊 评估评分预测能力...")
    
    predictions = []
    targets = []
    
    # 随机采样测试数据（避免过多计算）
    sample_size = min(5000, len(test_data))
    test_sample = test_data.sample(n=sample_size, random_state=42)
    
    print(f"   使用 {sample_size} 个样本进行评估...")
    
    for idx, row in test_sample.iterrows():
        try:
            user_id = int(row['user_id'])
            item_id = int(row['item_id'])
            true_rating = float(row['rating'])
            
            # 获取预测评分
            pred_rating = model.predict_rating(user_id, item_id)
            
            if pred_rating is not None:
                predictions.append(pred_rating)
                targets.append(true_rating)
                
        except Exception as e:
            continue  # 跳过有问题的样本
    
    if len(predictions) == 0:
        print("❌ 没有有效的预测结果")
        return {
            'rmse': float('inf'),
            'mae': float('inf'),
            'num_predictions': 0
        }
    
    # 计算指标
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    
    print(f"✅ 评分预测评估完成:")
    print(f"   有效预测: {len(predictions)}/{sample_size}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE: {mae:.4f}")
    
    return {
        'rmse': rmse,
        'mae': mae,
        'num_predictions': len(predictions)
    }

def evaluate_recommendation_quality(model, test_data, num_users=50):
    """评估推荐质量"""
    print(f"\n🎯 评估推荐质量 (测试{num_users}个用户)...")
    
    # 选择有足够评分记录的用户
    user_counts = test_data['user_id'].value_counts()
    eligible_users = user_counts[user_counts >= 5].index.tolist()
    
    if len(eligible_users) < num_users:
        num_users = len(eligible_users)
        
    test_users = np.random.choice(eligible_users, num_users, replace=False)
    
    precision_scores = []
    ndcg_scores = []
    coverage_items = set()
    all_recommended_items = []
    
    for i, user_id in enumerate(test_users):
        if i % 10 == 0:
            print(f"   进度: {i}/{num_users}")
            
        try:
            # 获取用户的高评分物品（>=4分）
            user_test_data = test_data[test_data['user_id'] == user_id]
            high_rated_items = set(user_test_data[user_test_data['rating'] >= 4]['item_id'].tolist())
            
            if len(high_rated_items) == 0:
                continue
            
            # 获取推荐
            recommendations = model.get_user_recommendations(user_id, top_k=10, enable_explanation=False)
            
            if len(recommendations) == 0:
                continue
                
            recommended_items = [rec['item_id'] for rec in recommendations]
            all_recommended_items.extend(recommended_items)
            coverage_items.update(recommended_items)
            
            # Precision@10
            relevant_count = len(set(recommended_items) & high_rated_items)
            precision = relevant_count / min(len(recommended_items), 10)
            precision_scores.append(precision)
            
            # NDCG@10
            dcg = 0
            for j, item_id in enumerate(recommended_items[:10]):
                if item_id in high_rated_items:
                    dcg += 1 / np.log2(j + 2)
            
            idcg = sum(1 / np.log2(j + 2) for j in range(min(len(high_rated_items), 10)))
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_scores.append(ndcg)
            
        except Exception as e:
            continue
    
    # 计算多样性（平均两两相似度的补）
    diversity = 0.0
    if len(all_recommended_items) > 1:
        unique_items = list(set(all_recommended_items))
        if len(unique_items) > 1:
            diversity = len(unique_items) / len(all_recommended_items)
    
    # 计算覆盖率
    total_items = test_data['item_id'].nunique()
    coverage = len(coverage_items) / total_items
    
    avg_precision = np.mean(precision_scores) if precision_scores else 0.0
    avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
    
    print(f"✅ 推荐质量评估完成:")
    print(f"   评估用户数: {len(test_users)}")
    print(f"   有效评估: {len(precision_scores)} 个用户")
    print(f"   Precision@10: {avg_precision:.4f}")
    print(f"   NDCG@10: {avg_ndcg:.4f}")
    print(f"   Coverage: {coverage:.4f}")
    print(f"   Diversity: {diversity:.4f}")
    
    return {
        'precision_at_10': avg_precision,
        'ndcg_at_10': avg_ndcg,
        'coverage': coverage,
        'diversity': diversity,
        'evaluated_users': len(precision_scores)
    }

def main():
    """主函数"""
    print("=" * 80)
    print("🔧 修正版Ensemble基准模型训练 - MovieLens 1M")
    print("=" * 80)
    
    try:
        # 1. 数据准备
        train_data, val_data, test_data = load_and_prepare_data()
        
        # 2. 模型训练
        model, training_time, successful_algorithms = train_ensemble_model(train_data)
        
        # 3. 评分预测评估
        rating_metrics = evaluate_rating_prediction(model, test_data)
        
        # 4. 推荐质量评估
        rec_metrics = evaluate_recommendation_quality(model, test_data, num_users=50)
        
        # 5. 汇总结果
        final_results = {
            'model_type': 'ExplainableRecommender Ensemble (Fixed)',
            'dataset': 'MovieLens-1M',
            'successful_algorithms': successful_algorithms,
            'training_time_seconds': training_time,
            'rating_prediction': rating_metrics,
            'recommendation_quality': rec_metrics,
            'overall_metrics': {
                'rmse': rating_metrics['rmse'],
                'mae': rating_metrics['mae'],
                'precision_at_10': rec_metrics['precision_at_10'],
                'ndcg_at_10': rec_metrics['ndcg_at_10'],
                'coverage': rec_metrics['coverage'],
                'diversity': rec_metrics['diversity']
            }
        }
        
        # 6. 保存结果
        results_file = '/home/coder-gw/7Projects_in_7Days/online-inference-system/layerwise_adapter_enhanced/experiments/baselines/ensemble_baseline_1m_fixed_results.json'
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        # 7. 打印结果
        print("\n" + "=" * 60)
        print("🎯 最终评估结果")
        print("=" * 60)
        print(f"模型: ExplainableRecommender Ensemble")
        print(f"成功算法: {successful_algorithms}")
        print(f"训练时间: {training_time:.1f}秒")
        print("\n📈 核心指标:")
        print(f"- RMSE: {rating_metrics['rmse']:.4f}")
        print(f"- MAE: {rating_metrics['mae']:.4f}")
        print(f"- Precision@10: {rec_metrics['precision_at_10']:.4f}")
        print(f"- NDCG@10: {rec_metrics['ndcg_at_10']:.4f}")
        print(f"- Coverage: {rec_metrics['coverage']:.4f}")
        print(f"- Diversity: {rec_metrics['diversity']:.4f}")
        
        print(f"\n💾 结果已保存到: {results_file}")
        print("✅ 修正版基准评估完成!")
        
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
