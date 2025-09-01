#!/usr/bin/env python3
"""
简化版Ensemble基准测试 - 直接使用底层模型
绕过ExplainableRecommender的复杂性，直接测试核心算法
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

# 直接导入核心算法
from models.algorithm_factory import create_recommender

def load_data():
    """加载数据"""
    print("📁 加载MovieLens 1M数据...")
    
    ratings_file = '/home/coder-gw/7Projects_in_7Days/online-inference-system/data/movielens/1m/ratings.csv'
    df = pd.read_csv(ratings_file)
    
    # 统一列名
    df = df.rename(columns={'userId': 'user_id', 'movieId': 'item_id'})
    if 'timestamp' in df.columns:
        df = df.drop('timestamp', axis=1)
    
    print(f"✅ 数据加载完成: {len(df)} 条记录")
    print(f"   用户数: {df['user_id'].nunique()}, 物品数: {df['item_id'].nunique()}")
    
    # 数据分割
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
    print(f"   训练集: {len(train_data)}, 测试集: {len(test_data)}")
    
    return train_data, test_data

def test_individual_algorithms(train_data, test_data):
    """测试各个算法的性能"""
    print("\n🔍 测试各个算法...")
    
    algorithms = ['svd', 'autoint', 'deepfm']
    results = {}
    
    for alg_name in algorithms:
        print(f"\n--- 测试 {alg_name} ---")
        try:
            # 创建模型
            model = create_recommender(alg_name)
            if model is None:
                print(f"❌ 无法创建 {alg_name} 模型")
                continue
            
            # 训练
            print(f"🚀 训练 {alg_name}...")
            start_time = time.time()
            model.fit(train_data)
            training_time = time.time() - start_time
            print(f"✅ {alg_name} 训练完成，耗时: {training_time:.1f}秒")
            
            # 评估
            print(f"📊 评估 {alg_name}...")
            metrics = evaluate_model(model, test_data, alg_name)
            metrics['training_time'] = training_time
            
            results[alg_name] = metrics
            
        except Exception as e:
            print(f"❌ {alg_name} 失败: {e}")
            results[alg_name] = {'error': str(e)}
    
    return results

def evaluate_model(model, test_data, model_name):
    """评估单个模型"""
    # 评分预测
    predictions = []
    targets = []
    
    # 采样测试数据
    sample_size = min(2000, len(test_data))
    test_sample = test_data.sample(n=sample_size, random_state=42)
    
    print(f"   评分预测 ({sample_size} 样本)...")
    
    for _, row in test_sample.iterrows():
        try:
            user_id = int(row['user_id'])
            item_id = int(row['item_id'])
            true_rating = float(row['rating'])
            
            # 尝试预测
            pred_rating = model.predict(user_id, item_id)
            
            if pred_rating is not None and not np.isnan(pred_rating):
                predictions.append(float(pred_rating))
                targets.append(true_rating)
                
        except:
            continue
    
    # 计算指标
    if len(predictions) > 0:
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        mae = mean_absolute_error(targets, predictions)
        print(f"   ✅ 有效预测: {len(predictions)}/{sample_size}")
        print(f"   RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    else:
        rmse = float('inf')
        mae = float('inf')
        print(f"   ❌ 没有有效预测")
    
    # 推荐质量评估
    print(f"   推荐质量评估...")
    rec_quality = evaluate_recommendations(model, test_data)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'num_predictions': len(predictions),
        'recommendation_quality': rec_quality
    }

def evaluate_recommendations(model, test_data, num_users=20):
    """评估推荐质量"""
    # 选择测试用户
    user_counts = test_data['user_id'].value_counts()
    eligible_users = user_counts[user_counts >= 5].index.tolist()[:num_users]
    
    precision_scores = []
    ndcg_scores = []
    
    for user_id in eligible_users:
        try:
            # 用户的高评分物品
            user_data = test_data[test_data['user_id'] == user_id]
            high_rated = set(user_data[user_data['rating'] >= 4]['item_id'].tolist())
            
            if len(high_rated) == 0:
                continue
            
            # 获取推荐
            try:
                recs = model.get_user_recommendations(user_id, top_k=10)
                if isinstance(recs, list) and len(recs) > 0:
                    # 处理不同的返回格式
                    if isinstance(recs[0], tuple):
                        rec_items = [item_id for item_id, _ in recs]
                    elif isinstance(recs[0], dict):
                        rec_items = [rec.get('item_id', rec.get('movie_id')) for rec in recs]
                    else:
                        rec_items = recs
                    
                    # 过滤无效推荐
                    rec_items = [item for item in rec_items if item is not None]
                    
                    if len(rec_items) > 0:
                        # 计算precision
                        relevant = len(set(rec_items) & high_rated)
                        precision = relevant / min(len(rec_items), 10)
                        precision_scores.append(precision)
                        
                        # 计算NDCG
                        dcg = sum(1 / np.log2(i + 2) for i, item in enumerate(rec_items[:10]) if item in high_rated)
                        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(high_rated), 10)))
                        ndcg = dcg / idcg if idcg > 0 else 0
                        ndcg_scores.append(ndcg)
                        
            except Exception as e:
                continue
                
        except Exception as e:
            continue
    
    avg_precision = np.mean(precision_scores) if precision_scores else 0.0
    avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
    
    return {
        'precision_at_10': avg_precision,
        'ndcg_at_10': avg_ndcg,
        'evaluated_users': len(precision_scores)
    }

def create_ensemble_baseline(individual_results):
    """基于个体结果创建ensemble基准"""
    print("\n🔧 创建Ensemble基准...")
    
    # 过滤有效结果
    valid_results = {name: res for name, res in individual_results.items() 
                    if 'error' not in res and res['num_predictions'] > 0}
    
    if len(valid_results) == 0:
        print("❌ 没有有效的个体模型结果")
        return None
    
    print(f"✅ 基于 {list(valid_results.keys())} 创建ensemble基准")
    
    # 简单平均ensemble指标
    ensemble_rmse = np.mean([res['rmse'] for res in valid_results.values() if not np.isinf(res['rmse'])])
    ensemble_mae = np.mean([res['mae'] for res in valid_results.values() if not np.isinf(res['mae'])])
    
    # 推荐质量指标
    precision_scores = [res['recommendation_quality']['precision_at_10'] for res in valid_results.values()]
    ndcg_scores = [res['recommendation_quality']['ndcg_at_10'] for res in valid_results.values()]
    
    ensemble_precision = np.mean(precision_scores) if precision_scores else 0.0
    ensemble_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
    
    return {
        'model_type': 'Simple Ensemble Baseline',
        'component_algorithms': list(valid_results.keys()),
        'ensemble_metrics': {
            'rmse': float(ensemble_rmse) if not np.isnan(ensemble_rmse) else float('inf'),
            'mae': float(ensemble_mae) if not np.isnan(ensemble_mae) else float('inf'),
            'precision_at_10': float(ensemble_precision),
            'ndcg_at_10': float(ensemble_ndcg)
        },
        'individual_results': valid_results
    }

def main():
    """主函数"""
    print("=" * 80)
    print("🧪 简化版Ensemble基准测试")
    print("=" * 80)
    
    try:
        # 加载数据
        train_data, test_data = load_data()
        
        # 测试各个算法
        individual_results = test_individual_algorithms(train_data, test_data)
        
        # 创建ensemble基准
        ensemble_result = create_ensemble_baseline(individual_results)
        
        # 保存结果
        results_file = '/home/coder-gw/7Projects_in_7Days/online-inference-system/layerwise_adapter_enhanced/experiments/baselines/simple_ensemble_baseline_results.json'
        
        final_results = {
            'dataset': 'MovieLens-1M',
            'evaluation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'individual_algorithms': individual_results,
            'ensemble_baseline': ensemble_result
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        # 打印结果
        print("\n" + "=" * 60)
        print("📊 最终结果汇总")
        print("=" * 60)
        
        print("\n🔍 个体算法结果:")
        for alg_name, result in individual_results.items():
            if 'error' not in result:
                print(f"{alg_name}:")
                print(f"  RMSE: {result['rmse']:.4f}, MAE: {result['mae']:.4f}")
                print(f"  P@10: {result['recommendation_quality']['precision_at_10']:.4f}")
                print(f"  NDCG@10: {result['recommendation_quality']['ndcg_at_10']:.4f}")
            else:
                print(f"{alg_name}: 失败 - {result['error']}")
        
        if ensemble_result:
            print(f"\n🎯 Ensemble基准:")
            metrics = ensemble_result['ensemble_metrics']
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  MAE: {metrics['mae']:.4f}")
            print(f"  Precision@10: {metrics['precision_at_10']:.4f}")
            print(f"  NDCG@10: {metrics['ndcg_at_10']:.4f}")
        
        print(f"\n💾 详细结果保存到: {results_file}")
        print("✅ 简化版基准测试完成!")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
