#!/usr/bin/env python3
"""
调试ensemble基准模型的问题
主要检查：
1. 模型是否正确训练
2. 推荐生成是否正常
3. 评估指标计算是否正确
"""

import sys
import os
sys.path.append('/home/coder-gw/7Projects_in_7Days/online-inference-system')

import pandas as pd
import numpy as np
from models.ensemble_recommender import ExplainableRecommender

def load_movielens_1m():
    """加载MovieLens 1M数据"""
    ratings_file = '/home/coder-gw/7Projects_in_7Days/online-inference-system/data/movielens/1m/ratings.csv'
    
    print(f"🔍 检查数据文件: {ratings_file}")
    if not os.path.exists(ratings_file):
        print(f"❌ 文件不存在: {ratings_file}")
        return None
    
    df = pd.read_csv(ratings_file)
    print(f"✅ 加载了 {len(df)} 条评分记录")
    print(f"用户数量: {df['user_id'].nunique()}")
    print(f"物品数量: {df['item_id'].nunique()}")
    print(f"评分范围: {df['rating'].min()} - {df['rating'].max()}")
    print("前5条记录:")
    print(df.head())
    return df

def debug_model_training():
    """调试模型训练过程"""
    print("=" * 60)
    print("🔍 调试模型训练过程")
    print("=" * 60)
    
    # 加载数据
    df = load_movielens_1m()
    if df is None:
        return
    
    # 使用小样本进行测试
    print("\n📝 使用小样本进行调试...")
    sample_df = df.sample(n=1000, random_state=42)
    print(f"调试样本: {len(sample_df)} 条记录")
    print(f"用户数量: {sample_df['user_id'].nunique()}")
    print(f"物品数量: {sample_df['item_id'].nunique()}")
    
    # 创建模型
    print("\n🤖 创建ExplainableRecommender...")
    algorithms = ['svd']  # 先测试单一算法
    model = ExplainableRecommender(
        algorithm_names=algorithms,
        diversity_weight=0.1,
        popularity_penalty=0.05
    )
    
    # 训练模型
    print("\n🚀 训练模型...")
    try:
        model.fit(sample_df)
        print(f"✅ 训练完成，成功训练的模型: {list(model.models.keys())}")
        
        # 检查模型状态
        print(f"模型训练状态: {model.is_trained}")
        print(f"物品流行度统计: {len(model.item_popularity)} 个物品")
        
        # 测试推荐生成
        print("\n🎯 测试推荐生成...")
        test_users = sample_df['user_id'].unique()[:3]
        
        for user_id in test_users:
            print(f"\n用户 {user_id} 的推荐:")
            try:
                recs = model.get_user_recommendations(user_id, top_k=5, enable_explanation=False)
                print(f"  推荐数量: {len(recs)}")
                
                if len(recs) > 0:
                    for i, rec in enumerate(recs[:3]):
                        print(f"  {i+1}. 物品{rec.get('item_id', 'N/A')}, 评分:{rec.get('score', 'N/A'):.3f}")
                else:
                    print("  ❌ 没有生成推荐")
                    
                    # 调试：检查底层模型
                    for alg_name, alg_model in model.models.items():
                        try:
                            raw_recs = alg_model.get_user_recommendations(user_id, top_k=5)
                            print(f"    {alg_name} 原始推荐数量: {len(raw_recs)}")
                            if len(raw_recs) > 0:
                                print(f"    {alg_name} 第一个推荐: {raw_recs[0]}")
                        except Exception as e:
                            print(f"    {alg_name} 推荐失败: {e}")
                    
            except Exception as e:
                print(f"  ❌ 推荐生成失败: {e}")
                import traceback
                traceback.print_exc()
                
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()

def debug_evaluation_metrics():
    """调试评估指标计算"""
    print("\n" + "=" * 60)
    print("🔍 调试评估指标计算")
    print("=" * 60)
    
    # 模拟一些测试数据
    print("\n📊 测试评估指标计算...")
    
    # 创建模拟数据
    predictions = np.array([3.5, 4.2, 2.8, 4.1, 3.9])
    targets = np.array([4.0, 4.0, 3.0, 4.5, 4.0])
    
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    mae = np.mean(np.abs(predictions - targets))
    
    print(f"模拟RMSE: {rmse:.4f}")
    print(f"模拟MAE: {mae:.4f}")
    
    # 模拟推荐质量指标
    recommended_items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    high_rated_items = {2, 4, 5, 8}
    
    relevant_recommended = len(set(recommended_items) & high_rated_items)
    precision_at_10 = relevant_recommended / min(len(recommended_items), 10)
    
    print(f"模拟Precision@10: {precision_at_10:.4f}")
    
    # 计算NDCG
    dcg = 0
    for i, item_id in enumerate(recommended_items[:10]):
        if item_id in high_rated_items:
            dcg += 1 / np.log2(i + 2)
    
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(high_rated_items), 10)))
    ndcg = dcg / idcg if idcg > 0 else 0
    
    print(f"模拟NDCG@10: {ndcg:.4f}")

if __name__ == "__main__":
    debug_model_training()
    debug_evaluation_metrics()
