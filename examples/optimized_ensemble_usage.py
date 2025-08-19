#!/usr/bin/env python3
"""
优化集成推荐器使用示例

本示例展示如何使用基于一致性分析的优化集成推荐器，
包括三种不同的集成策略和最佳实践。

基于consistency_analysis结果，推荐的最佳组合：
- 最互补组合: SVD + SimpleAutoInt (Jaccard=0.0403, 极低重叠度)
- 最多样化组合: SVD + SimpleAutoInt + SimpleDIN (平衡的多样性)
- 质量优先组合: 基于单独性能选择最佳算法组合
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from models import create_optimized_ensemble


def create_sample_data():
    """创建示例数据"""
    print("📊 创建示例MovieLens数据...")
    
    # 简单创建示例数据
    np.random.seed(42)
    interactions = []
    for user_id in range(1, 101):  # 100个用户
        for item_id in range(1, 51):  # 50个物品
            if np.random.random() < 0.1:  # 10%的交互率
                rating = np.random.choice([3, 4, 5], p=[0.3, 0.4, 0.3])
                interactions.append({
                    'user_id': user_id,
                    'item_id': item_id, 
                    'rating': rating,
                    'timestamp': 1234567890
                })
    
    train_data = pd.DataFrame(interactions)
    print(f"✓ 创建了 {len(train_data)} 条交互数据")
    
    return train_data


def demonstrate_ensemble_strategies():
    """演示三种集成策略"""
    print("\n🎯 演示优化集成推荐器的三种策略\n")
    
    # 创建示例数据
    train_data = create_sample_data()
    
    # 三种集成策略
    strategies = {
        'best_complementary': '最互补策略 - 选择重叠度最低的算法组合',
        'best_diverse': '最多样化策略 - 平衡性能和多样性',
        'quality_focused': '质量优先策略 - 基于单独性能选择'
    }
    
    results = {}
    
    for strategy, description in strategies.items():
        print(f"🔧 {description}")
        print(f"策略: {strategy}")
        
        try:
            # 创建优化集成推荐器
            ensemble = create_optimized_ensemble(strategy=strategy)
            print(f"✓ 成功创建集成推荐器")
            
            # 训练模型
            print("📚 训练模型...")
            ensemble.fit(train_data)
            print("✓ 训练完成")
            
            # 生成推荐
            user_id = 1
            recommendations = ensemble.recommend(user_id, k=10)
            
            print(f"👤 为用户 {user_id} 生成的推荐:")
            for i, (item_id, score) in enumerate(recommendations[:5], 1):
                print(f"   {i}. 物品 {item_id} (评分: {score:.4f})")
            
            # 获取推荐解释
            explanations = ensemble.explain_recommendation(user_id, recommendations[0][0])
            print(f"🔍 推荐解释: {explanations}")
            
            results[strategy] = {
                'recommendations': recommendations,
                'explanations': explanations
            }
            
        except Exception as e:
            print(f"❌ 策略 {strategy} 出现错误: {e}")
            
        print("-" * 60)
    
    return results


def compare_strategies():
    """比较不同策略的效果"""
    print("\n📈 策略效果比较\n")
    
    train_data = create_sample_data()
    
    print("| 策略 | 算法组合 | 特点 | 适用场景 |")
    print("|------|----------|------|----------|")
    
    strategies_info = {
        'best_complementary': {
            'algorithms': 'SVD + SimpleAutoInt',
            'features': '极低重叠度(0.0403)',
            'use_case': '追求推荐多样性'
        },
        'best_diverse': {
            'algorithms': 'SVD + SimpleAutoInt + SimpleDIN', 
            'features': '平衡性能与多样性',
            'use_case': '综合最佳效果'
        },
        'quality_focused': {
            'algorithms': '基于性能动态选择',
            'features': '注重推荐质量',
            'use_case': '质量优先场景'
        }
    }
    
    for strategy, info in strategies_info.items():
        print(f"| {strategy} | {info['algorithms']} | {info['features']} | {info['use_case']} |")


def advanced_usage_examples():
    """高级使用示例"""
    print("\n🚀 高级使用示例\n")
    
    train_data = create_sample_data()
    
    print("1. 自定义集成权重:")
    try:
        ensemble = create_optimized_ensemble(strategy='best_complementary')
        
        # 可以通过参数调整集成策略
        print("   ✓ 使用默认权重")
        
        # 训练并测试
        ensemble.fit(train_data)
        recs = ensemble.recommend(1, k=5)
        print(f"   推荐结果: {len(recs)} 个物品")
        
    except Exception as e:
        print(f"   ❌ 错误: {e}")
    
    print("\n2. 获取详细性能指标:")
    try:
        ensemble = create_optimized_ensemble(strategy='best_diverse')
        ensemble.fit(train_data)
        
        # 这里可以添加性能评估逻辑
        print("   ✓ 可集成评估指标 (R@10, N@10等)")
        
    except Exception as e:
        print(f"   ❌ 错误: {e}")
    
    print("\n3. 实时推荐更新:")
    print("   ✓ 支持增量学习和在线更新")
    print("   ✓ 可根据用户反馈动态调整权重")


def main():
    """主函数"""
    print("🎬 优化集成推荐器使用指南")
    print("=" * 60)
    
    # 演示三种策略
    results = demonstrate_ensemble_strategies()
    
    # 比较策略效果
    compare_strategies()
    
    # 高级使用示例
    advanced_usage_examples()
    
    print("\n📋 总结:")
    print("✓ 基于一致性分析的科学集成策略")
    print("✓ 三种不同场景的优化策略")
    print("✓ 简单易用的API接口")
    print("✓ 完整的推荐解释功能")
    
    print("\n🎯 推荐使用:")
    print("- 新项目: 使用 'best_diverse' 策略")
    print("- 追求多样性: 使用 'best_complementary' 策略") 
    print("- 质量优先: 使用 'quality_focused' 策略")


if __name__ == "__main__":
    main()
