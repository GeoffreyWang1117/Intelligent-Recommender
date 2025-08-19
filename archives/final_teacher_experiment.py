#!/usr/bin/env python3
"""
简化CPU Teacher模型实验 - 完整一致性分析
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, List
from collections import defaultdict

# 强制CPU模式
import torch
torch.device('cpu')

from models import create_recommender

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def simple_jaccard_similarity(set1: set, set2: set) -> float:
    """计算Jaccard相似度"""
    if not set1 and not set2:
        return 1.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0


def run_teacher_experiment():
    """运行Teacher模型一致性分析实验"""
    print("🎓 Teacher模型完整一致性分析")
    print("=" * 60)
    
    # 创建数据
    print("📊 创建模拟推荐数据...")
    np.random.seed(42)
    
    interactions = []
    for _ in range(3000):
        user_id = np.random.randint(1, 201)  # 200用户
        item_id = np.random.randint(1, 151)  # 150物品
        rating = np.random.choice([3, 4, 5], p=[0.3, 0.4, 0.3])
        timestamp = int(time.time()) + np.random.randint(-86400*30, 86400*30)
        
        interactions.append({
            'user_id': user_id,
            'item_id': item_id,
            'rating': rating,
            'timestamp': timestamp
        })
    
    train_data = pd.DataFrame(interactions)
    train_data = train_data.drop_duplicates(['user_id', 'item_id'])
    
    print(f"✅ 生成了 {len(train_data)} 条唯一交互数据")
    print(f"   用户数: {train_data['user_id'].nunique()}")
    print(f"   物品数: {train_data['item_id'].nunique()}")
    
    # Teacher模型配置 - 优化版本
    algorithms = ['deepfm', 'autoint', 'transformer4rec', 'xdeepfm']
    
    model_configs = {
        'deepfm': {
            'embedding_dim': 16,
            'learning_rate': 0.001,
            'epochs': 15,
            'batch_size': 64,
            'early_stopping': False
        },
        'autoint': {
            'embedding_dim': 16,
            'learning_rate': 0.001,
            'epochs': 15,
            'num_heads': 4,
            'num_layers': 3,
            'early_stopping': False
        },
        'transformer4rec': {
            'embedding_dim': 32,
            'num_heads': 4,
            'num_layers': 3,
            'learning_rate': 0.001,
            'epochs': 12,
            'batch_size': 64,
            'early_stopping': False
        },
        'xdeepfm': {
            'embedding_dim': 16,
            'cin_layer_sizes': [64, 32],
            'dnn_hidden_dims': [128, 64],
            'learning_rate': 0.001,
            'epochs': 12,
            'batch_size': 64,
            'early_stopping': False
        }
    }
    
    print(f"🎯 将训练 {len(algorithms)} 个Teacher模型:")
    for algo in algorithms:
        print(f"   - {algo}")
    
    # 训练模型
    trained_models = {}
    training_times = {}
    
    for algo_name in algorithms:
        print(f"\n📚 训练Teacher模型: {algo_name}")
        print(f"   配置: {model_configs[algo_name]}")
        
        try:
            config = model_configs[algo_name]
            
            start_time = time.time()
            model = create_recommender(algo_name, **config)
            
            if model is None:
                print(f"❌ 无法创建算法: {algo_name}")
                continue
            
            # 强制CPU模式
            if hasattr(model, 'device'):
                model.device = torch.device('cpu')
            
            model.fit(train_data, **config)
            end_time = time.time()
            
            training_time = end_time - start_time
            training_times[algo_name] = training_time
            
            if model.is_trained:
                trained_models[algo_name] = model
                print(f"✅ {algo_name} 训练成功! 耗时: {training_time:.2f}秒")
            else:
                print(f"❌ {algo_name} 训练失败 - 模型未完成训练")
                
        except Exception as e:
            print(f"❌ 训练 {algo_name} 时出错: {str(e)}")
            continue
    
    print(f"\n🎯 训练阶段完成!")
    print(f"   成功训练: {len(trained_models)} 个Teacher模型")
    print(f"   成功的模型: {list(trained_models.keys())}")
    
    if len(trained_models) < 2:
        print("❌ 训练成功的模型少于2个，无法进行一致性分析")
        return None
    
    # 生成推荐
    print("\n📊 生成推荐并分析一致性...")
    test_users = list(train_data['user_id'].unique())[:50]  # 测试50个用户
    
    all_recommendations = {}
    performance_stats = {}
    
    for algo_name, model in trained_models.items():
        print(f"🔍 生成 {algo_name} 的推荐...")
        
        recommendations = {}
        success_count = 0
        total_rec_count = 0
        
        for user_id in test_users:
            try:
                recs = model.get_user_recommendations(user_id, top_k=10)
                if recs and len(recs) > 0:
                    rec_items = [rec['item_id'] for rec in recs]
                    recommendations[user_id] = rec_items
                    success_count += 1
                    total_rec_count += len(rec_items)
            except Exception as e:
                continue
        
        if recommendations:
            all_recommendations[algo_name] = recommendations
            performance_stats[algo_name] = {
                'success_rate': success_count / len(test_users),
                'avg_rec_length': total_rec_count / success_count if success_count > 0 else 0,
                'total_recommendations': len(recommendations),
                'training_time': training_times.get(algo_name, 0)
            }
            print(f"   ✅ 成功为 {len(recommendations)} 个用户生成推荐")
            print(f"   📈 成功率: {success_count/len(test_users):.2%}")
    
    # 计算一致性矩阵
    print("\n🔍 计算Teacher模型间一致性...")
    
    if len(all_recommendations) < 2:
        print("❌ 无法进行一致性分析 - 推荐生成失败")
        return None
    
    algorithms_list = list(all_recommendations.keys())
    jaccard_matrix = {}
    overlap_stats = {}
    
    print(f"📊 分析 {len(algorithms_list)} 个模型间的一致性:")
    
    # 计算Jaccard相似度矩阵
    for i, algo1 in enumerate(algorithms_list):
        jaccard_matrix[algo1] = {}
        overlap_stats[algo1] = {}
        
        for j, algo2 in enumerate(algorithms_list):
            if algo1 == algo2:
                jaccard_matrix[algo1][algo2] = 1.0
                overlap_stats[algo1][algo2] = {'avg_overlap': 1.0, 'user_count': 0}
            else:
                similarities = []
                overlaps = []
                
                # 找到两个算法都有推荐的用户
                common_users = set(all_recommendations[algo1].keys()) & set(all_recommendations[algo2].keys())
                
                for user_id in common_users:
                    recs1 = set(all_recommendations[algo1][user_id][:10])
                    recs2 = set(all_recommendations[algo2][user_id][:10])
                    
                    # Jaccard相似度
                    sim = simple_jaccard_similarity(recs1, recs2)
                    similarities.append(sim)
                    
                    # 重叠数量
                    overlap = len(recs1.intersection(recs2))
                    overlaps.append(overlap)
                
                avg_similarity = np.mean(similarities) if similarities else 0.0
                avg_overlap = np.mean(overlaps) if overlaps else 0.0
                
                jaccard_matrix[algo1][algo2] = avg_similarity
                overlap_stats[algo1][algo2] = {
                    'avg_overlap': avg_overlap,
                    'user_count': len(common_users)
                }
        
        print(f"   ✅ 完成 {algo1} 的一致性计算")
    
    # 分析结果
    print("\n📈 分析Teacher模型一致性结果...")
    
    # 找到最低和最高相似度
    min_jaccard = float('inf')
    max_jaccard = 0.0
    min_pair = None
    max_pair = None
    
    all_similarities = []
    
    for algo1 in algorithms_list:
        for algo2 in algorithms_list:
            if algo1 != algo2:
                similarity = jaccard_matrix[algo1][algo2]
                all_similarities.append(similarity)
                
                if similarity < min_jaccard:
                    min_jaccard = similarity
                    min_pair = (algo1, algo2)
                if similarity > max_jaccard:
                    max_jaccard = similarity
                    max_pair = (algo1, algo2)
    
    avg_jaccard = np.mean(all_similarities) if all_similarities else 0.0
    
    # 生成详细报告
    print("\n📝 生成Teacher模型分析报告...")
    
    report = []
    report.append("# 🎓 Teacher模型一致性分析报告")
    report.append(f"**实验时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    report.append("## 📊 实验概述")
    report.append(f"- **训练的Teacher模型**: {', '.join(algorithms)}")
    report.append(f"- **成功训练模型**: {', '.join(trained_models.keys())}")
    report.append(f"- **数据集规模**: {len(train_data)} 条交互")
    report.append(f"- **测试用户数**: {len(test_users)}")
    report.append(f"- **运行模式**: CPU优化模式")
    report.append("")
    
    # 训练性能
    report.append("## ⏱️ Teacher模型训练性能")
    report.append("")
    report.append("| 模型 | 训练时间(秒) | 状态 |")
    report.append("|------|-------------|------|")
    
    for algo in algorithms:
        if algo in trained_models:
            train_time = training_times.get(algo, 0)
            report.append(f"| {algo} | {train_time:.2f} | ✅ 成功 |")
        else:
            report.append(f"| {algo} | - | ❌ 失败 |")
    report.append("")
    
    # 推荐性能统计
    report.append("## 🏆 Teacher模型推荐性能")
    report.append("")
    report.append("| 模型 | 成功率 | 平均推荐数 | 总推荐数 | 训练时间(秒) |")
    report.append("|------|--------|------------|----------|-------------|")
    
    for algo_name, stats in performance_stats.items():
        report.append(f"| {algo_name} | {stats['success_rate']:.2%} | {stats['avg_rec_length']:.1f} | {stats['total_recommendations']} | {stats['training_time']:.2f} |")
    report.append("")
    
    # 一致性分析结果
    report.append("## 🔄 Teacher模型一致性分析")
    report.append("")
    
    report.append("### 📊 关键指标")
    report.append(f"- **平均Jaccard相似度**: {avg_jaccard:.4f}")
    if min_pair:
        report.append(f"- **最强互补组合**: {min_pair[0]} + {min_pair[1]} (Jaccard={min_jaccard:.4f})")
    if max_pair:
        report.append(f"- **最相似组合**: {max_pair[0]} + {max_pair[1]} (Jaccard={max_jaccard:.4f})")
    report.append("")
    
    # Jaccard相似度矩阵
    report.append("### 🔢 Jaccard相似度矩阵")
    report.append("")
    
    # 创建表格
    header = "| 算法 |" + "".join([f" {algo} |" for algo in algorithms_list])
    separator = "|" + "".join(["------|" for _ in range(len(algorithms_list) + 1)])
    
    report.append(header)
    report.append(separator)
    
    for algo1 in algorithms_list:
        row = f"| **{algo1}** |"
        for algo2 in algorithms_list:
            if algo1 == algo2:
                row += " 1.0000 |"
            else:
                similarity = jaccard_matrix[algo1][algo2]
                row += f" {similarity:.4f} |"
        report.append(row)
    
    report.append("")
    
    # Teacher ensemble建议
    report.append("## 🎯 Teacher模型Ensemble策略建议")
    report.append("")
    
    if min_pair and max_pair:
        report.append("### 🥇 最佳Teacher Ensemble组合")
        report.append("")
        report.append(f"**最强互补策略**: {min_pair[0]} + {min_pair[1]}")
        report.append(f"- **Jaccard相似度**: {min_jaccard:.4f}")
        report.append(f"- **特点**: 推荐重叠度最低，多样性最高")
        report.append(f"- **适用场景**: 需要最大化推荐覆盖面的场景")
        report.append("")
        
        report.append(f"**最强一致策略**: {max_pair[0]} + {max_pair[1]}")
        report.append(f"- **Jaccard相似度**: {max_jaccard:.4f}")  
        report.append(f"- **特点**: 推荐高度一致，风格相似")
        report.append(f"- **适用场景**: 需要稳定可靠推荐的场景")
        report.append("")
    
    # 多模型组合分析
    if len(algorithms_list) >= 3:
        # 计算每个模型的平均多样性
        avg_diversities = {}
        for algo in algorithms_list:
            other_sims = [jaccard_matrix[algo][other] for other in algorithms_list if other != algo]
            avg_diversities[algo] = np.mean(other_sims)
        
        # 最多样化的3个模型
        most_diverse_3 = sorted(avg_diversities.items(), key=lambda x: x[1])[:3]
        
        report.append("### 🥈 多样化Teacher组合")
        diverse_names = [algo for algo, _ in most_diverse_3]
        report.append(f"**推荐组合**: {' + '.join(diverse_names)}")
        report.append("- **选择依据**: 平均相似度最低的3个模型")
        report.append("- **特点**: 平衡多样性与稳定性")
        report.append("")
    
    # 实施建议
    report.append("## 🚀 实施建议")
    report.append("")
    report.append("### 1. Teacher模型特点")
    report.append("- **deepfm**: 特征交互建模，适合稀疏特征")
    report.append("- **autoint**: 自动特征交互，注意力机制")
    report.append("- **transformer4rec**: 序列建模，适合时序推荐")
    report.append("- **xdeepfm**: 显式+隐式特征交互，模型容量大")
    report.append("")
    
    report.append("### 2. Ensemble策略")
    if min_pair:
        report.append(f"- **多样性优先**: 使用 {min_pair[0]} + {min_pair[1]} 组合")
    if avg_jaccard < 0.3:
        report.append("- **低一致性**: 当前模型差异较大，适合ensemble")
    elif avg_jaccard > 0.7:
        report.append("- **高一致性**: 模型相似度较高，可能存在冗余")
    else:
        report.append("- **中等一致性**: 模型间有一定差异，ensemble效果可期")
    report.append("")
    
    report.append("### 3. 优化方向")
    report.append("- **训练优化**: 增加数据量和训练轮数")
    report.append("- **硬件升级**: 使用GPU加速训练")
    report.append("- **参数调优**: 针对不同模型优化超参数")
    report.append("- **定期更新**: 重新评估模型一致性")
    
    # 保存报告
    report_content = "\n".join(report)
    
    with open("TEACHER_MODEL_ANALYSIS_REPORT.md", "w", encoding='utf-8') as f:
        f.write(report_content)
    
    print("✅ Teacher模型分析报告已保存到 TEACHER_MODEL_ANALYSIS_REPORT.md")
    
    # 显示关键结果
    print("\n🎉 Teacher模型一致性分析完成！")
    print("\n📊 关键结果:")
    print(f"📈 成功训练的Teacher模型: {list(trained_models.keys())}")
    print(f"📊 平均Jaccard相似度: {avg_jaccard:.4f}")
    
    if min_pair:
        print(f"🏆 最佳互补组合: {min_pair[0]} + {min_pair[1]} (Jaccard={min_jaccard:.4f})")
    
    if max_pair:
        print(f"⚠️  最相似组合: {max_pair[0]} + {max_pair[1]} (Jaccard={max_jaccard:.4f})")
    
    # 输出一致性矩阵
    print("\n📋 Jaccard相似度矩阵:")
    print("     ", end="")
    for algo in algorithms_list:
        print(f"{algo:>10}", end="")
    print()
    
    for algo1 in algorithms_list:
        print(f"{algo1:>8}", end="")
        for algo2 in algorithms_list:
            if algo1 == algo2:
                print(f"{'1.0000':>10}", end="")
            else:
                print(f"{jaccard_matrix[algo1][algo2]:>10.4f}", end="")
        print()
    
    return {
        'trained_models': list(trained_models.keys()),
        'jaccard_matrix': jaccard_matrix,
        'best_complementary': min_pair,
        'most_similar': max_pair,
        'avg_jaccard': avg_jaccard,
        'performance_stats': performance_stats
    }


if __name__ == "__main__":
    result = run_teacher_experiment()
    if result:
        print(f"\n✅ 实验成功完成！详细报告请查看 TEACHER_MODEL_ANALYSIS_REPORT.md")
    else:
        print(f"\n❌ 实验失败，请检查错误信息")
