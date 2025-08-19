#!/usr/bin/env python3
"""
简化CPU Teacher模型实验
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


def run_simple_teacher_experiment():
    """运行简化Teacher模型实验"""
    print("🎓 简化Teacher模型一致性分析")
    print("=" * 60)
    
    # 创建数据
    print("📊 创建模拟数据...")
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
    
    # Teacher模型配置
    algorithms = ['deepfm', 'autoint', 'transformer4rec', 'xdeepfm']
    
    model_configs = {
        'deepfm': {
            'embedding_dim': 8,
            'learning_rate': 0.001,
            'epochs': 10,
            'batch_size': 64,
            'early_stopping': False
        },
        'autoint': {
            'embedding_dim': 8,
            'learning_rate': 0.001,
            'epochs': 10,
            'num_heads': 2,
            'num_layers': 2,
            'early_stopping': False
        },
        'transformer4rec': {
            'embedding_dim': 16,
            'num_heads': 2,
            'num_layers': 2,
            'learning_rate': 0.001,
            'epochs': 8,
            'batch_size': 64,
            'early_stopping': False
        },
        'xdeepfm': {
            'embedding_dim': 8,
            'cin_layer_sizes': [32, 32],
            'dnn_hidden_dims': [64, 32],
            'learning_rate': 0.001,
            'epochs': 8,
            'batch_size': 64,
            'early_stopping': False
        }
    }
    
    # 训练模型
    trained_models = {}
    
    for algo_name in algorithms:
        print(f"\n📚 训练 {algo_name}...")
        
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
            
            if model.is_trained:
                trained_models[algo_name] = model
                print(f"✅ {algo_name} 训练成功 (耗时: {end_time - start_time:.2f}秒)")
            else:
                print(f"❌ {algo_name} 训练失败")
                
        except Exception as e:
            print(f"❌ 训练 {algo_name} 时出错: {str(e)[:100]}...")
    
    print(f"\n🎯 训练完成！成功训练了 {len(trained_models)} 个Teacher模型")
    print(f"成功的模型: {list(trained_models.keys())}")
    
    if len(trained_models) < 2:
        print("❌ 训练成功的模型少于2个，无法进行一致性分析")
        return
    
    # 生成推荐
    print("\n📊 生成推荐并分析一致性...")
    test_users = train_data['user_id'].unique()[:30]  # 测试30个用户
    
    all_recommendations = {}
    performance_stats = {}
    
    for algo_name, model in trained_models.items():
        print(f"生成 {algo_name} 的推荐...")
        
        recommendations = {}
        success_count = 0
        
        for user_id in test_users:
            try:
                recs = model.get_user_recommendations(user_id, top_k=10)
                if recs and len(recs) > 0:
                    recommendations[user_id] = [rec['item_id'] for rec in recs]
                    success_count += 1
            except Exception as e:
                continue
        
        if recommendations:
            all_recommendations[algo_name] = recommendations
            performance_stats[algo_name] = {
                'success_rate': success_count / len(test_users),
                'avg_rec_length': np.mean([len(recs) for recs in recommendations.values()]),
                'total_recommendations': len(recommendations)
            }
            print(f"  成功为 {len(recommendations)} 个用户生成推荐 (成功率: {success_count/len(test_users):.2%})")
    
    # 计算一致性
    print("\n🔍 计算Teacher模型间一致性...")
    
    if len(all_recommendations) < 2:
        print("❌ 无法进行一致性分析")
        return
    
    algorithms_list = list(all_recommendations.keys())
    jaccard_matrix = {}
    
    # 计算Jaccard相似度矩阵
    for algo1 in algorithms_list:
        jaccard_matrix[algo1] = {}
        for algo2 in algorithms_list:
            if algo1 == algo2:
                jaccard_matrix[algo1][algo2] = 1.0
            else:
                similarities = []
                
                # 找到两个算法都有推荐的用户
                common_users = set(all_recommendations[algo1].keys()) & set(all_recommendations[algo2].keys())
                
                for user_id in common_users:
                    recs1 = set(all_recommendations[algo1][user_id][:10])
                    recs2 = set(all_recommendations[algo2][user_id][:10])
                    sim = simple_jaccard_similarity(recs1, recs2)
                    similarities.append(sim)
                
                avg_similarity = np.mean(similarities) if similarities else 0.0
                jaccard_matrix[algo1][algo2] = avg_similarity\n    \n    # 生成报告\n    print(\"\\n📝 生成Teacher模型分析报告...\")\n    \n    report = []\n    report.append(\"# 🎓 Teacher模型一致性分析报告 (简化版)\")\n    report.append(f\"实验时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\")\n    report.append(\"\")\n    \n    report.append(\"## 📊 实验概述\")\n    report.append(f\"- **成功训练模型**: {list(trained_models.keys())}\")\n    report.append(f\"- **数据集规模**: {len(train_data)} 条交互\")\n    report.append(f\"- **测试用户数**: {len(test_users)}\")\n    report.append(f\"- **运行模式**: CPU (简化版)\")\n    report.append(\"\")\n    \n    # 性能统计\n    report.append(\"## 🏆 Teacher模型性能统计\")\n    report.append(\"\")\n    report.append(\"| 模型 | 成功率 | 平均推荐数 | 总推荐数 |\")\n    report.append(\"|------|--------|------------|----------|\")\n    \n    for algo_name, stats in performance_stats.items():\n        report.append(f\"| {algo_name} | {stats['success_rate']:.2%} | {stats['avg_rec_length']:.1f} | {stats['total_recommendations']} |\")\n    report.append(\"\")\n    \n    # Jaccard相似度矩阵\n    report.append(\"## 🔄 Teacher模型Jaccard相似度矩阵\")\n    report.append(\"\")\n    \n    # 找到最低和最高相似度\n    min_jaccard = float('inf')\n    max_jaccard = 0.0\n    min_pair = None\n    max_pair = None\n    \n    for algo1 in algorithms_list:\n        for algo2 in algorithms_list:\n            if algo1 != algo2:\n                similarity = jaccard_matrix[algo1][algo2]\n                if similarity < min_jaccard:\n                    min_jaccard = similarity\n                    min_pair = (algo1, algo2)\n                if similarity > max_jaccard:\n                    max_jaccard = similarity\n                    max_pair = (algo1, algo2)\n    \n    report.append(\"### 🎯 关键发现\")\n    if min_pair:\n        report.append(f\"- **最强互补组合**: {min_pair[0]} + {min_pair[1]} (Jaccard={min_jaccard:.4f})\")\n    if max_pair:\n        report.append(f\"- **最相似组合**: {max_pair[0]} + {max_pair[1]} (Jaccard={max_jaccard:.4f})\")\n    report.append(\"\")\n    \n    # 相似度矩阵表格\n    report.append(\"### Jaccard相似度矩阵\")\n    report.append(\"\")\n    \n    # 创建表格标题\n    header = \"| 算法 |\" + \"|\".join([f\" {algo} \" for algo in algorithms_list]) + \"|\"\n    separator = \"|\" + \"|\".join([\"------\" for _ in range(len(algorithms_list) + 1)]) + \"|\"\n    \n    report.append(header)\n    report.append(separator)\n    \n    for algo1 in algorithms_list:\n        row = f\"| {algo1} |\"\n        for algo2 in algorithms_list:\n            if algo1 == algo2:\n                row += \" 1.0000 |\"\n            else:\n                row += f\" {jaccard_matrix[algo1][algo2]:.4f} |\"\n        report.append(row)\n    \n    report.append(\"\")\n    \n    # Teacher ensemble建议\n    report.append(\"## 🎯 Teacher模型Ensemble策略建议\")\n    report.append(\"\")\n    \n    if min_pair:\n        report.append(\"### 🥇 推荐的最佳Teacher ensemble组合\")\n        report.append(f\"**最强互补策略**: {min_pair[0]} + {min_pair[1]}\")\n        report.append(f\"- Jaccard相似度: {min_jaccard:.4f}\")\n        report.append(f\"- 特点: 重叠度最低，推荐多样性最高\")\n        report.append(\"\")\n    \n    # 三模型组合\n    if len(algorithms_list) >= 3:\n        # 按平均相似度排序，选择最多样化的3个模型\n        avg_similarities = {}\n        for algo in algorithms_list:\n            other_sims = [jaccard_matrix[algo][other] for other in algorithms_list if other != algo]\n            avg_similarities[algo] = np.mean(other_sims)\n        \n        most_diverse_3 = sorted(avg_similarities.items(), key=lambda x: x[1])[:3]\n        \n        report.append(\"### 🥈 多样化三模型组合\")\n        report.append(f\"**推荐组合**: {' + '.join([algo for algo, _ in most_diverse_3])}\")\n        report.append(f\"- 基于最低平均相似度选择\")\n        report.append(f\"- 特点: 平衡多样性与覆盖度\")\n        report.append(\"\")\n    \n    # 实施建议\n    report.append(\"## 🚀 实施建议\")\n    report.append(\"\")\n    report.append(\"1. **Teacher模型特点**:\")\n    report.append(\"   - 比simple版本提供更好的特征学习能力\")\n    report.append(\"   - 训练时间较长，建议离线训练\")\n    report.append(\"   - 适合对推荐质量要求较高的场景\")\n    report.append(\"\")\n    report.append(\"2. **Ensemble策略**:\")\n    if min_pair:\n        report.append(f\"   - 多样性优先: 使用 {min_pair[0]} + {min_pair[1]} 组合\")\n    report.append(\"   - 可以根据业务需求调整权重分配\")\n    report.append(\"\")\n    report.append(\"3. **优化方向**:\")\n    report.append(\"   - 增加训练数据量和epochs提升模型效果\")\n    report.append(\"   - 使用GPU加速训练过程\")\n    report.append(\"   - 定期重新评估模型一致性\")\n    \n    # 保存报告\n    report_content = \"\\n\".join(report)\n    \n    with open(\"SIMPLIFIED_TEACHER_ANALYSIS.md\", \"w\", encoding='utf-8') as f:\n        f.write(report_content)\n    \n    print(\"✅ Teacher模型分析报告已保存到 SIMPLIFIED_TEACHER_ANALYSIS.md\")\n    \n    # 显示关键结果\n    print(\"\\n🎉 Teacher模型一致性分析完成！\")\n    print(\"\\n📊 关键结果:\")\n    \n    if min_pair:\n        print(f\"🏆 最佳互补组合: {min_pair[0]} + {min_pair[1]} (Jaccard={min_jaccard:.4f})\")\n    \n    if max_pair:\n        print(f\"⚠️  最相似组合: {max_pair[0]} + {max_pair[1]} (Jaccard={max_jaccard:.4f})\")\n    \n    print(f\"📈 成功训练的Teacher模型: {list(trained_models.keys())}\")\n    \n    return {\n        'trained_models': list(trained_models.keys()),\n        'jaccard_matrix': jaccard_matrix,\n        'best_complementary': min_pair if min_pair else None,\n        'performance_stats': performance_stats\n    }\n\n\nif __name__ == \"__main__\":\n    run_simple_teacher_experiment()
