#!/usr/bin/env python3
"""
Teacher模型一致性分析实验

使用完整版Teacher模型重新执行一致性分析，包括：
- DeepFM、AutoInt、DIN的完整版本
- 新增Transformer4Rec、DCNv2、xDeepFM
- 重新计算Jaccard相似度和Kendall Tau相关性
- 基于新结果优化ensemble策略
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import logging
import time
from collections import defaultdict

# 导入评估和一致性分析模块
from evaluation.consistency_analysis import AlgorithmConsistencyEvaluator
from evaluation.metrics import RecommendationMetrics
from models import create_recommender, get_available_algorithms

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_enhanced_movielens_simulation(num_users: int = 1000, num_items: int = 500,
                                       num_interactions: int = 10000) -> pd.DataFrame:
    """创建增强的MovieLens模拟数据"""
    print(f"🎬 创建增强MovieLens模拟数据...")
    print(f"用户数: {num_users}, 物品数: {num_items}, 交互数: {num_interactions}")
    
    np.random.seed(42)  # 确保可重复性
    
    interactions = []
    
    # 模拟用户行为模式
    user_preferences = {}
    for user_id in range(1, num_users + 1):
        # 每个用户有不同的评分偏好
        user_preferences[user_id] = {
            'mean_rating': np.random.normal(3.5, 0.5),
            'rating_std': np.random.uniform(0.5, 1.5),
            'activity_level': np.random.exponential(0.1)  # 活跃度
        }
    
    # 生成交互数据
    for _ in range(num_interactions):
        # 根据活跃度选择用户（更活跃的用户更可能产生交互）
        user_weights = [user_preferences[uid]['activity_level'] for uid in range(1, num_users + 1)]
        user_id = np.random.choice(range(1, num_users + 1), p=user_weights/np.sum(user_weights))
        
        # 随机选择物品（可以加入流行度偏向）
        item_id = np.random.randint(1, num_items + 1)
        
        # 生成评分（基于用户偏好）
        user_pref = user_preferences[user_id]
        rating = np.clip(
            np.random.normal(user_pref['mean_rating'], user_pref['rating_std']),
            1.0, 5.0
        )
        rating = round(rating)
        
        # 生成时间戳（模拟时序）
        timestamp = int(time.time()) + np.random.randint(-86400*30, 86400*30)  # ±30天
        
        interactions.append({
            'user_id': user_id,
            'item_id': item_id,
            'rating': rating,
            'timestamp': timestamp
        })
    
    train_data = pd.DataFrame(interactions)
    
    # 去重（用户-物品对）保留最新的交互
    train_data = train_data.sort_values('timestamp').groupby(['user_id', 'item_id']).last().reset_index()
    
    print(f"✅ 生成了 {len(train_data)} 条唯一交互数据")
    print(f"评分分布: {train_data['rating'].value_counts().sort_index().to_dict()}")
    
    return train_data


class TeacherModelConsistencyExperiment:
    """Teacher模型一致性实验"""
    
    def __init__(self):
        self.algorithms = [
            'svd',              # 基础算法
            'deepfm',           # 完整版DeepFM Teacher
            'autoint',          # 完整版AutoInt Teacher  
            'din',              # 完整版DIN Teacher
            'transformer4rec',  # 新Teacher模型
            'dcnv2',           # 新Teacher模型
            'xdeepfm'          # 新Teacher模型
        ]
        
        self.train_data = None
        self.trained_models = {}
        self.evaluator = AlgorithmConsistencyEvaluator()
        self.metrics = RecommendationMetrics()
        
        # 优化的Teacher模型配置
        self.model_configs = {
            'svd': {
                'n_components': 50,
                'learning_rate': 0.01,
                'regularization': 0.02,
                'epochs': 100
            },
            'deepfm': {
                'embedding_dim': 16,
                'learning_rate': 0.001,
                'epochs': 60,
                'batch_size': 256,
                'early_stopping': True,
                'patience': 8
            },
            'autoint': {
                'embedding_dim': 16,
                'learning_rate': 0.001,
                'epochs': 60,
                'num_heads': 8,
                'num_layers': 6,
                'early_stopping': True,
                'patience': 8
            },
            'din': {
                'embedding_dim': 16,
                'learning_rate': 0.001,
                'epochs': 60,
                'hidden_dim': 128,
                'early_stopping': True,
                'patience': 8
            },
            'transformer4rec': {
                'embedding_dim': 64,
                'num_heads': 8,
                'num_layers': 6,
                'learning_rate': 0.001,
                'epochs': 50,
                'batch_size': 256,
                'early_stopping': True,
                'patience': 8
            },
            'dcnv2': {
                'embedding_dim': 16,
                'cross_layers': 4,
                'deep_layers': [512, 256, 128],
                'learning_rate': 0.001,
                'epochs': 50,
                'batch_size': 256,
                'early_stopping': True,
                'patience': 8
            },
            'xdeepfm': {
                'embedding_dim': 16,
                'cin_layer_sizes': [200, 200, 200],
                'dnn_hidden_dims': [400, 400, 400],
                'learning_rate': 0.001,
                'epochs': 50,
                'batch_size': 256,
                'early_stopping': True,
                'patience': 8
            }
        }
    
    def train_all_models(self, train_data: pd.DataFrame):
        """训练所有Teacher模型"""
        print("\n🚀 开始训练所有Teacher模型...")
        self.train_data = train_data
        
        for algo_name in self.algorithms:
            print(f"\n📚 训练 {algo_name} (Teacher模型)...")
            
            try:
                # 获取配置
                config = self.model_configs.get(algo_name, {})
                
                # 创建模型
                model = create_recommender(algo_name, **config)
                if model is None:
                    print(f"❌ 无法创建算法: {algo_name}")
                    continue
                
                # 训练模型
                start_time = time.time()
                model.fit(train_data, **config)
                end_time = time.time()
                
                if model.is_trained:
                    self.trained_models[algo_name] = model
                    print(f"✅ {algo_name} 训练成功 (耗时: {end_time - start_time:.2f}秒)")
                else:
                    print(f"❌ {algo_name} 训练失败")
                    
            except Exception as e:
                print(f"❌ 训练 {algo_name} 时出错: {e}")
        
        print(f"\n🎯 训练完成！成功训练了 {len(self.trained_models)} 个Teacher模型")
        print(f"成功的模型: {list(self.trained_models.keys())}")
    
    def evaluate_individual_performance(self):
        """评估单个模型性能"""
        print("\n📊 评估Teacher模型个体性能...")
        
        results = {}
        
        for algo_name, model in self.trained_models.items():
            print(f"评估 {algo_name}...")
            
            try:
                # 生成推荐
                test_users = self.train_data['user_id'].unique()[:100]  # 测试前100个用户
                recommendations = {}
                
                for user_id in test_users:
                    try:
                        recs = model.get_user_recommendations(user_id, top_k=50)
                        if recs:
                            recommendations[user_id] = [rec['item_id'] for rec in recs]
                    except:
                        continue
                
                if not recommendations:
                    print(f"❌ {algo_name} 无法生成推荐")
                    continue
                
                # 计算指标
                metrics = self.metrics.calculate_metrics(
                    recommendations, self.train_data, k_values=[10, 50]
                )
                
                results[algo_name] = {
                    'R@10': metrics.get('R@10', 0.0),
                    'N@10': metrics.get('N@10', 0.0),
                    'R@50': metrics.get('R@50', 0.0),
                    'N@50': metrics.get('N@50', 0.0),
                    'Coverage': metrics.get('Coverage', 0.0),
                    'num_recommendations': len(recommendations)
                }
                
                print(f"  R@10: {results[algo_name]['R@10']:.4f}")
                print(f"  N@10: {results[algo_name]['N@10']:.4f}")
                print(f"  R@50: {results[algo_name]['R@50']:.4f}")
                print(f"  N@50: {results[algo_name]['N@50']:.4f}")
                
            except Exception as e:
                print(f"❌ 评估 {algo_name} 时出错: {e}")
        
        return results
    
    def analyze_teacher_consistency(self):
        """分析Teacher模型间的一致性"""
        print("\n🔍 分析Teacher模型一致性...")
        
        # 生成所有模型的推荐
        all_recommendations = {}
        test_users = self.train_data['user_id'].unique()[:50]  # 测试50个用户
        
        for algo_name, model in self.trained_models.items():
            print(f"生成 {algo_name} 的推荐...")
            recommendations = {}
            
            for user_id in test_users:
                try:
                    recs = model.get_user_recommendations(user_id, top_k=10)
                    if recs:
                        recommendations[user_id] = [rec['item_id'] for rec in recs]
                except:
                    continue
            
            if recommendations:
                all_recommendations[algo_name] = recommendations
                print(f"  成功为 {len(recommendations)} 个用户生成推荐")
        
        if len(all_recommendations) < 2:
            print("❌ 无法进行一致性分析，成功的模型少于2个")
            return None
        
        # 计算一致性指标
        print("\n计算一致性指标...")
        consistency_results = self.evaluator.analyze_algorithm_consistency(
            all_recommendations, k_values=[5, 10]
        )
        
        return consistency_results
    
    def generate_teacher_report(self, performance_results, consistency_results):
        """生成Teacher模型分析报告"""
        print("\n📝 生成Teacher模型分析报告...")
        
        report = []
        report.append("# 🎓 Teacher模型一致性分析报告")
        report.append(f"实验时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 实验概述
        report.append("## 📊 实验概述")
        report.append(f"- **Teacher模型数量**: {len(self.trained_models)}")
        report.append(f"- **成功训练模型**: {list(self.trained_models.keys())}")
        report.append(f"- **数据集规模**: {len(self.train_data)} 条交互")
        report.append(f"- **用户数**: {len(self.train_data['user_id'].unique())}")
        report.append(f"- **物品数**: {len(self.train_data['item_id'].unique())}")
        report.append("")
        
        # 个体性能
        if performance_results:
            report.append("## 🏆 Teacher模型个体性能")
            report.append("")
            report.append("| 模型 | R@10 | N@10 | R@50 | N@50 | Coverage |")
            report.append("|------|------|------|------|------|----------|")
            
            for algo_name, metrics in performance_results.items():
                report.append(f"| {algo_name} | {metrics['R@10']:.4f} | {metrics['N@10']:.4f} | "
                            f"{metrics['R@50']:.4f} | {metrics['N@50']:.4f} | {metrics['Coverage']:.4f} |")
            report.append("")
            
            # 性能排名
            sorted_by_r10 = sorted(performance_results.items(), 
                                 key=lambda x: x[1]['R@10'], reverse=True)
            
            report.append("### 📈 性能排名 (按R@10)")
            for i, (algo_name, metrics) in enumerate(sorted_by_r10, 1):
                report.append(f"{i}. **{algo_name}**: R@10={metrics['R@10']:.4f}")
            report.append("")
        
        # 一致性分析
        if consistency_results:
            report.append("## 🔄 Teacher模型一致性分析")
            report.append("")
            
            if 'jaccard_matrix' in consistency_results:
                report.append("### Jaccard相似度矩阵")
                jaccard_matrix = consistency_results['jaccard_matrix']
                algorithms = list(jaccard_matrix.keys())
                
                # 找到最低和最高Jaccard相似度
                min_jaccard = float('inf')
                max_jaccard = 0.0
                min_pair = None
                max_pair = None
                
                for algo1 in algorithms:
                    for algo2 in algorithms:
                        if algo1 != algo2:
                            similarity = jaccard_matrix[algo1][algo2]
                            if similarity < min_jaccard:
                                min_jaccard = similarity
                                min_pair = (algo1, algo2)
                            if similarity > max_jaccard:
                                max_jaccard = similarity
                                max_pair = (algo1, algo2)
                
                report.append(f"- **最强互补组合**: {min_pair[0]} + {min_pair[1]} (Jaccard={min_jaccard:.4f})")
                report.append(f"- **最相似组合**: {max_pair[0]} + {max_pair[1]} (Jaccard={max_jaccard:.4f})")
                report.append("")
            
            # 多样性分析
            if 'diversity_scores' in consistency_results:
                report.append("### 📊 Teacher模型多样性评分")
                diversity_scores = consistency_results['diversity_scores']
                sorted_diversity = sorted(diversity_scores.items(), 
                                        key=lambda x: x[1], reverse=True)
                
                for i, (algo_name, score) in enumerate(sorted_diversity, 1):
                    report.append(f"{i}. **{algo_name}**: {score:.4f}")
                report.append("")
        
        # Teacher模型ensemble建议
        report.append("## 🎯 Teacher模型Ensemble策略建议")
        report.append("")
        
        if consistency_results and 'jaccard_matrix' in consistency_results:
            # 基于一致性分析的建议
            jaccard_matrix = consistency_results['jaccard_matrix']
            algorithms = list(jaccard_matrix.keys())
            
            # 找到最互补的组合
            min_jaccard = float('inf')
            best_complementary = None
            
            for i, algo1 in enumerate(algorithms):
                for j, algo2 in enumerate(algorithms[i+1:], i+1):
                    similarity = jaccard_matrix[algo1][algo2]
                    if similarity < min_jaccard:
                        min_jaccard = similarity
                        best_complementary = (algo1, algo2)
            
            if best_complementary:
                report.append(f"### 🥇 最佳互补策略")
                report.append(f"**推荐组合**: {best_complementary[0]} + {best_complementary[1]}")
                report.append(f"- Jaccard相似度: {min_jaccard:.4f}")
                report.append(f"- 特点: 极低重叠度，最大化推荐多样性")
                report.append("")
            
            # 三模型组合建议
            if len(algorithms) >= 3:
                diversity_scores = consistency_results.get('diversity_scores', {})
                top_3_diverse = sorted(diversity_scores.items(), 
                                     key=lambda x: x[1], reverse=True)[:3]
                
                report.append(f"### 🥈 最佳多样化策略")
                report.append(f"**推荐组合**: {' + '.join([algo for algo, _ in top_3_diverse])}")
                report.append(f"- 多样性评分: {[f'{algo}({score:.3f})' for algo, score in top_3_diverse]}")
                report.append(f"- 特点: 平衡性能与多样性，综合效果最佳")
                report.append("")
            
            # 质量优先建议
            if performance_results:
                top_performer = max(performance_results.items(), key=lambda x: x[1]['R@10'])
                report.append(f"### 🥉 质量优先策略")
                report.append(f"**主导模型**: {top_performer[0]} (R@10={top_performer[1]['R@10']:.4f})")
                report.append(f"- 特点: 以最高性能模型为主，辅以互补模型")
                report.append("")
        
        # 实施建议
        report.append("## 🚀 实施建议")
        report.append("")
        report.append("1. **生产环境部署**:")
        report.append("   - 推荐使用最佳互补策略以获得最高多样性")
        report.append("   - 在计算资源有限时使用质量优先策略")
        report.append("")
        report.append("2. **模型更新策略**:")
        report.append("   - Teacher模型需要更多训练时间，建议离线训练")
        report.append("   - 定期重新评估一致性，动态调整ensemble权重")
        report.append("")
        report.append("3. **性能监控**:")
        report.append("   - 监控ensemble与单模型的性能差异")
        report.append("   - 关注用户反馈，调整推荐策略")
        
        # 保存报告
        report_content = "\n".join(report)
        
        with open("TEACHER_MODEL_CONSISTENCY_REPORT.md", "w", encoding='utf-8') as f:
            f.write(report_content)
        
        print("✅ Teacher模型分析报告已保存到 TEACHER_MODEL_CONSISTENCY_REPORT.md")
        
        return report_content
    
    def run_complete_experiment(self):
        """运行完整的Teacher模型实验"""
        print("🎓 Teacher模型一致性分析实验开始！")
        print("=" * 60)
        
        # 1. 创建数据
        train_data = create_enhanced_movielens_simulation(
            num_users=800, num_items=300, num_interactions=8000
        )
        
        # 2. 训练所有Teacher模型
        self.train_all_models(train_data)
        
        if len(self.trained_models) < 2:
            print("❌ 训练成功的模型少于2个，无法进行一致性分析")
            return
        
        # 3. 评估个体性能
        performance_results = self.evaluate_individual_performance()
        
        # 4. 分析一致性
        consistency_results = self.analyze_teacher_consistency()
        
        # 5. 生成报告
        report = self.generate_teacher_report(performance_results, consistency_results)
        
        print("\n🎉 Teacher模型一致性分析实验完成！")
        print(f"📋 报告已生成，详细结果请查看 TEACHER_MODEL_CONSISTENCY_REPORT.md")
        
        return {
            'performance_results': performance_results,
            'consistency_results': consistency_results,
            'report': report,
            'trained_models': list(self.trained_models.keys())
        }


def main():
    """主函数"""
    print("🎓 Teacher模型一致性分析实验")
    print("使用完整版Teacher模型重新评估算法一致性")
    print("=" * 60)
    
    # 检查可用算法
    available_algorithms = get_available_algorithms()
    print(f"可用算法: {available_algorithms}")
    
    # 运行实验
    experiment = TeacherModelConsistencyExperiment()
    results = experiment.run_complete_experiment()
    
    if results:
        print("\n🎯 实验结果摘要:")
        print(f"- 成功训练模型: {results['trained_models']}")
        
        if results['performance_results']:
            best_model = max(results['performance_results'].items(), 
                           key=lambda x: x[1]['R@10'])
            print(f"- 最佳性能模型: {best_model[0]} (R@10={best_model[1]['R@10']:.4f})")
        
        if results['consistency_results'] and 'jaccard_matrix' in results['consistency_results']:
            print("- 一致性分析完成，详细结果请查看报告")


if __name__ == "__main__":
    main()
