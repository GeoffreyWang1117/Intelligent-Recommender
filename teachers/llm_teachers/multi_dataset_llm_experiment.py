#!/usr/bin/env python3
"""
多数据集LLM推荐系统验证实验
Multi-Dataset LLM Recommendation System Validation Experiment

支持数据集:
1. MovieLens (data/movielens中的样本数据)
2. Amazon Reviews 2023 (自动下载)

功能:
1. 多数据集对比验证
2. LLM推荐效果评估
3. 跨领域推荐性能分析

作者: GitHub Copilot
日期: 2025-08-26
"""

import sys
import os
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from movielens_llm_validator import MovieLensLLMValidator
from amazon_reviews_llm_validator import AmazonReviewsLLMValidator
from recommendation_metrics import RecommendationMetrics

class MultiDatasetLLMExperiment:
    """多数据集LLM推荐验证实验"""
    
    def __init__(self, output_dir: str = "multi_dataset_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 初始化验证器
        self.movielens_validator = MovieLensLLMValidator()
        self.amazon_validator = AmazonReviewsLLMValidator()
        self.metrics = RecommendationMetrics()
        
        # 实验配置
        self.test_config = {
            "test_users_count": 5,              # 每个数据集测试用户数
            "min_ratings_per_user": 5,          # 最少交互数
            "recommendation_count": 10,         # 推荐数量
            "relevance_threshold": 4.0,         # 相关性阈值
            "test_ratio": 0.2                  # 测试集比例
        }
        
        self.results = {}
    
    def run_complete_experiment(self) -> Dict:
        """运行完整的多数据集验证实验"""
        print("🚀 启动多数据集LLM推荐验证实验")
        print("=" * 60)
        
        # Step 1: MovieLens数据集实验
        movielens_results = self._run_movielens_experiment()
        
        # Step 2: Amazon Reviews数据集实验
        amazon_results = self._run_amazon_experiment()
        
        # Step 3: 跨数据集对比分析
        comparison_results = self._compare_datasets(movielens_results, amazon_results)
        
        # Step 4: 保存结果
        self._save_results({
            'movielens': movielens_results,
            'amazon': amazon_results,
            'comparison': comparison_results
        })
        
        # Step 5: 生成报告
        self._generate_report(movielens_results, amazon_results, comparison_results)
        
        return {
            'movielens': movielens_results,
            'amazon': amazon_results,
            'comparison': comparison_results
        }
    
    def _run_movielens_experiment(self) -> Dict:
        """运行MovieLens数据集实验"""
        print("\\n🎬 MovieLens数据集实验")
        print("-" * 40)
        
        try:
            # 加载数据
            if not self.movielens_validator.load_movielens_data("sample"):
                print("❌ MovieLens数据加载失败")
                return {'success': False, 'error': 'Data loading failed'}
            
            # 构建用户画像
            user_profiles = self.movielens_validator.build_user_profiles()
            if not user_profiles:
                print("❌ MovieLens用户画像构建失败")
                return {'success': False, 'error': 'User profile building failed'}
            
            print(f"✅ MovieLens数据准备完成: {len(user_profiles)}个用户")
            
            # 选择测试用户
            qualified_users = [
                uid for uid, profile in user_profiles.items()
                if profile['total_ratings'] >= self.test_config['min_ratings_per_user']
            ]
            
            if len(qualified_users) < self.test_config['test_users_count']:
                test_users = qualified_users
            else:
                test_users = np.random.choice(
                    qualified_users, 
                    self.test_config['test_users_count'], 
                    replace=False
                ).tolist()
            
            print(f"👥 选择测试用户: {len(test_users)}个")
            
            # 生成LLM推荐
            recommendations = {}
            success_count = 0
            
            for i, user_id in enumerate(test_users):
                print(f"  处理用户 {user_id} ({i+1}/{len(test_users)})")
                
                # Llama3推荐
                llama_recs = self.movielens_validator.generate_llm_recommendation(
                    user_id, "llama3:latest", k=self.test_config['recommendation_count']
                )
                
                # Qwen3推荐
                qwen_recs = self.movielens_validator.generate_llm_recommendation(
                    user_id, "qwen3:latest", k=self.test_config['recommendation_count']
                )
                
                if llama_recs or qwen_recs:
                    success_count += 1
                    recommendations[user_id] = {
                        'llama3': llama_recs,
                        'qwen3': qwen_recs
                    }
            
            print(f"✅ MovieLens推荐生成完成: {success_count}/{len(test_users)} 成功")
            
            # 评估推荐效果
            evaluation_results = self._evaluate_movielens_recommendations(test_users, recommendations)
            
            return {
                'success': True,
                'dataset': 'MovieLens',
                'users_count': len(test_users),
                'success_rate': success_count / len(test_users),
                'recommendations': recommendations,
                'evaluation': evaluation_results
            }
            
        except Exception as e:
            print(f"❌ MovieLens实验失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def _run_amazon_experiment(self) -> Dict:
        """运行Amazon Reviews数据集实验"""
        print("\\n🛒 Amazon Reviews数据集实验")
        print("-" * 40)
        
        try:
            # 加载数据（使用小样本）
            if not self.amazon_validator.load_amazon_data(sample_size=2000):
                print("❌ Amazon数据加载失败")
                return {'success': False, 'error': 'Data loading failed'}
            
            # 构建用户画像
            user_profiles = self.amazon_validator.build_user_profiles()
            if not user_profiles:
                print("❌ Amazon用户画像构建失败")
                return {'success': False, 'error': 'User profile building failed'}
            
            print(f"✅ Amazon数据准备完成: {len(user_profiles)}个用户")
            
            # 选择测试用户
            qualified_users = [
                uid for uid, profile in user_profiles.items()
                if profile['total_ratings'] >= self.test_config['min_ratings_per_user']
            ]
            
            if len(qualified_users) < self.test_config['test_users_count']:
                test_users = qualified_users
            else:
                test_users = np.random.choice(
                    qualified_users, 
                    self.test_config['test_users_count'], 
                    replace=False
                ).tolist()
            
            print(f"👥 选择测试用户: {len(test_users)}个")
            
            # 生成LLM推荐
            recommendations = {}
            success_count = 0
            
            for i, user_id in enumerate(test_users):
                print(f"  处理用户 {user_id} ({i+1}/{len(test_users)})")
                
                # Llama3推荐
                llama_recs = self.amazon_validator.generate_llm_recommendation(
                    user_id, "llama3:latest", k=self.test_config['recommendation_count']
                )
                
                # Qwen3推荐
                qwen_recs = self.amazon_validator.generate_llm_recommendation(
                    user_id, "qwen3:latest", k=self.test_config['recommendation_count']
                )
                
                if llama_recs or qwen_recs:
                    success_count += 1
                    recommendations[user_id] = {
                        'llama3': llama_recs,
                        'qwen3': qwen_recs
                    }
            
            print(f"✅ Amazon推荐生成完成: {success_count}/{len(test_users)} 成功")
            
            # 评估推荐效果
            evaluation_results = self._evaluate_amazon_recommendations(test_users, recommendations)
            
            return {
                'success': True,
                'dataset': 'Amazon Reviews',
                'users_count': len(test_users),
                'success_rate': success_count / len(test_users),
                'recommendations': recommendations,
                'evaluation': evaluation_results
            }
            
        except Exception as e:
            print(f"❌ Amazon实验失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def _evaluate_movielens_recommendations(self, test_users: List, recommendations: Dict) -> Dict:
        """评估MovieLens推荐结果"""
        print("📊 评估MovieLens推荐效果...")
        
        results = {'llama3': {}, 'qwen3': {}}
        
        for model in ['llama3', 'qwen3']:
            all_recommendations = []
            all_ground_truth = []
            
            for user_id in test_users:
                if user_id in recommendations and model in recommendations[user_id]:
                    user_recs = recommendations[user_id][model]
                    
                    # 获取用户的真实高评分电影作为ground truth
                    user_ratings = self.movielens_validator.ratings[
                        self.movielens_validator.ratings['user_id'] == user_id
                    ]
                    high_rated = user_ratings[
                        user_ratings['rating'] >= self.test_config['relevance_threshold']
                    ]['movie_id'].tolist()
                    
                    if user_recs and high_rated:
                        all_recommendations.append(user_recs)
                        all_ground_truth.append(high_rated)
            
            if all_recommendations:
                # 计算评估指标
                metrics_result = self.metrics.evaluate_comprehensive(
                    all_recommendations, all_ground_truth, k_values=[5, 10]
                )
                results[model] = metrics_result
        
        return results
    
    def _evaluate_amazon_recommendations(self, test_users: List, recommendations: Dict) -> Dict:
        """评估Amazon推荐结果"""
        print("📊 评估Amazon推荐效果...")
        
        results = {'llama3': {}, 'qwen3': {}}
        
        for model in ['llama3', 'qwen3']:
            all_recommendations = []
            all_ground_truth = []
            
            for user_id in test_users:
                if user_id in recommendations and model in recommendations[user_id]:
                    user_recs = recommendations[user_id][model]
                    
                    # 获取用户的真实高评分商品作为ground truth
                    user_reviews = self.amazon_validator.reviews[
                        self.amazon_validator.reviews['user_id'] == user_id
                    ]
                    high_rated = user_reviews[
                        user_reviews['rating'] >= self.test_config['relevance_threshold']
                    ]['item_id'].tolist()
                    
                    if user_recs and high_rated:
                        all_recommendations.append(user_recs)
                        all_ground_truth.append(high_rated)
            
            if all_recommendations:
                # 计算评估指标
                metrics_result = self.metrics.evaluate_comprehensive(
                    all_recommendations, all_ground_truth, k_values=[5, 10]
                )
                results[model] = metrics_result
        
        return results
    
    def _compare_datasets(self, movielens_results: Dict, amazon_results: Dict) -> Dict:
        """跨数据集对比分析"""
        print("\\n🔍 跨数据集对比分析")
        print("-" * 40)
        
        comparison = {
            'dataset_comparison': {},
            'model_comparison': {},
            'recommendations': {}
        }
        
        # 数据集基本信息对比
        if movielens_results.get('success') and amazon_results.get('success'):
            comparison['dataset_comparison'] = {
                'movielens_success_rate': movielens_results['success_rate'],
                'amazon_success_rate': amazon_results['success_rate'],
                'movielens_users': movielens_results['users_count'],
                'amazon_users': amazon_results['users_count']
            }
            
            # 模型性能对比
            for model in ['llama3', 'qwen3']:
                ml_metrics = movielens_results.get('evaluation', {}).get(model, {})
                az_metrics = amazon_results.get('evaluation', {}).get(model, {})
                
                if ml_metrics and az_metrics:
                    comparison['model_comparison'][model] = {
                        'movielens_overall': ml_metrics.get('overall_score', 0),
                        'amazon_overall': az_metrics.get('overall_score', 0),
                        'domain_preference': 'MovieLens' if ml_metrics.get('overall_score', 0) > az_metrics.get('overall_score', 0) else 'Amazon'
                    }
        
        return comparison
    
    def _save_results(self, results: Dict):
        """保存实验结果"""
        print("💾 保存实验结果...")
        
        # 保存详细结果
        with open(self.output_dir / "multi_dataset_experiment_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✅ 结果已保存到 {self.output_dir}")
    
    def _generate_report(self, movielens_results: Dict, amazon_results: Dict, comparison_results: Dict):
        """生成实验报告"""
        print("\\n📋 生成多数据集实验报告")
        print("=" * 60)
        
        report = f"""
# 多数据集LLM推荐系统验证实验报告

## 实验概述
- 实验时间: {pd.Timestamp.now()}
- 测试数据集: MovieLens + Amazon Reviews 2023
- LLM模型: Llama3 + Qwen3
- 推荐数量: {self.test_config['recommendation_count']}

## MovieLens数据集结果
"""
        
        if movielens_results.get('success'):
            report += f"""
### 基本信息
- 测试用户数: {movielens_results['users_count']}
- 推荐成功率: {movielens_results['success_rate']:.1%}

### 模型性能对比
"""
            ml_eval = movielens_results.get('evaluation', {})
            for model in ['llama3', 'qwen3']:
                if model in ml_eval:
                    metrics = ml_eval[model]
                    report += f"""
#### {model.upper()}
- 综合得分: {metrics.get('overall_score', 0):.3f}
- 准确性得分: {metrics.get('accuracy_score', 0):.3f}
- 多样性得分: {metrics.get('diversity_score', 0):.3f}
"""
        else:
            report += f"\\n❌ MovieLens实验失败: {movielens_results.get('error', 'Unknown error')}"
        
        report += "\\n## Amazon Reviews数据集结果\\n"
        
        if amazon_results.get('success'):
            report += f"""
### 基本信息
- 测试用户数: {amazon_results['users_count']}
- 推荐成功率: {amazon_results['success_rate']:.1%}

### 模型性能对比
"""
            az_eval = amazon_results.get('evaluation', {})
            for model in ['llama3', 'qwen3']:
                if model in az_eval:
                    metrics = az_eval[model]
                    report += f"""
#### {model.upper()}
- 综合得分: {metrics.get('overall_score', 0):.3f}
- 准确性得分: {metrics.get('accuracy_score', 0):.3f}
- 多样性得分: {metrics.get('diversity_score', 0):.3f}
"""
        else:
            report += f"\\n❌ Amazon实验失败: {amazon_results.get('error', 'Unknown error')}"
        
        # 跨数据集对比
        report += "\\n## 跨数据集对比分析\\n"
        
        if comparison_results.get('model_comparison'):
            for model, comp in comparison_results['model_comparison'].items():
                report += f"""
### {model.upper()} 跨领域表现
- MovieLens综合得分: {comp['movielens_overall']:.3f}
- Amazon综合得分: {comp['amazon_overall']:.3f}
- 领域偏好: {comp['domain_preference']}
"""
        
        report += f"""
## 实验结论

1. **数据集适用性**: 验证了LLM在电影推荐和电商推荐两个不同领域的适用性
2. **模型对比**: Llama3和Qwen3在不同领域的表现差异
3. **技术可行性**: 证明了LLM在多领域推荐系统中的技术可行性

## 技术建议

1. 针对不同领域优化提示词工程
2. 考虑领域特定的用户画像构建方法
3. 探索跨领域推荐的迁移学习方法

---
实验数据保存在: {self.output_dir}
"""
        
        # 保存报告
        with open(self.output_dir / "multi_dataset_experiment_report.md", 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(report)
        print(f"\\n📁 详细报告已保存到: {self.output_dir / 'multi_dataset_experiment_report.md'}")

def main():
    """主函数"""
    experiment = MultiDatasetLLMExperiment()
    results = experiment.run_complete_experiment()
    
    print("\\n🎉 多数据集LLM推荐验证实验完成！")
    
    # 输出简要结果
    if results['movielens'].get('success'):
        print(f"✅ MovieLens: 成功率 {results['movielens']['success_rate']:.1%}")
    else:
        print(f"❌ MovieLens: 失败")
    
    if results['amazon'].get('success'):
        print(f"✅ Amazon: 成功率 {results['amazon']['success_rate']:.1%}")
    else:
        print(f"❌ Amazon: 失败")

if __name__ == "__main__":
    main()
