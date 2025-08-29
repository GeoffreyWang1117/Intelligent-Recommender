#!/usr/bin/env python3
"""
LLM推荐系统完整验证实验
Complete LLM Recommendation System Validation Experiment

功能:
1. 使用真实MovieLens数据
2. 生成LLM推荐结果
3. 使用标准推荐系统评价指标验证效果
4. 对比双语LLM模型性能
5. 输出详细的评估报告

作者: GitHub Copilot
日期: 2025-08-18
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
from recommendation_metrics import RecommendationMetrics

class CompleteLLMValidationExperiment:
    """完整的LLM推荐系统验证实验"""
    
    def __init__(self, output_dir: str = "experiment_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 初始化组件
        self.validator = MovieLensLLMValidator()
        self.metrics = RecommendationMetrics()
        
        # 实验配置
        self.test_config = {
            "test_users_count": 10,         # 测试用户数量
            "min_ratings_per_user": 10,     # 每个用户最少评分数
            "candidate_movies_count": 30,   # 候选电影数量
            "recommendation_count": 10,     # 推荐电影数量
            "relevance_threshold": 4.0,     # 相关性阈值 (>=4.0认为相关)
            "test_ratio": 0.2              # 测试集比例
        }
        
        self.experiment_results = {}
    
    def run_complete_experiment(self) -> Dict:
        """运行完整的验证实验"""
        print("🧪 开始LLM推荐系统完整验证实验")
        print("=" * 60)
        
        # Step 1: 数据准备
        if not self._prepare_data():
            print("❌ 数据准备失败，实验终止")
            return {}
        
        # Step 2: 选择测试用户
        test_users = self._select_test_users()
        if not test_users:
            print("❌ 测试用户选择失败，实验终止")
            return {}
        
        # Step 3: 划分训练/测试集
        train_data, test_data = self._split_train_test(test_users)
        
        # Step 4: 生成LLM推荐
        llm_recommendations = self._generate_llm_recommendations(test_users, train_data)
        
        # Step 5: 评估推荐效果
        evaluation_results = self._evaluate_recommendations(
            test_users, test_data, llm_recommendations
        )
        
        # Step 6: 保存实验结果
        self._save_experiment_results(evaluation_results)
        
        # Step 7: 生成报告
        self._generate_experiment_report(evaluation_results)
        
        return evaluation_results
    
    def _prepare_data(self) -> bool:
        """准备实验数据"""
        print("📊 准备实验数据...")
        
        # 优先尝试加载sample数据
        if self.validator.load_movielens_data("sample"):
            print("✅ 成功加载data/movielens中的MovieLens数据")
        elif self.validator.load_movielens_data("small"):
            print("✅ 成功加载标准MovieLens 100K数据")
        else:
            print("⚠️  真实MovieLens数据不可用，使用样本数据")
            if not self.validator.create_sample_data():
                return False
        
        # 构建用户画像
        user_profiles = self.validator.build_user_profiles()
        if not user_profiles:
            print("❌ 用户画像构建失败")
            return False
        
        print(f"✅ 数据准备完成: {len(user_profiles)}个用户画像")
        return True
    
    def _select_test_users(self) -> List[int]:
        """选择符合条件的测试用户"""
        print("👥 选择测试用户...")
        
        qualified_users = []
        for user_id, profile in self.validator.user_profiles.items():
            if profile['total_ratings'] >= self.test_config['min_ratings_per_user']:
                qualified_users.append(user_id)
        
        # 随机选择测试用户
        if len(qualified_users) > self.test_config['test_users_count']:
            test_users = np.random.choice(
                qualified_users, 
                self.test_config['test_users_count'], 
                replace=False
            ).tolist()
        else:
            test_users = qualified_users
        
        print(f"✅ 选择了{len(test_users)}个测试用户")
        return test_users
    
    def _split_train_test(self, test_users: List[int]) -> Tuple[Dict, Dict]:
        """划分训练集和测试集"""
        print("✂️  划分训练/测试集...")
        
        train_data = {}
        test_data = {}
        
        for user_id in test_users:
            user_ratings = self.validator.ratings[
                self.validator.ratings['user_id'] == user_id
            ].copy()
            
            # 按时间排序
            user_ratings = user_ratings.sort_values('timestamp')
            
            # 划分训练/测试
            n_test = max(1, int(len(user_ratings) * self.test_config['test_ratio']))
            
            train_ratings = user_ratings[:-n_test]
            test_ratings = user_ratings[-n_test:]
            
            train_data[user_id] = train_ratings
            test_data[user_id] = test_ratings
        
        total_train = sum(len(ratings) for ratings in train_data.values())
        total_test = sum(len(ratings) for ratings in test_data.values())
        
        print(f"✅ 数据集划分完成: 训练{total_train}条, 测试{total_test}条")
        return train_data, test_data
    
    def _generate_llm_recommendations(self, test_users: List[int], 
                                    train_data: Dict) -> Dict[str, Dict]:
        """生成LLM推荐结果"""
        print("🤖 生成LLM推荐...")
        
        recommendations = {
            "llama3": {},
            "qwen3": {},
            "generation_info": {}
        }
        
        for i, user_id in enumerate(test_users, 1):
            print(f"  处理用户 {user_id} ({i}/{len(test_users)})")
            
            # 获取候选电影 (用户在训练集中未评分的电影)
            train_movies = set(train_data[user_id]['movie_id']) if user_id in train_data else set()
            all_movies = set(self.validator.movies['movie_id'])
            candidate_movie_ids = list(all_movies - train_movies)
            
            # 限制候选数量以控制计算时间
            if len(candidate_movie_ids) > self.test_config['candidate_movies_count']:
                candidate_movie_ids = np.random.choice(
                    candidate_movie_ids, 
                    self.test_config['candidate_movies_count'], 
                    replace=False
                ).tolist()
            
            # 构建候选电影信息
            candidate_movies = []
            for movie_id in candidate_movie_ids:
                movie_info = self.validator.movies[
                    self.validator.movies['movie_id'] == movie_id
                ].iloc[0]
                candidate_movies.append({
                    'movie_id': movie_id,
                    'title': movie_info['title'],
                    'genres': self._extract_genres(movie_info)
                })
            
            # 生成Llama3推荐
            print(f"    🇺🇸 Llama3推荐...")
            llama3_result = self.validator.get_llm_recommendations(
                user_id, candidate_movies, "primary", 
                top_k=self.test_config['recommendation_count']
            )
            recommendations["llama3"][user_id] = llama3_result
            
            # 生成Qwen3推荐
            print(f"    🇨🇳 Qwen3推荐...")
            qwen3_result = self.validator.get_llm_recommendations(
                user_id, candidate_movies, "secondary",
                top_k=self.test_config['recommendation_count']
            )
            recommendations["qwen3"][user_id] = qwen3_result
            
            # 记录生成信息
            recommendations["generation_info"][user_id] = {
                "candidate_count": len(candidate_movies),
                "llama3_status": llama3_result.get("status", "unknown"),
                "qwen3_status": qwen3_result.get("status", "unknown")
            }
        
        # 统计生成成功率
        llama3_success = sum(1 for info in recommendations["generation_info"].values() 
                           if info["llama3_status"] == "success")
        qwen3_success = sum(1 for info in recommendations["generation_info"].values() 
                          if info["qwen3_status"] == "success")
        
        print(f"✅ LLM推荐生成完成:")
        print(f"   Llama3成功率: {llama3_success}/{len(test_users)} ({llama3_success/len(test_users)*100:.1f}%)")
        print(f"   Qwen3成功率: {qwen3_success}/{len(test_users)} ({qwen3_success/len(test_users)*100:.1f}%)")
        
        return recommendations
    
    def _extract_genres(self, movie_info) -> str:
        """提取电影类型信息"""
        genre_names = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 
                      'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 
                      'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        
        genres = []
        for i, genre in enumerate(genre_names):
            if i < len([col for col in movie_info.index if col.startswith('genre_')]):
                genre_col = f'genre_{i}'
                if genre_col in movie_info.index and movie_info[genre_col] == 1:
                    genres.append(genre)
        
        return ', '.join(genres) if genres else 'Unknown'
    
    def _evaluate_recommendations(self, test_users: List[int], 
                                test_data: Dict, 
                                llm_recommendations: Dict) -> Dict:
        """评估推荐效果"""
        print("📏 评估推荐效果...")
        
        evaluation_results = {}
        
        for model_name in ["llama3", "qwen3"]:
            print(f"  评估{model_name}模型...")
            
            # 准备评估数据
            eval_data = self._prepare_evaluation_data(
                test_users, test_data, llm_recommendations[model_name]
            )
            
            if not eval_data:
                print(f"    ❌ {model_name}评估数据准备失败")
                continue
            
            # 执行综合评估
            results = self.metrics.comprehensive_evaluation(
                eval_data["true_ratings"],
                eval_data["predicted_ratings"], 
                eval_data["test_users_relevant"],
                eval_data["recommendations"],
                eval_data["item_features"],
                eval_data["item_popularity"],
                eval_data["all_items"]
            )
            
            evaluation_results[model_name] = results
            
            print(f"    ✅ {model_name}评估完成")
        
        return evaluation_results
    
    def _prepare_evaluation_data(self, test_users: List[int], 
                               test_data: Dict, 
                               model_recommendations: Dict) -> Dict:
        """准备评估所需的数据格式"""
        
        # 收集真实评分
        true_ratings = []
        for user_id in test_users:
            if user_id in test_data:
                for _, row in test_data[user_id].iterrows():
                    true_ratings.append((user_id, row['movie_id'], row['rating']))
        
        # 收集预测评分
        predicted_ratings = []
        recommendations = {}
        
        for user_id in test_users:
            if (user_id in model_recommendations and 
                model_recommendations[user_id].get("status") == "success"):
                
                recs = model_recommendations[user_id]["recommendations"]
                user_recs = []
                
                for rec in recs:
                    movie_id = rec.get("movie_id")
                    predicted_rating = rec.get("predicted_rating", 3.0)
                    
                    if movie_id:
                        predicted_ratings.append((user_id, movie_id, predicted_rating))
                        user_recs.append(movie_id)
                
                recommendations[user_id] = user_recs
        
        # 收集相关物品 (测试集中评分>=阈值的物品)
        test_users_relevant = {}
        for user_id in test_users:
            relevant_items = []
            if user_id in test_data:
                relevant_movies = test_data[user_id][
                    test_data[user_id]['rating'] >= self.test_config['relevance_threshold']
                ]
                relevant_items = relevant_movies['movie_id'].tolist()
            test_users_relevant[user_id] = relevant_items
        
        # 构建物品特征
        item_features = {}
        for _, movie in self.validator.movies.iterrows():
            genres = self._extract_genres(movie).split(', ')
            item_features[movie['movie_id']] = [g for g in genres if g != 'Unknown']
        
        # 计算物品流行度
        item_popularity = {}
        item_counts = self.validator.ratings['movie_id'].value_counts()
        max_count = item_counts.max()
        for movie_id in self.validator.movies['movie_id']:
            count = item_counts.get(movie_id, 0)
            item_popularity[movie_id] = count / max_count if max_count > 0 else 0
        
        # 所有物品集合
        all_items = set(self.validator.movies['movie_id'])
        
        return {
            "true_ratings": true_ratings,
            "predicted_ratings": predicted_ratings,
            "test_users_relevant": test_users_relevant,
            "recommendations": recommendations,
            "item_features": item_features,
            "item_popularity": item_popularity,
            "all_items": all_items
        }
    
    def _save_experiment_results(self, evaluation_results: Dict):
        """保存实验结果"""
        print("💾 保存实验结果...")
        
        # 保存完整结果
        results_file = self.output_dir / "llm_recommendation_evaluation_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存简化摘要
        summary = {}
        for model_name, results in evaluation_results.items():
            if "overall_score" in results:
                summary[model_name] = {
                    "overall_score": results["overall_score"]["overall_score"],
                    "accuracy_rmse": results["accuracy_metrics"].get("RMSE", "N/A"),
                    "precision_at_10": results["ranking_metrics"].get("Precision@10", "N/A"),
                    "coverage": results["diversity_metrics"].get("Coverage", "N/A"),
                    "novelty": results["novelty_metrics"].get("Novelty", "N/A")
                }
        
        summary_file = self.output_dir / "experiment_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 结果已保存到 {self.output_dir}")
    
    def _generate_experiment_report(self, evaluation_results: Dict):
        """生成实验报告"""
        print("\n" + "="*80)
        print("📋 LLM推荐系统验证实验报告")
        print("="*80)
        
        print(f"📅 实验时间: {pd.Timestamp.now()}")
        print(f"📊 实验配置:")
        for key, value in self.test_config.items():
            print(f"   {key}: {value}")
        
        # 模型对比
        if "llama3" in evaluation_results and "qwen3" in evaluation_results:
            print(f"\n🔄 双语模型对比:")
            
            for metric_category in ["overall_score", "accuracy_metrics", "ranking_metrics", 
                                  "diversity_metrics", "novelty_metrics"]:
                print(f"\n📊 {metric_category}:")
                
                llama3_metrics = evaluation_results["llama3"].get(metric_category, {})
                qwen3_metrics = evaluation_results["qwen3"].get(metric_category, {})
                
                # 找到共同指标
                common_metrics = set(llama3_metrics.keys()) & set(qwen3_metrics.keys())
                
                for metric in sorted(common_metrics):
                    llama3_val = llama3_metrics[metric]
                    qwen3_val = qwen3_metrics[metric]
                    
                    if isinstance(llama3_val, (int, float)) and isinstance(qwen3_val, (int, float)):
                        diff = llama3_val - qwen3_val
                        winner = "🇺🇸" if diff > 0 else "🇨🇳" if diff < 0 else "🤝"
                        print(f"   {metric:20s}: Llama3={llama3_val:7.4f}, Qwen3={qwen3_val:7.4f} {winner}")
        
        # 详细报告
        for model_name, results in evaluation_results.items():
            print(f"\n" + "="*60)
            print(f"📈 {model_name.upper()} 详细评估报告")
            print("="*60)
            self.metrics.print_evaluation_report(results)
        
        print(f"\n🎯 实验结论:")
        if "llama3" in evaluation_results and "qwen3" in evaluation_results:
            llama3_score = evaluation_results["llama3"].get("overall_score", {}).get("overall_score", 0)
            qwen3_score = evaluation_results["qwen3"].get("overall_score", {}).get("overall_score", 0)
            
            if llama3_score > qwen3_score:
                print(f"   🇺🇸 Llama3表现更优 (得分: {llama3_score:.3f} vs {qwen3_score:.3f})")
                print(f"   ✅ 英文原声模型在MovieLens英文数据集上确实有优势")
            elif qwen3_score > llama3_score:
                print(f"   🇨🇳 Qwen3表现更优 (得分: {qwen3_score:.3f} vs {llama3_score:.3f})")
                print(f"   🤔 中文模型在英文数据集上表现意外优秀，值得深入研究")
            else:
                print(f"   🤝 两个模型表现相当 (Llama3: {llama3_score:.3f}, Qwen3: {qwen3_score:.3f})")
                print(f"   📊 语言差异对推荐效果影响较小")
        
        print(f"\n💡 技术建议:")
        print(f"   1. LLM推荐系统在MovieLens数据集上具备可行性")
        print(f"   2. 双语模型对比为跨语言推荐研究提供了有价值的基准")
        print(f"   3. 建议进一步优化提示词工程以提升推荐质量")
        print(f"   4. 可以考虑将LLM与传统推荐算法进行融合")


def main():
    """主函数"""
    print("🚀 启动LLM推荐系统完整验证实验")
    
    # 创建实验
    experiment = CompleteLLMValidationExperiment()
    
    # 运行实验
    try:
        results = experiment.run_complete_experiment()
        
        if results:
            print(f"\n🎉 实验成功完成！")
            print(f"📁 结果文件保存在: {experiment.output_dir}")
        else:
            print(f"\n❌ 实验失败，请检查日志")
            
    except KeyboardInterrupt:
        print(f"\n⚠️  实验被用户中断")
    except Exception as e:
        print(f"\n❌ 实验异常终止: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
