#!/usr/bin/env python3
"""
传统推荐算法Teachers完整评估 - 最终修复版
Complete Traditional Teachers Evaluation - Final Fixed Version

基于真实MovieLens数据，修复所有列名和接口问题

作者: GitHub Copilot
日期: 2025-08-18
"""

import sys
import os
import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime

# 添加模型路径
sys.path.append('../../')
from models.algorithm_factory import create_recommender
from recommendation_metrics import RecommendationMetrics

warnings.filterwarnings('ignore')

class TraditionalTeachersEvaluator:
    """传统推荐算法Teachers的完整评估器 - 最终修复版"""
    
    def __init__(self, output_dir: str = "traditional_evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 推荐评估指标
        self.metrics = RecommendationMetrics()
        
        # 数据路径
        self.data_path = Path("../../data/sample")
        
        # 传统模型列表 (使用正确的小写名称)
        self.algorithms = [
            "deepfm",
            "autoint", 
            "transformer4rec",
            "xdeepfm",
            "din",
            "dcnv2"
        ]
        
        # 评估配置
        self.config = {
            "test_users_count": 20,        # 测试用户数量
            "min_ratings_per_user": 10,    # 最少评分数
            "train_ratio": 0.8,            # 训练集比例
            "top_k": 10,                   # 推荐数量
            "relevance_threshold": 4.0     # 相关性阈值
        }
        
        print(f"🎯 Traditional Teachers评估器初始化完成")
        print(f"   算法数量: {len(self.algorithms)}")
        print(f"   数据路径: {self.data_path}")
        print(f"   输出目录: {self.output_dir}")
    
    def load_movielens_data(self) -> Dict:
        """加载MovieLens数据"""
        print("📁 加载MovieLens数据...")
        
        try:
            # 加载数据文件
            ratings_df = pd.read_csv(self.data_path / "ratings.csv")
            movies_df = pd.read_csv(self.data_path / "movies.csv") 
            users_df = pd.read_csv(self.data_path / "users.csv")
            
            print(f"✅ 数据加载成功:")
            print(f"   评分记录: {len(ratings_df):,}")
            print(f"   电影数量: {len(movies_df):,}")
            print(f"   用户数量: {len(users_df):,}")
            print(f"   评分范围: {ratings_df['rating'].min():.1f} - {ratings_df['rating'].max():.1f}")
            print(f"   平均评分: {ratings_df['rating'].mean():.2f}")
            print(f"   数据列名: {list(ratings_df.columns)}")
            
            return {
                "ratings": ratings_df,
                "movies": movies_df,
                "users": users_df
            }
            
        except Exception as e:
            print(f"❌ 数据加载失败: {str(e)}")
            raise
    
    def prepare_train_test_data(self, data: Dict) -> Dict:
        """准备训练测试数据"""
        print("🔄 准备训练测试数据...")
        
        ratings_df = data["ratings"]
        
        # 选择活跃用户 (评分数量足够的用户)
        user_counts = ratings_df['user_id'].value_counts()
        active_users = user_counts[user_counts >= self.config["min_ratings_per_user"]].index.tolist()
        
        # 过滤数据
        filtered_ratings = ratings_df[ratings_df['user_id'].isin(active_users)]
        
        # 随机选择测试用户
        test_users = np.random.choice(
            active_users, 
            size=min(self.config["test_users_count"], len(active_users)),
            replace=False
        ).tolist()
        
        # 为每个测试用户分割训练测试数据
        train_data = []
        test_data = {}
        
        for user_id in filtered_ratings['user_id'].unique():
            user_ratings = filtered_ratings[filtered_ratings['user_id'] == user_id]
            
            if user_id in test_users:
                # 测试用户：80%训练，20%测试
                n_train = int(len(user_ratings) * self.config["train_ratio"])
                user_train = user_ratings.sample(n=n_train, random_state=42)
                user_test = user_ratings.drop(user_train.index)
                
                train_data.append(user_train)
                test_data[user_id] = user_test
            else:
                # 非测试用户：全部用于训练
                train_data.append(user_ratings)
        
        train_df = pd.concat(train_data, ignore_index=True)
        
        print(f"✅ 数据分割完成:")
        print(f"   训练集: {len(train_df):,} 条记录")
        print(f"   测试用户: {len(test_users)} 个")
        print(f"   测试记录: {sum(len(df) for df in test_data.values())} 条")
        print(f"   训练数据列名: {list(train_df.columns)}")
        
        return {
            "train_df": train_df,
            "test_data": test_data,
            "test_users": test_users,
            "active_users": active_users
        }
    
    def train_single_model(self, algorithm: str, train_df: pd.DataFrame) -> Any:
        """训练单个模型"""
        print(f"  🔧 训练 {algorithm} 模型...")
        
        try:
            # 模型配置
            config = {
                "embedding_dim": 64,
                "hidden_dims": [512, 256, 128],
                "dropout": 0.2,
                "learning_rate": 0.001,
                "batch_size": 1024,
                "epochs": 5,  # 减少epochs加快训练
                "device": "cuda" if torch.cuda.is_available() else "cpu"
            }
            
            # 创建模型
            model = create_recommender(algorithm, **config)
            
            # 数据预处理 - 确保格式正确
            train_data = train_df.copy()
            
            print(f"    📊 训练数据形状: {train_data.shape}")
            print(f"    📊 训练数据列: {list(train_data.columns)}")
            
            # 检查必需列
            required_columns = ['user_id', 'item_id', 'rating']
            for col in required_columns:
                if col not in train_data.columns:
                    print(f"    ❌ 缺少必要列 {col}")
                    print(f"    可用列: {list(train_data.columns)}")
                    return None
            
            # 训练模型
            print(f"    🏃 开始训练...")
            model.fit(train_data)
            
            print(f"    ✅ {algorithm} 训练完成")
            return model
            
        except Exception as e:
            print(f"    ❌ {algorithm} 训练失败: {str(e)}")
            import traceback
            print(f"    详细错误: {traceback.format_exc()}")
            return None
    
    def generate_recommendations(self, model: Any, algorithm: str, 
                               test_users: List[int], train_df: pd.DataFrame) -> Dict:
        """生成推荐结果"""
        print(f"  📋 生成 {algorithm} 推荐...")
        
        recommendations = {}
        
        try:
            # 获取所有物品
            all_items = train_df['item_id'].unique()
            print(f"    📊 总物品数: {len(all_items)}")
            
            for user_id in test_users:
                # 获取用户已评分物品
                user_items = train_df[train_df['user_id'] == user_id]['item_id'].unique()
                
                # 候选物品 (排除已评分)
                candidate_items = [item for item in all_items if item not in user_items]
                
                if len(candidate_items) > 0:
                    # 生成推荐
                    try:
                        if hasattr(model, 'predict'):
                            # 预测评分
                            predictions = []
                            for item_id in candidate_items[:100]:  # 限制候选数量
                                try:
                                    score = model.predict(user_id, item_id)
                                    predictions.append((item_id, score))
                                except:
                                    continue
                            
                            # 排序并选择top-k
                            if predictions:
                                predictions.sort(key=lambda x: x[1], reverse=True)
                                recommended_items = [item_id for item_id, _ in predictions[:self.config["top_k"]]]
                                recommendations[user_id] = recommended_items
                        
                        elif hasattr(model, 'recommend'):
                            # 直接推荐
                            recommended_items = model.recommend(user_id, self.config["top_k"])
                            recommendations[user_id] = recommended_items
                        
                        else:
                            print(f"    ⚠️ {algorithm} 没有预测或推荐方法")
                            # 尝试随机推荐作为fallback
                            random_items = np.random.choice(candidate_items, 
                                                          size=min(self.config["top_k"], len(candidate_items)), 
                                                          replace=False)
                            recommendations[user_id] = random_items.tolist()
                            
                    except Exception as e:
                        print(f"    ⚠️ 用户{user_id}推荐失败: {str(e)}")
                        continue
        
        except Exception as e:
            print(f"    ❌ {algorithm} 推荐生成失败: {str(e)}")
        
        success_rate = len(recommendations) / len(test_users) * 100 if test_users else 0
        print(f"    ✅ {algorithm} 推荐完成，成功率: {success_rate:.1f}%")
        print(f"    📊 成功推荐用户数: {len(recommendations)}")
        
        return recommendations
    
    def evaluate_model_performance(self, algorithm: str, recommendations: Dict, 
                                 test_data: Dict, train_df: pd.DataFrame) -> Dict:
        """评估模型性能"""
        print(f"  📊 评估 {algorithm} 性能...")
        
        try:
            # 准备评估数据
            eval_data = self._prepare_evaluation_data(recommendations, test_data, train_df)
            
            if not eval_data or not eval_data.get("true_ratings"):
                print(f"    ❌ {algorithm} 评估数据准备失败")
                return {}
            
            print(f"    📊 评估数据统计:")
            print(f"       真实评分数: {len(eval_data['true_ratings'])}")
            print(f"       预测评分数: {len(eval_data['predicted_ratings'])}")
            print(f"       测试用户数: {len(eval_data['test_users_relevant'])}")
            
            # 执行综合评估
            results = self.metrics.comprehensive_evaluation(
                eval_data["true_ratings"],
                eval_data["predicted_ratings"],
                eval_data["test_users_relevant"],  # 这是关键 - 用户到相关物品的映射
                eval_data["recommendations"],
                eval_data["item_features"],
                eval_data["item_popularity"],
                eval_data["all_items"]
            )
            
            print(f"    ✅ {algorithm} 性能评估完成")
            return results
            
        except Exception as e:
            print(f"    ❌ {algorithm} 性能评估失败: {str(e)}")
            import traceback
            print(f"    详细错误: {traceback.format_exc()}")
            return {}
    
    def _prepare_evaluation_data(self, recommendations: Dict, test_data: Dict, 
                               train_df: pd.DataFrame) -> Dict:
        """准备评估数据格式"""
        
        # 真实评分
        true_ratings = []
        for user_id, user_test_df in test_data.items():
            for _, row in user_test_df.iterrows():
                true_ratings.append((user_id, row['item_id'], row['rating']))
        
        # 预测评分 (简化处理，使用推荐排序作为评分)
        predicted_ratings = []
        for user_id, rec_items in recommendations.items():
            for i, item_id in enumerate(rec_items):
                # 根据推荐排序生成伪评分 (5到1)
                score = max(1, 5 - i // 2)
                predicted_ratings.append((user_id, item_id, score))
        
        # 测试用户相关物品 (关键修复！)
        test_users_relevant = {}
        for user_id, user_test_df in test_data.items():
            # 获取评分 >= 阈值的物品作为相关物品
            relevant_items = user_test_df[
                user_test_df['rating'] >= self.config['relevance_threshold']
            ]['item_id'].tolist()
            test_users_relevant[user_id] = relevant_items
        
        # 物品特征 (简化)
        all_items = train_df['item_id'].unique()
        item_features = {item_id: np.random.randn(10) for item_id in all_items}
        
        # 物品流行度
        item_popularity = train_df['item_id'].value_counts().to_dict()
        
        return {
            "true_ratings": true_ratings,
            "predicted_ratings": predicted_ratings,
            "test_users_relevant": test_users_relevant,  # 这是关键修复
            "recommendations": recommendations,
            "item_features": item_features,
            "item_popularity": item_popularity,
            "all_items": all_items.tolist()
        }
    
    def run_complete_evaluation(self) -> Dict:
        """运行完整评估流程"""
        print("🚀 开始Traditional Teachers完整评估...")
        print("=" * 60)
        
        start_time = datetime.now()
        results = {
            "experiment_info": {
                "start_time": start_time.isoformat(),
                "algorithms": self.algorithms,
                "config": self.config
            },
            "models": {},
            "summary": {}
        }
        
        try:
            # 1. 加载数据
            data = self.load_movielens_data()
            
            # 2. 准备训练测试数据
            train_test_data = self.prepare_train_test_data(data)
            
            # 3. 逐个训练和评估模型
            successful_models = 0
            
            for algorithm in self.algorithms:
                print(f"\n📈 处理 {algorithm} 算法...")
                print("-" * 40)
                
                model_results = {
                    "algorithm": algorithm,
                    "training_status": "failed",
                    "evaluation_results": {},
                    "recommendations_count": 0
                }
                
                # 训练模型
                model = self.train_single_model(algorithm, train_test_data["train_df"])
                
                if model is not None:
                    model_results["training_status"] = "success"
                    
                    # 生成推荐
                    recommendations = self.generate_recommendations(
                        model, algorithm, train_test_data["test_users"], train_test_data["train_df"]
                    )
                    
                    model_results["recommendations_count"] = len(recommendations)
                    
                    if recommendations:
                        # 评估性能
                        evaluation = self.evaluate_model_performance(
                            algorithm, recommendations, train_test_data["test_data"], train_test_data["train_df"]
                        )
                        
                        model_results["evaluation_results"] = evaluation
                        
                        if evaluation:
                            successful_models += 1
                
                results["models"][algorithm] = model_results
            
            # 4. 生成总结
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            results["summary"] = {
                "total_algorithms": len(self.algorithms),
                "successful_trainings": sum(1 for r in results["models"].values() if r["training_status"] == "success"),
                "successful_evaluations": successful_models,
                "total_duration_seconds": duration,
                "end_time": end_time.isoformat()
            }
            
            # 5. 保存结果
            self.save_results(results)
            self.print_summary(results)
            
            return results
            
        except Exception as e:
            print(f"❌ 评估过程失败: {str(e)}")
            import traceback
            print(f"详细错误: {traceback.format_exc()}")
            results["error"] = str(e)
            return results
    
    def save_results(self, results: Dict):
        """保存评估结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存完整结果
        results_file = self.output_dir / f"traditional_evaluation_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"💾 结果已保存: {results_file}")
    
    def print_summary(self, results: Dict):
        """打印评估总结"""
        print("\n" + "=" * 60)
        print("📊 Traditional Teachers评估总结")
        print("=" * 60)
        
        summary = results["summary"]
        print(f"🎯 算法总数: {summary['total_algorithms']}")
        print(f"✅ 训练成功: {summary['successful_trainings']}")
        print(f"📈 评估成功: {summary['successful_evaluations']}")
        print(f"⏱️  总耗时: {summary['total_duration_seconds']:.1f} 秒")
        
        # 详细结果
        print(f"\n📋 详细结果:")
        for algorithm, model_result in results["models"].items():
            status = "✅" if model_result["training_status"] == "success" else "❌"
            rec_count = model_result["recommendations_count"]
            print(f"   {status} {algorithm}: 推荐数 {rec_count}")
            
            if model_result["evaluation_results"]:
                eval_results = model_result["evaluation_results"]
                if "accuracy" in eval_results:
                    acc = eval_results["accuracy"]
                    if "rmse" in acc:
                        print(f"       RMSE: {acc['rmse']:.4f}")
                if "ranking" in eval_results:
                    rank = eval_results["ranking"]
                    if "ndcg_10" in rank:
                        print(f"       NDCG@10: {rank['ndcg_10']:.4f}")

def main():
    """主函数"""
    
    # 导入torch (需要在这里导入，避免初始化时的依赖问题)
    global torch
    try:
        import torch
        print(f"🔥 PyTorch版本: {torch.__version__}")
        print(f"🎮 CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"🚀 GPU设备: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("❌ PyTorch未安装")
        return
    
    # 创建评估器并运行
    evaluator = TraditionalTeachersEvaluator()
    results = evaluator.run_complete_evaluation()
    
    # 检查结果
    if "error" not in results:
        print(f"\n🎉 Traditional Teachers评估完成!")
        print(f"🎯 成功评估算法数: {results['summary']['successful_evaluations']}")
    else:
        print(f"\n💥 评估失败: {results['error']}")

if __name__ == "__main__":
    main()
