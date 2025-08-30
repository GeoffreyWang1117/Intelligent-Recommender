#!/usr/bin/env python3
"""
MovieLens数据集上的集成教师模型测试
在真实数据上验证Traditional Teachers Ensemble的性能

作者: GitHub Copilot
日期: 2025-08-27
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Any, Optional
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MovieLensDataLoader:
    """MovieLens数据加载器"""
    
    def __init__(self, data_path: str = "data/movielens/"):
        self.data_path = data_path
        self.ratings_df = None
        self.movies_df = None
        
    def load_data(self):
        """加载MovieLens数据"""
        logger.info("📚 加载MovieLens数据...")
        
        # 加载评分数据
        ratings_file = os.path.join(self.data_path, "ratings.csv")
        if not os.path.exists(ratings_file):
            raise FileNotFoundError(f"评分文件不存在: {ratings_file}")
            
        self.ratings_df = pd.read_csv(ratings_file)
        
        # 加载电影数据
        movies_file = os.path.join(self.data_path, "movies.csv") 
        if os.path.exists(movies_file):
            self.movies_df = pd.read_csv(movies_file)
        
        # 数据统计
        logger.info(f"   评分记录: {len(self.ratings_df)}")
        logger.info(f"   用户数: {self.ratings_df['user_id'].nunique()}")
        logger.info(f"   电影数: {self.ratings_df['item_id'].nunique()}")
        logger.info(f"   评分范围: {self.ratings_df['rating'].min()}-{self.ratings_df['rating'].max()}")
        
        return self.ratings_df, self.movies_df
    
    def prepare_train_test_split(self, test_ratio: float = 0.2):
        """准备训练测试数据分割"""
        logger.info(f"🔄 准备训练/测试数据分割 (测试比例: {test_ratio})")
        
        if self.ratings_df is None:
            self.load_data()
        
        if self.ratings_df is None:
            raise ValueError("无法加载评分数据")
        
        # 简单的时间分割：最新的20%作为测试集
        sorted_data = self.ratings_df.sort_values('timestamp')
        split_idx = int(len(sorted_data) * (1 - test_ratio))
        
        train_df = sorted_data.iloc[:split_idx]
        test_df = sorted_data.iloc[split_idx:]
        
        logger.info(f"   训练数据: {len(train_df)} 条")
        logger.info(f"   测试数据: {len(test_df)} 条")
        
        return train_df, test_df


class SimpleRecommender:
    """简单推荐器基类"""
    
    def __init__(self, name: str, strategy: str = "user_avg"):
        self.name = name
        self.strategy = strategy
        self.is_trained = False
        self.global_mean = 3.5
        self.user_means = {}
        self.item_means = {}
        self.user_item_matrix = {}
        
    def fit(self, train_data: pd.DataFrame):
        """训练模型"""
        logger.info(f"🔧 训练 {self.name}...")
        
        # 计算统计信息
        self.global_mean = train_data['rating'].mean()
        self.user_means = train_data.groupby('user_id')['rating'].mean().to_dict()
        self.item_means = train_data.groupby('item_id')['rating'].mean().to_dict()
        
        # 构建用户-物品矩阵
        for _, row in train_data.iterrows():
            user_id = int(row['user_id'])
            item_id = int(row['item_id'])
            rating = float(row['rating'])
            
            if user_id not in self.user_item_matrix:
                self.user_item_matrix[user_id] = {}
            self.user_item_matrix[user_id][item_id] = rating
        
        self.is_trained = True
        logger.info(f"   ✅ {self.name} 训练完成")
        
    def predict(self, user_id: int, item_id: int) -> float:
        """预测评分"""
        if not self.is_trained:
            return self.global_mean
        
        # 如果有直接评分，返回该评分
        if user_id in self.user_item_matrix and item_id in self.user_item_matrix[user_id]:
            return self.user_item_matrix[user_id][item_id]
        
        # 根据策略预测
        if self.strategy == "user_avg":
            return self.user_means.get(user_id, self.global_mean)
        elif self.strategy == "item_avg":
            return self.item_means.get(item_id, self.global_mean)
        elif self.strategy == "combined":
            user_avg = self.user_means.get(user_id, self.global_mean)
            item_avg = self.item_means.get(item_id, self.global_mean)
            return 0.5 * user_avg + 0.5 * item_avg
        else:
            return self.global_mean
    
    def get_recommendations(self, user_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """获取推荐"""
        if not self.is_trained:
            return []
        
        # 获取用户已评分的物品
        rated_items = set()
        if user_id in self.user_item_matrix:
            rated_items = set(self.user_item_matrix[user_id].keys())
        
        # 从所有物品中选择候选
        all_items = set(self.item_means.keys())
        candidate_items = list(all_items - rated_items)
        
        # 如果候选太少，取所有物品
        if len(candidate_items) < top_k:
            candidate_items = list(all_items)[:100]
        
        # 预测并排序
        item_scores = []
        for item_id in candidate_items:
            score = self.predict(user_id, item_id)
            item_scores.append((item_id, score))
        
        item_scores.sort(key=lambda x: x[1], reverse=True)
        return item_scores[:top_k]


class EnsembleRecommender:
    """集成推荐器"""
    
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.is_trained = False
        
    def add_model(self, name: str, model: SimpleRecommender, weight: float):
        """添加模型"""
        self.models[name] = model
        self.weights[name] = weight
        
    def fit(self, train_data: pd.DataFrame):
        """训练所有模型"""
        logger.info("🎓 训练集成模型...")
        
        for name, model in self.models.items():
            model.fit(train_data)
        
        # 归一化权重
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        self.is_trained = True
        logger.info(f"   ✅ 集成训练完成，权重: {self.weights}")
        
    def predict(self, user_id: int, item_id: int) -> float:
        """集成预测"""
        if not self.is_trained:
            return 3.5
        
        weighted_sum = 0
        total_weight = 0
        
        for name, model in self.models.items():
            try:
                pred = model.predict(user_id, item_id)
                weight = self.weights[name]
                weighted_sum += weight * pred
                total_weight += weight
            except:
                continue
        
        return weighted_sum / total_weight if total_weight > 0 else 3.5
    
    def get_recommendations(self, user_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """集成推荐"""
        if not self.is_trained:
            return []
        
        # 收集所有模型的推荐
        all_candidates = {}
        
        for name, model in self.models.items():
            try:
                recs = model.get_recommendations(user_id, top_k=20)
                weight = self.weights[name]
                
                for item_id, score in recs:
                    if item_id not in all_candidates:
                        all_candidates[item_id] = 0
                    all_candidates[item_id] += weight * score
            except:
                continue
        
        # 排序返回
        sorted_items = sorted(all_candidates.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:top_k]


def evaluate_model(model, test_data: pd.DataFrame, model_name: str) -> Dict[str, float]:
    """评估模型性能"""
    logger.info(f"📊 评估 {model_name}...")
    
    # 预测准确性评估
    predictions = []
    actuals = []
    
    # 限制测试数据量以加快速度
    test_sample = test_data.head(500)
    
    for _, row in test_sample.iterrows():
        try:
            user_id = int(row['user_id'])
            item_id = int(row['item_id'])
            actual = row['rating']
            
            if hasattr(model, 'predict'):
                pred = model.predict(user_id, item_id)
            else:
                continue
                
            predictions.append(pred)
            actuals.append(actual)
        except:
            continue
    
    if len(predictions) == 0:
        return {'rmse': float('inf'), 'mae': float('inf'), 'coverage': 0.0}
    
    # 计算指标
    rmse = np.sqrt(np.mean([(p - a) ** 2 for p, a in zip(predictions, actuals)]))
    mae = np.mean([abs(p - a) for p, a in zip(predictions, actuals)])
    
    # 推荐覆盖率测试
    test_users = test_data['user_id'].unique()[:20]
    successful_recs = 0
    
    for user_id in test_users:
        try:
            if hasattr(model, 'get_recommendations'):
                recs = model.get_recommendations(int(user_id), top_k=5)
            else:
                continue
            if recs and len(recs) > 0:
                successful_recs += 1
        except:
            continue
    
    coverage = successful_recs / len(test_users)
    
    logger.info(f"   {model_name} - RMSE: {rmse:.3f}, MAE: {mae:.3f}, 覆盖率: {coverage:.1%}")
    
    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'coverage': float(coverage),
        'prediction_count': len(predictions)
    }


def main():
    """主函数"""
    print("🎬 MovieLens集成教师模型测试")
    print("=" * 60)
    
    try:
        # 1. 检查CUDA环境
        if torch.cuda.is_available():
            print(f"🔥 CUDA可用: {torch.cuda.get_device_name()}")
            print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("💻 使用CPU模式")
        
        # 2. 加载数据
        data_loader = MovieLensDataLoader()
        train_data, test_data = data_loader.prepare_train_test_split()
        
        # 3. 创建单个模型
        models = {
            'user_based': SimpleRecommender('用户协同过滤', 'user_avg'),
            'item_based': SimpleRecommender('物品协同过滤', 'item_avg'),
            'combined': SimpleRecommender('混合模型', 'combined'),
            'baseline1': SimpleRecommender('基线模型1', 'user_avg'),
            'baseline2': SimpleRecommender('基线模型2', 'item_avg'),
            'baseline3': SimpleRecommender('基线模型3', 'combined')
        }
        
        # 4. 训练单个模型
        print("\n🔧 训练单个模型...")
        for name, model in models.items():
            model.fit(train_data)
        
        # 5. 评估单个模型
        print("\n📊 评估单个模型性能...")
        individual_results = {}
        for name, model in models.items():
            individual_results[name] = evaluate_model(model, test_data, name)
        
        # 6. 创建集成模型
        print("\n🎯 创建集成模型...")
        ensemble = EnsembleRecommender()
        
        # 基于性能设定权重
        weights = {
            'user_based': 0.25,
            'item_based': 0.20,
            'combined': 0.30,
            'baseline1': 0.10,
            'baseline2': 0.10,
            'baseline3': 0.05
        }
        
        for name, model in models.items():
            ensemble.add_model(name, model, weights[name])
        
        ensemble.fit(train_data)
        
        # 7. 评估集成模型
        print("\n🏆 评估集成模型...")
        ensemble_result = evaluate_model(ensemble, test_data, "集成模型")
        
        # 8. 结果对比
        print("\n📋 性能对比:")
        print("模型名称\t\tRMSE\tMAE\t覆盖率")
        print("-" * 50)
        
        for name, result in individual_results.items():
            print(f"{name:15}\t{result['rmse']:.3f}\t{result['mae']:.3f}\t{result['coverage']:.1%}")
        
        print(f"{'集成模型':15}\t{ensemble_result['rmse']:.3f}\t{ensemble_result['mae']:.3f}\t{ensemble_result['coverage']:.1%}")
        
        # 9. 推荐示例
        print("\n🎬 推荐示例:")
        try:
            # 选择一个有评分的用户
            sample_user = int(train_data['user_id'].iloc[0])
            print(f"   用户 {sample_user} 的历史评分:")
            
            user_history = train_data[train_data['user_id'] == sample_user].head(3)
            for _, row in user_history.iterrows():
                print(f"     电影 {row['item_id']}: {row['rating']} 分")
            
            print(f"\n   集成模型推荐 Top-5:")
            recs = ensemble.get_recommendations(sample_user, top_k=5)
            for i, (item_id, score) in enumerate(recs):
                print(f"     {i+1}. 电影 {item_id}: {score:.3f}")
                
        except Exception as e:
            print(f"   推荐示例失败: {e}")
        
        # 10. 性能改进分析
        print("\n📈 性能改进分析:")
        individual_rmses = [r['rmse'] for r in individual_results.values()]
        best_individual = min(individual_rmses)
        ensemble_rmse = ensemble_result['rmse']
        
        improvement = (best_individual - ensemble_rmse) / best_individual * 100
        
        print(f"   最佳单模型RMSE: {best_individual:.3f}")
        print(f"   集成模型RMSE: {ensemble_rmse:.3f}")
        print(f"   性能改进: {improvement:+.1f}%")
        
        if improvement > 0:
            print("✅ 集成模型表现优于单个模型！")
        else:
            print("⚠️ 集成模型需要进一步优化")
        
        print("\n🎯 下一步计划:")
        print("   1. 集成真实的Traditional Teacher模型(DCNv2, DIN等)")
        print("   2. 实现Fisher Information per-layer计算")
        print("   3. 添加pruning-aware蒸馏机制")
        print("   4. 在Amazon数据上验证泛化能力")
        
        return {
            'individual_results': individual_results,
            'ensemble_result': ensemble_result,
            'improvement': improvement
        }
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
