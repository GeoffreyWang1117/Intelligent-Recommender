"""
优化的Ensemble推荐器
基于算法一致性分析的最佳组合策略实现
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
import logging
from collections import defaultdict

from .base_recommender import BaseRecommender
from .algorithm_factory import create_recommender

logger = logging.getLogger(__name__)


class OptimizedEnsembleRecommender(BaseRecommender):
    """
    基于一致性分析优化的Ensemble推荐器
    
    根据算法间Jaccard相似度和Kendall Tau相关性分析，
    选择最优的算法组合策略，最大化推荐效果。
    """
    
    def __init__(self, strategy: str = 'best_complementary', **kwargs):
        """
        初始化优化Ensemble推荐器
        
        Args:
            strategy: 组合策略
                - 'best_complementary': SVD + SimpleAutoInt (最强互补)
                - 'best_diverse': SVD + SimpleAutoInt + SimpleDeepFM (最佳多样性)
                - 'quality_focused': SVD主导的高质量组合
                - 'custom': 自定义算法组合
            **kwargs: 其他参数
        """
        super().__init__("OptimizedEnsemble", **kwargs)
        self.strategy = strategy
        self.models = {}
        self.algorithm_weights = {}
        self.consistency_info = {}
        
        # 根据一致性分析结果定义策略
        self._setup_strategy()
        
    def _setup_strategy(self):
        """根据选择的策略设置算法组合"""
        if self.strategy == 'best_complementary':
            # 基于最低Jaccard相似度的最强互补组合 - 使用完整Teacher模型
            self.algorithm_names = ['svd', 'autoint']  # 使用完整版AutoInt
            self.algorithm_weights = {
                'svd': 0.6,           # 质量最佳，权重更高
                'autoint': 0.4        # 提供多样性补充
            }
            self.consistency_info = {
                'jaccard_similarity': 0.0403,
                'kendall_tau': -0.0007,
                'complementarity': 'extremely_high',
                'teacher_models': True
            }
            
        elif self.strategy == 'best_diverse':
            # 基于多样性排序的Teacher模型组合
            self.algorithm_names = ['svd', 'autoint', 'deepfm', 'transformer4rec']
            self.algorithm_weights = {
                'svd': 0.4,           # 多样性得分最高
                'autoint': 0.25,      # Teacher模型
                'deepfm': 0.2,        # Teacher模型  
                'transformer4rec': 0.15  # 新的Teacher模型
            }
            self.consistency_info = {
                'diversity_scores': {'svd': 0.9572, 'autoint': 0.7496, 'deepfm': 0.5617, 'transformer4rec': 0.8000},
                'strategy_focus': 'maximum_diversity_with_teachers',
                'teacher_models': True
            }
            
        elif self.strategy == 'quality_focused':
            # 以质量为主的Teacher模型组合
            self.algorithm_names = ['svd', 'dcnv2', 'xdeepfm'] 
            self.algorithm_weights = {
                'svd': 0.5,           # 高权重，质量最优
                'dcnv2': 0.3,         # 强力Teacher模型
                'xdeepfm': 0.2        # 复杂特征交互
            }
            self.consistency_info = {
                'focus': 'quality_with_teachers',
                'primary_performance': {'R@10': 0.0596, 'R@50': 0.2897},
                'teacher_models': True
            }
            
        elif self.strategy == 'all_teachers':
            # 全Teacher模型组合策略
            self.algorithm_names = ['svd', 'deepfm', 'autoint', 'din', 'transformer4rec', 'dcnv2', 'xdeepfm']
            self.algorithm_weights = {
                'svd': 0.2,              # 基础强算法
                'deepfm': 0.15,          # Teacher模型
                'autoint': 0.15,         # Teacher模型
                'din': 0.1,              # Teacher模型
                'transformer4rec': 0.15, # 序列建模Teacher
                'dcnv2': 0.125,          # 特征交互Teacher
                'xdeepfm': 0.125         # 综合Teacher
            }
            self.consistency_info = {
                'strategy_focus': 'comprehensive_teacher_ensemble',
                'teacher_models': True,
                'model_count': 7
            }
            
        else:  # custom
            # 默认Teacher模型组合，可以通过参数自定义
            self.algorithm_names = ['svd', 'autoint', 'deepfm']
            self.algorithm_weights = {'svd': 0.5, 'autoint': 0.3, 'deepfm': 0.2}
            
        logger.info(f"Ensemble策略: {self.strategy}")
        logger.info(f"算法组合: {self.algorithm_names}")
        logger.info(f"权重分配: {self.algorithm_weights}")
    
    def fit(self, train_data: pd.DataFrame, **kwargs) -> None:
        """
        训练所有子算法
        
        Args:
            train_data: 训练数据
            **kwargs: 训练参数
        """
        logger.info(f"开始训练优化Ensemble推荐器 (策略: {self.strategy})...")
        
        # 为每个算法设置优化的训练配置 - 使用完整Teacher模型
        algorithm_configs = {
            'svd': {
                'n_components': 50,
                'learning_rate': 0.01,
                'regularization': 0.02,
                'epochs': 100
            },
            'deepfm': {  # 完整版DeepFM Teacher模型
                'embedding_dim': 16,
                'learning_rate': 0.001,
                'epochs': 80,
                'batch_size': 256,
                'early_stopping': True,
                'patience': 8
            },
            'autoint': {  # 完整版AutoInt Teacher模型
                'embedding_dim': 16,
                'learning_rate': 0.001,
                'epochs': 80,
                'num_heads': 8,
                'num_layers': 6,
                'early_stopping': True,
                'patience': 8
            },
            'din': {  # 完整版DIN Teacher模型
                'embedding_dim': 16,
                'learning_rate': 0.001,
                'epochs': 80,
                'hidden_dim': 128,
                'early_stopping': True,
                'patience': 8
            },
            'transformer4rec': {  # Transformer4Rec Teacher模型
                'embedding_dim': 64,
                'num_heads': 8,
                'num_layers': 6,
                'learning_rate': 0.001,
                'epochs': 60,
                'batch_size': 256,
                'early_stopping': True,
                'patience': 8
            },
            'dcnv2': {  # DCNv2 Teacher模型
                'embedding_dim': 16,
                'cross_layers': 4,
                'deep_layers': [512, 256, 128],
                'learning_rate': 0.001,
                'epochs': 60,
                'batch_size': 256,
                'early_stopping': True,
                'patience': 8
            },
            'xdeepfm': {  # xDeepFM Teacher模型
                'embedding_dim': 16,
                'cin_layer_sizes': [200, 200, 200],
                'dnn_hidden_dims': [400, 400, 400],
                'learning_rate': 0.001,
                'epochs': 60,
                'batch_size': 256,
                'early_stopping': True,
                'patience': 8
            }
        }
        
        # 训练每个算法
        successful_algorithms = []
        
        for algo_name in self.algorithm_names:
            try:
                logger.info(f"训练算法: {algo_name}")
                config = algorithm_configs.get(algo_name, {})
                
                model = create_recommender(algo_name, **config)
                if model is None:
                    logger.error(f"无法创建算法: {algo_name}")
                    continue
                
                model.fit(train_data, **config)
                
                if model.is_trained:
                    self.models[algo_name] = model
                    successful_algorithms.append(algo_name)
                    logger.info(f"算法 {algo_name} 训练成功")
                else:
                    logger.error(f"算法 {algo_name} 训练失败")
                    
            except Exception as e:
                logger.error(f"训练算法 {algo_name} 时出错: {e}")
        
        # 更新权重，只保留成功训练的算法
        if successful_algorithms:
            # 重新归一化权重
            total_weight = sum(self.algorithm_weights[algo] for algo in successful_algorithms)
            self.algorithm_weights = {
                algo: self.algorithm_weights[algo] / total_weight 
                for algo in successful_algorithms
            }
            self.algorithm_names = successful_algorithms
            self.is_trained = True
            logger.info(f"Ensemble训练完成，成功算法: {successful_algorithms}")
            logger.info(f"最终权重: {self.algorithm_weights}")
        else:
            logger.error("没有算法训练成功，Ensemble失败")
            self.is_trained = False
    
    def predict(self, user_id: int, item_id: int) -> float:
        """
        预测用户对物品的评分
        
        Args:
            user_id: 用户ID
            item_id: 物品ID
            
        Returns:
            加权预测评分
        """
        if not self.is_trained or not self.models:
            return 3.0  # 默认中等评分
        
        weighted_prediction = 0.0
        total_weight = 0.0
        
        for algo_name, model in self.models.items():
            try:
                prediction = model.predict(user_id, item_id)
                weight = self.algorithm_weights.get(algo_name, 0.0)
                weighted_prediction += prediction * weight
                total_weight += weight
            except Exception as e:
                logger.warning(f"算法 {algo_name} 预测失败: {e}")
        
        if total_weight > 0:
            return weighted_prediction / total_weight
        else:
            return 3.0
    
    def get_user_recommendations(self, user_id: int, top_k: int = 10, 
                               fusion_method: str = 'weighted_rank') -> List[Dict[str, Any]]:
        """
        获取用户推荐，使用多种融合策略
        
        Args:
            user_id: 用户ID
            top_k: 推荐数量
            fusion_method: 融合方法
                - 'weighted_rank': 加权排序融合
                - 'score_fusion': 评分加权融合
                - 'round_robin': 轮询选择
                
        Returns:
            推荐列表
        """
        if not self.is_trained or not self.models:
            return []
        
        # 获取各算法的推荐结果
        algorithm_recommendations = {}
        
        for algo_name, model in self.models.items():
            try:
                # 获取更多推荐以便融合
                recs = model.get_user_recommendations(user_id, top_k=top_k * 2)
                algorithm_recommendations[algo_name] = recs
            except Exception as e:
                logger.warning(f"算法 {algo_name} 推荐生成失败: {e}")
                algorithm_recommendations[algo_name] = []
        
        if not algorithm_recommendations:
            return []
        
        # 根据融合方法合并推荐
        if fusion_method == 'weighted_rank':
            return self._weighted_rank_fusion(algorithm_recommendations, top_k)
        elif fusion_method == 'score_fusion':
            return self._score_fusion(algorithm_recommendations, user_id, top_k)
        elif fusion_method == 'round_robin':
            return self._round_robin_fusion(algorithm_recommendations, top_k)
        else:
            return self._weighted_rank_fusion(algorithm_recommendations, top_k)
    
    def _weighted_rank_fusion(self, algorithm_recommendations: Dict[str, List], 
                             top_k: int) -> List[Dict[str, Any]]:
        """加权排序融合"""
        item_scores = defaultdict(float)
        item_counts = defaultdict(int)
        
        for algo_name, recs in algorithm_recommendations.items():
            weight = self.algorithm_weights.get(algo_name, 0.0)
            
            for rank, rec in enumerate(recs):
                if isinstance(rec, dict) and 'item_id' in rec:
                    item_id = rec['item_id']
                    # 排序分数：权重 / (排名 + 1)
                    rank_score = weight / (rank + 1)
                    item_scores[item_id] += rank_score
                    item_counts[item_id] += 1
        
        # 排序并返回top-k
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for item_id, score in sorted_items[:top_k]:
            recommendations.append({
                'item_id': item_id,
                'score': score,
                'fusion_method': 'weighted_rank',
                'algorithm_count': item_counts[item_id],
                'ensemble_strategy': self.strategy
            })
        
        return recommendations
    
    def _score_fusion(self, algorithm_recommendations: Dict[str, List],
                     user_id: int, top_k: int) -> List[Dict[str, Any]]:
        """评分加权融合"""
        item_scores = defaultdict(float)
        item_details = defaultdict(dict)
        
        for algo_name, recs in algorithm_recommendations.items():
            weight = self.algorithm_weights.get(algo_name, 0.0)
            model = self.models[algo_name]
            
            for rec in recs:
                if isinstance(rec, dict) and 'item_id' in rec:
                    item_id = rec['item_id']
                    
                    # 获取预测评分
                    try:
                        pred_score = model.predict(user_id, item_id)
                        weighted_score = pred_score * weight
                        item_scores[item_id] += weighted_score
                        
                        if item_id not in item_details:
                            item_details[item_id] = {'algorithms': [], 'scores': []}
                        item_details[item_id]['algorithms'].append(algo_name)
                        item_details[item_id]['scores'].append(pred_score)
                    except:
                        pass
        
        # 排序并返回top-k
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for item_id, score in sorted_items[:top_k]:
            rec = {
                'item_id': item_id,
                'score': score,
                'fusion_method': 'score_fusion',
                'contributing_algorithms': item_details[item_id]['algorithms'],
                'individual_scores': item_details[item_id]['scores'],
                'ensemble_strategy': self.strategy
            }
            recommendations.append(rec)
        
        return recommendations
    
    def _round_robin_fusion(self, algorithm_recommendations: Dict[str, List],
                           top_k: int) -> List[Dict[str, Any]]:
        """轮询融合"""
        recommendations = []
        seen_items = set()
        max_length = max(len(recs) for recs in algorithm_recommendations.values())
        
        # 按权重排序算法
        sorted_algorithms = sorted(self.algorithm_weights.items(), 
                                 key=lambda x: x[1], reverse=True)
        
        for round_idx in range(max_length):
            for algo_name, weight in sorted_algorithms:
                if len(recommendations) >= top_k:
                    break
                
                recs = algorithm_recommendations.get(algo_name, [])
                if round_idx < len(recs):
                    rec = recs[round_idx]
                    if isinstance(rec, dict) and 'item_id' in rec:
                        item_id = rec['item_id']
                        if item_id not in seen_items:
                            recommendations.append({
                                'item_id': item_id,
                                'score': rec.get('score', 0.0),
                                'fusion_method': 'round_robin',
                                'selected_from': algo_name,
                                'round': round_idx,
                                'ensemble_strategy': self.strategy
                            })
                            seen_items.add(item_id)
            
            if len(recommendations) >= top_k:
                break
        
        return recommendations[:top_k]
    
    def recommend(self, user_id: int, k: int = 10) -> List[Tuple[int, float]]:
        """
        为指定用户生成推荐 (兼容接口)
        
        Args:
            user_id: 用户ID
            k: 推荐数量
            
        Returns:
            推荐列表 [(item_id, score), ...]
        """
        recs = self.get_user_recommendations(user_id, top_k=k)
        return [(rec['item_id'], rec['score']) for rec in recs]
    
    def explain_recommendation(self, user_id: int, item_id: int) -> str:
        """
        解释推荐结果 (兼容接口)
        
        Args:
            user_id: 用户ID  
            item_id: 物品ID
            
        Returns:
            推荐解释文本
        """
        explanation = self.get_ensemble_explanation(user_id, item_id)
        strategy_desc = {
            'best_complementary': '最强互补策略',
            'best_diverse': '最佳多样性策略', 
            'quality_focused': '质量优先策略'
        }
        
        desc = strategy_desc.get(self.strategy, self.strategy)
        return f"基于{desc}的Ensemble推荐，算法权重: {explanation['algorithm_weights']}"
    
    def get_ensemble_explanation(self, user_id: int, item_id: int) -> Dict[str, Any]:
        """
        获取ensemble推荐的解释信息
        
        Args:
            user_id: 用户ID
            item_id: 物品ID
            
        Returns:
            解释信息字典
        """
        explanation = {
            'ensemble_strategy': self.strategy,
            'algorithm_predictions': {},
            'algorithm_weights': self.algorithm_weights.copy(),
            'consistency_info': self.consistency_info.copy(),
            'final_prediction': self.predict(user_id, item_id)
        }
        
        for algo_name, model in self.models.items():
            try:
                pred = model.predict(user_id, item_id)
                explanation['algorithm_predictions'][algo_name] = pred
            except (AttributeError, ValueError, TypeError) as e:
                explanation['algorithm_predictions'][algo_name] = f"Error: {e}"
        
        return explanation
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """获取当前策略的详细信息"""
        return {
            'strategy': self.strategy,
            'algorithms': self.algorithm_names,
            'weights': self.algorithm_weights,
            'consistency_analysis': self.consistency_info,
            'model_status': {algo: model.is_trained for algo, model in self.models.items()},
            'is_trained': self.is_trained
        }


# 工厂函数
def create_optimized_ensemble(strategy_name: str = 'best_complementary', **kwargs) -> OptimizedEnsembleRecommender:
    """
    创建优化的Ensemble推荐器
    
    Args:
        strategy_name: 组合策略
            - 'best_complementary': 最强互补的Teacher模型组合  
            - 'best_diverse': 最多样化的Teacher模型组合
            - 'quality_focused': 质量优先的Teacher模型组合
            - 'all_teachers': 全Teacher模型组合
        **kwargs: 其他参数
        
    Returns:
        OptimizedEnsembleRecommender实例
    """
    return OptimizedEnsembleRecommender(strategy=strategy_name, **kwargs)


# 示例使用
if __name__ == "__main__":
    # 创建不同策略的ensemble
    strategies = ['best_complementary', 'best_diverse', 'quality_focused']
    
    for strategy in strategies:
        print(f"\n=== {strategy.upper()} 策略 ===")
        ensemble = create_optimized_ensemble(strategy)
        info = ensemble.get_strategy_info()
        print(f"算法组合: {info['algorithms']}")
        print(f"权重分配: {info['weights']}")
        print(f"一致性信息: {info['consistency_analysis']}")
