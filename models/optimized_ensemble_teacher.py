#!/usr/bin/env python3
"""
基于最新评估结果的优化Ensemble推荐器

结合SVD(最佳推荐质量) + xDeepFM(最佳预测精度) + AutoInt(综合均衡)
构建高性能Teacher模型，为后续Fisher信息分析和知识蒸馏做准备。

核心优势:
- 基于真实性能数据的算法选择和权重分配
- 多任务融合：排序 + 评分预测 + 特征学习
- 为Fisher Analysis提供强Teacher模型
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any, Optional, Tuple
import pickle
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class OptimizedEnsembleTeacher:
    """基于评估结果优化的集成教师模型"""
    
    def __init__(self, **kwargs):
        """
        初始化优化集成推荐器
        
        基于最新评估结果的最佳算法组合:
        - SVD: 推荐质量最佳 (Recall@10: 0.030, NDCG@10: 0.128)
        - xDeepFM: 评分预测最佳 (RMSE: 0.491, MAE: 0.256)  
        - AutoInt: 综合性能均衡 (Recall@10: 0.007, RMSE: 0.517)
        """
        # 基于评估结果的算法配置
        self.algorithm_config = {
            'svd': {
                'weight': 0.4,           # 推荐质量最佳，权重最高
                'primary_task': 'ranking',  # 主要负责排序推荐
                'model_path': 'models/saved/SVD_real_movielens.pkl',
                'performance': {
                    'recall_10': 0.030,
                    'precision_10': 0.126, 
                    'ndcg_10': 0.128,
                    'speed': 0.2  # 秒
                }
            },
            'xdeepfm': {
                'weight': 0.4,           # 预测精度最佳，权重最高
                'primary_task': 'rating',   # 主要负责评分预测
                'model_path': 'models/saved/xDeepFM_real_movielens.pkl',
                'performance': {
                    'rmse': 0.491,
                    'mae': 0.256,
                    'speed': 131.9  # 秒
                }
            },
            'autoint': {
                'weight': 0.2,           # 综合均衡，补充权重
                'primary_task': 'balance',  # 增强鲁棒性
                'model_path': 'models/saved/AutoInt_real_movielens.pkl', 
                'performance': {
                    'recall_10': 0.007,
                    'rmse': 0.517,
                    'speed': 1012.6  # 秒
                }
            }
        }
        
        self.models = {}
        self.is_trained = False
        self.feature_importance = {}
        self.ensemble_stats = {}
        
        # 多任务融合参数
        self.task_weights = kwargs.get('task_weights', {
            'ranking': 0.5,    # 排序任务权重
            'rating': 0.4,     # 评分预测权重
            'diversity': 0.1   # 多样性权重
        })
        
        logger.info("初始化优化集成推荐器")
        logger.info("算法配置: %s", list(self.algorithm_config.keys()))
    
    def load_pretrained_models(self) -> bool:
        """加载预训练的模型"""
        logger.info("加载预训练模型...")
        
        success_count = 0
        for algo_name, config in self.algorithm_config.items():
            model_path = Path(config['model_path'])
            
            try:
                if model_path.exists():
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    
                    self.models[algo_name] = model
                    success_count += 1
                    logger.info(f"✓ 成功加载 {algo_name} 模型")
                else:
                    logger.error(f"✗ 模型文件不存在: {model_path}")
                    
            except Exception as e:
                logger.error(f"✗ 加载模型失败 {algo_name}: {str(e)}")
        
        if success_count == len(self.algorithm_config):
            self.is_trained = True
            logger.info(f"所有模型加载成功: {success_count}/{len(self.algorithm_config)}")
            return True
        else:
            logger.warning(f"部分模型加载失败: {success_count}/{len(self.algorithm_config)}")
            return False
    
    def get_user_recommendations(self, user_id: int, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        获取用户推荐
        
        多任务融合策略:
        1. SVD提供高质量排序候选
        2. xDeepFM提供精确评分预测
        3. AutoInt提供特征均衡补充
        """
        if not self.is_trained:
            logger.error("模型未训练或加载")
            return []
        
        try:
            # 收集各算法的推荐结果
            algo_recommendations = {}
            algo_scores = {}
            
            for algo_name, model in self.models.items():
                try:
                    # 获取推荐
                    recs = model.get_user_recommendations(user_id, top_k * 2)  # 扩大候选集
                    algo_recommendations[algo_name] = recs
                    
                    # 转换为item_id -> score映射
                    scores = {}
                    if recs:
                        if isinstance(recs[0], tuple):
                            scores = {item_id: score for item_id, score in recs}
                        elif isinstance(recs[0], dict):
                            scores = {rec.get('item_id', 0): rec.get('score', 0.0) for rec in recs}
                    
                    algo_scores[algo_name] = scores
                    
                except Exception as e:
                    logger.debug(f"算法 {algo_name} 推荐失败: {str(e)}")
                    algo_recommendations[algo_name] = []
                    algo_scores[algo_name] = {}
            
            # 多任务融合策略
            final_scores = self._fuse_recommendations(algo_scores, user_id)
            
            # 生成最终推荐
            final_recommendations = []
            for item_id, score in sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
                rec = {
                    'item_id': int(item_id),
                    'score': float(score),
                    'explanation': self._generate_explanation(user_id, item_id, algo_scores),
                    'algorithm_details': {
                        algo: {'score': float(algo_scores[algo].get(item_id, 0.0)), 
                               'weight': self.algorithm_config[algo]['weight']}
                        for algo in self.algorithm_config.keys()
                        if item_id in algo_scores.get(algo, {})
                    }
                }
                final_recommendations.append(rec)
            
            return final_recommendations
            
        except Exception as e:
            logger.error(f"生成推荐失败: {str(e)}")
            return []
    
    def _fuse_recommendations(self, algo_scores: Dict[str, Dict[int, float]], user_id: int) -> Dict[int, float]:
        """
        多任务推荐融合
        
        融合策略:
        1. 基于任务权重的加权融合
        2. 考虑算法特长分配任务
        3. 加入多样性和新颖性调节
        """
        all_items = set()
        for scores in algo_scores.values():
            all_items.update(scores.keys())
        
        final_scores = {}
        
        for item_id in all_items:
            # 任务特定得分计算
            ranking_score = 0.0
            rating_score = 0.0
            balance_score = 0.0
            
            # SVD贡献 - 主要负责排序
            if 'svd' in algo_scores and item_id in algo_scores['svd']:
                svd_score = algo_scores['svd'][item_id]
                ranking_score += svd_score * self.algorithm_config['svd']['weight']
            
            # xDeepFM贡献 - 主要负责评分预测
            if 'xdeepfm' in algo_scores and item_id in algo_scores['xdeepfm']:
                xdeepfm_score = algo_scores['xdeepfm'][item_id]
                rating_score += xdeepfm_score * self.algorithm_config['xdeepfm']['weight']
            
            # AutoInt贡献 - 综合均衡
            if 'autoint' in algo_scores and item_id in algo_scores['autoint']:
                autoint_score = algo_scores['autoint'][item_id]
                balance_score += autoint_score * self.algorithm_config['autoint']['weight']
            
            # 多任务融合
            final_score = (
                ranking_score * self.task_weights['ranking'] +
                rating_score * self.task_weights['rating'] +
                balance_score * (1 - self.task_weights['ranking'] - self.task_weights['rating'])
            )
            
            # 多样性调节（可选）
            diversity_bonus = self._calculate_diversity_bonus(item_id, user_id)
            final_score += diversity_bonus * self.task_weights.get('diversity', 0.1)
            
            final_scores[item_id] = final_score
        
        return final_scores
    
    def _calculate_diversity_bonus(self, item_id: int, user_id: int) -> float:
        """计算多样性奖励"""
        # 简化的多样性计算
        # 实际应用中可以基于类型、特征等计算
        return np.random.uniform(0, 0.1)  # 0-10%的随机多样性奖励
    
    def _generate_explanation(self, user_id: int, item_id: int, algo_scores: Dict[str, Dict[int, float]]) -> Dict[str, Any]:
        """生成推荐解释"""
        explanation = {
            'primary_reason': '',
            'supporting_factors': [],
            'confidence': 0.0,
            'algorithm_contributions': {}
        }
        
        # 分析各算法贡献
        contributions = []
        for algo_name, scores in algo_scores.items():
            if item_id in scores:
                score = scores[item_id]
                weight = self.algorithm_config[algo_name]['weight']
                contribution = score * weight
                
                contributions.append((algo_name, contribution, score))
                explanation['algorithm_contributions'][algo_name] = {
                    'score': float(score),
                    'weight': float(weight),
                    'contribution': float(contribution)
                }
        
        # 确定主要推荐理由
        if contributions:
            # 按贡献排序
            contributions.sort(key=lambda x: x[1], reverse=True)
            top_algo = contributions[0][0]
            
            if top_algo == 'svd':
                explanation['primary_reason'] = "基于协同过滤的高质量推荐"
                explanation['supporting_factors'].append("排序质量最佳")
            elif top_algo == 'xdeepfm':
                explanation['primary_reason'] = "基于深度特征学习的精准预测"
                explanation['supporting_factors'].append("评分预测最准确")
            elif top_algo == 'autoint':
                explanation['primary_reason'] = "基于注意力机制的综合推荐"
                explanation['supporting_factors'].append("特征交互均衡")
            
            # 计算置信度
            total_contribution = sum(c[1] for c in contributions)
            explanation['confidence'] = min(total_contribution / len(contributions), 1.0)
        
        return explanation
    
    def predict(self, user_id: int, item_id: int) -> float:
        """
        预测用户对物品的评分
        
        主要由xDeepFM负责，其他算法提供辅助
        """
        if not self.is_trained:
            return 0.0
        
        predictions = []
        weights = []
        
        for algo_name, model in self.models.items():
            try:
                pred = model.predict(user_id, item_id)
                if pred is not None and not np.isnan(pred):
                    predictions.append(pred)
                    
                    # xDeepFM在评分预测上权重更高
                    if algo_name == 'xdeepfm':
                        weights.append(0.6)  # 评分预测主导
                    elif algo_name == 'autoint':
                        weights.append(0.3)  # 辅助预测
                    else:  # svd评分预测较弱
                        weights.append(0.1)  # 最小权重
                        
            except Exception as e:
                logger.debug(f"算法 {algo_name} 预测失败: {str(e)}")
        
        if predictions:
            # 加权平均
            weights = np.array(weights)
            weights = weights / weights.sum()  # 归一化
            return float(np.average(predictions, weights=weights))
        else:
            return 3.0  # 默认评分
    
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """获取模型性能摘要"""
        return {
            'algorithm_config': self.algorithm_config,
            'task_weights': self.task_weights,
            'models_loaded': list(self.models.keys()),
            'ensemble_strategy': {
                'ranking_leader': 'svd',
                'rating_leader': 'xdeepfm', 
                'balance_support': 'autoint',
                'fusion_method': 'weighted_multi_task'
            },
            'expected_performance': {
                'ranking_quality': 'high',      # 基于SVD的强排序能力
                'rating_accuracy': 'high',      # 基于xDeepFM的强预测能力
                'computational_cost': 'medium', # 平衡的计算开销
                'recommendation_speed': 'fast'  # SVD的高效推荐
            }
        }
    
    def save_ensemble(self, save_path: str) -> bool:
        """保存集成模型"""
        try:
            ensemble_data = {
                'algorithm_config': self.algorithm_config,
                'task_weights': self.task_weights,
                'models': self.models,
                'feature_importance': self.feature_importance,
                'ensemble_stats': self.ensemble_stats,
                'is_trained': self.is_trained
            }
            
            with open(save_path, 'wb') as f:
                pickle.dump(ensemble_data, f)
            
            logger.info(f"集成模型已保存: {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存集成模型失败: {str(e)}")
            return False
    
    def load_ensemble(self, load_path: str) -> bool:
        """加载集成模型"""
        try:
            with open(load_path, 'rb') as f:
                ensemble_data = pickle.load(f)
            
            self.algorithm_config = ensemble_data['algorithm_config']
            self.task_weights = ensemble_data['task_weights']
            self.models = ensemble_data['models']
            self.feature_importance = ensemble_data.get('feature_importance', {})
            self.ensemble_stats = ensemble_data.get('ensemble_stats', {})
            self.is_trained = ensemble_data['is_trained']
            
            logger.info(f"集成模型已加载: {load_path}")
            return True
            
        except Exception as e:
            logger.error(f"加载集成模型失败: {str(e)}")
            return False


def main():
    """测试优化集成推荐器"""
    # 创建集成推荐器
    ensemble = OptimizedEnsembleTeacher()
    
    # 加载预训练模型
    if ensemble.load_pretrained_models():
        logger.info("✅ 优化集成推荐器准备就绪")
        
        # 测试推荐
        test_user_id = 1
        recommendations = ensemble.get_user_recommendations(test_user_id, top_k=5)
        
        print(f"\n🎯 用户 {test_user_id} 的推荐结果:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. 物品 {rec['item_id']}: 分数 {rec['score']:.3f}")
            print(f"   推荐理由: {rec['explanation']['primary_reason']}")
            print(f"   算法贡献: {list(rec['algorithm_details'].keys())}")
        
        # 输出性能摘要
        summary = ensemble.get_model_performance_summary()
        print(f"\n📊 集成模型性能摘要:")
        print(f"   排序领导者: {summary['ensemble_strategy']['ranking_leader']}")
        print(f"   评分领导者: {summary['ensemble_strategy']['rating_leader']}")
        print(f"   预期性能: {summary['expected_performance']}")
        
    else:
        logger.error("❌ 优化集成推荐器初始化失败")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
