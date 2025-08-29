#!/usr/bin/env python3
"""
Traditional Teachers Ensemble Model
传统推荐算法集成教师模型

基于Phase 1的6个SOTA算法性能排名构建加权集成:
- DCNv2 (权重: 0.25) - 最优算法
- DIN (权重: 0.20) - 次优算法  
- xDeepFM (权重: 0.18) - 第三名
- DeepFM (权重: 0.15) - 基础模型
- AutoInt (权重: 0.12) - 注意力模型
- Transformer4Rec (权重: 0.10) - 序列模型

作者: GitHub Copilot
日期: 2025-08-27
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from collections import defaultdict
import pickle
import json

# 导入6个traditional teacher模型
try:
    from models.algorithm_factory import create_recommender
    from models.base_recommender import BaseRecommender
except ImportError:
    # 备用导入路径
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))
    from algorithm_factory import create_recommender
    from base_recommender import BaseRecommender

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleTeacher:
    """6个SOTA算法的集成教师模型"""
    
    def __init__(self, model_weights: Optional[Dict[str, float]] = None, device: str = 'auto'):
        """
        初始化集成教师模型
        
        Args:
            model_weights: 模型权重字典，如果为None则使用默认权重
            device: 计算设备
        """
        # 基于Phase 1性能结果的默认权重
        self.default_weights = {
            'dcnv2': 0.25,         # 最优: 综合得分 0.3676
            'din': 0.20,           # 次优: 注意力机制强
            'xdeepfm': 0.18,       # 第三: 显式+隐式交互
            'deepfm': 0.15,        # 基础: FM+DNN结合
            'autoint': 0.12,       # 自动特征交互
            'transformer4rec': 0.10 # 序列建模
        }
        
        self.weights = model_weights if model_weights else self.default_weights
        self.device = self._setup_device(device)
        self.models = {}
        self.is_trained = False
        
        # 集成策略配置
        self.ensemble_strategy = 'weighted_average'  # 'weighted_average', 'rank_fusion', 'adaptive'
        self.consensus_threshold = 0.7  # 一致性阈值
        
        logger.info(f"初始化Ensemble Teacher，使用设备: {self.device}")
        logger.info(f"模型权重: {self.weights}")
    
    def _setup_device(self, device: str) -> torch.device:
        """设置计算设备"""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                logger.info(f"检测到CUDA，使用GPU: {torch.cuda.get_device_name()}")
            else:
                device = 'cpu'
                logger.info("未检测到CUDA，使用CPU")
        return torch.device(device)
    
    def load_trained_models(self, model_dir: str = "teachers/traditional_teachers/models/saved/"):
        """
        加载已训练的6个模型
        
        Args:
            model_dir: 模型保存目录
        """
        logger.info("🔄 加载已训练的Traditional Teacher模型...")
        
        model_configs = {
            'dcnv2': {'name': 'DCNv2', 'type': 'dcnv2'},
            'din': {'name': 'DIN', 'type': 'din'},
            'xdeepfm': {'name': 'xDeepFM', 'type': 'xdeepfm'},
            'deepfm': {'name': 'DeepFM', 'type': 'deepfm'},
            'autoint': {'name': 'AutoInt', 'type': 'autoint'},
            'transformer4rec': {'name': 'Transformer4Rec', 'type': 'transformer4rec'}
        }
        
        successful_loads = 0
        
        for model_key, config in model_configs.items():
            try:
                # 创建模型实例
                model = create_recommender(
                    config['type'],
                    num_users=610,  # MovieLens默认
                    num_items=9742,
                    embedding_dim=64
                )
                
                if model is None:
                    logger.warning(f"⚠️ {config['name']} 创建失败，跳过")
                    continue
                
                # 尝试加载预训练权重（如果存在）
                model_path = os.path.join(model_dir, f"{model_key}_best.pth")
                if os.path.exists(model_path):
                    try:
                        # 对于PyTorch模型，尝试加载state_dict
                        if hasattr(model, 'load_state_dict') and hasattr(model, 'state_dict'):
                            state_dict = torch.load(model_path, map_location=self.device)
                            model.load_state_dict(state_dict)
                            logger.info(f"✅ {config['name']} PyTorch权重加载成功")
                        else:
                            # 对于非PyTorch模型，使用pickle加载
                            model.load_model(model_path)
                            logger.info(f"✅ {config['name']} 模型加载成功")
                    except Exception as load_e:
                        logger.warning(f"⚠️ {config['name']} 权重加载失败: {load_e}，使用随机初始化")
                else:
                    logger.warning(f"⚠️ {config['name']} 权重文件未找到，使用随机初始化")
                
                # 设置为评估模式（如果支持）
                if hasattr(model, 'eval'):
                    model.eval()
                if hasattr(model, 'to'):
                    model.to(self.device)
                
                # 包装为EnsembleMember
                self.models[model_key] = EnsembleMember(
                    model=model,
                    weight=self.weights[model_key],
                    name=config['name'],
                    model_type=config['type']
                )
                
                successful_loads += 1
                
            except Exception as e:
                logger.error(f"❌ {config['name']} 加载失败: {str(e)}")
                # 移除失败模型的权重
                if model_key in self.weights:
                    del self.weights[model_key]
        
        # 重新归一化权重
        if successful_loads > 0:
            total_weight = sum(self.weights.values())
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
            self.is_trained = True
            logger.info(f"✅ 成功加载 {successful_loads}/6 个模型")
            logger.info(f"重新归一化权重: {self.weights}")
        else:
            raise RuntimeError("❌ 所有模型加载失败，无法构建Ensemble")
    
    def get_ensemble_predictions(self, user_ids: List[int], item_ids: List[int]) -> Dict[str, Any]:
        """
        获取集成预测结果
        
        Args:
            user_ids: 用户ID列表
            item_ids: 物品ID列表
            
        Returns:
            包含预测结果和元信息的字典
        """
        if not self.is_trained:
            raise RuntimeError("模型未训练，请先调用load_trained_models()")
        
        logger.info(f"🔮 集成预测: {len(user_ids)} 用户 × {len(item_ids)} 物品")
        
        # 收集各模型预测
        model_predictions = {}
        model_confidences = {}
        
        for model_key, member in self.models.items():
            try:
                with torch.no_grad():
                    # 获取模型预测
                    predictions = member.predict(user_ids, item_ids)
                    confidences = member.get_confidence(predictions)
                    
                    model_predictions[model_key] = predictions
                    model_confidences[model_key] = confidences
                    
                logger.debug(f"{member.name} 预测完成: {predictions.shape}")
                
            except Exception as e:
                logger.error(f"{member.name} 预测失败: {str(e)}")
                continue
        
        # 执行集成
        if self.ensemble_strategy == 'weighted_average':
            ensemble_pred = self._weighted_average_ensemble(model_predictions)
        elif self.ensemble_strategy == 'rank_fusion':
            ensemble_pred = self._rank_fusion_ensemble(model_predictions)
        elif self.ensemble_strategy == 'adaptive':
            ensemble_pred = self._adaptive_ensemble(model_predictions, model_confidences)
        else:
            raise ValueError(f"未知的集成策略: {self.ensemble_strategy}")
        
        return {
            'ensemble_predictions': ensemble_pred,
            'individual_predictions': model_predictions,
            'model_confidences': model_confidences,
            'weights_used': self.weights,
            'strategy': self.ensemble_strategy
        }
    
    def get_user_recommendations(self, user_id: int, top_k: int = 10, 
                               candidate_items: Optional[List[int]] = None) -> List[Tuple[int, float]]:
        """
        为单个用户获取Top-K推荐
        
        Args:
            user_id: 用户ID
            top_k: 推荐数量
            candidate_items: 候选物品列表，如果为None则使用所有物品
            
        Returns:
            [(item_id, score), ...] 推荐列表
        """
        if candidate_items is None:
            # 默认候选物品范围（可根据实际数据调整）
            candidate_items = list(range(1, 1000))  # 示例范围
        
        # 获取预测分数
        result = self.get_ensemble_predictions([user_id], candidate_items)
        predictions = result['ensemble_predictions']
        
        # 组合物品ID和分数
        item_scores = list(zip(candidate_items, predictions[0]))
        
        # 按分数降序排序并返回Top-K
        item_scores.sort(key=lambda x: x[1], reverse=True)
        
        return item_scores[:top_k]
    
    def _weighted_average_ensemble(self, model_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """加权平均集成"""
        if not model_predictions:
            raise ValueError("没有可用的模型预测")
        
        # 初始化结果
        first_pred = next(iter(model_predictions.values()))
        ensemble_pred = np.zeros_like(first_pred)
        total_weight = 0
        
        # 加权求和
        for model_key, predictions in model_predictions.items():
            if model_key in self.weights:
                weight = self.weights[model_key]
                ensemble_pred += weight * predictions
                total_weight += weight
        
        # 归一化
        if total_weight > 0:
            ensemble_pred /= total_weight
        
        return ensemble_pred
    
    def _rank_fusion_ensemble(self, model_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """排序融合集成（适用于推荐场景）"""
        if not model_predictions:
            raise ValueError("没有可用的模型预测")
        
        # 将预测转换为排序
        model_ranks = {}
        for model_key, predictions in model_predictions.items():
            # 对每个用户的预测进行排序
            ranks = np.argsort(np.argsort(-predictions, axis=1), axis=1)
            model_ranks[model_key] = ranks
        
        # 加权排序融合
        ensemble_ranks = self._weighted_average_ensemble(model_ranks)
        
        # 将排序转换回分数（倒排序）
        max_rank = ensemble_ranks.max()
        ensemble_pred = max_rank - ensemble_ranks
        
        return ensemble_pred
    
    def _adaptive_ensemble(self, model_predictions: Dict[str, np.ndarray], 
                          model_confidences: Dict[str, np.ndarray]) -> np.ndarray:
        """自适应集成（基于模型置信度动态调整权重）"""
        if not model_predictions:
            raise ValueError("没有可用的模型预测")
        
        # 动态权重计算
        adaptive_weights = {}
        for model_key in model_predictions.keys():
            if model_key in self.weights and model_key in model_confidences:
                base_weight = self.weights[model_key]
                confidence = model_confidences[model_key].mean()
                adaptive_weights[model_key] = base_weight * confidence
        
        # 归一化自适应权重
        total_weight = sum(adaptive_weights.values())
        if total_weight > 0:
            adaptive_weights = {k: v/total_weight for k, v in adaptive_weights.items()}
        
        # 使用自适应权重进行加权平均
        first_pred = next(iter(model_predictions.values()))
        ensemble_pred = np.zeros_like(first_pred)
        
        for model_key, predictions in model_predictions.items():
            if model_key in adaptive_weights:
                ensemble_pred += adaptive_weights[model_key] * predictions
        
        return ensemble_pred
    
    def analyze_model_consensus(self, user_ids: List[int], item_ids: List[int]) -> Dict[str, Any]:
        """
        分析模型一致性
        
        Returns:
            一致性分析结果
        """
        result = self.get_ensemble_predictions(user_ids, item_ids)
        predictions = result['individual_predictions']
        
        if len(predictions) < 2:
            return {'consensus_score': 1.0, 'agreement_matrix': None}
        
        # 计算两两相关性
        model_keys = list(predictions.keys())
        n_models = len(model_keys)
        agreement_matrix = np.zeros((n_models, n_models))
        
        for i, key1 in enumerate(model_keys):
            for j, key2 in enumerate(model_keys):
                if i <= j:
                    corr = np.corrcoef(predictions[key1].flatten(), 
                                     predictions[key2].flatten())[0, 1]
                    agreement_matrix[i, j] = agreement_matrix[j, i] = corr if not np.isnan(corr) else 0
        
        # 计算平均一致性
        consensus_score = agreement_matrix[np.triu_indices(n_models, k=1)].mean()
        
        return {
            'consensus_score': consensus_score,
            'agreement_matrix': agreement_matrix,
            'model_keys': model_keys,
            'high_consensus': consensus_score > self.consensus_threshold
        }
    
    def save_ensemble_config(self, save_path: str):
        """保存集成配置"""
        config = {
            'weights': self.weights,
            'ensemble_strategy': self.ensemble_strategy,
            'consensus_threshold': self.consensus_threshold,
            'device': str(self.device)
        }
        
        with open(save_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"集成配置已保存到: {save_path}")
    
    def load_ensemble_config(self, config_path: str):
        """加载集成配置"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.weights = config['weights']
        self.ensemble_strategy = config['ensemble_strategy']
        self.consensus_threshold = config['consensus_threshold']
        
        logger.info(f"集成配置已加载: {config_path}")


class EnsembleMember:
    """集成成员包装器"""
    
    def __init__(self, model: BaseRecommender, weight: float, name: str, model_type: str):
        self.model = model
        self.weight = weight
        self.name = name
        self.model_type = model_type
    
    def predict(self, user_ids: List[int], item_ids: List[int]) -> np.ndarray:
        """获取模型预测"""
        predictions = []
        
        # 对每个用户-物品对进行预测
        for user_id in user_ids:
            user_predictions = []
            for item_id in item_ids:
                try:
                    score = self.model.predict(user_id, item_id)
                    user_predictions.append(float(score))
                except Exception as e:
                    # 预测失败时使用默认值
                    user_predictions.append(0.0)
            predictions.append(user_predictions)
        
        return np.array(predictions)
    
    def get_confidence(self, predictions: np.ndarray) -> np.ndarray:
        """计算预测置信度（简单版本：使用预测分布特征）"""
        # 基于预测值的分布计算置信度
        # 1. 计算每行的标准差（预测一致性）
        std_per_user = np.std(predictions, axis=1, keepdims=True)
        
        # 2. 计算每行的平均值（预测强度）
        mean_per_user = np.mean(predictions, axis=1, keepdims=True)
        
        # 3. 置信度 = 预测强度 / (1 + 预测方差)
        # 预测越强、方差越小，置信度越高
        confidence = np.abs(mean_per_user) / (1.0 + std_per_user + 1e-8)
        
        return confidence
    
    def get_user_recommendations(self, user_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """获取用户推荐（委托给底层模型）"""
        try:
            recs = self.model.get_user_recommendations(user_id, top_k)
            result = []
            
            for rec in recs:
                if isinstance(rec, dict) and 'item_id' in rec and 'score' in rec:
                    result.append((int(rec['item_id']), float(rec['score'])))
                elif isinstance(rec, (list, tuple)) and len(rec) >= 2:
                    result.append((int(rec[0]), float(rec[1])))
                    
            return result
        except Exception as e:
            logger.warning(f"{self.name} 推荐失败: {e}")
            return []


def demo_ensemble_teacher():
    """演示Ensemble Teacher使用"""
    print("🚀 演示Traditional Teachers Ensemble...")
    
    # 创建集成教师
    ensemble = EnsembleTeacher()
    
    try:
        # 加载模型
        ensemble.load_trained_models()
        
        # 示例用户和物品
        test_users = [1, 2, 3, 4, 5]
        test_items = [10, 20, 30, 40, 50]
        
        # 获取集成预测
        result = ensemble.get_ensemble_predictions(test_users, test_items)
        
        print(f"✅ 集成预测完成")
        print(f"   集成策略: {result['strategy']}")
        print(f"   使用权重: {result['weights_used']}")
        print(f"   预测形状: {result['ensemble_predictions'].shape}")
        
        # 获取用户推荐
        user_recs = ensemble.get_user_recommendations(user_id=1, top_k=5)
        print(f"\n📋 用户1的Top-5推荐:")
        for i, (item_id, score) in enumerate(user_recs):
            print(f"   {i+1}. 物品{item_id}: {score:.4f}")
        
        # 分析模型一致性
        consensus = ensemble.analyze_model_consensus(test_users, test_items)
        print(f"\n🤝 模型一致性分析:")
        print(f"   一致性得分: {consensus['consensus_score']:.4f}")
        print(f"   高一致性: {consensus['high_consensus']}")
        
        print("\n🎉 Ensemble Teacher演示完成！")
        
    except Exception as e:
        print(f"❌ 演示失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demo_ensemble_teacher()
