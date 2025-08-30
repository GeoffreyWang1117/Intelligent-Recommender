"""
Ensemble Teacher实现
集成SVD + xDeepFM + AutoInt三个算法的Teacher
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple
import os
import logging

from .base_teacher import BaseTeacher, UserProfile, ItemProfile, RecommendationResult


class EnsembleTeacher(BaseTeacher):
    """
    Ensemble Teacher - 集成SVD + xDeepFM + AutoInt
    基于已验证的最优三算法组合
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("EnsembleTeacher", config)
        
        # 模型组件
        self.svd_model = None
        self.xdeepfm_model = None  
        self.autoint_model = None
        
        # 配置参数
        self.teacher_weights = config.get("teacher_weights", {
            "svd": 0.5,      # 最佳推荐质量
            "xdeepfm": 0.3,  # 最佳评分预测  
            "autoint": 0.2   # 注意力机制增强
        })
        
        # 模型路径
        self.svd_model_path = config.get("svd_model_path", "models/saved/SVD_real_movielens.pkl")
        self.xdeepfm_model_path = config.get("xdeepfm_model_path", "models/saved/xDeepFM_model.pth")
        self.autoint_model_path = config.get("autoint_model_path", "models/saved/AutoInt_model.pth")
        
        # 数据映射 (如果需要从原有系统加载)
        self.user_id_map = {}
        self.item_id_map = {}
        
    def initialize(self) -> bool:
        """初始化所有三个模型"""
        try:
            # 初始化SVD模型
            success_svd = self._initialize_svd()
            
            # 初始化xDeepFM模型  
            success_xdeepfm = self._initialize_xdeepfm()
            
            # 初始化AutoInt模型
            success_autoint = self._initialize_autoint()
            
            # 所有模型都成功初始化
            self.is_initialized = success_svd and success_xdeepfm and success_autoint
            
            if self.is_initialized:
                self.logger.info("All three ensemble models initialized successfully")
            else:
                self.logger.warning("Some ensemble models failed to initialize")
                
            return self.is_initialized
            
        except Exception as e:
            self.logger.error("Failed to initialize ensemble teacher: %s", str(e))
            return False
    
    def _initialize_svd(self) -> bool:
        """初始化SVD模型"""
        try:
            if os.path.exists(self.svd_model_path):
                with open(self.svd_model_path, 'rb') as f:
                    self.svd_model = pickle.load(f)
                self.logger.info("SVD model loaded from %s", self.svd_model_path)
                return True
            else:
                self.logger.warning("SVD model file not found: %s", self.svd_model_path)
                return False
        except Exception as e:
            self.logger.error("Failed to load SVD model: %s", str(e))
            return False
    
    def _initialize_xdeepfm(self) -> bool:
        """初始化xDeepFM模型"""
        try:
            if os.path.exists(self.xdeepfm_model_path):
                # 这里需要根据实际的xDeepFM模型结构来加载
                # 暂时返回True，实际实现时需要完善
                self.logger.info("xDeepFM model path found: %s", self.xdeepfm_model_path)
                return True
            else:
                self.logger.warning("xDeepFM model file not found: %s", self.xdeepfm_model_path)
                return False
        except Exception as e:
            self.logger.error("Failed to load xDeepFM model: %s", str(e))
            return False
    
    def _initialize_autoint(self) -> bool:
        """初始化AutoInt模型"""
        try:
            if os.path.exists(self.autoint_model_path):
                # 这里需要根据实际的AutoInt模型结构来加载
                # 暂时返回True，实际实现时需要完善
                self.logger.info("AutoInt model path found: %s", self.autoint_model_path)
                return True
            else:
                self.logger.warning("AutoInt model file not found: %s", self.autoint_model_path)
                return False
        except Exception as e:
            self.logger.error("Failed to load AutoInt model: %s", str(e))
            return False
    
    def predict(self, user_profile: UserProfile, candidate_items: List[ItemProfile], 
                num_recommendations: int = 10) -> RecommendationResult:
        """生成Ensemble推荐结果"""
        
        if not self.is_initialized:
            return RecommendationResult(
                user_id=user_profile.user_id,
                item_recommendations=[],
                confidence=0.0,
                reasoning="Ensemble teacher not initialized",
                teacher_name=self.teacher_name
            )
        
        try:
            # 获取各个模型的预测结果
            svd_scores = self._predict_svd(user_profile, candidate_items)
            xdeepfm_scores = self._predict_xdeepfm(user_profile, candidate_items)
            autoint_scores = self._predict_autoint(user_profile, candidate_items)
            
            # 加权融合
            final_scores = self._weighted_fusion(svd_scores, xdeepfm_scores, autoint_scores)
            
            # 排序并选择Top-K
            item_score_pairs = [(item.item_id, score) for item, score in zip(candidate_items, final_scores)]
            item_score_pairs.sort(key=lambda x: x[1], reverse=True)
            top_recommendations = item_score_pairs[:num_recommendations]
            
            # 计算置信度
            confidence = self._calculate_confidence(svd_scores, xdeepfm_scores, autoint_scores)
            
            # 生成推理解释
            reasoning = self._generate_reasoning(user_profile, top_recommendations)
            
            return RecommendationResult(
                user_id=user_profile.user_id,
                item_recommendations=top_recommendations,
                confidence=confidence,
                reasoning=reasoning,
                teacher_name=self.teacher_name,
                metadata={
                    "ensemble_weights": self.teacher_weights,
                    "individual_predictions": {
                        "svd_available": self.svd_model is not None,
                        "xdeepfm_available": self.xdeepfm_model is not None,
                        "autoint_available": self.autoint_model is not None
                    }
                }
            )
            
        except Exception as e:
            self.logger.error("Ensemble prediction failed: %s", str(e))
            return RecommendationResult(
                user_id=user_profile.user_id,
                item_recommendations=[],
                confidence=0.0,
                reasoning=f"Ensemble prediction error: {str(e)}",
                teacher_name=self.teacher_name
            )
    
    def _predict_svd(self, user_profile: UserProfile, candidate_items: List[ItemProfile]) -> List[float]:
        """SVD模型预测"""
        if self.svd_model is None:
            return [0.0] * len(candidate_items)
        
        scores = []
        for item in candidate_items:
            try:
                # 使用SVD模型预测评分
                # 这里需要根据实际SVD模型接口调整
                if hasattr(self.svd_model, 'predict'):
                    score = self.svd_model.predict(user_profile.user_id, item.item_id)
                else:
                    score = 3.0  # 默认评分
                scores.append(float(score))
            except Exception:
                scores.append(0.0)
        
        return scores
    
    def _predict_xdeepfm(self, user_profile: UserProfile, candidate_items: List[ItemProfile]) -> List[float]:
        """xDeepFM模型预测"""
        if self.xdeepfm_model is None:
            return [0.0] * len(candidate_items)
        
        # 这里需要实现xDeepFM的预测逻辑
        # 暂时返回随机分数
        return [3.5 + np.random.normal(0, 0.5) for _ in candidate_items]
    
    def _predict_autoint(self, user_profile: UserProfile, candidate_items: List[ItemProfile]) -> List[float]:
        """AutoInt模型预测"""
        if self.autoint_model is None:
            return [0.0] * len(candidate_items)
        
        # 这里需要实现AutoInt的预测逻辑
        # 暂时返回随机分数
        return [3.0 + np.random.normal(0, 0.3) for _ in candidate_items]
    
    def _weighted_fusion(self, svd_scores: List[float], xdeepfm_scores: List[float], 
                        autoint_scores: List[float]) -> List[float]:
        """加权融合三个模型的分数"""
        final_scores = []
        
        for i in range(len(svd_scores)):
            weighted_score = (
                self.teacher_weights["svd"] * svd_scores[i] +
                self.teacher_weights["xdeepfm"] * xdeepfm_scores[i] +
                self.teacher_weights["autoint"] * autoint_scores[i]
            )
            final_scores.append(weighted_score)
        
        return final_scores
    
    def _calculate_confidence(self, svd_scores: List[float], xdeepfm_scores: List[float], 
                            autoint_scores: List[float]) -> float:
        """计算预测置信度"""
        # 基于模型间的一致性计算置信度
        if len(svd_scores) == 0:
            return 0.0
        
        agreements = []
        for i in range(len(svd_scores)):
            scores = [svd_scores[i], xdeepfm_scores[i], autoint_scores[i]]
            score_std = np.std(scores)
            agreement = 1.0 / (1.0 + score_std)  # 方差越小，一致性越高
            agreements.append(agreement)
        
        return float(np.mean(agreements))
    
    def _generate_reasoning(self, user_profile: UserProfile, 
                          recommendations: List[Tuple[int, float]]) -> str:
        """生成推荐理由"""
        reasoning_parts = [
            f"Based on ensemble of SVD, xDeepFM, and AutoInt models",
            f"User {user_profile.user_id} has {user_profile.num_ratings} rating history",
            f"Average rating: {user_profile.avg_rating:.2f}",
            f"Top recommendation score: {recommendations[0][1]:.3f}" if recommendations else "No recommendations"
        ]
        
        return "; ".join(reasoning_parts)
    
    def get_user_embedding(self, user_profile: UserProfile) -> List[float]:
        """获取用户嵌入 (暂时返回随机向量)"""
        # 这里应该从三个模型中提取用户嵌入并融合
        embedding_dim = 64
        return [float(x) for x in np.random.normal(0, 0.1, embedding_dim)]
    
    def get_item_embedding(self, item_profile: ItemProfile) -> List[float]:
        """获取物品嵌入 (暂时返回随机向量)"""
        # 这里应该从三个模型中提取物品嵌入并融合
        embedding_dim = 64
        return [float(x) for x in np.random.normal(0, 0.1, embedding_dim)]


if __name__ == "__main__":
    # 测试EnsembleTeacher
    config = {
        "teacher_weights": {"svd": 0.5, "xdeepfm": 0.3, "autoint": 0.2}
    }
    
    teacher = EnsembleTeacher(config)
    print(f"Teacher: {teacher}")
    print(f"Weights: {teacher.teacher_weights}")
    
    # 测试初始化
    success = teacher.initialize()
    print(f"Initialization success: {success}")
