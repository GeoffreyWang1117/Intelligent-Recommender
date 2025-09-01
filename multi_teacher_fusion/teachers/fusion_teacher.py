"""
Multi-Teacher Fusion System - Fusion Manager
融合多个Teacher的智能管理器，支持Ensemble + LLM的协同推荐
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
from dataclasses import dataclass
import requests
import json

from .base_teacher import BaseTeacher, UserProfile, ItemProfile, RecommendationResult
from .ensemble_teacher import EnsembleTeacher

@dataclass
class FusionConfig:
    """融合配置"""
    ensemble_weight: float = 0.7
    llm_weight: float = 0.3
    
    # LLM配置
    llm_model: str = "llama3:latest"
    llm_base_url: str = "http://localhost:11434"
    llm_temperature: float = 0.7
    llm_timeout: int = 15
    
    # 融合策略
    fusion_method: str = "weighted_average"  # weighted_average, rank_fusion, neural_fusion
    diversification: bool = True
    max_candidates: int = 100

class MultiTeacherFusion(BaseTeacher):
    """多Teacher融合系统"""
    
    def __init__(self, 
                 fusion_config: FusionConfig,
                 ensemble_teacher: EnsembleTeacher,
                 movie_data: Optional[pd.DataFrame] = None):
        """
        初始化多Teacher融合系统
        
        Args:
            fusion_config: 融合配置
            ensemble_teacher: 已训练的Ensemble Teacher
            movie_data: 电影元数据
        """
        super().__init__("MultiTeacherFusion")
        self.config = fusion_config
        self.ensemble_teacher = ensemble_teacher
        self.movie_data = movie_data
        self.logger = logging.getLogger(__name__)
        
        # 构建电影映射
        self.movie_id_to_title = {}
        if movie_data is not None:
            for _, row in movie_data.iterrows():
                self.movie_id_to_title[int(row['movieId'])] = row['title']
        
        # 测试LLM连接
        self.llm_available = self._test_llm_connection()
        
        self.logger.info(f"多Teacher融合系统初始化完成")
        self.logger.info(f"  Ensemble权重: {fusion_config.ensemble_weight}")
        self.logger.info(f"  LLM权重: {fusion_config.llm_weight}")
        self.logger.info(f"  LLM可用: {self.llm_available}")
    
    def _test_llm_connection(self) -> bool:
        """测试LLM连接"""
        try:
            response = requests.get(f"{self.config.llm_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                
                if any(self.config.llm_model in name for name in model_names):
                    self.logger.info(f"✅ LLM连接成功: {self.config.llm_model}")
                    return True
                else:
                    self.logger.warning(f"⚠️ LLM模型未找到: {self.config.llm_model}")
                    return False
            else:
                self.logger.warning(f"⚠️ LLM连接失败: HTTP {response.status_code}")
                return False
        except Exception as e:
            self.logger.warning(f"⚠️ LLM连接测试失败: {e}")
            return False
    
    def _call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """调用LLM"""
        if not self.llm_available:
            return ""
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "model": self.config.llm_model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.config.llm_temperature,
                    "num_predict": 256
                }
            }
            
            response = requests.post(
                f"{self.config.llm_base_url}/api/chat",
                json=payload,
                timeout=self.config.llm_timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['message']['content'].strip()
            else:
                self.logger.warning(f"LLM调用失败: HTTP {response.status_code}")
                return ""
                
        except Exception as e:
            self.logger.warning(f"LLM调用异常: {e}")
            return ""
    
    def initialize(self, **kwargs) -> bool:
        """初始化融合系统"""
        try:
            # 初始化Ensemble Teacher
            if hasattr(self.ensemble_teacher, 'initialize'):
                ensemble_init = self.ensemble_teacher.initialize(**kwargs)
            else:
                ensemble_init = True
            
            self.logger.info(f"融合系统初始化: Ensemble={ensemble_init}, LLM={self.llm_available}")
            return ensemble_init or self.llm_available
            
        except Exception as e:
            self.logger.error(f"融合系统初始化失败: {e}")
            return False
    
    def predict(self, user_id: int, item_id: int, **kwargs) -> float:
        """预测用户对物品的评分"""
        try:
            # Ensemble预测
            ensemble_score = 3.5  # 默认分数
            if hasattr(self.ensemble_teacher, 'predict'):
                ensemble_score = self.ensemble_teacher.predict(user_id, item_id, **kwargs)
            elif hasattr(self.ensemble_teacher, 'predict_rating'):
                ensemble_score = self.ensemble_teacher.predict_rating(user_id, item_id)
            
            # LLM增强（如果可用）
            llm_adjustment = 0.0
            if self.llm_available:
                llm_adjustment = self._get_llm_rating_adjustment(user_id, item_id)
            
            # 融合评分
            if self.config.fusion_method == "weighted_average":
                final_score = (
                    ensemble_score * self.config.ensemble_weight + 
                    (ensemble_score + llm_adjustment) * self.config.llm_weight
                )
            else:
                final_score = ensemble_score + llm_adjustment * 0.1
            
            return max(1.0, min(5.0, final_score))
            
        except Exception as e:
            self.logger.warning(f"预测失败: {e}")
            return 3.5
    
    def _get_llm_rating_adjustment(self, user_id: int, item_id: int) -> float:
        """获取LLM的评分调整"""
        try:
            movie_title = self.movie_id_to_title.get(item_id, f"Movie_{item_id}")
            
            prompt = f"""请评估用户{user_id}对电影"{movie_title}"的可能喜好程度。
返回一个-1到1之间的调整值，其中：
-1表示不太喜欢（降低评分）
0表示中性
1表示很喜欢（提高评分）

只返回数字，不要解释。"""
            
            response = self._call_llm(prompt)
            if response:
                # 尝试提取数值
                import re
                numbers = re.findall(r'-?\d+\.?\d*', response)
                if numbers:
                    adjustment = float(numbers[0])
                    return max(-1.0, min(1.0, adjustment))
            
            return 0.0
            
        except:
            return 0.0
    
    def get_recommendations(self, user_id: int, num_recommendations: int = 10, **kwargs) -> RecommendationResult:
        """生成融合推荐"""
        try:
            # 1. 获取Ensemble推荐
            ensemble_recs = []
            if hasattr(self.ensemble_teacher, 'get_recommendations'):
                ensemble_result = self.ensemble_teacher.get_recommendations(
                    user_id, num_recommendations * 2, **kwargs
                )
                if hasattr(ensemble_result, 'item_recommendations'):
                    ensemble_recs = ensemble_result.item_recommendations
            
            # 如果Ensemble没有推荐，生成候选列表
            if not ensemble_recs:
                # 简单的候选生成（实际应用中需要更复杂的逻辑）
                all_items = list(self.movie_id_to_title.keys())[:self.config.max_candidates]
                ensemble_recs = [(item_id, 3.5) for item_id in all_items[:num_recommendations * 2]]
            
            # 2. LLM重排序和优化（如果可用）
            if self.llm_available and len(ensemble_recs) > 0:
                optimized_recs = self._llm_rerank_recommendations(
                    user_id, ensemble_recs, num_recommendations
                )
            else:
                optimized_recs = ensemble_recs[:num_recommendations]
            
            # 3. 构建最终推荐结果
            final_recs = []
            confidence_scores = []
            
            for i, (item_id, score) in enumerate(optimized_recs[:num_recommendations]):
                # 计算融合置信度
                base_confidence = max(0.1, 1.0 - i * 0.1)
                if self.llm_available:
                    llm_boost = 0.2  # LLM可用时提升置信度
                else:
                    llm_boost = 0.0
                
                confidence = min(1.0, base_confidence + llm_boost)
                confidence_scores.append(confidence)
                final_recs.append((item_id, score))
            
            return RecommendationResult(
                user_id=user_id,
                item_recommendations=final_recs,
                confidence=np.mean(confidence_scores) if confidence_scores else 0.5,
                reasoning=f"融合推荐：Ensemble({self.config.ensemble_weight}) + LLM({self.config.llm_weight})"
            )
            
        except Exception as e:
            self.logger.error(f"生成推荐失败: {e}")
            # 返回默认推荐
            default_items = list(self.movie_id_to_title.keys())[:num_recommendations]
            default_recs = [(item_id, 3.5) for item_id in default_items]
            
            return RecommendationResult(
                user_id=user_id,
                item_recommendations=default_recs,
                confidence=0.3,
                reasoning="默认推荐（融合失败）"
            )
    
    def _llm_rerank_recommendations(self, 
                                  user_id: int, 
                                  candidate_recs: List[Tuple[int, float]], 
                                  num_final: int) -> List[Tuple[int, float]]:
        """使用LLM重排序推荐"""
        try:
            # 构建候选电影列表
            movie_list = []
            for item_id, score in candidate_recs[:20]:  # 限制候选数量
                title = self.movie_id_to_title.get(item_id, f"Movie_{item_id}")
                movie_list.append(f"{item_id}: {title} (基础分数: {score:.2f})")
            
            movies_text = "\n".join(movie_list)
            
            prompt = f"""用户{user_id}的推荐候选列表：
{movies_text}

请从中选择最适合的{num_final}部电影，考虑多样性和个性化。
返回格式：电影ID:调整后分数，每行一个，按推荐优先级排序。
例如：
123:4.5
456:4.2
789:4.0"""
            
            response = self._call_llm(prompt)
            
            if response:
                # 解析LLM响应
                reranked = []
                for line in response.split('\n'):
                    line = line.strip()
                    if ':' in line:
                        try:
                            parts = line.split(':')
                            item_id = int(parts[0].strip())
                            score = float(parts[1].strip())
                            reranked.append((item_id, score))
                        except:
                            continue
                
                if reranked:
                    return reranked[:num_final]
            
            # 如果LLM重排序失败，返回原始排序
            return candidate_recs[:num_final]
            
        except Exception as e:
            self.logger.warning(f"LLM重排序失败: {e}")
            return candidate_recs[:num_final]
    
    def explain_recommendation(self, user_id: int, item_id: int) -> str:
        """解释推荐原因"""
        try:
            movie_title = self.movie_id_to_title.get(item_id, f"Movie_{item_id}")
            
            # 获取Ensemble的解释
            ensemble_explanation = "基于协同过滤和深度学习的综合分析"
            if hasattr(self.ensemble_teacher, 'explain_recommendation'):
                ensemble_explanation = self.ensemble_teacher.explain_recommendation(user_id, item_id)
            
            # LLM增强解释
            if self.llm_available:
                prompt = f"""为什么向用户{user_id}推荐电影"{movie_title}"？
基础分析：{ensemble_explanation}

请提供一个简洁的个性化解释（50字以内）。"""
                
                llm_explanation = self._call_llm(prompt)
                if llm_explanation:
                    return f"{llm_explanation} (融合推荐)"
            
            return f"推荐《{movie_title}》：{ensemble_explanation}"
            
        except Exception as e:
            self.logger.warning(f"解释生成失败: {e}")
            return f"基于多Teacher融合算法的推荐"
    
    def get_user_embedding(self, user_id: int) -> np.ndarray:
        """获取用户嵌入表示"""
        if self.ensemble_teacher and hasattr(self.ensemble_teacher, 'get_user_embedding'):
            return self.ensemble_teacher.get_user_embedding(user_id)
        else:
            # 返回默认嵌入或随机嵌入
            return np.random.randn(64)  # 默认64维嵌入
    
    def get_item_embedding(self, item_id: int) -> np.ndarray:
        """获取物品嵌入表示"""
        if self.ensemble_teacher and hasattr(self.ensemble_teacher, 'get_item_embedding'):
            return self.ensemble_teacher.get_item_embedding(item_id)
        else:
            # 返回默认嵌入或随机嵌入
            return np.random.randn(64)  # 默认64维嵌入

if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 创建配置
    config = FusionConfig(
        ensemble_weight=0.7,
        llm_weight=0.3,
        llm_model="llama3:latest"
    )
    
    print("多Teacher融合系统配置完成!")
    print(f"配置: Ensemble({config.ensemble_weight}) + LLM({config.llm_weight})")
    print(f"LLM模型: {config.llm_model}")
    print(f"融合方法: {config.fusion_method}")
