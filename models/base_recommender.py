#!/usr/bin/env python3
"""
推荐系统基础接口模块

定义了所有推荐算法的统一接口，便于算法的替换和扩展。
"""

import logging
import pickle
import os
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class BaseRecommender(ABC):
    """推荐模型基类
    
    所有推荐算法都应该继承此基类，并实现其抽象方法。
    这确保了算法间的一致性接口，便于在系统中替换不同的算法。
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        初始化推荐器
        
        Args:
            model_name: 模型名称
            **kwargs: 其他模型特定参数
        """
        self.model_name = model_name
        self.is_trained = False
        self.model_params = kwargs
        
    @abstractmethod
    def fit(self, train_data: pd.DataFrame, **kwargs) -> None:
        """训练模型
        
        Args:
            train_data: 训练数据，包含user_id, item_id, rating列
            **kwargs: 训练相关参数
        """
        pass
    
    @abstractmethod
    def predict(self, user_id: int, item_id: int) -> float:
        """预测用户对物品的评分
        
        Args:
            user_id: 用户ID
            item_id: 物品ID
            
        Returns:
            预测评分
        """
        pass
    
    @abstractmethod
    def get_user_recommendations(self, user_id: int, top_k: int = 10) -> List[Dict[str, Any]]:
        """获取用户推荐
        
        Args:
            user_id: 用户ID
            top_k: 推荐数量
            
        Returns:
            推荐列表，格式为[{'item_id': int, 'score': float}, ...]
        """
        pass
    
    def _format_recommendations(self, recommendations: List[Tuple[int, float]]) -> List[Dict[str, Any]]:
        """格式化推荐结果为标准格式
        
        Args:
            recommendations: 原始推荐结果 [(item_id, score), ...]
            
        Returns:
            标准格式的推荐结果 [{'item_id': int, 'score': float}, ...]
        """
        if not recommendations:
            return []
        
        formatted_recs = []
        for item in recommendations:
            if isinstance(item, tuple) and len(item) >= 2:
                formatted_recs.append({
                    'item_id': int(item[0]),
                    'score': float(item[1])
                })
            elif isinstance(item, dict):
                # 已经是字典格式，确保包含必需字段
                if 'item_id' in item and 'score' in item:
                    formatted_recs.append({
                        'item_id': int(item['item_id']),
                        'score': float(item['score'])
                    })
        
        return formatted_recs
    
    def save_model(self, filepath: str) -> bool:
        """保存模型
        
        Args:
            filepath: 保存路径
            
        Returns:
            是否保存成功
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            logger.info(f"模型已保存到: {filepath}")
            return True
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """加载模型到当前实例
        
        Args:
            filepath: 模型文件路径
            
        Returns:
            是否加载成功
        """
        try:
            with open(filepath, 'rb') as f:
                loaded_model = pickle.load(f)
            
            # 将加载的模型属性复制到当前实例
            for attr_name, attr_value in loaded_model.__dict__.items():
                setattr(self, attr_name, attr_value)
            
            logger.info("模型已从 %s 加载", filepath)
            return True
        except Exception as e:
            logger.error("加载模型失败: %s", e)
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息
        
        Returns:
            模型信息字典
        """
        return {
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'model_params': self.model_params
        }
