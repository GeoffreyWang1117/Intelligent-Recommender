"""
统一Teacher接口定义

为多种Teacher类型提供统一的接口，支持：
- Ensemble Teacher (SVD + xDeepFM + AutoInt)
- LLM Teacher (Llama3.1)
- 标准化的预测和特征提取接口
"""

from abc import ABC, abstractmethod
import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Union

class BaseTeacher(ABC):
    """统一Teacher接口基类"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    @abstractmethod
    def predict(self, user_id: int, item_id: int) -> float:
        """标准预测接口"""
        pass
    
    @abstractmethod
    def get_embeddings(self, user_id: int, item_id: int) -> torch.Tensor:
        """获取嵌入表示"""
        pass
    
    @abstractmethod
    def get_confidence(self, user_id: int, item_id: int) -> float:
        """获取预测置信度"""
        pass
    
    @abstractmethod
    def batch_predict(self, user_ids: List[int], item_ids: List[int]) -> np.ndarray:
        """批量预测"""
        pass

class TeacherType:
    """Teacher类型枚举"""
    ENSEMBLE = "ensemble"
    LLM = "llm"
    TRADITIONAL = "traditional"

class TeacherCapability:
    """Teacher能力标识"""
    SEMANTIC_UNDERSTANDING = "semantic"
    COLLABORATIVE_FILTERING = "cf"
    FEATURE_INTERACTION = "interaction"
    ATTENTION_MECHANISM = "attention"
    FAST_INFERENCE = "fast"
    HIGH_ACCURACY = "accurate"
