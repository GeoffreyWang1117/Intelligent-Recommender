"""
Teacher接口基类定义
所有Teacher必须实现的统一接口
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
import time
import logging
from dataclasses import dataclass


@dataclass
class UserProfile:
    """用户画像数据结构"""
    user_id: int
    rating_history: List[Tuple[int, float]]  # [(item_id, rating), ...]
    preferred_genres: List[str] = None
    avg_rating: float = 0.0
    num_ratings: int = 0
    
    def __post_init__(self):
        if self.rating_history:
            self.num_ratings = len(self.rating_history)
            self.avg_rating = sum(rating for _, rating in self.rating_history) / self.num_ratings


@dataclass
class ItemProfile:
    """物品画像数据结构"""
    item_id: int
    title: str = ""
    genres: List[str] = None
    year: Optional[int] = None
    features: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.genres is None:
            self.genres = []
        if self.features is None:
            self.features = {}


@dataclass
class RecommendationResult:
    """推荐结果数据结构"""
    user_id: int
    item_recommendations: List[Tuple[int, float]]  # [(item_id, score), ...]
    confidence: float = 0.0
    reasoning: str = ""
    teacher_name: str = ""
    inference_time: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseTeacher(ABC):
    """Teacher基类 - 所有Teacher必须继承此类"""
    
    def __init__(self, teacher_name: str, config: Dict[str, Any] = None):
        self.teacher_name = teacher_name
        self.config = config or {}
        self.logger = logging.getLogger(f"Teacher.{teacher_name}")
        self.is_initialized = False
        self.model = None
        
        # 性能统计
        self.total_inference_time = 0.0
        self.total_inference_count = 0
        self.error_count = 0
    
    @abstractmethod
    def initialize(self) -> bool:
        """初始化Teacher模型"""
        pass
    
    @abstractmethod
    def predict(self, user_profile: UserProfile, candidate_items: List[ItemProfile], 
                num_recommendations: int = 10) -> RecommendationResult:
        """生成推荐结果"""
        pass
    
    @abstractmethod
    def get_user_embedding(self, user_profile: UserProfile) -> List[float]:
        """获取用户嵌入向量"""
        pass
    
    @abstractmethod
    def get_item_embedding(self, item_profile: ItemProfile) -> List[float]:
        """获取物品嵌入向量"""
        pass
    
    def predict_with_timing(self, user_profile: UserProfile, 
                          candidate_items: List[ItemProfile],
                          num_recommendations: int = 10) -> RecommendationResult:
        """带时间统计的推荐预测"""
        start_time = time.time()
        
        try:
            result = self.predict(user_profile, candidate_items, num_recommendations)
            result.inference_time = time.time() - start_time
            result.teacher_name = self.teacher_name
            
            # 更新统计信息
            self.total_inference_time += result.inference_time
            self.total_inference_count += 1
            
            return result
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Prediction error: {e}")
            
            # 返回空结果
            return RecommendationResult(
                user_id=user_profile.user_id,
                item_recommendations=[],
                confidence=0.0,
                reasoning=f"Error: {str(e)}",
                teacher_name=self.teacher_name,
                inference_time=time.time() - start_time
            )
    
    def get_average_inference_time(self) -> float:
        """获取平均推理时间"""
        if self.total_inference_count == 0:
            return 0.0
        return self.total_inference_time / self.total_inference_count
    
    def get_error_rate(self) -> float:
        """获取错误率"""
        total_requests = self.total_inference_count + self.error_count
        if total_requests == 0:
            return 0.0
        return self.error_count / total_requests
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        return {
            "teacher_name": self.teacher_name,
            "total_inferences": self.total_inference_count,
            "total_errors": self.error_count,
            "average_inference_time": self.get_average_inference_time(),
            "error_rate": self.get_error_rate(),
            "is_initialized": self.is_initialized
        }
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self.total_inference_time = 0.0
        self.total_inference_count = 0
        self.error_count = 0
    
    def validate_inputs(self, user_profile: UserProfile, 
                       candidate_items: List[ItemProfile]) -> bool:
        """验证输入参数"""
        if not isinstance(user_profile, UserProfile):
            raise ValueError("user_profile must be UserProfile instance")
        
        if not isinstance(candidate_items, list):
            raise ValueError("candidate_items must be a list")
        
        if len(candidate_items) == 0:
            raise ValueError("candidate_items cannot be empty")
        
        return True
    
    def __str__(self) -> str:
        return f"Teacher({self.teacher_name})"
    
    def __repr__(self) -> str:
        return f"Teacher(name='{self.teacher_name}', initialized={self.is_initialized})"


if __name__ == "__main__":
    # 测试数据结构
    user = UserProfile(
        user_id=1,
        rating_history=[(1, 4.5), (2, 3.0), (3, 5.0)]
    )
    
    item = ItemProfile(
        item_id=10,
        title="Test Movie",
        genres=["Action", "Drama"],
        year=2023
    )
    
    print(f"User: {user}")
    print(f"Item: {item}")
    print(f"User avg rating: {user.avg_rating}")
    print(f"User num ratings: {user.num_ratings}")
