import logging
from typing import List, Dict, Any, Optional, Tuple
from models.svd_model import SVDRecommender as LightFMRecommender
from models.faiss_index import FAISSIndex
from services.cache import CacheService
from services.ab_testing import ABTestManager
from services.explainability import RecommendationExplainer
from services.multi_objective import MultiObjectiveOptimizer, DiversityReranker

logger = logging.getLogger(__name__)

class RecommendationService:
    """推荐服务核心类"""
    
    def __init__(self, lightfm_model: LightFMRecommender, 
                 faiss_index: FAISSIndex, cache_service: CacheService,
                 item_metadata: Optional[Dict] = None,
                 ab_test_manager: Optional[ABTestManager] = None,
                 explainer: Optional[RecommendationExplainer] = None,
                 multi_objective_optimizer: Optional[MultiObjectiveOptimizer] = None):
        self.lightfm_model = lightfm_model
        self.faiss_index = faiss_index
        self.cache_service = cache_service
        self.item_metadata = item_metadata or {}
        self.ab_test_manager = ab_test_manager
        self.explainer = explainer
        self.multi_objective_optimizer = multi_objective_optimizer
        self.diversity_reranker = DiversityReranker(diversity_weight=0.3)
    
    def get_recommendations(self, user_id: int, top_k: int = 10, 
                          use_cache: bool = True, 
                          enable_ab_test: bool = True,
                          enable_explanation: bool = False,
                          enable_multi_objective: bool = False,
                          diversify: bool = False) -> List[Dict[str, Any]]:
        """获取用户推荐列表（增强版）"""
        
        # A/B测试：获取用户实验变体
        experiment_variant = None
        if enable_ab_test and self.ab_test_manager:
            experiment_variant = self.ab_test_manager.get_user_variant(
                user_id, "recommendation_algorithm_test"
            )
        
        # 检查缓存
        if use_cache:
            cache_key = self.cache_service.generate_recommendation_key(user_id, top_k)
            if experiment_variant:
                cache_key += f":variant:{experiment_variant}"
            
            cached_result = self.cache_service.get(cache_key)
            if cached_result is not None:
                logger.info(f"从缓存获取用户 {user_id} 的推荐")
                return self._format_recommendations(cached_result['recommendations'], cached=True)
        
        try:
            # 根据A/B测试变体调整推荐策略
            recommendations = self._generate_base_recommendations(
                user_id, top_k, experiment_variant
            )
            
            if not recommendations:
                # 如果基础推荐为空，使用冷启动策略
                logger.info(f"用户 {user_id} 基础推荐为空，使用冷启动策略")
                recommendations = self._cold_start_recommendations(user_id, top_k)
            
            # 格式化推荐结果
            formatted_recs = self._format_recommendations(recommendations)
            
            # 多目标优化
            if enable_multi_objective and self.multi_objective_optimizer:
                user_context = self._build_user_context(user_id)
                formatted_recs = self.multi_objective_optimizer.rerank_recommendations(
                    formatted_recs, user_context, top_k
                )
            
            # 多样性重排
            if diversify:
                formatted_recs = self.diversity_reranker.rerank_with_diversity(
                    formatted_recs, top_k
                )
            
            # 添加推荐解释
            if enable_explanation and self.explainer:
                for rec in formatted_recs:
                    explanation = self.explainer.explain_recommendation(
                        user_id, rec['item_id']
                    )
                    rec['explanation'] = explanation
            
            # 记录A/B测试事件
            if enable_ab_test and self.ab_test_manager and experiment_variant:
                self.ab_test_manager.log_event(
                    user_id, "recommendation_algorithm_test", "impression",
                    {"variant": experiment_variant, "recommendation_count": len(formatted_recs)}
                )
            
            # 缓存结果
            if use_cache and recommendations:
                cache_data = {
                    'recommendations': recommendations,
                    'user_id': user_id,
                    'top_k': top_k,
                    'variant': experiment_variant
                }
                cache_key = self.cache_service.generate_recommendation_key(user_id, top_k)
                if experiment_variant:
                    cache_key += f":variant:{experiment_variant}"
                self.cache_service.set(cache_key, cache_data)
            
            return formatted_recs
            
        except Exception as e:
            logger.error(f"获取推荐失败 user_id={user_id}: {e}")
            return []
    
    def _generate_base_recommendations(self, user_id: int, top_k: int, 
                                     experiment_variant: Optional[str] = None) -> List[Tuple[int, float]]:
        """根据实验变体生成基础推荐"""
        try:
            if experiment_variant == "lightfm_enhanced":
                # 使用增强参数的LightFM模型
                # 这里可以调整模型参数或使用不同的模型
                return self.lightfm_model.predict(user_id, top_k=top_k * 2)[:top_k]
            else:
                # 默认LightFM模型
                return self.lightfm_model.predict(user_id, top_k=top_k)
        except Exception as e:
            logger.error(f"生成基础推荐失败: {e}")
            return []
    
    def _build_user_context(self, user_id: int) -> Dict[str, Any]:
        """构建用户上下文信息"""
        try:
            # 从缓存或数据库获取用户历史数据
            user_history_key = f"user_history:{user_id}"
            user_history = self.cache_service.get(user_history_key) or []
            
            # 分析用户偏好
            preferred_genres = self._extract_user_preferred_genres(user_history)
            
            return {
                'user_id': user_id,
                'user_history': user_history,
                'preferred_genres': preferred_genres,
                'total_interactions': len(user_history)
            }
        except Exception as e:
            logger.error(f"构建用户上下文失败: {e}")
            return {'user_id': user_id, 'user_history': []}
    
    def _extract_user_preferred_genres(self, user_history: List[int]) -> List[str]:
        """从用户历史中提取偏好类型"""
        genre_counts = {}
        for item_id in user_history:
            item_info = self.item_metadata.get(item_id, {})
            genres = item_info.get('genres', [])
            for genre in genres:
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        # 返回最喜欢的前5个类型
        sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
        return [genre for genre, _ in sorted_genres[:5]]
    
    def log_recommendation_interaction(self, user_id: int, item_id: int, 
                                     interaction_type: str, experiment_name: str = None):
        """记录推荐交互事件（用于A/B测试）"""
        if self.ab_test_manager and experiment_name:
            self.ab_test_manager.log_event(
                user_id, experiment_name, interaction_type,
                {'item_id': item_id}
            )
    
    def get_recommendation_explanation(self, user_id: int, item_id: int) -> Dict[str, Any]:
        """获取单个推荐的解释"""
        if not self.explainer:
            return {'error': '解释器未初始化'}
        
        return self.explainer.explain_recommendation(user_id, item_id)
    
    def get_ab_test_results(self, experiment_name: str) -> Dict[str, Any]:
        """获取A/B测试结果"""
        if not self.ab_test_manager:
            return {'error': 'A/B测试管理器未初始化'}
        
        return self.ab_test_manager.get_experiment_results(experiment_name)
    
    def get_similar_items(self, item_id: int, top_k: int = 10,
                         use_cache: bool = True) -> List[Dict[str, Any]]:
        """获取相似物品列表"""
        
        # 检查缓存
        if use_cache:
            cache_key = self.cache_service.generate_similar_key(item_id, top_k)
            cached_result = self.cache_service.get(cache_key)
            if cached_result is not None:
                logger.info(f"从缓存获取物品 {item_id} 的相似推荐")
                return self._format_similar_items(cached_result['similar_items'], cached=True)
        
        try:
            # 从FAISS索引获取物品向量
            item_vector = self.faiss_index.get_vector_by_id(item_id)
            if item_vector is None:
                logger.warning(f"物品 {item_id} 的向量不存在")
                return []
            
            # 搜索相似物品
            similar_items = self.faiss_index.search(item_vector, top_k + 1)  # +1 因为会包含自己
            
            # 过滤掉自己
            similar_items = [item for item in similar_items if item['item_id'] != item_id][:top_k]
            
            # 格式化结果
            formatted_items = self._format_similar_items(similar_items)
            
            # 缓存结果
            if use_cache and similar_items:
                cache_data = {
                    'similar_items': similar_items,
                    'item_id': item_id,
                    'top_k': top_k
                }
                self.cache_service.set(cache_key, cache_data)
            
            return formatted_items
            
        except Exception as e:
            logger.error(f"获取相似物品失败 item_id={item_id}: {e}")
            return []
    
    def update_user_rating(self, user_id: int, item_id: int, rating: float):
        """更新用户评分并清除相关缓存"""
        try:
            # 这里可以实现实时更新逻辑，比如更新数据库
            # 目前先清除缓存，强制重新计算推荐
            
            # 清除用户推荐缓存
            pattern = f"rec:user:{user_id}:*"
            cache_keys = self.cache_service.keys(pattern)
            for key in cache_keys:
                self.cache_service.delete(key)
            
            logger.info(f"用户 {user_id} 对物品 {item_id} 的评分已更新为 {rating}")
            
            # 这里可以添加实时模型更新逻辑
            # 比如增量更新LightFM模型或重新训练
            
        except Exception as e:
            logger.error(f"更新用户评分失败: {e}")
            raise
    
    def _cold_start_recommendations(self, user_id: int, top_k: int) -> List[Tuple[int, float]]:
        """冷启动推荐策略"""
        try:
            # 策略1: 推荐热门物品
            popular_items = self._get_popular_items(top_k)
            if popular_items:
                return popular_items
            
            # 策略2: 随机推荐
            return self._get_random_items(top_k)
            
        except Exception as e:
            logger.error(f"冷启动推荐失败: {e}")
            return []
    
    def _get_popular_items(self, top_k: int) -> List[Tuple[int, float]]:
        """获取热门物品"""
        # 这里可以从数据库或缓存中获取热门物品
        # 目前返回空列表，实际项目中需要实现
        return []
    
    def _get_random_items(self, top_k: int) -> List[Tuple[int, float]]:
        """获取随机物品"""
        # 这里可以随机选择一些物品
        # 目前返回空列表，实际项目中需要实现
        return []
    
    def _format_recommendations(self, recommendations: List[Tuple[int, float]], 
                              cached: bool = False) -> List[Dict[str, Any]]:
        """格式化推荐结果"""
        formatted = []
        for item_id, score in recommendations:
            item_info = {
                'item_id': item_id,
                'score': float(score),
                'cached': cached
            }
            
            # 添加元数据信息
            if item_id in self.item_metadata:
                item_info.update(self.item_metadata[item_id])
            
            formatted.append(item_info)
        
        return formatted
    
    def _format_similar_items(self, similar_items: List[Dict[str, Any]], 
                            cached: bool = False) -> List[Dict[str, Any]]:
        """格式化相似物品结果"""
        formatted = []
        for item in similar_items:
            item_info = {
                'item_id': item['item_id'],
                'similarity': item['similarity'],
                'cached': cached
            }
            
            # 添加元数据信息
            if item['item_id'] in self.item_metadata:
                item_info.update(self.item_metadata[item['item_id']])
            
            formatted.append(item_info)
        
        return formatted
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = {
            'lightfm_loaded': self.lightfm_model.model is not None,
            'faiss_loaded': self.faiss_index.index is not None,
            'cache_connected': self.cache_service.is_connected(),
            'item_metadata_count': len(self.item_metadata)
        }
        
        # 添加FAISS索引信息
        if self.faiss_index.index is not None:
            faiss_info = self.faiss_index.get_index_info()
            if faiss_info:
                info['faiss_info'] = faiss_info
        
        return info
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            'status': 'healthy' if self._is_healthy() else 'unhealthy',
            'model_loaded': self.lightfm_model.model is not None,
            'faiss_loaded': self.faiss_index.index is not None,
            'redis_connected': self.cache_service.is_connected(),
            'item_metadata_loaded': len(self.item_metadata) > 0
        }
    
    def _is_healthy(self) -> bool:
        """检查服务是否健康"""
        return (
            self.lightfm_model.model is not None and
            self.faiss_index.index is not None and
            self.cache_service.is_connected()
        )
