import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

class RecommendationExplainer:
    """推荐解释性分析器"""
    
    def __init__(self, lightfm_model, item_metadata: Dict = None, 
                 user_metadata: Dict = None):
        self.lightfm_model = lightfm_model
        self.item_metadata = item_metadata or {}
        self.user_metadata = user_metadata or {}
    
    def explain_recommendation(self, user_id: int, item_id: int, 
                             explanation_types: List[str] = None) -> Dict[str, Any]:
        """生成推荐解释"""
        if explanation_types is None:
            explanation_types = ['collaborative', 'content', 'popularity', 'diversity']
        
        explanations = {}
        
        try:
            # 协同过滤解释
            if 'collaborative' in explanation_types:
                explanations['collaborative'] = self._explain_collaborative_filtering(
                    user_id, item_id
                )
            
            # 内容相似性解释
            if 'content' in explanation_types:
                explanations['content'] = self._explain_content_similarity(
                    user_id, item_id
                )
            
            # 流行度解释
            if 'popularity' in explanation_types:
                explanations['popularity'] = self._explain_popularity(item_id)
            
            # 多样性解释
            if 'diversity' in explanation_types:
                explanations['diversity'] = self._explain_diversity(user_id, item_id)
            
            # 生成综合解释文本
            explanations['summary'] = self._generate_explanation_summary(
                user_id, item_id, explanations
            )
            
        except Exception as e:
            logger.error(f"生成推荐解释失败: {e}")
            explanations['error'] = str(e)
        
        return explanations
    
    def _explain_collaborative_filtering(self, user_id: int, item_id: int) -> Dict[str, Any]:
        """协同过滤解释"""
        try:
            # 获取用户嵌入向量
            user_embeddings = self.lightfm_model.get_user_embeddings([user_id])
            if len(user_embeddings) == 0:
                return {'type': 'collaborative', 'explanation': '用户数据不足'}
            
            user_embedding = user_embeddings[0]
            
            # 获取物品嵌入向量
            item_embeddings = self.lightfm_model.get_item_embeddings([item_id])
            if len(item_embeddings) == 0:
                return {'type': 'collaborative', 'explanation': '物品数据不足'}
            
            item_embedding = item_embeddings[0]
            
            # 计算相似度
            similarity = np.dot(user_embedding, item_embedding) / (
                np.linalg.norm(user_embedding) * np.linalg.norm(item_embedding)
            )
            
            # 找到相似用户
            similar_users = self._find_similar_users(user_id, top_k=5)
            
            # 找到相似物品
            similar_items = self._find_similar_items_for_user(user_id, item_id, top_k=5)
            
            return {
                'type': 'collaborative',
                'similarity_score': float(similarity),
                'similar_users': similar_users,
                'similar_items': similar_items,
                'explanation': f'基于与您兴趣相似的{len(similar_users)}位用户的喜好推荐'
            }
            
        except Exception as e:
            logger.error(f"协同过滤解释失败: {e}")
            return {'type': 'collaborative', 'error': str(e)}
    
    def _explain_content_similarity(self, user_id: int, item_id: int) -> Dict[str, Any]:
        """内容相似性解释"""
        try:
            # 获取物品元数据
            item_info = self.item_metadata.get(item_id, {})
            if not item_info:
                return {'type': 'content', 'explanation': '物品信息不足'}
            
            # 获取用户历史偏好
            user_preferences = self._analyze_user_preferences(user_id)
            
            # 分析内容匹配度
            content_matches = []
            
            # 类型匹配
            item_genres = item_info.get('genres', [])
            preferred_genres = user_preferences.get('preferred_genres', [])
            
            matching_genres = set(item_genres) & set(preferred_genres)
            if matching_genres:
                content_matches.append({
                    'feature': 'genres',
                    'matches': list(matching_genres),
                    'explanation': f'符合您喜欢的类型: {", ".join(matching_genres)}'
                })
            
            return {
                'type': 'content',
                'item_features': item_info,
                'user_preferences': user_preferences,
                'content_matches': content_matches,
                'explanation': self._format_content_explanation(content_matches)
            }
            
        except Exception as e:
            logger.error(f"内容相似性解释失败: {e}")
            return {'type': 'content', 'error': str(e)}
    
    def _explain_popularity(self, item_id: int) -> Dict[str, Any]:
        """流行度解释"""
        try:
            # 这里需要从数据库或缓存获取物品流行度数据
            # 当前返回模拟数据
            popularity_score = np.random.random()  # 模拟流行度分数
            
            if popularity_score > 0.8:
                popularity_level = 'very_high'
                explanation = '这是一部非常热门的作品，很多用户都喜欢'
            elif popularity_score > 0.6:
                popularity_level = 'high'
                explanation = '这是一部比较热门的作品'
            elif popularity_score > 0.4:
                popularity_level = 'medium'
                explanation = '这是一部中等热度的作品'
            else:
                popularity_level = 'low'
                explanation = '这是一部小众作品，可能正符合您的独特品味'
            
            return {
                'type': 'popularity',
                'popularity_score': float(popularity_score),
                'popularity_level': popularity_level,
                'explanation': explanation
            }
            
        except Exception as e:
            logger.error(f"流行度解释失败: {e}")
            return {'type': 'popularity', 'error': str(e)}
    
    def _explain_diversity(self, user_id: int, item_id: int) -> Dict[str, Any]:
        """多样性解释"""
        try:
            # 获取用户最近的推荐历史
            recent_recommendations = self._get_user_recent_recommendations(user_id)
            
            # 分析当前推荐与历史推荐的多样性
            item_info = self.item_metadata.get(item_id, {})
            item_genres = set(item_info.get('genres', []))
            
            # 计算类型多样性
            historical_genres = set()
            for rec_item_id in recent_recommendations:
                rec_item_info = self.item_metadata.get(rec_item_id, {})
                historical_genres.update(rec_item_info.get('genres', []))
            
            new_genres = item_genres - historical_genres
            diversity_score = len(new_genres) / len(item_genres) if item_genres else 0
            
            explanation = ''
            if diversity_score > 0.5:
                explanation = '为您推荐不同类型的内容，丰富您的观影体验'
            elif diversity_score > 0:
                explanation = '在您熟悉的类型基础上，尝试一些新的元素'
            else:
                explanation = '基于您一贯的喜好推荐'
            
            return {
                'type': 'diversity',
                'diversity_score': float(diversity_score),
                'new_genres': list(new_genres),
                'explanation': explanation
            }
            
        except Exception as e:
            logger.error(f"多样性解释失败: {e}")
            return {'type': 'diversity', 'error': str(e)}
    
    def _find_similar_users(self, user_id: int, top_k: int = 5) -> List[Dict]:
        """找到相似用户"""
        try:
            # 获取用户嵌入
            user_embeddings = self.lightfm_model.get_user_embeddings()
            if user_id not in self.lightfm_model.user_mapping:
                return []
            
            user_idx = self.lightfm_model.user_mapping[user_id]
            user_embedding = user_embeddings[user_idx]
            
            # 计算与所有用户的相似度
            similarities = np.dot(user_embeddings, user_embedding)
            
            # 排序并获取top_k
            similar_indices = np.argsort(similarities)[::-1][1:top_k+1]  # 排除自己
            
            similar_users = []
            for idx in similar_indices:
                similar_user_id = self.lightfm_model.reverse_user_mapping.get(idx)
                if similar_user_id:
                    similar_users.append({
                        'user_id': similar_user_id,
                        'similarity': float(similarities[idx])
                    })
            
            return similar_users
            
        except Exception as e:
            logger.error(f"查找相似用户失败: {e}")
            return []
    
    def _find_similar_items_for_user(self, user_id: int, item_id: int, 
                                   top_k: int = 5) -> List[Dict]:
        """为用户找到相似物品"""
        # 这里应该获取用户评价过且评分较高的物品
        # 当前返回模拟数据
        return []
    
    def _analyze_user_preferences(self, user_id: int) -> Dict[str, Any]:
        """分析用户偏好"""
        try:
            # 这里应该分析用户的历史行为数据
            # 当前返回模拟偏好数据
            return {
                'preferred_genres': ['Action', 'Comedy', 'Drama'],
                'avg_rating': 4.2,
                'total_ratings': 85,
                'favorite_actors': ['Tom Hanks', 'Leonardo DiCaprio'],
                'preferred_decades': ['1990s', '2000s']
            }
            
        except Exception as e:
            logger.error(f"分析用户偏好失败: {e}")
            return {}
    
    def _format_content_explanation(self, content_matches: List[Dict]) -> str:
        """格式化内容解释"""
        if not content_matches:
            return '根据内容特征为您推荐'
        
        explanations = []
        for match in content_matches:
            explanations.append(match['explanation'])
        
        return '；'.join(explanations)
    
    def _get_user_recent_recommendations(self, user_id: int) -> List[int]:
        """获取用户最近的推荐历史"""
        # 这里应该从数据库或缓存获取
        # 当前返回模拟数据
        return [1, 2, 3, 4, 5]
    
    def _generate_explanation_summary(self, user_id: int, item_id: int, 
                                    explanations: Dict) -> str:
        """生成综合解释摘要"""
        try:
            item_info = self.item_metadata.get(item_id, {})
            item_title = item_info.get('title', f'物品{item_id}')
            
            summary_parts = []
            
            # 添加协同过滤解释
            if 'collaborative' in explanations:
                collab = explanations['collaborative']
                if 'similar_users' in collab and collab['similar_users']:
                    summary_parts.append(f"有{len(collab['similar_users'])}位和您兴趣相似的用户也喜欢这部作品")
            
            # 添加内容解释
            if 'content' in explanations:
                content = explanations['content']
                if 'content_matches' in content and content['content_matches']:
                    match_explanations = [match['explanation'] for match in content['content_matches']]
                    summary_parts.extend(match_explanations)
            
            # 添加流行度解释
            if 'popularity' in explanations:
                popularity = explanations['popularity']
                if 'explanation' in popularity:
                    summary_parts.append(popularity['explanation'])
            
            # 添加多样性解释
            if 'diversity' in explanations:
                diversity = explanations['diversity']
                if 'explanation' in diversity:
                    summary_parts.append(diversity['explanation'])
            
            if summary_parts:
                return f'推荐《{item_title}》给您，因为：' + '；'.join(summary_parts) + '。'
            else:
                return f'根据您的个人喜好，推荐《{item_title}》给您。'
                
        except Exception as e:
            logger.error(f"生成解释摘要失败: {e}")
            return '为您智能推荐此内容。'

class ExplanationTemplates:
    """解释模板"""
    
    COLLABORATIVE_TEMPLATES = [
        "喜欢{similar_items}的用户也喜欢这个",
        "基于{user_count}位相似用户的喜好推荐",
        "您和喜欢这个的用户有相似的兴趣爱好"
    ]
    
    CONTENT_TEMPLATES = [
        "因为您喜欢{genre}类型的内容",
        "与您之前观看的{similar_item}风格相似",
        "符合您对{feature}的偏好"
    ]
    
    POPULARITY_TEMPLATES = [
        "这是最近的热门内容",
        "很多用户都在观看这个",
        "口碑很好的优质内容"
    ]
    
    DIVERSITY_TEMPLATES = [
        "为您尝试新的{genre}类型",
        "丰富您的观看体验",
        "发现新的兴趣点"
    ]

class ExplanationMetrics:
    """解释性评估指标"""
    
    @staticmethod
    def calculate_explanation_quality(explanations: Dict) -> Dict[str, float]:
        """计算解释质量指标"""
        metrics = {}
        
        # 完整性：解释覆盖的维度数量
        total_dimensions = 4  # collaborative, content, popularity, diversity
        covered_dimensions = len([k for k in explanations.keys() 
                                if k in ['collaborative', 'content', 'popularity', 'diversity']])
        metrics['completeness'] = covered_dimensions / total_dimensions
        
        # 准确性：基于可用信息的准确度
        accuracy_scores = []
        for explanation_type, explanation_data in explanations.items():
            if isinstance(explanation_data, dict) and 'error' not in explanation_data:
                accuracy_scores.append(1.0)
            else:
                accuracy_scores.append(0.0)
        
        metrics['accuracy'] = np.mean(accuracy_scores) if accuracy_scores else 0.0
        
        # 可理解性：解释文本的长度和复杂度
        summary = explanations.get('summary', '')
        if summary:
            # 简单的可理解性评分：基于句子长度
            word_count = len(summary.split())
            metrics['understandability'] = max(0, 1 - (word_count - 20) / 50)  # 20词最佳，超过减分
        else:
            metrics['understandability'] = 0.0
        
        # 综合评分
        metrics['overall_quality'] = (
            metrics['completeness'] * 0.3 +
            metrics['accuracy'] * 0.4 +
            metrics['understandability'] * 0.3
        )
        
        return metrics
