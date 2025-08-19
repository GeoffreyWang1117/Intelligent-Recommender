import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class ObjectiveFunction(ABC):
    """目标函数抽象基类"""
    
    @abstractmethod
    def calculate(self, recommendations: List[Dict], user_context: Dict) -> float:
        """计算目标函数值"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """目标函数名称"""
        pass
    
    @property
    @abstractmethod
    def weight(self) -> float:
        """目标函数权重"""
        pass

class AccuracyObjective(ObjectiveFunction):
    """准确性目标函数"""
    
    def __init__(self, weight: float = 1.0):
        self._weight = weight
    
    @property
    def name(self) -> str:
        return "accuracy"
    
    @property
    def weight(self) -> float:
        return self._weight
    
    def calculate(self, recommendations: List[Dict], user_context: Dict) -> float:
        """基于预测评分计算准确性"""
        if not recommendations:
            return 0.0
        
        # 使用推荐分数作为准确性指标
        scores = [rec.get('score', 0.0) for rec in recommendations]
        return np.mean(scores) if scores else 0.0

class DiversityObjective(ObjectiveFunction):
    """多样性目标函数"""
    
    def __init__(self, weight: float = 1.0, diversity_type: str = 'genre'):
        self._weight = weight
        self.diversity_type = diversity_type
    
    @property
    def name(self) -> str:
        return "diversity"
    
    @property
    def weight(self) -> float:
        return self._weight
    
    def calculate(self, recommendations: List[Dict], user_context: Dict) -> float:
        """计算推荐列表的多样性"""
        if len(recommendations) <= 1:
            return 0.0
        
        if self.diversity_type == 'genre':
            return self._calculate_genre_diversity(recommendations)
        elif self.diversity_type == 'embedding':
            return self._calculate_embedding_diversity(recommendations)
        else:
            return 0.0
    
    def _calculate_genre_diversity(self, recommendations: List[Dict]) -> float:
        """计算类型多样性"""
        all_genres = set()
        for rec in recommendations:
            genres = rec.get('genres', [])
            all_genres.update(genres)
        
        # 多样性 = 不同类型数量 / 总推荐数量
        return len(all_genres) / len(recommendations) if recommendations else 0.0
    
    def _calculate_embedding_diversity(self, recommendations: List[Dict]) -> float:
        """基于嵌入向量计算多样性"""
        embeddings = []
        for rec in recommendations:
            embedding = rec.get('embedding')
            if embedding is not None:
                embeddings.append(embedding)
        
        if len(embeddings) <= 1:
            return 0.0
        
        # 计算平均余弦距离
        embeddings = np.array(embeddings)
        similarities = []
        
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                similarities.append(sim)
        
        # 多样性 = 1 - 平均相似度
        return 1.0 - np.mean(similarities) if similarities else 0.0

class NoveltyObjective(ObjectiveFunction):
    """新颖性目标函数"""
    
    def __init__(self, weight: float = 1.0):
        self._weight = weight
    
    @property
    def name(self) -> str:
        return "novelty"
    
    @property
    def weight(self) -> float:
        return self._weight
    
    def calculate(self, recommendations: List[Dict], user_context: Dict) -> float:
        """计算推荐的新颖性"""
        if not recommendations:
            return 0.0
        
        user_history = user_context.get('user_history', [])
        if not user_history:
            return 1.0  # 新用户，所有推荐都是新颖的
        
        # 计算推荐中有多少是用户未见过的
        history_items = set(user_history)
        recommended_items = [rec.get('item_id') for rec in recommendations]
        
        novel_items = [item for item in recommended_items if item not in history_items]
        return len(novel_items) / len(recommended_items) if recommended_items else 0.0

class PopularityObjective(ObjectiveFunction):
    """流行度目标函数"""
    
    def __init__(self, weight: float = 1.0, popularity_data: Dict = None):
        self._weight = weight
        self.popularity_data = popularity_data or {}
    
    @property
    def name(self) -> str:
        return "popularity"
    
    @property
    def weight(self) -> float:
        return self._weight
    
    def calculate(self, recommendations: List[Dict], user_context: Dict) -> float:
        """计算推荐的流行度"""
        if not recommendations:
            return 0.0
        
        popularity_scores = []
        for rec in recommendations:
            item_id = rec.get('item_id')
            popularity = self.popularity_data.get(item_id, 0.0)
            popularity_scores.append(popularity)
        
        return np.mean(popularity_scores) if popularity_scores else 0.0

class FairnessObjective(ObjectiveFunction):
    """公平性目标函数"""
    
    def __init__(self, weight: float = 1.0, fairness_type: str = 'provider'):
        self._weight = weight
        self.fairness_type = fairness_type
    
    @property
    def name(self) -> str:
        return "fairness"
    
    @property
    def weight(self) -> float:
        return self._weight
    
    def calculate(self, recommendations: List[Dict], user_context: Dict) -> float:
        """计算推荐的公平性"""
        if not recommendations:
            return 0.0
        
        if self.fairness_type == 'provider':
            return self._calculate_provider_fairness(recommendations)
        elif self.fairness_type == 'category':
            return self._calculate_category_fairness(recommendations)
        else:
            return 0.0
    
    def _calculate_provider_fairness(self, recommendations: List[Dict]) -> float:
        """计算提供商公平性（推荐分布的均匀度）"""
        providers = [rec.get('provider', 'unknown') for rec in recommendations]
        
        # 计算提供商分布的熵
        from collections import Counter
        provider_counts = Counter(providers)
        total = len(providers)
        
        entropy = 0.0
        for count in provider_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        
        # 归一化熵（最大值为log2(n)，其中n是提供商数量）
        max_entropy = np.log2(len(provider_counts)) if len(provider_counts) > 1 else 1
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _calculate_category_fairness(self, recommendations: List[Dict]) -> float:
        """计算类别公平性"""
        categories = []
        for rec in recommendations:
            genres = rec.get('genres', [])
            categories.extend(genres)
        
        if not categories:
            return 0.0
        
        # 计算类别分布的基尼系数
        from collections import Counter
        category_counts = Counter(categories)
        counts = list(category_counts.values())
        
        if len(counts) <= 1:
            return 1.0
        
        # 基尼系数计算
        counts.sort()
        n = len(counts)
        gini = (2 * sum((i + 1) * count for i, count in enumerate(counts))) / (n * sum(counts)) - (n + 1) / n
        
        return 1 - gini  # 转换为公平性分数（越高越公平）

class MultiObjectiveOptimizer:
    """多目标优化器"""
    
    def __init__(self, objectives: List[ObjectiveFunction]):
        self.objectives = objectives
        self._validate_objectives()
    
    def _validate_objectives(self):
        """验证目标函数配置"""
        if not self.objectives:
            raise ValueError("至少需要一个目标函数")
        
        total_weight = sum(obj.weight for obj in self.objectives)
        if abs(total_weight - 1.0) > 0.001:
            logger.warning(f"目标函数权重总和不为1.0: {total_weight}")
    
    def optimize_recommendations(self, candidate_recommendations: List[List[Dict]], 
                                user_context: Dict) -> Tuple[List[Dict], Dict[str, float]]:
        """
        优化推荐列表
        
        Args:
            candidate_recommendations: 候选推荐列表的列表
            user_context: 用户上下文
            
        Returns:
            最优推荐列表和目标函数值
        """
        if not candidate_recommendations:
            return [], {}
        
        best_recommendations = None
        best_score = -float('inf')
        best_objectives = {}
        
        # 评估每个候选推荐列表
        for recommendations in candidate_recommendations:
            objective_scores = self._evaluate_objectives(recommendations, user_context)
            
            # 计算加权总分
            total_score = sum(
                score * obj.weight 
                for obj, score in zip(self.objectives, objective_scores.values())
            )
            
            if total_score > best_score:
                best_score = total_score
                best_recommendations = recommendations
                best_objectives = objective_scores
        
        return best_recommendations, best_objectives
    
    def _evaluate_objectives(self, recommendations: List[Dict], 
                           user_context: Dict) -> Dict[str, float]:
        """评估所有目标函数"""
        objective_scores = {}
        
        for objective in self.objectives:
            try:
                score = objective.calculate(recommendations, user_context)
                objective_scores[objective.name] = score
            except Exception as e:
                logger.error(f"计算目标函数 {objective.name} 失败: {e}")
                objective_scores[objective.name] = 0.0
        
        return objective_scores
    
    def rerank_recommendations(self, recommendations: List[Dict], 
                             user_context: Dict, top_k: int = None) -> List[Dict]:
        """
        基于多目标优化重排推荐列表
        """
        if not recommendations:
            return []
        
        if top_k is None:
            top_k = len(recommendations)
        
        # 为每个推荐计算综合分数
        scored_recommendations = []
        
        for rec in recommendations:
            # 为单个推荐创建列表进行评估
            single_rec_list = [rec]
            objective_scores = self._evaluate_objectives(single_rec_list, user_context)
            
            # 计算综合分数
            total_score = sum(
                score * obj.weight 
                for obj, score in zip(self.objectives, objective_scores.values())
            )
            
            rec_with_score = rec.copy()
            rec_with_score['multi_objective_score'] = total_score
            rec_with_score['objective_breakdown'] = objective_scores
            
            scored_recommendations.append(rec_with_score)
        
        # 按综合分数排序
        scored_recommendations.sort(key=lambda x: x['multi_objective_score'], reverse=True)
        
        return scored_recommendations[:top_k]

class DiversityReranker:
    """多样性重排器"""
    
    def __init__(self, diversity_weight: float = 0.3):
        self.diversity_weight = diversity_weight
    
    def rerank_with_diversity(self, recommendations: List[Dict], 
                            top_k: int = None) -> List[Dict]:
        """
        使用贪心算法进行多样性重排
        """
        if not recommendations or len(recommendations) <= 1:
            return recommendations
        
        if top_k is None:
            top_k = len(recommendations)
        
        # 初始化
        selected = []
        remaining = recommendations.copy()
        
        # 选择第一个（评分最高的）
        if remaining:
            first_item = max(remaining, key=lambda x: x.get('score', 0))
            selected.append(first_item)
            remaining.remove(first_item)
        
        # 贪心选择剩余物品
        while len(selected) < top_k and remaining:
            best_item = None
            best_score = -float('inf')
            
            for item in remaining:
                # 计算与已选择物品的多样性
                diversity_score = self._calculate_diversity_with_selected(item, selected)
                
                # 综合评分 = 原始评分 * (1 - diversity_weight) + 多样性评分 * diversity_weight
                original_score = item.get('score', 0)
                combined_score = (original_score * (1 - self.diversity_weight) + 
                                diversity_score * self.diversity_weight)
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_item = item
            
            if best_item:
                selected.append(best_item)
                remaining.remove(best_item)
        
        return selected
    
    def _calculate_diversity_with_selected(self, item: Dict, selected: List[Dict]) -> float:
        """计算物品与已选择物品的多样性"""
        if not selected:
            return 1.0
        
        # 基于类型的多样性
        item_genres = set(item.get('genres', []))
        
        min_similarity = float('inf')
        for selected_item in selected:
            selected_genres = set(selected_item.get('genres', []))
            
            # 计算Jaccard相似度
            intersection = len(item_genres & selected_genres)
            union = len(item_genres | selected_genres)
            
            similarity = intersection / union if union > 0 else 0
            min_similarity = min(min_similarity, similarity)
        
        # 多样性 = 1 - 最小相似度
        return 1.0 - min_similarity if min_similarity != float('inf') else 1.0
