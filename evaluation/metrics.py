"""
推荐系统评价指标模块
提供各种推荐系统常用的评价指标，包括：
- Recall@K, Precision@K
- NDCG@K (Normalized Discounted Cumulative Gain)
- MAP (Mean Average Precision)
- MRR (Mean Reciprocal Rank)
- Hit Rate
- Coverage
- Diversity
- Novelty
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Set, Tuple, Union
from collections import defaultdict
import math
import logging

logger = logging.getLogger(__name__)


class RecommendationMetrics:
    """推荐系统评价指标计算器"""
    
    def __init__(self):
        self.item_popularity = {}
        self.total_items = 0
        
    def prepare_data(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        """准备评价所需的数据"""
        # 计算物品流行度（用于计算novelty）
        item_counts = train_data['item_id'].value_counts()
        total_interactions = len(train_data)
        self.item_popularity = {item: count/total_interactions 
                               for item, count in item_counts.items()}
        self.total_items = train_data['item_id'].nunique()
        
        # 构建测试集真实标签
        self.test_user_items = defaultdict(set)
        for _, row in test_data.iterrows():
            self.test_user_items[row['user_id']].add(row['item_id'])
    
    def recall_at_k(self, recommendations: Dict[int, List[int]], k: int = 10) -> float:
        """
        计算Recall@K
        
        Args:
            recommendations: {user_id: [item_id1, item_id2, ...]}
            k: Top-K
            
        Returns:
            Recall@K值
        """
        total_recall = 0.0
        user_count = 0
        
        for user_id, rec_items in recommendations.items():
            if user_id not in self.test_user_items:
                continue
                
            true_items = self.test_user_items[user_id]
            rec_items_k = set(rec_items[:k])
            
            if len(true_items) > 0:
                recall = len(rec_items_k & true_items) / len(true_items)
                total_recall += recall
                user_count += 1
        
        return total_recall / user_count if user_count > 0 else 0.0
    
    def precision_at_k(self, recommendations: Dict[int, List[int]], k: int = 10) -> float:
        """
        计算Precision@K
        
        Args:
            recommendations: {user_id: [item_id1, item_id2, ...]}
            k: Top-K
            
        Returns:
            Precision@K值
        """
        total_precision = 0.0
        user_count = 0
        
        for user_id, rec_items in recommendations.items():
            if user_id not in self.test_user_items:
                continue
                
            true_items = self.test_user_items[user_id]
            rec_items_k = set(rec_items[:k])
            
            if len(rec_items_k) > 0:
                precision = len(rec_items_k & true_items) / len(rec_items_k)
                total_precision += precision
                user_count += 1
        
        return total_precision / user_count if user_count > 0 else 0.0
    
    def ndcg_at_k(self, recommendations: Dict[int, List[int]], 
                  relevance_scores: Dict[int, Dict[int, float]] = None, 
                  k: int = 10) -> float:
        """
        计算NDCG@K (Normalized Discounted Cumulative Gain)
        
        Args:
            recommendations: {user_id: [item_id1, item_id2, ...]}
            relevance_scores: {user_id: {item_id: relevance_score}}
            k: Top-K
            
        Returns:
            NDCG@K值
        """
        total_ndcg = 0.0
        user_count = 0
        
        for user_id, rec_items in recommendations.items():
            if user_id not in self.test_user_items:
                continue
                
            true_items = self.test_user_items[user_id]
            rec_items_k = rec_items[:k]
            
            # 计算DCG
            dcg = 0.0
            for i, item_id in enumerate(rec_items_k):
                if item_id in true_items:
                    # 如果有显式的相关性分数，使用它；否则使用二元相关性
                    if relevance_scores and user_id in relevance_scores:
                        rel_score = relevance_scores[user_id].get(item_id, 0)
                    else:
                        rel_score = 1.0 if item_id in true_items else 0.0
                    
                    dcg += rel_score / math.log2(i + 2)
            
            # 计算IDCG (Ideal DCG)
            ideal_items = sorted(true_items)[:k]
            idcg = 0.0
            for i, item_id in enumerate(ideal_items):
                if relevance_scores and user_id in relevance_scores:
                    rel_score = relevance_scores[user_id].get(item_id, 1.0)
                else:
                    rel_score = 1.0
                idcg += rel_score / math.log2(i + 2)
            
            # 计算NDCG
            if idcg > 0:
                ndcg = dcg / idcg
                total_ndcg += ndcg
                user_count += 1
        
        return total_ndcg / user_count if user_count > 0 else 0.0
    
    def mean_average_precision(self, recommendations: Dict[int, List[int]], 
                              k: int = 10) -> float:
        """
        计算MAP (Mean Average Precision)
        
        Args:
            recommendations: {user_id: [item_id1, item_id2, ...]}
            k: Top-K
            
        Returns:
            MAP值
        """
        total_ap = 0.0
        user_count = 0
        
        for user_id, rec_items in recommendations.items():
            if user_id not in self.test_user_items:
                continue
                
            true_items = self.test_user_items[user_id]
            rec_items_k = rec_items[:k]
            
            # 计算Average Precision
            ap = 0.0
            hit_count = 0
            
            for i, item_id in enumerate(rec_items_k):
                if item_id in true_items:
                    hit_count += 1
                    precision_at_i = hit_count / (i + 1)
                    ap += precision_at_i
            
            if len(true_items) > 0:
                ap /= min(len(true_items), k)
                total_ap += ap
                user_count += 1
        
        return total_ap / user_count if user_count > 0 else 0.0
    
    def mean_reciprocal_rank(self, recommendations: Dict[int, List[int]]) -> float:
        """
        计算MRR (Mean Reciprocal Rank)
        
        Args:
            recommendations: {user_id: [item_id1, item_id2, ...]}
            
        Returns:
            MRR值
        """
        total_rr = 0.0
        user_count = 0
        
        for user_id, rec_items in recommendations.items():
            if user_id not in self.test_user_items:
                continue
                
            true_items = self.test_user_items[user_id]
            
            # 找到第一个相关物品的位置
            for i, item_id in enumerate(rec_items):
                if item_id in true_items:
                    total_rr += 1.0 / (i + 1)
                    break
            
            user_count += 1
        
        return total_rr / user_count if user_count > 0 else 0.0
    
    def hit_rate_at_k(self, recommendations: Dict[int, List[int]], k: int = 10) -> float:
        """
        计算Hit Rate@K
        
        Args:
            recommendations: {user_id: [item_id1, item_id2, ...]}
            k: Top-K
            
        Returns:
            Hit Rate@K值
        """
        hit_count = 0
        user_count = 0
        
        for user_id, rec_items in recommendations.items():
            if user_id not in self.test_user_items:
                continue
                
            true_items = self.test_user_items[user_id]
            rec_items_k = set(rec_items[:k])
            
            if len(rec_items_k & true_items) > 0:
                hit_count += 1
            user_count += 1
        
        return hit_count / user_count if user_count > 0 else 0.0
    
    def coverage(self, recommendations: Dict[int, List[int]], k: int = 10) -> float:
        """
        计算推荐覆盖率
        
        Args:
            recommendations: {user_id: [item_id1, item_id2, ...]}
            k: Top-K
            
        Returns:
            覆盖率
        """
        recommended_items = set()
        
        for rec_items in recommendations.values():
            recommended_items.update(rec_items[:k])
        
        return len(recommended_items) / self.total_items if self.total_items > 0 else 0.0
    
    def diversity(self, recommendations: Dict[int, List[int]], 
                  item_features: Dict[int, List] = None, k: int = 10) -> float:
        """
        计算推荐多样性
        
        Args:
            recommendations: {user_id: [item_id1, item_id2, ...]}
            item_features: {item_id: [feature1, feature2, ...]}
            k: Top-K
            
        Returns:
            多样性分数
        """
        if item_features is None:
            # 如果没有特征信息，使用简单的不重复率
            total_diversity = 0.0
            user_count = 0
            
            for rec_items in recommendations.values():
                rec_items_k = rec_items[:k]
                unique_items = len(set(rec_items_k))
                diversity = unique_items / len(rec_items_k) if len(rec_items_k) > 0 else 0.0
                total_diversity += diversity
                user_count += 1
            
            return total_diversity / user_count if user_count > 0 else 0.0
        
        # 基于特征的多样性计算
        def jaccard_distance(features1, features2):
            set1, set2 = set(features1), set(features2)
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            return 1 - (intersection / union) if union > 0 else 0.0
        
        total_diversity = 0.0
        user_count = 0
        
        for rec_items in recommendations.values():
            rec_items_k = rec_items[:k]
            
            if len(rec_items_k) < 2:
                continue
            
            diversity_sum = 0.0
            pair_count = 0
            
            for i in range(len(rec_items_k)):
                for j in range(i + 1, len(rec_items_k)):
                    item1, item2 = rec_items_k[i], rec_items_k[j]
                    if item1 in item_features and item2 in item_features:
                        distance = jaccard_distance(item_features[item1], item_features[item2])
                        diversity_sum += distance
                        pair_count += 1
            
            if pair_count > 0:
                total_diversity += diversity_sum / pair_count
                user_count += 1
        
        return total_diversity / user_count if user_count > 0 else 0.0
    
    def novelty(self, recommendations: Dict[int, List[int]], k: int = 10) -> float:
        """
        计算推荐新颖性
        
        Args:
            recommendations: {user_id: [item_id1, item_id2, ...]}
            k: Top-K
            
        Returns:
            新颖性分数
        """
        total_novelty = 0.0
        user_count = 0
        
        for rec_items in recommendations.values():
            rec_items_k = rec_items[:k]
            
            item_novelty = 0.0
            item_count = 0
            
            for item_id in rec_items_k:
                if item_id in self.item_popularity:
                    # 新颖性 = -log(popularity)
                    novelty = -math.log(self.item_popularity[item_id] + 1e-10)
                    item_novelty += novelty
                    item_count += 1
            
            if item_count > 0:
                total_novelty += item_novelty / item_count
                user_count += 1
        
        return total_novelty / user_count if user_count > 0 else 0.0
    
    def evaluate_all(self, recommendations: Dict[int, List[int]], 
                     relevance_scores: Dict[int, Dict[int, float]] = None,
                     item_features: Dict[int, List] = None,
                     k_values: List[int] = [10, 20, 50]) -> Dict[str, float]:
        """
        计算所有评价指标
        
        Args:
            recommendations: {user_id: [item_id1, item_id2, ...]}
            relevance_scores: {user_id: {item_id: relevance_score}}
            item_features: {item_id: [feature1, feature2, ...]}
            k_values: 要计算的K值列表
            
        Returns:
            包含所有指标的字典
        """
        results = {}
        
        for k in k_values:
            results[f'Recall@{k}'] = self.recall_at_k(recommendations, k)
            results[f'Precision@{k}'] = self.precision_at_k(recommendations, k)
            results[f'NDCG@{k}'] = self.ndcg_at_k(recommendations, relevance_scores, k)
            results[f'MAP@{k}'] = self.mean_average_precision(recommendations, k)
            results[f'HitRate@{k}'] = self.hit_rate_at_k(recommendations, k)
            results[f'Coverage@{k}'] = self.coverage(recommendations, k)
            results[f'Diversity@{k}'] = self.diversity(recommendations, item_features, k)
            results[f'Novelty@{k}'] = self.novelty(recommendations, k)
        
        # 计算MRR（不依赖K值）
        results['MRR'] = self.mean_reciprocal_rank(recommendations)
        
        return results


def format_evaluation_results(results: Dict[str, float], 
                             algorithm_name: str = "Algorithm") -> str:
    """
    格式化评价结果输出
    
    Args:
        results: 评价结果字典
        algorithm_name: 算法名称
        
    Returns:
        格式化的结果字符串
    """
    output = [f"\n=== {algorithm_name} 评价结果 ==="]
    
    # 按指标类型分组
    metrics_groups = {
        'Accuracy': ['Recall', 'Precision', 'NDCG', 'MAP', 'HitRate', 'MRR'],
        'Diversity': ['Coverage', 'Diversity', 'Novelty']
    }
    
    for group_name, metric_types in metrics_groups.items():
        output.append(f"\n{group_name} Metrics:")
        output.append("-" * 30)
        
        for metric_type in metric_types:
            # 找到所有相关指标
            relevant_metrics = {k: v for k, v in results.items() 
                              if metric_type in k}
            
            if relevant_metrics:
                for metric_name, value in relevant_metrics.items():
                    output.append(f"{metric_name:<15}: {value:.4f}")
    
    return "\n".join(output)


# 示例使用
if __name__ == "__main__":
    # 创建示例数据
    train_data = pd.DataFrame({
        'user_id': [1, 1, 2, 2, 3, 3] * 10,
        'item_id': [1, 2, 2, 3, 1, 3] * 10,
        'rating': [5, 4, 3, 5, 4, 2] * 10
    })
    
    test_data = pd.DataFrame({
        'user_id': [1, 2, 3],
        'item_id': [3, 1, 2],
        'rating': [4, 5, 3]
    })
    
    # 模拟推荐结果
    recommendations = {
        1: [3, 1, 2, 4, 5],
        2: [1, 2, 3, 4, 5],
        3: [2, 1, 3, 4, 5]
    }
    
    # 计算评价指标
    metrics = RecommendationMetrics()
    metrics.prepare_data(train_data, test_data)
    results = metrics.evaluate_all(recommendations, k_values=[5, 10])
    
    print(format_evaluation_results(results, "Example Algorithm"))
