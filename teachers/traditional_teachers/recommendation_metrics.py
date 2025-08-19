#!/usr/bin/env python3
"""
推荐系统评价指标模块
Recommendation System Evaluation Metrics

实现标准的推荐系统评价指标：
- 准确性指标: RMSE, MAE, Precision, Recall, F1-Score
- 排序指标: NDCG, MAP, MRR
- 多样性指标: Coverage, Diversity, Novelty
- 鲁棒性指标: Serendipity, Popularity Bias

作者: GitHub Copilot
日期: 2025-08-18
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Any
from collections import defaultdict, Counter
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import sparse
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')

class RecommendationMetrics:
    """推荐系统评价指标计算器"""
    
    def __init__(self):
        self.metrics_results = {}
    
    def calculate_accuracy_metrics(self, true_ratings: List[Tuple[int, int, float]], 
                                 predicted_ratings: List[Tuple[int, int, float]]) -> Dict[str, float]:
        """
        计算准确性指标
        
        Args:
            true_ratings: [(user_id, item_id, rating), ...]
            predicted_ratings: [(user_id, item_id, predicted_rating), ...]
        
        Returns:
            Dict with RMSE, MAE, etc.
        """
        # 创建评分映射
        true_dict = {(user, item): rating for user, item, rating in true_ratings}
        pred_dict = {(user, item): rating for user, item, rating in predicted_ratings}
        
        # 找到交集
        common_pairs = set(true_dict.keys()) & set(pred_dict.keys())
        
        if not common_pairs:
            return {"error": "没有共同的用户-物品对"}
        
        true_values = [true_dict[pair] for pair in common_pairs]
        pred_values = [pred_dict[pair] for pair in common_pairs]
        
        # 计算指标
        rmse = np.sqrt(mean_squared_error(true_values, pred_values))
        mae = mean_absolute_error(true_values, pred_values)
        
        # 相关系数
        correlation = np.corrcoef(true_values, pred_values)[0, 1] if len(true_values) > 1 else 0
        
        return {
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            "Correlation": round(correlation, 4),
            "Coverage": len(common_pairs)
        }
    
    def calculate_ranking_metrics(self, test_users: Dict[int, List[int]], 
                                 recommendations: Dict[int, List[int]], 
                                 relevance_threshold: float = 4.0,
                                 k_values: List[int] = [5, 10, 20]) -> Dict[str, Dict[int, float]]:
        """
        计算排序指标
        
        Args:
            test_users: {user_id: [relevant_item_ids]}
            recommendations: {user_id: [recommended_item_ids]}
            relevance_threshold: 相关性阈值
            k_values: 计算top-k指标的k值列表
        
        Returns:
            Dict with Precision@K, Recall@K, NDCG@K, MAP@K
        """
        results = {}
        
        for k in k_values:
            precision_scores = []
            recall_scores = []
            ndcg_scores = []
            ap_scores = []
            
            for user_id in test_users:
                if user_id not in recommendations:
                    continue
                
                relevant_items = set(test_users[user_id])
                recommended_items = recommendations[user_id][:k]
                
                # Precision@K
                if recommended_items:
                    relevant_recommended = len(set(recommended_items) & relevant_items)
                    precision = relevant_recommended / len(recommended_items)
                    precision_scores.append(precision)
                
                # Recall@K
                if relevant_items:
                    relevant_recommended = len(set(recommended_items) & relevant_items)
                    recall = relevant_recommended / len(relevant_items)
                    recall_scores.append(recall)
                
                # NDCG@K
                ndcg = self._calculate_ndcg(recommended_items, relevant_items, k)
                ndcg_scores.append(ndcg)
                
                # Average Precision
                ap = self._calculate_average_precision(recommended_items, relevant_items)
                ap_scores.append(ap)
            
            results[f"Precision@{k}"] = np.mean(precision_scores) if precision_scores else 0
            results[f"Recall@{k}"] = np.mean(recall_scores) if recall_scores else 0
            results[f"NDCG@{k}"] = np.mean(ndcg_scores) if ndcg_scores else 0
            results[f"MAP@{k}"] = np.mean(ap_scores) if ap_scores else 0
            results[f"F1@{k}"] = self._calculate_f1(
                results[f"Precision@{k}"], results[f"Recall@{k}"]
            )
        
        return {metric: round(value, 4) for metric, value in results.items()}
    
    def _calculate_ndcg(self, recommended_items: List[int], 
                       relevant_items: Set[int], k: int) -> float:
        """计算NDCG@K"""
        if not recommended_items or not relevant_items:
            return 0.0
        
        # DCG
        dcg = 0.0
        for i, item in enumerate(recommended_items[:k]):
            if item in relevant_items:
                dcg += 1.0 / math.log2(i + 2)  # i+2 because log2(1) is undefined
        
        # IDCG (理想情况下的DCG)
        idcg = 0.0
        for i in range(min(k, len(relevant_items))):
            idcg += 1.0 / math.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _calculate_average_precision(self, recommended_items: List[int], 
                                   relevant_items: Set[int]) -> float:
        """计算Average Precision"""
        if not recommended_items or not relevant_items:
            return 0.0
        
        relevant_count = 0
        precision_sum = 0.0
        
        for i, item in enumerate(recommended_items):
            if item in relevant_items:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i
        
        return precision_sum / len(relevant_items) if relevant_items else 0.0
    
    def _calculate_f1(self, precision: float, recall: float) -> float:
        """计算F1分数"""
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
    
    def calculate_diversity_metrics(self, recommendations: Dict[int, List[int]], 
                                  item_features: Dict[int, List[str]], 
                                  all_items: Set[int]) -> Dict[str, float]:
        """
        计算多样性指标
        
        Args:
            recommendations: {user_id: [recommended_item_ids]}
            item_features: {item_id: [feature_list]} (e.g., genres)
            all_items: 所有可推荐的物品集合
        
        Returns:
            Dict with Coverage, Intra-list Diversity, Inter-list Diversity
        """
        results = {}
        
        # 1. Coverage (推荐覆盖率)
        recommended_items = set()
        for user_recs in recommendations.values():
            recommended_items.update(user_recs)
        
        coverage = len(recommended_items) / len(all_items) if all_items else 0
        results["Coverage"] = round(coverage, 4)
        
        # 2. Intra-list Diversity (列表内多样性)
        intra_diversities = []
        for user_id, recs in recommendations.items():
            if len(recs) > 1:
                diversity = self._calculate_intra_list_diversity(recs, item_features)
                intra_diversities.append(diversity)
        
        results["Intra_List_Diversity"] = round(np.mean(intra_diversities), 4) if intra_diversities else 0
        
        # 3. Gini Coefficient (推荐分布均匀性)
        item_counts = Counter()
        for user_recs in recommendations.values():
            item_counts.update(user_recs)
        
        gini = self._calculate_gini_coefficient(list(item_counts.values()))
        results["Gini_Coefficient"] = round(gini, 4)
        
        return results
    
    def _calculate_intra_list_diversity(self, items: List[int], 
                                      item_features: Dict[int, List[str]]) -> float:
        """计算推荐列表内的多样性"""
        if len(items) < 2:
            return 0.0
        
        similarities = []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                item1, item2 = items[i], items[j]
                if item1 in item_features and item2 in item_features:
                    sim = self._jaccard_similarity(
                        set(item_features[item1]), 
                        set(item_features[item2])
                    )
                    similarities.append(sim)
        
        # 多样性 = 1 - 平均相似度
        avg_similarity = np.mean(similarities) if similarities else 0
        return 1 - avg_similarity
    
    def _jaccard_similarity(self, set1: Set, set2: Set) -> float:
        """计算Jaccard相似度"""
        if not set1 and not set2:
            return 1.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    def _calculate_gini_coefficient(self, values: List[int]) -> float:
        """计算基尼系数 (衡量分布不均匀程度)"""
        if not values:
            return 0.0
        
        values = sorted(values)
        n = len(values)
        cumsum = np.cumsum(values)
        
        return (n + 1 - 2 * sum((n + 1 - i) * y for i, y in enumerate(values))) / (n * sum(values))
    
    def calculate_novelty_metrics(self, recommendations: Dict[int, List[int]], 
                                item_popularity: Dict[int, float],
                                popularity_threshold: float = 0.1) -> Dict[str, float]:
        """
        计算新颖性指标
        
        Args:
            recommendations: {user_id: [recommended_item_ids]}
            item_popularity: {item_id: popularity_score} (0-1)
            popularity_threshold: 流行度阈值，低于此值被认为是新颖的
        
        Returns:
            Dict with Novelty, Serendipity metrics
        """
        results = {}
        
        novelty_scores = []
        serendipity_scores = []
        
        for user_id, recs in recommendations.items():
            # Novelty: 推荐非流行物品的比例
            novel_items = [item for item in recs 
                          if item_popularity.get(item, 0) < popularity_threshold]
            novelty = len(novel_items) / len(recs) if recs else 0
            novelty_scores.append(novelty)
            
            # Serendipity: 平均负对数流行度
            if recs:
                pop_scores = [item_popularity.get(item, 0.01) for item in recs]  # 避免log(0)
                serendipity = -np.mean([math.log(max(pop, 0.001)) for pop in pop_scores])
                serendipity_scores.append(serendipity)
        
        results["Novelty"] = round(np.mean(novelty_scores), 4) if novelty_scores else 0
        results["Serendipity"] = round(np.mean(serendipity_scores), 4) if serendipity_scores else 0
        
        return results
    
    def comprehensive_evaluation(self, 
                               true_ratings: List[Tuple[int, int, float]],
                               predicted_ratings: List[Tuple[int, int, float]],
                               test_users: Dict[int, List[int]],
                               recommendations: Dict[int, List[int]],
                               item_features: Dict[int, List[str]],
                               item_popularity: Dict[int, float],
                               all_items: Set[int]) -> Dict[str, Any]:
        """
        综合评估推荐系统性能
        
        Returns:
            Dict containing all evaluation metrics
        """
        print("📊 开始综合评估...")
        
        results = {
            "evaluation_timestamp": pd.Timestamp.now().isoformat(),
            "dataset_info": {
                "num_users": len(set([x[0] for x in true_ratings])),
                "num_items": len(set([x[1] for x in true_ratings])),
                "num_ratings": len(true_ratings),
                "num_recommendations": sum(len(recs) for recs in recommendations.values())
            }
        }
        
        # 1. 准确性指标
        print("  计算准确性指标...")
        accuracy_metrics = self.calculate_accuracy_metrics(true_ratings, predicted_ratings)
        results["accuracy_metrics"] = accuracy_metrics
        
        # 2. 排序指标
        print("  计算排序指标...")
        ranking_metrics = self.calculate_ranking_metrics(test_users, recommendations)
        results["ranking_metrics"] = ranking_metrics
        
        # 3. 多样性指标
        print("  计算多样性指标...")
        diversity_metrics = self.calculate_diversity_metrics(recommendations, item_features, all_items)
        results["diversity_metrics"] = diversity_metrics
        
        # 4. 新颖性指标
        print("  计算新颖性指标...")
        novelty_metrics = self.calculate_novelty_metrics(recommendations, item_popularity)
        results["novelty_metrics"] = novelty_metrics
        
        # 5. 综合指标
        results["overall_score"] = self._calculate_overall_score(
            accuracy_metrics, ranking_metrics, diversity_metrics, novelty_metrics
        )
        
        print("✅ 综合评估完成")
        return results
    
    def _calculate_overall_score(self, accuracy: Dict, ranking: Dict, 
                               diversity: Dict, novelty: Dict) -> Dict[str, float]:
        """计算综合评分"""
        
        # 提取关键指标
        rmse = accuracy.get("RMSE", 5.0)  # 越小越好
        precision_10 = ranking.get("Precision@10", 0)  # 越大越好
        recall_10 = ranking.get("Recall@10", 0)  # 越大越好
        ndcg_10 = ranking.get("NDCG@10", 0)  # 越大越好
        coverage = diversity.get("Coverage", 0)  # 越大越好
        intra_diversity = diversity.get("Intra_List_Diversity", 0)  # 越大越好
        novelty_score = novelty.get("Novelty", 0)  # 越大越好
        
        # 标准化分数 (0-1)
        accuracy_score = max(0, 1 - rmse / 5.0)  # RMSE标准化
        relevance_score = (precision_10 + recall_10 + ndcg_10) / 3  # 相关性平均
        diversity_score = (coverage + intra_diversity) / 2  # 多样性平均
        novelty_score = novelty_score  # 已经是0-1范围
        
        # 加权综合分数
        overall = (
            0.4 * relevance_score +      # 相关性权重40%
            0.3 * accuracy_score +       # 准确性权重30%
            0.2 * diversity_score +      # 多样性权重20%
            0.1 * novelty_score          # 新颖性权重10%
        )
        
        return {
            "overall_score": round(overall, 4),
            "accuracy_score": round(accuracy_score, 4),
            "relevance_score": round(relevance_score, 4),
            "diversity_score": round(diversity_score, 4),
            "novelty_score": round(novelty_score, 4)
        }
    
    def print_evaluation_report(self, results: Dict[str, Any]):
        """打印评估报告"""
        print("\n" + "="*60)
        print("📋 推荐系统评估报告")
        print("="*60)
        
        # 数据集信息
        dataset_info = results["dataset_info"]
        print(f"📊 数据集信息:")
        print(f"   用户数: {dataset_info['num_users']}")
        print(f"   物品数: {dataset_info['num_items']}")  
        print(f"   评分数: {dataset_info['num_ratings']}")
        print(f"   推荐数: {dataset_info['num_recommendations']}")
        
        # 准确性指标
        print(f"\n🎯 准确性指标:")
        accuracy = results["accuracy_metrics"]
        for metric, value in accuracy.items():
            if metric != "error":
                print(f"   {metric}: {value}")
        
        # 排序指标
        print(f"\n📈 排序指标:")
        ranking = results["ranking_metrics"]
        for metric, value in ranking.items():
            print(f"   {metric}: {value}")
        
        # 多样性指标
        print(f"\n🌈 多样性指标:")
        diversity = results["diversity_metrics"]
        for metric, value in diversity.items():
            print(f"   {metric}: {value}")
        
        # 新颖性指标
        print(f"\n💫 新颖性指标:")
        novelty = results["novelty_metrics"]
        for metric, value in novelty.items():
            print(f"   {metric}: {value}")
        
        # 综合评分
        print(f"\n🏆 综合评分:")
        overall = results["overall_score"]
        for metric, value in overall.items():
            print(f"   {metric}: {value}")
        
        # 性能等级评定
        overall_score = overall["overall_score"]
        if overall_score >= 0.8:
            grade = "🥇 优秀"
        elif overall_score >= 0.6:
            grade = "🥈 良好"
        elif overall_score >= 0.4:
            grade = "🥉 一般"
        else:
            grade = "❌ 待改进"
        
        print(f"\n🎖️  总体评级: {grade} (综合得分: {overall_score:.3f})")
        print("="*60)


# 使用示例和测试
if __name__ == "__main__":
    # 创建示例数据进行测试
    metrics = RecommendationMetrics()
    
    # 示例评分数据
    true_ratings = [(1, 101, 4.0), (1, 102, 3.0), (2, 101, 5.0), (2, 103, 2.0)]
    predicted_ratings = [(1, 101, 4.2), (1, 102, 3.1), (2, 101, 4.8), (2, 103, 2.5)]
    
    # 示例推荐数据
    test_users = {1: [101, 102], 2: [101]}
    recommendations = {1: [101, 103, 104], 2: [101, 102, 105]}
    
    # 示例物品特征
    item_features = {
        101: ["Action", "Sci-Fi"],
        102: ["Drama", "Romance"],  
        103: ["Comedy"],
        104: ["Action", "Thriller"],
        105: ["Drama"]
    }
    
    # 示例物品流行度
    item_popularity = {101: 0.8, 102: 0.3, 103: 0.6, 104: 0.2, 105: 0.4}
    
    all_items = {101, 102, 103, 104, 105, 106, 107}
    
    # 综合评估
    results = metrics.comprehensive_evaluation(
        true_ratings, predicted_ratings, test_users, recommendations,
        item_features, item_popularity, all_items
    )
    
    # 打印报告
    metrics.print_evaluation_report(results)
