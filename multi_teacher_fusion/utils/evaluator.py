"""
Multi-Teacher Fusion System - Evaluation Metrics
推荐系统评估指标集合，包括准确性、多样性、覆盖度等
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass
import logging
from collections import defaultdict
import math

@dataclass
class EvaluationResult:
    """评估结果数据类"""
    # 准确性指标
    rmse: float
    mae: float
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    ndcg_at_k: Dict[int, float]
    hit_rate_at_k: Dict[int, float]
    
    # 多样性指标
    intra_list_diversity: float
    inter_list_diversity: float
    coverage: float
    gini_coefficient: float
    
    # 排序指标
    auc: float
    ranking_loss: float
    
    # 其他指标
    novelty: float
    serendipity: float

class RecommendationEvaluator:
    """推荐系统评估器"""
    
    def __init__(self, k_values: Optional[List[int]] = None):
        """
        初始化评估器
        
        Args:
            k_values: Top-K评估的K值列表，默认[5, 10, 20, 50]
        """
        self.k_values = k_values or [5, 10, 20, 50]
        self.logger = logging.getLogger(__name__)
    
    def rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """均方根误差 (Root Mean Square Error)"""
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    def mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """平均绝对误差 (Mean Absolute Error)"""
        return float(np.mean(np.abs(y_true - y_pred)))
    
    def precision_at_k(self, y_true: List[List[int]], y_pred: List[List[int]], k: int) -> float:
        """
        Precision@K - 推荐列表中相关物品的比例
        
        Args:
            y_true: 真实相关物品列表，每个用户一个列表
            y_pred: 预测推荐物品列表，每个用户一个列表
            k: Top-K
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true和y_pred长度不匹配")
        
        precisions = []
        for true_items, pred_items in zip(y_true, y_pred):
            if len(pred_items) == 0:
                precisions.append(0.0)
                continue
                
            true_set = set(true_items)
            pred_k = pred_items[:k]
            
            relevant_retrieved = len([item for item in pred_k if item in true_set])
            precision = relevant_retrieved / min(k, len(pred_k))
            precisions.append(precision)
        
        return float(np.mean(precisions))
    
    def recall_at_k(self, y_true: List[List[int]], y_pred: List[List[int]], k: int) -> float:
        """
        Recall@K - 真实相关物品中被推荐的比例
        
        Args:
            y_true: 真实相关物品列表
            y_pred: 预测推荐物品列表
            k: Top-K
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true和y_pred长度不匹配")
        
        recalls = []
        for true_items, pred_items in zip(y_true, y_pred):
            if len(true_items) == 0:
                recalls.append(0.0)
                continue
                
            true_set = set(true_items)
            pred_k = pred_items[:k]
            
            relevant_retrieved = len([item for item in pred_k if item in true_set])
            recall = relevant_retrieved / len(true_items)
            recalls.append(recall)
        
        return float(np.mean(recalls))
    
    def dcg_at_k(self, relevance_scores: List[float], k: int) -> float:
        """
        Discounted Cumulative Gain at K
        
        Args:
            relevance_scores: 相关性分数列表（按排序位置）
            k: Top-K
        """
        if len(relevance_scores) == 0:
            return 0.0
        
        relevance_k = relevance_scores[:k]
        dcg = relevance_k[0]
        
        for i in range(1, len(relevance_k)):
            dcg += relevance_k[i] / math.log2(i + 1)
        
        return dcg
    
    def ndcg_at_k(self, y_true: List[List[int]], y_pred: List[List[int]], 
                  ratings: Dict[Tuple[int, int], float], k: int) -> float:
        """
        Normalized Discounted Cumulative Gain at K
        
        Args:
            y_true: 真实相关物品列表
            y_pred: 预测推荐物品列表
            ratings: (user_id, item_id) -> rating的字典
            k: Top-K
        """
        ndcg_scores = []
        
        for user_idx, (true_items, pred_items) in enumerate(zip(y_true, y_pred)):
            if len(pred_items) == 0 or len(true_items) == 0:
                ndcg_scores.append(0.0)
                continue
            
            # 计算预测列表的相关性分数
            pred_relevance = []
            for item in pred_items[:k]:
                if (user_idx, item) in ratings:
                    pred_relevance.append(ratings[(user_idx, item)])
                else:
                    pred_relevance.append(0.0)
            
            # 计算理想列表的相关性分数（按评分排序）
            ideal_relevance = []
            for item in true_items:
                if (user_idx, item) in ratings:
                    ideal_relevance.append(ratings[(user_idx, item)])
                else:
                    ideal_relevance.append(0.0)
            ideal_relevance.sort(reverse=True)
            
            # 计算DCG和IDCG
            dcg = self.dcg_at_k(pred_relevance, k)
            idcg = self.dcg_at_k(ideal_relevance, k)
            
            if idcg == 0:
                ndcg_scores.append(0.0)
            else:
                ndcg_scores.append(dcg / idcg)
        
        return float(np.mean(ndcg_scores))
    
    def hit_rate_at_k(self, y_true: List[List[int]], y_pred: List[List[int]], k: int) -> float:
        """
        Hit Rate@K - 至少有一个相关物品被推荐的用户比例
        
        Args:
            y_true: 真实相关物品列表
            y_pred: 预测推荐物品列表
            k: Top-K
        """
        hits = 0
        total_users = len(y_true)
        
        for true_items, pred_items in zip(y_true, y_pred):
            true_set = set(true_items)
            pred_k = pred_items[:k]
            
            if any(item in true_set for item in pred_k):
                hits += 1
        
        return hits / total_users if total_users > 0 else 0.0
    
    def intra_list_diversity(self, recommendations: List[List[int]], 
                           item_features: Dict[int, List[float]]) -> float:
        """
        列表内多样性 - 单个推荐列表内物品的多样性
        
        Args:
            recommendations: 推荐列表
            item_features: 物品特征字典
        """
        diversities = []
        
        for rec_list in recommendations:
            if len(rec_list) <= 1:
                diversities.append(0.0)
                continue
            
            # 计算列表内物品间的平均距离
            distances = []
            for i in range(len(rec_list)):
                for j in range(i + 1, len(rec_list)):
                    item_i, item_j = rec_list[i], rec_list[j]
                    
                    if item_i in item_features and item_j in item_features:
                        feat_i = np.array(item_features[item_i])
                        feat_j = np.array(item_features[item_j])
                        distance = np.linalg.norm(feat_i - feat_j)
                        distances.append(distance)
            
            if distances:
                diversities.append(np.mean(distances))
            else:
                diversities.append(0.0)
        
        return float(np.mean(diversities))
    
    def coverage(self, recommendations: List[List[int]], total_items: int) -> float:
        """
        覆盖度 - 被推荐的物品占总物品的比例
        
        Args:
            recommendations: 推荐列表
            total_items: 总物品数量
        """
        recommended_items = set()
        for rec_list in recommendations:
            recommended_items.update(rec_list)
        
        return len(recommended_items) / total_items if total_items > 0 else 0.0
    
    def gini_coefficient(self, recommendations: List[List[int]]) -> float:
        """
        基尼系数 - 衡量推荐物品分布的不平等程度
        
        Args:
            recommendations: 推荐列表
        """
        # 计算每个物品被推荐的次数
        item_counts = defaultdict(int)
        for rec_list in recommendations:
            for item in rec_list:
                item_counts[item] += 1
        
        if not item_counts:
            return 0.0
        
        # 计算基尼系数
        counts = sorted(item_counts.values())
        n = len(counts)
        cumsum = np.cumsum(counts)
        
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    
    def auc_score(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """
        AUC Score - 二分类问题的AUC
        
        Args:
            y_true: 真实标签 (0/1)
            y_scores: 预测分数
        """
        from sklearn.metrics import roc_auc_score
        try:
            return float(roc_auc_score(y_true, y_scores))
        except ValueError:
            # 处理只有一个类别的情况
            return 0.5
    
    def novelty_score(self, recommendations: List[List[int]], 
                     item_popularity: Dict[int, float]) -> float:
        """
        新颖性 - 推荐物品的平均新颖性（负流行度的对数）
        
        Args:
            recommendations: 推荐列表
            item_popularity: 物品流行度字典
        """
        novelty_scores = []
        
        for rec_list in recommendations:
            list_novelty = []
            for item in rec_list:
                if item in item_popularity and item_popularity[item] > 0:
                    novelty = -math.log2(item_popularity[item])
                    list_novelty.append(novelty)
            
            if list_novelty:
                novelty_scores.append(np.mean(list_novelty))
        
        return float(np.mean(novelty_scores)) if novelty_scores else 0.0
    
    def comprehensive_evaluate(self, 
                             y_true_ratings: np.ndarray,
                             y_pred_ratings: np.ndarray,
                             y_true_items: List[List[int]],
                             y_pred_items: List[List[int]],
                             ratings_dict: Dict[Tuple[int, int], float],
                             item_features: Optional[Dict[int, List[float]]] = None,
                             item_popularity: Optional[Dict[int, float]] = None,
                             total_items: Optional[int] = None) -> EvaluationResult:
        """
        综合评估，计算所有指标
        
        Args:
            y_true_ratings: 真实评分
            y_pred_ratings: 预测评分
            y_true_items: 真实相关物品列表
            y_pred_items: 预测推荐物品列表
            ratings_dict: 评分字典
            item_features: 物品特征字典
            item_popularity: 物品流行度字典
            total_items: 总物品数量
        """
        self.logger.info("开始综合评估...")
        
        # 准确性指标
        rmse = self.rmse(y_true_ratings, y_pred_ratings)
        mae = self.mae(y_true_ratings, y_pred_ratings)
        
        # Top-K指标
        precision_at_k = {}
        recall_at_k = {}
        ndcg_at_k = {}
        hit_rate_at_k = {}
        
        for k in self.k_values:
            precision_at_k[k] = self.precision_at_k(y_true_items, y_pred_items, k)
            recall_at_k[k] = self.recall_at_k(y_true_items, y_pred_items, k)
            ndcg_at_k[k] = self.ndcg_at_k(y_true_items, y_pred_items, ratings_dict, k)
            hit_rate_at_k[k] = self.hit_rate_at_k(y_true_items, y_pred_items, k)
        
        # 多样性指标
        intra_diversity = 0.0
        if item_features:
            intra_diversity = self.intra_list_diversity(y_pred_items, item_features)
        
        coverage = 0.0
        if total_items:
            coverage = self.coverage(y_pred_items, total_items)
        
        gini = self.gini_coefficient(y_pred_items)
        
        # 新颖性
        novelty = 0.0
        if item_popularity:
            novelty = self.novelty_score(y_pred_items, item_popularity)
        
        # 创建结果对象
        result = EvaluationResult(
            rmse=rmse,
            mae=mae,
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            ndcg_at_k=ndcg_at_k,
            hit_rate_at_k=hit_rate_at_k,
            intra_list_diversity=intra_diversity,
            inter_list_diversity=0.0,  # 需要额外计算
            coverage=coverage,
            gini_coefficient=gini,
            auc=0.0,  # 需要额外计算
            ranking_loss=0.0,  # 需要额外计算
            novelty=novelty,
            serendipity=0.0  # 需要额外计算
        )
        
        self.logger.info("综合评估完成")
        return result
    
    def print_evaluation_report(self, result: EvaluationResult) -> None:
        """打印评估报告"""
        print("\n" + "="*60)
        print("📊 Multi-Teacher Fusion System - Evaluation Report")
        print("="*60)
        
        print(f"\n🎯 准确性指标:")
        print(f"   RMSE: {result.rmse:.4f}")
        print(f"   MAE:  {result.mae:.4f}")
        
        print(f"\n📈 Top-K排序指标:")
        for k in sorted(result.precision_at_k.keys()):
            print(f"   Precision@{k}: {result.precision_at_k[k]:.4f}")
            print(f"   Recall@{k}:    {result.recall_at_k[k]:.4f}")
            print(f"   NDCG@{k}:      {result.ndcg_at_k[k]:.4f}")
            print(f"   HitRate@{k}:   {result.hit_rate_at_k[k]:.4f}")
            print()
        
        print(f"🌈 多样性指标:")
        print(f"   List Diversity: {result.intra_list_diversity:.4f}")
        print(f"   Coverage:       {result.coverage:.4f}")
        print(f"   Gini Coeff:     {result.gini_coefficient:.4f}")
        
        print(f"\n✨ 其他指标:")
        print(f"   Novelty:       {result.novelty:.4f}")
        
        print("="*60)

if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 创建示例数据
    np.random.seed(42)
    
    # 评分预测示例
    y_true_ratings = np.random.uniform(1, 5, 1000)
    y_pred_ratings = y_true_ratings + np.random.normal(0, 0.5, 1000)
    
    # 推荐列表示例
    num_users = 100
    y_true_items = [list(np.random.choice(100, size=5, replace=False)) for _ in range(num_users)]
    y_pred_items = [list(np.random.choice(100, size=10, replace=False)) for _ in range(num_users)]
    
    # 评分字典示例
    ratings_dict = {(i, j): np.random.uniform(1, 5) for i in range(num_users) for j in range(10)}
    
    # 初始化评估器
    evaluator = RecommendationEvaluator()
    
    # 执行评估
    result = evaluator.comprehensive_evaluate(
        y_true_ratings=y_true_ratings,
        y_pred_ratings=y_pred_ratings,
        y_true_items=y_true_items,
        y_pred_items=y_pred_items,
        ratings_dict=ratings_dict,
        total_items=100
    )
    
    # 打印报告
    evaluator.print_evaluation_report(result)
