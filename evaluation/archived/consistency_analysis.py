"""
算法一致性评估模块
用于评估不同推荐算法之间的重叠度和一致性，以便优化ensemble效果
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set
import logging
from scipy.stats import kendalltau
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

logger = logging.getLogger(__name__)


class AlgorithmConsistencyEvaluator:
    """算法一致性评估器"""
    
    def __init__(self):
        self.algorithm_recommendations = {}
        self.users_tested = set()
        
    def add_algorithm_recommendations(self, algorithm_name: str, 
                                    user_recommendations: Dict[int, List[int]]):
        """
        添加算法的推荐结果
        
        Args:
            algorithm_name: 算法名称
            user_recommendations: {user_id: [item_id1, item_id2, ...]}
        """
        self.algorithm_recommendations[algorithm_name] = user_recommendations
        self.users_tested.update(user_recommendations.keys())
        
    def jaccard_similarity(self, set1: Set, set2: Set) -> float:
        """
        计算两个集合的Jaccard相似度
        
        Args:
            set1, set2: 两个推荐物品集合
            
        Returns:
            Jaccard相似度 [0, 1]
        """
        if not set1 and not set2:
            return 1.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0
    
    def kendall_tau_distance(self, list1: List[int], list2: List[int]) -> float:
        """
        计算两个排序列表的Kendall Tau距离
        
        Args:
            list1, list2: 两个推荐物品排序列表
            
        Returns:
            Kendall Tau相关系数 [-1, 1]
        """
        # 找到共同物品
        common_items = set(list1).intersection(set(list2))
        if len(common_items) < 2:
            return 0.0  # 需要至少2个共同物品才能计算相关性
        
        # 为共同物品创建排序
        rank1 = {item: idx for idx, item in enumerate(list1) if item in common_items}
        rank2 = {item: idx for idx, item in enumerate(list2) if item in common_items}
        
        # 提取排序值
        ranks1 = [rank1[item] for item in common_items]
        ranks2 = [rank2[item] for item in common_items]
        
        # 计算Kendall Tau
        try:
            tau, _ = kendalltau(ranks1, ranks2)
            return tau if not np.isnan(tau) else 0.0
        except:
            return 0.0
    
    def calculate_pairwise_consistency(self, top_k: int = 10) -> Dict[str, Dict[str, float]]:
        """
        计算所有算法对之间的一致性
        
        Args:
            top_k: 考虑top-k推荐
            
        Returns:
            一致性矩阵字典
        """
        algorithms = list(self.algorithm_recommendations.keys())
        jaccard_matrix = defaultdict(dict)
        kendall_matrix = defaultdict(dict)
        
        logger.info(f"计算 {len(algorithms)} 个算法之间的一致性...")
        
        for algo1, algo2 in itertools.combinations(algorithms, 2):
            jaccard_scores = []
            kendall_scores = []
            
            common_users = (set(self.algorithm_recommendations[algo1].keys()) & 
                          set(self.algorithm_recommendations[algo2].keys()))
            
            for user_id in common_users:
                recs1 = self.algorithm_recommendations[algo1][user_id][:top_k]
                recs2 = self.algorithm_recommendations[algo2][user_id][:top_k]
                
                # Jaccard相似度
                jaccard = self.jaccard_similarity(set(recs1), set(recs2))
                jaccard_scores.append(jaccard)
                
                # Kendall Tau相关性
                kendall = self.kendall_tau_distance(recs1, recs2)
                kendall_scores.append(kendall)
            
            # 计算平均值
            avg_jaccard = np.mean(jaccard_scores) if jaccard_scores else 0.0
            avg_kendall = np.mean(kendall_scores) if kendall_scores else 0.0
            
            # 填充矩阵（对称）
            jaccard_matrix[algo1][algo2] = avg_jaccard
            jaccard_matrix[algo2][algo1] = avg_jaccard
            kendall_matrix[algo1][algo2] = avg_kendall
            kendall_matrix[algo2][algo1] = avg_kendall
        
        # 对角线为1.0
        for algo in algorithms:
            jaccard_matrix[algo][algo] = 1.0
            kendall_matrix[algo][algo] = 1.0
        
        return {
            'jaccard': dict(jaccard_matrix),
            'kendall': dict(kendall_matrix)
        }
    
    def calculate_diversity_metrics(self, top_k: int = 10) -> Dict[str, float]:
        """
        计算算法多样性指标
        
        Args:
            top_k: 考虑top-k推荐
            
        Returns:
            多样性指标字典
        """
        algorithms = list(self.algorithm_recommendations.keys())
        
        # 算法间平均Jaccard相似度
        consistency_matrices = self.calculate_pairwise_consistency(top_k)
        jaccard_matrix = consistency_matrices['jaccard']
        
        avg_similarities = []
        for algo1 in algorithms:
            for algo2 in algorithms:
                if algo1 != algo2:
                    avg_similarities.append(jaccard_matrix[algo1][algo2])
        
        avg_jaccard = np.mean(avg_similarities) if avg_similarities else 0.0
        
        # 算法覆盖度多样性
        all_recommended_items = set()
        algorithm_item_counts = {}
        
        for algo in algorithms:
            algo_items = set()
            for user_recs in self.algorithm_recommendations[algo].values():
                algo_items.update(user_recs[:top_k])
            algorithm_item_counts[algo] = len(algo_items)
            all_recommended_items.update(algo_items)
        
        # 覆盖度多样性：所有算法推荐的不同物品数量
        total_coverage = len(all_recommended_items)
        avg_coverage = np.mean(list(algorithm_item_counts.values()))
        coverage_diversity = total_coverage / avg_coverage if avg_coverage > 0 else 1.0
        
        return {
            'avg_jaccard_similarity': avg_jaccard,
            'total_item_coverage': total_coverage,
            'avg_algorithm_coverage': avg_coverage,
            'coverage_diversity_ratio': coverage_diversity,
            'algorithm_item_counts': algorithm_item_counts
        }
    
    def recommend_ensemble_strategy(self, top_k: int = 10) -> Dict[str, any]:
        """
        基于一致性分析推荐最佳ensemble策略
        
        Args:
            top_k: 考虑top-k推荐
            
        Returns:
            推荐的ensemble策略
        """
        consistency_matrices = self.calculate_pairwise_consistency(top_k)
        diversity_metrics = self.calculate_diversity_metrics(top_k)
        
        jaccard_matrix = consistency_matrices['jaccard']
        kendall_matrix = consistency_matrices['kendall']
        algorithms = list(self.algorithm_recommendations.keys())
        
        # 找到最互补的算法对（低相似度）
        min_similarity = float('inf')
        most_complementary_pair = None
        
        for algo1 in algorithms:
            for algo2 in algorithms:
                if algo1 != algo2:
                    similarity = jaccard_matrix[algo1][algo2]
                    if similarity < min_similarity:
                        min_similarity = similarity
                        most_complementary_pair = (algo1, algo2)
        
        # 找到最相似的算法对（高相似度）
        max_similarity = -1
        most_similar_pair = None
        
        for algo1 in algorithms:
            for algo2 in algorithms:
                if algo1 != algo2:
                    similarity = jaccard_matrix[algo1][algo2]
                    if similarity > max_similarity:
                        max_similarity = similarity
                        most_similar_pair = (algo1, algo2)
        
        # 基于多样性评分排序算法
        diversity_scores = {}
        for algo in algorithms:
            # 该算法与其他算法的平均相似度（越低越好，表示越独特）
            similarities = [jaccard_matrix[algo][other] for other in algorithms if other != algo]
            avg_similarity = np.mean(similarities) if similarities else 0.0
            diversity_scores[algo] = 1.0 - avg_similarity  # 转换为多样性分数
        
        # 推荐策略
        strategy = {
            'most_complementary_pair': {
                'algorithms': most_complementary_pair,
                'jaccard_similarity': min_similarity,
                'recommendation': 'suitable_for_ensemble'
            },
            'most_similar_pair': {
                'algorithms': most_similar_pair, 
                'jaccard_similarity': max_similarity,
                'recommendation': 'redundant_for_ensemble'
            },
            'diversity_ranking': sorted(diversity_scores.items(), key=lambda x: x[1], reverse=True),
            'ensemble_recommendations': []
        }
        
        # 生成ensemble建议
        if min_similarity < 0.3:
            strategy['ensemble_recommendations'].append(
                f"推荐ensemble: {most_complementary_pair[0]} + {most_complementary_pair[1]} (互补性强)"
            )
        
        if len(algorithms) >= 3:
            # 选择多样性最高的3个算法
            top_diverse = [algo for algo, _ in strategy['diversity_ranking'][:3]]
            strategy['ensemble_recommendations'].append(
                f"推荐3-算法ensemble: {' + '.join(top_diverse)} (多样性最佳)"
            )
        
        return strategy
    
    def visualize_consistency_matrix(self, top_k: int = 10, save_path: str = None):
        """
        可视化一致性矩阵
        
        Args:
            top_k: 考虑top-k推荐
            save_path: 保存路径
        """
        consistency_matrices = self.calculate_pairwise_consistency(top_k)
        algorithms = list(self.algorithm_recommendations.keys())
        
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Jaccard相似度矩阵
        jaccard_data = []
        for algo1 in algorithms:
            row = []
            for algo2 in algorithms:
                row.append(consistency_matrices['jaccard'][algo1][algo2])
            jaccard_data.append(row)
        
        sns.heatmap(jaccard_data, annot=True, fmt='.3f', 
                   xticklabels=algorithms, yticklabels=algorithms,
                   cmap='Blues', ax=ax1)
        ax1.set_title(f'Jaccard Similarity Matrix (Top-{top_k})')
        
        # Kendall Tau相关性矩阵
        kendall_data = []
        for algo1 in algorithms:
            row = []
            for algo2 in algorithms:
                row.append(consistency_matrices['kendall'][algo1][algo2])
            kendall_data.append(row)
        
        sns.heatmap(kendall_data, annot=True, fmt='.3f',
                   xticklabels=algorithms, yticklabels=algorithms, 
                   cmap='RdBu_r', center=0, ax=ax2)
        ax2.set_title(f'Kendall Tau Correlation Matrix (Top-{top_k})')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_consistency_report(self, top_k_values: List[int] = [5, 10]) -> str:
        """
        生成详细的一致性分析报告
        
        Args:
            top_k_values: 要分析的top-k值列表
            
        Returns:
            格式化的报告字符串
        """
        report = []
        report.append("=" * 60)
        report.append("算法一致性分析报告")
        report.append("=" * 60)
        
        algorithms = list(self.algorithm_recommendations.keys())
        report.append(f"\n分析算法: {', '.join(algorithms)}")
        report.append(f"测试用户数: {len(self.users_tested)}")
        
        for top_k in top_k_values:
            report.append(f"\n{'-' * 40}")
            report.append(f"Top-{top_k} 推荐一致性分析")
            report.append(f"{'-' * 40}")
            
            # 计算一致性矩阵
            consistency_matrices = self.calculate_pairwise_consistency(top_k)
            diversity_metrics = self.calculate_diversity_metrics(top_k)
            
            # Jaccard相似度统计
            jaccard_matrix = consistency_matrices['jaccard']
            jaccard_values = []
            for algo1 in algorithms:
                for algo2 in algorithms:
                    if algo1 != algo2:
                        jaccard_values.append(jaccard_matrix[algo1][algo2])
            
            report.append(f"\nJaccard相似度统计:")
            report.append(f"  平均相似度: {np.mean(jaccard_values):.4f}")
            report.append(f"  最小相似度: {np.min(jaccard_values):.4f}")
            report.append(f"  最大相似度: {np.max(jaccard_values):.4f}")
            report.append(f"  标准差: {np.std(jaccard_values):.4f}")
            
            # Kendall Tau相关性统计
            kendall_matrix = consistency_matrices['kendall']
            kendall_values = []
            for algo1 in algorithms:
                for algo2 in algorithms:
                    if algo1 != algo2:
                        kendall_values.append(kendall_matrix[algo1][algo2])
            
            report.append(f"\nKendall Tau相关性统计:")
            report.append(f"  平均相关性: {np.mean(kendall_values):.4f}")
            report.append(f"  最小相关性: {np.min(kendall_values):.4f}")
            report.append(f"  最大相关性: {np.max(kendall_values):.4f}")
            report.append(f"  标准差: {np.std(kendall_values):.4f}")
            
            # 详细的成对比较
            report.append(f"\n成对算法Jaccard相似度:")
            for algo1 in algorithms:
                for algo2 in algorithms:
                    if algo1 < algo2:  # 避免重复
                        similarity = jaccard_matrix[algo1][algo2]
                        correlation = kendall_matrix[algo1][algo2]
                        report.append(f"  {algo1:<15} vs {algo2:<15}: Jaccard={similarity:.4f}, Kendall={correlation:.4f}")
            
            # 多样性指标
            report.append(f"\n多样性指标:")
            report.append(f"  总物品覆盖度: {diversity_metrics['total_item_coverage']}")
            report.append(f"  平均算法覆盖度: {diversity_metrics['avg_algorithm_coverage']:.1f}")
            report.append(f"  覆盖度多样性比率: {diversity_metrics['coverage_diversity_ratio']:.2f}")
            
            # Ensemble建议
            ensemble_strategy = self.recommend_ensemble_strategy(top_k)
            report.append(f"\nEnsemble策略建议:")
            
            if ensemble_strategy['most_complementary_pair']:
                pair = ensemble_strategy['most_complementary_pair']
                report.append(f"  最互补算法对: {pair['algorithms'][0]} + {pair['algorithms'][1]} (Jaccard={pair['jaccard_similarity']:.4f})")
            
            report.append(f"  算法多样性排序:")
            for i, (algo, score) in enumerate(ensemble_strategy['diversity_ranking']):
                report.append(f"    {i+1}. {algo}: {score:.4f}")
            
            for recommendation in ensemble_strategy['ensemble_recommendations']:
                report.append(f"  • {recommendation}")
        
        return "\n".join(report)


def format_recommendations_for_consistency(user_recommendations: Dict[int, List[Dict]]) -> Dict[int, List[int]]:
    """
    将推荐结果格式转换为一致性分析所需的格式
    
    Args:
        user_recommendations: {user_id: [{'item_id': int, 'score': float}, ...]}
        
    Returns:
        {user_id: [item_id1, item_id2, ...]}
    """
    formatted = {}
    for user_id, recs in user_recommendations.items():
        if isinstance(recs, list) and len(recs) > 0:
            if isinstance(recs[0], dict):
                formatted[user_id] = [rec['item_id'] for rec in recs if 'item_id' in rec]
            else:
                formatted[user_id] = recs
        else:
            formatted[user_id] = []
    return formatted


# 示例使用
if __name__ == "__main__":
    # 模拟推荐结果
    user_recs_algo1 = {
        1: [101, 102, 103, 104, 105],
        2: [201, 202, 203, 204, 205],
        3: [301, 302, 303, 304, 305]
    }
    
    user_recs_algo2 = {
        1: [101, 103, 106, 107, 108],
        2: [201, 203, 206, 207, 208], 
        3: [301, 309, 310, 311, 312]
    }
    
    # 创建评估器
    evaluator = AlgorithmConsistencyEvaluator()
    evaluator.add_algorithm_recommendations('SVD', user_recs_algo1)
    evaluator.add_algorithm_recommendations('DeepFM', user_recs_algo2)
    
    # 生成报告
    report = evaluator.generate_consistency_report([5, 10])
    print(report)
