#!/usr/bin/env python3
"""
集成推荐系统

结合多种推荐算法的优势，提供更准确和可解释的推荐结果。
包括SVD、DeepFM等多种算法的集成，以及可解释性分析。

特点：
- 多算法集成：结合传统协同过滤和深度学习方法
- 可解        # 训练各个子模型
        for algorithm in self.algorithm_names:：提供推荐理由和特征重要性分析
- 多样性：确保推荐结果的多样性
- 实时更新：支持在线学习和模型更新
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict
import os
import pickle

from .algorithm_factory import create_recommender
from .base_recommender import BaseRecommender

logger = logging.getLogger(__name__)


class ExplainableRecommender:
    """可解释的集成推荐器"""
    
    def __init__(self, algorithm_names: List[str], **kwargs):
        """
        初始化集成推荐器
        
        Args:
            algorithm_names: 算法名称列表，如 ['svd', 'deepfm', 'autoInt']
        """
        # 解析算法名称（支持逗号分隔的字符串）
        if len(algorithm_names) == 1 and ',' in algorithm_names[0]:
            algorithm_names = [name.strip() for name in algorithm_names[0].split(',')]
        
        self.algorithm_names = algorithm_names
        self.models = {}
        self.weights = {}
        self.movie_metadata = None
        self.user_stats = None
        self.item_stats = None
        self.item_features = {}  # 添加缺少的属性
        self.user_features = {}  # 添加缺少的属性
        self.user_profiles = {}  # 用户档案
        self.item_profiles = {}  # 物品档案
        self.item_popularity = {}  # 添加缺少的属性
        self.is_trained = False  # 添加缺少的属性
        
        # 集成参数
        self.diversity_weight = kwargs.get('diversity_weight', 0.1)
        self.popularity_penalty = kwargs.get('popularity_penalty', 0.05)
        
        logger.info("初始化集成推荐器，算法: %s", algorithm_names)
    
    def _load_movie_metadata(self, movies_path: str = 'data/sample/movies.csv') -> None:
        """加载电影元数据"""
        try:
            if os.path.exists(movies_path):
                movies_df = pd.read_csv(movies_path)
                for _, row in movies_df.iterrows():
                    item_id = int(row['item_id'])
                    self.item_features[item_id] = {
                        'title': row['title'],
                        'genres': row['genres'].split('|') if pd.notna(row['genres']) else [],
                        'year': self._extract_year(row['title'])
                    }
                logger.info("加载了 %d 个物品的元数据", len(self.item_features))
            else:
                logger.warning("电影元数据文件不存在: %s", movies_path)
        except (FileNotFoundError, KeyError, ValueError, pd.errors.EmptyDataError) as e:
            logger.error("加载电影元数据失败: %s", str(e))
    
    def _extract_year(self, title: str) -> Optional[int]:
        """从标题中提取年份"""
        import re
        match = re.search(r'\((\d{4})\)', title)
        return int(match.group(1)) if match else None
    
    def _analyze_data_statistics(self, train_data: pd.DataFrame) -> None:
        """分析数据统计信息"""
        # 计算物品流行度
        item_counts = train_data['item_id'].value_counts()
        total_interactions = len(train_data)
        self.item_popularity = {item_id: count / total_interactions 
                              for item_id, count in item_counts.items()}
        
        # 分析用户偏好
        for user_id in train_data['user_id'].unique():
            user_ratings = train_data[train_data['user_id'] == user_id]
            
            # 用户评分分布
            avg_rating = user_ratings['rating'].mean()
            rating_std = user_ratings['rating'].std()
            
            # 用户喜欢的类型（基于高评分物品）
            high_rated_items = user_ratings[user_ratings['rating'] >= 4]['item_id'].tolist()
            preferred_genres = []
            
            for item_id in high_rated_items:
                if item_id in self.item_features:
                    preferred_genres.extend(self.item_features[item_id]['genres'])
            
            genre_counts = pd.Series(preferred_genres).value_counts()
            
            self.user_profiles[user_id] = {
                'avg_rating': avg_rating,
                'rating_std': rating_std,
                'num_ratings': len(user_ratings),
                'preferred_genres': genre_counts.head(3).to_dict(),
                'activity_level': 'high' if len(user_ratings) > train_data.groupby('user_id').size().quantile(0.7) else 'low'
            }
        
        logger.info("数据统计分析完成")
    
    def fit(self, train_data: pd.DataFrame) -> None:
        """训练集成模型"""
        logger.info("开始训练集成推荐模型...")
        
        # 加载元数据和分析统计信息
        self._load_movie_metadata()
        self._analyze_data_statistics(train_data)
        
        # 使用算法工厂创建和训练各个模型
        for algorithm in self.algorithm_names:
            try:
                # 使用算法工厂创建模型
                if algorithm == 'svd':
                    model = create_recommender('svd', n_components=50)
                elif algorithm in ['deepfm', 'simple_deepfm']:
                    model = create_recommender(algorithm, 
                                             embedding_dim=8, 
                                             epochs=5,
                                             batch_size=256)
                elif algorithm in ['autoint', 'simple_autoint']:
                    model = create_recommender(algorithm,
                                             embedding_dim=8,
                                             num_heads=2,
                                             epochs=5)
                elif algorithm in ['din', 'simple_din']:
                    model = create_recommender(algorithm,
                                             embedding_dim=8,
                                             epochs=5)
                else:
                    # 尝试通用创建
                    model = create_recommender(algorithm)
                
                if model is not None:
                    logger.info("开始训练%s模型...", algorithm)
                    model.fit(train_data)
                    self.models[algorithm] = model
                    logger.info("%s模型训练完成", algorithm)
                else:
                    logger.warning("无法创建算法: %s", algorithm)
                    
            except (ImportError, ValueError, RuntimeError) as e:
                logger.error("训练模型 %s 失败: %s", algorithm, str(e))
                # 如果主算法失败，尝试简化版本
                if not algorithm.startswith('simple_'):
                    simple_algorithm = f"simple_{algorithm}"
                    try:
                        logger.info("尝试使用简化版本: %s", simple_algorithm)
                        model = create_recommender(simple_algorithm)
                        if model is not None:
                            model.fit(train_data)
                            self.models[simple_algorithm] = model
                            logger.info("%s模型训练完成", simple_algorithm)
                    except (ImportError, ValueError, RuntimeError) as e2:
                        logger.error("简化版本 %s 也失败: %s", simple_algorithm, str(e2))
        
        self.is_trained = True
        logger.info("集成模型训练完成，成功训练 %d 个模型", len(self.models))
    
    def predict(self, user_id: int, item_id: int) -> float:
        """预测评分"""
        if not self.is_trained:
            return 0.0
        
        predictions = []
        weights = []
        
        for algorithm, model in self.models.items():
            try:
                pred = model.predict(user_id, item_id)
                if pred > 0:  # 有效预测
                    predictions.append(pred)
                    weights.append(self.weights.get(algorithm, 1.0))
            except Exception as e:
                logger.warning("模型 %s 预测失败: %s", algorithm, str(e))
        
        if not predictions:
            return 0.0
        
        # 加权平均
        weighted_pred = np.average(predictions, weights=weights)
        
        # 流行度惩罚
        popularity = self.item_popularity.get(item_id, 0)
        popularity_penalty = self.popularity_penalty * popularity
        
        return max(1.0, min(5.0, weighted_pred - popularity_penalty))
    
    def get_user_recommendations(self, user_id: int, top_k: int = 10, 
                               enable_explanation: bool = True) -> List[Dict[str, Any]]:
        """获取用户推荐（带解释）"""
        if not self.is_trained:
            return []
        
        try:
            # 收集所有模型的推荐
            all_recommendations = defaultdict(list)
            
            for algorithm, model in self.models.items():
                try:
                    model_recs = model.get_user_recommendations(user_id, top_k * 2)
                    for item_id, score in model_recs:
                        all_recommendations[item_id].append({
                            'algorithm': algorithm,
                            'score': score,
                            'weight': self.weights.get(algorithm, 1.0)
                        })
                except Exception as e:
                    logger.warning("获取模型 %s 推荐失败: %s", algorithm, str(e))
            
            # 集成评分和多样性处理
            final_recommendations = []
            
            for item_id, model_scores in all_recommendations.items():
                # 加权集成
                weighted_scores = [ms['score'] * ms['weight'] for ms in model_scores]
                total_weight = sum(ms['weight'] for ms in model_scores)
                ensemble_score = sum(weighted_scores) / total_weight if total_weight > 0 else 0
                
                # 流行度惩罚
                popularity = self.item_popularity.get(item_id, 0)
                adjusted_score = ensemble_score - self.popularity_penalty * popularity
                
                # 构建推荐项
                recommendation = {
                    'item_id': item_id,
                    'score': adjusted_score,
                    'ensemble_details': model_scores,
                    'popularity': popularity
                }
                
                # 添加物品信息
                if item_id in self.item_features:
                    recommendation.update(self.item_features[item_id])
                
                # 添加解释
                if enable_explanation:
                    recommendation['explanation'] = self._generate_explanation(user_id, item_id, model_scores)
                
                final_recommendations.append(recommendation)
            
            # 排序
            final_recommendations.sort(key=lambda x: x['score'], reverse=True)
            
            # 多样性处理
            if self.diversity_weight > 0:
                final_recommendations = self._apply_diversity(final_recommendations, user_id)
            
            return final_recommendations[:top_k]
            
        except Exception as e:
            logger.error("获取推荐失败: %s", str(e))
            return []
    
    def _generate_explanation(self, user_id: int, item_id: int, model_scores: List[Dict]) -> Dict[str, Any]:
        """生成推荐解释"""
        explanation = {
            'primary_reason': '',
            'supporting_factors': [],
            'algorithm_contributions': {},
            'confidence': 0.0
        }
        
        try:
            # 算法贡献分析
            for ms in model_scores:
                explanation['algorithm_contributions'][ms['algorithm']] = {
                    'score': ms['score'],
                    'weight': ms['weight'],
                    'contribution': ms['score'] * ms['weight']
                }
            
            # 置信度（基于模型一致性）
            scores = [ms['score'] for ms in model_scores]
            if len(scores) > 1:
                explanation['confidence'] = 1.0 - (np.std(scores) / np.mean(scores))
            else:
                explanation['confidence'] = 0.5
            
            # 用户偏好匹配
            user_profile = self.user_profiles.get(user_id, {})
            item_info = self.item_features.get(item_id, {})
            
            if user_profile and item_info:
                # 类型匹配
                user_genres = set(user_profile.get('preferred_genres', {}).keys())
                item_genres = set(item_info.get('genres', []))
                genre_overlap = user_genres.intersection(item_genres)
                
                if genre_overlap:
                    explanation['primary_reason'] = f"匹配您喜欢的类型: {', '.join(genre_overlap)}"
                    explanation['supporting_factors'].append("类型偏好匹配")
                
                # 评分习惯
                if user_profile.get('avg_rating', 0) > 3.5:
                    explanation['supporting_factors'].append("根据您的高标准推荐")
                
                # 流行度分析
                popularity = self.item_popularity.get(item_id, 0)
                if popularity < 0.01:
                    explanation['supporting_factors'].append("发现小众优质内容")
                elif popularity > 0.05:
                    explanation['supporting_factors'].append("热门推荐")
            
            if not explanation['primary_reason']:
                explanation['primary_reason'] = "基于您的观影历史智能推荐"
                
        except Exception as e:
            logger.warning("生成解释失败: %s", str(e))
            explanation['primary_reason'] = "智能算法推荐"
        
        return explanation
    
    def _apply_diversity(self, recommendations: List[Dict], _: int) -> List[Dict]:
        """应用多样性策略"""
        if not recommendations:
            return recommendations
        
        try:
            # MMR (Maximal Marginal Relevance) 算法
            selected = []
            candidates = recommendations.copy()
            
            # 选择第一个（最高分）
            if candidates:
                selected.append(candidates.pop(0))
            
            # 后续选择平衡相关性和多样性
            while candidates and len(selected) < len(recommendations):
                best_score = -1
                best_idx = 0
                
                for i, candidate in enumerate(candidates):
                    # 相关性分数
                    relevance = candidate['score']
                    
                    # 多样性分数（与已选择项目的最小相似度）
                    min_similarity = float('inf')
                    for selected_item in selected:
                        similarity = self._calculate_item_similarity(candidate, selected_item)
                        min_similarity = min(min_similarity, similarity)
                    
                    if min_similarity == float('inf'):
                        min_similarity = 0
                    
                    # MMR分数
                    mmr_score = (1 - self.diversity_weight) * relevance - self.diversity_weight * min_similarity
                    
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = i
                
                selected.append(candidates.pop(best_idx))
            
            return selected
            
        except Exception as e:
            logger.warning("应用多样性策略失败: %s", str(e))
            return recommendations
    
    def _calculate_item_similarity(self, item1: Dict, item2: Dict) -> float:
        """计算物品相似度"""
        try:
            # 类型相似度
            genres1 = set(item1.get('genres', []))
            genres2 = set(item2.get('genres', []))
            
            if not genres1 or not genres2:
                return 0.0
            
            # Jaccard相似度
            intersection = len(genres1.intersection(genres2))
            union = len(genres1.union(genres2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def save_model(self, filepath: str) -> bool:
        """保存集成模型"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # 分别保存各个模型
            model_data = {
                'algorithms': self.algorithm_names,
                'weights': self.weights,
                'diversity_weight': self.diversity_weight,
                'popularity_penalty': self.popularity_penalty,
                'item_popularity': self.item_popularity,
                'user_profiles': self.user_profiles,
                'item_features': self.item_features,
                'is_trained': self.is_trained
            }
            
            # 保存主数据
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            # 保存各个子模型
            models_dir = os.path.splitext(filepath)[0] + '_models'
            os.makedirs(models_dir, exist_ok=True)
            
            for algorithm, model in self.models.items():
                model_path = os.path.join(models_dir, f'{algorithm}_model.pkl')
                model.save_model(model_path)
            
            logger.info("集成模型已保存到: %s", filepath)
            return True
            
        except Exception as e:
            logger.error("保存集成模型失败: %s", str(e))
            return False
    
    def load_model(self, filepath: str) -> bool:
        """加载集成模型"""
        try:
            # 加载主数据
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.algorithm_names = model_data['algorithms']
            self.weights = model_data['weights']
            self.diversity_weight = model_data['diversity_weight']
            self.popularity_penalty = model_data['popularity_penalty']
            self.item_popularity = model_data['item_popularity']
            self.user_profiles = model_data['user_profiles']
            self.item_features = model_data['item_features']
            self.is_trained = model_data['is_trained']
            
            # 加载各个子模型
            models_dir = os.path.splitext(filepath)[0] + '_models'
            self.models = {}
            
            for algorithm in self.algorithm_names:
                model_path = os.path.join(models_dir, f'{algorithm}_model.pkl')
                if os.path.exists(model_path):
                    try:
                        # 使用算法工厂创建模型
                        model = create_recommender(algorithm)
                        
                        if model is not None:
                            # 尝试加载模型
                            if hasattr(model, 'load_model') and model.load_model(model_path):
                                self.models[algorithm] = model
                            else:
                                # 尝试通用加载方法
                                loaded_model = BaseRecommender.load_model(model_path)
                                if loaded_model:
                                    self.models[algorithm] = loaded_model
                    except Exception as e:
                        logger.warning("加载模型 %s 失败: %s", algorithm, str(e))
            
            logger.info("集成模型已从 %s 加载，包含 %d 个子模型", filepath, len(self.models))
            return True
            
        except Exception as e:
            logger.error("加载集成模型失败: %s", str(e))
            return False


if __name__ == "__main__":
    # 简单测试
    recommender = ExplainableRecommender(['svd'])
    print("集成推荐系统初始化完成")
