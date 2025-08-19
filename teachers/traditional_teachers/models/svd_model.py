"""
SVD推荐模型

基于SVD（奇异值分解）的协同过滤推荐算法实现。
该模型使用TruncatedSVD对用户-物品评分矩阵进行降维，
提取用户和物品的潜在特征，并基于这些特征进行推荐。

主要功能：
- 协同过滤推荐
- 冷启动处理（热门物品推荐）
- 相似物品推荐
- 实时评分更新
- 模型持久化

Author: GitHub Copilot
Date: 2025-08-17
"""

import numpy as np
import pandas as pd
import pickle
import logging
from typing import List, Tuple, Dict, Optional, Any
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from .base_recommender import BaseRecommender

logger = logging.getLogger(__name__)
import logging
from typing import List, Tuple, Dict, Any, Optional
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import warnings

# 忽略sklearn的警告信息
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class SVDRecommender(BaseRecommender):
    """
    基于SVD的推荐系统
    
    使用截断奇异值分解(TruncatedSVD)对用户-物品评分矩阵进行降维，
    提取用户和物品的潜在特征向量，用于预测评分和生成推荐。
    
    Attributes:
        n_components (int): SVD降维后的维度数
        model (TruncatedSVD): SVD模型实例
        user_encoder (LabelEncoder): 用户ID编码器
        item_encoder (LabelEncoder): 物品ID编码器
        user_item_matrix (np.ndarray): 用户-物品评分矩阵
        user_features (np.ndarray): 用户特征矩阵
        item_features (np.ndarray): 物品特征矩阵
        global_mean (float): 全局平均评分
        is_trained (bool): 模型是否已训练
        ratings_df (pd.DataFrame): 原始评分数据
    """
    
    def __init__(self, n_components: int = 50, random_state: int = 42, **kwargs):
        """
        初始化SVD推荐器
        
        Args:
            n_components (int): SVD分解的维度，默认50
            random_state (int): 随机种子，默认42
        """
        super().__init__("SVD", n_components=n_components, random_state=random_state, **kwargs)
        self.n_components = min(n_components, 50)  # 限制最大维度
        self.model = TruncatedSVD(n_components=self.n_components, random_state=random_state)
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        
        # 初始化属性
        self.user_item_matrix: Optional[np.ndarray] = None
        self.user_features: Optional[np.ndarray] = None
        self.item_features: Optional[np.ndarray] = None
        self.global_mean: float = 3.0
        self.is_trained: bool = False
        self.ratings_df: Optional[pd.DataFrame] = None
        
    def fit(self, ratings_df: pd.DataFrame) -> None:
        """
        训练SVD推荐模型
        
        Args:
            ratings_df (pd.DataFrame): 评分数据，必须包含列：
                - user_id: 用户ID
                - item_id: 物品ID  
                - rating: 评分值
                - timestamp: 时间戳（可选）
        
        Raises:
            ValueError: 当输入数据格式不正确时
            RuntimeError: 当模型训练失败时
        """
        # 验证输入数据
        required_columns = ['user_id', 'item_id', 'rating']
        missing_columns = [col for col in required_columns if col not in ratings_df.columns]
        if missing_columns:
            raise ValueError(f"缺少必需的列: {missing_columns}")
        
        if ratings_df.empty:
            raise ValueError("评分数据不能为空")
        
        logger.info("开始训练SVD推荐模型...")
        
        try:
            # 保存原始数据副本
            self.ratings_df = ratings_df.copy()
            
            # 编码用户和物品ID
            self.user_encoder.fit(ratings_df['user_id'])
            self.item_encoder.fit(ratings_df['item_id'])
            
            # 创建用户-物品评分矩阵
            self._create_user_item_matrix(ratings_df)
            
            # 计算全局平均评分
            self.global_mean = float(ratings_df['rating'].mean())
            
            n_users = len(self.user_encoder.classes_)
            n_items = len(self.item_encoder.classes_)
            logger.info("数据准备完成: %d 用户, %d 物品", n_users, n_items)
            
            # 训练SVD模型
            logger.info("开始SVD分解...")
            
            # 使用中心化的评分矩阵（减去全局均值）
            centered_matrix = self.user_item_matrix - self.global_mean
            
            # 执行SVD分解
            self.user_features = self.model.fit_transform(centered_matrix)
            self.item_features = self.model.components_.T
            
            self.is_trained = True
            logger.info("SVD模型训练完成")
            
        except Exception as e:
            logger.error("模型训练失败: %s", str(e))
            self.is_trained = False
            raise RuntimeError(f"SVD模型训练失败: {e}") from e
        
    def _create_user_item_matrix(self, ratings_df: pd.DataFrame) -> None:
        """
        创建用户-物品评分矩阵
        
        Args:
            ratings_df (pd.DataFrame): 评分数据
        """
        n_users = len(self.user_encoder.classes_)
        n_items = len(self.item_encoder.classes_)
        
        # 初始化矩阵（用0填充，表示未评分）
        self.user_item_matrix = np.zeros((n_users, n_items))
        
        # 填充已知评分
        for _, row in ratings_df.iterrows():
            user_idx = self.user_encoder.transform([row['user_id']])[0]
            item_idx = self.item_encoder.transform([row['item_id']])[0]
            self.user_item_matrix[user_idx, item_idx] = row['rating']
    
    def predict(self, user_id: int, item_id: int) -> float:
        """
        预测用户对物品的评分
        
        Args:
            user_id (int): 用户ID
            item_id (int): 物品ID
            
        Returns:
            float: 预测评分，范围[1, 5]
        """
        if not self.is_trained:
            logger.warning("模型尚未训练，返回全局平均评分")
            return self.global_mean
            
        try:
            # 检查用户是否存在
            if user_id not in self.user_encoder.classes_:
                logger.debug("用户 %d 不在训练数据中，返回全局平均评分", user_id)
                return self.global_mean
            
            # 检查物品是否存在
            if item_id not in self.item_encoder.classes_:
                logger.debug("物品 %d 不在训练数据中，返回全局平均评分", item_id)
                return self.global_mean
            
            # 获取编码后的索引
            user_idx = self.user_encoder.transform([user_id])[0]
            item_idx = self.item_encoder.transform([item_id])[0]
            
            # 计算预测评分 = 全局均值 + 用户特征 · 物品特征
            prediction = self.global_mean + np.dot(
                self.user_features[user_idx], 
                self.item_features[item_idx]
            )
            
            # 限制评分范围在[1, 5]
            return float(max(1.0, min(5.0, prediction)))
            
        except (IndexError, ValueError, TypeError) as e:
            logger.error("预测失败 (user=%d, item=%d): %s", user_id, item_id, str(e))
            return self.global_mean
    
    def get_user_recommendations(self, user_id: int, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        获取用户推荐列表
        
        Args:
            user_id (int): 用户ID
            top_k (int): 推荐物品数量，默认10
            
        Returns:
            List[Dict[str, Any]]: 推荐列表，每个元素为{'item_id': int, 'score': float}
        """
        if not self.is_trained:
            logger.warning("模型尚未训练")
            return []
            
        try:
            # 冷启动处理：用户不在训练数据中
            if user_id not in self.user_encoder.classes_:
                logger.info("用户 %d 为新用户，返回热门物品推荐", user_id)
                return self._get_popular_items(top_k)
            
            # 获取用户已评分的物品
            rated_items = set()
            if hasattr(self, 'ratings_df') and self.ratings_df is not None:
                user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]
                if not user_ratings.empty:
                    rated_items = set(user_ratings['item_id'].values)
            
            # 为所有未评分物品计算预测评分
            recommendations = []
            for item_id in self.item_encoder.classes_:
                if item_id not in rated_items:  # 只推荐未评分的物品
                    score = self.predict(user_id, item_id)
                    recommendations.append((int(item_id), float(score)))
            
            # 按评分降序排序并返回top-k
            recommendations.sort(key=lambda x: x[1], reverse=True)
            return self._format_recommendations(recommendations[:top_k])
            
        except (ValueError, TypeError, KeyError) as e:
            logger.error("获取推荐失败 (user=%d): %s", user_id, str(e))
            return self._format_recommendations(self._get_popular_items(top_k))
    
    def _get_popular_items(self, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        获取热门物品（用于冷启动推荐）
        
        基于物品的平均评分和评分次数计算热门度评分
        
        Args:
            top_k (int): 返回热门物品数量
            
        Returns:
            List[Tuple[int, float]]: 热门物品列表，每个元素为(物品ID, 热门度评分)
        """
        try:
            if self.ratings_df is None or self.ratings_df.empty:
                logger.warning("无评分数据，无法计算热门物品")
                return []
            
            # 计算每个物品的统计信息
            item_stats = self.ratings_df.groupby('item_id').agg({
                'rating': ['mean', 'count']
            }).reset_index()
            
            item_stats.columns = ['item_id', 'avg_rating', 'rating_count']
            
            # 使用加权评分：平均评分 * log(1 + 评分次数)
            # 这样既考虑了评分质量，也考虑了评分数量
            item_stats['weighted_score'] = (
                item_stats['avg_rating'] * np.log(1 + item_stats['rating_count'])
            )
            
            # 按加权评分降序排序
            popular_items = item_stats.nlargest(top_k, 'weighted_score')
            
            return [
                (int(row['item_id']), float(row['weighted_score']))
                for _, row in popular_items.iterrows()
            ]
            
        except (ValueError, TypeError, KeyError) as e:
            logger.error("获取热门物品失败: %s", str(e))
            return []
    
    def get_item_recommendations(self, user_id: int, k: int = 10) -> List[Dict[str, Any]]:
        """
        获取物品推荐（兼容接口）
        
        返回格式化的推荐结果，包含物品信息
        
        Args:
            user_id (int): 用户ID
            k (int): 推荐数量
            
        Returns:
            List[Dict[str, Any]]: 格式化的推荐列表
        """
        try:
            # 获取基础推荐
            recommendations = self.get_user_recommendations(user_id, k)
            
            # 转换为字典格式，添加物品信息
            results = []
            for item_id, score in recommendations:
                results.append({
                    'item_id': int(item_id),
                    'score': float(score),
                    'title': f'Movie {item_id}',  # 简化的标题
                    'genres': ['Drama']  # 默认类型
                })
            
            return results
            
        except (ValueError, TypeError) as e:
            logger.error("获取推荐失败: %s", str(e))
            return []
    
    def get_similar_items(self, item_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        获取相似物品
        
        基于物品特征向量的余弦相似度计算相似物品
        
        Args:
            item_id (int): 目标物品ID
            top_k (int): 返回相似物品数量
            
        Returns:
            List[Tuple[int, float]]: 相似物品列表，每个元素为(物品ID, 相似度)
        """
        if not self.is_trained:
            logger.warning("模型尚未训练")
            return []
            
        try:
            # 检查物品是否存在
            if item_id not in self.item_encoder.classes_:
                logger.warning("物品 %d 不在训练数据中", item_id)
                return []
            
            item_idx = self.item_encoder.transform([item_id])[0]
            
            # 计算物品间的余弦相似度
            item_vector = self.item_features[item_idx].reshape(1, -1)
            similarities = cosine_similarity(item_vector, self.item_features)[0]
            
            # 获取最相似的物品（排除自身）
            similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
            
            similar_items = []
            for idx in similar_indices:
                similar_item_id = self.item_encoder.classes_[idx]
                similarity = similarities[idx]
                similar_items.append((int(similar_item_id), float(similarity)))
            
            return similar_items
            
        except (IndexError, ValueError, TypeError) as e:
            logger.error("获取相似物品失败 (item=%d): %s", item_id, str(e))
            return []

    def get_item_embeddings(self, item_ids: List[int]) -> Optional[np.ndarray]:
        """
        获取物品的嵌入向量
        
        Args:
            item_ids: 物品ID列表
            
        Returns:
            Optional[np.ndarray]: 物品嵌入向量矩阵，形状为(n_items, n_components)
        """
        try:
            if not self.is_trained or self.item_features is None:
                logger.warning("模型未训练，无法获取物品嵌入向量")
                return None
            
            embeddings = []
            for item_id in item_ids:
                if item_id in self.item_encoder.classes_:
                    item_idx = self.item_encoder.transform([item_id])[0]
                    embedding = self.item_features[item_idx]
                    embeddings.append(embedding)
                else:
                    logger.warning("物品不存在: %d", item_id)
                    # 使用零向量
                    embedding = np.zeros(self.n_components)
                    embeddings.append(embedding)
            
            return np.array(embeddings) if embeddings else None
            
        except Exception as e:
            logger.error("获取物品嵌入向量失败: %s", str(e))
            return None
    
    def update_rating(self, user_id: int, item_id: int, rating: float) -> bool:
        """
        更新用户评分
        
        Args:
            user_id (int): 用户ID
            item_id (int): 物品ID
            rating (float): 新评分
            
        Returns:
            bool: 更新是否成功
        """
        try:
            # 验证评分范围
            if not 1 <= rating <= 5:
                logger.warning("评分值 %f 超出范围[1,5]", rating)
                return False
            
            # 创建新评分记录
            new_rating = pd.DataFrame({
                'user_id': [user_id],
                'item_id': [item_id], 
                'rating': [rating],
                'timestamp': [pd.Timestamp.now().timestamp()]
            })
            
            # 添加到评分数据中
            if self.ratings_df is not None:
                self.ratings_df = pd.concat([self.ratings_df, new_rating], ignore_index=True)
            else:
                self.ratings_df = new_rating
            
            # 检查是否需要重新训练（新用户或新物品）
            if (user_id not in self.user_encoder.classes_ or 
                item_id not in self.item_encoder.classes_):
                logger.info("检测到新用户或新物品，建议重新训练模型")
                return False  # 需要重新训练
            
            # 更新评分矩阵
            user_idx = self.user_encoder.transform([user_id])[0]
            item_idx = self.item_encoder.transform([item_id])[0]
            self.user_item_matrix[user_idx, item_idx] = rating
            
            return True
            
        except (ValueError, TypeError, IndexError) as e:
            logger.error("更新评分失败: %s", str(e))
            return False
    
    def save_model(self, filepath: str) -> bool:
        """
        保存模型到文件
        
        Args:
            filepath (str): 保存文件路径
            
        Returns:
            bool: 保存是否成功
        """
        try:
            # 组装所有模型数据
            model_data = {
                'model': self.model,
                'user_encoder': self.user_encoder,
                'item_encoder': self.item_encoder,
                'user_item_matrix': self.user_item_matrix,
                'user_features': self.user_features,
                'item_features': self.item_features,
                'global_mean': self.global_mean,
                'ratings_df': self.ratings_df,
                'is_trained': self.is_trained,
                'n_components': self.n_components
            }
            
            # 保存到文件
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info("模型已保存到: %s", filepath)
            return True
            
        except (IOError, pickle.PickleError, TypeError) as e:
            logger.error("保存模型失败: %s", str(e))
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        从文件加载模型
        
        Args:
            filepath (str): 模型文件路径
            
        Returns:
            bool: 加载是否成功
        """
        try:
            # 从文件加载数据
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # 恢复所有属性
            self.model = model_data['model']
            self.user_encoder = model_data['user_encoder']
            self.item_encoder = model_data['item_encoder']
            self.user_item_matrix = model_data['user_item_matrix']
            self.user_features = model_data['user_features']
            self.item_features = model_data['item_features']
            self.global_mean = model_data['global_mean']
            self.ratings_df = model_data['ratings_df']
            self.is_trained = model_data['is_trained']
            self.n_components = model_data['n_components']
            
            logger.info("模型已从 %s 加载", filepath)
            return True
            
        except (IOError, pickle.PickleError, KeyError, TypeError) as e:
            logger.error("加载模型失败: %s", str(e))
            return False
