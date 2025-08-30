"""
Real data processing for MovieLens dataset integration with LayerwiseAdapter.
"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.base import UserProfile, ItemProfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MovieLensDataProcessor:
    """MovieLens数据处理器，为LayerwiseAdapter准备真实数据"""
    
    def __init__(self, data_path: str, embedding_dim: int = 128):
        self.data_path = Path(data_path)
        self.embedding_dim = embedding_dim
        
        # 编码器
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.genre_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        # 数据存储
        self.users_df: Optional[pd.DataFrame] = None
        self.movies_df: Optional[pd.DataFrame] = None
        self.ratings_df: Optional[pd.DataFrame] = None
        
        # 特征映射
        self.user_features = {}
        self.item_features = {}
        self.genre_list = []
        
        logger.info(f"MovieLensDataProcessor initialized with embedding_dim={embedding_dim}")
    
    def load_data(self) -> None:
        """加载MovieLens数据文件"""
        logger.info(f"Loading data from {self.data_path}")
        
        try:
            # 加载ratings数据
            ratings_file = self.data_path / "ratings.csv"
            self.ratings_df = pd.read_csv(ratings_file)
            
            # 加载movies数据
            movies_file = self.data_path / "movies.csv"
            self.movies_df = pd.read_csv(movies_file)
            
            # 尝试加载users数据（如果存在）
            users_file = self.data_path / "users.csv"
            if users_file.exists():
                self.users_df = pd.read_csv(users_file)
            else:
                # 从ratings数据创建基础用户信息
                unique_users = self.ratings_df['userId'].unique()
                self.users_df = pd.DataFrame({
                    'userId': unique_users,
                    'age': np.random.randint(18, 65, len(unique_users)),
                    'gender': np.random.choice(['M', 'F'], len(unique_users)),
                    'occupation': np.random.randint(0, 20, len(unique_users))
                })
            
            logger.info(f"Loaded {len(self.ratings_df)} ratings, {len(self.movies_df)} movies, {len(self.users_df)} users")
            
            # 重新映射用户和物品ID以确保连续性（修复CUDA索引问题）
            self._remap_ids()
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _remap_ids(self) -> None:
        """重新映射用户和物品ID，确保从0开始连续编号"""
        logger.info("Remapping user and item IDs...")
        
        # 获取唯一的用户和物品ID
        unique_users = sorted(self.ratings_df['userId'].unique())
        unique_movies = sorted(self.ratings_df['movieId'].unique())
        
        # 创建映射字典：原始ID -> 新ID (0, 1, 2, ...)
        self.user_id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_users)}
        self.movie_id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_movies)}
        
        # 反向映射：新ID -> 原始ID
        self.reverse_user_mapping = {new_id: old_id for old_id, new_id in self.user_id_mapping.items()}
        self.reverse_movie_mapping = {new_id: old_id for old_id, new_id in self.movie_id_mapping.items()}
        
        # 应用映射到ratings数据
        self.ratings_df['userId'] = self.ratings_df['userId'].map(self.user_id_mapping)
        self.ratings_df['movieId'] = self.ratings_df['movieId'].map(self.movie_id_mapping)
        
        # 应用映射到用户数据
        self.users_df['userId'] = self.users_df['userId'].map(self.user_id_mapping)
        
        # 应用映射到电影数据
        self.movies_df['movieId'] = self.movies_df['movieId'].map(self.movie_id_mapping)
        
        # 过滤掉映射失败的数据
        self.ratings_df = self.ratings_df.dropna()
        self.users_df = self.users_df.dropna()
        self.movies_df = self.movies_df.dropna()
        
        logger.info(f"Remapped IDs: {len(unique_users)} users (0-{len(unique_users)-1}), "
                   f"{len(unique_movies)} movies (0-{len(unique_movies)-1})")
        
        # 存储最大ID以供后续使用
        self.max_user_id = len(unique_users) - 1
        self.max_movie_id = len(unique_movies) - 1
    
    def _generate_user_info(self) -> None:
        """从评分数据生成用户信息"""
        user_stats = self.ratings_df.groupby('userId').agg({
            'rating': ['count', 'mean', 'std'],
            'movieId': 'nunique'
        }).round(3)
        
        user_stats.columns = ['rating_count', 'rating_mean', 'rating_std', 'movie_count']
        user_stats['rating_std'] = user_stats['rating_std'].fillna(0)
        
        self.users_df = user_stats.reset_index()
        
        # 添加用户活跃度等级
        self.users_df['activity_level'] = pd.cut(
            self.users_df['rating_count'], 
            bins=5, 
            labels=['low', 'medium_low', 'medium', 'medium_high', 'high']
        ).astype(str)
    
    def _preprocess_data(self) -> None:
        """数据预处理"""
        logger.info("Preprocessing data...")
        
        # 处理电影类型
        self.movies_df['genres_list'] = self.movies_df['genres'].str.split('|')
        all_genres = []
        for genres in self.movies_df['genres_list']:
            all_genres.extend(genres)
        self.genre_list = list(set(all_genres))
        
        # 创建类型向量
        self.movies_df['genre_vector'] = self.movies_df['genres_list'].apply(
            lambda x: self._create_genre_vector(x)
        )
        
        # 编码用户和物品ID
        self.ratings_df['user_encoded'] = self.user_encoder.fit_transform(self.ratings_df['userId'])
        self.ratings_df['item_encoded'] = self.item_encoder.fit_transform(self.ratings_df['movieId'])
        
        # 计算统计特征
        self._compute_statistical_features()
        
        logger.info("Data preprocessing completed")
    
    def _create_genre_vector(self, genres: List[str]) -> np.ndarray:
        """创建类型向量"""
        vector = np.zeros(len(self.genre_list))
        for genre in genres:
            if genre in self.genre_list:
                idx = self.genre_list.index(genre)
                vector[idx] = 1
        return vector
    
    def _compute_statistical_features(self) -> None:
        """计算统计特征"""
        # 用户统计
        user_stats = self.ratings_df.groupby('userId').agg({
            'rating': ['count', 'mean', 'std'],
            'movieId': 'nunique'
        })
        user_stats.columns = ['_'.join(col) for col in user_stats.columns]
        user_stats = user_stats.fillna(0)
        
        # 物品统计
        item_stats = self.ratings_df.groupby('movieId').agg({
            'rating': ['count', 'mean', 'std'],
            'userId': 'nunique'
        })
        item_stats.columns = ['_'.join(col) for col in item_stats.columns]
        item_stats = item_stats.fillna(0)
        
        # 合并统计信息
        self.users_df = self.users_df.merge(user_stats, on='userId', how='left')
        self.movies_df = self.movies_df.merge(item_stats, on='movieId', how='left')
        
        # 标准化数值特征
        numeric_user_cols = [col for col in self.users_df.columns if col.endswith('_count') or col.endswith('_mean') or col.endswith('_std')]
        if numeric_user_cols:
            self.users_df[numeric_user_cols] = self.scaler.fit_transform(self.users_df[numeric_user_cols])
        
        numeric_item_cols = [col for col in self.movies_df.columns if col.endswith('_count') or col.endswith('_mean') or col.endswith('_std')]
        if numeric_item_cols:
            self.movies_df[numeric_item_cols] = self.scaler.fit_transform(self.movies_df[numeric_item_cols])
    
    def create_user_profiles(self) -> Dict[int, UserProfile]:
        """创建用户Profile对象"""
        logger.info("Creating user profiles...")
        
        user_profiles = {}
        
        for _, user_row in self.users_df.iterrows():
            user_id = int(user_row['userId'])
            
            # 创建用户特征向量
            feature_vector = self._create_user_feature_vector(user_row)
            
            # 获取用户历史交互
            user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
            interaction_history = user_ratings['movieId'].tolist()[:50]  # 限制历史长度
            
            # 创建UserProfile
            user_profile = UserProfile(
                user_id=user_id,
                feature_vector=torch.tensor(feature_vector, dtype=torch.float32),
                interaction_history=interaction_history,
                metadata={
                    'rating_count': user_ratings.shape[0],
                    'avg_rating': float(user_ratings['rating'].mean()),
                    'activity_level': getattr(user_row, 'activity_level', 'medium')
                }
            )
            
            user_profiles[user_id] = user_profile
        
        logger.info(f"Created {len(user_profiles)} user profiles")
        return user_profiles
    
    def create_item_profiles(self) -> Dict[int, ItemProfile]:
        """创建物品Profile对象"""
        logger.info("Creating item profiles...")
        
        item_profiles = {}
        
        for _, movie_row in self.movies_df.iterrows():
            movie_id = int(movie_row['movieId'])
            
            # 创建物品特征向量
            feature_vector = self._create_item_feature_vector(movie_row)
            
            # 获取主要类型
            genres = movie_row['genres'].split('|')
            main_category = genres[0] if genres else 'Unknown'
            
            # 创建ItemProfile
            item_profile = ItemProfile(
                item_id=movie_id,
                feature_vector=torch.tensor(feature_vector, dtype=torch.float32),
                category=main_category,
                semantic_embedding=torch.tensor(movie_row['genre_vector'], dtype=torch.float32),
                metadata={
                    'title': movie_row['title'],
                    'genres': movie_row['genres'],
                    'genre_count': len(genres),
                    'avg_rating': float(self.ratings_df[self.ratings_df['movieId'] == movie_id]['rating'].mean()) if movie_id in self.ratings_df['movieId'].values else 3.0
                }
            )
            
            item_profiles[movie_id] = item_profile
        
        logger.info(f"Created {len(item_profiles)} item profiles")
        return item_profiles
    
    def _create_user_feature_vector(self, user_row: pd.Series) -> np.ndarray:
        """创建用户特征向量"""
        features = []
        
        # 基础统计特征
        if 'rating_count' in user_row:
            features.extend([
                user_row.get('rating_count', 0),
                user_row.get('rating_mean', 3.0),
                user_row.get('rating_std', 0),
                user_row.get('movie_count', 0)
            ])
        else:
            # 使用原始统计特征
            features.extend([
                user_row.get('rating_rating_count', 0),
                user_row.get('rating_rating_mean', 3.0),
                user_row.get('rating_rating_std', 0),
                user_row.get('movieId_movieId_nunique', 0)
            ])
        
        # 活跃度编码
        activity_map = {'low': 0.2, 'medium_low': 0.4, 'medium': 0.6, 'medium_high': 0.8, 'high': 1.0}
        activity_score = activity_map.get(user_row.get('activity_level', 'medium'), 0.6)
        features.append(activity_score)
        
        # 填充到目标维度
        current_dim = len(features)
        if current_dim < self.embedding_dim:
            # 用随机噪声填充剩余维度
            remaining = np.random.normal(0, 0.1, self.embedding_dim - current_dim)
            features.extend(remaining)
        
        return np.array(features[:self.embedding_dim])
    
    def _create_item_feature_vector(self, movie_row: pd.Series) -> np.ndarray:
        """创建物品特征向量"""
        features = []
        
        # 基础统计特征
        features.extend([
            movie_row.get('rating_rating_count', 0),
            movie_row.get('rating_rating_mean', 3.0),
            movie_row.get('rating_rating_std', 0),
            movie_row.get('userId_userId_nunique', 0)
        ])
        
        # 类型特征 (取前20个维度)
        genre_vector = movie_row['genre_vector']
        if len(genre_vector) >= 20:
            features.extend(genre_vector[:20])
        else:
            features.extend(genre_vector)
            features.extend([0] * (20 - len(genre_vector)))
        
        # 类型数量
        features.append(len(movie_row['genres_list']))
        
        # 填充到目标维度
        current_dim = len(features)
        if current_dim < self.embedding_dim:
            remaining = np.random.normal(0, 0.1, self.embedding_dim - current_dim)
            features.extend(remaining)
        
        return np.array(features[:self.embedding_dim])
    
    def create_training_data(self, user_profiles: Dict[int, UserProfile], 
                           item_profiles: Dict[int, ItemProfile],
                           test_size: float = 0.2,
                           min_interactions: int = 5) -> Tuple[List, List, List, List, List, List]:
        """创建训练数据"""
        logger.info("Creating training data...")
        
        # 过滤低交互用户
        user_counts = self.ratings_df['userId'].value_counts()
        valid_users = user_counts[user_counts >= min_interactions].index
        
        filtered_ratings = self.ratings_df[self.ratings_df['userId'].isin(valid_users)].copy()
        logger.info(f"Filtered to {len(filtered_ratings)} ratings from {len(valid_users)} users")
        
        # 准备训练数据
        train_users, train_items, train_ratings = [], [], []
        test_users, test_items, test_ratings = [], [], []
        
        for user_id in valid_users:
            if user_id not in user_profiles:
                continue
                
            user_ratings = filtered_ratings[filtered_ratings['userId'] == user_id]
            
            # 按时间排序 (如果有timestamp)
            if 'timestamp' in user_ratings.columns:
                user_ratings = user_ratings.sort_values('timestamp')
            
            # 分割训练/测试
            n_ratings = len(user_ratings)
            n_test = max(1, int(n_ratings * test_size))
            
            train_rating_subset = user_ratings.iloc[:-n_test]
            test_rating_subset = user_ratings.iloc[-n_test:]
            
            # 添加训练数据
            for _, rating_row in train_rating_subset.iterrows():
                movie_id = int(rating_row['movieId'])
                if movie_id in item_profiles:
                    train_users.append(user_profiles[user_id])
                    train_items.append(item_profiles[movie_id])
                    train_ratings.append(float(rating_row['rating']))
            
            # 添加测试数据
            for _, rating_row in test_rating_subset.iterrows():
                movie_id = int(rating_row['movieId'])
                if movie_id in item_profiles:
                    test_users.append(user_profiles[user_id])
                    test_items.append(item_profiles[movie_id])
                    test_ratings.append(float(rating_row['rating']))
        
        logger.info(f"Created training data: {len(train_users)} train, {len(test_users)} test")
        
        return train_users, train_items, train_ratings, test_users, test_items, test_ratings
    
    def save_processed_data(self, save_path: str) -> None:
        """保存处理后的数据"""
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True)
        
        # 保存编码器
        with open(save_path / "encoders.pkl", "wb") as f:
            pickle.dump({
                'user_encoder': self.user_encoder,
                'item_encoder': self.item_encoder,
                'genre_encoder': self.genre_encoder,
                'scaler': self.scaler,
                'genre_list': self.genre_list
            }, f)
        
        # 保存处理后的数据
        self.users_df.to_csv(save_path / "processed_users.csv", index=False)
        self.movies_df.to_csv(save_path / "processed_movies.csv", index=False)
        self.ratings_df.to_csv(save_path / "processed_ratings.csv", index=False)
        
        logger.info(f"Processed data saved to {save_path}")


def load_movielens_data(data_path: str, embedding_dim: int = 128) -> Tuple[Dict, Dict, Tuple]:
    """便捷函数：加载并处理MovieLens数据"""
    processor = MovieLensDataProcessor(data_path, embedding_dim)
    processor.load_data()
    
    user_profiles = processor.create_user_profiles()
    item_profiles = processor.create_item_profiles()
    
    training_data = processor.create_training_data(user_profiles, item_profiles)
    
    return user_profiles, item_profiles, training_data


if __name__ == "__main__":
    # 测试数据处理
    data_path = "/home/coder-gw/7Projects_in_7Days/online-inference-system/data/movielens/1m"
    
    processor = MovieLensDataProcessor(data_path, embedding_dim=128)
    processor.load_data()
    
    user_profiles = processor.create_user_profiles()
    item_profiles = processor.create_item_profiles()
    
    print(f"Created {len(user_profiles)} user profiles")
    print(f"Created {len(item_profiles)} item profiles")
    
    # 示例用户和物品
    sample_user = list(user_profiles.values())[0]
    sample_item = list(item_profiles.values())[0]
    
    print(f"\nSample user: {sample_user.user_id}")
    print(f"Feature shape: {sample_user.feature_vector.shape}")
    print(f"Interactions: {len(sample_user.interaction_history)}")
    
    print(f"\nSample item: {sample_item.item_id}")
    print(f"Feature shape: {sample_item.feature_vector.shape}")
    print(f"Category: {sample_item.category}")
    print(f"Title: {sample_item.metadata['title']}")
