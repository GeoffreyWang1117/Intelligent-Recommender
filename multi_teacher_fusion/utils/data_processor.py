"""
Multi-Teacher Fusion System - Data Processor
专用于MovieLens数据集的特征工程和预处理
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import logging
from dataclasses import dataclass

@dataclass
class DataStats:
    """数据统计信息"""
    num_users: int
    num_items: int
    num_ratings: int
    sparsity: float
    rating_range: Tuple[float, float]
    avg_rating: float

class MovieLensProcessor:
    """MovieLens数据处理器"""
    
    def __init__(self, data_path: str, min_ratings_per_user: int = 20, min_ratings_per_item: int = 10):
        """
        初始化数据处理器
        
        Args:
            data_path: MovieLens数据集路径
            min_ratings_per_user: 用户最小评分数量
            min_ratings_per_item: 物品最小评分数量
        """
        self.data_path = Path(data_path)
        self.min_ratings_per_user = min_ratings_per_user
        self.min_ratings_per_item = min_ratings_per_item
        self.logger = logging.getLogger(__name__)
        
        # 数据存储
        self.ratings_df: Optional[pd.DataFrame] = None
        self.movies_df: Optional[pd.DataFrame] = None
        self.users_df: Optional[pd.DataFrame] = None
        
        # 映射表
        self.user_id_map: Optional[Dict[int, int]] = None
        self.item_id_map: Optional[Dict[int, int]] = None
        self.reverse_user_map: Optional[Dict[int, int]] = None
        self.reverse_item_map: Optional[Dict[int, int]] = None
        
        # 统计信息
        self.stats: Optional[DataStats] = None
    
    def load_raw_data(self) -> bool:
        """加载原始MovieLens数据"""
        try:
            # 加载评分数据
            ratings_file = self.data_path / "ratings.csv"
            if ratings_file.exists():
                self.ratings_df = pd.read_csv(ratings_file)
                self.logger.info(f"Loaded {len(self.ratings_df)} ratings")
            else:
                self.logger.error(f"Ratings file not found: {ratings_file}")
                return False
            
            # 加载电影数据
            movies_file = self.data_path / "movies.csv"
            if movies_file.exists():
                self.movies_df = pd.read_csv(movies_file)
                self.logger.info(f"Loaded {len(self.movies_df)} movies")
            else:
                self.logger.warning(f"Movies file not found: {movies_file}")
                self.movies_df = pd.DataFrame()
            
            # 检查数据格式
            required_columns = ['userId', 'movieId', 'rating', 'timestamp']
            if not all(col in self.ratings_df.columns for col in required_columns):
                self.logger.error(f"Missing required columns in ratings data")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return False
    
    def filter_sparse_data(self) -> pd.DataFrame:
        """过滤稀疏数据，保留活跃用户和热门物品"""
        if self.ratings_df is None:
            raise ValueError("数据未加载，请先调用 load_raw_data()")
        
        original_size = len(self.ratings_df)
        filtered_df = self.ratings_df.copy()
        
        # 迭代过滤，直到稳定
        prev_size = 0
        iteration = 0
        
        while len(filtered_df) != prev_size and iteration < 10:
            prev_size = len(filtered_df)
            iteration += 1
            
            # 计算用户评分次数
            user_counts = filtered_df['userId'].value_counts()
            valid_users = user_counts[user_counts >= self.min_ratings_per_user].index
            
            # 计算物品评分次数
            item_counts = filtered_df['movieId'].value_counts()
            valid_items = item_counts[item_counts >= self.min_ratings_per_item].index
            
            # 过滤数据
            filtered_df = filtered_df[
                (filtered_df['userId'].isin(valid_users)) & 
                (filtered_df['movieId'].isin(valid_items))
            ]
            
            self.logger.info(f"Iteration {iteration}: {len(filtered_df)} ratings remaining")
        
        self.logger.info(f"Filtered from {original_size} to {len(filtered_df)} ratings "
                        f"({len(filtered_df)/original_size*100:.1f}% retained)")
        
        return filtered_df
    
    def create_mappings(self, filtered_df: pd.DataFrame) -> None:
        """创建用户和物品的ID映射"""
        # 创建连续的ID映射
        unique_users = sorted(filtered_df['userId'].unique())
        unique_items = sorted(filtered_df['movieId'].unique())
        
        self.user_id_map = {original_id: new_id for new_id, original_id in enumerate(unique_users)}
        self.item_id_map = {original_id: new_id for new_id, original_id in enumerate(unique_items)}
        
        # 创建反向映射
        self.reverse_user_map = {new_id: original_id for original_id, new_id in self.user_id_map.items()}
        self.reverse_item_map = {new_id: original_id for original_id, new_id in self.item_id_map.items()}
        
        self.logger.info(f"Created mappings: {len(self.user_id_map)} users, {len(self.item_id_map)} items")
    
    def apply_mappings(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用ID映射到数据框"""
        if self.user_id_map is None or self.item_id_map is None:
            raise ValueError("映射未创建，请先调用 create_mappings()")
        
        mapped_df = df.copy()
        mapped_df['user_id'] = mapped_df['userId'].map(self.user_id_map)
        mapped_df['item_id'] = mapped_df['movieId'].map(self.item_id_map)
        
        # 移除映射失败的行
        original_size = len(mapped_df)
        mapped_df = mapped_df.dropna(subset=['user_id', 'item_id'])
        mapped_df['user_id'] = mapped_df['user_id'].astype(int)
        mapped_df['item_id'] = mapped_df['item_id'].astype(int)
        
        if len(mapped_df) < original_size:
            self.logger.warning(f"Lost {original_size - len(mapped_df)} rows during mapping")
        
        return mapped_df
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取特征工程"""
        feature_df = df.copy()
        
        # 时间特征
        feature_df['timestamp'] = pd.to_datetime(feature_df['timestamp'], unit='s')
        feature_df['year'] = feature_df['timestamp'].dt.year
        feature_df['month'] = feature_df['timestamp'].dt.month
        feature_df['day_of_week'] = feature_df['timestamp'].dt.dayofweek
        feature_df['hour'] = feature_df['timestamp'].dt.hour
        
        # 用户特征
        user_features = feature_df.groupby('user_id').agg({
            'rating': ['count', 'mean', 'std'],
            'year': ['min', 'max']
        }).round(4)
        
        user_features.columns = [
            'user_rating_count', 'user_avg_rating', 'user_rating_std',
            'user_first_year', 'user_last_year'
        ]
        user_features['user_rating_std'] = user_features['user_rating_std'].fillna(0)
        user_features['user_active_years'] = user_features['user_last_year'] - user_features['user_first_year'] + 1
        
        # 物品特征
        item_features = feature_df.groupby('item_id').agg({
            'rating': ['count', 'mean', 'std'],
            'year': ['min', 'max']
        }).round(4)
        
        item_features.columns = [
            'item_rating_count', 'item_avg_rating', 'item_rating_std',
            'item_first_year', 'item_last_year'
        ]
        item_features['item_rating_std'] = item_features['item_rating_std'].fillna(0)
        item_features['item_active_years'] = item_features['item_last_year'] - item_features['item_first_year'] + 1
        
        # 合并特征
        feature_df = feature_df.merge(user_features, left_on='user_id', right_index=True, how='left')
        feature_df = feature_df.merge(item_features, left_on='item_id', right_index=True, how='left')
        
        # 相对特征
        feature_df['user_item_rating_diff'] = feature_df['rating'] - feature_df['item_avg_rating']
        feature_df['item_user_rating_diff'] = feature_df['rating'] - feature_df['user_avg_rating']
        
        self.logger.info(f"Generated {len(feature_df.columns)} features")
        return feature_df
    
    def calculate_statistics(self, df: pd.DataFrame) -> DataStats:
        """计算数据集统计信息"""
        num_users = df['user_id'].nunique()
        num_items = df['item_id'].nunique()
        num_ratings = len(df)
        sparsity = 1 - (num_ratings / (num_users * num_items))
        rating_range = (df['rating'].min(), df['rating'].max())
        avg_rating = df['rating'].mean()
        
        stats = DataStats(
            num_users=num_users,
            num_items=num_items,
            num_ratings=num_ratings,
            sparsity=sparsity,
            rating_range=rating_range,
            avg_rating=avg_rating
        )
        
        self.logger.info(f"Dataset stats: {num_users} users, {num_items} items, "
                        f"{num_ratings} ratings, {sparsity:.3f} sparsity")
        
        return stats
    
    def train_test_split(self, df: pd.DataFrame, test_ratio: float = 0.2, 
                        time_based: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """划分训练测试集"""
        if time_based:
            # 基于时间划分
            df_sorted = df.sort_values('timestamp')
            split_idx = int(len(df_sorted) * (1 - test_ratio))
            train_df = df_sorted.iloc[:split_idx].copy()
            test_df = df_sorted.iloc[split_idx:].copy()
        else:
            # 随机划分
            test_df = df.sample(frac=test_ratio, random_state=42)
            train_df = df.drop(test_df.index)
        
        self.logger.info(f"Split: {len(train_df)} train, {len(test_df)} test")
        return train_df, test_df
    
    def process_full_pipeline(self) -> Tuple[pd.DataFrame, pd.DataFrame, DataStats]:
        """执行完整的数据处理流程"""
        self.logger.info("开始完整数据处理流程...")
        
        # 1. 加载原始数据
        if not self.load_raw_data():
            raise RuntimeError("数据加载失败")
        
        # 2. 过滤稀疏数据
        filtered_df = self.filter_sparse_data()
        
        # 3. 创建ID映射
        self.create_mappings(filtered_df)
        
        # 4. 应用映射
        mapped_df = self.apply_mappings(filtered_df)
        
        # 5. 特征工程
        feature_df = self.extract_features(mapped_df)
        
        # 6. 计算统计信息
        self.stats = self.calculate_statistics(feature_df)
        
        # 7. 划分训练测试集
        train_df, test_df = self.train_test_split(feature_df)
        
        self.logger.info("数据处理流程完成!")
        return train_df, test_df, self.stats
    
    def save_processed_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                           output_path: str) -> None:
        """保存处理后的数据"""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存数据
        train_df.to_csv(output_dir / "train.csv", index=False)
        test_df.to_csv(output_dir / "test.csv", index=False)
        
        # 保存映射
        mapping_data = {
            'user_id_map': self.user_id_map,
            'item_id_map': self.item_id_map,
            'reverse_user_map': self.reverse_user_map,
            'reverse_item_map': self.reverse_item_map,
            'stats': self.stats.__dict__ if self.stats else None
        }
        
        import json
        with open(output_dir / "mappings.json", 'w') as f:
            json.dump(mapping_data, f, indent=2, default=str)
        
        self.logger.info(f"数据已保存到: {output_dir}")

if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 示例用法
    processor = MovieLensProcessor(
        data_path="/home/coder-gw/7Projects_in_7Days/online-inference-system/data/movielens",
        min_ratings_per_user=20,
        min_ratings_per_item=10
    )
    
    try:
        train_df, test_df, stats = processor.process_full_pipeline()
        print(f"\n处理完成:")
        print(f"训练集: {len(train_df)} 样本")
        print(f"测试集: {len(test_df)} 样本")
        print(f"用户数: {stats.num_users}")
        print(f"物品数: {stats.num_items}")
        print(f"稀疏度: {stats.sparsity:.3f}")
        
        # 保存数据
        processor.save_processed_data(train_df, test_df, 
                                    "/home/coder-gw/7Projects_in_7Days/online-inference-system/multi_teacher_fusion/data")
        
    except Exception as e:
        print(f"处理失败: {e}")
