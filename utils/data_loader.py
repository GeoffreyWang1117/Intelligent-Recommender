import pandas as pd
import numpy as np
import os
import logging
from typing import Tuple, Dict, Optional

logger = logging.getLogger(__name__)

class MovieLensLoader:
    """MovieLens数据加载器"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.ratings = None
        self.movies = None
        self.users = None
    
    def load_ratings(self) -> pd.DataFrame:
        """加载评分数据"""
        try:
            ratings_file = os.path.join(self.data_path, 'ratings.dat')
            if not os.path.exists(ratings_file):
                raise FileNotFoundError(f"评分文件不存在: {ratings_file}")
            
            # MovieLens 1M格式: UserID::MovieID::Rating::Timestamp
            self.ratings = pd.read_csv(
                ratings_file,
                sep='::',
                names=['user_id', 'item_id', 'rating', 'timestamp'],
                engine='python'
            )
            
            logger.info(f"加载评分数据: {len(self.ratings)} 条记录")
            return self.ratings
            
        except Exception as e:
            logger.error(f"加载评分数据失败: {e}")
            raise
    
    def load_movies(self) -> pd.DataFrame:
        """加载电影数据"""
        try:
            movies_file = os.path.join(self.data_path, 'movies.dat')
            if not os.path.exists(movies_file):
                raise FileNotFoundError(f"电影文件不存在: {movies_file}")
            
            # MovieLens 1M格式: MovieID::Title::Genres
            self.movies = pd.read_csv(
                movies_file,
                sep='::',
                names=['item_id', 'title', 'genres'],
                engine='python',
                encoding='iso-8859-1'
            )
            
            logger.info(f"加载电影数据: {len(self.movies)} 部电影")
            return self.movies
            
        except Exception as e:
            logger.error(f"加载电影数据失败: {e}")
            raise
    
    def load_users(self) -> pd.DataFrame:
        """加载用户数据"""
        try:
            users_file = os.path.join(self.data_path, 'users.dat')
            if not os.path.exists(users_file):
                raise FileNotFoundError(f"用户文件不存在: {users_file}")
            
            # MovieLens 1M格式: UserID::Gender::Age::Occupation::Zip-code
            self.users = pd.read_csv(
                users_file,
                sep='::',
                names=['user_id', 'gender', 'age', 'occupation', 'zipcode'],
                engine='python'
            )
            
            logger.info(f"加载用户数据: {len(self.users)} 个用户")
            return self.users
            
        except Exception as e:
            logger.error(f"加载用户数据失败: {e}")
            raise
    
    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """加载所有数据"""
        ratings = self.load_ratings()
        movies = self.load_movies()
        users = self.load_users()
        return ratings, movies, users
    
    def get_dataset_info(self) -> Dict:
        """获取数据集信息"""
        if self.ratings is None:
            self.load_ratings()
        if self.movies is None:
            self.load_movies()
        if self.users is None:
            self.load_users()
        
        info = {
            'num_ratings': len(self.ratings),
            'num_users': self.ratings['user_id'].nunique(),
            'num_items': self.ratings['item_id'].nunique(),
            'num_movies': len(self.movies),
            'rating_scale': (self.ratings['rating'].min(), self.ratings['rating'].max()),
            'sparsity': 1 - len(self.ratings) / (self.ratings['user_id'].nunique() * self.ratings['item_id'].nunique()),
            'avg_rating': self.ratings['rating'].mean(),
            'ratings_per_user': len(self.ratings) / self.ratings['user_id'].nunique(),
            'ratings_per_item': len(self.ratings) / self.ratings['item_id'].nunique()
        }
        
        return info
    
    def create_item_metadata(self) -> Dict:
        """创建物品元数据字典"""
        if self.movies is None:
            self.load_movies()
        
        metadata = {}
        for _, row in self.movies.iterrows():
            metadata[row['item_id']] = {
                'title': row['title'],
                'genres': row['genres'].split('|') if pd.notna(row['genres']) else []
            }
        
        return metadata

def download_movielens_1m(data_path: str) -> bool:
    """下载MovieLens 1M数据集"""
    try:
        import urllib.request
        import zipfile
        
        # 创建数据目录
        os.makedirs(data_path, exist_ok=True)
        
        # 检查是否已经存在
        ratings_file = os.path.join(data_path, 'ratings.dat')
        if os.path.exists(ratings_file):
            logger.info("MovieLens 1M数据集已存在")
            return True
        
        # 下载URL
        url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
        zip_path = os.path.join(data_path, "ml-1m.zip")
        
        logger.info("开始下载MovieLens 1M数据集...")
        urllib.request.urlretrieve(url, zip_path)
        
        # 解压
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_path)
        
        # 移动文件到正确位置
        ml_1m_path = os.path.join(data_path, 'ml-1m')
        if os.path.exists(ml_1m_path):
            import shutil
            for filename in os.listdir(ml_1m_path):
                shutil.move(
                    os.path.join(ml_1m_path, filename),
                    os.path.join(data_path, filename)
                )
            os.rmdir(ml_1m_path)
        
        # 删除zip文件
        os.remove(zip_path)
        
        logger.info("MovieLens 1M数据集下载完成")
        return True
        
    except Exception as e:
        logger.error(f"下载MovieLens 1M数据集失败: {e}")
        return False

def split_train_test(ratings: pd.DataFrame, test_ratio: float = 0.2, 
                    random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """按用户分割训练测试集"""
    try:
        np.random.seed(random_state)
        
        train_data = []
        test_data = []
        
        for user_id, user_ratings in ratings.groupby('user_id'):
            user_ratings = user_ratings.sort_values('timestamp')
            n_ratings = len(user_ratings)
            
            if n_ratings >= 5:  # 确保每个用户至少有5个评分
                n_test = max(1, int(n_ratings * test_ratio))
                test_indices = np.random.choice(n_ratings, n_test, replace=False)
                
                for i, (_, rating) in enumerate(user_ratings.iterrows()):
                    if i in test_indices:
                        test_data.append(rating)
                    else:
                        train_data.append(rating)
            else:
                # 评分太少的用户全部放入训练集
                for _, rating in user_ratings.iterrows():
                    train_data.append(rating)
        
        train_df = pd.DataFrame(train_data)
        test_df = pd.DataFrame(test_data)
        
        logger.info(f"数据分割完成 - 训练集: {len(train_df)}, 测试集: {len(test_df)}")
        
        return train_df, test_df
        
    except Exception as e:
        logger.error(f"数据分割失败: {e}")
        raise

def create_sample_data(n_users: int = 500, n_items: int = 200, n_ratings: int = 5000) -> pd.DataFrame:
    """创建示例推荐数据"""
    np.random.seed(42)
    
    # 生成用户ID和物品ID
    user_ids = np.random.randint(1, n_users + 1, n_ratings)
    item_ids = np.random.randint(1, n_items + 1, n_ratings)
    
    # 生成评分 (1-5)
    ratings = np.random.choice([1, 2, 3, 4, 5], n_ratings, p=[0.05, 0.1, 0.3, 0.35, 0.2])
    
    # 生成时间戳
    timestamps = np.random.randint(946684800, 1640995200, n_ratings)  # 2000-2022
    
    # 创建DataFrame
    ratings_df = pd.DataFrame({
        'user_id': user_ids,
        'item_id': item_ids,
        'rating': ratings,
        'timestamp': timestamps
    })
    
    # 去重
    ratings_df = ratings_df.drop_duplicates(subset=['user_id', 'item_id'])
    
    # 保存额外的数据文件
    data_dir = 'data/sample'
    os.makedirs(data_dir, exist_ok=True)
    
    # 创建电影数据
    movies_data = []
    genres_list = ['Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime',
                   'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
                   'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    
    for item_id in range(1, n_items + 1):
        title = f"Movie {item_id} ({2000 + item_id % 23})"
        selected_genres = np.random.choice(genres_list, np.random.randint(1, 4), replace=False)
        genres = '|'.join(selected_genres)
        movies_data.append({'item_id': item_id, 'title': title, 'genres': genres})
    
    movies_df = pd.DataFrame(movies_data)
    movies_df.to_csv(f'{data_dir}/movies.csv', index=False)
    
    # 创建用户数据
    users_data = []
    genders = ['M', 'F']
    ages = [1, 18, 25, 35, 45, 50, 56]
    occupations = list(range(21))  # 0-20
    
    for user_id in range(1, n_users + 1):
        gender = np.random.choice(genders)
        age = np.random.choice(ages)
        occupation = np.random.choice(occupations)
        zipcode = f"{np.random.randint(10000, 99999)}"
        users_data.append({
            'user_id': user_id, 
            'gender': gender, 
            'age': age, 
            'occupation': occupation, 
            'zipcode': zipcode
        })
    
    users_df = pd.DataFrame(users_data)
    users_df.to_csv(f'{data_dir}/users.csv', index=False)
    
    return ratings_df
