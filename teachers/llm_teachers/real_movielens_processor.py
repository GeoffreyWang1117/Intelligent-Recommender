#!/usr/bin/env python3
"""
真实MovieLens数据下载和高性能预处理系统
使用Trie、哈希表等高级数据结构优化数据管理和检索

功能:
1. 下载真实MovieLens数据集
2. 使用Trie结构管理电影标题和类型
3. 构建高效的用户画像索引
4. 实现快速相似度计算和推荐生成

作者: GitHub Copilot
日期: 2025-08-27
"""

import os
import sys
import requests
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import defaultdict, deque
import pickle
import json
from dataclasses import dataclass, field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MovieNode:
    """电影节点数据结构"""
    movie_id: int
    title: str
    year: int
    genres: List[str]
    imdb_id: Optional[str] = None
    tmdb_id: Optional[str] = None
    
    def __hash__(self):
        return hash(self.movie_id)

@dataclass 
class UserProfile:
    """用户画像数据结构"""
    user_id: int
    rated_movies: Dict[int, float] = field(default_factory=dict)  # movie_id -> rating
    preferred_genres: Dict[str, float] = field(default_factory=dict)  # genre -> preference_score
    avg_rating: float = 0.0
    rating_count: int = 0
    rating_variance: float = 0.0
    
    def add_rating(self, movie_id: int, rating: float, genres: List[str]):
        """添加评分并更新用户画像"""
        self.rated_movies[movie_id] = rating
        
        # 更新类型偏好
        for genre in genres:
            if genre not in self.preferred_genres:
                self.preferred_genres[genre] = 0.0
            self.preferred_genres[genre] += rating
            
        # 更新统计信息
        self._update_stats()
    
    def _update_stats(self):
        """更新用户统计信息"""
        if self.rated_movies:
            ratings = list(self.rated_movies.values())
            self.avg_rating = float(np.mean(ratings))
            self.rating_count = len(ratings)
            self.rating_variance = float(np.var(ratings))
            
            # 标准化类型偏好
            for genre in self.preferred_genres:
                self.preferred_genres[genre] /= self.rating_count

class TrieNode:
    """Trie树节点"""
    def __init__(self):
        self.children: Dict[str, 'TrieNode'] = {}
        self.is_end_word: bool = False
        self.movie_ids: Set[int] = set()  # 存储匹配的电影ID
        self.frequency: int = 0

class MovieTrieIndex:
    """基于Trie的电影索引系统"""
    
    def __init__(self):
        self.root = TrieNode()
        self.movie_nodes: Dict[int, MovieNode] = {}
        self.genre_index: Dict[str, Set[int]] = defaultdict(set)
        self.year_index: Dict[int, Set[int]] = defaultdict(set)
        
    def insert_movie(self, movie: MovieNode):
        """插入电影到Trie索引"""
        self.movie_nodes[movie.movie_id] = movie
        
        # 插入标题到Trie
        self._insert_title(movie.title.lower(), movie.movie_id)
        
        # 构建类型索引
        for genre in movie.genres:
            self.genre_index[genre.lower()].add(movie.movie_id)
            
        # 构建年份索引
        self.year_index[movie.year].add(movie.movie_id)
        
    def _insert_title(self, title: str, movie_id: int):
        """将电影标题插入Trie树"""
        # 插入完整标题
        words = title.replace('(', '').replace(')', '').split()
        for word in words:
            if len(word) > 2:  # 过滤短词
                self._insert_word(word, movie_id)
                
    def _insert_word(self, word: str, movie_id: int):
        """插入单词到Trie"""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.movie_ids.add(movie_id)
            
        node.is_end_word = True
        node.frequency += 1
        
    def search_movies(self, query: str, limit: int = 10) -> List[MovieNode]:
        """搜索电影"""
        query = query.lower().strip()
        candidate_ids = set()
        
        # 按词搜索
        words = query.split()
        for word in words:
            if len(word) > 2:
                ids = self._search_word(word)
                if candidate_ids:
                    candidate_ids &= ids  # 交集
                else:
                    candidate_ids = ids
                    
        # 转换为电影对象并排序
        movies = [self.movie_nodes[mid] for mid in candidate_ids if mid in self.movie_nodes]
        
        # 按相关性排序（简单的字符串匹配）
        movies.sort(key=lambda m: self._relevance_score(m.title.lower(), query), reverse=True)
        
        return movies[:limit]
        
    def _search_word(self, word: str) -> Set[int]:
        """搜索单词"""
        node = self.root
        for char in word:
            if char not in node.children:
                return set()
            node = node.children[char]
        return node.movie_ids.copy()
        
    def _relevance_score(self, title: str, query: str) -> float:
        """计算相关性评分"""
        score = 0.0
        query_words = set(query.split())
        title_words = set(title.split())
        
        # 精确匹配加分
        exact_matches = len(query_words & title_words)
        score += exact_matches * 2
        
        # 部分匹配加分
        for qw in query_words:
            for tw in title_words:
                if qw in tw or tw in qw:
                    score += 0.5
                    
        return score
        
    def get_movies_by_genre(self, genre: str) -> List[MovieNode]:
        """按类型获取电影"""
        movie_ids = self.genre_index.get(genre.lower(), set())
        return [self.movie_nodes[mid] for mid in movie_ids if mid in self.movie_nodes]
        
    def get_movies_by_year_range(self, start_year: int, end_year: int) -> List[MovieNode]:
        """按年份范围获取电影"""
        movie_ids = set()
        for year in range(start_year, end_year + 1):
            movie_ids.update(self.year_index.get(year, set()))
        return [self.movie_nodes[mid] for mid in movie_ids if mid in self.movie_nodes]

class UserProfileManager:
    """用户画像管理器"""
    
    def __init__(self):
        self.profiles: Dict[int, UserProfile] = {}
        self.genre_similarity_cache: Dict[Tuple[int, int], float] = {}
        self.rating_matrix = None
        self.user_similarity_cache: Dict[Tuple[int, int], float] = {}
        
    def add_user_rating(self, user_id: int, movie: MovieNode, rating: float):
        """添加用户评分"""
        if user_id not in self.profiles:
            self.profiles[user_id] = UserProfile(user_id=user_id)
            
        self.profiles[user_id].add_rating(movie.movie_id, rating, movie.genres)
        
    def get_user_profile(self, user_id: int) -> Optional[UserProfile]:
        """获取用户画像"""
        return self.profiles.get(user_id)
        
    def compute_user_similarity(self, user1_id: int, user2_id: int) -> float:
        """计算用户相似度（使用缓存）"""
        cache_key = (min(user1_id, user2_id), max(user1_id, user2_id))
        
        if cache_key in self.user_similarity_cache:
            return self.user_similarity_cache[cache_key]
            
        profile1 = self.profiles.get(user1_id)
        profile2 = self.profiles.get(user2_id)
        
        if not profile1 or not profile2:
            return 0.0
            
        # 计算共同评分电影的余弦相似度
        common_movies = set(profile1.rated_movies.keys()) & set(profile2.rated_movies.keys())
        
        if len(common_movies) < 2:
            similarity = 0.0
        else:
            ratings1 = [profile1.rated_movies[mid] for mid in common_movies]
            ratings2 = [profile2.rated_movies[mid] for mid in common_movies]
            
            # 余弦相似度
            similarity = np.dot(ratings1, ratings2) / (np.linalg.norm(ratings1) * np.linalg.norm(ratings2))
            
        self.user_similarity_cache[cache_key] = similarity
        return similarity
        
    def get_similar_users(self, user_id: int, k: int = 20) -> List[Tuple[int, float]]:
        """获取相似用户"""
        if user_id not in self.profiles:
            return []
            
        similarities = []
        for other_id in self.profiles:
            if other_id != user_id:
                sim = self.compute_user_similarity(user_id, other_id)
                if sim > 0:
                    similarities.append((other_id, sim))
                    
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
        
    def recommend_for_user(self, user_id: int, movie_index: MovieTrieIndex, k: int = 10) -> List[Tuple[MovieNode, float]]:
        """为用户推荐电影"""
        if user_id not in self.profiles:
            return []
            
        profile = self.profiles[user_id]
        similar_users = self.get_similar_users(user_id, k=20)
        
        # 基于协同过滤的推荐
        candidate_scores = defaultdict(float)
        
        for similar_user_id, similarity in similar_users:
            similar_profile = self.profiles[similar_user_id]
            
            for movie_id, rating in similar_profile.rated_movies.items():
                if movie_id not in profile.rated_movies:  # 用户未评分的电影
                    candidate_scores[movie_id] += similarity * rating
                    
        # 基于内容的推荐（类型偏好）
        for genre, preference in profile.preferred_genres.items():
            if preference > profile.avg_rating:  # 喜欢的类型
                genre_movies = movie_index.get_movies_by_genre(genre)
                for movie in genre_movies:
                    if movie.movie_id not in profile.rated_movies:
                        candidate_scores[movie.movie_id] += preference * 0.5
                        
        # 转换为电影对象并排序
        recommendations = []
        for movie_id, score in candidate_scores.items():
            if movie_id in movie_index.movie_nodes:
                movie = movie_index.movie_nodes[movie_id]
                recommendations.append((movie, score))
                
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:k]

class RealMovieLensProcessor:
    """真实MovieLens数据处理器"""
    
    def __init__(self, data_dir: str = "data/movielens"):
        self.data_dir = Path(data_dir)
        self.movie_index = MovieTrieIndex()
        self.user_manager = UserProfileManager()
        
        # 数据URL
        self.ml_25m_url = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
        self.ml_latest_small_url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
        
    def download_movielens_data(self, dataset: str = "small") -> bool:
        """下载真实MovieLens数据"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        if dataset == "small":
            url = self.ml_latest_small_url
            zip_name = "ml-latest-small.zip"
            extract_folder = "ml-latest-small"
        elif dataset == "25m":
            url = self.ml_25m_url
            zip_name = "ml-25m.zip" 
            extract_folder = "ml-25m"
        else:
            logger.error(f"不支持的数据集: {dataset}")
            return False
            
        zip_path = self.data_dir / zip_name
        
        # 检查是否已存在
        expected_files = ["movies.csv", "ratings.csv", "tags.csv", "links.csv"]
        if all((self.data_dir / f).exists() for f in expected_files):
            logger.info("MovieLens数据已存在，跳过下载")
            return True
            
        try:
            logger.info(f"正在下载 {dataset} MovieLens数据集...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        if total_size > 0:
                            progress = downloaded_size / total_size * 100
                            print(f"\r下载进度: {progress:.1f}%", end="", flush=True)
                            
            print()  # 换行
            logger.info("下载完成，正在解压...")
            
            # 解压文件
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
                
            # 移动文件到根目录
            extract_path = self.data_dir / extract_folder
            if extract_path.exists():
                for file in extract_path.glob("*.csv"):
                    file.rename(self.data_dir / file.name)
                    
            # 清理
            zip_path.unlink()
            if extract_path.exists():
                import shutil
                shutil.rmtree(extract_path)
                
            logger.info("MovieLens数据下载和处理完成")
            return True
            
        except Exception as e:
            logger.error(f"下载失败: {str(e)}")
            return False
    
    def load_and_process_data(self, sample_size: Optional[int] = None) -> bool:
        """加载和处理数据"""
        try:
            # 加载电影数据
            movies_file = self.data_dir / "movies.csv"
            if not movies_file.exists():
                logger.error("movies.csv 不存在")
                return False
                
            logger.info("正在加载电影数据...")
            movies_df = pd.read_csv(movies_file)
            
            # 处理电影数据
            for _, row in movies_df.iterrows():
                # 解析标题和年份
                title = row['title']
                year = 2000  # 默认值
                
                # 提取年份
                if '(' in title and ')' in title:
                    try:
                        year_str = title.split('(')[-1].split(')')[0]
                        if year_str.isdigit() and len(year_str) == 4:
                            year = int(year_str)
                            title = title.rsplit('(', 1)[0].strip()
                    except (IndexError, ValueError, AttributeError):
                        # 无法解析年份，使用默认值
                        pass
                        
                # 解析类型
                genres = []
                if pd.notna(row['genres']) and row['genres'] != '(no genres listed)':
                    genres = [g.strip() for g in row['genres'].split('|')]
                    
                # 创建电影节点
                movie = MovieNode(
                    movie_id=row['movieId'],
                    title=title,
                    year=year,
                    genres=genres
                )
                
                self.movie_index.insert_movie(movie)
                
            logger.info(f"已加载 {len(self.movie_index.movie_nodes)} 部电影")
            
            # 加载评分数据
            ratings_file = self.data_dir / "ratings.csv"
            if not ratings_file.exists():
                logger.error("ratings.csv 不存在")
                return False
                
            logger.info("正在加载评分数据...")
            
            # 如果指定了样本大小，只加载部分数据
            if sample_size:
                ratings_df = pd.read_csv(ratings_file, nrows=sample_size)
            else:
                ratings_df = pd.read_csv(ratings_file)
                
            # 处理评分数据
            processed_count = 0
            for _, row in ratings_df.iterrows():
                movie_id = row['movieId']
                if movie_id in self.movie_index.movie_nodes:
                    movie = self.movie_index.movie_nodes[movie_id]
                    self.user_manager.add_user_rating(
                        user_id=row['userId'],
                        movie=movie,
                        rating=row['rating']
                    )
                    processed_count += 1
                    
                if processed_count % 10000 == 0:
                    logger.info(f"已处理 {processed_count} 条评分")
                    
            logger.info(f"已处理 {processed_count} 条评分，{len(self.user_manager.profiles)} 个用户")
            return True
            
        except Exception as e:
            logger.error(f"数据加载失败: {str(e)}")
            return False
    
    def save_processed_data(self, cache_file: str = "movielens_cache.pkl"):
        """保存处理后的数据"""
        cache_path = self.data_dir / cache_file
        
        data = {
            'movie_index': self.movie_index,
            'user_manager': self.user_manager
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
            
        logger.info(f"数据已缓存到 {cache_path}")
        
    def load_processed_data(self, cache_file: str = "movielens_cache.pkl") -> bool:
        """加载处理后的数据"""
        cache_path = self.data_dir / cache_file
        
        if not cache_path.exists():
            return False
            
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
                
            self.movie_index = data['movie_index']
            self.user_manager = data['user_manager']
            
            logger.info("已从缓存加载数据")
            return True
            
        except Exception as e:
            logger.error(f"缓存加载失败: {str(e)}")
            return False
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        stats = {
            'total_movies': len(self.movie_index.movie_nodes),
            'total_users': len(self.user_manager.profiles),
            'total_ratings': sum(len(profile.rated_movies) for profile in self.user_manager.profiles.values()),
            'genres': list(self.movie_index.genre_index.keys()),
            'year_range': (
                min(movie.year for movie in self.movie_index.movie_nodes.values()),
                max(movie.year for movie in self.movie_index.movie_nodes.values())
            ) if self.movie_index.movie_nodes else (0, 0),
            'avg_ratings_per_user': np.mean([len(profile.rated_movies) for profile in self.user_manager.profiles.values()]) if self.user_manager.profiles else 0,
            'sparsity': 1.0 - (sum(len(profile.rated_movies) for profile in self.user_manager.profiles.values()) / 
                              (len(self.user_manager.profiles) * len(self.movie_index.movie_nodes))) if self.movie_index.movie_nodes and self.user_manager.profiles else 0
        }
        
        return stats

def demo_real_movielens_processing():
    """演示真实MovieLens数据处理"""
    print("🎬 真实MovieLens数据处理演示")
    print("=" * 50)
    
    processor = RealMovieLensProcessor()
    
    # 尝试加载缓存
    if not processor.load_processed_data():
        print("📥 下载并处理真实MovieLens数据...")
        
        # 下载数据
        if not processor.download_movielens_data("small"):
            print("❌ 数据下载失败")
            return
            
        # 处理数据（使用样本以加快演示）
        if not processor.load_and_process_data(sample_size=50000):
            print("❌ 数据处理失败")
            return
            
        # 保存缓存
        processor.save_processed_data()
    
    # 展示统计信息
    stats = processor.get_dataset_stats()
    print("\n📊 数据集统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 演示Trie搜索
    print("\n🔍 电影搜索演示:")
    search_queries = ["Matrix", "Star Wars", "Godfather"]
    
    for query in search_queries:
        movies = processor.movie_index.search_movies(query, limit=3)
        print(f"\n搜索 '{query}':")
        for movie in movies:
            print(f"  - {movie.title} ({movie.year}) [{', '.join(movie.genres)}]")
    
    # 演示推荐
    print("\n🎯 推荐演示:")
    if processor.user_manager.profiles:
        sample_user_id = list(processor.user_manager.profiles.keys())[0]
        profile = processor.user_manager.get_user_profile(sample_user_id)
        
        if profile:
            print(f"\n用户 {sample_user_id} 的画像:")
            print(f"  评分数量: {profile.rating_count}")
            print(f"  平均评分: {profile.avg_rating:.2f}")
            print(f"  喜欢的类型: {sorted(profile.preferred_genres.items(), key=lambda x: x[1], reverse=True)[:3]}")
            
            recommendations = processor.user_manager.recommend_for_user(
                sample_user_id, processor.movie_index, k=5
            )
            
            print(f"\n为用户 {sample_user_id} 推荐:")
            for movie, score in recommendations:
                print(f"  - {movie.title} ({movie.year}) [评分: {score:.2f}]")

if __name__ == "__main__":
    demo_real_movielens_processing()
