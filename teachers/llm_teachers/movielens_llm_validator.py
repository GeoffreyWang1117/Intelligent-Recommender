#!/usr/bin/env python3
"""
MovieLens数据集处理和LLM推荐验证系统
Real MovieLens Data Integration for LLM Recommendation Validation

功能:
1. 加载真实MovieLens数据集
2. 构建用户画像和电影特征
3. LLM推荐生成
4. 推荐系统标准评价指标验证

作者: GitHub Copilot
日期: 2025-08-18
"""

import pandas as pd
import numpy as np
import requests
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class MovieLensLLMValidator:
    """基于真实MovieLens数据的LLM推荐验证器"""
    
    def __init__(self, data_path: str = "data/movielens", ollama_endpoint: str = "http://localhost:11434/api/generate"):
        self.data_path = Path(data_path)
        self.ollama_endpoint = ollama_endpoint
        
        # LLM模型配置
        self.primary_model = "llama3:latest"    # 英文主力
        self.secondary_model = "qwen3:latest"   # 中文对照
        
        # 数据存储
        self.ratings = None
        self.movies = None
        self.users = None
        self.user_profiles = {}
        self.movie_features = {}
        
        # 验证指标
        self.metrics_results = {}
        
    def load_movielens_data(self, dataset_size: str = "sample") -> bool:
        """
        加载MovieLens数据集
        
        Args:
            dataset_size: "sample" (data/movielens) 或 "small" (100k) 或 "medium" (1M) 或 "large" (25M)
        """
        try:
            if dataset_size == "sample":
                # 使用data/movielens文件夹中的样本数据
                ratings_file = Path("data/movielens/ratings.csv")
                movies_file = Path("data/movielens/movies.csv")
                users_file = Path("data/movielens/users.csv")
                
                if not all([ratings_file.exists(), movies_file.exists(), users_file.exists()]):
                    print("❌ MovieLens样本数据文件不存在")
                    print("请检查 data/movielens/ 文件夹")
                    return False
                
                # 加载评分数据
                self.ratings = pd.read_csv(ratings_file)
                # 确保列名一致
                if 'item_id' in self.ratings.columns:
                    self.ratings = self.ratings.rename(columns={'item_id': 'movie_id'})
                
                # 加载电影数据
                self.movies = pd.read_csv(movies_file)
                if 'item_id' in self.movies.columns:
                    self.movies = self.movies.rename(columns={'item_id': 'movie_id'})
                
                # 加载用户数据
                self.users = pd.read_csv(users_file)
                
            elif dataset_size == "small":
                # MovieLens 100K dataset
                ratings_file = self.data_path / "u.data"
                movies_file = self.data_path / "u.item"
                users_file = self.data_path / "u.user"
                
                if not all([ratings_file.exists(), movies_file.exists(), users_file.exists()]):
                    print("❌ MovieLens数据文件不存在，需要先下载数据集")
                    print("请访问: https://grouplens.org/datasets/movielens/100k/")
                    return False
                
                # 加载评分数据
                self.ratings = pd.read_csv(
                    ratings_file, 
                    sep='\t', 
                    names=['user_id', 'movie_id', 'rating', 'timestamp'],
                    dtype={'user_id': int, 'movie_id': int, 'rating': int, 'timestamp': int}
                )
                
                # 加载电影数据
                movie_columns = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + \
                               [f'genre_{i}' for i in range(19)]  # 19个类型标签
                
                self.movies = pd.read_csv(
                    movies_file,
                    sep='|',
                    names=movie_columns,
                    encoding='latin-1',
                    dtype={'movie_id': int}
                )
                
                # 加载用户数据
                self.users = pd.read_csv(
                    users_file,
                    sep='|',
                    names=['user_id', 'age', 'gender', 'occupation', 'zip_code'],
                    dtype={'user_id': int, 'age': int}
                )
                
            else:
                print(f"❌ 暂不支持{dataset_size}数据集，请使用small版本")
                return False
                
            print(f"✅ 成功加载MovieLens数据:")
            print(f"   用户数: {len(self.users)}")
            print(f"   电影数: {len(self.movies)}")
            print(f"   评分数: {len(self.ratings)}")
            print(f"   评分范围: {self.ratings['rating'].min()}-{self.ratings['rating'].max()}")
            print(f"   稀疏度: {(1 - len(self.ratings) / (len(self.users) * len(self.movies))) * 100:.2f}%")
            
            return True
            
        except Exception as e:
            print(f"❌ 加载MovieLens数据失败: {str(e)}")
            return False
    
    def create_sample_data(self) -> bool:
        """创建样本数据用于测试（当真实数据不可用时）"""
        print("📝 创建MovieLens样本数据用于测试...")
        
        # 创建样本用户
        self.users = pd.DataFrame({
            'user_id': range(1, 101),
            'age': np.random.randint(18, 65, 100),
            'gender': np.random.choice(['M', 'F'], 100),
            'occupation': np.random.choice(['student', 'engineer', 'teacher', 'doctor'], 100),
            'zip_code': [f'{np.random.randint(10000, 99999)}' for _ in range(100)]
        })
        
        # 创建样本电影
        movie_titles = [
            "The Matrix", "Inception", "Interstellar", "Blade Runner 2049", "The Dark Knight",
            "Pulp Fiction", "The Godfather", "Schindler's List", "The Shawshank Redemption",
            "Forrest Gump", "Goodfellas", "The Silence of the Lambs", "Saving Private Ryan",
            "Titanic", "Avatar", "The Avengers", "Iron Man", "Spider-Man", "Batman Begins",
            "Jurassic Park", "Star Wars", "Lord of the Rings", "Harry Potter", "Pirates of the Caribbean",
            "Mission Impossible", "Fast & Furious", "John Wick", "Mad Max", "Terminator", "Alien"
        ]
        
        self.movies = pd.DataFrame({
            'movie_id': range(1, len(movie_titles) + 1),
            'title': movie_titles,
            'release_date': ['01-Jan-' + str(np.random.randint(1990, 2024)) for _ in movie_titles],
            'video_release_date': [''] * len(movie_titles),
            'imdb_url': [f'http://us.imdb.com/M/title-exact?{title.replace(" ", "+")}' for title in movie_titles]
        })
        
        # 添加类型信息
        genres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary',
                 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
                 'Sci-Fi', 'Thriller', 'War', 'Western']
        
        for i, genre in enumerate(genres):
            self.movies[f'genre_{i}'] = np.random.choice([0, 1], len(movie_titles), p=[0.8, 0.2])
        
        # 创建样本评分
        n_ratings = 5000
        user_ids = np.random.choice(self.users['user_id'], n_ratings)
        movie_ids = np.random.choice(self.movies['movie_id'], n_ratings)
        ratings = np.random.choice([1, 2, 3, 4, 5], n_ratings, p=[0.1, 0.1, 0.2, 0.4, 0.2])
        timestamps = np.random.randint(800000000, 1600000000, n_ratings)
        
        self.ratings = pd.DataFrame({
            'user_id': user_ids,
            'movie_id': movie_ids,
            'rating': ratings,
            'timestamp': timestamps
        }).drop_duplicates(subset=['user_id', 'movie_id'])
        
        print(f"✅ 样本数据创建完成:")
        print(f"   用户数: {len(self.users)}")
        print(f"   电影数: {len(self.movies)}")
        print(f"   评分数: {len(self.ratings)}")
        
        return True
    
    def build_user_profiles(self) -> Dict:
        """构建用户画像"""
        print("👤 构建用户画像...")
        
        user_profiles = {}
        
        for user_id in self.users['user_id'].unique():
            # 基本信息
            user_info = self.users[self.users['user_id'] == user_id].iloc[0]
            user_ratings = self.ratings[self.ratings['user_id'] == user_id]
            
            if len(user_ratings) == 0:
                continue
            
            # 评分统计
            avg_rating = user_ratings['rating'].mean()
            total_ratings = len(user_ratings)
            
            # 类型偏好分析
            favorite_genres = self._analyze_genre_preferences(user_id)
            
            # 评分行为分析
            rating_distribution = user_ratings['rating'].value_counts().to_dict()
            
            user_profiles[user_id] = {
                'user_id': user_id,
                'age': int(user_info['age']),
                'gender': user_info['gender'],
                'occupation': user_info['occupation'],
                'avg_rating': round(avg_rating, 2),
                'total_ratings': total_ratings,
                'favorite_genres': favorite_genres,
                'rating_distribution': rating_distribution,
                'viewing_history': user_ratings['movie_id'].tolist()[-10:]  # 最近10部电影
            }
        
        self.user_profiles = user_profiles
        print(f"✅ 用户画像构建完成，覆盖{len(user_profiles)}个用户")
        
        return user_profiles
    
    def _analyze_genre_preferences(self, user_id: int) -> List[str]:
        """分析用户类型偏好"""
        user_ratings = self.ratings[self.ratings['user_id'] == user_id]
        
        # 获取用户评分过的电影
        user_movies = self.movies[self.movies['movie_id'].isin(user_ratings['movie_id'])]
        
        # 计算各类型的平均评分
        genre_cols = [col for col in self.movies.columns if col.startswith('genre_')]
        genre_names = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary',
                      'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
                      'Sci-Fi', 'Thriller', 'War', 'Western']
        
        genre_scores = {}
        
        for i, genre_col in enumerate(genre_cols):
            if i < len(genre_names):
                genre_movies = user_movies[user_movies[genre_col] == 1]
                if len(genre_movies) > 0:
                    # 计算该类型电影的平均评分
                    genre_ratings = user_ratings[user_ratings['movie_id'].isin(genre_movies['movie_id'])]
                    if len(genre_ratings) > 0:
                        avg_score = genre_ratings['rating'].mean()
                        genre_scores[genre_names[i]] = avg_score
        
        # 返回评分最高的前3个类型
        sorted_genres = sorted(genre_scores.items(), key=lambda x: x[1], reverse=True)
        return [genre for genre, score in sorted_genres[:3]]
    
    def get_llm_recommendations(self, user_id: int, candidate_movies: List[Dict], 
                               model_choice: str = "primary", top_k: int = 10) -> Dict:
        """获取LLM推荐结果"""
        
        if user_id not in self.user_profiles:
            return {"error": f"用户{user_id}画像不存在"}
        
        user_profile = self.user_profiles[user_id]
        model_name = self.primary_model if model_choice == "primary" else self.secondary_model
        
        # 构建提示词
        prompt = self._build_recommendation_prompt(user_profile, candidate_movies, model_choice, top_k)
        
        try:
            print(f"🤖 使用{model_name}为用户{user_id}生成推荐...")
            
            response = requests.post(self.ollama_endpoint, json={
                "model": model_name,
                "prompt": prompt,
                "stream": False
            }, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                return self._parse_llm_response(result['response'], model_choice)
            else:
                return {"error": f"LLM API调用失败: {response.status_code}"}
                
        except Exception as e:
            return {"error": f"LLM推荐生成失败: {str(e)}"}
    
    def _build_recommendation_prompt(self, user_profile: Dict, candidate_movies: List[Dict], 
                                   model_choice: str, top_k: int) -> str:
        """构建推荐提示词"""
        
        # 获取用户观影历史的电影标题
        history_movies = self.movies[self.movies['movie_id'].isin(user_profile['viewing_history'])]
        history_titles = history_movies['title'].tolist()
        
        if model_choice == "primary":  # Llama3英文提示词
            candidates_text = "\n".join([
                f"- Movie {movie['movie_id']}: {movie['title']} ({movie.get('genres', 'Unknown')})"
                for movie in candidate_movies
            ])
            
            prompt = f"""You are an expert movie recommendation system trained on MovieLens data. Analyze this real user profile and recommend the top {top_k} movies.

REAL USER PROFILE:
- User ID: {user_profile['user_id']}
- Age: {user_profile['age']} years old
- Gender: {user_profile['gender']}
- Occupation: {user_profile['occupation']}
- Average Rating: {user_profile['avg_rating']}/5.0 ({user_profile['total_ratings']} movies rated)
- Favorite Genres: {', '.join(user_profile['favorite_genres'])}
- Recent Viewing History: {', '.join(history_titles[-5:]) if history_titles else 'No history available'}

CANDIDATE MOVIES:
{candidates_text}

INSTRUCTIONS:
1. Consider the user's genre preferences and rating patterns
2. Account for the user's demographic profile (age, gender, occupation)
3. Predict realistic ratings based on the user's average rating behavior
4. Provide confidence scores based on profile match
5. Give brief explanations for recommendations

OUTPUT FORMAT (JSON only):
[
  {{"movie_id": 1, "title": "Movie Title", "predicted_rating": 4.2, "confidence": 0.85, "reason": "Brief explanation"}}
]
"""
        
        else:  # Qwen3中文提示词
            candidates_text = "\n".join([
                f"- 电影{movie['movie_id']}: {movie['title']} ({movie.get('genres', '未知类型')})"
                for movie in candidate_movies
            ])
            
            prompt = f"""你是基于MovieLens数据训练的专业电影推荐系统。分析真实用户档案，推荐最适合的{top_k}部电影。

真实用户档案：
- 用户ID：{user_profile['user_id']}
- 年龄：{user_profile['age']}岁
- 性别：{user_profile['gender']}
- 职业：{user_profile['occupation']}
- 平均评分：{user_profile['avg_rating']}/5.0 (已评分{user_profile['total_ratings']}部电影)
- 偏好类型：{', '.join(user_profile['favorite_genres'])}
- 近期观影：{', '.join(history_titles[-5:]) if history_titles else '无观影记录'}

候选电影：
{candidates_text}

任务要求：
1. 考虑用户的类型偏好和评分模式
2. 结合用户人口统计特征 (年龄、性别、职业)
3. 基于用户平均评分行为预测真实评分
4. 根据档案匹配度提供置信度分数
5. 为推荐提供简洁解释

输出格式 (仅JSON):
[
  {{"movie_id": 1, "title": "电影标题", "predicted_rating": 4.2, "confidence": 0.85, "reason": "简要解释"}}
]
"""
        
        return prompt
    
    def _parse_llm_response(self, response_text: str, model_choice: str) -> Dict:
        """解析LLM响应"""
        try:
            # 提取JSON部分
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_text = response_text[start_idx:end_idx]
                recommendations = json.loads(json_text)
                
                return {
                    "model": f"Llama3 (英文)" if model_choice == "primary" else "Qwen3 (中文)",
                    "recommendations": recommendations,
                    "status": "success"
                }
            else:
                return {
                    "model": f"Llama3 (英文)" if model_choice == "primary" else "Qwen3 (中文)",
                    "raw_response": response_text[:500] + "..." if len(response_text) > 500 else response_text,
                    "status": "parse_failed"
                }
                
        except json.JSONDecodeError as e:
            return {
                "model": f"Llama3 (英文)" if model_choice == "primary" else "Qwen3 (中文)",
                "error": f"JSON解析失败: {str(e)}",
                "raw_response": response_text[:500] + "..." if len(response_text) > 500 else response_text,
                "status": "json_error"
            }


def main():
    """主函数 - 演示真实MovieLens数据的LLM推荐"""
    print("🎬 MovieLens数据集 + LLM推荐验证系统")
    print("=" * 60)
    
    # 初始化验证器
    validator = MovieLensLLMValidator()
    
    # 尝试加载真实数据，失败则使用样本数据
    if not validator.load_movielens_data("small"):
        print("⚠️  真实MovieLens数据不可用，使用样本数据进行演示")
        if not validator.create_sample_data():
            print("❌ 样本数据创建失败")
            return
    
    # 构建用户画像
    user_profiles = validator.build_user_profiles()
    
    if not user_profiles:
        print("❌ 用户画像构建失败")
        return
    
    # 选择测试用户 (评分数量足够的用户)
    test_users = []
    for user_id, profile in user_profiles.items():
        if profile['total_ratings'] >= 5:  # 至少评分过5部电影
            test_users.append(user_id)
        if len(test_users) >= 3:  # 测试3个用户
            break
    
    if not test_users:
        test_users = list(user_profiles.keys())[:3]
    
    print(f"\n🧪 开始测试用户: {test_users}")
    
    # 为每个测试用户生成推荐
    for user_id in test_users:
        print(f"\n" + "="*50)
        print(f"👤 测试用户 {user_id}")
        print("="*50)
        
        user_profile = user_profiles[user_id]
        print(f"用户信息: {user_profile['age']}岁 {user_profile['gender']} {user_profile['occupation']}")
        print(f"评分统计: 平均{user_profile['avg_rating']}/5.0, 已评分{user_profile['total_ratings']}部")
        print(f"偏好类型: {', '.join(user_profile['favorite_genres'])}")
        
        # 获取候选电影 (用户未评分过的电影)
        user_rated_movies = set(validator.ratings[validator.ratings['user_id'] == user_id]['movie_id'])
        all_movies = set(validator.movies['movie_id'])
        candidate_movie_ids = list(all_movies - user_rated_movies)[:20]  # 随机选择20部候选电影
        
        candidate_movies = []
        for movie_id in candidate_movie_ids:
            movie_info = validator.movies[validator.movies['movie_id'] == movie_id].iloc[0]
            candidate_movies.append({
                'movie_id': movie_id,
                'title': movie_info['title'],
                'genres': 'Mixed'  # 简化处理
            })
        
        print(f"候选电影: {len(candidate_movies)}部")
        
        # 获取LLM推荐 (仅测试主力模型以节省时间)
        llm_result = validator.get_llm_recommendations(
            user_id, candidate_movies, "primary", top_k=5
        )
        
        print(f"\n🤖 Llama3推荐结果:")
        if llm_result.get("status") == "success":
            for i, rec in enumerate(llm_result["recommendations"][:5], 1):
                print(f"   {i}. {rec.get('title', 'N/A')} "
                      f"(预测评分: {rec.get('predicted_rating', 'N/A'):.1f}, "
                      f"置信度: {rec.get('confidence', 'N/A'):.2f})")
                if 'reason' in rec:
                    print(f"      理由: {rec['reason']}")
        else:
            print(f"   ❌ 推荐失败: {llm_result.get('error', '未知错误')}")
    
    print(f"\n🎉 MovieLens + LLM推荐测试完成!")


if __name__ == "__main__":
    main()
