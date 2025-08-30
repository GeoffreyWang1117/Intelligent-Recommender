#!/usr/bin/env python3
"""
Real MovieLens LLM Recommender - 基于真实数据的LLM推荐系统
Real Data-Only LLM Recommendation System with Fisher Information & PAKD Support

功能:
1. 仅使用真实MovieLens数据（禁止模拟数据）
2. Ollama LLM推荐生成（Llama3 + Qwen3）
3. 支持Fisher Information分析
4. 支持PAKD（Pruning-Aware Knowledge Distillation）
5. 完整的推荐系统评估

作者: GitHub Copilot
日期: 2025-08-29
"""

import pandas as pd
import numpy as np
import requests
import json
import time
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class RealMovieLensLLMRecommender:
    """基于真实MovieLens数据的LLM推荐系统（支持Fisher+PAKD）"""
    
    def __init__(self, data_path: str = "../../data/movielens", ollama_endpoint: str = "http://localhost:11434/api/generate"):
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
        
        # Fisher Information & PAKD相关
        self.fisher_calculator = None
        self.pakd_distiller = None
        self.llm_embeddings = {}
        
        # 验证指标
        self.metrics_results = {}
        
        print("🎬 Real MovieLens LLM Recommender 初始化完成")
        print("🔍 支持Fisher Information分析和PAKD")
        print("⚠️  仅支持真实数据，不支持模拟数据")
    
    def load_real_movielens_data(self, dataset_size: str = "small") -> bool:
        """
        加载真实MovieLens数据集（严格禁止模拟数据）
        
        Args:
            dataset_size: "small" (100k), "100k", "1m", "10m", "25m"
        
        Returns:
            bool: 加载成功返回True，失败返回False
        """
        print(f"📊 开始加载真实MovieLens数据 (size: {dataset_size})...")
        
        # 支持的真实数据集路径
        dataset_paths = {
            "small": self.data_path / "small",
            "100k": self.data_path / "100k", 
            "1m": self.data_path / "1m",
            "10m": self.data_path / "10m",
            "25m": self.data_path / "25m"
        }
        
        if dataset_size not in dataset_paths:
            print(f"❌ 不支持的数据集大小: {dataset_size}")
            print(f"📋 支持的数据集: {list(dataset_paths.keys())}")
            return False
        
        data_dir = dataset_paths[dataset_size]
        if not data_dir.exists():
            print(f"❌ 数据集目录不存在: {data_dir}")
            print(f"💡 请下载真实MovieLens数据集到: {data_dir}")
            return False
        
        try:
            # 加载评分数据
            ratings_file = data_dir / "ratings.csv"
            if not ratings_file.exists():
                print(f"❌ 评分文件不存在: {ratings_file}")
                return False
            
            self.ratings = pd.read_csv(ratings_file)
            if self.ratings.empty:
                print("❌ 评分数据为空")
                return False
            
            # 加载电影数据
            movies_file = data_dir / "movies.csv"
            if not movies_file.exists():
                print(f"❌ 电影文件不存在: {movies_file}")
                return False
                
            self.movies = pd.read_csv(movies_file)
            if self.movies.empty:
                print("❌ 电影数据为空")
                return False
            
            # 检查评分数据质量
            if len(self.ratings) < 1000:
                print(f"⚠️  评分数据过少 ({len(self.ratings)} < 1000)，可能不是真实数据")
                return False
            
            # 检查电影数据质量  
            if len(self.movies) < 100:
                print(f"⚠️  电影数据过少 ({len(self.movies)} < 100)，可能不是真实数据")
                return False
            
            # 数据一致性检查
            rating_movie_ids = set(self.ratings['movieId'].unique())
            movie_ids = set(self.movies['movieId'].unique())
            
            if len(rating_movie_ids & movie_ids) < len(rating_movie_ids) * 0.8:
                print("⚠️  评分和电影数据不匹配，可能存在数据质量问题")
                return False
            
            print(f"✅ 真实MovieLens数据加载成功:")
            print(f"   📊 评分数: {len(self.ratings):,}")
            print(f"   🎬 电影数: {len(self.movies):,}")
            print(f"   👥 用户数: {self.ratings['userId'].nunique():,}")
            print(f"   ⭐ 评分范围: {self.ratings['rating'].min()}-{self.ratings['rating'].max()}")
            print(f"   📈 稀疏度: {(1 - len(self.ratings) / (self.ratings['userId'].nunique() * len(self.movies))) * 100:.2f}%")
            
            # 构建用户和电影特征
            self._build_user_profiles()
            self._build_movie_features()
            
            return True
            
        except Exception as e:
            print(f"❌ 加载真实MovieLens数据失败: {str(e)}")
            return False
    
    def _build_user_profiles(self):
        """构建用户画像"""
        print("👥 构建用户画像...")
        
        user_stats = self.ratings.groupby('userId').agg({
            'rating': ['count', 'mean', 'std'],
            'movieId': 'nunique'
        }).round(2)
        
        # 扁平化列名
        user_stats.columns = ['rating_count', 'avg_rating', 'rating_std', 'unique_movies']
        user_stats['rating_std'] = user_stats['rating_std'].fillna(0)
        
        # 获取用户喜好的电影类型
        movie_ratings = self.ratings.merge(self.movies[['movieId', 'genres']], on='movieId')
        
        for user_id in self.ratings['userId'].unique():
            user_movies = movie_ratings[movie_ratings['userId'] == user_id]
            high_rated = user_movies[user_movies['rating'] >= 4.0]
            
            # 提取喜好类型
            liked_genres = []
            for genres_str in high_rated['genres']:
                if pd.notna(genres_str) and genres_str != '(no genres listed)':
                    genres = genres_str.split('|')
                    liked_genres.extend(genres)
            
            genre_counts = pd.Series(liked_genres).value_counts()
            top_genres = genre_counts.head(3).index.tolist()
            
            self.user_profiles[user_id] = {
                **user_stats.loc[user_id].to_dict(),
                'preferred_genres': top_genres,
                'genre_diversity': len(set(liked_genres))
            }
        
        print(f"✅ 用户画像构建完成 ({len(self.user_profiles)} 个用户)")
    
    def _build_movie_features(self):
        """构建电影特征"""
        print("🎬 构建电影特征...")
        
        # 计算电影统计信息
        movie_stats = self.ratings.groupby('movieId').agg({
            'rating': ['count', 'mean', 'std'],
            'userId': 'nunique'
        }).round(2)
        
        movie_stats.columns = ['rating_count', 'avg_rating', 'rating_std', 'unique_users']
        movie_stats['rating_std'] = movie_stats['rating_std'].fillna(0)
        
        # 合并电影基本信息
        for _, movie in self.movies.iterrows():
            movie_id = movie['movieId']
            
            features = {
                'title': movie['title'],
                'genres': movie['genres'].split('|') if pd.notna(movie['genres']) else [],
                'year': self._extract_year(movie['title'])
            }
            
            # 添加统计信息
            if movie_id in movie_stats.index:
                features.update(movie_stats.loc[movie_id].to_dict())
            else:
                features.update({
                    'rating_count': 0,
                    'avg_rating': 0.0,
                    'rating_std': 0.0,
                    'unique_users': 0
                })
            
            self.movie_features[movie_id] = features
        
        print(f"✅ 电影特征构建完成 ({len(self.movie_features)} 部电影)")
    
    def _extract_year(self, title: str) -> int:
        """从电影标题提取年份"""
        import re
        match = re.search(r'\((\d{4})\)', title)
        return int(match.group(1)) if match else 0
    
    def generate_llm_recommendations(self, user_id: int, num_recommendations: int = 10, 
                                   model: str = "llama3", explain: bool = True) -> Dict[str, Any]:
        """
        生成LLM推荐结果
        
        Args:
            user_id: 用户ID
            num_recommendations: 推荐数量
            model: 使用的模型 ("llama3" 或 "qwen3")
            explain: 是否需要推荐解释
        
        Returns:
            Dict: 推荐结果和解释
        """
        if user_id not in self.user_profiles:
            print(f"❌ 用户 {user_id} 不存在")
            return {}
        
        print(f"🤖 为用户 {user_id} 生成LLM推荐 (模型: {model})...")
        
        # 准备用户上下文
        user_context = self._prepare_user_context(user_id)
        
        # 准备候选电影
        candidate_movies = self._get_candidate_movies(user_id, num_candidates=50)
        
        # 构建LLM提示词
        prompt = self._build_recommendation_prompt(user_context, candidate_movies, num_recommendations, explain)
        
        # 调用LLM
        model_name = self.primary_model if model == "llama3" else self.secondary_model
        llm_response = self._call_ollama_llm(prompt, model_name)
        
        if not llm_response:
            print(f"❌ LLM调用失败")
            return {}
        
        # 解析推荐结果
        recommendations = self._parse_llm_recommendations(llm_response, candidate_movies)
        
        # 记录为训练数据（用于Fisher/PAKD分析）
        self._record_llm_interaction(user_id, prompt, llm_response, recommendations)
        
        return {
            'user_id': user_id,
            'model': model,
            'recommendations': recommendations,
            'llm_response': llm_response,
            'timestamp': time.time()
        }
    
    def _prepare_user_context(self, user_id: int) -> Dict[str, Any]:
        """准备用户上下文信息"""
        profile = self.user_profiles[user_id]
        
        # 获取用户历史高分电影
        user_ratings = self.ratings[self.ratings['userId'] == user_id]
        high_rated = user_ratings[user_ratings['rating'] >= 4.0].merge(
            self.movies[['movieId', 'title', 'genres']], on='movieId'
        )
        
        liked_movies = high_rated.nlargest(5, 'rating')[['title', 'rating', 'genres']].to_dict('records')
        
        return {
            'user_id': user_id,
            'profile': profile,
            'liked_movies': liked_movies,
            'total_ratings': len(user_ratings),
            'avg_rating': profile['avg_rating']
        }
    
    def _get_candidate_movies(self, user_id: int, num_candidates: int = 50) -> List[Dict]:
        """获取候选推荐电影"""
        # 获取用户已评分电影
        user_movies = set(self.ratings[self.ratings['userId'] == user_id]['movieId'])
        
        # 筛选候选电影（未评分且有足够评分数）
        candidates = []
        for movie_id, features in self.movie_features.items():
            if movie_id not in user_movies and features['rating_count'] >= 10:
                candidates.append({
                    'movieId': movie_id,
                    'title': features['title'],
                    'genres': features['genres'],
                    'avg_rating': features['avg_rating'],
                    'rating_count': features['rating_count'],
                    'year': features['year']
                })
        
        # 按评分和流行度排序
        candidates.sort(key=lambda x: (x['avg_rating'], x['rating_count']), reverse=True)
        
        return candidates[:num_candidates]
    
    def _build_recommendation_prompt(self, user_context: Dict, candidates: List[Dict], 
                                   num_rec: int, explain: bool) -> str:
        """构建LLM推荐提示词"""
        
        liked_movies_str = "\n".join([
            f"- {movie['title']} (评分: {movie['rating']}, 类型: {movie['genres']})"
            for movie in user_context['liked_movies']
        ])
        
        candidates_str = "\n".join([
            f"{i+1}. {movie['title']} ({movie['year']}) - 类型: {', '.join(movie['genres'][:3])} - 平均评分: {movie['avg_rating']:.1f}"
            for i, movie in enumerate(candidates[:30])
        ])
        
        prompt = f"""你是一个专业的电影推荐系统。基于用户的观影历史，从候选电影中推荐最适合的电影。

用户信息:
- 用户ID: {user_context['user_id']}
- 总评分数: {user_context['total_ratings']}
- 平均评分: {user_context['avg_rating']:.2f}
- 偏好类型: {', '.join(user_context['profile']['preferred_genres'])}

用户喜欢的电影:
{liked_movies_str}

候选电影列表:
{candidates_str}

请从上述候选电影中选择 {num_rec} 部最适合推荐的电影，按推荐程度排序。

返回格式要求 (严格JSON格式):
{{
    "recommendations": [
        {{
            "rank": 1,
            "title": "电影标题",
            "reason": "推荐理由"
        }}
    ]
}}

推荐理由应该简洁明了，说明为什么这部电影适合该用户。"""

        return prompt
    
    def _call_ollama_llm(self, prompt: str, model: str, max_retries: int = 3) -> Optional[str]:
        """调用Ollama LLM API"""
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 1000
            }
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.ollama_endpoint,
                    json=payload,
                    timeout=60,
                    headers={'Content-Type': 'application/json'}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get('response', '').strip()
                else:
                    print(f"⚠️  API请求失败 (状态码: {response.status_code})")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                    
            except Exception as e:
                print(f"⚠️  LLM调用异常: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        print(f"❌ LLM调用失败，重试 {max_retries} 次后仍然失败")
        return None
    
    def _parse_llm_recommendations(self, llm_response: str, candidates: List[Dict]) -> List[Dict]:
        """解析LLM推荐结果"""
        try:
            # 尝试解析JSON格式
            if '{' in llm_response and '}' in llm_response:
                json_start = llm_response.find('{')
                json_end = llm_response.rfind('}') + 1
                json_str = llm_response[json_start:json_end]
                
                result = json.loads(json_str)
                if 'recommendations' in result:
                    return result['recommendations']
            
            # 如果JSON解析失败，尝试文本解析
            recommendations = []
            lines = llm_response.split('\n')
            
            for line in lines:
                line = line.strip()
                if any(keyword in line.lower() for keyword in ['推荐', 'recommend', '1.', '2.', '3.']):
                    # 提取电影标题
                    for candidate in candidates:
                        if candidate['title'].lower() in line.lower():
                            recommendations.append({
                                'rank': len(recommendations) + 1,
                                'title': candidate['title'],
                                'movieId': candidate['movieId'],
                                'reason': line
                            })
                            break
            
            return recommendations[:10]  # 最多返回10个推荐
            
        except Exception as e:
            print(f"⚠️  解析LLM推荐结果失败: {str(e)}")
            return []
    
    def _record_llm_interaction(self, user_id: int, prompt: str, response: str, recommendations: List[Dict]):
        """记录LLM交互数据用于Fisher/PAKD分析"""
        if not hasattr(self, 'llm_interactions'):
            self.llm_interactions = []
        
        interaction = {
            'user_id': user_id,
            'prompt': prompt,
            'response': response,
            'recommendations': recommendations,
            'timestamp': time.time(),
            'prompt_length': len(prompt),
            'response_length': len(response),
            'num_recommendations': len(recommendations)
        }
        
        self.llm_interactions.append(interaction)
    
    def initialize_fisher_calculator(self):
        """初始化Fisher Information计算器"""
        try:
            from llm_fisher_calculator import LLMFisherCalculator
            self.fisher_calculator = LLMFisherCalculator()
            print("✅ Fisher Information计算器初始化完成")
            return True
        except ImportError as e:
            print(f"⚠️  Fisher计算器导入失败: {e}")
            return False
    
    def initialize_pakd_distiller(self):
        """初始化PAKD蒸馏器"""
        try:
            from llm_pakd_distiller import LLMPAKDDistiller
            self.pakd_distiller = LLMPAKDDistiller()
            print("✅ PAKD蒸馏器初始化完成")
            return True
        except ImportError as e:
            print(f"⚠️  PAKD蒸馏器导入失败: {e}")
            return False
    
    def run_llm_fisher_analysis(self, sample_size: int = 100) -> Dict[str, Any]:
        """运行LLM的Fisher Information分析"""
        if not self.fisher_calculator:
            if not self.initialize_fisher_calculator():
                return {}
        
        print(f"🔍 开始LLM Fisher Information分析 (样本大小: {sample_size})...")
        
        # 收集LLM交互数据
        if not hasattr(self, 'llm_interactions') or len(self.llm_interactions) < 10:
            print("📊 生成LLM交互数据...")
            self._generate_llm_interactions(sample_size)
        
        # 计算Fisher Information
        fisher_results = self.fisher_calculator.compute_llm_fisher_information(
            self.llm_interactions[:sample_size]
        )
        
        print("✅ LLM Fisher Information分析完成")
        return fisher_results
    
    def run_llm_pakd_experiment(self, teacher_model: str = "llama3", student_model: str = "qwen3", 
                               num_distillation_samples: int = 200) -> Dict[str, Any]:
        """运行LLM的PAKD实验"""
        if not self.pakd_distiller:
            if not self.initialize_pakd_distiller():
                return {}
        
        print(f"🧠 开始LLM PAKD实验 (Teacher: {teacher_model}, Student: {student_model})...")
        
        # 生成teacher-student数据对
        distillation_data = self._generate_teacher_student_pairs(
            teacher_model, student_model, num_distillation_samples
        )
        
        # 运行PAKD
        pakd_results = self.pakd_distiller.run_llm_pakd(
            teacher_data=distillation_data['teacher'],
            student_data=distillation_data['student'],
            distillation_config={
                'temperature': 3.0,
                'alpha': 0.7,
                'pruning_ratio': 0.1
            }
        )
        
        print("✅ LLM PAKD实验完成")
        return pakd_results
    
    def _generate_llm_interactions(self, sample_size: int):
        """生成LLM交互数据"""
        print(f"🤖 生成 {sample_size} 个LLM交互样本...")
        
        # 随机选择用户
        user_ids = np.random.choice(
            list(self.user_profiles.keys()),
            size=min(sample_size // 2, len(self.user_profiles)),
            replace=False
        )
        
        for user_id in user_ids:
            # 使用两个模型生成推荐
            for model in ["llama3", "qwen3"]:
                try:
                    recommendation = self.generate_llm_recommendations(
                        user_id=user_id,
                        num_recommendations=5,
                        model=model,
                        explain=True
                    )
                    
                    if len(self.llm_interactions) >= sample_size:
                        break
                        
                except Exception as e:
                    print(f"⚠️  生成推荐失败 (用户: {user_id}, 模型: {model}): {e}")
                    continue
            
            if len(self.llm_interactions) >= sample_size:
                break
        
        print(f"✅ 生成了 {len(self.llm_interactions)} 个LLM交互样本")
    
    def _generate_teacher_student_pairs(self, teacher_model: str, student_model: str, 
                                      num_samples: int) -> Dict[str, List]:
        """生成teacher-student数据对"""
        print(f"👥 生成Teacher-Student数据对 ({num_samples} 对)...")
        
        teacher_data = []
        student_data = []
        
        # 随机选择用户
        user_ids = np.random.choice(
            list(self.user_profiles.keys()),
            size=min(num_samples, len(self.user_profiles)),
            replace=True
        )
        
        for user_id in user_ids:
            try:
                # 生成teacher推荐
                teacher_rec = self.generate_llm_recommendations(
                    user_id=user_id,
                    model=teacher_model,
                    num_recommendations=5
                )
                
                # 生成student推荐
                student_rec = self.generate_llm_recommendations(
                    user_id=user_id,
                    model=student_model,
                    num_recommendations=5
                )
                
                if teacher_rec and student_rec:
                    teacher_data.append(teacher_rec)
                    student_data.append(student_rec)
                
            except Exception as e:
                print(f"⚠️  生成数据对失败 (用户: {user_id}): {e}")
                continue
        
        print(f"✅ 生成了 {len(teacher_data)} 个Teacher-Student数据对")
        
        return {
            'teacher': teacher_data,
            'student': student_data
        }
    
    def run_comprehensive_experiment(self, output_dir: str = "experiment_results") -> Dict[str, Any]:
        """运行综合实验（推荐+Fisher+PAKD）"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("🧪 开始综合LLM推荐实验...")
        print("=" * 60)
        
        results = {
            'dataset_info': self._get_dataset_info(),
            'recommendation_results': {},
            'fisher_analysis': {},
            'pakd_experiment': {},
            'timestamp': time.time()
        }
        
        # 1. 推荐性能评估
        print("\n📊 1. 推荐性能评估...")
        rec_results = self._evaluate_recommendation_performance()
        results['recommendation_results'] = rec_results
        
        # 2. Fisher Information分析
        print("\n🔍 2. Fisher Information分析...")
        fisher_results = self.run_llm_fisher_analysis(sample_size=50)
        results['fisher_analysis'] = fisher_results
        
        # 3. PAKD实验
        print("\n🧠 3. PAKD实验...")
        pakd_results = self.run_llm_pakd_experiment(num_distillation_samples=100)
        results['pakd_experiment'] = pakd_results
        
        # 保存结果
        results_file = output_path / "real_movielens_llm_comprehensive_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n✅ 综合实验完成，结果保存至: {results_file}")
        return results
    
    def _get_dataset_info(self) -> Dict[str, Any]:
        """获取数据集信息"""
        return {
            'num_users': len(self.user_profiles),
            'num_movies': len(self.movie_features),
            'num_ratings': len(self.ratings),
            'rating_range': [float(self.ratings['rating'].min()), float(self.ratings['rating'].max())],
            'sparsity': float((1 - len(self.ratings) / (len(self.user_profiles) * len(self.movie_features))) * 100),
            'data_source': 'Real MovieLens Dataset'
        }
    
    def _evaluate_recommendation_performance(self) -> Dict[str, Any]:
        """评估推荐性能"""
        print("⚡ 评估推荐系统性能...")
        
        # 选择测试用户
        test_users = np.random.choice(
            list(self.user_profiles.keys()),
            size=min(20, len(self.user_profiles)),
            replace=False
        )
        
        results = {
            'llama3': {'successful_recommendations': 0, 'total_attempts': 0, 'avg_response_time': 0},
            'qwen3': {'successful_recommendations': 0, 'total_attempts': 0, 'avg_response_time': 0}
        }
        
        for user_id in test_users:
            for model in ['llama3', 'qwen3']:
                start_time = time.time()
                
                try:
                    recommendation = self.generate_llm_recommendations(
                        user_id=user_id,
                        model=model,
                        num_recommendations=10
                    )
                    
                    response_time = time.time() - start_time
                    
                    if recommendation and recommendation.get('recommendations'):
                        results[model]['successful_recommendations'] += 1
                        results[model]['avg_response_time'] += response_time
                    
                    results[model]['total_attempts'] += 1
                    
                except Exception as e:
                    print(f"⚠️  推荐生成失败 (用户: {user_id}, 模型: {model}): {e}")
                    results[model]['total_attempts'] += 1
        
        # 计算平均响应时间
        for model in results:
            if results[model]['successful_recommendations'] > 0:
                results[model]['avg_response_time'] /= results[model]['successful_recommendations']
                results[model]['success_rate'] = results[model]['successful_recommendations'] / results[model]['total_attempts']
            else:
                results[model]['success_rate'] = 0.0
        
        return results


def main():
    """主函数 - 运行真实数据LLM推荐实验"""
    print("🎬 Real MovieLens LLM Recommender 启动")
    print("📋 仅支持真实数据，确保推荐系统的真实性和可靠性")
    print("=" * 60)
    
    # 初始化推荐系统
    recommender = RealMovieLensLLMRecommender()
    
    # 加载真实数据
    if not recommender.load_real_movielens_data("small"):
        print("❌ 无法加载真实MovieLens数据，实验终止")
        return
    
    # 运行综合实验
    results = recommender.run_comprehensive_experiment()
    
    print("\n🎉 实验完成！")
    print(f"📊 数据集信息: {results['dataset_info']['num_users']} 用户, {results['dataset_info']['num_movies']} 电影")
    print(f"🤖 推荐成功率: Llama3 {results['recommendation_results']['llama3']['success_rate']:.2%}, Qwen3 {results['recommendation_results']['qwen3']['success_rate']:.2%}")
    
    if results['fisher_analysis']:
        print("🔍 Fisher分析: 完成")
    if results['pakd_experiment']:
        print("🧠 PAKD实验: 完成")


if __name__ == "__main__":
    main()
