"""
Multi-Teacher Fusion System - LLM Teacher
基于Ollama的Llama3.1 Teacher实现，提供语义理解和推荐推理能力
"""

import json
import requests
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
from dataclasses import dataclass
import time
import re

from .base_teacher import BaseTeacher, UserProfile, ItemProfile, RecommendationResult

@dataclass
class LLMConfig:
    """LLM配置"""
    model_name: str = "llama3.1"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.7
    max_tokens: int = 512
    timeout: int = 30
    
    # 推荐相关配置
    max_history_items: int = 10
    recommendation_count: int = 20
    explanation_length: int = 100

class LLMTeacher(BaseTeacher):
    """基于Llama3.1的LLM Teacher"""
    
    def __init__(self, config: LLMConfig, movie_data: Optional[pd.DataFrame] = None):
        """
        初始化LLM Teacher
        
        Args:
            config: LLM配置
            movie_data: 电影元数据（包含电影ID、标题、类型等）
        """
        super().__init__("LLMTeacher")
        self.config = config
        self.movie_data = movie_data
        self.logger = logging.getLogger(__name__)
        
        # 电影ID到标题的映射
        self.movie_id_to_title = {}
        self.movie_title_to_id = {}
        if movie_data is not None:
            self._build_movie_mappings()
        
        # 测试Ollama连接
        self._test_connection()
    
    def _build_movie_mappings(self):
        """构建电影ID和标题的映射"""
        if self.movie_data is not None:
            for _, row in self.movie_data.iterrows():
                movie_id = row['movieId']
                title = row['title']
                self.movie_id_to_title[movie_id] = title
                self.movie_title_to_id[title] = movie_id
        
        self.logger.info(f"构建了 {len(self.movie_id_to_title)} 部电影的映射")
    
    def _test_connection(self):
        """测试Ollama连接"""
        try:
            response = requests.get(f"{self.config.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                
                if any(self.config.model_name in name for name in model_names):
                    self.logger.info(f"✅ Ollama连接成功，找到模型: {self.config.model_name}")
                else:
                    self.logger.warning(f"⚠️ 模型 {self.config.model_name} 未找到，可用模型: {model_names}")
            else:
                self.logger.error(f"❌ Ollama连接失败: HTTP {response.status_code}")
        except Exception as e:
            self.logger.error(f"❌ Ollama连接测试失败: {e}")
    
    def _call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        调用LLM生成响应
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示
        
        Returns:
            LLM生成的响应
        """
        try:
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "model": self.config.model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens
                }
            }
            
            response = requests.post(
                f"{self.config.base_url}/api/chat",
                json=payload,
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['message']['content'].strip()
            else:
                self.logger.error(f"LLM调用失败: HTTP {response.status_code}")
                return ""
                
        except Exception as e:
            self.logger.error(f"LLM调用异常: {e}")
            return ""
    
    def get_user_profile(self, user_id: int, interaction_history: List[Tuple[int, float]]) -> UserProfile:
        """
        通过LLM分析生成用户画像
        
        Args:
            user_id: 用户ID
            interaction_history: 交互历史 [(item_id, rating), ...]
        
        Returns:
            用户画像
        """
        # 限制历史记录长度
        recent_history = interaction_history[-self.config.max_history_items:]
        
        # 构建用户观影历史的文本描述
        movie_descriptions = []
        for item_id, rating in recent_history:
            title = self.movie_id_to_title.get(item_id, f"Movie_{item_id}")
            movie_descriptions.append(f"- {title} (评分: {rating}/5.0)")
        
        history_text = "\n".join(movie_descriptions)
        
        # 构建系统提示
        system_prompt = """你是一个专业的电影推荐分析师。请基于用户的观影历史，分析用户的偏好特征，包括：
1. 偏好的电影类型（如动作、喜剧、剧情等）
2. 评分习惯（严格/宽松）
3. 观影品味（商业片/艺术片）
4. 其他特征

请用简洁的JSON格式回答，包含preferences, rating_style, taste_level等字段。"""
        
        # 构建用户提示
        user_prompt = f"""用户ID: {user_id}
观影历史:
{history_text}

请分析这个用户的偏好特征。"""
        
        # 调用LLM
        llm_response = self._call_llm(user_prompt, system_prompt)
        
        # 解析LLM响应
        preferences = self._parse_user_preferences(llm_response, recent_history)
        
        return UserProfile(
            user_id=user_id,
            rating_history=recent_history,
            preferred_genres=preferences.get('genres', []),
            avg_rating=preferences.get('avg_rating', 3.5)
        )
    
    def _parse_user_preferences(self, llm_response: str, history: List[Tuple[int, float]]) -> Dict[str, Any]:
        """解析LLM生成的用户偏好"""
        preferences = {
            "genres": [],
            "rating_style": "moderate",
            "taste_level": "mainstream",
            "avg_rating": 3.5,
            "rating_variance": 1.0
        }
        
        try:
            # 尝试解析JSON格式的响应
            if '{' in llm_response and '}' in llm_response:
                json_start = llm_response.find('{')
                json_end = llm_response.rfind('}') + 1
                json_str = llm_response[json_start:json_end]
                parsed = json.loads(json_str)
                preferences.update(parsed)
        except:
            # 如果JSON解析失败，使用正则表达式提取信息
            genre_patterns = [
                r'动作|action',
                r'喜剧|comedy',
                r'剧情|drama',
                r'科幻|sci-fi|science fiction',
                r'恐怖|horror',
                r'爱情|romance',
                r'惊悚|thriller'
            ]
            
            detected_genres = []
            for pattern in genre_patterns:
                if re.search(pattern, llm_response, re.IGNORECASE):
                    detected_genres.append(pattern.split('|')[0])
            
            preferences["genres"] = detected_genres
        
        # 计算统计特征
        if history:
            ratings = [rating for _, rating in history]
            preferences["avg_rating"] = float(np.mean(ratings))
            preferences["rating_variance"] = float(np.var(ratings))
        
        return preferences
    
    def get_item_profile(self, item_id: int, user_ratings: List[Tuple[int, float]]) -> ItemProfile:
        """
        通过LLM分析生成物品画像
        
        Args:
            item_id: 物品ID
            user_ratings: 用户评分历史 [(user_id, rating), ...]
        
        Returns:
            物品画像
        """
        title = self.movie_id_to_title.get(item_id, f"Movie_{item_id}")
        
        # 构建评分统计
        ratings = [rating for _, rating in user_ratings] if user_ratings else []
        rating_stats = {
            "avg_rating": np.mean(ratings) if ratings else 3.5,
            "rating_count": len(ratings),
            "rating_std": np.std(ratings) if ratings else 1.0
        }
        
        # 构建系统提示
        system_prompt = """你是一个电影分析专家。请基于电影标题和评分数据，分析这部电影的特征，包括：
1. 电影类型
2. 目标观众
3. 艺术性/商业性
4. 受欢迎程度

请用简洁的JSON格式回答。"""
        
        # 构建用户提示
        user_prompt = f"""电影: {title}
平均评分: {rating_stats['avg_rating']:.2f}
评分人数: {rating_stats['rating_count']}
评分标准差: {rating_stats['rating_std']:.2f}

请分析这部电影的特征。"""
        
        # 调用LLM
        llm_response = self._call_llm(user_prompt, system_prompt)
        
        # 解析特征
        features = self._parse_item_features(llm_response, rating_stats)
        
        return ItemProfile(
            item_id=item_id,
            title=title,
            genres=features.get('genres', []),
            features=features
        )
    
    def _parse_item_features(self, llm_response: str, rating_stats: Dict[str, float]) -> Dict[str, Any]:
        """解析LLM生成的物品特征"""
        features = {
            "genres": [],
            "target_audience": "general",
            "artistic_level": "mainstream",
            "popularity": "medium"
        }
        
        # 添加统计特征
        features.update(rating_stats)
        
        try:
            # 尝试解析JSON格式的响应
            if '{' in llm_response and '}' in llm_response:
                json_start = llm_response.find('{')
                json_end = llm_response.rfind('}') + 1
                json_str = llm_response[json_start:json_end]
                parsed = json.loads(json_str)
                features.update(parsed)
        except:
            # 简单的文本分析
            if "艺术" in llm_response or "art" in llm_response.lower():
                features["artistic_level"] = "artistic"
            elif "商业" in llm_response or "commercial" in llm_response.lower():
                features["artistic_level"] = "commercial"
        
        return features
    
    def generate_recommendations(self, 
                               user_profile: UserProfile, 
                               candidate_items: List[int],
                               num_recommendations: int = 10) -> List[RecommendationResult]:
        """
        基于LLM推理生成推荐
        
        Args:
            user_profile: 用户画像
            candidate_items: 候选物品列表
            num_recommendations: 推荐数量
        
        Returns:
            推荐结果列表
        """
        # 构建用户偏好描述
        preferences = user_profile.preferences
        user_desc = f"""用户偏好:
- 平均评分: {preferences.get('avg_rating', 3.5):.2f}
- 偏好类型: {', '.join(preferences.get('genres', ['未知']))}
- 评分风格: {preferences.get('rating_style', '中等')}
- 观影品味: {preferences.get('taste_level', '主流')}"""
        
        # 构建候选电影列表
        candidate_titles = []
        for item_id in candidate_items[:50]:  # 限制候选数量以避免prompt太长
            title = self.movie_id_to_title.get(item_id, f"Movie_{item_id}")
            candidate_titles.append(f"{item_id}: {title}")
        
        candidates_text = "\n".join(candidate_titles)
        
        # 构建系统提示
        system_prompt = f"""你是一个专业的电影推荐系统。请根据用户偏好，从候选电影中选择 {num_recommendations} 部最适合的电影。

要求:
1. 选择的电影要符合用户的偏好类型和品味
2. 考虑用户的评分习惯
3. 确保推荐的多样性
4. 为每部电影提供简短的推荐理由

请按以下JSON格式回答:
{{
  "recommendations": [
    {{
      "movie_id": 电影ID,
      "confidence": 0.0-1.0的置信度,
      "reason": "推荐理由"
    }}
  ]
}}"""
        
        # 构建用户提示
        user_prompt = f"""{user_desc}

候选电影列表:
{candidates_text}

请从中推荐 {num_recommendations} 部电影。"""
        
        # 调用LLM
        llm_response = self._call_llm(user_prompt, system_prompt)
        
        # 解析推荐结果
        recommendations = self._parse_recommendations(llm_response, candidate_items, num_recommendations)
        
        return recommendations
    
    def _parse_recommendations(self, 
                             llm_response: str, 
                             candidate_items: List[int],
                             num_recommendations: int) -> List[RecommendationResult]:
        """解析LLM生成的推荐结果"""
        recommendations = []
        
        try:
            # 尝试解析JSON格式
            if '{' in llm_response and '}' in llm_response:
                json_start = llm_response.find('{')
                json_end = llm_response.rfind('}') + 1
                json_str = llm_response[json_start:json_end]
                parsed = json.loads(json_str)
                
                if 'recommendations' in parsed:
                    for rec in parsed['recommendations'][:num_recommendations]:
                        item_id = int(rec.get('movie_id', 0))
                        if item_id in candidate_items:
                            recommendations.append(RecommendationResult(
                                item_id=item_id,
                                score=float(rec.get('confidence', 0.5)),
                                explanation=rec.get('reason', '基于用户偏好推荐'),
                                metadata={
                                    "source": "llm_teacher",
                                    "title": self.movie_id_to_title.get(item_id, f"Movie_{item_id}")
                                }
                            ))
        except Exception as e:
            self.logger.warning(f"JSON解析失败: {e}")
        
        # 如果JSON解析失败，使用正则表达式提取
        if not recommendations:
            # 查找所有提到的电影ID
            id_pattern = r'(\d+):'
            mentioned_ids = re.findall(id_pattern, llm_response)
            
            for i, id_str in enumerate(mentioned_ids[:num_recommendations]):
                try:
                    item_id = int(id_str)
                    if item_id in candidate_items:
                        score = max(0.1, 1.0 - i * 0.1)  # 递减的置信度
                        recommendations.append(RecommendationResult(
                            item_id=item_id,
                            score=score,
                            explanation="基于LLM分析推荐",
                            metadata={
                                "source": "llm_teacher",
                                "title": self.movie_id_to_title.get(item_id, f"Movie_{item_id}")
                            }
                        ))
                except ValueError:
                    continue
        
        # 如果仍然没有推荐，返回随机选择
        if not recommendations:
            self.logger.warning("LLM推荐解析完全失败，使用随机推荐")
            import random
            random_items = random.sample(candidate_items, min(num_recommendations, len(candidate_items)))
            for i, item_id in enumerate(random_items):
                recommendations.append(RecommendationResult(
                    item_id=item_id,
                    score=0.3,  # 低置信度
                    explanation="LLM推荐解析失败，随机选择",
                    metadata={
                        "source": "llm_teacher_fallback",
                        "title": self.movie_id_to_title.get(item_id, f"Movie_{item_id}")
                    }
                ))
        
        return recommendations
    
    def predict_rating(self, user_id: int, item_id: int, 
                      user_profile: UserProfile = None, 
                      item_profile: ItemProfile = None) -> float:
        """
        预测用户对物品的评分
        
        Args:
            user_id: 用户ID
            item_id: 物品ID
            user_profile: 用户画像（可选）
            item_profile: 物品画像（可选）
        
        Returns:
            预测评分
        """
        # 简化版评分预测，基于用户偏好和物品特征
        if user_profile and user_profile.preferences:
            user_avg = user_profile.preferences.get('avg_rating', 3.5)
        else:
            user_avg = 3.5
        
        if item_profile and item_profile.features:
            item_avg = item_profile.features.get('avg_rating', 3.5)
        else:
            item_avg = 3.5
        
        # 简单的加权平均
        predicted_rating = (user_avg * 0.6 + item_avg * 0.4)
        
        # 确保在合理范围内
        return max(1.0, min(5.0, predicted_rating))

if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 创建配置
    config = LLMConfig(
        model_name="llama3.1",
        temperature=0.7,
        max_tokens=512
    )
    
    # 创建LLM Teacher
    llm_teacher = LLMTeacher(config)
    
    print("LLM Teacher 初始化完成!")
    print(f"配置: {config.model_name} @ {config.base_url}")
    
    # 测试用户画像生成
    test_history = [(1, 4.5), (2, 3.0), (3, 5.0)]
    user_profile = llm_teacher.get_user_profile(100, test_history)
    print(f"测试用户画像: {user_profile.preferences}")
