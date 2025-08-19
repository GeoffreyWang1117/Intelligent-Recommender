#!/usr/bin/env python3
"""
双语LLM Teacher推荐系统演示
Demo: Dual Language LLM Teacher Recommendation System

支持的模型:
- Llama3 (英文主力) - 适配MovieLens英文数据
- Qwen3 (中文对照) - 跨语言推荐研究

作者: GitHub Copilot
日期: 2025-08-18
"""

import json
import requests
import time
from typing import Dict, List, Any

class DualLanguageLLMTeacher:
    """双语LLM推荐Teacher - 英文主力+中文对照"""
    
    def __init__(self, ollama_endpoint="http://localhost:11434/api/generate"):
        self.ollama_endpoint = ollama_endpoint
        self.primary_model = "llama3:latest"    # 英文主力 (MovieLens适配)
        self.secondary_model = "qwen3:latest"   # 中文对照 (跨语言研究)
        
    def get_recommendation(self, user_profile: Dict, candidates: List[Dict], 
                          model_choice: str = "primary", top_k: int = 5) -> Dict:
        """
        获取LLM推荐结果
        
        Args:
            user_profile: 用户档案
            candidates: 候选电影列表
            model_choice: "primary" (Llama3) 或 "secondary" (Qwen3)
            top_k: 推荐数量
        """
        model_name = self.primary_model if model_choice == "primary" else self.secondary_model
        
        # 构建提示词
        prompt = self._build_prompt(user_profile, candidates, model_choice, top_k)
        
        try:
            print(f"🤖 Using {model_name} ({'英文主力' if model_choice == 'primary' else '中文对照'})...")
            
            # 调用Ollama API
            response = requests.post(self.ollama_endpoint, json={
                "model": model_name,
                "prompt": prompt,
                "stream": False
            }, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return self._parse_response(result['response'], model_choice)
            else:
                return {"error": f"API调用失败: {response.status_code}"}
                
        except Exception as e:
            return {"error": f"推荐生成失败: {str(e)}"}
    
    def _build_prompt(self, user_profile: Dict, candidates: List[Dict], 
                     model_choice: str, top_k: int) -> str:
        """构建针对不同语言模型的提示词"""
        
        if model_choice == "primary":  # Llama3 英文提示词
            candidates_text = "\n".join([
                f"- Movie {i+1}: {movie['title']} (Genre: {movie['genre']}, Year: {movie['year']})"
                for i, movie in enumerate(candidates)
            ])
            
            prompt = f"""You are an expert movie recommendation system. Based on the user's profile, recommend the top {top_k} movies from the candidate list.

User Profile:
- User ID: {user_profile['user_id']}
- Age: {user_profile['age']}
- Favorite Genres: {', '.join(user_profile['favorite_genres'])}
- Previous Ratings: {user_profile['avg_rating']:.1f}/5.0 average
- Viewing History: {user_profile['movies_watched']} movies watched

Candidate Movies:
{candidates_text}

Please analyze the user's preferences and recommend the most suitable movies. 
Output format: JSON array with movie_id, title, predicted_rating (1-5), confidence (0-1), and brief reason.

Example output:
[
  {{"movie_id": 1, "title": "Movie Title", "predicted_rating": 4.2, "confidence": 0.85, "reason": "Matches user's preference for action films"}}
]
"""
        
        else:  # Qwen3 中文提示词
            candidates_text = "\n".join([
                f"- 电影{i+1}: {movie['title']} (类型: {movie['genre']}, 年份: {movie['year']})"
                for i, movie in enumerate(candidates)
            ])
            
            prompt = f"""你是专业的电影推荐系统专家。根据用户档案，从候选电影中推荐最适合的{top_k}部电影。

用户档案：
- 用户ID：{user_profile['user_id']}
- 年龄：{user_profile['age']}岁
- 偏好类型：{', '.join(user_profile['favorite_genres'])}
- 历史评分：平均{user_profile['avg_rating']:.1f}/5.0分
- 观影数量：已观看{user_profile['movies_watched']}部电影

候选电影：
{candidates_text}

请分析用户偏好，推荐最适合的电影。
输出格式：JSON数组，包含movie_id, title, predicted_rating (1-5), confidence (0-1), reason。

示例输出：
[
  {{"movie_id": 1, "title": "电影标题", "predicted_rating": 4.2, "confidence": 0.85, "reason": "符合用户对动作片的偏好"}}
]
"""
        
        return prompt
    
    def _parse_response(self, response_text: str, model_choice: str) -> Dict:
        """解析LLM响应"""
        try:
            # 尝试提取JSON部分
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_text = response_text[start_idx:end_idx]
                recommendations = json.loads(json_text)
                
                return {
                    "model": "Llama3 (英文)" if model_choice == "primary" else "Qwen3 (中文)",
                    "recommendations": recommendations,
                    "status": "success"
                }
            else:
                return {
                    "model": "Llama3 (英文)" if model_choice == "primary" else "Qwen3 (中文)",
                    "raw_response": response_text,
                    "status": "parse_failed"
                }
                
        except json.JSONDecodeError:
            return {
                "model": "Llama3 (英文)" if model_choice == "primary" else "Qwen3 (中文)",
                "raw_response": response_text,
                "status": "json_error"
            }
    
    def compare_dual_recommendations(self, user_profile: Dict, candidates: List[Dict], 
                                   top_k: int = 5) -> Dict:
        """对比双语模型推荐效果"""
        print("🌐 开始双语LLM推荐对比实验...")
        
        # 获取Llama3英文推荐
        llama3_result = self.get_recommendation(user_profile, candidates, "primary", top_k)
        time.sleep(1)  # 避免API调用过快
        
        # 获取Qwen3中文推荐
        qwen3_result = self.get_recommendation(user_profile, candidates, "secondary", top_k)
        
        # 计算重叠度
        overlap_score = self._calculate_overlap(llama3_result, qwen3_result)
        
        return {
            "llama3_english": llama3_result,
            "qwen3_chinese": qwen3_result,
            "overlap_score": overlap_score,
            "comparison_summary": self._generate_comparison_summary(llama3_result, qwen3_result)
        }
    
    def _calculate_overlap(self, result1: Dict, result2: Dict) -> float:
        """计算两个推荐结果的重叠度"""
        try:
            if (result1.get("status") != "success" or 
                result2.get("status") != "success"):
                return 0.0
            
            recs1 = {rec["movie_id"] for rec in result1["recommendations"]}
            recs2 = {rec["movie_id"] for rec in result2["recommendations"]}
            
            if not recs1 or not recs2:
                return 0.0
            
            intersection = len(recs1 & recs2)
            union = len(recs1 | recs2)
            
            return intersection / union if union > 0 else 0.0
            
        except:
            return 0.0
    
    def _generate_comparison_summary(self, llama3_result: Dict, qwen3_result: Dict) -> str:
        """生成对比摘要"""
        summary = []
        
        if llama3_result.get("status") == "success":
            summary.append("✅ Llama3英文推荐成功")
        else:
            summary.append("❌ Llama3英文推荐失败")
        
        if qwen3_result.get("status") == "success":
            summary.append("✅ Qwen3中文推荐成功")
        else:
            summary.append("❌ Qwen3中文推荐失败")
        
        return " | ".join(summary)


def demo_movie_recommendation():
    """演示双语电影推荐"""
    print("🎬 双语LLM Teacher电影推荐系统演示")
    print("=" * 50)
    
    # 初始化双语推荐系统
    dual_teacher = DualLanguageLLMTeacher()
    
    # 模拟用户档案 (MovieLens风格)
    user_profile = {
        "user_id": 123,
        "age": 28,
        "favorite_genres": ["Action", "Sci-Fi", "Thriller"],
        "avg_rating": 4.2,
        "movies_watched": 156
    }
    
    # 模拟候选电影
    candidates = [
        {"movie_id": 1, "title": "The Matrix", "genre": "Sci-Fi", "year": 1999},
        {"movie_id": 2, "title": "John Wick", "genre": "Action", "year": 2014},
        {"movie_id": 3, "title": "Inception", "genre": "Sci-Fi", "year": 2010},
        {"movie_id": 4, "title": "The Notebook", "genre": "Romance", "year": 2004},
        {"movie_id": 5, "title": "Mad Max: Fury Road", "genre": "Action", "year": 2015},
        {"movie_id": 6, "title": "Interstellar", "genre": "Sci-Fi", "year": 2014},
        {"movie_id": 7, "title": "Die Hard", "genre": "Action", "year": 1988},
        {"movie_id": 8, "title": "Blade Runner 2049", "genre": "Sci-Fi", "year": 2017}
    ]
    
    print(f"👤 用户档案:")
    print(f"   ID: {user_profile['user_id']}")
    print(f"   年龄: {user_profile['age']}岁")
    print(f"   偏好类型: {', '.join(user_profile['favorite_genres'])}")
    print(f"   平均评分: {user_profile['avg_rating']}/5.0")
    print(f"   观影数量: {user_profile['movies_watched']}部")
    print()
    
    print(f"🎭 候选电影 ({len(candidates)}部):")
    for movie in candidates:
        print(f"   [{movie['movie_id']}] {movie['title']} ({movie['genre']}, {movie['year']})")
    print()
    
    # 执行双语推荐对比
    comparison_result = dual_teacher.compare_dual_recommendations(
        user_profile, candidates, top_k=5
    )
    
    # 展示结果
    print("📊 双语推荐对比结果:")
    print("=" * 50)
    
    # Llama3英文结果
    llama3_result = comparison_result["llama3_english"]
    print("🇺🇸 Llama3 (英文主力) 推荐结果:")
    if llama3_result.get("status") == "success":
        for i, rec in enumerate(llama3_result["recommendations"][:5], 1):
            print(f"   {i}. {rec.get('title', 'N/A')} "
                  f"(预测评分: {rec.get('predicted_rating', 'N/A')}, "
                  f"置信度: {rec.get('confidence', 'N/A')})")
            if 'reason' in rec:
                print(f"      理由: {rec['reason']}")
    else:
        print(f"   ❌ 推荐失败: {llama3_result.get('error', '未知错误')}")
    print()
    
    # Qwen3中文结果
    qwen3_result = comparison_result["qwen3_chinese"]
    print("🇨🇳 Qwen3 (中文对照) 推荐结果:")
    if qwen3_result.get("status") == "success":
        for i, rec in enumerate(qwen3_result["recommendations"][:5], 1):
            print(f"   {i}. {rec.get('title', 'N/A')} "
                  f"(预测评分: {rec.get('predicted_rating', 'N/A')}, "
                  f"置信度: {rec.get('confidence', 'N/A')})")
            if 'reason' in rec:
                print(f"      理由: {rec['reason']}")
    else:
        print(f"   ❌ 推荐失败: {qwen3_result.get('error', '未知错误')}")
    print()
    
    # 对比分析
    print("🔄 跨语言推荐分析:")
    print(f"   重叠度: {comparison_result['overlap_score']:.3f}")
    print(f"   状态摘要: {comparison_result['comparison_summary']}")
    
    return comparison_result


if __name__ == "__main__":
    try:
        # 检查Ollama服务
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama服务运行正常")
            demo_result = demo_movie_recommendation()
        else:
            print("❌ Ollama服务无法访问")
    
    except requests.exceptions.RequestException:
        print("❌ 无法连接到Ollama服务，请确保Ollama正在运行")
        print("   启动命令: ollama serve")
