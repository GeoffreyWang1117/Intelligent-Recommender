"""
Real LLM Teacher implementation using Ollama for knowledge distillation.
This module provides real Llama3 and Qwen3 integration for recommendation systems.
"""

import logging
import json
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import time

# Import with error handling
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from .base import BaseTeacher, TeacherType, TeacherOutput, UserProfile, ItemProfile


class RealLlamaTeacher(BaseTeacher):
    """Real Llama3 Teacher using Ollama for recommendation knowledge distillation."""
    
    def __init__(self, model_name: str = "llama3", config: Optional[Dict] = None):
        super().__init__(f"RealLlama_{model_name}", TeacherType.LLM)
        self.model_name = model_name
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.embedding_dim = self.config.get('embedding_dim', 384)
        self.max_context_length = self.config.get('max_context_length', 2048)
        self.temperature = self.config.get('temperature', 0.7)
        self.top_k = self.config.get('top_k', 5)
        
        # Initialize sentence transformer for embeddings
        self.sentence_model = None
        self.ollama_client = None
        
        # Data mappings (will be loaded from MovieLens data)
        self.user_profiles = {}
        self.item_profiles = {}
        
    def load_model(self, model_path: Optional[str] = None) -> None:
        """Load the Llama3 model via Ollama."""
        try:
            if not OLLAMA_AVAILABLE:
                raise ImportError("Ollama package not available")
                
            # Initialize Ollama client
            self.ollama_client = ollama
            
            # Test connection to Ollama
            models_response = self.ollama_client.list()
            
            # Handle different response formats
            available_models = []
            if hasattr(models_response, 'models'):
                models_list = models_response.models
            elif isinstance(models_response, dict) and 'models' in models_response:
                models_list = models_response['models']
            else:
                models_list = []
            
            for model in models_list:
                if hasattr(model, 'model'):
                    available_models.append(model.model)
                elif isinstance(model, dict):
                    if 'model' in model:
                        available_models.append(model['model'])
                    elif 'name' in model:
                        available_models.append(model['name'])
                else:
                    available_models.append(str(model))
            
            if f"{self.model_name}:latest" not in available_models:
                self.logger.warning(f"Model {self.model_name}:latest not found in Ollama. Available: {available_models}")
                self.logger.info("Continuing with available model list for testing...")
            
            # Initialize sentence transformer for embeddings
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            self.is_ready = True
            self.logger.info(f"Real Llama Teacher ({self.model_name}) loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load real Llama model: {e}")
            self.is_ready = False
            raise
    
    def load_movielens_data(self, data_path: str) -> None:
        """Load MovieLens data for context-aware recommendations."""
        try:
            import pandas as pd
            
            # Load movies data
            movies_path = f"{data_path}/movies.csv"
            movies_df = pd.read_csv(movies_path)
            
            # Create item profiles
            for _, row in movies_df.iterrows():
                movie_id = int(row['movieId'])  # Ensure int type
                title = row['title']
                genres = row['genres'].split('|') if '|' in row['genres'] else [row['genres']]
                
                self.item_profiles[movie_id] = {
                    'title': title,
                    'genres': genres,
                    'description': f"{title} ({', '.join(genres)})"
                }
            
            # Load ratings for user profiles (sample)
            ratings_path = f"{data_path}/ratings.csv"
            ratings_df = pd.read_csv(ratings_path)
            
            # Create basic user profiles based on rating history
            user_ratings = ratings_df.groupby('userId').agg({
                'movieId': 'count',
                'rating': 'mean'
            }).rename(columns={'movieId': 'num_ratings', 'rating': 'avg_rating'})
            
            for user_id, row in user_ratings.iterrows():
                user_id = int(user_id) if isinstance(user_id, (int, float, str)) else 0  # Safe conversion
                user_movie_ratings = ratings_df[ratings_df['userId'] == user_id]
                top_movies = user_movie_ratings.nlargest(5, 'rating')['movieId'].tolist()
                
                self.user_profiles[user_id] = {
                    'num_ratings': int(row['num_ratings']),
                    'avg_rating': float(row['avg_rating']),
                    'top_movies': top_movies,
                    'favorite_genres': self._get_user_favorite_genres(user_id, ratings_df)
                }
            
            self.logger.info(f"Loaded {len(self.item_profiles)} movies and {len(self.user_profiles)} user profiles")
            
        except Exception as e:
            self.logger.error(f"Failed to load MovieLens data: {e}")
            # Use minimal fallback data
            self._create_fallback_data()
    
    def _get_user_favorite_genres(self, user_id: int, ratings_df) -> List[str]:
        """Get user's favorite genres based on rating history."""
        user_ratings = ratings_df[ratings_df['userId'] == user_id]
        genre_scores = {}
        
        for _, rating in user_ratings.iterrows():
            movie_id = rating['movieId']
            if movie_id in self.item_profiles:
                genres = self.item_profiles[movie_id]['genres']
                for genre in genres:
                    if genre not in genre_scores:
                        genre_scores[genre] = []
                    genre_scores[genre].append(rating['rating'])
        
        # Calculate average rating per genre
        avg_genre_scores = {
            genre: np.mean(scores) 
            for genre, scores in genre_scores.items() 
            if len(scores) >= 2  # At least 2 movies
        }
        
        # Return top 3 genres
        sorted_genres = sorted(avg_genre_scores.items(), key=lambda x: x[1], reverse=True)
        return [genre for genre, _ in sorted_genres[:3]]
    
    def _create_fallback_data(self) -> None:
        """Create minimal fallback data if loading fails."""
        self.item_profiles = {
            1: {'title': 'Toy Story', 'genres': ['Animation', 'Children', 'Comedy'], 'description': 'Toy Story (Animation, Children, Comedy)'},
            2: {'title': 'Jumanji', 'genres': ['Adventure', 'Children', 'Fantasy'], 'description': 'Jumanji (Adventure, Children, Fantasy)'},
            3: {'title': 'Heat', 'genres': ['Action', 'Crime', 'Thriller'], 'description': 'Heat (Action, Crime, Thriller)'}
        }
        self.user_profiles = {
            1: {'num_ratings': 20, 'avg_rating': 3.5, 'top_movies': [1, 2], 'favorite_genres': ['Comedy', 'Animation']},
            2: {'num_ratings': 15, 'avg_rating': 4.0, 'top_movies': [3], 'favorite_genres': ['Action', 'Thriller']}
        }
    
    def _build_recommendation_prompt(
        self, 
        user_profile: UserProfile, 
        candidate_items: List[ItemProfile]
    ) -> str:
        """Build a comprehensive prompt for Llama3 recommendation."""
        
        user_id = user_profile.user_id
        user_data = self.user_profiles.get(user_id, {})
        
        # User context
        user_context = f"""User Profile:
- User ID: {user_id}
- Total Ratings: {user_data.get('num_ratings', 'Unknown')}
- Average Rating: {user_data.get('avg_rating', 'Unknown'):.2f}
- Favorite Genres: {', '.join(user_data.get('favorite_genres', ['Unknown']))}
"""
        
        # Candidate items context
        candidates_context = "Candidate Movies:\n"
        for i, item in enumerate(candidate_items[:10]):  # Limit to top 10 for context
            item_id = item.item_id
            item_data = self.item_profiles.get(item_id, {})
            candidates_context += f"{i+1}. Movie ID {item_id}: {item_data.get('description', 'Unknown Movie')}\n"
        
        # Build complete prompt
        prompt = f"""You are an expert movie recommendation system. Given a user's profile and a list of candidate movies, predict rating scores (1.0-5.0) for each movie.

{user_context}

{candidates_context}

Task: Predict rating scores (1.0-5.0) for each candidate movie based on the user's preferences.

Requirements:
1. Consider the user's favorite genres and rating patterns
2. Provide realistic scores between 1.0 and 5.0
3. Higher scores for movies matching user preferences
4. Return ONLY a JSON array of scores in the same order as candidates

Example output format: [3.2, 4.1, 2.8, 4.5, 3.0]

Prediction:"""
        
        return prompt
    
    def predict(
        self, 
        user_profile: UserProfile, 
        candidate_items: List[ItemProfile]
    ) -> TeacherOutput:
        """Generate predictions using real Llama3 model."""
        if not self.is_ready:
            raise RuntimeError("Model not loaded")
        
        try:
            start_time = time.time()
            
            # Build prompt
            prompt = self._build_recommendation_prompt(user_profile, candidate_items)
            
            # Call Llama3 via Ollama (with safe checks)
            if self.ollama_client and OLLAMA_AVAILABLE:
                response = self.ollama_client.generate(
                    model=f"{self.model_name}:latest",
                    prompt=prompt,
                    options={
                        'temperature': self.temperature,
                        'num_predict': 256,
                        'top_k': 40,
                        'top_p': 0.9
                    }
                )
                response_text = response['response']
            else:
                raise RuntimeError("Ollama client not available")
            
            # Parse response
            predictions = self._parse_llm_response(response_text, len(candidate_items))
            
            # Generate embeddings using sentence transformer (with safe checks)
            if self.sentence_model and SENTENCE_TRANSFORMERS_AVAILABLE:
                texts = [self._get_item_text(item) for item in candidate_items]
                embeddings = self.sentence_model.encode(texts, convert_to_tensor=True)
            else:
                # Fallback to random embeddings
                embeddings = torch.randn(len(candidate_items), self.embedding_dim)
            
            # Create explanation
            reasoning_chain = f"Llama3 prediction based on user preferences for {user_profile.user_id}"
            
            inference_time = time.time() - start_time
            
            return TeacherOutput(
                predictions=torch.tensor(predictions, dtype=torch.float32),
                embeddings=embeddings,
                confidence=0.85,  # High confidence for real LLM
                reasoning_chain=reasoning_chain,
                attention_weights=None  # Not available from Ollama
            )
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            # Fallback to random predictions
            predictions = torch.rand(len(candidate_items)) * 4 + 1  # 1-5 range
            embeddings = torch.randn(len(candidate_items), self.embedding_dim)
            
            return TeacherOutput(
                predictions=predictions,
                embeddings=embeddings,
                confidence=0.3,
                reasoning_chain=f"Fallback prediction due to error: {str(e)}",
                attention_weights=None
            )
    
    def _parse_llm_response(self, response: str, expected_length: int) -> List[float]:
        """Parse LLM response to extract numerical predictions."""
        try:
            # Try to find JSON array in response
            import re
            
            # Look for JSON array pattern
            json_pattern = r'\[[\d\.,\s]+\]'
            matches = re.findall(json_pattern, response)
            
            if matches:
                # Parse the first JSON array found
                scores_str = matches[0]
                scores = json.loads(scores_str)
                
                # Ensure all scores are valid numbers in range [1, 5]
                valid_scores = []
                for score in scores:
                    if isinstance(score, (int, float)):
                        valid_scores.append(max(1.0, min(5.0, float(score))))
                    else:
                        valid_scores.append(3.0)  # Default neutral score
                
                # Pad or truncate to expected length
                if len(valid_scores) < expected_length:
                    valid_scores.extend([3.0] * (expected_length - len(valid_scores)))
                elif len(valid_scores) > expected_length:
                    valid_scores = valid_scores[:expected_length]
                
                return valid_scores
            
            # Fallback: extract individual numbers
            numbers = re.findall(r'\b\d+\.?\d*\b', response)
            if numbers:
                scores = [max(1.0, min(5.0, float(num))) for num in numbers[:expected_length]]
                if len(scores) < expected_length:
                    scores.extend([3.0] * (expected_length - len(scores)))
                return scores
            
            # Ultimate fallback: random scores
            return [np.random.uniform(2.5, 4.5) for _ in range(expected_length)]
            
        except Exception as e:
            self.logger.warning(f"Failed to parse LLM response: {e}")
            return [3.0] * expected_length  # Neutral scores
    
    def _get_item_text(self, item: ItemProfile) -> str:
        """Get text representation of an item for embedding."""
        item_data = self.item_profiles.get(item.item_id, {})
        return item_data.get('description', f"Movie {item.item_id}")
    
    def get_embeddings(self, user_profile: UserProfile, item_profile: ItemProfile) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get embeddings for user and item using sentence transformer."""
        if not self.is_ready:
            # Fallback to random embeddings
            user_emb = torch.randn(1, self.embedding_dim)
            item_emb = torch.randn(1, self.embedding_dim)
            return user_emb.squeeze(0), item_emb.squeeze(0)
        
        try:
            if self.sentence_model and SENTENCE_TRANSFORMERS_AVAILABLE:
                # User text representation
                user_data = self.user_profiles.get(user_profile.user_id, {})
                user_text = f"User likes {', '.join(user_data.get('favorite_genres', ['movies']))} with average rating {user_data.get('avg_rating', 3.0):.1f}"
                
                # Item text representation
                item_text = self._get_item_text(item_profile)
                
                # Generate embeddings
                user_emb = self.sentence_model.encode([user_text], convert_to_tensor=True)
                item_emb = self.sentence_model.encode([item_text], convert_to_tensor=True)
                
                return user_emb.squeeze(0), item_emb.squeeze(0)
            else:
                # Fallback if sentence transformer not available
                user_emb = torch.randn(1, self.embedding_dim)
                item_emb = torch.randn(1, self.embedding_dim)
                return user_emb.squeeze(0), item_emb.squeeze(0)
            
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            # Fallback
            user_emb = torch.randn(1, self.embedding_dim)
            item_emb = torch.randn(1, self.embedding_dim)
            return user_emb.squeeze(0), item_emb.squeeze(0)
    
    def get_embedding(self, user_profile: UserProfile, item_profile: ItemProfile) -> Tuple[torch.Tensor, torch.Tensor]:
        """Backward compatibility alias for get_embeddings."""
        return self.get_embeddings(user_profile, item_profile)


class RealQwenTeacher(RealLlamaTeacher):
    """Real Qwen3 Teacher using Ollama - inherits from Llama with model-specific tweaks."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(model_name="qwen3", config=config)
        self.name = "RealQwen_qwen3"
        
        # Qwen-specific configurations
        self.temperature = self.config.get('temperature', 0.6)  # Slightly lower for Qwen
    
    def get_embeddings(self, user_profile: UserProfile, item_profile: ItemProfile) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get embeddings for user and item - inherits from parent but can be overridden."""
        return super().get_embeddings(user_profile, item_profile)
        
    def _build_recommendation_prompt(
        self, 
        user_profile: UserProfile, 
        candidate_items: List[ItemProfile]
    ) -> str:
        """Build a Qwen-optimized prompt for recommendation."""
        
        user_id = user_profile.user_id
        user_data = self.user_profiles.get(user_id, {})
        
        # Qwen prefers more structured prompts
        prompt = f"""<|system|>
You are a professional movie recommendation expert with deep understanding of user preferences and movie characteristics.

<|user|>
Please analyze the following user profile and predict rating scores for candidate movies.

User Information:
- ID: {user_id}
- Rating History: {user_data.get('num_ratings', 0)} movies
- Average Rating: {user_data.get('avg_rating', 3.0):.2f}/5.0
- Preferred Genres: {', '.join(user_data.get('favorite_genres', ['Various']))}

Candidate Movies:
"""
        
        for i, item in enumerate(candidate_items[:10]):
            item_id = item.item_id
            item_data = self.item_profiles.get(item_id, {})
            prompt += f"{i+1}. {item_data.get('description', f'Movie {item_id}')}\n"
        
        prompt += """
Please predict rating scores (1.0-5.0) for each movie based on user preferences.
Return only a JSON array of numerical scores.

<|assistant|>
Based on the user's preferences, here are the predicted ratings:

"""
        
        return prompt


def create_real_llm_teacher(model_name: str = "llama3", config: Optional[Dict] = None) -> BaseTeacher:
    """Factory function to create real LLM teachers."""
    if model_name.lower() in ["llama3", "llama"]:
        return RealLlamaTeacher(model_name="llama3", config=config)
    elif model_name.lower() in ["qwen3", "qwen"]:
        return RealQwenTeacher(config=config)
    else:
        raise ValueError(f"Unsupported model: {model_name}. Use 'llama3' or 'qwen3'")
