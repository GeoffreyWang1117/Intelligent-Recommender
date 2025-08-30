"""
Teacher model adapters for integrating existing models into the layerwise adapter system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

from .base import BaseTeacher, TeacherType, UserProfile, ItemProfile, TeacherOutput

# Import teacher models with error handling
try:
    from models.optimized_ensemble import OptimizedEnsembleTeacher
except ImportError:
    OptimizedEnsembleTeacher = None

try:
    from teachers.llm_teachers.real_movielens_llm_recommender import RealMovieLensLLMRecommender
except ImportError:
    RealMovieLensLLMRecommender = None


class EnsembleTeacherAdapter(BaseTeacher):
    """Adapter for the OptimizedEnsembleTeacher."""
    
    def __init__(self, config_path: Optional[str] = None):
        super().__init__("OptimizedEnsemble", TeacherType.ENSEMBLE)
        self.model = None
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """Load the ensemble teacher model."""
        try:
            if OptimizedEnsembleTeacher is None:
                raise ImportError("OptimizedEnsembleTeacher not available")
                
            # Use the existing optimized ensemble teacher
            self.model = OptimizedEnsembleTeacher()
            
            # If model_path is provided, load weights
            if model_path and os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location='cpu')
                self.model.load_state_dict(state_dict)
                self.logger.info(f"Loaded ensemble model from {model_path}")
            
            self.model.eval()
            self.is_ready = True
            self.logger.info("EnsembleTeacherAdapter ready")
            
        except Exception as e:
            self.logger.error(f"Failed to load ensemble model: {e}")
            self.is_ready = False
            # Create a mock model for testing
            self._create_mock_ensemble()
    
    def _create_mock_ensemble(self) -> None:
        """Create a mock ensemble model for testing."""
        class MockEnsemble:
            def __init__(self):
                self.num_models = 3
                
            def predict_batch(self, user_features, item_features):
                batch_size = item_features.size(0)
                return torch.randn(batch_size)
                
            def get_user_embedding(self, user_features):
                return torch.randn(1, 128)
                
            def get_item_embeddings(self, item_features):
                return torch.randn(item_features.size(0), 128)
                
            def eval(self):
                pass
        
        self.model = MockEnsemble()
        self.is_ready = True
        self.logger.warning("Using mock ensemble model for testing")
    
    def predict(
        self, 
        user_profile: UserProfile, 
        candidate_items: List[ItemProfile]
    ) -> TeacherOutput:
        """Generate predictions using ensemble teacher."""
        if not self.is_ready:
            raise RuntimeError("Model not loaded")
        
        try:
            # Prepare input data for ensemble model
            user_features = user_profile.feature_vector.unsqueeze(0)  # Add batch dim
            
            # Collect item features
            item_features = torch.stack([item.feature_vector for item in candidate_items])
            
            # Get ensemble predictions
            with torch.no_grad():
                # Assuming the ensemble model takes user and item features
                predictions = self.model.predict_batch(user_features, item_features)
                
                # Get embeddings for distillation
                user_emb = self.model.get_user_embedding(user_features)
                item_embs = self.model.get_item_embeddings(item_features)
                
                # Combine embeddings
                combined_emb = torch.cat([user_emb.repeat(len(candidate_items), 1), item_embs], dim=1)
            
            # Calculate confidence based on prediction variance
            confidence = float(1.0 - torch.std(predictions).item())
            confidence = max(0.1, min(1.0, confidence))  # Clamp to [0.1, 1.0]
            
            return TeacherOutput(
                predictions=predictions.squeeze(),
                embeddings=combined_emb,
                confidence=confidence,
                reasoning_chain=f"Ensemble prediction using {self.model.num_models} models"
            )
            
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            # Return dummy output on error
            return TeacherOutput(
                predictions=torch.zeros(len(candidate_items)),
                embeddings=torch.zeros(len(candidate_items), 256),
                confidence=0.1
            )
    
    def get_embeddings(
        self, 
        user_profile: UserProfile, 
        item_profile: ItemProfile
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract user and item embeddings."""
        if not self.is_ready:
            raise RuntimeError("Model not loaded")
        
        user_features = user_profile.feature_vector.unsqueeze(0)
        item_features = item_profile.feature_vector.unsqueeze(0)
        
        with torch.no_grad():
            user_emb = self.model.get_user_embedding(user_features)
            item_emb = self.model.get_item_embeddings(item_features)
        
        return user_emb.squeeze(0), item_emb.squeeze(0)


class LLMTeacherAdapter(BaseTeacher):
    """Adapter for LLM teachers (Llama3/Qwen3)."""
    
    def __init__(self, model_name: str = "llama3", config: Optional[Dict] = None):
        super().__init__(f"LLM_{model_name}", TeacherType.LLM)
        self.model_name = model_name.lower()
        self.model = None
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Default embedding dimension for LLM
        self.embedding_dim = self.config.get('embedding_dim', 512)
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """Load the LLM teacher model."""
        try:
            if RealMovieLensLLMRecommender is None:
                raise ImportError("RealMovieLensLLMRecommender not available")
                
            # Initialize the LLM recommender
            self.model = RealMovieLensLLMRecommender()
            
            # Mock initialization for testing
            self.is_ready = True
            self.logger.info(f"LLMTeacherAdapter ({self.model_name}) ready")
            
        except Exception as e:
            self.logger.error(f"Failed to load LLM model: {e}")
            self.is_ready = False
            # Create a mock model for testing
            self._create_mock_llm()
    
    def _create_mock_llm(self) -> None:
        """Create a mock LLM model for testing."""
        class MockLLM:
            def recommend_for_user(self, user_id, candidate_items, top_k):
                predictions = torch.randn(len(candidate_items)).tolist()
                explanations = [f"Mock explanation for item {item}" for item in candidate_items[:3]]
                return predictions, explanations
        
        self.model = MockLLM()
        self.is_ready = True
        self.logger.warning("Using mock LLM model for testing")
    
    def predict(
        self, 
        user_profile: UserProfile, 
        candidate_items: List[ItemProfile]
    ) -> TeacherOutput:
        """Generate predictions using LLM teacher."""
        if not self.is_ready:
            raise RuntimeError("Model not loaded")
        
        try:
            # Prepare data for LLM prediction
            user_id = user_profile.user_id
            item_ids = [item.item_id for item in candidate_items]
            
            # Get LLM predictions
            predictions, explanations = self.model.recommend_for_user(
                user_id=user_id,
                candidate_items=item_ids,
                top_k=len(item_ids)
            )
            
            # Convert to tensor
            pred_tensor = torch.tensor(predictions, dtype=torch.float32)
            
            # Generate embeddings (using feature vectors as proxy)
            user_emb = user_profile.feature_vector
            item_embs = torch.stack([item.feature_vector for item in candidate_items])
            
            # Create combined embeddings with LLM-style processing
            combined_emb = torch.cat([
                user_emb.unsqueeze(0).repeat(len(candidate_items), 1),
                item_embs
            ], dim=1)
            
            # Add semantic processing (simplified)
            combined_emb = self._add_semantic_processing(combined_emb)
            
            # Calculate confidence based on prediction distribution
            confidence = self._calculate_llm_confidence(pred_tensor)
            
            # Create reasoning chain
            reasoning_chain = self._create_reasoning_chain(explanations)
            
            return TeacherOutput(
                predictions=pred_tensor,
                embeddings=combined_emb,
                confidence=confidence,
                reasoning_chain=reasoning_chain
            )
            
        except Exception as e:
            self.logger.error(f"LLM prediction error: {e}")
            # Return dummy output on error
            return TeacherOutput(
                predictions=torch.zeros(len(candidate_items)),
                embeddings=torch.zeros(len(candidate_items), self.embedding_dim),
                confidence=0.1,
                reasoning_chain="Error in LLM prediction"
            )
    
    def get_embeddings(
        self, 
        user_profile: UserProfile, 
        item_profile: ItemProfile
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract user and item embeddings with LLM processing."""
        if not self.is_ready:
            raise RuntimeError("Model not loaded")
        
        user_emb = user_profile.feature_vector
        item_emb = item_profile.feature_vector
        
        # Add LLM-style semantic processing
        user_emb = self._add_semantic_processing(user_emb.unsqueeze(0)).squeeze(0)
        item_emb = self._add_semantic_processing(item_emb.unsqueeze(0)).squeeze(0)
        
        return user_emb, item_emb
    
    def _add_semantic_processing(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Add semantic processing to embeddings (simplified LLM simulation)."""
        # Simple transformation to simulate LLM semantic processing
        processed = embeddings * 1.1 + 0.1  # Scale and shift
        processed = torch.tanh(processed)  # Non-linear activation
        
        # Pad or truncate to target embedding dimension
        if processed.size(-1) < self.embedding_dim:
            padding = torch.zeros(*processed.shape[:-1], 
                                self.embedding_dim - processed.size(-1))
            processed = torch.cat([processed, padding], dim=-1)
        elif processed.size(-1) > self.embedding_dim:
            processed = processed[..., :self.embedding_dim]
        
        return processed
    
    def _calculate_llm_confidence(self, predictions: torch.Tensor) -> float:
        """Calculate confidence for LLM predictions."""
        # Use entropy as confidence measure
        probs = torch.softmax(predictions, dim=0)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))
        max_entropy = torch.log(torch.tensor(len(predictions), dtype=torch.float32))
        
        # Higher entropy = lower confidence
        confidence = 1.0 - (entropy / max_entropy).item()
        return max(0.1, min(1.0, confidence))
    
    def _create_reasoning_chain(self, explanations: List[str]) -> str:
        """Create reasoning chain from explanations."""
        if not explanations:
            return f"LLM {self.model_name} recommendation based on user preferences"
        
        # Take first few explanations and combine
        reasoning = f"LLM {self.model_name}: " + "; ".join(explanations[:3])
        return reasoning[:200]  # Limit length


class MultiTeacherManager:
    """Manager for coordinating multiple teacher models."""
    
    def __init__(self):
        self.teachers = {}
        self.logger = logging.getLogger(__name__)
    
    def add_teacher(self, teacher: BaseTeacher, name: Optional[str] = None) -> None:
        """Add a teacher model to the manager."""
        teacher_name = name or teacher.model_name
        self.teachers[teacher_name] = teacher
        self.logger.info(f"Added teacher: {teacher_name}")
    
    def load_all_teachers(self) -> None:
        """Load all registered teacher models."""
        for name, teacher in self.teachers.items():
            if not teacher.is_model_ready():
                try:
                    teacher.load_model()
                    self.logger.info(f"Loaded teacher: {name}")
                except Exception as e:
                    self.logger.error(f"Failed to load teacher {name}: {e}")
    
    def get_all_predictions(
        self,
        user_profile: UserProfile,
        candidate_items: List[ItemProfile]
    ) -> Dict[str, TeacherOutput]:
        """Get predictions from all teachers."""
        predictions = {}
        
        for name, teacher in self.teachers.items():
            if teacher.is_model_ready():
                try:
                    output = teacher.predict(user_profile, candidate_items)
                    predictions[name] = output
                except Exception as e:
                    self.logger.error(f"Prediction error for {name}: {e}")
        
        return predictions
    
    def get_teacher_weights(self, confidence_based: bool = True) -> Dict[str, float]:
        """Calculate teacher weights for ensemble."""
        if not confidence_based:
            # Equal weights
            num_teachers = len(self.teachers)
            return {name: 1.0 / num_teachers for name in self.teachers.keys()}
        
        # Confidence-based weights (placeholder - would use validation performance)
        weights = {}
        total_confidence = 0.0
        
        for name, teacher in self.teachers.items():
            if teacher.teacher_type == TeacherType.ENSEMBLE:
                confidence = 0.7  # Base confidence for ensemble
            else:  # LLM
                confidence = 0.8  # Base confidence for LLM
            
            weights[name] = confidence
            total_confidence += confidence
        
        # Normalize
        if total_confidence > 0:
            weights = {name: w / total_confidence for name, w in weights.items()}
        
        return weights
    
    def get_ready_teachers(self) -> List[str]:
        """Get list of ready teacher names."""
        return [name for name, teacher in self.teachers.items() 
                if teacher.is_model_ready()]


# Factory function for easy teacher creation
def create_teacher_manager(config: Dict) -> MultiTeacherManager:
    """Create and configure a multi-teacher manager."""
    manager = MultiTeacherManager()
    
    # Add ensemble teacher if configured
    if config.get('use_ensemble_teacher', True):
        ensemble_teacher = EnsembleTeacherAdapter()
        manager.add_teacher(ensemble_teacher, "ensemble")
    
    # Add LLM teachers if configured
    llm_models = config.get('llm_models', ['llama3'])
    for model_name in llm_models:
        llm_teacher = LLMTeacherAdapter(model_name, config)
        manager.add_teacher(llm_teacher, f"llm_{model_name}")
    
    return manager
