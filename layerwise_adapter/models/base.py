"""
Base interfaces and abstract classes for the Layerwise Adapter system.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union
import torch
import torch.nn as nn
from dataclasses import dataclass
from enum import Enum


class TeacherType(Enum):
    """Teacher model types."""
    ENSEMBLE = "ensemble"
    LLM = "llm"


@dataclass
class UserProfile:
    """User profile data structure."""
    user_id: int
    feature_vector: torch.Tensor
    interaction_history: List[int]
    preferences: Optional[torch.Tensor] = None
    metadata: Optional[Dict] = None


@dataclass
class ItemProfile:
    """Item profile data structure."""
    item_id: int
    feature_vector: torch.Tensor
    category: str
    semantic_embedding: Optional[torch.Tensor] = None
    metadata: Optional[Dict] = None


@dataclass
class TeacherOutput:
    """Teacher model output structure."""
    predictions: torch.Tensor
    embeddings: torch.Tensor
    confidence: float
    reasoning_chain: Optional[str] = None
    attention_weights: Optional[torch.Tensor] = None


@dataclass
class AdapterOutput:
    """Adapter model output structure."""
    recommendations: torch.Tensor
    confidence_scores: torch.Tensor
    explanations: Optional[List[str]] = None
    layer_outputs: Optional[Dict[str, torch.Tensor]] = None


class BaseTeacher(ABC):
    """Abstract base class for teacher models."""
    
    def __init__(self, model_name: str, teacher_type: TeacherType):
        self.model_name = model_name
        self.teacher_type = teacher_type
        self.is_ready = False
    
    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """Load the teacher model."""
        pass
    
    @abstractmethod
    def predict(
        self, 
        user_profile: UserProfile, 
        candidate_items: List[ItemProfile]
    ) -> TeacherOutput:
        """Generate predictions for user-item pairs."""
        pass
    
    @abstractmethod
    def get_embeddings(
        self, 
        user_profile: UserProfile, 
        item_profile: ItemProfile
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract user and item embeddings."""
        pass
    
    def is_model_ready(self) -> bool:
        """Check if model is ready for inference."""
        return self.is_ready


class BaseAdapter(nn.Module, ABC):
    """Abstract base class for adapter layers."""
    
    def __init__(self, input_dim: int, output_dim: int, layer_name: str):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_name = layer_name
    
    @abstractmethod
    def forward(
        self, 
        x: torch.Tensor, 
        teacher_guidance: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Forward pass with optional teacher guidance."""
        pass
    
    @abstractmethod
    def compute_distillation_loss(
        self,
        student_output: torch.Tensor,
        teacher_outputs: Dict[str, TeacherOutput],
        temperature: float = 3.0
    ) -> torch.Tensor:
        """Compute knowledge distillation loss."""
        pass


class BaseDistillationLoss(nn.Module, ABC):
    """Abstract base class for distillation loss functions."""
    
    def __init__(self, temperature: float = 3.0):
        super().__init__()
        self.temperature = temperature
    
    @abstractmethod
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute distillation loss."""
        pass


class BaseDataProcessor(ABC):
    """Abstract base class for data processing."""
    
    @abstractmethod
    def prepare_user_features(self, user_data: Dict) -> UserProfile:
        """Prepare user features for model input."""
        pass
    
    @abstractmethod
    def prepare_item_features(self, item_data: Dict) -> ItemProfile:
        """Prepare item features for model input."""
        pass
    
    @abstractmethod
    def create_training_batch(
        self, 
        user_profiles: List[UserProfile],
        item_profiles: List[ItemProfile],
        labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Create training batch from user-item pairs."""
        pass


class ModelConfig:
    """Configuration class for the layerwise adapter."""
    
    def __init__(self):
        # Model architecture
        self.embedding_dim = 128
        self.hidden_dim = 256
        self.num_heads = 8
        self.num_layers = 3
        self.dropout = 0.1
        self.activation = "gelu"
        
        # Training parameters
        self.batch_size = 256
        self.learning_rate = 1e-4
        self.warmup_steps = 1000
        self.max_epochs = 50
        self.patience = 5
        
        # Distillation parameters
        self.alpha_ensemble = 0.4
        self.beta_llm = 0.4
        self.gamma_task = 0.2
        self.temperature = 3.0
        
        # Teacher model paths
        self.ensemble_teacher_path = ""
        self.llm_teacher_path = ""
        
        # Data parameters
        self.max_sequence_length = 100
        self.vocab_size = 10000
        self.num_items = 50000
        self.num_users = 10000
    
    def update_from_dict(self, config_dict: Dict) -> None:
        """Update configuration from dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_')
        }


class ExperimentLogger:
    """Logger for experimental results and metrics."""
    
    def __init__(self, experiment_name: str, log_dir: str):
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.metrics_history = {}
        self.step = 0
    
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log metrics for current step."""
        for metric_name, value in metrics.items():
            if metric_name not in self.metrics_history:
                self.metrics_history[metric_name] = []
            self.metrics_history[metric_name].append((self.step, value))
        self.step += 1
    
    def log_model_info(self, model: nn.Module) -> None:
        """Log model architecture information."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Model: {model.__class__.__name__}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    def save_results(self, results: Dict) -> None:
        """Save experimental results."""
        import json
        import os
        
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Save metrics history
        with open(f"{self.log_dir}/metrics_history.json", "w") as f:
            json.dump(self.metrics_history, f, indent=2)
        
        # Save final results
        with open(f"{self.log_dir}/final_results.json", "w") as f:
            json.dump(results, f, indent=2)


def calculate_model_efficiency(model: nn.Module, input_tensor: torch.Tensor) -> Dict[str, float]:
    """Calculate model efficiency metrics."""
    import time
    
    model.eval()
    
    # Calculate model size
    param_size = sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024  # MB
    
    # Calculate inference speed
    warmup_runs = 10
    test_runs = 100
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)
    
    # Timing
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(test_runs):
            _ = model(input_tensor)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    avg_inference_time = (end_time - start_time) / test_runs * 1000  # ms
    
    return {
        "model_size_mb": param_size,
        "avg_inference_time_ms": avg_inference_time,
        "parameters_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
    }
