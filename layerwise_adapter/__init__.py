"""
LayerwiseAdapter package - A three-layer knowledge distillation system for recommendation.
"""

__version__ = "0.1.0"
__author__ = "LayerwiseAdapter Team"

from .models import (
    ModelConfig,
    LayerwiseAdapter,
    MultiTeacherManager,
    create_teacher_manager
)

from .utils import (
    RecommendationDataset,
    LayerwiseTrainer,
    create_trainer
)

__all__ = [
    'ModelConfig',
    'LayerwiseAdapter', 
    'MultiTeacherManager',
    'create_teacher_manager',
    'RecommendationDataset',
    'LayerwiseTrainer',
    'create_trainer'
]
