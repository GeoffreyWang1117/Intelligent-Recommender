"""
Main LayerwiseAdapter model that combines all three adapter layers.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import logging

try:
    from .base import ModelConfig, AdapterOutput, UserProfile, ItemProfile
    from .adapters import EmbeddingAdapter, InteractionAdapter, ReasoningAdapter
    from .teacher_adapters import MultiTeacherManager
except ImportError:
    # Handle relative import issues
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from base import ModelConfig, AdapterOutput, UserProfile, ItemProfile
    from adapters import EmbeddingAdapter, InteractionAdapter, ReasoningAdapter
    from teacher_adapters import MultiTeacherManager


class LayerwiseAdapter(nn.Module):
    """
    Main LayerwiseAdapter model implementing three-layer knowledge distillation.
    
    Architecture:
    1. EmbeddingAdapter: Feature alignment and embedding learning
    2. InteractionAdapter: User-item interaction modeling  
    3. ReasoningAdapter: Final prediction and explainability
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize the three adapter layers
        self.embedding_adapter = EmbeddingAdapter(config)
        self.interaction_adapter = InteractionAdapter(config)
        self.reasoning_adapter = ReasoningAdapter(config)
        
        # Teacher manager for multi-teacher distillation
        self.teacher_manager = None
        
        # Training state
        self.training_step = 0
        self.distillation_weights = {
            'alpha_ensemble': config.alpha_ensemble,
            'beta_llm': config.beta_llm,
            'gamma_task': config.gamma_task
        }
        
        self.logger.info("LayerwiseAdapter initialized with 3 layers")
    
    def set_teacher_manager(self, teacher_manager: MultiTeacherManager) -> None:
        """Set the teacher manager for distillation."""
        self.teacher_manager = teacher_manager
        self.logger.info("Teacher manager set for distillation")
    
    def forward(self, 
                user_profiles: List[UserProfile], 
                item_profiles: List[ItemProfile],
                return_layer_outputs: bool = False) -> AdapterOutput:
        """
        Forward pass through all three adapter layers.
        
        Args:
            user_profiles: List of user profiles
            item_profiles: List of item profiles  
            return_layer_outputs: Whether to return intermediate layer outputs
            
        Returns:
            AdapterOutput with predictions, confidence scores, and optional explanations
        """
        batch_size = len(user_profiles)
        
        # Prepare input tensor from profiles
        input_tensor = self._prepare_input_tensor(user_profiles, item_profiles)
        
        # Get teacher guidance if available
        teacher_guidance = self._get_teacher_guidance(user_profiles, item_profiles)
        
        # Layer 1: Embedding Adapter
        embedding_output = self.embedding_adapter(
            input_tensor, 
            teacher_guidance.get('embedding', None)
        )
        
        # Layer 2: Interaction Adapter
        interaction_output = self.interaction_adapter(
            embedding_output,
            teacher_guidance.get('interaction', None)
        )
        
        # Layer 3: Reasoning Adapter
        predictions = self.reasoning_adapter(
            interaction_output,
            teacher_guidance.get('reasoning', None)
        )
        
        # Get confidence scores
        confidence_scores = self.reasoning_adapter.get_confidence(interaction_output)
        
        # Prepare layer outputs if requested
        layer_outputs = None
        if return_layer_outputs:
            layer_outputs = {
                'embedding': embedding_output,
                'interaction': interaction_output,
                'reasoning': predictions
            }
        
        return AdapterOutput(
            recommendations=predictions.squeeze(-1),
            confidence_scores=confidence_scores.squeeze(-1),
            explanations=None,  # TODO: Implement explanation generation
            layer_outputs=layer_outputs
        )
    
    def compute_distillation_loss(self,
                                student_output: AdapterOutput,
                                user_profiles: List[UserProfile],
                                item_profiles: List[ItemProfile],
                                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute multi-layer distillation loss.
        
        Returns:
            Dictionary containing different loss components
        """
        losses = {}
        
        if self.teacher_manager is None:
            # Return only task loss if no teachers available
            if labels is not None:
                task_loss = nn.functional.mse_loss(student_output.recommendations, labels)
                losses['task_loss'] = task_loss
                losses['total_loss'] = task_loss
            return losses
        
        # Get teacher outputs
        teacher_outputs = {}
        for user_profile, item_profile in zip(user_profiles, item_profiles):
            teacher_preds = self.teacher_manager.get_all_predictions(
                user_profile, [item_profile]
            )
            for teacher_name, teacher_output in teacher_preds.items():
                if teacher_name not in teacher_outputs:
                    teacher_outputs[teacher_name] = []
                teacher_outputs[teacher_name].append(teacher_output)
        
        # Convert teacher outputs to tensors
        teacher_tensors = {}
        for teacher_name, outputs_list in teacher_outputs.items():
            if outputs_list:
                predictions = torch.stack([out.predictions for out in outputs_list])
                embeddings = torch.stack([out.embeddings for out in outputs_list])
                teacher_tensors[teacher_name] = {
                    'predictions': predictions,
                    'embeddings': embeddings
                }
        
        # Compute layer-wise distillation losses
        total_distillation_loss = torch.tensor(0.0, device=student_output.recommendations.device)
        
        if student_output.layer_outputs is not None:
            # Embedding layer distillation
            embedding_teacher_outputs = {
                name: tensors['embeddings'] 
                for name, tensors in teacher_tensors.items()
            }
            if embedding_teacher_outputs:
                embedding_loss = self.embedding_adapter.compute_distillation_loss(
                    student_output.layer_outputs['embedding'],
                    embedding_teacher_outputs
                )
                losses['embedding_distillation'] = embedding_loss
                total_distillation_loss += embedding_loss
            
            # Interaction layer distillation  
            interaction_loss = self.interaction_adapter.compute_distillation_loss(
                student_output.layer_outputs['interaction'],
                embedding_teacher_outputs  # Use same teacher embeddings
            )
            losses['interaction_distillation'] = interaction_loss
            total_distillation_loss += interaction_loss
            
            # Reasoning layer distillation
            reasoning_teacher_outputs = {
                name: tensors['predictions']
                for name, tensors in teacher_tensors.items()
            }
            if reasoning_teacher_outputs:
                reasoning_loss = self.reasoning_adapter.compute_distillation_loss(
                    student_output.recommendations.unsqueeze(-1),
                    reasoning_teacher_outputs
                )
                losses['reasoning_distillation'] = reasoning_loss
                total_distillation_loss += reasoning_loss
        
        # Task loss (if labels provided)
        task_loss = torch.tensor(0.0, device=student_output.recommendations.device)
        if labels is not None:
            task_loss = nn.functional.mse_loss(student_output.recommendations, labels)
            losses['task_loss'] = task_loss
        
        # Combine losses with weights
        total_loss = (
            self.distillation_weights['alpha_ensemble'] * total_distillation_loss +
            self.distillation_weights['gamma_task'] * task_loss
        )
        
        losses['total_distillation'] = total_distillation_loss
        losses['total_loss'] = total_loss
        
        return losses
    
    def _prepare_input_tensor(self, 
                            user_profiles: List[UserProfile], 
                            item_profiles: List[ItemProfile]) -> torch.Tensor:
        """Prepare input tensor from user and item profiles."""
        batch_size = len(user_profiles)
        max_items = len(item_profiles)
        
        # Create input tensor [batch_size, max_items, feature_dim]
        # For now, use a simple concatenation of user_id, item_id, and features
        device = next(self.parameters()).device  # Get model device
        input_tensor = torch.zeros(batch_size, max_items, self.config.embedding_dim, device=device)
        
        for i, (user_profile, item_profile) in enumerate(zip(user_profiles, item_profiles)):
            # Simple feature preparation
            features = torch.zeros(self.config.embedding_dim, device=device)
            features[0] = user_profile.user_id  # User ID
            features[1] = item_profile.item_id   # Item ID
            
            # Add user features if available
            if user_profile.feature_vector is not None:
                feat_size = min(user_profile.feature_vector.size(0), 
                              self.config.embedding_dim - 2)
                features[2:2+feat_size] = user_profile.feature_vector[:feat_size].to(device)
            
            input_tensor[i, 0] = features  # Single item per user for now
        
        return input_tensor
    
    def _get_teacher_guidance(self, 
                            user_profiles: List[UserProfile], 
                            item_profiles: List[ItemProfile]) -> Dict[str, Dict[str, torch.Tensor]]:
        """Get teacher guidance for each layer."""
        guidance = {
            'embedding': {},
            'interaction': {},
            'reasoning': {}
        }
        
        if self.teacher_manager is None:
            return guidance
        
        # Get teacher outputs
        try:
            for user_profile, item_profile in zip(user_profiles, item_profiles):
                teacher_outputs = self.teacher_manager.get_all_predictions(
                    user_profile, [item_profile]
                )
                
                for teacher_name, teacher_output in teacher_outputs.items():
                    # Add to embedding guidance
                    if teacher_name not in guidance['embedding']:
                        guidance['embedding'][teacher_name] = []
                    guidance['embedding'][teacher_name].append(teacher_output.embeddings)
                    
                    # Add to interaction guidance (use same embeddings)
                    if teacher_name not in guidance['interaction']:
                        guidance['interaction'][teacher_name] = []
                    guidance['interaction'][teacher_name].append(teacher_output.embeddings)
                    
                    # Add to reasoning guidance  
                    if teacher_name not in guidance['reasoning']:
                        guidance['reasoning'][teacher_name] = []
                    guidance['reasoning'][teacher_name].append(teacher_output.predictions)
            
            # Convert lists to tensors
            for layer_name in guidance.keys():
                for teacher_name in list(guidance[layer_name].keys()):
                    try:
                        tensor_list = guidance[layer_name][teacher_name]
                        if tensor_list:
                            guidance[layer_name][teacher_name] = torch.stack(tensor_list)
                    except Exception as e:
                        self.logger.warning(f"Failed to stack {teacher_name} guidance for {layer_name}: {e}")
                        del guidance[layer_name][teacher_name]
        
        except Exception as e:
            self.logger.warning(f"Failed to get teacher guidance: {e}")
        
        return guidance
    
    def predict(self, 
                user_profiles: List[UserProfile], 
                item_profiles: List[ItemProfile]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions for user-item pairs.
        
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(user_profiles, item_profiles)
            return output.recommendations, output.confidence_scores
    
    def get_model_size(self) -> Dict[str, float]:
        """Get model size information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Calculate size for each layer
        embedding_params = sum(p.numel() for p in self.embedding_adapter.parameters())
        interaction_params = sum(p.numel() for p in self.interaction_adapter.parameters())
        reasoning_params = sum(p.numel() for p in self.reasoning_adapter.parameters())
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / 1024 / 1024,
            'embedding_layer_params': embedding_params,
            'interaction_layer_params': interaction_params,
            'reasoning_layer_params': reasoning_params
        }
    
    def update_distillation_weights(self, step: int, total_steps: int) -> None:
        """Update distillation weights during training (progressive training)."""
        self.training_step = step
        
        # Progressive distillation: start with more task loss, gradually increase distillation
        progress = step / total_steps
        
        # Adjust weights based on training progress
        base_alpha = self.config.alpha_ensemble
        base_beta = self.config.beta_llm
        base_gamma = self.config.gamma_task
        
        # Increase distillation weight as training progresses
        self.distillation_weights['alpha_ensemble'] = base_alpha * min(1.0, progress * 2)
        self.distillation_weights['beta_llm'] = base_beta * min(1.0, progress * 2)
        self.distillation_weights['gamma_task'] = base_gamma * (2 - progress)
        
        # Normalize weights
        total_weight = sum(self.distillation_weights.values())
        if total_weight > 0:
            for key in self.distillation_weights:
                self.distillation_weights[key] /= total_weight
