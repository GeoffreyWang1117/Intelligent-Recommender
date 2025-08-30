"""
Three-layer adapter implementation for the layerwise knowledge distillation system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import math

from .base import BaseAdapter, ModelConfig


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)
        
        # Linear transformations and reshape
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads and put through final linear layer
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.w_o(context)
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding for sequence modeling."""
    
    def __init__(self, d_model: int, max_length: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe = getattr(self, 'pe')[:x.size(0), :].to(x.device)
        return x + pe


class EmbeddingAdapter(BaseAdapter):
    """Layer 1: Embedding Adapter for feature alignment and embedding learning."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(
            input_dim=config.embedding_dim,
            output_dim=config.hidden_dim,
            layer_name="embedding_adapter"
        )
        
        self.config = config
        
        # User and item embedding tables
        self.user_embedding = nn.Embedding(config.num_users, config.embedding_dim)
        self.item_embedding = nn.Embedding(config.num_items, config.embedding_dim)
        
        # Feature transformation layers
        self.user_transform = nn.Linear(config.embedding_dim, config.hidden_dim)
        self.item_transform = nn.Linear(config.embedding_dim, config.hidden_dim)
        
        # Multi-head attention for feature fusion
        self.attention = MultiHeadAttention(config.hidden_dim, config.num_heads)
        
        # Teacher alignment layers
        self.ensemble_align = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.llm_align = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        
        for module in [self.user_transform, self.item_transform, 
                      self.ensemble_align, self.llm_align]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, 
                teacher_guidance: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        Forward pass for embedding adapter.
        
        Args:
            x: Input tensor [batch_size, seq_len, feature_dim]
            teacher_guidance: Teacher embeddings for alignment
        """
        batch_size, seq_len = x.size(0), x.size(1)
        _ = batch_size, seq_len  # Suppress unused variable warnings
        
        # Extract user and item IDs (assuming they are in the input)
        user_ids = x[:, :, 0].long()  # First feature is user ID
        item_ids = x[:, :, 1].long()  # Second feature is item ID
        
        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Transform embeddings
        user_emb = self.user_transform(user_emb)
        item_emb = self.item_transform(item_emb)
        
        # Combine user and item embeddings
        combined_emb = user_emb + item_emb
        
        # Apply attention for feature fusion
        attended_emb = self.attention(combined_emb, combined_emb, combined_emb)
        
        # Residual connection and normalization
        output = self.layer_norm(combined_emb + attended_emb)
        output = self.dropout(output)
        
        # Teacher alignment if guidance is provided
        if teacher_guidance is not None:
            if 'ensemble' in teacher_guidance:
                ensemble_aligned = self.ensemble_align(output)
                output = output + 0.1 * ensemble_aligned
            
            if 'llm' in teacher_guidance:
                llm_aligned = self.llm_align(output)
                output = output + 0.1 * llm_aligned
        
        return output
    
    def compute_distillation_loss(self, student_output: torch.Tensor,
                                teacher_outputs: Dict[str, torch.Tensor],
                                temperature: float = 3.0) -> torch.Tensor:
        """Compute knowledge distillation loss for embedding layer."""
        total_loss = torch.tensor(0.0, device=student_output.device)
        
        # Ensure student output has the right shape
        if len(student_output.shape) > 2:
            student_output = student_output.view(-1, student_output.size(-1))
        
        # Distillation from ensemble teacher
        if 'ensemble' in teacher_outputs:
            ensemble_emb = teacher_outputs['ensemble']
            if len(ensemble_emb.shape) > 2:
                ensemble_emb = ensemble_emb.view(-1, ensemble_emb.size(-1))
            
            # Align dimensions if needed
            if ensemble_emb.size(-1) != student_output.size(-1):
                ensemble_emb = self.ensemble_align(ensemble_emb)
            
            ensemble_loss = F.mse_loss(student_output, ensemble_emb)
            total_loss += ensemble_loss
        
        # Distillation from LLM teacher
        if 'llm' in teacher_outputs:
            llm_emb = teacher_outputs['llm']
            if len(llm_emb.shape) > 2:
                llm_emb = llm_emb.view(-1, llm_emb.size(-1))
            
            # Align dimensions if needed
            if llm_emb.size(-1) != student_output.size(-1):
                llm_emb = self.llm_align(llm_emb)
            
            llm_loss = F.mse_loss(student_output, llm_emb)
            total_loss += llm_loss
        
        return total_loss


class InteractionAdapter(BaseAdapter):
    """Layer 2: Interaction Adapter for user-item interaction modeling."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(
            input_dim=config.hidden_dim,
            output_dim=config.hidden_dim,
            layer_name="interaction_adapter"
        )
        
        self.config = config
        
        # Cross-attention for user-item interaction
        self.cross_attention = MultiHeadAttention(config.hidden_dim, config.num_heads)
        self.self_attention = MultiHeadAttention(config.hidden_dim, config.num_heads)
        
        # Positional encoding for sequence modeling
        self.pos_encoding = PositionalEncoding(config.hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.norm3 = nn.LayerNorm(config.hidden_dim)
        
        # Teacher alignment
        self.teacher_fusion = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor,
                teacher_guidance: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        Forward pass for interaction adapter.
        
        Args:
            x: Input tensor from embedding adapter [batch_size, seq_len, hidden_dim]
            teacher_guidance: Teacher interaction patterns
        """
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Self-attention for sequence modeling
        attn_output = self.self_attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention for interaction modeling (simplified)
        cross_output = self.cross_attention(x, x, x)
        x = self.norm2(x + self.dropout(cross_output))
        
        # Teacher guidance integration
        if teacher_guidance is not None:
            # Combine teacher guidance
            teacher_signals = []
            for teacher_name, teacher_emb in teacher_guidance.items():
                if teacher_emb.size(-1) == x.size(-1):
                    teacher_signals.append(teacher_emb)
            
            if teacher_signals:
                combined_teacher = torch.stack(teacher_signals, dim=0).mean(dim=0)
                if combined_teacher.shape == x.shape:
                    fused = self.teacher_fusion(torch.cat([x, combined_teacher], dim=-1))
                    x = x + 0.1 * fused
        
        # Feed-forward network
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))
        
        return x
    
    def compute_distillation_loss(self, student_output: torch.Tensor,
                                teacher_outputs: Dict[str, torch.Tensor],
                                temperature: float = 3.0) -> torch.Tensor:
        """Compute distillation loss for interaction patterns."""
        total_loss = torch.tensor(0.0, device=student_output.device)
        
        for teacher_name, teacher_output in teacher_outputs.items():
            # Compute attention-based similarity loss
            if teacher_output.shape == student_output.shape:
                # Use cosine similarity loss for interaction patterns
                student_norm = F.normalize(student_output, p=2, dim=-1)
                teacher_norm = F.normalize(teacher_output, p=2, dim=-1)
                
                similarity_loss = 1 - torch.cosine_similarity(
                    student_norm.view(-1, student_norm.size(-1)),
                    teacher_norm.view(-1, teacher_norm.size(-1)),
                    dim=-1
                ).mean()
                
                total_loss += similarity_loss
        
        return total_loss


class ReasoningAdapter(BaseAdapter):
    """Layer 3: Reasoning Adapter for final prediction and explainability."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(
            input_dim=config.hidden_dim,
            output_dim=1,  # Final prediction score
            layer_name="reasoning_adapter"
        )
        
        self.config = config
        
        # Multi-task prediction heads
        self.prediction_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1)
        )
        
        # Confidence estimation head
        self.confidence_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Explanation generation (simplified)
        self.explanation_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4)
        )
        
        # Teacher alignment for final predictions
        self.teacher_weight_net = nn.Sequential(
            nn.Linear(config.hidden_dim, config.num_heads),
            nn.Softmax(dim=-1)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
    
    def forward(self, x: torch.Tensor,
                teacher_guidance: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        Forward pass for reasoning adapter.
        
        Args:
            x: Input tensor from interaction adapter [batch_size, seq_len, hidden_dim]
            teacher_guidance: Teacher prediction guidance
        """
        # Normalize input
        x = self.layer_norm(x)
        
        # Global pooling for final representation
        if len(x.shape) > 2:
            x = x.mean(dim=1)  # Average pooling over sequence
        
        # Generate predictions
        predictions = self.prediction_head(x)
        confidence = self.confidence_head(x)
        explanation_emb = self.explanation_head(x)
        
        # Suppress unused variable warnings for now
        _ = confidence, explanation_emb
        
        # Teacher-guided prediction refinement
        if teacher_guidance is not None:
            teacher_weights = self.teacher_weight_net(x)
            
            # Weighted combination of teacher guidance
            refined_pred = predictions
            weight_idx = 0
            
            for teacher_name, teacher_pred in teacher_guidance.items():
                if teacher_pred.size(-1) == 1 and weight_idx < teacher_weights.size(-1):
                    weight = teacher_weights[:, weight_idx:weight_idx+1]
                    refined_pred += weight * teacher_pred
                    weight_idx += 1
            
            predictions = refined_pred
        
        # Combine outputs (prediction is the main output)
        return predictions
    
    def compute_distillation_loss(self, student_output: torch.Tensor,
                                teacher_outputs: Dict[str, torch.Tensor],
                                temperature: float = 3.0) -> torch.Tensor:
        """Compute distillation loss for final predictions."""
        total_loss = torch.tensor(0.0, device=student_output.device)
        
        for teacher_name, teacher_pred in teacher_outputs.items():
            if teacher_pred.size(-1) == student_output.size(-1):
                # Use temperature-scaled KL divergence for final predictions
                student_soft = F.log_softmax(student_output / temperature, dim=-1)
                teacher_soft = F.softmax(teacher_pred / temperature, dim=-1)
                
                kl_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
                total_loss += kl_loss * (temperature ** 2)
        
        return total_loss
    
    def get_confidence(self, x: torch.Tensor) -> torch.Tensor:
        """Get confidence scores for predictions."""
        x = self.layer_norm(x)
        if len(x.shape) > 2:
            x = x.mean(dim=1)
        return self.confidence_head(x)
    
    def get_explanation_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get explanation embeddings."""
        x = self.layer_norm(x)
        if len(x.shape) > 2:
            x = x.mean(dim=1)
        return self.explanation_head(x)
