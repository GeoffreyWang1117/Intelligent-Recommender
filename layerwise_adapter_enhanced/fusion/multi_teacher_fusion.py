"""
多Teacher融合核心模块

实现基于LayerwiseAdapter的多Teacher融合架构
支持Ensemble + LLM Teacher的深度融合
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Union, Optional
import time

class MultiTeacherFusion(nn.Module):
    """多Teacher融合模块
    
    核心功能：
    1. 整合多个Teacher的知识
    2. 提供统一的融合接口
    3. 支持Fisher引导的权重分配
    4. 保持LayerwiseAdapter的高效架构
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 融合配置
        self.embedding_dim = config.get('embedding_dim', 64)
        self.teacher_count = config.get('teacher_count', 2)  # Ensemble + LLM
        self.fusion_hidden_dim = config.get('fusion_hidden_dim', 128)
        
        # 融合层架构
        self._build_fusion_layers()
        
        # Teacher权重管理
        self.teacher_weights = nn.Parameter(
            torch.ones(self.teacher_count, device=self.device) / self.teacher_count
        )
        
        # 自适应融合门控
        self.adaptive_gate = nn.Sequential(
            nn.Linear(self.embedding_dim * self.teacher_count, self.fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.fusion_hidden_dim, self.teacher_count),
            nn.Softmax(dim=-1)
        )
        
        # 特征对齐层
        self.feature_alignment = nn.ModuleDict({
            f'teacher_{i}': nn.Linear(self.embedding_dim, self.embedding_dim)
            for i in range(self.teacher_count)
        })
        
        # 融合输出层
        self.fusion_output = nn.Sequential(
            nn.Linear(self.embedding_dim, self.fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.fusion_hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 损失函数
        self.mse_loss = nn.MSELoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        
        print(f"✅ MultiTeacherFusion初始化完成")
        print(f"   - 嵌入维度: {self.embedding_dim}")
        print(f"   - Teacher数量: {self.teacher_count}")
        print(f"   - 融合隐藏维度: {self.fusion_hidden_dim}")
        
    def _build_fusion_layers(self):
        """构建融合层架构"""
        # 注意力融合层
        self.attention_fusion = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # 跨Teacher交互层
        self.cross_teacher_interaction = nn.Sequential(
            nn.Linear(self.embedding_dim * self.teacher_count, self.fusion_hidden_dim),
            nn.LayerNorm(self.fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.fusion_hidden_dim, self.embedding_dim)
        )
        
        # Fisher引导融合层
        self.fisher_guided_fusion = nn.Sequential(
            nn.Linear(self.embedding_dim, self.fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.fusion_hidden_dim // 2, self.embedding_dim)
        )
    
    def forward(self, 
                teacher_embeddings: List[torch.Tensor],
                teacher_predictions: List[float],
                fisher_weights: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """前向传播"""
        try:
            batch_size = 1  # 当前支持单样本
            
            # 1. 特征对齐
            aligned_embeddings = []
            for i, embedding in enumerate(teacher_embeddings):
                if embedding.dim() == 1:
                    embedding = embedding.unsqueeze(0)  # 添加batch维度
                
                # 对齐维度
                if embedding.size(-1) != self.embedding_dim:
                    if embedding.size(-1) < self.embedding_dim:
                        padding = torch.zeros(
                            embedding.size(0), 
                            self.embedding_dim - embedding.size(-1),
                            device=self.device
                        )
                        embedding = torch.cat([embedding, padding], dim=-1)
                    else:
                        embedding = embedding[:, :self.embedding_dim]
                
                # 特征对齐
                teacher_key = f'teacher_{min(i, self.teacher_count-1)}'
                if teacher_key in self.feature_alignment:
                    aligned_embedding = self.feature_alignment[teacher_key](embedding)
                else:
                    aligned_embedding = embedding
                
                aligned_embeddings.append(aligned_embedding)
            
            # 2. 注意力融合
            if len(aligned_embeddings) >= 2:
                # 堆叠Teacher嵌入
                stacked_embeddings = torch.stack(aligned_embeddings, dim=1)  # [batch, teachers, dim]
                
                # 多头注意力
                attended_embeddings, attention_weights = self.attention_fusion(
                    stacked_embeddings, stacked_embeddings, stacked_embeddings
                )
                
                # 平均池化
                attention_fused = attended_embeddings.mean(dim=1)  # [batch, dim]
            else:
                attention_fused = aligned_embeddings[0] if aligned_embeddings else torch.zeros(
                    batch_size, self.embedding_dim, device=self.device
                )
            
            # 3. 跨Teacher交互
            if len(aligned_embeddings) >= 2:
                concatenated = torch.cat(aligned_embeddings, dim=-1)  # [batch, teachers*dim]
                interaction_output = self.cross_teacher_interaction(concatenated)
            else:
                interaction_output = attention_fused
            
            # 4. Fisher引导融合
            if fisher_weights is not None and len(aligned_embeddings) >= 2:
                # 应用Fisher权重
                fisher_weighted = torch.zeros_like(aligned_embeddings[0])
                total_weight = 0.0
                
                for i, (embedding, weight) in enumerate(zip(aligned_embeddings, fisher_weights)):
                    fisher_weighted += embedding * weight
                    total_weight += weight
                
                if total_weight > 0:
                    fisher_weighted = fisher_weighted / total_weight
                
                fisher_output = self.fisher_guided_fusion(fisher_weighted)
            else:
                fisher_output = attention_fused
            
            # 5. 自适应门控融合
            if len(aligned_embeddings) >= 2:
                gate_input = torch.cat(aligned_embeddings, dim=-1)
                adaptive_weights = self.adaptive_gate(gate_input)  # [batch, teachers]
                
                adaptive_fused = torch.zeros_like(aligned_embeddings[0])
                for i, (embedding, weight) in enumerate(zip(aligned_embeddings, adaptive_weights[0])):
                    adaptive_fused += embedding * weight
            else:
                adaptive_fused = attention_fused
            
            # 6. 最终融合
            # 组合多种融合策略
            final_embedding = (
                attention_fused * 0.3 +
                interaction_output * 0.3 + 
                fisher_output * 0.2 +
                adaptive_fused * 0.2
            )
            
            # 7. 输出预测
            fusion_prediction = self.fusion_output(final_embedding)  # [batch, 1]
            
            # 转换为评分范围 [1, 5]
            fusion_rating = fusion_prediction * 4.0 + 1.0  # [1, 5]
            
            return {
                'fusion_embedding': final_embedding,
                'fusion_prediction': fusion_rating,
                'attention_weights': attention_weights if 'attention_weights' in locals() else None,
                'adaptive_weights': adaptive_weights if 'adaptive_weights' in locals() else None,
                'teacher_embeddings': aligned_embeddings
            }
            
        except Exception as e:
            print(f"❌ MultiTeacherFusion前向传播失败: {e}")
            # 返回默认输出
            return {
                'fusion_embedding': torch.zeros(batch_size, self.embedding_dim, device=self.device),
                'fusion_prediction': torch.tensor([[3.0]], device=self.device),
                'attention_weights': None,
                'adaptive_weights': None,
                'teacher_embeddings': []
            }
    
    def compute_fusion_loss(self,
                           fusion_output: Dict[str, torch.Tensor],
                           teacher_predictions: List[float],
                           target_rating: float,
                           loss_weights: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """计算融合损失"""
        try:
            if loss_weights is None:
                loss_weights = {
                    'prediction': 0.6,
                    'consistency': 0.3,
                    'regularization': 0.1
                }
            
            total_loss = torch.tensor(0.0, device=self.device)
            
            # 1. 预测损失
            fusion_pred = fusion_output['fusion_prediction']
            target_tensor = torch.tensor([[target_rating]], device=self.device)
            prediction_loss = self.mse_loss(fusion_pred, target_tensor)
            total_loss += prediction_loss * loss_weights['prediction']
            
            # 2. Teacher一致性损失
            if len(teacher_predictions) >= 2:
                teacher_tensor = torch.tensor(teacher_predictions, device=self.device)
                teacher_mean = teacher_tensor.mean()
                teacher_var = teacher_tensor.var()
                
                # 融合预测应该接近Teacher平均值
                consistency_loss = self.mse_loss(
                    fusion_pred.squeeze(), 
                    teacher_mean.unsqueeze(0)
                )
                total_loss += consistency_loss * loss_weights['consistency']
            
            # 3. 正则化损失
            # Teacher权重正则化
            weight_reg = torch.sum(self.teacher_weights ** 2)
            total_loss += weight_reg * loss_weights['regularization']
            
            return total_loss
            
        except Exception as e:
            print(f"❌ 计算融合损失失败: {e}")
            return torch.tensor(0.0, device=self.device)
    
    def update_teacher_weights(self, teacher_performance: Dict[str, float]):
        """更新Teacher权重"""
        try:
            with torch.no_grad():
                # 基于性能更新权重
                performance_values = list(teacher_performance.values())
                if len(performance_values) == self.teacher_count:
                    performance_tensor = torch.tensor(performance_values, device=self.device)
                    # 归一化权重
                    updated_weights = torch.softmax(performance_tensor, dim=0)
                    self.teacher_weights.data = updated_weights
                    
        except Exception as e:
            print(f"❌ 更新Teacher权重失败: {e}")
    
    def get_fusion_info(self) -> Dict[str, Any]:
        """获取融合信息"""
        return {
            'embedding_dim': self.embedding_dim,
            'teacher_count': self.teacher_count,
            'fusion_hidden_dim': self.fusion_hidden_dim,
            'teacher_weights': self.teacher_weights.detach().cpu().tolist(),
            'parameter_count': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
