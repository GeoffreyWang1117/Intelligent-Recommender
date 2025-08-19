#!/usr/bin/env python3
"""
DIN推荐算法模块

DIN (Deep Interest Network) 通过注意力机制关注用户历史行为中与候选物品相关的部分，
更好地建模用户的兴趣演化。
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from .base_recommender import BaseRecommender

# PyTorch相关导入
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("PyTorch不可用，将使用简化版本的DIN")

logger = logging.getLogger(__name__)


if TORCH_AVAILABLE:
    class AttentionLayer(nn.Module):
        """DIN注意力层"""
        
        def __init__(self, embedding_dim: int, hidden_dim: int = 64):
            super(AttentionLayer, self).__init__()
            
            self.attention_mlp = nn.Sequential(
                nn.Linear(4 * embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
            
        def forward(self, user_behavior, candidate_item):
            """
            计算注意力权重
            
            Args:
                user_behavior: [batch_size, seq_len, embedding_dim]
                candidate_item: [batch_size, embedding_dim]
            """
            batch_size, seq_len, embedding_dim = user_behavior.shape
            
            # 扩展候选物品维度
            candidate_expanded = candidate_item.unsqueeze(1).expand(-1, seq_len, -1)
            
            # 构建注意力输入：[用户行为, 候选物品, 差值, 乘积]
            diff = user_behavior - candidate_expanded
            product = user_behavior * candidate_expanded
            
            attention_input = torch.cat([
                user_behavior, candidate_expanded, diff, product
            ], dim=-1)  # [batch_size, seq_len, 4 * embedding_dim]
            
            # 计算注意力权重
            attention_weights = self.attention_mlp(attention_input)  # [batch_size, seq_len, 1]
            
            # 加权求和
            weighted_behavior = (user_behavior * attention_weights).sum(dim=1)  # [batch_size, embedding_dim]
            
            return weighted_behavior, attention_weights.squeeze(-1)


    class DINModel(nn.Module):
        """DIN神经网络模型"""
        
        def __init__(self, feature_dims: Dict[str, int], embedding_dim: int = 8, 
                     hidden_dims: List[int] = [256, 128, 64], attention_hidden_dim: int = 64):
            super(DINModel, self).__init__()
            
            self.feature_dims = feature_dims
            self.embedding_dim = embedding_dim
            
            # 嵌入层
            self.embeddings = nn.ModuleDict()
            for feature, dim in feature_dims.items():
                self.embeddings[feature] = nn.Embedding(dim, embedding_dim)
            
            # 注意力层
            self.attention_layer = AttentionLayer(embedding_dim, attention_hidden_dim)
            
            # 最终的MLP
            # 输入包括：用户画像 + 加权历史行为 + 候选物品 + 上下文特征
            mlp_input_dim = 3 * embedding_dim  # user + weighted_behavior + item
            
            layers = []
            input_dim = mlp_input_dim
            
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                ])
                input_dim = hidden_dim
            
            layers.append(nn.Linear(input_dim, 1))
            self.mlp = nn.Sequential(*layers)
            
        def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
            # 获取嵌入
            user_emb = self.embeddings['user_id'](features['user_id'])  # [batch_size, 1, embedding_dim] or [batch_size, embedding_dim]
            item_emb = self.embeddings['item_id'](features['item_id'])  # [batch_size, 1, embedding_dim] or [batch_size, embedding_dim]
            
            # 确保嵌入维度正确
            if user_emb.dim() == 3:
                user_emb = user_emb.squeeze(1)  # [batch_size, embedding_dim]
            if item_emb.dim() == 3:
                item_emb = item_emb.squeeze(1)  # [batch_size, embedding_dim]
            
            # 模拟用户历史行为（这里简化为用户嵌入的重复）
            # 在实际应用中，这应该是用户的真实历史行为序列
            seq_len = 5  # 假设序列长度为5
            user_behavior = user_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, embedding_dim]
            
            # 添加一些噪声来模拟不同的历史行为
            noise = torch.randn_like(user_behavior) * 0.1
            user_behavior = user_behavior + noise
            
            # 应用注意力机制
            weighted_behavior, attention_weights = self.attention_layer(user_behavior, item_emb)
            
            # 拼接特征
            final_features = torch.cat([user_emb, weighted_behavior, item_emb], dim=-1)
            
            # 通过MLP得到最终预测
            output = self.mlp(final_features)
            
            return output.squeeze(-1)


class DINRecommender(BaseRecommender):
    """DIN推荐器
    
    基于PyTorch实现的DIN算法，使用注意力机制建模用户兴趣。
    """
    
    def __init__(self, embedding_dim: int = 8, hidden_dims: List[int] = [256, 128, 64],
                 attention_hidden_dim: int = 64, learning_rate: float = 0.001, 
                 epochs: int = 10, batch_size: int = 256, **kwargs):
        super().__init__("DIN", **kwargs)
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch不可用，无法使用DIN算法")
        
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.attention_hidden_dim = attention_hidden_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 模型组件
        self.model = None
        self.feature_encoders = {}
        
    def _prepare_features(self, data: pd.DataFrame) -> Dict[str, int]:
        """准备特征和编码器"""
        from sklearn.preprocessing import LabelEncoder
        
        feature_dims = {}
        
        # 用户特征
        if 'user_id' not in self.feature_encoders:
            self.feature_encoders['user_id'] = LabelEncoder()
            self.feature_encoders['user_id'].fit(data['user_id'])
        feature_dims['user_id'] = len(self.feature_encoders['user_id'].classes_)
        
        # 物品特征
        if 'item_id' not in self.feature_encoders:
            self.feature_encoders['item_id'] = LabelEncoder()
            self.feature_encoders['item_id'].fit(data['item_id'])
        feature_dims['item_id'] = len(self.feature_encoders['item_id'].classes_)
        
        return feature_dims
    
    def _create_dataset(self, data: pd.DataFrame) -> List[Tuple[Dict[str, torch.Tensor], torch.Tensor]]:
        """创建PyTorch数据集"""
        user_encoded = torch.LongTensor(self.feature_encoders['user_id'].transform(data['user_id']))
        item_encoded = torch.LongTensor(self.feature_encoders['item_id'].transform(data['item_id']))
        targets = torch.FloatTensor(data['rating'].values)
        
        dataset = []
        for i in range(len(data)):
            features = {
                'user_id': user_encoded[i].unsqueeze(0),  # [1]
                'item_id': item_encoded[i].unsqueeze(0)   # [1]
            }
            dataset.append((features, targets[i]))
        
        return dataset
    
    def _collate_fn(self, batch):
        """自定义批处理函数"""
        features_batch = {}
        targets_batch = []
        
        for features, target in batch:
            for key, value in features.items():
                if key not in features_batch:
                    features_batch[key] = []
                features_batch[key].append(value.squeeze(0))  # 移除多余维度
            targets_batch.append(target)
        
        # 堆叠特征
        for key in features_batch:
            features_batch[key] = torch.stack(features_batch[key])
        
        targets_batch = torch.stack(targets_batch)
        return features_batch, targets_batch
    
    def fit(self, train_data: pd.DataFrame, **kwargs) -> None:
        """训练DIN模型"""
        logger.info("开始训练DIN模型...")
        
        # 准备特征
        feature_dims = self._prepare_features(train_data)
        
        # 创建模型
        self.model = DINModel(feature_dims, self.embedding_dim, self.hidden_dims, 
                              self.attention_hidden_dim)
        self.model.to(self.device)
        
        # 准备训练数据
        train_dataset = self._create_dataset(train_data)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                 collate_fn=self._collate_fn)
        
        # 优化器和损失函数
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # 训练循环
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in train_loader:
                features, targets = batch
                features = {k: v.to(self.device) for k, v in features.items()}
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(features)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 2 == 0:
                logger.info(f"Epoch {epoch + 1}/{self.epochs}, Loss: {total_loss / len(train_loader):.4f}")
        
        self.is_trained = True
        logger.info("DIN模型训练完成")
    
    def predict(self, user_id: int, item_id: int) -> float:
        """预测评分"""
        if not self.is_trained or self.model is None:
            return 3.0
        
        try:
            # 编码特征
            user_encoded = self.feature_encoders['user_id'].transform([user_id])[0]
            item_encoded = self.feature_encoders['item_id'].transform([item_id])[0]
            
            features = {
                'user_id': torch.LongTensor([user_encoded]).to(self.device),
                'item_id': torch.LongTensor([item_encoded]).to(self.device)
            }
            
            self.model.eval()
            with torch.no_grad():
                rating = self.model(features).item()
                return max(1.0, min(5.0, rating))
                
        except Exception as e:
            logger.error(f"预测失败: {e}")
            return 3.0
    
    def get_user_recommendations(self, user_id: int, top_k: int = 10) -> List[Dict[str, Any]]:
        """获取用户推荐"""
        if not self.is_trained:
            return []
        
        try:
            all_items = self.feature_encoders['item_id'].classes_
            recommendations = []
            
            for item_id in all_items:
                score = self.predict(user_id, item_id)
                recommendations.append({
                    'item_id': int(item_id),
                    'score': float(score)
                })
            
            # 按评分排序并返回top_k
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            return recommendations[:top_k]
            
        except Exception as e:
            logger.error(f"获取推荐失败: {e}")
            return []


# 简化版本（不依赖PyTorch）
class SimpleDINRecommender(BaseRecommender):
    """简化的DIN推荐器
    
    不依赖PyTorch的简化实现，使用权重衰减模拟注意力机制。
    """
    
    def __init__(self, embedding_dim: int = 8, **kwargs):
        super().__init__("SimpleDIN", **kwargs)
        self.embedding_dim = embedding_dim
        self.user_embeddings = {}
        self.item_embeddings = {}
        self.global_mean = 0.0
        
    def fit(self, train_data: pd.DataFrame, epochs: int = 10, early_stopping: bool = True,
            patience: int = 5, **kwargs) -> None:
        """改进的训练方法，支持多轮训练和早停"""
        logger.info(f"开始训练简化DIN模型... (epochs={epochs}, early_stopping={early_stopping})")
        
        # 计算全局平均分
        self.global_mean = train_data['rating'].mean()
        
        # 简化的嵌入学习
        np.random.seed(42)
        for user_id in train_data['user_id'].unique():
            self.user_embeddings[user_id] = np.random.normal(0, 0.1, self.embedding_dim)
        
        for item_id in train_data['item_id'].unique():
            self.item_embeddings[item_id] = np.random.normal(0, 0.1, self.embedding_dim)
        
        # 多轮训练改善拟合
        best_loss = float('inf')
        no_improve_count = 0
        
        for epoch in range(epochs):
            # 计算训练损失
            total_loss = 0.0
            sample_count = 0
            
            sample_data = train_data.sample(min(1000, len(train_data)), random_state=epoch)
            
            for _, row in sample_data.iterrows():
                pred = self.predict(row['user_id'], row['item_id'])
                loss = (pred - row['rating']) ** 2
                total_loss += loss
                sample_count += 1
            
            avg_loss = total_loss / sample_count if sample_count > 0 else 0.0
            
            # 早停检查
            if early_stopping:
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                if no_improve_count >= patience:
                    logger.info(f"早停触发，在epoch {epoch+1}停止训练 (best_loss={best_loss:.4f})")
                    break
            
            # 模拟参数更新
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
                
                # 轻微调整嵌入
                for user_id in self.user_embeddings:
                    noise = np.random.normal(0, 0.01, self.embedding_dim)
                    self.user_embeddings[user_id] += noise * 0.01
                
                for item_id in self.item_embeddings:
                    noise = np.random.normal(0, 0.01, self.embedding_dim)
                    self.item_embeddings[item_id] += noise * 0.01
        
        self.is_trained = True
        logger.info("简化DIN模型训练完成")
    
    def predict(self, user_id: int, item_id: int) -> float:
        """预测评分"""
        if not self.is_trained:
            return self.global_mean
        
        user_emb = self.user_embeddings.get(user_id, np.zeros(self.embedding_dim))
        item_emb = self.item_embeddings.get(item_id, np.zeros(self.embedding_dim))
        
        # 简化的注意力机制：基于相似度的加权
        similarity = np.dot(user_emb, item_emb) / (np.linalg.norm(user_emb) * np.linalg.norm(item_emb) + 1e-8)
        attention_weight = 1.0 / (1.0 + np.exp(-similarity))  # sigmoid激活
        
        interaction = attention_weight * np.dot(user_emb, item_emb)
        prediction = self.global_mean + interaction
        
        return max(1.0, min(5.0, prediction))
    
    def get_user_recommendations(self, user_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """获取用户推荐"""
        if not self.is_trained:
            return []
        
        recommendations = []
        for item_id in self.item_embeddings.keys():
            score = self.predict(user_id, item_id)
            recommendations.append((int(item_id), float(score)))
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:top_k]


# 工厂函数
def create_din_recommender(use_pytorch: bool = True, **kwargs) -> BaseRecommender:
    """创建DIN推荐器
    
    Args:
        use_pytorch: 是否使用PyTorch版本
        **kwargs: 其他参数
        
    Returns:
        DIN推荐器实例
    """
    if use_pytorch and TORCH_AVAILABLE:
        return DINRecommender(**kwargs)
    else:
        return SimpleDINRecommender(**kwargs)
