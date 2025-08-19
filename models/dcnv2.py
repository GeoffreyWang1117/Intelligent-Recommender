#!/usr/bin/env python3
"""
DCNv2 (Deep & Cross Network v2) 推荐算法实现

DCNv2是DCN的改进版本，通过改进的交叉网络和深度网络的结合，
更好地学习特征交互和非线性变换。
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Any, Optional
import logging
from sklearn.preprocessing import LabelEncoder

from .base_recommender import BaseRecommender

logger = logging.getLogger(__name__)


class CrossNetworkV2(nn.Module):
    """DCNv2的交叉网络"""
    
    def __init__(self, input_dim: int, num_layers: int = 3):
        super().__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        
        # 每层的权重矩阵和偏置
        self.cross_layers = nn.ModuleList([
            nn.Linear(input_dim, input_dim, bias=True) 
            for _ in range(num_layers)
        ])
        
        # 初始化权重
        for layer in self.cross_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, x0):
        """
        前向传播
        
        Args:
            x0: 初始输入 [batch_size, input_dim]
        
        Returns:
            交叉网络的输出
        """
        x_l = x0  # 第l层的输入
        
        for layer in self.cross_layers:
            # DCNv2改进: x_{l+1} = x_0 ⊙ (W_l * x_l + b_l) + x_l
            xl_transformed = layer(x_l)  # W_l * x_l + b_l
            x_l = x0 * xl_transformed + x_l  # 元素级乘法和残差连接
        
        return x_l


class DeepNetwork(nn.Module):
    """深度网络"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.5):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.deep_layers = nn.Sequential(*layers)
        
        # 权重初始化
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.deep_layers(x)


class DCNv2Model(nn.Module):
    """DCNv2模型"""
    
    def __init__(self, feature_dims: Dict[str, int], embedding_dim: int = 16,
                 cross_layers: int = 3, deep_layers: List[int] = [512, 256, 128],
                 dropout: float = 0.5):
        super().__init__()
        
        self.feature_dims = feature_dims
        self.embedding_dim = embedding_dim
        
        # 特征嵌入层
        self.embeddings = nn.ModuleDict()
        total_embedding_dim = 0
        
        for feature_name, feature_dim in feature_dims.items():
            if feature_dim > 1:  # 类别特征
                self.embeddings[feature_name] = nn.Embedding(feature_dim, embedding_dim)
                total_embedding_dim += embedding_dim
            else:  # 数值特征
                total_embedding_dim += 1
        
        self.total_embedding_dim = total_embedding_dim
        
        # 交叉网络
        self.cross_network = CrossNetworkV2(total_embedding_dim, cross_layers)
        
        # 深度网络
        self.deep_network = DeepNetwork(total_embedding_dim, deep_layers, dropout)
        
        # 最终输出层
        final_input_dim = total_embedding_dim + deep_layers[-1]  # 交叉网络输出 + 深度网络输出
        self.final_layer = nn.Sequential(
            nn.Linear(final_input_dim, 1),
            nn.Sigmoid()
        )
        
        logger.info(f"DCNv2模型初始化: 特征维度={feature_dims}, "
                   f"嵌入维度={embedding_dim}, 交叉层数={cross_layers}")
    
    def forward(self, categorical_features: Dict[str, torch.Tensor], 
                numerical_features: torch.Tensor = None):
        """
        前向传播
        
        Args:
            categorical_features: 类别特征字典
            numerical_features: 数值特征
        """
        # 特征嵌入
        embeddings = []
        
        for feature_name, feature_tensor in categorical_features.items():
            if feature_name in self.embeddings:
                emb = self.embeddings[feature_name](feature_tensor)
                embeddings.append(emb)
        
        # 添加数值特征
        if numerical_features is not None:
            embeddings.append(numerical_features.unsqueeze(-1))
        
        # 拼接所有嵌入
        x = torch.cat(embeddings, dim=-1)  # [batch_size, total_embedding_dim]
        
        # 交叉网络
        cross_output = self.cross_network(x)
        
        # 深度网络
        deep_output = self.deep_network(x)
        
        # 拼接交叉网络和深度网络输出
        final_input = torch.cat([cross_output, deep_output], dim=-1)
        
        # 最终预测
        output = self.final_layer(final_input)
        
        return output.squeeze(-1)


class DCNv2Dataset(Dataset):
    """DCNv2数据集"""
    
    def __init__(self, categorical_features: Dict[str, np.ndarray], 
                 numerical_features: Optional[np.ndarray] = None,
                 labels: Optional[np.ndarray] = None):
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.labels = labels
        
        # 检查数据长度一致性
        lengths = [len(feat) for feat in categorical_features.values()]
        if numerical_features is not None:
            lengths.append(len(numerical_features))
        if labels is not None:
            lengths.append(len(labels))
        
        assert len(set(lengths)) == 1, "所有特征的长度必须相同"
        self.length = lengths[0]
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # 类别特征
        cat_features = {
            name: torch.tensor(feat[idx], dtype=torch.long)
            for name, feat in self.categorical_features.items()
        }
        
        # 数值特征 - 如果没有数值特征，返回空tensor
        if self.numerical_features is not None:
            num_features = torch.tensor(self.numerical_features[idx], dtype=torch.float)
        else:
            num_features = torch.tensor([], dtype=torch.float)
        
        # 标签
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float)
        else:
            label = torch.tensor(0.0, dtype=torch.float)
        
        return cat_features, num_features, label


class DCNv2Recommender(BaseRecommender):
    """DCNv2推荐器"""
    
    def __init__(self, embedding_dim: int = 16, cross_layers: int = 3,
                 deep_layers: List[int] = [512, 256, 128], 
                 learning_rate: float = 0.001, epochs: int = 50,
                 batch_size: int = 256, dropout: float = 0.5,
                 early_stopping: bool = True, patience: int = 5,
                 **kwargs):
        super().__init__("DCNv2", **kwargs)
        
        self.embedding_dim = embedding_dim
        self.cross_layers = cross_layers
        self.deep_layers = deep_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.early_stopping = early_stopping
        self.patience = patience
        
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_encoders = {}
        self.feature_dims = {}
        self.all_items = set()
        self.user_item_matrix = {}
        
        logger.info(f"初始化DCNv2: 嵌入维度={embedding_dim}, "
                   f"交叉层数={cross_layers}, 深度层={deep_layers}")
    
    def _prepare_features(self, train_data: pd.DataFrame):
        """准备特征数据"""
        # 编码类别特征
        categorical_features = ['user_id', 'item_id']
        
        for feature in categorical_features:
            le = LabelEncoder()
            train_data[f'{feature}_encoded'] = le.fit_transform(train_data[feature])
            self.label_encoders[feature] = le
            self.feature_dims[feature] = len(le.classes_)
        
        # 构建用户-物品交互矩阵用于推荐
        self.all_items = set(train_data['item_id'].unique())
        for _, row in train_data.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            rating = row['rating']
            
            if user_id not in self.user_item_matrix:
                self.user_item_matrix[user_id] = {}
            self.user_item_matrix[user_id][item_id] = rating
        
        logger.info(f"特征维度: {self.feature_dims}")
        logger.info(f"用户数: {len(self.label_encoders['user_id'].classes_)}, "
                   f"物品数: {len(self.label_encoders['item_id'].classes_)}")
    
    def fit(self, train_data: pd.DataFrame, **kwargs) -> None:
        """训练模型"""
        logger.info("开始训练DCNv2模型...")
        
        # 准备特征
        self._prepare_features(train_data)
        
        # 创建模型
        self.model = DCNv2Model(
            feature_dims=self.feature_dims,
            embedding_dim=self.embedding_dim,
            cross_layers=self.cross_layers,
            deep_layers=self.deep_layers,
            dropout=self.dropout
        ).to(self.device)
        
        # 准备训练数据
        categorical_features = {
            'user_id': train_data['user_id_encoded'].values.astype(np.int64),
            'item_id': train_data['item_id_encoded'].values.astype(np.int64)
        }
        
        # 将评分标准化到[0,1]，确保float32类型
        labels = ((train_data['rating'].values - 1) / 4.0).astype(np.float32)  # 1-5 -> 0-1
        
        dataset = DCNv2Dataset(categorical_features, None, labels)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 优化器和损失函数
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # 训练循环
        self.model.train()
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            total_loss = 0.0
            num_batches = 0
            
            for cat_features, num_features, labels_batch in dataloader:
                # 移动到设备
                cat_features = {k: v.to(self.device) for k, v in cat_features.items()}
                labels_batch = labels_batch.to(self.device)
                
                # 处理数值特征 - 如果为空则设为None
                if num_features.numel() == 0:
                    num_features = None
                else:
                    num_features = num_features.to(self.device)
                
                optimizer.zero_grad()
                
                # 前向传播
                predictions = self.model(cat_features, num_features)
                loss = criterion(predictions, labels_batch)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")
            
            # 早停检查
            if self.early_stopping:
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.info(f"早停在第 {epoch + 1} 轮")
                        break
        
        self.is_trained = True
        logger.info("DCNv2训练完成")
    
    def predict(self, user_id: int, item_id: int) -> float:
        """预测用户对物品的评分"""
        if not self.is_trained or self.model is None:
            return 3.0
        
        # 检查用户和物品是否在训练集中
        if user_id not in self.label_encoders['user_id'].classes_:
            return 3.0
        if item_id not in self.label_encoders['item_id'].classes_:
            return 3.0
        
        self.model.eval()
        with torch.no_grad():
            # 编码特征
            user_encoded = self.label_encoders['user_id'].transform([user_id])[0]
            item_encoded = self.label_encoders['item_id'].transform([item_id])[0]
            
            # 准备输入
            cat_features = {
                'user_id': torch.tensor([user_encoded], dtype=torch.long).to(self.device),
                'item_id': torch.tensor([item_encoded], dtype=torch.long).to(self.device)
            }
            
            # 预测
            prediction = self.model(cat_features, None).item()
            
            # 转换回1-5评分
            return 1.0 + prediction * 4.0
    
    def get_user_recommendations(self, user_id: int, top_k: int = 10) -> List[Dict[str, Any]]:
        """获取用户推荐"""
        if not self.is_trained or self.model is None:
            return []
        
        if user_id not in self.label_encoders['user_id'].classes_:
            return []
        
        # 获取用户已交互的物品
        user_items = set()
        if user_id in self.user_item_matrix:
            user_items = set(self.user_item_matrix[user_id].keys())
        
        # 候选物品（未交互的物品）
        candidate_items = self.all_items - user_items
        if not candidate_items:
            return []
        
        self.model.eval()
        with torch.no_grad():
            recommendations = []
            
            # 批量预测
            user_encoded = self.label_encoders['user_id'].transform([user_id])[0]
            
            batch_size = min(512, len(candidate_items))
            item_batches = [list(candidate_items)[i:i + batch_size] 
                           for i in range(0, len(candidate_items), batch_size)]
            
            for item_batch in item_batches:
                # 编码物品
                item_encoded_batch = [
                    self.label_encoders['item_id'].transform([item])[0] 
                    for item in item_batch
                ]
                
                # 准备输入
                batch_size_actual = len(item_batch)
                cat_features = {
                    'user_id': torch.tensor([user_encoded] * batch_size_actual, 
                                          dtype=torch.long).to(self.device),
                    'item_id': torch.tensor(item_encoded_batch, 
                                          dtype=torch.long).to(self.device)
                }
                
                # 批量预测
                predictions = self.model(cat_features, None)
                
                # 收集推荐
                for i, (item_id, pred) in enumerate(zip(item_batch, predictions)):
                    score = 1.0 + pred.item() * 4.0  # 转换为1-5评分
                    recommendations.append({
                        'item_id': item_id,
                        'score': score
                    })
            
            # 排序并返回top-k
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            return recommendations[:top_k]


# 工厂函数
def create_dcnv2_recommender(**kwargs) -> DCNv2Recommender:
    """创建DCNv2推荐器"""
    return DCNv2Recommender(**kwargs)


if __name__ == "__main__":
    # 测试代码
    import pandas as pd
    
    # 创建测试数据
    data = {
        'user_id': [1, 1, 1, 2, 2, 3],
        'item_id': [1, 2, 3, 1, 4, 2],
        'rating': [5, 4, 3, 4, 5, 3],
        'timestamp': [1, 2, 3, 1, 2, 1]
    }
    
    train_data = pd.DataFrame(data)
    
    # 创建推荐器
    recommender = DCNv2Recommender(
        embedding_dim=16,
        cross_layers=2,
        deep_layers=[128, 64],
        epochs=5
    )
    
    # 训练
    recommender.fit(train_data)
    
    # 推荐
    recs = recommender.get_user_recommendations(1, top_k=3)
    print("推荐结果:", recs)
