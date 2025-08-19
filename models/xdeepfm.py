#!/usr/bin/env python3
"""
xDeepFM (eXtreme Deep Factorization Machine) 推荐算法实现

xDeepFM结合了FM、DNN和CIN（Compressed Interaction Network）的优势，
能够在向量级别上显式地学习特征交互。
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


class CompressedInteractionNetwork(nn.Module):
    """压缩交互网络(CIN)"""
    
    def __init__(self, field_dims: List[int], layer_sizes: List[int], 
                 split_half: bool = True):
        super().__init__()
        
        self.layer_sizes = layer_sizes
        self.split_half = split_half
        self.num_fields = len(field_dims)
        
        # CIN卷积层
        self.conv_layers = nn.ModuleList()
        prev_layer_size = self.num_fields
        
        for layer_size in layer_sizes:
            self.conv_layers.append(
                nn.Conv1d(prev_layer_size * self.num_fields, layer_size, 1)
            )
            
            if split_half and layer_size > 1:
                prev_layer_size = layer_size // 2
            else:
                prev_layer_size = layer_size
        
        # 输出层权重
        self.output_dim = sum(layer_sizes) if not split_half else \
                         sum([l // 2 for l in layer_sizes[:-1]]) + layer_sizes[-1]
    
    def forward(self, inputs):
        """
        前向传播
        
        Args:
            inputs: [batch_size, num_fields, embedding_dim]
        """
        batch_size, num_fields, embedding_dim = inputs.shape
        
        # 存储每层的输出用于最终的pooling
        pooling_outputs = []
        
        # 第一层输入
        x_0 = inputs  # [batch_size, num_fields, embedding_dim]
        x_i = x_0
        
        for i, conv_layer in enumerate(self.conv_layers):
            # 计算Hadamard乘积
            # x_0: [batch_size, num_fields, embedding_dim]
            # x_i: [batch_size, H_i, embedding_dim]
            
            # 扩展维度进行outer product
            x_0_expanded = x_0.unsqueeze(2)  # [batch_size, num_fields, 1, embedding_dim]
            x_i_expanded = x_i.unsqueeze(1)  # [batch_size, 1, H_i, embedding_dim]
            
            # Hadamard乘积
            hadamard_product = x_0_expanded * x_i_expanded  # [batch_size, num_fields, H_i, embedding_dim]
            
            # 重塑为卷积输入格式
            hadamard_reshaped = hadamard_product.view(
                batch_size, num_fields * x_i.shape[1], embedding_dim
            )  # [batch_size, num_fields * H_i, embedding_dim]
            
            # 1D卷积
            conv_out = conv_layer(hadamard_reshaped)  # [batch_size, H_{i+1}, embedding_dim]
            conv_out = torch.relu(conv_out)
            
            # Sum pooling在embedding维度上
            pooled = torch.sum(conv_out, dim=2)  # [batch_size, H_{i+1}]
            
            if self.split_half and i < len(self.conv_layers) - 1:
                # 分割输出，一半用于下一层，一半用于最终输出
                next_layer_input, pooling_output = torch.split(
                    conv_out, [conv_out.shape[1] // 2, conv_out.shape[1] - conv_out.shape[1] // 2], 
                    dim=1
                )
                x_i = next_layer_input
                pooling_outputs.append(torch.sum(pooling_output, dim=2))
            else:
                # 最后一层或不分割
                x_i = conv_out
                pooling_outputs.append(pooled)
        
        # 拼接所有pooling输出
        cin_output = torch.cat(pooling_outputs, dim=1)  # [batch_size, output_dim]
        
        return cin_output


class xDeepFMModel(nn.Module):
    """xDeepFM模型"""
    
    def __init__(self, field_dims: List[int], embedding_dim: int = 10,
                 cin_layer_sizes: List[int] = [128, 128],
                 dnn_hidden_dims: List[int] = [400, 400, 400],
                 dropout: float = 0.5):
        super().__init__()
        
        self.field_dims = field_dims
        self.embedding_dim = embedding_dim
        self.num_fields = len(field_dims)
        
        # 嵌入层
        self.embeddings = nn.ModuleList([
            nn.Embedding(field_dim, embedding_dim) 
            for field_dim in field_dims
        ])
        
        # Linear部分(一阶特征)
        self.linear = nn.Embedding(sum(field_dims), 1)
        self.bias = nn.Parameter(torch.zeros(1))
        
        # FM部分(二阶特征)
        # FM在嵌入层中隐式计算
        
        # CIN部分
        self.cin = CompressedInteractionNetwork(field_dims, cin_layer_sizes)
        
        # DNN部分
        dnn_input_dim = self.num_fields * embedding_dim
        dnn_layers = []
        prev_dim = dnn_input_dim
        
        for hidden_dim in dnn_hidden_dims:
            dnn_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.dnn = nn.Sequential(*dnn_layers)
        
        # 最终输出层
        final_dim = 1 + 1 + self.cin.output_dim + prev_dim  # linear + FM + CIN + DNN
        self.final_linear = nn.Linear(final_dim, 1)
        
        # 权重初始化
        self._init_weights()
        
        logger.info(f"xDeepFM模型初始化: 字段维度={field_dims}, "
                   f"嵌入维度={embedding_dim}, CIN层={cin_layer_sizes}")
    
    def _init_weights(self):
        """初始化权重"""
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, mean=0, std=0.1)
        
        nn.init.normal_(self.linear.weight, mean=0, std=0.1)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, categorical_inputs: torch.Tensor):
        """
        前向传播
        
        Args:
            categorical_inputs: [batch_size, num_fields] 类别特征输入
        """
        batch_size = categorical_inputs.shape[0]
        
        # 线性部分(一阶)
        linear_part = self.linear(categorical_inputs).sum(dim=1) + self.bias  # [batch_size, 1]
        
        # 嵌入
        embeddings = []
        for i, embedding_layer in enumerate(self.embeddings):
            emb = embedding_layer(categorical_inputs[:, i])  # [batch_size, embedding_dim]
            embeddings.append(emb)
        
        embeddings = torch.stack(embeddings, dim=1)  # [batch_size, num_fields, embedding_dim]
        
        # FM部分(二阶)
        # Sum of squares
        sum_of_embeddings = torch.sum(embeddings, dim=1)  # [batch_size, embedding_dim]
        sum_of_squares = torch.sum(sum_of_embeddings ** 2, dim=1, keepdim=True)  # [batch_size, 1]
        
        # Square of sums
        square_of_embeddings = embeddings ** 2  # [batch_size, num_fields, embedding_dim]
        square_of_sums = torch.sum(square_of_embeddings, dim=1)  # [batch_size, embedding_dim]
        square_of_sums = torch.sum(square_of_sums, dim=1, keepdim=True)  # [batch_size, 1]
        
        # FM交互项
        fm_part = 0.5 * (sum_of_squares - square_of_sums)  # [batch_size, 1]
        
        # CIN部分
        cin_part = self.cin(embeddings)  # [batch_size, cin_output_dim]
        
        # DNN部分
        dnn_input = embeddings.view(batch_size, -1)  # [batch_size, num_fields * embedding_dim]
        dnn_part = self.dnn(dnn_input)  # [batch_size, last_hidden_dim]
        
        # 拼接所有部分
        final_input = torch.cat([
            linear_part,  # [batch_size, 1]
            fm_part,      # [batch_size, 1]
            cin_part,     # [batch_size, cin_output_dim]
            dnn_part      # [batch_size, dnn_output_dim]
        ], dim=1)
        
        # 最终预测
        output = torch.sigmoid(self.final_linear(final_input))  # [batch_size, 1]
        
        return output.squeeze(-1)


class xDeepFMDataset(Dataset):
    """xDeepFM数据集"""
    
    def __init__(self, categorical_features: torch.Tensor, labels: torch.Tensor):
        self.categorical_features = categorical_features
        self.labels = labels
    
    def __len__(self):
        return len(self.categorical_features)
    
    def __getitem__(self, idx):
        return self.categorical_features[idx], self.labels[idx]


class xDeepFMRecommender(BaseRecommender):
    """xDeepFM推荐器"""
    
    def __init__(self, embedding_dim: int = 10, 
                 cin_layer_sizes: List[int] = [128, 128],
                 dnn_hidden_dims: List[int] = [400, 400, 400],
                 learning_rate: float = 0.001, epochs: int = 50,
                 batch_size: int = 256, dropout: float = 0.5,
                 early_stopping: bool = True, patience: int = 5,
                 **kwargs):
        super().__init__("xDeepFM", **kwargs)
        
        self.embedding_dim = embedding_dim
        self.cin_layer_sizes = cin_layer_sizes
        self.dnn_hidden_dims = dnn_hidden_dims
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.early_stopping = early_stopping
        self.patience = patience
        
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_encoders = {}
        self.field_dims = []
        self.all_items = set()
        self.user_item_matrix = {}
        
        logger.info(f"初始化xDeepFM: 嵌入维度={embedding_dim}, "
                   f"CIN层={cin_layer_sizes}, DNN层={dnn_hidden_dims}")
    
    def _prepare_features(self, train_data: pd.DataFrame):
        """准备特征数据"""
        # 编码类别特征
        categorical_features = ['user_id', 'item_id']
        
        for feature in categorical_features:
            le = LabelEncoder()
            train_data[f'{feature}_encoded'] = le.fit_transform(train_data[feature])
            self.label_encoders[feature] = le
            self.field_dims.append(len(le.classes_))
        
        # 构建用户-物品交互矩阵
        self.all_items = set(train_data['item_id'].unique())
        for _, row in train_data.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            rating = row['rating']
            
            if user_id not in self.user_item_matrix:
                self.user_item_matrix[user_id] = {}
            self.user_item_matrix[user_id][item_id] = rating
        
        logger.info(f"字段维度: {self.field_dims}")
        logger.info(f"用户数: {self.field_dims[0]}, 物品数: {self.field_dims[1]}")
    
    def fit(self, train_data: pd.DataFrame, **kwargs) -> None:
        """训练模型"""
        logger.info("开始训练xDeepFM模型...")
        
        # 准备特征
        self._prepare_features(train_data)
        
        # 创建模型
        self.model = xDeepFMModel(
            field_dims=self.field_dims,
            embedding_dim=self.embedding_dim,
            cin_layer_sizes=self.cin_layer_sizes,
            dnn_hidden_dims=self.dnn_hidden_dims,
            dropout=self.dropout
        ).to(self.device)
        
        # 准备训练数据
        categorical_features = torch.tensor(
            train_data[['user_id_encoded', 'item_id_encoded']].values,
            dtype=torch.long
        )
        
        # 将评分标准化到[0,1]
        labels = torch.tensor(
            (train_data['rating'].values - 1) / 4.0,  # 1-5 -> 0-1
            dtype=torch.float
        )
        
        dataset = xDeepFMDataset(categorical_features, labels)
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
            
            for cat_features, labels_batch in dataloader:
                cat_features = cat_features.to(self.device)
                labels_batch = labels_batch.to(self.device)
                
                optimizer.zero_grad()
                
                # 前向传播
                predictions = self.model(cat_features)
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
        logger.info("xDeepFM训练完成")
    
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
            cat_features = torch.tensor(
                [[user_encoded, item_encoded]], 
                dtype=torch.long
            ).to(self.device)
            
            # 预测
            prediction = self.model(cat_features).item()
            
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
                cat_features = torch.tensor([
                    [user_encoded, item_encoded] 
                    for item_encoded in item_encoded_batch
                ], dtype=torch.long).to(self.device)
                
                # 批量预测
                predictions = self.model(cat_features)
                
                # 收集推荐
                for item_id, pred in zip(item_batch, predictions):
                    score = 1.0 + pred.item() * 4.0  # 转换为1-5评分
                    recommendations.append({
                        'item_id': item_id,
                        'score': score
                    })
            
            # 排序并返回top-k
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            return recommendations[:top_k]


# 工厂函数
def create_xdeepfm_recommender(**kwargs) -> xDeepFMRecommender:
    """创建xDeepFM推荐器"""
    return xDeepFMRecommender(**kwargs)


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
    recommender = xDeepFMRecommender(
        embedding_dim=8,
        cin_layer_sizes=[64, 64],
        dnn_hidden_dims=[128, 64],
        epochs=5
    )
    
    # 训练
    recommender.fit(train_data)
    
    # 推荐
    recs = recommender.get_user_recommendations(1, top_k=3)
    print("推荐结果:", recs)
