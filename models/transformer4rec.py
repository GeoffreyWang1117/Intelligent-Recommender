#!/usr/bin/env python3
"""
Transformer4Rec推荐算法实现

基于Transformer架构的序列推荐模型，能够捕捉用户行为的时序模式。
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Any, Optional
import logging
from collections import defaultdict
import math

from .base_recommender import BaseRecommender

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class Transformer4RecModel(nn.Module):
    """Transformer4Rec模型"""
    
    def __init__(self, num_items: int, embedding_dim: int = 64, 
                 num_heads: int = 8, num_layers: int = 6, 
                 max_seq_len: int = 100, dropout: float = 0.1):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        
        # 物品嵌入
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embedding_dim, max_seq_len)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 输出层
        self.output_layer = nn.Linear(embedding_dim, num_items)
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)
    
    def forward(self, item_seq, mask=None):
        """前向传播
        
        Args:
            item_seq: 物品序列 [batch_size, seq_len]
            mask: 注意力掩码 [batch_size, seq_len]
        """
        # 物品嵌入
        embeddings = self.item_embedding(item_seq)  # [batch_size, seq_len, embedding_dim]
        
        # 位置编码
        embeddings = self.pos_encoding(embeddings.transpose(0, 1)).transpose(0, 1)
        embeddings = self.dropout(embeddings)
        
        # 创建注意力掩码
        if mask is None:
            # 创建因果掩码（下三角矩阵）
            seq_len = item_seq.size(1)
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            mask = mask.to(item_seq.device)
        
        # Transformer编码
        transformer_output = self.transformer(embeddings, mask=mask)
        
        # 只使用最后一个时间步的输出进行预测
        last_hidden = transformer_output[:, -1, :]  # [batch_size, embedding_dim]
        
        # 输出层
        logits = self.output_layer(last_hidden)  # [batch_size, num_items]
        
        return logits


class Transformer4RecDataset(Dataset):
    """Transformer4Rec数据集"""
    
    def __init__(self, sequences: List[List[int]], max_seq_len: int = 100):
        self.sequences = sequences
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # 序列截断或填充
        if len(seq) > self.max_seq_len:
            seq = seq[-self.max_seq_len:]
        
        # 创建输入和目标
        if len(seq) < 2:
            # 序列太短，使用零填充
            input_seq = [0] * (self.max_seq_len - 1)
            target = 0
        else:
            input_seq = seq[:-1]
            target = seq[-1]
            
            # 填充到固定长度
            if len(input_seq) < self.max_seq_len - 1:
                input_seq = [0] * (self.max_seq_len - 1 - len(input_seq)) + input_seq
        
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)


class Transformer4RecRecommender(BaseRecommender):
    """Transformer4Rec推荐器"""
    
    def __init__(self, embedding_dim: int = 64, num_heads: int = 8, 
                 num_layers: int = 6, max_seq_len: int = 100, 
                 learning_rate: float = 0.001, epochs: int = 50,
                 batch_size: int = 256, dropout: float = 0.1,
                 early_stopping: bool = True, patience: int = 5,
                 **kwargs):
        super().__init__("Transformer4Rec", **kwargs)
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.early_stopping = early_stopping
        self.patience = patience
        
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.user_sequences = {}
        self.item_to_idx = {}
        self.idx_to_item = {}
        self.num_items = 0
        
        logger.info(f"初始化Transformer4Rec: 嵌入维度={embedding_dim}, "
                   f"注意力头数={num_heads}, 层数={num_layers}")
    
    def _prepare_sequences(self, train_data: pd.DataFrame):
        """准备用户序列数据"""
        # 创建物品映射
        unique_items = sorted(train_data['item_id'].unique())
        self.item_to_idx = {item: idx + 1 for idx, item in enumerate(unique_items)}  # 从1开始，0为padding
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        self.num_items = len(unique_items) + 1  # +1 for padding token
        
        # 按用户和时间戳排序，构建序列
        train_data_sorted = train_data.sort_values(['user_id', 'timestamp'])
        
        self.user_sequences = {}
        for user_id, group in train_data_sorted.groupby('user_id'):
            sequence = [self.item_to_idx[item_id] for item_id in group['item_id'].tolist()]
            self.user_sequences[user_id] = sequence
        
        logger.info(f"构建了 {len(self.user_sequences)} 个用户序列，"
                   f"物品数量: {self.num_items}")
    
    def fit(self, train_data: pd.DataFrame, **kwargs) -> None:
        """训练模型"""
        logger.info("开始训练Transformer4Rec模型...")
        
        # 准备序列数据
        self._prepare_sequences(train_data)
        
        # 创建模型
        self.model = Transformer4RecModel(
            num_items=self.num_items,
            embedding_dim=self.embedding_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            max_seq_len=self.max_seq_len,
            dropout=self.dropout
        ).to(self.device)
        
        # 准备训练数据
        sequences = list(self.user_sequences.values())
        dataset = Transformer4RecDataset(sequences, self.max_seq_len)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 优化器和损失函数
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding
        
        # 训练循环
        self.model.train()
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            total_loss = 0.0
            num_batches = 0
            
            for batch_input, batch_target in dataloader:
                batch_input = batch_input.to(self.device)
                batch_target = batch_target.to(self.device)
                
                optimizer.zero_grad()
                
                # 前向传播
                logits = self.model(batch_input)
                loss = criterion(logits, batch_target)
                
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
        logger.info("Transformer4Rec训练完成")
    
    def predict(self, user_id: int, item_id: int) -> float:
        """预测用户对物品的评分"""
        if not self.is_trained or self.model is None:
            return 3.0
        
        if user_id not in self.user_sequences:
            return 3.0
        
        if item_id not in self.item_to_idx:
            return 3.0
        
        self.model.eval()
        with torch.no_grad():
            # 获取用户序列
            user_seq = self.user_sequences[user_id]
            
            # 准备输入
            if len(user_seq) >= self.max_seq_len:
                input_seq = user_seq[-(self.max_seq_len-1):]
            else:
                input_seq = [0] * (self.max_seq_len - 1 - len(user_seq)) + user_seq
            
            input_tensor = torch.tensor([input_seq], dtype=torch.long).to(self.device)
            
            # 预测
            logits = self.model(input_tensor)
            item_idx = self.item_to_idx[item_id]
            score = torch.softmax(logits, dim=-1)[0, item_idx].item()
            
            # 转换为1-5评分
            return 1.0 + score * 4.0
    
    def get_user_recommendations(self, user_id: int, top_k: int = 10) -> List[Dict[str, Any]]:
        """获取用户推荐"""
        if not self.is_trained or self.model is None:
            return []
        
        if user_id not in self.user_sequences:
            return []
        
        self.model.eval()
        with torch.no_grad():
            # 获取用户序列
            user_seq = self.user_sequences[user_id]
            
            # 准备输入
            if len(user_seq) >= self.max_seq_len:
                input_seq = user_seq[-(self.max_seq_len-1):]
            else:
                input_seq = [0] * (self.max_seq_len - 1 - len(user_seq)) + user_seq
            
            input_tensor = torch.tensor([input_seq], dtype=torch.long).to(self.device)
            
            # 预测
            logits = self.model(input_tensor)
            scores = torch.softmax(logits, dim=-1)[0]
            
            # 排除已交互的物品
            user_items = set(user_seq)
            
            # 获取推荐
            recommendations = []
            _, top_indices = torch.topk(scores, k=min(top_k * 2, self.num_items))
            
            for idx in top_indices:
                item_idx = idx.item()
                if item_idx in self.idx_to_item and item_idx not in user_items:
                    item_id = self.idx_to_item[item_idx]
                    score = scores[idx].item()
                    recommendations.append({
                        'item_id': item_id,
                        'score': score
                    })
                    
                    if len(recommendations) >= top_k:
                        break
            
            return recommendations


# 工厂函数
def create_transformer4rec_recommender(**kwargs) -> Transformer4RecRecommender:
    """创建Transformer4Rec推荐器"""
    return Transformer4RecRecommender(**kwargs)


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
    recommender = Transformer4RecRecommender(
        embedding_dim=32,
        num_heads=4,
        num_layers=2,
        epochs=5
    )
    
    # 训练
    recommender.fit(train_data)
    
    # 推荐
    recs = recommender.get_user_recommendations(1, top_k=3)
    print("推荐结果:", recs)
