#!/usr/bin/env python3
"""
改进版LayerwiseAdapter训练脚本

修复学习问题，添加推荐质量评估
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Tuple, Any
import logging
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 添加路径
sys.path.append('/home/coder-gw/7Projects_in_7Days/online-inference-system')
sys.path.append('/home/coder-gw/7Projects_in_7Days/online-inference-system/layerwise_adapter_enhanced')

def load_movielens_1m():
    """加载MovieLens 1M数据"""
    data_path = "/home/coder-gw/7Projects_in_7Days/online-inference-system/data/movielens/1m"
    
    # 读取评分数据
    ratings_file = os.path.join(data_path, "ratings.csv")
    if not os.path.exists(ratings_file):
        # 尝试其他可能的文件名
        for fname in ["ratings.dat", "ml-1m.csv"]:
            test_path = os.path.join(data_path, fname)
            if os.path.exists(test_path):
                if fname.endswith('.dat'):
                    ratings = pd.read_csv(test_path, sep='::', names=['userId', 'movieId', 'rating', 'timestamp'], engine='python')
                else:
                    ratings = pd.read_csv(test_path)
                break
    else:
        ratings = pd.read_csv(ratings_file)
    
    print(f"📊 加载MovieLens 1M数据集...")
    print(f"   - 总评分数: {len(ratings):,}")
    print(f"   - 用户数: {ratings['userId'].nunique():,}")
    print(f"   - 电影数: {ratings['movieId'].nunique():,}")
    print(f"   - 评分范围: {ratings['rating'].min()} - {ratings['rating'].max()}")
    print(f"   - 平均评分: {ratings['rating'].mean():.2f}")
    
    return ratings

def prepare_data(ratings_df):
    """准备训练数据"""
    # 用户和物品编码
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    
    ratings_df['userId_encoded'] = user_encoder.fit_transform(ratings_df['userId'])
    ratings_df['movieId_encoded'] = item_encoder.fit_transform(ratings_df['movieId'])
    
    num_users = len(user_encoder.classes_)
    num_items = len(item_encoder.classes_)
    
    print(f"✅ 数据加载完成")
    print(f"   - 映射后用户数: {num_users}")
    print(f"   - 映射后物品数: {num_items}")
    
    return ratings_df, user_encoder, item_encoder, num_users, num_items

def create_dataloaders(ratings_df, batch_size=512):
    """创建数据加载器"""
    # 分割数据
    train_data, temp_data = train_test_split(ratings_df, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    print(f"🔄 准备训练/验证/测试数据...")
    print(f"   - 训练集: {len(train_data):,} 样本")
    print(f"   - 验证集: {len(val_data):,} 样本") 
    print(f"   - 测试集: {len(test_data):,} 样本")
    
    def create_dataloader(data, batch_size, shuffle=True):
        user_ids = torch.LongTensor(data['userId_encoded'].values)
        item_ids = torch.LongTensor(data['movieId_encoded'].values)
        ratings = torch.FloatTensor(data['rating'].values)
        
        dataset = TensorDataset(user_ids, item_ids, ratings)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    train_loader = create_dataloader(train_data, batch_size, shuffle=True)
    val_loader = create_dataloader(val_data, batch_size, shuffle=False)
    test_loader = create_dataloader(test_data, batch_size, shuffle=False)
    
    print(f"✅ 数据加载器准备完成")
    
    return train_loader, val_loader, test_loader, train_data, val_data, test_data

class ImprovedLayerwiseAdapter(nn.Module):
    """改进的LayerwiseAdapter - 简化版本专注于学习"""
    
    def __init__(self, num_users, num_items, embedding_dim=64, hidden_dim=128):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        
        # 用户和物品嵌入
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # 三层架构：嵌入层 -> 交互层 -> 推理层
        self.embedding_layer = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.interaction_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.reasoning_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, user_ids, item_ids):
        # 获取嵌入
        user_emb = self.user_embedding(user_ids)  # [batch_size, embedding_dim]
        item_emb = self.item_embedding(item_ids)  # [batch_size, embedding_dim]
        
        # 拼接嵌入
        combined = torch.cat([user_emb, item_emb], dim=1)  # [batch_size, embedding_dim * 2]
        
        # 三层前向传播
        emb_out = self.embedding_layer(combined)
        inter_out = self.interaction_layer(emb_out)
        rating = self.reasoning_layer(inter_out)
        
        return rating.squeeze()

def evaluate_model(model, data_loader, device):
    """评估模型"""
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for user_ids, item_ids, ratings in data_loader:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            ratings = ratings.to(device)
            
            pred = model(user_ids, item_ids)
            
            predictions.extend(pred.cpu().numpy())
            targets.extend(ratings.cpu().numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    
    # 准确率 (±0.5范围内)
    accuracy = np.mean(np.abs(predictions - targets) <= 0.5)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'accuracy': accuracy,
        'predictions': predictions,
        'targets': targets
    }

def compute_recommendation_metrics(model, test_data, user_encoder, item_encoder, device, top_k=10):
    """计算推荐质量指标"""
    print(f"🎯 计算推荐质量指标 (Top-{top_k})...")
    
    model.eval()
    
    # 随机选择一些用户进行评估
    unique_users = test_data['userId_encoded'].unique()
    eval_users = np.random.choice(unique_users, min(100, len(unique_users)), replace=False)
    
    precisions = []
    ndcgs = []
    
    with torch.no_grad():
        for user_idx in tqdm(eval_users, desc="评估用户"):
            # 获取用户的真实高评分物品 (>=4)
            user_items = test_data[test_data['userId_encoded'] == user_idx]
            true_items = set(user_items[user_items['rating'] >= 4]['movieId_encoded'].values)
            
            if len(true_items) == 0:
                continue
            
            # 获取所有物品的预测评分
            all_items = torch.arange(model.num_items).to(device)
            user_tensor = torch.full((len(all_items),), user_idx, dtype=torch.long).to(device)
            
            predictions = model(user_tensor, all_items)
            
            # 获取Top-K推荐
            _, top_items = torch.topk(predictions, top_k)
            recommended_items = set(top_items.cpu().numpy())
            
            # 计算Precision@K
            hits = len(recommended_items & true_items)
            precision = hits / top_k
            precisions.append(precision)
            
            # 计算NDCG@K
            dcg = 0
            idcg = 0
            for i, item in enumerate(top_items.cpu().numpy()):
                if item in true_items:
                    dcg += 1 / np.log2(i + 2)
                idcg += 1 / np.log2(i + 2)
            
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcgs.append(ndcg)
    
    return {
        'precision_at_k': np.mean(precisions),
        'ndcg_at_k': np.mean(ndcgs),
        'evaluated_users': len(eval_users)
    }

def train_improved_model():
    """训练改进版模型"""
    print("=" * 80)
    print("🧪 改进版LayerwiseAdapter训练与评估")
    print("数据集: MovieLens 1M")
    print("=" * 80)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")
    
    # 加载数据
    ratings_df = load_movielens_1m()
    ratings_df, user_encoder, item_encoder, num_users, num_items = prepare_data(ratings_df)
    
    # 创建数据加载器
    train_loader, val_loader, test_loader, train_data, val_data, test_data = create_dataloaders(ratings_df)
    
    # 创建模型
    print(f"🔧 创建改进版LayerwiseAdapter...")
    model = ImprovedLayerwiseAdapter(num_users, num_items, embedding_dim=64, hidden_dim=128)
    model = model.to(device)
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ 模型设置完成")
    print(f"   - 用户数: {num_users}, 物品数: {num_items}")
    print(f"   - 总参数数量: {total_params:,}")
    print(f"   - 可训练参数: {trainable_params:,}")
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)  # 提高学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = nn.MSELoss()
    
    # 训练配置
    num_epochs = 10
    best_val_rmse = float('inf')
    training_history = {
        'train_losses': [],
        'val_rmses': [],
        'val_maes': [],
        'learning_rates': []
    }
    
    print(f"\n🚀 开始训练 - {num_epochs} epochs")
    print(f"学习率: {optimizer.param_groups[0]['lr']}")
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # 训练阶段
        model.train()
        train_loss = 0
        num_batches = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (user_ids, item_ids, ratings) in enumerate(train_pbar):
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            ratings = ratings.to(device)
            
            optimizer.zero_grad()
            predictions = model(user_ids, item_ids)
            loss = criterion(predictions, ratings)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / num_batches
        
        # 验证阶段
        val_metrics = evaluate_model(model, val_loader, device)
        val_rmse = val_metrics['rmse']
        val_mae = val_metrics['mae']
        
        # 记录历史
        training_history['train_losses'].append(avg_train_loss)
        training_history['val_rmses'].append(val_rmse)
        training_history['val_maes'].append(val_mae)
        training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # 学习率调度
        scheduler.step(val_rmse)
        
        # 保存最佳模型
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), 
                      '/home/coder-gw/7Projects_in_7Days/online-inference-system/layerwise_adapter_enhanced/models/improved_layerwise_adapter.pth')
            print(f"   🎯 新的最佳验证RMSE: {val_rmse:.4f}")
        
        epoch_time = time.time() - epoch_start
        print(f"\n📈 Epoch {epoch+1}/{num_epochs} 完成")
        print(f"   训练损失: {avg_train_loss:.4f}")
        print(f"   验证RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}")
        print(f"   学习率: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"   用时: {epoch_time:.2f}s")
        print("-" * 60)
    
    # 测试集评估
    print("\n🧪 测试集评估...")
    test_metrics = evaluate_model(model, test_loader, device)
    
    # 推荐质量评估
    rec_metrics = compute_recommendation_metrics(model, test_data, user_encoder, item_encoder, device)
    
    # 汇总结果
    final_results = {
        'model_name': 'ImprovedLayerwiseAdapter',
        'dataset': 'MovieLens-1M',
        'model_config': {
            'num_users': num_users,
            'num_items': num_items,
            'embedding_dim': 64,
            'hidden_dim': 128,
            'total_parameters': total_params
        },
        'training_config': {
            'epochs': num_epochs,
            'batch_size': 512,
            'learning_rate': 0.01,
            'optimizer': 'Adam',
            'weight_decay': 1e-5
        },
        'training_history': training_history,
        'final_metrics': {
            'test_rmse': test_metrics['rmse'],
            'test_mae': test_metrics['mae'],
            'test_accuracy': test_metrics['accuracy'],
            'best_val_rmse': best_val_rmse,
            'precision_at_10': rec_metrics['precision_at_k'],
            'ndcg_at_10': rec_metrics['ndcg_at_k']
        },
        'recommendation_quality': rec_metrics,
        'timestamp': time.time()
    }
    
    # 保存结果
    results_file = '/home/coder-gw/7Projects_in_7Days/online-inference-system/layerwise_adapter_enhanced/experiments/improved_layerwise_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    # 打印最终结果
    print("\n" + "=" * 80)
    print("🎯 最终评估结果")
    print("=" * 80)
    print(f"数据集: MovieLens 1M")
    print(f"模型: ImprovedLayerwiseAdapter")
    print(f"参数量: {total_params:,}")
    print(f"\n📈 评分预测性能:")
    print(f"- 测试RMSE: {test_metrics['rmse']:.4f}")
    print(f"- 测试MAE: {test_metrics['mae']:.4f}")
    print(f"- 测试准确率: {test_metrics['accuracy']*100:.2f}%")
    print(f"- 最佳验证RMSE: {best_val_rmse:.4f}")
    print(f"\n🎯 推荐质量:")
    print(f"- Precision@10: {rec_metrics['precision_at_k']:.4f}")
    print(f"- NDCG@10: {rec_metrics['ndcg_at_k']:.4f}")
    print(f"- 评估用户数: {rec_metrics['evaluated_users']}")
    
    print(f"\n💾 结果已保存到: {results_file}")
    print("✅ 训练和评估完成!")
    
    return final_results

if __name__ == "__main__":
    results = train_improved_model()
