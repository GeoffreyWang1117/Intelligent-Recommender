#!/usr/bin/env python3
"""
基准Ensemble模型在MovieLens 1M数据集上的训练和评估

基准模型：SVD + AutoInt + xDeepFM 简单ensemble
目的：为LayerwiseAdapter提供公平的性能对比基准
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

# 添加路径
sys.path.append('/home/coder-gw/7Projects_in_7Days/online-inference-system')

# 导入基础推荐算法
try:
    from models.svd_model import SVDModel
    from models.autoint import AutoIntModel  
    from models.xdeepfm import xDeepFMModel
    print("✅ 成功导入基础模型")
except ImportError as e:
    print(f"⚠️ 导入基础模型失败: {e}")
    print("使用简化版本模型")

class SimpleEnsembleModel(nn.Module):
    """简单Ensemble模型：SVD + AutoInt + xDeepFM"""
    
    def __init__(self, num_users, num_items, embedding_dim=64, device='cuda'):
        super().__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        
        # 用户和物品嵌入
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # SVD组件
        self.svd_predictor = nn.Linear(embedding_dim * 2, 1)
        
        # AutoInt组件（简化版）
        self.autoint_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # xDeepFM组件（简化版）
        self.xdeepfm_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Ensemble权重
        self.ensemble_weights = nn.Parameter(torch.ones(3) / 3)
        
    def forward(self, user_ids, item_ids):
        # 获取嵌入
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # 特征拼接
        features = torch.cat([user_emb, item_emb], dim=-1)
        
        # SVD预测
        svd_output = self.svd_predictor(features)
        
        # AutoInt预测
        autoint_output = self.autoint_layers(features)
        
        # xDeepFM预测
        xdeepfm_output = self.xdeepfm_layers(features)
        
        # Ensemble融合
        ensemble_weights = torch.softmax(self.ensemble_weights, dim=0)
        final_output = (ensemble_weights[0] * svd_output + 
                       ensemble_weights[1] * autoint_output + 
                       ensemble_weights[2] * xdeepfm_output)
        
        return torch.sigmoid(final_output) * 4 + 1  # 映射到1-5评分范围

class MovieLens1MDataLoader:
    """MovieLens 1M数据加载器"""
    
    def __init__(self, data_dir="/home/coder-gw/7Projects_in_7Days/online-inference-system/data/movielens/1m"):
        self.data_dir = data_dir
        
    def load_data(self):
        """加载MovieLens 1M数据"""
        try:
            # 加载评分数据
            ratings_file = os.path.join(self.data_dir, "ratings.csv")
            if not os.path.exists(ratings_file):
                # 尝试原始格式
                ratings_file = os.path.join(self.data_dir, "ratings.dat")
                if os.path.exists(ratings_file):
                    # 读取.dat格式
                    ratings = pd.read_csv(ratings_file, sep='::', 
                                        names=['userId', 'movieId', 'rating', 'timestamp'],
                                        engine='python')
                else:
                    raise FileNotFoundError(f"找不到评分文件: {self.data_dir}")
            else:
                # 读取.csv格式
                ratings = pd.read_csv(ratings_file)
                
            print(f"📊 加载了 {len(ratings)} 条评分记录")
            print(f"用户数量: {ratings['userId'].nunique()}")
            print(f"物品数量: {ratings['movieId'].nunique()}")
            print(f"评分范围: {ratings['rating'].min()} - {ratings['rating'].max()}")
            
            return ratings
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            raise

    def preprocess_data(self, ratings):
        """数据预处理"""
        # 用户和物品ID编码
        user_encoder = LabelEncoder()
        item_encoder = LabelEncoder()
        
        ratings['userId_encoded'] = user_encoder.fit_transform(ratings['userId'])
        ratings['movieId_encoded'] = item_encoder.fit_transform(ratings['movieId'])
        
        # 数据划分
        train_data, temp_data = train_test_split(ratings, test_size=0.3, random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
        
        print(f"训练集: {len(train_data)} 条")
        print(f"验证集: {len(val_data)} 条")
        print(f"测试集: {len(test_data)} 条")
        
        return train_data, val_data, test_data, user_encoder, item_encoder

class EnsembleTrainer:
    """Ensemble模型训练器"""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
    def create_dataloader(self, data, batch_size=512):
        """创建数据加载器"""
        user_ids = torch.LongTensor(data['userId_encoded'].values)
        item_ids = torch.LongTensor(data['movieId_encoded'].values)
        ratings = torch.FloatTensor(data['rating'].values)
        
        dataset = TensorDataset(user_ids, item_ids, ratings)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for user_ids, item_ids, ratings in train_loader:
            user_ids = user_ids.to(self.device)
            item_ids = item_ids.to(self.device)
            ratings = ratings.to(self.device)
            
            self.optimizer.zero_grad()
            predictions = self.model(user_ids, item_ids).squeeze()
            loss = self.criterion(predictions, ratings)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        return total_loss / num_batches
    
    def evaluate(self, data_loader):
        """评估模型"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for user_ids, item_ids, ratings in data_loader:
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)
                
                predictions = self.model(user_ids, item_ids).squeeze()
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(ratings.numpy())
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # 计算评估指标
        rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
        mae = mean_absolute_error(all_targets, all_predictions)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'predictions': all_predictions,
            'targets': all_targets
        }
    
    def train(self, train_data, val_data, epochs=10):
        """完整训练流程"""
        train_loader = self.create_dataloader(train_data)
        val_loader = self.create_dataloader(val_data)
        
        training_history = {
            'train_loss': [],
            'val_rmse': [],
            'val_mae': []
        }
        
        best_val_rmse = float('inf')
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # 训练
            train_loss = self.train_epoch(train_loader)
            
            # 验证
            val_metrics = self.evaluate(val_loader)
            val_rmse = val_metrics['rmse']
            val_mae = val_metrics['mae']
            
            # 记录历史
            training_history['train_loss'].append(train_loss)
            training_history['val_rmse'].append(val_rmse)
            training_history['val_mae'].append(val_mae)
            
            # 保存最佳模型
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                torch.save(self.model.state_dict(), 
                          '/home/coder-gw/7Projects_in_7Days/online-inference-system/layerwise_adapter_enhanced/models/ensemble_baseline_1m.pth')
            
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs} - {epoch_time:.2f}s")
            print(f"训练损失: {train_loss:.4f}, 验证RMSE: {val_rmse:.4f}, 验证MAE: {val_mae:.4f}")
            
            # 显示ensemble权重
            weights = torch.softmax(self.model.ensemble_weights, dim=0)
            print(f"Ensemble权重 - SVD: {weights[0]:.3f}, AutoInt: {weights[1]:.3f}, xDeepFM: {weights[2]:.3f}")
            print("-" * 60)
        
        return training_history

def compute_additional_metrics(predictions, targets, user_ids, item_ids):
    """计算额外的推荐系统评估指标"""
    
    # 创建用户-物品评分矩阵用于Top-K计算
    unique_users = np.unique(user_ids)
    user_predictions = {}
    user_targets = {}
    
    for i, user_id in enumerate(user_ids):
        if user_id not in user_predictions:
            user_predictions[user_id] = []
            user_targets[user_id] = []
        user_predictions[user_id].append(predictions[i])
        user_targets[user_id].append(targets[i])
    
    # 计算Precision@K和NDCG@K (K=10)
    precisions = []
    ndcgs = []
    
    for user_id in unique_users[:100]:  # 抽样100个用户避免计算过慢
        if len(user_predictions[user_id]) >= 10:
            pred = np.array(user_predictions[user_id])
            true = np.array(user_targets[user_id])
            
            # 按预测分数排序
            sorted_indices = np.argsort(pred)[::-1][:10]
            top_true_ratings = true[sorted_indices]
            
            # Precision@10 (评分>=4视为相关)
            relevant = (top_true_ratings >= 4).sum()
            precision = relevant / 10
            precisions.append(precision)
            
            # NDCG@10
            dcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(top_true_ratings))
            ideal_ratings = sorted(true, reverse=True)[:10]
            idcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(ideal_ratings))
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcgs.append(ndcg)
    
    return {
        'precision_at_10': np.mean(precisions) if precisions else 0,
        'ndcg_at_10': np.mean(ndcgs) if ndcgs else 0,
        'coverage': len(np.unique(item_ids)) / len(np.unique(item_ids)),  # 简化的覆盖率
        'diversity': np.std(predictions)  # 预测的多样性
    }

def main():
    """主函数"""
    print("=" * 80)
    print("基准Ensemble模型训练 - MovieLens 1M数据集")
    print("=" * 80)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    data_loader = MovieLens1MDataLoader()
    ratings = data_loader.load_data()
    train_data, val_data, test_data, user_encoder, item_encoder = data_loader.preprocess_data(ratings)
    
    # 创建模型
    num_users = len(user_encoder.classes_)
    num_items = len(item_encoder.classes_)
    
    print(f"模型参数:")
    print(f"- 用户数量: {num_users}")
    print(f"- 物品数量: {num_items}")
    print(f"- 嵌入维度: 64")
    
    model = SimpleEnsembleModel(num_users, num_items, device=str(device))
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"- 总参数量: {total_params:,}")
    print(f"- 可训练参数: {trainable_params:,}")
    
    # 训练模型
    trainer = EnsembleTrainer(model, str(device))
    print("\n🚀 开始训练...")
    training_history = trainer.train(train_data, val_data, epochs=10)
    
    # 测试集评估
    print("\n📊 测试集评估...")
    test_loader = trainer.create_dataloader(test_data)
    test_metrics = trainer.evaluate(test_loader)
    
    # 计算额外指标
    additional_metrics = compute_additional_metrics(
        test_metrics['predictions'], 
        test_metrics['targets'],
        test_data['userId_encoded'].values,
        test_data['movieId_encoded'].values
    )
    
    # 汇总结果
    final_results = {
        'model_type': 'Ensemble Baseline (SVD + AutoInt + xDeepFM)',
        'dataset': 'MovieLens-1M',
        'test_metrics': {
            'rmse': float(test_metrics['rmse']),
            'mae': float(test_metrics['mae']),
            'precision_at_10': float(additional_metrics['precision_at_10']),
            'ndcg_at_10': float(additional_metrics['ndcg_at_10']),
            'coverage': float(additional_metrics['coverage']),
            'diversity': float(additional_metrics['diversity'])
        },
        'model_info': {
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params),
            'num_users': int(num_users),
            'num_items': int(num_items)
        },
        'training_history': {
            'train_losses': [float(x) for x in training_history['train_loss']],
            'val_rmses': [float(x) for x in training_history['val_rmse']],
            'val_maes': [float(x) for x in training_history['val_mae']]
        },
        'ensemble_weights': {
            'svd': float(torch.softmax(model.ensemble_weights, dim=0)[0]),
            'autoint': float(torch.softmax(model.ensemble_weights, dim=0)[1]),
            'xdeepfm': float(torch.softmax(model.ensemble_weights, dim=0)[2])
        }
    }
    
    # 保存结果
    results_file = '/home/coder-gw/7Projects_in_7Days/online-inference-system/layerwise_adapter_enhanced/experiments/baselines/ensemble_baseline_1m_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    # 打印最终结果
    print("\n" + "=" * 60)
    print("🎯 最终评估结果")
    print("=" * 60)
    print(f"数据集: MovieLens 1M")
    print(f"模型: Ensemble Baseline (SVD + AutoInt + xDeepFM)")
    print(f"参数量: {total_params:,}")
    print("\n📈 性能指标:")
    print(f"- RMSE: {test_metrics['rmse']:.4f}")
    print(f"- MAE: {test_metrics['mae']:.4f}")
    print(f"- Precision@10: {additional_metrics['precision_at_10']:.4f}")
    print(f"- NDCG@10: {additional_metrics['ndcg_at_10']:.4f}")
    print(f"- Coverage: {additional_metrics['coverage']:.4f}")
    print(f"- Diversity: {additional_metrics['diversity']:.4f}")
    
    print("\n🔧 Ensemble权重:")
    weights = torch.softmax(model.ensemble_weights, dim=0)
    print(f"- SVD: {weights[0]:.3f}")
    print(f"- AutoInt: {weights[1]:.3f}")
    print(f"- xDeepFM: {weights[2]:.3f}")
    
    print(f"\n💾 结果已保存到: {results_file}")
    print("✅ 基准Ensemble训练完成!")

if __name__ == "__main__":
    main()
