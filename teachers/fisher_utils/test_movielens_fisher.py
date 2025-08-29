#!/usr/bin/env python3
"""
基于真实MovieLens数据的Fisher Information测试
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
import sys

# 添加项目根目录到路径
sys.path.append('/home/coder-gw/7Projects_in_7Days/online-inference-system')

from fisher_calculator import FisherInformationCalculator

class RecommenderModel(nn.Module):
    """更现实的推荐模型"""
    
    def __init__(self, num_users=610, num_items=9724, embedding_dim=64):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        
        # 用户和物品嵌入
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # 多层感知器
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 权重初始化
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.1)
    
    def forward(self, batch_data):
        if isinstance(batch_data, (tuple, list)):
            user_ids, item_ids = batch_data[0], batch_data[1]
        else:
            user_ids = batch_data[:, 0].long()
            item_ids = batch_data[:, 1].long()
            
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # 特征融合
        features = torch.cat([user_emb, item_emb], dim=-1)
        output = self.mlp(features)
        
        return torch.sigmoid(output.squeeze())

def load_movielens_data():
    """加载MovieLens数据"""
    data_path = '/home/coder-gw/7Projects_in_7Days/online-inference-system/data/movielens'
    
    # 读取评分数据
    ratings_file = os.path.join(data_path, 'ratings.csv')
    if not os.path.exists(ratings_file):
        print(f"❌ 未找到数据文件: {ratings_file}")
        return None
        
    ratings = pd.read_csv(ratings_file)
    print(f"✓ 加载 {len(ratings)} 条评分数据")
    
    # 数据预处理
    # 重新编码用户和物品ID
    user_ids = ratings['user_id'].unique()
    item_ids = ratings['item_id'].unique()
    
    user_map = {uid: idx for idx, uid in enumerate(user_ids)}
    item_map = {iid: idx for idx, iid in enumerate(item_ids)}
    
    ratings['user_idx'] = ratings['user_id'].map(user_map)
    ratings['item_idx'] = ratings['item_id'].map(item_map)
    
    # 标准化评分到0-1
    ratings['rating_norm'] = (ratings['rating'] - 1) / 4.0
    
    print(f"  - 用户数: {len(user_ids)}")
    print(f"  - 物品数: {len(item_ids)}")
    print(f"  - 评分范围: {ratings['rating'].min()}-{ratings['rating'].max()}")
    
    return ratings, len(user_ids), len(item_ids)

def create_movielens_dataloader(ratings_df, batch_size=64, sample_size=1000):
    """创建MovieLens数据加载器"""
    # 采样数据
    if len(ratings_df) > sample_size:
        ratings_sample = ratings_df.sample(n=sample_size, random_state=42)
    else:
        ratings_sample = ratings_df
    
    # 创建张量
    user_ids = torch.tensor(ratings_sample['user_idx'].values, dtype=torch.long)
    item_ids = torch.tensor(ratings_sample['item_idx'].values, dtype=torch.long)
    ratings = torch.tensor(ratings_sample['rating_norm'].values, dtype=torch.float32)
    
    # 组合输入
    inputs = torch.stack([user_ids, item_ids], dim=1).float()
    
    dataset = TensorDataset(inputs, ratings)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"✓ 创建数据加载器: {len(dataset)} 样本, {len(dataloader)} 批次")
    return dataloader

def test_fisher_on_movielens():
    """在真实MovieLens数据上测试Fisher Information"""
    print("=== MovieLens数据上的Fisher Information测试 ===")
    
    # 加载数据
    print("1. 加载MovieLens数据...")
    result = load_movielens_data()
    if result is None:
        return False
        
    ratings_df, num_users, num_items = result
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    print("\n2. 创建推荐模型...")
    model = RecommenderModel(num_users, num_items, embedding_dim=64).to(device)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建数据加载器
    print("\n3. 准备数据...")
    dataloader = create_movielens_dataloader(ratings_df, batch_size=64, sample_size=2000)
    
    # 初始化Fisher计算器
    print("\n4. 初始化Fisher Information计算器...")
    fisher_calc = FisherInformationCalculator(model, device=str(device))
    
    # 定义损失函数
    criterion = nn.MSELoss()
    
    # 计算Fisher Information
    print("\n5. 计算Fisher Information...")
    try:
        fisher_info = fisher_calc.compute_fisher_information(
            dataloader, criterion, num_batches=10
        )
        
        print("✓ Fisher Information计算成功！")
        print(f"  - 覆盖层数: {len(fisher_info)}")
        
        # 分析Fisher信息分布
        all_values = []
        layer_info = {}
        for name, tensor in fisher_info.items():
            values = tensor.flatten().cpu().numpy()
            all_values.extend(values)
            layer_info[name] = {
                'mean': float(values.mean()),
                'std': float(values.std()),
                'max': float(values.max())
            }
        
        all_values = np.array(all_values)
        print(f"  - 全局统计: mean={all_values.mean():.6f}, std={all_values.std():.6f}")
        print(f"  - 非零值比例: {(all_values > 1e-8).mean():.3f}")
        
        # 显示最重要的层
        sorted_layers = sorted(layer_info.items(), key=lambda x: x[1]['mean'], reverse=True)
        print("  - 最重要的5层:")
        for name, stats in sorted_layers[:5]:
            print(f"    * {name}: mean={stats['mean']:.6f}, max={stats['max']:.6f}")
            
    except Exception as e:
        print(f"✗ Fisher Information计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试剪枝掩码生成
    print("\n6. 测试剪枝掩码生成...")
    try:
        # 使用更合理的剪枝比例
        for ratio in [0.1, 0.3, 0.5]:
            masks = fisher_calc.generate_pruning_mask(fisher_info, ratio, 'global')
            
            total_params = sum(mask.numel() for mask in masks.values())
            kept_params = sum(mask.sum().item() for mask in masks.values())
            actual_ratio = 1 - (kept_params / total_params)
            
            print(f"  - 目标剪枝比例 {ratio:.1%}: 实际 {actual_ratio:.1%}")
        
        # 分析每层的剪枝情况
        masks_30 = fisher_calc.generate_pruning_mask(fisher_info, 0.3, 'layer_wise')
        sparsity_stats = fisher_calc.get_sparsity_stats(masks_30)
        
        print("  - 逐层剪枝稀疏度分析:")
        for layer, sparsity in list(sparsity_stats['layer_sparsity'].items())[:5]:
            print(f"    * {layer}: {sparsity:.1%}")
            
    except Exception as e:
        print(f"✗ 剪枝掩码生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n=== MovieLens Fisher Information测试完成 ===")
    print("✓ 所有功能在真实数据上正常工作！")
    return True

if __name__ == "__main__":
    success = test_fisher_on_movielens()
    if success:
        print("\n🎉 Fisher Information模块在真实数据上验证成功！")
        print("准备集成到pruning-aware蒸馏流程中...")
    else:
        print("\n❌ 测试失败，需要进一步调试。")
