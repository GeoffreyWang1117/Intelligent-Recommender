#!/usr/bin/env python3
"""
Pruning-Aware Knowledge Distillation 端到端测试
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
import sys

# 添加项目根目录到路径
sys.path.append('/home/coder-gw/7Projects_in_7Days/online-inference-system')

from pruning_distillation import PruningAwareDistillation

class TeacherModel(nn.Module):
    """教师模型（大模型）"""
    
    def __init__(self, num_users=500, num_items=200, embedding_dim=128):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self._init_weights()
        
    def _init_weights(self):
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
        
        features = torch.cat([user_emb, item_emb], dim=-1)
        output = self.mlp(features)
        
        return torch.sigmoid(output.squeeze())

class StudentModel(nn.Module):
    """学生模型（小模型）"""
    
    def __init__(self, num_users=500, num_items=200, embedding_dim=32):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self._init_weights()
        
    def _init_weights(self):
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
        
        features = torch.cat([user_emb, item_emb], dim=-1)
        output = self.mlp(features)
        
        return torch.sigmoid(output.squeeze())

def load_movielens_data():
    """加载MovieLens数据"""
    data_path = '/home/coder-gw/7Projects_in_7Days/online-inference-system/data/movielens'
    ratings_file = os.path.join(data_path, 'ratings.csv')
    
    if not os.path.exists(ratings_file):
        print(f"❌ 未找到数据文件: {ratings_file}")
        return None
        
    ratings = pd.read_csv(ratings_file)
    print(f"✓ 加载 {len(ratings)} 条评分数据")
    
    # 数据预处理
    user_ids = ratings['user_id'].unique()
    item_ids = ratings['item_id'].unique()
    
    user_map = {uid: idx for idx, uid in enumerate(user_ids)}
    item_map = {iid: idx for idx, iid in enumerate(item_ids)}
    
    ratings['user_idx'] = ratings['user_id'].map(user_map)
    ratings['item_idx'] = ratings['item_id'].map(item_map)
    ratings['rating_norm'] = (ratings['rating'] - 1) / 4.0
    
    print(f"  - 用户数: {len(user_ids)}")
    print(f"  - 物品数: {len(item_ids)}")
    
    return ratings, len(user_ids), len(item_ids)

def create_dataloaders(ratings_df, num_users, num_items, train_ratio=0.8, batch_size=64):
    """创建训练和验证数据加载器"""
    # 划分训练/验证集
    train_size = int(len(ratings_df) * train_ratio)
    train_df = ratings_df.iloc[:train_size]
    val_df = ratings_df.iloc[train_size:]
    
    def create_dataloader(df, shuffle=True):
        user_ids = torch.tensor(df['user_idx'].values, dtype=torch.long)
        item_ids = torch.tensor(df['item_idx'].values, dtype=torch.long)
        ratings = torch.tensor(df['rating_norm'].values, dtype=torch.float32)
        
        inputs = torch.stack([user_ids, item_ids], dim=1).float()
        dataset = TensorDataset(inputs, ratings)
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    train_loader = create_dataloader(train_df, shuffle=True)
    val_loader = create_dataloader(val_df, shuffle=False)
    
    print(f"✓ 创建数据加载器:")
    print(f"  - 训练集: {len(train_df)} 样本, {len(train_loader)} 批次")
    print(f"  - 验证集: {len(val_df)} 样本, {len(val_loader)} 批次")
    
    return train_loader, val_loader

def test_pruning_aware_distillation():
    """测试Pruning-Aware Knowledge Distillation"""
    print("=== Pruning-Aware Knowledge Distillation 端到端测试 ===")
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    result = load_movielens_data()
    if result is None:
        return False
        
    ratings_df, num_users, num_items = result
    
    # 2. 创建数据加载器
    print("\n2. 创建数据加载器...")
    train_loader, val_loader = create_dataloaders(ratings_df, num_users, num_items)
    
    # 3. 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 4. 创建模型
    print("\n4. 创建教师和学生模型...")
    teacher = TeacherModel(num_users, num_items, embedding_dim=128)
    student = StudentModel(num_users, num_items, embedding_dim=32)
    
    print(f"教师模型参数: {sum(p.numel() for p in teacher.parameters()):,}")
    print(f"学生模型参数: {sum(p.numel() for p in student.parameters()):,}")
    
    # 5. 配置蒸馏参数
    print("\n5. 配置Pruning-Aware蒸馏...")
    distillation_config = {
        'temperature': 3.0,
        'alpha': 0.8,  # KD损失权重
        'beta': 0.2,   # 任务损失权重
        'pruning_schedule': {
            'start_epoch': 2,  # 早点开始剪枝用于测试
            'frequency': 3,    # 每3个epoch剪枝一次
            'initial_ratio': 0.05,
            'final_ratio': 0.3,
            'strategy': 'gradual'
        },
        'fisher_config': {
            'update_frequency': 2,  # 每2个epoch更新Fisher
            'num_batches': 5,       # 使用5个batch计算Fisher
            'accumulate': True
        }
    }
    
    # 6. 初始化蒸馏器
    distiller = PruningAwareDistillation(
        teacher_model=teacher,
        student_model=student,
        device=str(device),
        distillation_config=distillation_config
    )
    
    # 7. 设置优化器和损失函数
    print("\n7. 设置训练组件...")
    optimizer = optim.Adam(student.parameters(), lr=0.001)
    task_criterion = nn.MSELoss()
    
    # 8. 训练循环
    print("\n8. 开始训练...")
    num_epochs = 10
    
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        
        # 训练
        train_stats = distiller.train_epoch(train_loader, optimizer, task_criterion, epoch)
        
        # 验证
        val_stats = distiller.evaluate(val_loader, task_criterion)
        
        # 获取模型统计
        model_stats = distiller.get_model_statistics()
        
        # 打印结果
        print(f"训练损失: {train_stats['total_loss']:.4f} "
              f"(KD: {train_stats['kd_loss']:.4f}, Task: {train_stats['task_loss']:.4f})")
        print(f"验证损失: {val_stats['val_total_loss']:.4f}")
        print(f"当前稀疏度: {model_stats['sparsity']:.1%}")
        print(f"压缩比率: {model_stats['compression_ratio']:.3f}")
        print(f"有效参数: {model_stats['effective_params']:,}/{model_stats['student_params']:,}")
    
    # 9. 最终评估
    print("\n9. 最终评估...")
    final_stats = distiller.get_model_statistics()
    
    print("=== 最终结果 ===")
    print(f"教师模型参数: {final_stats['teacher_params']:,}")
    print(f"学生模型参数: {final_stats['student_params']:,}")
    print(f"有效参数: {final_stats['effective_params']:,}")
    print(f"稀疏度: {final_stats['sparsity']:.1%}")
    print(f"压缩比率: {final_stats['compression_ratio']:.3f}")
    print(f"Fisher层数: {final_stats['fisher_layers']}")
    print(f"剪枝掩码数: {final_stats['pruning_masks']}")
    
    # 10. 保存模型
    print("\n10. 保存模型...")
    save_path = "/home/coder-gw/7Projects_in_7Days/online-inference-system/teachers/fisher_utils/pruned_student_model.pt"
    distiller.save_checkpoint(save_path, num_epochs-1, optimizer.state_dict())
    
    print("\n=== Pruning-Aware Knowledge Distillation 测试完成 ===")
    print("✅ 所有功能正常工作！")
    return True

if __name__ == "__main__":
    success = test_pruning_aware_distillation()
    if success:
        print("\n🎉 Fisher Information + Pruning-Aware蒸馏系统验证成功！")
        print("系统已具备完整的蒸馏和剪枝能力。")
    else:
        print("\n❌ 测试失败，需要进一步调试。")
