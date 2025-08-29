#!/usr/bin/env python3
"""
测试Fisher Information per-layer计算模块
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

class MockStudentModel(nn.Module):
    """简单的mock学生模型"""
    
    def __init__(self, embedding_dim=32, num_users=1000, num_items=1000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_users = num_users
        self.num_items = num_items
        
        # 用户和物品嵌入
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # 简单的预测头
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, batch_data):
        """处理batch数据"""
        if isinstance(batch_data, (tuple, list)):
            user_ids, item_ids = batch_data[0], batch_data[1]
        else:
            # 假设输入是[batch_size, 2]格式
            user_ids = batch_data[:, 0].long()
            item_ids = batch_data[:, 1].long()
            
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        combined = torch.cat([user_emb, item_emb], dim=-1)
        return self.predictor(combined).squeeze()

def create_mock_dataloader(batch_size=32, num_batches=10, num_users=1000, num_items=1000):
    """创建mock DataLoader"""
    total_samples = batch_size * num_batches
    
    # 创建随机用户ID和物品ID
    user_ids = torch.randint(0, num_users, (total_samples,))
    item_ids = torch.randint(0, num_items, (total_samples,))
    ratings = torch.rand(total_samples)
    
    # 创建输入张量 [user_id, item_id]
    inputs = torch.stack([user_ids, item_ids], dim=1).float()
    
    dataset = TensorDataset(inputs, ratings)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader

def test_fisher_calculator():
    """测试Fisher Information计算器"""
    print("=== 测试Fisher Information per-layer计算器 ===")
    
    # 检查CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建学生模型
    student_model = MockStudentModel().to(device)
    print(f"学生模型参数数量: {sum(p.numel() for p in student_model.parameters())}")
    
    # 创建Fisher计算器
    fisher_calc = FisherInformationCalculator(student_model, device=str(device))
    
    # 创建数据加载器
    dataloader = create_mock_dataloader(batch_size=32, num_batches=5)
    print(f"准备了数据加载器，batch数量: 5")
    
    # 定义损失函数
    criterion = nn.MSELoss()
    
    # 计算Fisher Information
    print("\n1. 计算Fisher Information...")
    try:
        fisher_info = fisher_calc.compute_fisher_information(
            dataloader, criterion, num_batches=5
        )
        print(f"✓ Fisher Information计算成功！")
        print(f"  - 覆盖层数: {len(fisher_info)}")
        print(f"  - 层名称: {list(fisher_info.keys())[:3]}...")  # 显示前3个
        
        # 检查每层的Fisher信息
        for name, fisher_tensor in list(fisher_info.items())[:3]:
            print(f"  - {name}: shape={fisher_tensor.shape}, mean={fisher_tensor.mean():.6f}")
            
    except Exception as e:
        print(f"✗ Fisher Information计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 分析重要性
    print("\n2. 分析参数重要性...")
    try:
        importance = fisher_calc.analyze_importance(fisher_info)
        print(f"✓ 重要性分析成功！")
        print(f"  - 总体统计: mean={importance['global_stats']['mean']:.6f}")
        print(f"  - 层级统计数量: {len(importance['layer_stats'])}")
        
        # 显示最重要的几层
        layer_means = {name: stats['mean'] for name, stats in importance['layer_stats'].items()}
        top_layers = sorted(layer_means.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"  - 最重要的3层:")
        for name, mean_val in top_layers:
            print(f"    * {name}: {mean_val:.6f}")
            
    except Exception as e:
        print(f"✗ 重要性分析失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 生成剪枝掩码
    print("\n3. 生成剪枝掩码...")
    try:
        # 测试全局策略
        global_masks = fisher_calc.generate_pruning_mask(
            fisher_info, 0.2, 'global'
        )
        print("✓ 全局剪枝掩码生成成功！")
        print(f"  - 掩码层数: {len(global_masks)}")
        
        # 统计剪枝比例
        total_params = 0
        pruned_params = 0
        for name, mask in global_masks.items():
            total_params += mask.numel()
            pruned_params += (mask == 0).sum().item()
        
        actual_ratio = pruned_params / total_params
        print(f"  - 实际剪枝比例: {actual_ratio:.3f} (目标: 0.2)")
        
        # 测试逐层策略
        layer_masks = fisher_calc.generate_pruning_mask(
            fisher_info, 0.3, 'layer_wise'
        )
        print("✓ 逐层剪枝掩码生成成功！")
        print(f"  - 掩码层数: {len(layer_masks)}")
        
    except Exception as e:
        print(f"✗ 剪枝掩码生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试稀疏性检查
    print("\n4. 测试稀疏性检查...")
    try:
        sparsity_stats = fisher_calc.get_sparsity_stats(global_masks)
        print(f"✓ 稀疏性统计成功！")
        print(f"  - 总体稀疏度: {sparsity_stats['overall_sparsity']:.3f}")
        print(f"  - 非零参数比例: {sparsity_stats['non_zero_ratio']:.3f}")
        print(f"  - 层级稀疏度范围: {sparsity_stats['layer_sparsity_range']}")
        
    except Exception as e:
        print(f"✗ 稀疏性检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n=== Fisher Information模块测试完成 ===")
    print("✓ 所有功能正常工作！")
    return True

if __name__ == "__main__":
    success = test_fisher_calculator()
    if success:
        print("\n🎉 Fisher Information per-layer模块已就绪！")
        print("可以继续集成到蒸馏流程中。")
    else:
        print("\n❌ 测试失败，需要修复问题。")
