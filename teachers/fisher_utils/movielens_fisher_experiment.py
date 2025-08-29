#!/usr/bin/env python3
"""
基于真实MovieLens数据和Ensemble Teacher的Fisher Information实验
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
import sys
import pickle
from typing import Dict, Any, List, Tuple

# 添加项目根目录到路径
sys.path.append('/home/coder-gw/7Projects_in_7Days/online-inference-system')

from teachers.fisher_utils.ensemble_fisher_calculator import EnsembleFisherCalculator
from models.optimized_ensemble_teacher import OptimizedEnsembleTeacher
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MovieLensDataLoader:
    """MovieLens数据加载器"""
    
    def __init__(self, data_path='/home/coder-gw/7Projects_in_7Days/online-inference-system/data/movielens'):
        self.data_path = data_path
        self.ratings_data = None
        self.user_map = {}
        self.item_map = {}
        self.num_users = 0
        self.num_items = 0
        
    def load_data(self) -> bool:
        """加载MovieLens数据"""
        logger.info("📥 加载MovieLens数据...")
        
        try:
            # 尝试加载评分数据
            ratings_file = os.path.join(self.data_path, 'ratings.csv')
            if os.path.exists(ratings_file):
                self.ratings_data = pd.read_csv(ratings_file)
                logger.info(f"✅ 从ratings.csv加载 {len(self.ratings_data)} 条评分")
            else:
                # 使用模拟数据
                logger.warning("⚠️ 未找到ratings.csv，生成模拟数据")
                self.ratings_data = self._generate_mock_data()
            
            # 构建用户和物品映射
            self._build_mappings()
            return True
            
        except Exception as e:
            logger.error(f"❌ 数据加载失败: {e}")
            return False
    
    def _generate_mock_data(self) -> pd.DataFrame:
        """生成模拟MovieLens数据"""
        np.random.seed(42)
        
        num_users = 610
        num_items = 9724  
        num_ratings = 100836
        
        # 生成符合MovieLens分布的数据
        user_ids = np.random.choice(num_users, num_ratings, replace=True)
        item_ids = np.random.choice(num_items, num_ratings, replace=True)
        
        # 生成评分（偏向高分，符合MovieLens特性）
        ratings = np.random.choice([1, 2, 3, 4, 5], num_ratings, 
                                 p=[0.05, 0.1, 0.2, 0.35, 0.3])
        
        # 添加时间戳
        timestamps = np.random.randint(900000000, 1500000000, num_ratings)
        
        return pd.DataFrame({
            'user_id': user_ids,
            'item_id': item_ids, 
            'rating': ratings,
            'timestamp': timestamps
        })
    
    def _build_mappings(self):
        """构建用户和物品ID映射"""
        unique_users = self.ratings_data['user_id'].unique()
        unique_items = self.ratings_data['item_id'].unique()
        
        self.user_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.item_map = {iid: idx for idx, iid in enumerate(unique_items)}
        
        self.num_users = len(unique_users)
        self.num_items = len(unique_items)
        
        # 添加映射后的索引
        self.ratings_data['user_idx'] = self.ratings_data['user_id'].map(self.user_map)
        self.ratings_data['item_idx'] = self.ratings_data['item_id'].map(self.item_map)
        
        logger.info(f"📊 数据统计: {self.num_users}用户, {self.num_items}物品, {len(self.ratings_data)}评分")
    
    def get_dataloader(self, batch_size=1024, split='train') -> DataLoader:
        """获取数据加载器"""
        if self.ratings_data is None:
            raise ValueError("数据未加载，请先调用load_data()")
        
        # 简单的训练/测试分割
        if split == 'train':
            data = self.ratings_data.sample(frac=0.8, random_state=42)
        else:
            data = self.ratings_data.drop(self.ratings_data.sample(frac=0.8, random_state=42).index)
        
        # 准备张量数据
        user_tensor = torch.LongTensor(data['user_idx'].values)
        item_tensor = torch.LongTensor(data['item_idx'].values)
        rating_tensor = torch.FloatTensor((data['rating'].values - 1) / 4.0)  # 归一化到[0,1]
        
        dataset = TensorDataset(user_tensor, item_tensor, rating_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class SimpleRecommenderModel(nn.Module):
    """简化的推荐模型，用于Fisher分析演示"""
    
    def __init__(self, num_users, num_items, embedding_dim=64):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        
        # 嵌入层
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # 预测网络
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        
        for layer in self.predictor:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.1)
    
    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        features = torch.cat([user_emb, item_emb], dim=-1)
        prediction = self.predictor(features)
        
        return prediction.squeeze()


def run_movielens_fisher_experiment():
    """运行MovieLens Fisher Information实验"""
    logger.info("🚀 开始MovieLens Fisher Information实验")
    logger.info("="*60)
    
    # 1. 加载数据
    logger.info("📋 步骤1: 数据加载")
    data_loader = MovieLensDataLoader()
    if not data_loader.load_data():
        logger.error("❌ 数据加载失败，终止实验")
        return False
    
    train_dataloader = data_loader.get_dataloader(batch_size=512, split='train')
    test_dataloader = data_loader.get_dataloader(batch_size=512, split='test')
    
    # 2. 初始化Ensemble Teacher
    logger.info("📋 步骤2: Ensemble Teacher初始化")
    try:
        ensemble_teacher = OptimizedEnsembleTeacher()
        logger.info("✅ Ensemble Teacher初始化成功")
    except Exception as e:
        logger.error(f"❌ Ensemble Teacher初始化失败: {e}")
        return False
    
    # 3. 创建简化模型用于演示Fisher分析
    logger.info("📋 步骤3: 创建演示模型")
    demo_model = SimpleRecommenderModel(
        num_users=data_loader.num_users,
        num_items=data_loader.num_items,
        embedding_dim=64
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    demo_model.to(device)
    logger.info(f"✅ 演示模型创建成功，设备: {device}")
    
    # 4. 运行Fisher Information分析
    logger.info("📋 步骤4: Fisher Information分析")
    try:
        # 使用Ensemble Fisher计算器
        fisher_calc = EnsembleFisherCalculator(ensemble_teacher, device=str(device))
        
        # 运行完整分析
        analysis_results = fisher_calc.analyze_ensemble_fisher(
            dataloader=train_dataloader,
            num_batches=100
        )
        
        logger.info("✅ Fisher分析完成")
        
    except Exception as e:
        logger.error(f"❌ Fisher分析失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. 保存和展示结果
    logger.info("📋 步骤5: 结果保存与展示")
    try:
        # 保存分析结果
        save_path = '/home/coder-gw/7Projects_in_7Days/online-inference-system/analysis_results/movielens_fisher_experiment.json'
        fisher_calc.save_analysis_results(analysis_results, save_path)
        
        # 展示关键结果
        _display_key_results(analysis_results)
        
        logger.info("✅ 结果保存与展示完成")
        
    except Exception as e:
        logger.error(f"❌ 结果处理失败: {e}")
        return False
    
    logger.info("🎉 MovieLens Fisher Information实验成功完成！")
    return True


def _display_key_results(results: Dict[str, Any]):
    """展示关键实验结果"""
    print("\n" + "="*60)
    print("🎯 核心实验结果")
    print("="*60)
    
    # 1. 模型剪枝友好度排名
    if 'ensemble_comparison' in results:
        comparison = results['ensemble_comparison']
        if 'pruning_friendly_ranking' in comparison:
            print("\n📊 模型剪枝友好度排名:")
            for i, model in enumerate(comparison['pruning_friendly_ranking'], 1):
                print(f"   {i}. {model['model']} (分数: {model['score']}) - {model['reason']}")
    
    # 2. 推荐剪枝策略
    if 'pruning_suggestions' in results:
        pruning = results['pruning_suggestions']
        if 'recommended_strategy' in pruning:
            strategy = pruning['recommended_strategy']
            print(f"\n✂️ 推荐剪枝策略:")
            print(f"   目标压缩率: {strategy.get('target_compression', 0)*100:.1f}%")
            print(f"   预期性能损失: {strategy.get('expected_performance_loss', 0)*100:.1f}%")
    
    # 3. 性能预测
    if 'performance_prediction' in results:
        perf = results['performance_prediction']
        if 'efficiency_gains' in perf:
            gains = perf['efficiency_gains']
            print(f"\n🚀 效率提升预测:")
            print(f"   推理加速: {gains.get('inference_speedup', 0):.1f}x")
            print(f"   内存减少: {gains.get('memory_reduction', 0)*100:.1f}%")
            print(f"   能耗节省: {gains.get('energy_savings', 0)*100:.1f}%")
    
    # 4. 关键层重要性
    if 'layer_importance' in results:
        layers = results['layer_importance']
        if 'critical_layers' in layers:
            print(f"\n🎯 关键层重要性:")
            for layer in layers['critical_layers'][:3]:  # 显示前3个
                print(f"   {layer['model']}.{layer['layer']}: {layer['importance']:.2f} - {layer['reason']}")
    
    print("\n" + "="*60)


def main():
    """主函数"""
    success = run_movielens_fisher_experiment()
    if success:
        print("\n🎉 实验完成！检查analysis_results目录查看详细结果。")
    else:
        print("\n❌ 实验失败，请检查日志信息。")


if __name__ == "__main__":
    main()
