#!/usr/bin/env python3
"""
LayerwiseAdapter增强版训练和评估脚本 - MovieLens Small版本
在MovieLens Small数据集上进行端到端训练和评估，用于与基准模型的公平比较
"""

import sys
import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Tuple, Any
import logging

# 添加路径
sys.path.append('/home/coder-gw/7Projects_in_7Days/online-inference-system')
sys.path.append('/home/coder-gw/7Projects_in_7Days/online-inference-system/layerwise_adapter_enhanced')

# 导入LayerwiseAdapter增强版组件
def import_components():
    """导入必要的组件，带有降级机制"""
    try:
        # 直接导入Python文件，避免包结构问题
        import importlib.util
        
        # 导入LayerwiseAdapterV2
        spec = importlib.util.spec_from_file_location(
            "layerwise_adapter_v2", 
            "/home/coder-gw/7Projects_in_7Days/online-inference-system/layerwise_adapter_enhanced/models/layerwise_adapter_v2.py"
        )
        layerwise_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(layerwise_module)
        
        # 导入TeacherManager
        spec = importlib.util.spec_from_file_location(
            "multi_teacher_manager", 
            "/home/coder-gw/7Projects_in_7Days/online-inference-system/layerwise_adapter_enhanced/teachers/multi_teacher_manager.py"
        )
        manager_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(manager_module)
        
        return layerwise_module.LayerwiseAdapterV2, manager_module.MultiTeacherManager
        
    except Exception as e:
        print(f"❌ 导入增强版组件失败: {e}")
        print("🔄 使用简化版本...")
        return None, None

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleTrainer:
    """MovieLens Small数据集训练器"""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 使用设备: {self.device}")
        
        # 尝试导入增强版组件
        self.LayerwiseAdapterV2, self.MultiTeacherManager = import_components()
        self.use_full_system = self.LayerwiseAdapterV2 is not None
        
        print(f"📦 系统模式: {'完整多Teacher融合' if self.use_full_system else '简化模拟'}")
    
    def load_movielens_small_data(self) -> Tuple[pd.DataFrame, Dict]:
        """加载MovieLens Small数据集"""
        try:
            print("📊 加载MovieLens Small数据集...")
            
            # MovieLens Small数据路径
            data_dir = "/home/coder-gw/7Projects_in_7Days/online-inference-system/data/movielens/small"
            ratings_file = os.path.join(data_dir, "ratings.csv")
            
            # 读取评分数据
            ratings_df = pd.read_csv(ratings_file)
            print(f"📈 原始数据: {len(ratings_df)} 条评分")
            print(f"📊 数据列: {ratings_df.columns.tolist()}")
            print(f"📊 数据预览:\n{ratings_df.head()}")
            
            # 数据预处理
            user_encoder = LabelEncoder()
            item_encoder = LabelEncoder()
            
            ratings_df['user_id_encoded'] = user_encoder.fit_transform(ratings_df['userId'])
            ratings_df['item_id_encoded'] = item_encoder.fit_transform(ratings_df['movieId'])
            
            # 创建元数据
            metadata = {
                'num_users': len(user_encoder.classes_),
                'num_items': len(item_encoder.classes_),
                'num_ratings': len(ratings_df),
                'user_mapping': {str(orig): int(enc) for orig, enc in 
                               zip(user_encoder.classes_, range(len(user_encoder.classes_)))},
                'item_mapping': {str(orig): int(enc) for orig, enc in 
                               zip(item_encoder.classes_, range(len(item_encoder.classes_)))}
            }
            
            print(f"👥 用户数: {metadata['num_users']}")
            print(f"🎬 物品数: {metadata['num_items']}")
            print(f"⭐ 评分数: {metadata['num_ratings']}")
            
            return ratings_df, metadata
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            raise
    
    def prepare_datasets(self, ratings_df: pd.DataFrame) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """准备训练、验证、测试数据集"""
        try:
            print("🔄 准备数据集...")
            
            # 提取特征和标签
            user_ids = ratings_df['user_id_encoded'].values
            item_ids = ratings_df['item_id_encoded'].values
            ratings = ratings_df['rating'].values.astype(np.float32)
            
            # 数据分割
            train_idx, test_idx = train_test_split(
                range(len(ratings_df)), 
                test_size=0.2, 
                random_state=42,
                stratify=ratings_df['rating'].values  # 保持评分分布
            )
            
            train_idx, val_idx = train_test_split(
                train_idx, 
                test_size=0.2, 
                random_state=42,
                stratify=ratings[train_idx]
            )
            
            print(f"🚂 训练集: {len(train_idx)} 样本")
            print(f"✅ 验证集: {len(val_idx)} 样本") 
            print(f"🔍 测试集: {len(test_idx)} 样本")
            
            # 创建数据加载器
            def create_dataloader(indices, batch_size, shuffle=True):
                user_tensor = torch.LongTensor(user_ids[indices])
                item_tensor = torch.LongTensor(item_ids[indices])
                rating_tensor = torch.FloatTensor(ratings[indices])
                
                dataset = TensorDataset(user_tensor, item_tensor, rating_tensor)
                return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
            
            train_loader = create_dataloader(train_idx, self.config['batch_size'], shuffle=True)
            val_loader = create_dataloader(val_idx, self.config['batch_size'], shuffle=False)
            test_loader = create_dataloader(test_idx, self.config['batch_size'], shuffle=False)
            
            return train_loader, val_loader, test_loader
            
        except Exception as e:
            print(f"❌ 数据集准备失败: {e}")
            raise
    
    def create_model(self, metadata: Dict) -> nn.Module:
        """创建模型"""
        try:
            if self.use_full_system:
                print("🔧 创建完整多Teacher融合模型...")
                
                # 创建完整的配置
                model_config = {
                    'embedding_dim': self.config['embedding_dim'],
                    'hidden_dim': 128,
                    'num_users': metadata['num_users'],
                    'num_items': metadata['num_items'],
                    'teacher_config': {
                        'fusion_mode': 'weighted',
                        'default_weights': {
                            'ensemble': 0.6,
                            'llama': 0.4
                        },
                        'fisher_config': {}
                    },
                    'fusion_config': {
                        'embedding_dim': self.config['embedding_dim'],
                        'teacher_count': 2,
                        'fusion_hidden_dim': 128
                    }
                }
                
                # 创建LayerwiseAdapterV2
                model = self.LayerwiseAdapterV2(model_config).to(self.device)
                
                print(f"✅ 完整模型创建成功")
                return model
                
            else:
                print("🔧 创建简化模拟模型...")
                # 创建简化版的神经网络模型
                model = SimpleRecommenderModel(
                    num_users=metadata['num_users'],
                    num_items=metadata['num_items'],
                    embedding_dim=self.config['embedding_dim']
                ).to(self.device)
                
                print(f"✅ 简化模型创建成功")
                return model
                
        except Exception as e:
            print(f"❌ 模型创建失败: {e}")
            raise
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                   optimizer: optim.Optimizer, criterion: nn.Module) -> float:
        """训练一个epoch"""
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (user_ids, item_ids, ratings) in enumerate(train_loader):
            user_ids = user_ids.to(self.device)
            item_ids = item_ids.to(self.device)
            ratings = ratings.to(self.device)
            
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(user_ids, item_ids)
            
            # 处理模型输出
            if isinstance(outputs, dict):
                predictions = outputs.get('prediction', outputs.get('predictions', outputs))
                if isinstance(predictions, torch.Tensor):
                    predictions = predictions.squeeze()
                else:
                    predictions = torch.tensor(predictions, device=self.device).squeeze()
            else:
                predictions = outputs.squeeze()
            
            # 计算损失
            loss = criterion(predictions, ratings)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 50 == 0:
                print(f"  批次 {batch_idx}/{len(train_loader)}, 损失: {loss.item():.4f}")
        
        return total_loss / num_batches
    
    def validate(self, model: nn.Module, val_loader: DataLoader, 
                criterion: nn.Module) -> float:
        """验证模型"""
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for user_ids, item_ids, ratings in val_loader:
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)
                ratings = ratings.to(self.device)
                
                # 前向传播
                outputs = model(user_ids, item_ids)
                
                # 处理模型输出
                if isinstance(outputs, dict):
                    predictions = outputs.get('prediction', outputs.get('predictions', outputs))
                    if isinstance(predictions, torch.Tensor):
                        predictions = predictions.squeeze()
                    else:
                        predictions = torch.tensor(predictions, device=self.device).squeeze()
                else:
                    predictions = outputs.squeeze()
                
                # 计算损失
                loss = criterion(predictions, ratings)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def test(self, model: nn.Module, test_loader: DataLoader) -> Dict:
        """测试模型"""
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for user_ids, item_ids, ratings in test_loader:
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)
                ratings = ratings.to(self.device)
                
                # 前向传播
                outputs = model(user_ids, item_ids)
                
                # 处理模型输出
                if isinstance(outputs, dict):
                    predictions = outputs.get('prediction', outputs.get('predictions', outputs))
                    if isinstance(predictions, torch.Tensor):
                        predictions = predictions.squeeze()
                    else:
                        predictions = torch.tensor(predictions, device=self.device).squeeze()
                else:
                    predictions = outputs.squeeze()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(ratings.cpu().numpy())
        
        # 计算指标
        predictions_array = np.array(all_predictions)
        targets_array = np.array(all_targets)
        
        rmse = np.sqrt(mean_squared_error(targets_array, predictions_array))
        mae = mean_absolute_error(targets_array, predictions_array)
        
        # 计算准确率（±0.5的容忍度）
        tolerance = 0.5
        accurate_predictions = np.abs(predictions_array - targets_array) <= tolerance
        accuracy = np.mean(accurate_predictions)
        
        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'accuracy': float(accuracy),
            'num_samples': len(all_predictions),
            'mean_prediction': float(np.mean(predictions_array)),
            'mean_target': float(np.mean(targets_array))
        }
    
    def train_and_evaluate(self) -> Dict:
        """完整的训练和评估流程"""
        try:
            print("🚀 开始LayerwiseAdapter增强版训练和评估 (MovieLens Small)...")
            
            # 1. 加载数据
            ratings_df, metadata = self.load_movielens_small_data()
            
            # 2. 准备数据集
            train_loader, val_loader, test_loader = self.prepare_datasets(ratings_df)
            
            # 3. 创建模型
            model = self.create_model(metadata)
            
            # 4. 设置优化器和损失函数
            optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
            criterion = nn.MSELoss()
            
            # 5. 训练循环
            print("🏃‍♂️ 开始训练...")
            train_losses = []
            val_losses = []
            best_val_loss = float('inf')
            
            for epoch in range(self.config['num_epochs']):
                print(f"\n📈 Epoch {epoch+1}/{self.config['num_epochs']}")
                
                # 训练
                train_loss = self.train_epoch(model, train_loader, optimizer, criterion)
                train_losses.append(train_loss)
                
                # 验证
                val_loss = self.validate(model, val_loader, criterion)
                val_losses.append(val_loss)
                
                print(f"训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
                
                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(f"✅ 发现更好的模型 (验证损失: {best_val_loss:.4f})")
            
            # 6. 测试评估
            print("\n🔍 测试评估...")
            test_results = self.test(model, test_loader)
            
            # 7. 汇总结果
            results = {
                'config': self.config,
                'metadata': metadata,
                'train_results': {
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'best_val_loss': best_val_loss,
                    'num_epochs': self.config['num_epochs']
                },
                'test_results': test_results,
                'use_full_system': self.use_full_system,
                'timestamp': time.time()
            }
            
            # 8. 保存结果
            output_path = "/home/coder-gw/7Projects_in_7Days/online-inference-system/layerwise_adapter_enhanced/experiments/full_training_results_small.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                # 处理numpy类型的JSON序列化
                def convert_numpy_types(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return obj
                
                # 递归转换所有numpy类型
                def recursive_convert(data):
                    if isinstance(data, dict):
                        return {k: recursive_convert(v) for k, v in data.items()}
                    elif isinstance(data, list):
                        return [recursive_convert(v) for v in data]
                    else:
                        return convert_numpy_types(data)
                
                converted_results = recursive_convert(results)
                json.dump(converted_results, f, indent=2, ensure_ascii=False)
            
            print(f"💾 结果已保存到: {output_path}")
            
            # 9. 打印最终结果
            print(f"\n🎯 最终结果:")
            print(f"RMSE: {test_results['rmse']:.4f}")
            print(f"MAE: {test_results['mae']:.4f}")
            print(f"准确率: {test_results['accuracy']:.4f} ({test_results['accuracy']*100:.1f}%)")
            print(f"测试样本数: {test_results['num_samples']}")
            
            return results
            
        except Exception as e:
            print(f"❌ 训练评估失败: {e}")
            import traceback
            traceback.print_exc()
            raise

class SimpleRecommenderModel(nn.Module):
    """简化的推荐模型，用于降级情况"""
    
    def __init__(self, num_users: int, num_items: int, embedding_dim: int):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # 拼接用户和物品嵌入
        combined = torch.cat([user_emb, item_emb], dim=-1)
        
        # 预测评分
        prediction = self.fc(combined).squeeze()
        
        return prediction

def main():
    """主函数"""
    print("=" * 60)
    print("LayerwiseAdapter增强版 - MovieLens Small 训练和评估")
    print("=" * 60)
    print("数据集: MovieLens Small")
    print("目标: 与基准模型进行公平性能比较")
    print("=" * 60)
    
    # 配置
    config = {
        'batch_size': 512,
        'learning_rate': 0.001,
        'num_epochs': 10,  # 增加epoch数以获得更好的性能
        'embedding_dim': 64
    }
    
    print(f"📋 配置: {config}")
    
    # 创建训练器并运行
    trainer = SimpleTrainer(config)
    results = trainer.train_and_evaluate()
    
    print("\n🎉 训练评估完成!")
    return results

if __name__ == "__main__":
    results = main()
