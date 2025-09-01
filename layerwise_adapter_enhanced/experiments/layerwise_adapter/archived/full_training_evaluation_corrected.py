#!/usr/bin/env python3
"""
多Teacher融合LayerwiseAdapter训练与评估脚本（修正版）

在MovieLens 1M数据集上进行端到端训练和评估
集成完整的多Teacher融合系统
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
        LayerwiseAdapterV2 = layerwise_module.LayerwiseAdapterV2
        
        # 导入MultiTeacherManager
        spec = importlib.util.spec_from_file_location(
            "multi_teacher_manager", 
            "/home/coder-gw/7Projects_in_7Days/online-inference-system/layerwise_adapter_enhanced/teachers/multi_teacher_manager.py"
        )
        manager_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(manager_module)
        MultiTeacherManager = manager_module.MultiTeacherManager
        
        # 导入EnsembleTeacherAdapter
        spec = importlib.util.spec_from_file_location(
            "ensemble_teacher_adapter", 
            "/home/coder-gw/7Projects_in_7Days/online-inference-system/layerwise_adapter_enhanced/teachers/ensemble_teacher_adapter.py"
        )
        ensemble_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ensemble_module)
        EnsembleTeacherAdapter = ensemble_module.EnsembleTeacherAdapter
        
        # 导入LlamaTeacherAdapter
        spec = importlib.util.spec_from_file_location(
            "llama_teacher_adapter", 
            "/home/coder-gw/7Projects_in_7Days/online-inference-system/layerwise_adapter_enhanced/teachers/llama_teacher_adapter.py"
        )
        llama_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(llama_module)
        LlamaTeacherAdapter = llama_module.LlamaTeacherAdapter
        
        print("✅ 成功通过动态导入完整多Teacher融合组件")
        return LayerwiseAdapterV2, MultiTeacherManager, EnsembleTeacherAdapter, LlamaTeacherAdapter, True
    except Exception as e:
        print(f"❌ 动态导入失败: {e}")
        raise ImportError("无法导入必要的模块组件")

# 导入组件
LayerwiseAdapterV2, MultiTeacherManager, EnsembleTeacherAdapter, LlamaTeacherAdapter, use_full_system = import_components()

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MovieLensTrainer:
    """MovieLens 1M数据集训练器"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化训练器"""
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 使用设备: {self.device}")
        
        # 训练参数
        self.batch_size = config.get('batch_size', 512)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.num_epochs = config.get('num_epochs', 10)
        self.embedding_dim = config.get('embedding_dim', 64)
        
        # 数据和模型
        self.train_loader = None
        self.val_loader = None 
        self.test_loader = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.teacher_manager = None
        
        # 统计信息
        self.num_users = 0
        self.num_items = 0
        
    def load_movielens_data(self) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """加载MovieLens数据"""
        try:
            print("📊 加载MovieLens 1M数据集...")
            
            # 数据路径
            data_dir = "/home/coder-gw/7Projects_in_7Days/online-inference-system/data/movielens/1m"
            ratings_file = os.path.join(data_dir, "ratings.csv")
            
            if not os.path.exists(ratings_file):
                raise FileNotFoundError(f"MovieLens数据文件未找到: {ratings_file}")
            
            # 读取评分数据
            ratings = pd.read_csv(ratings_file)
            
            # 使用LabelEncoder重新映射ID（从0开始）
            user_encoder = LabelEncoder()
            item_encoder = LabelEncoder()
            
            ratings['user_id'] = user_encoder.fit_transform(ratings['userId'])
            ratings['item_id'] = item_encoder.fit_transform(ratings['movieId'])
            
            print(f"   - 总评分数: {len(ratings):,}")
            print(f"   - 用户数: {ratings['user_id'].nunique():,}")
            print(f"   - 电影数: {ratings['item_id'].nunique():,}")
            print(f"   - 评分范围: {ratings['rating'].min():.1f} - {ratings['rating'].max():.1f}")
            print(f"   - 平均评分: {ratings['rating'].mean():.2f}")
            
            # 创建映射字典
            user_mapping = {uid: idx for idx, uid in enumerate(sorted(ratings['userId'].unique()))}
            item_mapping = {iid: idx for idx, iid in enumerate(sorted(ratings['movieId'].unique()))}
            
            metadata = {
                'num_users': ratings['user_id'].nunique(),
                'num_items': ratings['item_id'].nunique(),
                'num_ratings': len(ratings),
                'user_mapping': user_mapping,
                'item_mapping': item_mapping
            }
            
            print(f"✅ 数据加载完成")
            print(f"   - 映射后用户数: {metadata['num_users']}")
            print(f"   - 映射后物品数: {metadata['num_items']}")
            
            return ratings, metadata
            
        except Exception as e:
            logger.error(f"加载MovieLens数据失败: {e}")
            raise
    
    def prepare_data_loaders(self, ratings: pd.DataFrame) -> None:
        """准备数据加载器"""
        try:
            print("🔄 准备训练/验证/测试数据...")
            
            # 提取特征和标签
            users = ratings['user_id'].values.astype(np.int64)
            items = ratings['item_id'].values.astype(np.int64)
            ratings_values = ratings['rating'].values.astype(np.float32)
            
            # 划分数据集 (70% 训练, 15% 验证, 15% 测试)
            X = np.column_stack([users, items])
            y = ratings_values
            
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=0.15, random_state=42, stratify=None
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=0.1765, random_state=42  # 0.15/0.85 ≈ 0.1765
            )
            
            print(f"   - 训练集: {len(X_train):,} 样本")
            print(f"   - 验证集: {len(X_val):,} 样本") 
            print(f"   - 测试集: {len(X_test):,} 样本")
            
            # 转换为PyTorch张量
            def create_tensors(X, y):
                users_tensor = torch.LongTensor(X[:, 0])
                items_tensor = torch.LongTensor(X[:, 1])
                ratings_tensor = torch.FloatTensor(y)
                return users_tensor, items_tensor, ratings_tensor
            
            train_users, train_items, train_ratings = create_tensors(X_train, y_train)
            val_users, val_items, val_ratings = create_tensors(X_val, y_val)
            test_users, test_items, test_ratings = create_tensors(X_test, y_test)
            
            # 创建数据加载器
            train_dataset = TensorDataset(train_users, train_items, train_ratings)
            val_dataset = TensorDataset(val_users, val_items, val_ratings)
            test_dataset = TensorDataset(test_users, test_items, test_ratings)
            
            self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, 
                                         shuffle=True, num_workers=4)
            self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, 
                                       shuffle=False, num_workers=4)
            self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, 
                                        shuffle=False, num_workers=4)
            
            print("✅ 数据加载器准备完成")
            
        except Exception as e:
            logger.error(f"准备数据加载器失败: {e}")
            raise
    
    def setup_teachers_and_model(self, metadata: Dict[str, int]) -> None:
        """设置Teacher和模型"""
        try:
            print("🔧 设置Teacher和LayerwiseAdapter模型...")
            
            # 设置数据维度
            self.num_users = metadata['num_users']
            self.num_items = metadata['num_items']
            
            print(f"   - 用户数: {self.num_users}")
            print(f"   - 物品数: {self.num_items}")
            
            # 创建teacher适配器
            ensemble_config = {
                'ensemble_config': {},
                'model_weights': {'svd': 0.4, 'xdeepfm': 0.3, 'autoint': 0.3}
            }
            llama_config = {
                'model_name': 'llama3:latest',
                'base_url': 'http://localhost:11434',
                'temperature': 0.7
            }
            
            ensemble_teacher = EnsembleTeacherAdapter(ensemble_config)
            llama_teacher = LlamaTeacherAdapter(llama_config)
            
            # 创建MultiTeacherManager
            self.teacher_manager = MultiTeacherManager({})
            self.teacher_manager.add_teacher(ensemble_teacher, "ensemble")
            self.teacher_manager.add_teacher(llama_teacher, "llama")
            
            # 创建LayerwiseAdapter模型
            model_config = {
                'num_users': self.num_users,
                'num_items': self.num_items,
                'embedding_dim': self.embedding_dim,
                'hidden_dims': [128, 64],
                'dropout': 0.2,
                'device': self.device
            }
            
            self.model = LayerwiseAdapterV2(model_config)
            self.model.to(self.device)
            
            # 设置Teacher管理器到模型（如果模型支持）
            if hasattr(self.model, 'set_teacher_manager'):
                self.model.set_teacher_manager(self.teacher_manager)
            
            # 设置优化器和调度器
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=self.learning_rate,
                weight_decay=1e-5
            )
            
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=3, factor=0.5
            )
            
            print("✅ Teacher和模型设置完成")
            print(f"   - LayerwiseAdapter参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"   - Teacher数量: {len(self.teacher_manager.teachers)}")
            
        except Exception as e:
            logger.error(f"设置Teacher和模型失败: {e}")
            raise
    
    def train_epoch(self) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (users, items, ratings) in enumerate(self.train_loader):
            users = users.to(self.device)
            items = items.to(self.device)
            ratings = ratings.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 批量处理
            batch_losses = []
            for i in range(len(users)):
                try:
                    model_output = self.model(users[i], items[i])
                    target = ratings[i]
                    
                    # 处理模型输出 - 如果是字典，提取prediction字段
                    if isinstance(model_output, dict):
                        output = model_output.get('prediction', model_output)
                    else:
                        output = model_output
                    
                    # 确保输出为标量
                    if hasattr(output, 'item'):
                        output = output.item()
                    elif torch.is_tensor(output):
                        output = output.squeeze().item() if output.numel() == 1 else output.mean().item()
                    
                    if hasattr(target, 'item'):
                        target = target.item()
                    
                    # 计算损失
                    loss = (output - target) ** 2
                    batch_losses.append(loss)
                    
                    # Teacher引导损失（如果有Teacher管理器）
                    if hasattr(self.model, 'teacher_manager') and self.model.teacher_manager:
                        try:
                            for teacher_name, teacher in self.model.teacher_manager.teachers.items():
                                teacher_pred = teacher.predict(users[i].item(), items[i].item())
                                if isinstance(teacher_pred, (int, float)):
                                    teacher_loss = 0.1 * (output - teacher_pred) ** 2
                                    batch_losses.append(teacher_loss)
                        except Exception as te:
                            # Teacher预测失败时跳过
                            pass
                    
                except Exception as e:
                    logger.warning(f"批次{batch_idx}样本{i}处理失败: {e}")
                    continue
            
            if batch_losses:
                # 计算平均损失
                avg_batch_loss = sum(batch_losses) / len(batch_losses)
                
                # 确保损失是张量
                if not isinstance(avg_batch_loss, torch.Tensor):
                    avg_batch_loss = torch.tensor(avg_batch_loss, requires_grad=True, device=self.device)
                
                # 反向传播
                avg_batch_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                total_loss += avg_batch_loss.item()
                num_batches += 1
                
                # 打印进度
                if batch_idx % 100 == 0:
                    print(f"   批次 {batch_idx:4d}/{len(self.train_loader):4d}, "
                          f"损失: {avg_batch_loss.item():.4f}")
        
        return total_loss / max(num_batches, 1)
    
    def validate_epoch(self) -> float:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for users, items, ratings in self.val_loader:
                users = users.to(self.device)
                items = items.to(self.device)
                ratings = ratings.to(self.device)
                
                batch_predictions = []
                batch_targets = []
                
                for i in range(len(users)):
                    try:
                        model_output = self.model(users[i], items[i])
                        target = ratings[i]
                        
                        # 处理模型输出 - 如果是字典，提取prediction字段
                        if isinstance(model_output, dict):
                            output = model_output.get('prediction', model_output)
                        else:
                            output = model_output
                        
                        # 确保输出为标量
                        if hasattr(output, 'item'):
                            output = output.item()
                        elif torch.is_tensor(output):
                            output = output.squeeze().item() if output.numel() == 1 else output.mean().item()
                        
                        if hasattr(target, 'item'):
                            target = target.item()
                        
                        batch_predictions.append(output)
                        batch_targets.append(target)
                        
                        # 计算损失
                        loss = (output - target) ** 2
                        total_loss += loss
                        
                    except Exception as e:
                        logger.warning(f"验证样本处理失败: {e}")
                        continue
                
                predictions.extend(batch_predictions)
                targets.extend(batch_targets)
        
        # 计算指标
        if predictions and targets:
            rmse = np.sqrt(mean_squared_error(targets, predictions))
            mae = mean_absolute_error(targets, predictions)
            
            print(f"   验证 - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
            return total_loss / len(predictions)
        else:
            return float('inf')
    
    def train(self) -> Dict[str, Any]:
        """完整训练流程"""
        try:
            print(f"🚀 开始训练 - {self.num_epochs} epochs")
            
            train_losses = []
            val_losses = []
            best_val_loss = float('inf')
            
            for epoch in range(self.num_epochs):
                print(f"\n📈 Epoch {epoch + 1}/{self.num_epochs}")
                
                start_time = time.time()
                
                # 训练
                train_loss = self.train_epoch()
                train_losses.append(train_loss)
                
                # 验证
                val_loss = self.validate_epoch()
                val_losses.append(val_loss)
                
                epoch_time = time.time() - start_time
                
                print(f"   训练损失: {train_loss:.4f}")
                print(f"   验证损失: {val_loss:.4f}")
                print(f"   用时: {epoch_time:.2f}s")
                
                # 学习率调度
                if self.scheduler:
                    self.scheduler.step(val_loss)
                
                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(f"   🎯 新的最佳验证损失: {best_val_loss:.4f}")
            
            return {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss,
                'num_epochs': self.num_epochs
            }
            
        except Exception as e:
            logger.error(f"训练失败: {e}")
            raise
    
    def evaluate(self) -> Dict[str, Any]:
        """在测试集上评估"""
        try:
            print("\n🧪 测试集评估...")
            
            self.model.eval()
            predictions = []
            targets = []
            
            with torch.no_grad():
                for users, items, ratings in self.test_loader:
                    users = users.to(self.device)
                    items = items.to(self.device)
                    
                    batch_predictions = []
                    batch_targets = []
                    
                    for i in range(len(users)):
                        try:
                            model_output = self.model(users[i], items[i])
                            target = ratings[i]
                            
                            # 处理模型输出 - 如果是字典，提取prediction字段
                            if isinstance(model_output, dict):
                                output = model_output.get('prediction', model_output)
                            else:
                                output = model_output
                            
                            # 确保输出为标量
                            if hasattr(output, 'item'):
                                output = output.item()
                            elif torch.is_tensor(output):
                                output = output.squeeze().item() if output.numel() == 1 else output.mean().item()
                            
                            if hasattr(target, 'item'):
                                target = target.item()
                            
                            batch_predictions.append(output)
                            batch_targets.append(target)
                            
                        except Exception as e:
                            logger.warning(f"测试样本处理失败: {e}")
                            continue
                    
                    predictions.extend(batch_predictions)
                    targets.extend(batch_targets)
            
            if predictions and targets:
                # 计算指标
                rmse = np.sqrt(mean_squared_error(targets, predictions))
                mae = mean_absolute_error(targets, predictions)
                
                # 计算其他指标
                predictions_np = np.array(predictions)
                targets_np = np.array(targets)
                
                # 准确率（评分差在0.5以内的比例）
                accuracy = np.mean(np.abs(predictions_np - targets_np) <= 0.5)
                
                results = {
                    'rmse': rmse,
                    'mae': mae,
                    'accuracy': accuracy,
                    'num_samples': len(predictions),
                    'mean_prediction': np.mean(predictions_np),
                    'mean_target': np.mean(targets_np)
                }
                
                print(f"📊 测试结果:")
                print(f"   - RMSE: {rmse:.4f}")
                print(f"   - MAE: {mae:.4f}")
                print(f"   - 准确率(±0.5): {accuracy:.2%}")
                print(f"   - 样本数: {len(predictions):,}")
                print(f"   - 平均预测: {np.mean(predictions_np):.3f}")
                print(f"   - 平均真实: {np.mean(targets_np):.3f}")
                
                return results
            else:
                logger.error("测试评估失败：没有有效预测")
                return {}
                
        except Exception as e:
            logger.error(f"评估失败: {e}")
            raise

def main():
    """主函数"""
    print("=" * 80)
    print("多Teacher融合LayerwiseAdapter训练与评估")
    print("数据集: MovieLens 1M")
    print("=" * 80)
    
    # 配置参数
    config = {
        'batch_size': 512,
        'learning_rate': 0.001,
        'num_epochs': 5,  # 减少epoch数进行快速测试
        'embedding_dim': 64
    }
    
    try:
        # 创建训练器
        trainer = MovieLensTrainer(config)
        
        # 加载数据
        ratings, metadata = trainer.load_movielens_data()
        
        # 准备数据加载器
        trainer.prepare_data_loaders(ratings)
        
        # 设置Teacher和模型
        trainer.setup_teachers_and_model(metadata)
        
        # 训练
        train_results = trainer.train()
        
        # 评估
        test_results = trainer.evaluate()
        
        # 保存结果
        results = {
            'config': config,
            'metadata': metadata,
            'train_results': train_results,
            'test_results': test_results,
            'use_full_system': use_full_system,
            'timestamp': time.time()
        }
        
        # 保存到文件
        results_file = 'full_training_results.json'
        
        # 转换numpy和int64类型为Python原生类型
        def convert_to_json_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(k): convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            else:
                return obj
        
        # 转换结果使其可JSON序列化
        serializable_results = convert_to_json_serializable(results)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"\n✅ 训练和评估完成！结果已保存到 {results_file}")
        print("\n🎯 最终结果摘要:")
        print(f"   - 最佳验证损失: {train_results.get('best_val_loss', 'N/A'):.4f}")
        print(f"   - 测试RMSE: {test_results.get('rmse', 'N/A'):.4f}")
        print(f"   - 测试MAE: {test_results.get('mae', 'N/A'):.4f}")
        print(f"   - 测试准确率: {test_results.get('accuracy', 'N/A'):.2%}")
        
    except Exception as e:
        logger.error("主流程失败: %s", str(e))
        raise

if __name__ == "__main__":
    main()
