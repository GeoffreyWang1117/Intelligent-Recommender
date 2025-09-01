"""
Multi-Teacher Fusion System - Model Trainer
统一训练器，支持单模型和多Teacher融合训练
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import logging
import json
import time
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..config.base_config import SystemConfig, DatasetConfig
from ..config.teacher_config import TeacherConfig, EnsembleConfig, FusionConfig
from ..teachers.base_teacher import BaseTeacher
from ..utils.evaluator import RecommendationEvaluator, EvaluationResult

@dataclass
class TrainingHistory:
    """训练历史记录"""
    epoch: int
    train_loss: float
    val_loss: float
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    training_time: float

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, 
                 system_config: SystemConfig,
                 dataset_config: DatasetConfig,
                 save_dir: str = "./saved_models"):
        """
        初始化训练器
        
        Args:
            system_config: 系统配置
            dataset_config: 数据集配置
            save_dir: 模型保存目录
        """
        self.system_config = system_config
        self.dataset_config = dataset_config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.evaluator = RecommendationEvaluator()
        
        # 训练状态
        self.training_history: List[TrainingHistory] = []
        self.best_model_state = None
        self.best_val_score = float('-inf')
        
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() and system_config.use_cuda else 'cpu')
        self.logger.info(f"使用设备: {self.device}")
    
    def prepare_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                    batch_size: int = 256) -> Tuple[DataLoader, DataLoader]:
        """
        准备训练数据
        
        Args:
            train_df: 训练数据
            val_df: 验证数据
            batch_size: 批次大小
        """
        self.logger.info("准备训练数据...")
        
        # 提取特征和标签
        feature_cols = ['user_id', 'item_id']
        
        # 添加额外特征
        extra_features = []
        for col in train_df.columns:
            if col not in ['rating', 'userId', 'movieId', 'timestamp'] and col not in feature_cols:
                if train_df[col].dtype in ['int64', 'float64']:
                    extra_features.append(col)
        
        feature_cols.extend(extra_features)
        
        # 训练集
        X_train = train_df[feature_cols].values.astype(np.float32)
        y_train = train_df['rating'].values.astype(np.float32)
        
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 验证集
        X_val = val_df[feature_cols].values.astype(np.float32)
        y_val = val_df['rating'].values.astype(np.float32)
        
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        self.logger.info(f"训练集: {len(train_dataset)} 样本, 验证集: {len(val_dataset)} 样本")
        self.logger.info(f"特征维度: {len(feature_cols)}")
        
        return train_loader, val_loader
    
    def train_single_teacher(self, 
                           teacher: BaseTeacher,
                           train_loader: DataLoader,
                           val_loader: DataLoader,
                           epochs: int = 100,
                           learning_rate: float = 0.001,
                           patience: int = 10,
                           save_best: bool = True) -> Dict[str, Any]:
        """
        训练单个Teacher模型
        
        Args:
            teacher: Teacher实例
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            learning_rate: 学习率
            patience: 早停耐心值
            save_best: 是否保存最佳模型
        """
        self.logger.info(f"开始训练 {teacher.__class__.__name__}...")
        
        # 检查Teacher是否支持训练
        if not hasattr(teacher, 'train_step'):
            self.logger.warning(f"{teacher.__class__.__name__} 不支持训练，跳过")
            return {"status": "skipped", "reason": "no training support"}
        
        # 初始化优化器和损失函数
        if hasattr(teacher, 'model') and teacher.model is not None:
            teacher.model.to(self.device)
            optimizer = torch.optim.Adam(teacher.model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()
        else:
            self.logger.warning(f"{teacher.__class__.__name__} 没有PyTorch模型，使用自定义训练")
            return self._train_custom_teacher(teacher, train_loader, val_loader, epochs)
        
        # 训练循环
        best_val_loss = float('inf')
        patience_counter = 0
        training_start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # 训练阶段
            teacher.model.train()
            train_losses = []
            
            for batch_idx, (features, targets) in enumerate(train_loader):
                features, targets = features.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                
                # 前向传播
                if hasattr(teacher, 'train_step'):
                    loss = teacher.train_step(features, targets, criterion)
                else:
                    predictions = teacher.model(features)
                    loss = criterion(predictions.squeeze(), targets)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # 验证阶段
            teacher.model.eval()
            val_losses = []
            
            with torch.no_grad():
                for features, targets in val_loader:
                    features, targets = features.to(self.device), targets.to(self.device)
                    
                    if hasattr(teacher, 'predict_batch'):
                        predictions = teacher.predict_batch(features)
                    else:
                        predictions = teacher.model(features).squeeze()
                    
                    val_loss = criterion(predictions, targets)
                    val_losses.append(val_loss.item())
            
            # 计算平均损失
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            epoch_time = time.time() - epoch_start_time
            
            # 记录训练历史
            history = TrainingHistory(
                epoch=epoch + 1,
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
                train_metrics={},
                val_metrics={},
                training_time=epoch_time
            )
            self.training_history.append(history)
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                self.logger.info(f"Epoch {epoch+1}/{epochs}: "
                               f"Train Loss: {avg_train_loss:.4f}, "
                               f"Val Loss: {avg_val_loss:.4f}, "
                               f"Time: {epoch_time:.2f}s")
            
            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                if save_best:
                    self.best_model_state = teacher.model.state_dict().copy()
                    self.best_val_score = -avg_val_loss  # 负损失作为分数
            else:
                patience_counter += 1
                
                if patience_counter >= patience:
                    self.logger.info(f"早停在第 {epoch+1} 轮，最佳验证损失: {best_val_loss:.4f}")
                    break
        
        total_training_time = time.time() - training_start_time
        
        # 恢复最佳模型
        if save_best and self.best_model_state is not None:
            teacher.model.load_state_dict(self.best_model_state)
        
        # 保存模型
        if save_best:
            model_path = self.save_dir / f"{teacher.__class__.__name__}_best.pth"
            torch.save(teacher.model.state_dict(), model_path)
            self.logger.info(f"模型已保存到: {model_path}")
        
        return {
            "status": "completed",
            "epochs_trained": len(self.training_history),
            "best_val_loss": best_val_loss,
            "total_training_time": total_training_time,
            "model_path": str(model_path) if save_best else None
        }
    
    def _train_custom_teacher(self, 
                            teacher: BaseTeacher,
                            train_loader: DataLoader,
                            val_loader: DataLoader,
                            epochs: int) -> Dict[str, Any]:
        """训练自定义Teacher（如SVD等）"""
        self.logger.info(f"使用自定义训练方法训练 {teacher.__class__.__name__}")
        
        # 转换数据格式
        train_data = []
        for features, targets in train_loader:
            for i in range(len(features)):
                user_id = int(features[i][0].item())
                item_id = int(features[i][1].item())
                rating = float(targets[i].item())
                train_data.append((user_id, item_id, rating))
        
        # 调用Teacher的训练方法
        training_start_time = time.time()
        
        if hasattr(teacher, 'fit'):
            teacher.fit(train_data)
        else:
            self.logger.warning(f"{teacher.__class__.__name__} 没有fit方法")
            return {"status": "failed", "reason": "no fit method"}
        
        total_training_time = time.time() - training_start_time
        
        # 验证
        val_predictions = []
        val_targets = []
        
        for features, targets in val_loader:
            for i in range(len(features)):
                user_id = int(features[i][0].item())
                item_id = int(features[i][1].item())
                rating = float(targets[i].item())
                
                try:
                    pred = teacher.predict_rating(user_id, item_id)
                    val_predictions.append(pred)
                    val_targets.append(rating)
                except:
                    # 如果预测失败，使用全局平均值
                    val_predictions.append(3.5)
                    val_targets.append(rating)
        
        # 计算验证损失
        val_loss = np.mean((np.array(val_predictions) - np.array(val_targets)) ** 2)
        
        return {
            "status": "completed",
            "val_loss": val_loss,
            "total_training_time": total_training_time,
            "num_predictions": len(val_predictions)
        }
    
    def train_multi_teacher_fusion(self,
                                 teachers: List[BaseTeacher],
                                 fusion_config: FusionConfig,
                                 train_loader: DataLoader,
                                 val_loader: DataLoader,
                                 teacher_epochs: int = 50,
                                 fusion_epochs: int = 30) -> Dict[str, Any]:
        """
        训练多Teacher融合系统
        
        Args:
            teachers: Teacher列表
            fusion_config: 融合配置
            train_loader: 训练数据
            val_loader: 验证数据
            teacher_epochs: 单Teacher训练轮数
            fusion_epochs: 融合训练轮数
        """
        self.logger.info("开始多Teacher融合训练...")
        
        results = {
            "teacher_results": {},
            "fusion_result": None,
            "total_time": 0
        }
        
        start_time = time.time()
        
        # 1. 首先训练各个Teacher
        self.logger.info("第1阶段: 训练各个Teacher...")
        for i, teacher in enumerate(teachers):
            teacher_name = teacher.__class__.__name__
            self.logger.info(f"训练Teacher {i+1}/{len(teachers)}: {teacher_name}")
            
            # 重置训练历史
            self.training_history = []
            
            # 训练Teacher
            teacher_result = self.train_single_teacher(
                teacher=teacher,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=teacher_epochs,
                save_best=True
            )
            
            results["teacher_results"][teacher_name] = teacher_result
            self.logger.info(f"{teacher_name} 训练完成: {teacher_result['status']}")
        
        # 2. 训练融合层
        if fusion_config.method == "weighted_average":
            self.logger.info("第2阶段: 优化融合权重...")
            fusion_result = self._train_weighted_fusion(teachers, val_loader, fusion_config)
            results["fusion_result"] = fusion_result
        elif fusion_config.method == "neural_fusion":
            self.logger.info("第2阶段: 训练神经融合层...")
            fusion_result = self._train_neural_fusion(teachers, train_loader, val_loader, fusion_epochs)
            results["fusion_result"] = fusion_result
        
        results["total_time"] = time.time() - start_time
        self.logger.info(f"多Teacher融合训练完成，总耗时: {results['total_time']:.2f}秒")
        
        return results
    
    def _train_weighted_fusion(self, 
                             teachers: List[BaseTeacher],
                             val_loader: DataLoader,
                             fusion_config: FusionConfig) -> Dict[str, Any]:
        """训练加权融合"""
        from scipy.optimize import minimize
        
        # 收集所有Teacher的预测
        teacher_predictions = [[] for _ in teachers]
        true_ratings = []
        
        for features, targets in val_loader:
            for i in range(len(features)):
                user_id = int(features[i][0].item())
                item_id = int(features[i][1].item())
                true_rating = float(targets[i].item())
                
                true_ratings.append(true_rating)
                
                for j, teacher in enumerate(teachers):
                    try:
                        pred = teacher.predict_rating(user_id, item_id)
                        teacher_predictions[j].append(pred)
                    except:
                        teacher_predictions[j].append(3.5)  # 默认值
        
        # 转换为numpy数组
        teacher_predictions = np.array(teacher_predictions).T  # [samples, teachers]
        true_ratings = np.array(true_ratings)
        
        # 定义优化目标函数
        def objective(weights):
            weights = weights / np.sum(weights)  # 归一化
            fusion_pred = np.dot(teacher_predictions, weights)
            return np.mean((fusion_pred - true_ratings) ** 2)
        
        # 优化权重
        initial_weights = np.ones(len(teachers)) / len(teachers)
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in teachers]
        
        result = minimize(objective, initial_weights, 
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        optimal_weights = result.x
        optimal_loss = result.fun
        
        # 更新融合配置
        teacher_names = [teacher.__class__.__name__ for teacher in teachers]
        fusion_config.weights = dict(zip(teacher_names, optimal_weights))
        
        self.logger.info(f"优化后的融合权重: {fusion_config.weights}")
        self.logger.info(f"融合验证损失: {optimal_loss:.4f}")
        
        return {
            "method": "weighted_average",
            "optimal_weights": fusion_config.weights,
            "val_loss": optimal_loss,
            "optimization_success": result.success
        }
    
    def _train_neural_fusion(self,
                           teachers: List[BaseTeacher],
                           train_loader: DataLoader,
                           val_loader: DataLoader,
                           epochs: int) -> Dict[str, Any]:
        """训练神经融合层"""
        # TODO: 实现神经融合层训练
        self.logger.info("神经融合层训练暂未实现，使用加权平均")
        
        # 暂时使用等权重
        teacher_names = [teacher.__class__.__name__ for teacher in teachers]
        equal_weights = {name: 1.0/len(teachers) for name in teacher_names}
        
        return {
            "method": "neural_fusion",
            "weights": equal_weights,
            "val_loss": 0.0,
            "status": "placeholder"
        }
    
    def evaluate_teacher(self,
                        teacher: BaseTeacher,
                        test_loader: DataLoader,
                        k_values: List[int] = None) -> EvaluationResult:
        """
        评估Teacher性能
        
        Args:
            teacher: 要评估的Teacher
            test_loader: 测试数据
            k_values: Top-K值列表
        """
        self.logger.info(f"评估 {teacher.__class__.__name__}...")
        
        # 收集预测和真实值
        y_pred_ratings = []
        y_true_ratings = []
        
        for features, targets in test_loader:
            for i in range(len(features)):
                user_id = int(features[i][0].item())
                item_id = int(features[i][1].item())
                true_rating = float(targets[i].item())
                
                try:
                    pred_rating = teacher.predict_rating(user_id, item_id)
                except:
                    pred_rating = 3.5  # 默认值
                
                y_pred_ratings.append(pred_rating)
                y_true_ratings.append(true_rating)
        
        # 生成推荐列表（简化版）
        y_true_items = [[]] * len(y_true_ratings)  # 简化处理
        y_pred_items = [[]] * len(y_pred_ratings)  # 简化处理
        
        # 评估
        result = self.evaluator.comprehensive_evaluate(
            y_true_ratings=np.array(y_true_ratings),
            y_pred_ratings=np.array(y_pred_ratings),
            y_true_items=y_true_items,
            y_pred_items=y_pred_items,
            ratings_dict={}
        )
        
        self.logger.info(f"{teacher.__class__.__name__} 评估完成: RMSE={result.rmse:.4f}, MAE={result.mae:.4f}")
        return result
    
    def save_training_history(self, filename: str = "training_history.json") -> None:
        """保存训练历史"""
        history_data = [asdict(h) for h in self.training_history]
        
        save_path = self.save_dir / filename
        with open(save_path, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        self.logger.info(f"训练历史已保存到: {save_path}")

if __name__ == "__main__":
    # 测试代码
    import sys
    sys.path.append('/home/coder-gw/7Projects_in_7Days/online-inference-system')
    
    logging.basicConfig(level=logging.INFO)
    
    # 创建配置
    system_config = SystemConfig()
    dataset_config = DatasetConfig()
    
    # 创建训练器
    trainer = ModelTrainer(system_config, dataset_config)
    
    print("ModelTrainer 初始化成功!")
    print(f"使用设备: {trainer.device}")
    print(f"保存目录: {trainer.save_dir}")
