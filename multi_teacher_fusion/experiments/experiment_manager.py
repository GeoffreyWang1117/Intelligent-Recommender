"""
Multi-Teacher Fusion System - Experiment Manager
实验管理器，负责端到端的实验流程
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import logging
import json
import time
from datetime import datetime
import traceback

from ..config.base_config import SystemConfig, DatasetConfig
from ..config.teacher_config import EnsembleConfig, FusionConfig
from ..teachers.ensemble_teacher import EnsembleTeacher
from ..utils.data_processor import MovieLensProcessor
from ..utils.trainer import ModelTrainer
from ..utils.evaluator import RecommendationEvaluator, EvaluationResult

class ExperimentManager:
    """实验管理器"""
    
    def __init__(self, 
                 data_path: str,
                 output_dir: str = "./experiment_outputs",
                 experiment_name: str = None):
        """
        初始化实验管理器
        
        Args:
            data_path: MovieLens数据路径
            output_dir: 实验输出目录
            experiment_name: 实验名称
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"experiment_{timestamp}"
        
        self.experiment_name = experiment_name
        self.experiment_dir = self.output_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.experiment_dir / "experiment.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 初始化组件
        self.system_config = SystemConfig()
        self.dataset_config = DatasetConfig()
        self.data_processor = None
        self.trainer = None
        self.evaluator = RecommendationEvaluator()
        
        # 实验结果
        self.experiment_results = {
            "experiment_name": experiment_name,
            "start_time": None,
            "end_time": None,
            "duration": None,
            "status": "initialized",
            "data_stats": None,
            "training_results": {},
            "evaluation_results": {},
            "errors": []
        }
        
        self.logger.info(f"实验管理器初始化完成: {experiment_name}")
        self.logger.info(f"实验目录: {self.experiment_dir}")
    
    def setup_data(self, 
                   min_ratings_per_user: int = 20,
                   min_ratings_per_item: int = 10,
                   test_ratio: float = 0.2) -> bool:
        """
        设置和处理数据
        
        Args:
            min_ratings_per_user: 用户最小评分数
            min_ratings_per_item: 物品最小评分数
            test_ratio: 测试集比例
        """
        try:
            self.logger.info("开始数据处理...")
            
            # 初始化数据处理器
            self.data_processor = MovieLensProcessor(
                data_path=self.data_path,
                min_ratings_per_user=min_ratings_per_user,
                min_ratings_per_item=min_ratings_per_item
            )
            
            # 执行数据处理流程
            train_df, test_df, stats = self.data_processor.process_full_pipeline()
            
            # 进一步划分验证集
            val_size = int(len(train_df) * 0.2)
            val_df = train_df.sample(n=val_size, random_state=42)
            train_df = train_df.drop(val_df.index)
            
            # 保存数据
            data_dir = self.experiment_dir / "data"
            data_dir.mkdir(exist_ok=True)
            
            train_df.to_csv(data_dir / "train.csv", index=False)
            val_df.to_csv(data_dir / "val.csv", index=False)
            test_df.to_csv(data_dir / "test.csv", index=False)
            
            # 保存统计信息
            self.experiment_results["data_stats"] = {
                "num_users": stats.num_users,
                "num_items": stats.num_items,
                "num_ratings": stats.num_ratings,
                "sparsity": stats.sparsity,
                "rating_range": stats.rating_range,
                "avg_rating": stats.avg_rating,
                "train_size": len(train_df),
                "val_size": len(val_df),
                "test_size": len(test_df)
            }
            
            # 存储数据集
            self.train_df = train_df
            self.val_df = val_df
            self.test_df = test_df
            
            self.logger.info(f"数据处理完成:")
            self.logger.info(f"  训练集: {len(train_df)} 样本")
            self.logger.info(f"  验证集: {len(val_df)} 样本")
            self.logger.info(f"  测试集: {len(test_df)} 样本")
            self.logger.info(f"  用户数: {stats.num_users}")
            self.logger.info(f"  物品数: {stats.num_items}")
            self.logger.info(f"  稀疏度: {stats.sparsity:.3f}")
            
            return True
            
        except Exception as e:
            error_msg = f"数据处理失败: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            self.experiment_results["errors"].append(error_msg)
            return False
    
    def create_ensemble_teacher(self) -> EnsembleTeacher:
        """创建Ensemble Teacher"""
        try:
            self.logger.info("创建Ensemble Teacher...")
            
            # 使用已验证的权重配置
            ensemble_config = EnsembleConfig(
                svd_weight=0.5,
                xdeepfm_weight=0.3,
                autoint_weight=0.2,
                num_factors=50,
                num_epochs=100,
                learning_rate=0.01
            )
            
            # 创建EnsembleTeacher
            teacher = EnsembleTeacher(
                ensemble_config=ensemble_config,
                num_users=self.experiment_results["data_stats"]["num_users"],
                num_items=self.experiment_results["data_stats"]["num_items"]
            )
            
            self.logger.info("Ensemble Teacher创建成功")
            return teacher
            
        except Exception as e:
            error_msg = f"Ensemble Teacher创建失败: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            self.experiment_results["errors"].append(error_msg)
            raise
    
    def run_training_experiment(self, 
                              epochs: int = 50,
                              batch_size: int = 256,
                              learning_rate: float = 0.001) -> bool:
        """
        运行训练实验
        
        Args:
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
        """
        try:
            self.logger.info("开始训练实验...")
            
            # 创建训练器
            self.trainer = ModelTrainer(
                system_config=self.system_config,
                dataset_config=self.dataset_config,
                save_dir=str(self.experiment_dir / "models")
            )
            
            # 准备数据
            train_loader, val_loader = self.trainer.prepare_data(
                train_df=self.train_df,
                val_df=self.val_df,
                batch_size=batch_size
            )
            
            # 创建和训练Ensemble Teacher
            ensemble_teacher = self.create_ensemble_teacher()
            
            # 训练
            training_result = self.trainer.train_single_teacher(
                teacher=ensemble_teacher,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs,
                learning_rate=learning_rate,
                save_best=True
            )
            
            # 保存训练结果
            self.experiment_results["training_results"]["ensemble_teacher"] = training_result
            
            # 保存训练历史
            self.trainer.save_training_history("ensemble_training_history.json")
            
            self.logger.info(f"训练实验完成: {training_result['status']}")
            if training_result["status"] == "completed":
                self.logger.info(f"  训练轮数: {training_result['epochs_trained']}")
                self.logger.info(f"  最佳验证损失: {training_result.get('best_val_loss', 'N/A')}")
                self.logger.info(f"  训练时间: {training_result['total_training_time']:.2f}秒")
            
            return training_result["status"] == "completed"
            
        except Exception as e:
            error_msg = f"训练实验失败: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            self.experiment_results["errors"].append(error_msg)
            return False
    
    def run_evaluation_experiment(self) -> bool:
        """运行评估实验"""
        try:
            self.logger.info("开始评估实验...")
            
            if self.trainer is None:
                self.logger.error("训练器未初始化，请先运行训练实验")
                return False
            
            # 准备测试数据
            test_loader, _ = self.trainer.prepare_data(
                train_df=self.test_df,
                val_df=self.test_df,  # 这里只是为了接口兼容
                batch_size=256
            )
            
            # 重新创建Teacher（加载最佳模型）
            ensemble_teacher = self.create_ensemble_teacher()
            
            # 加载最佳模型权重（如果存在）
            model_path = self.experiment_dir / "models" / "EnsembleTeacher_best.pth"
            if model_path.exists() and hasattr(ensemble_teacher, 'model') and ensemble_teacher.model is not None:
                import torch
                ensemble_teacher.model.load_state_dict(torch.load(model_path))
                self.logger.info(f"已加载最佳模型: {model_path}")
            
            # 评估Teacher
            evaluation_result = self.trainer.evaluate_teacher(
                teacher=ensemble_teacher,
                test_loader=test_loader
            )
            
            # 保存评估结果
            self.experiment_results["evaluation_results"]["ensemble_teacher"] = {
                "rmse": evaluation_result.rmse,
                "mae": evaluation_result.mae,
                "precision_at_k": evaluation_result.precision_at_k,
                "recall_at_k": evaluation_result.recall_at_k,
                "ndcg_at_k": evaluation_result.ndcg_at_k,
                "hit_rate_at_k": evaluation_result.hit_rate_at_k,
                "coverage": evaluation_result.coverage,
                "gini_coefficient": evaluation_result.gini_coefficient,
                "novelty": evaluation_result.novelty
            }
            
            # 打印评估报告
            self.evaluator.print_evaluation_report(evaluation_result)
            
            self.logger.info("评估实验完成")
            return True
            
        except Exception as e:
            error_msg = f"评估实验失败: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            self.experiment_results["errors"].append(error_msg)
            return False
    
    def run_complete_experiment(self,
                              min_ratings_per_user: int = 20,
                              min_ratings_per_item: int = 10,
                              epochs: int = 50,
                              batch_size: int = 256,
                              learning_rate: float = 0.001) -> bool:
        """
        运行完整实验流程
        
        Args:
            min_ratings_per_user: 用户最小评分数
            min_ratings_per_item: 物品最小评分数
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
        """
        self.experiment_results["start_time"] = datetime.now().isoformat()
        self.experiment_results["status"] = "running"
        
        try:
            self.logger.info(f"开始完整实验: {self.experiment_name}")
            
            # 1. 数据处理
            if not self.setup_data(min_ratings_per_user, min_ratings_per_item):
                self.experiment_results["status"] = "failed_data_setup"
                return False
            
            # 2. 训练实验
            if not self.run_training_experiment(epochs, batch_size, learning_rate):
                self.experiment_results["status"] = "failed_training"
                return False
            
            # 3. 评估实验
            if not self.run_evaluation_experiment():
                self.experiment_results["status"] = "failed_evaluation"
                return False
            
            # 实验成功完成
            self.experiment_results["status"] = "completed"
            self.experiment_results["end_time"] = datetime.now().isoformat()
            
            # 计算实验时间
            start_time = datetime.fromisoformat(self.experiment_results["start_time"])
            end_time = datetime.fromisoformat(self.experiment_results["end_time"])
            self.experiment_results["duration"] = (end_time - start_time).total_seconds()
            
            self.logger.info(f"完整实验成功完成!")
            self.logger.info(f"实验耗时: {self.experiment_results['duration']:.2f}秒")
            
            # 保存实验结果
            self.save_experiment_results()
            
            return True
            
        except Exception as e:
            error_msg = f"完整实验失败: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            self.experiment_results["errors"].append(error_msg)
            self.experiment_results["status"] = "failed_unknown"
            self.experiment_results["end_time"] = datetime.now().isoformat()
            
            # 即使失败也保存结果
            self.save_experiment_results()
            return False
    
    def save_experiment_results(self) -> None:
        """保存实验结果"""
        results_file = self.experiment_dir / "experiment_results.json"
        
        try:
            with open(results_file, 'w') as f:
                json.dump(self.experiment_results, f, indent=2, default=str)
            
            self.logger.info(f"实验结果已保存到: {results_file}")
            
        except Exception as e:
            self.logger.error(f"保存实验结果失败: {e}")
    
    def print_experiment_summary(self) -> None:
        """打印实验摘要"""
        print("\n" + "="*80)
        print(f"🧪 Multi-Teacher Fusion Experiment Summary")
        print("="*80)
        print(f"实验名称: {self.experiment_results['experiment_name']}")
        print(f"实验状态: {self.experiment_results['status']}")
        
        if self.experiment_results.get('start_time'):
            print(f"开始时间: {self.experiment_results['start_time']}")
        
        if self.experiment_results.get('end_time'):
            print(f"结束时间: {self.experiment_results['end_time']}")
        
        if self.experiment_results.get('duration'):
            print(f"实验耗时: {self.experiment_results['duration']:.2f}秒")
        
        # 数据统计
        if self.experiment_results.get('data_stats'):
            stats = self.experiment_results['data_stats']
            print(f"\n📊 数据统计:")
            print(f"  用户数: {stats['num_users']}")
            print(f"  物品数: {stats['num_items']}")
            print(f"  评分数: {stats['num_ratings']}")
            print(f"  稀疏度: {stats['sparsity']:.3f}")
            print(f"  训练集: {stats['train_size']}")
            print(f"  验证集: {stats['val_size']}")
            print(f"  测试集: {stats['test_size']}")
        
        # 训练结果
        if self.experiment_results.get('training_results'):
            print(f"\n🏋️ 训练结果:")
            for teacher_name, result in self.experiment_results['training_results'].items():
                print(f"  {teacher_name}: {result.get('status', 'unknown')}")
                if result.get('total_training_time'):
                    print(f"    训练时间: {result['total_training_time']:.2f}秒")
        
        # 评估结果
        if self.experiment_results.get('evaluation_results'):
            print(f"\n📈 评估结果:")
            for teacher_name, result in self.experiment_results['evaluation_results'].items():
                print(f"  {teacher_name}:")
                print(f"    RMSE: {result.get('rmse', 'N/A'):.4f}")
                print(f"    MAE: {result.get('mae', 'N/A'):.4f}")
                
                if 'precision_at_k' in result:
                    for k, v in result['precision_at_k'].items():
                        print(f"    Precision@{k}: {v:.4f}")
        
        # 错误信息
        if self.experiment_results.get('errors'):
            print(f"\n❌ 错误信息:")
            for error in self.experiment_results['errors']:
                print(f"  - {error}")
        
        print("="*80)

if __name__ == "__main__":
    # 示例运行
    experiment = ExperimentManager(
        data_path="/home/coder-gw/7Projects_in_7Days/online-inference-system/data/movielens",
        experiment_name="ensemble_baseline_test"
    )
    
    # 运行完整实验
    success = experiment.run_complete_experiment(
        epochs=10,  # 测试用较少轮数
        batch_size=128
    )
    
    # 打印摘要
    experiment.print_experiment_summary()
