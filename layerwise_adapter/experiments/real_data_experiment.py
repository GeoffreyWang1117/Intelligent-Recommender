"""
Real data experiment with MovieLens dataset for LayerwiseAdapter.
"""

import torch
import numpy as np
import logging
import os
import sys
import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.base import ModelConfig
from models.layerwise_adapter import LayerwiseAdapter
from utils.trainer import RecommendationDataset, LayerwiseTrainer, create_trainer
from utils.data_processor import MovieLensDataProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealDataExperiment:
    """真实数据实验管理器"""
    
    def __init__(self, data_path: str, experiment_name: str = "movielens_real_experiment"):
        self.data_path = data_path
        self.experiment_name = experiment_name
        self.results_dir = f"./real_experiment_results/{experiment_name}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 实验配置
        self.config = self._create_experiment_config()
        
        # 数据处理器
        self.data_processor = MovieLensDataProcessor(data_path, self.config.embedding_dim)
        
        # 结果存储
        self.experiment_results = {}
        
        logger.info(f"RealDataExperiment initialized: {experiment_name}")
    
    def _create_experiment_config(self) -> ModelConfig:
        """创建实验配置"""
        config = ModelConfig()
        
        # 模型架构配置 - 适合small数据集的轻量化配置
        config.embedding_dim = 64
        config.hidden_dim = 128
        config.num_heads = 4
        config.num_layers = 2
        config.dropout = 0.1
        
        # 训练配置 - small数据集友好
        config.batch_size = 128
        config.learning_rate = 1e-3
        config.max_epochs = 10
        config.patience = 3
        config.warmup_steps = 100
        
        # 蒸馏配置
        config.alpha_ensemble = 0.3
        config.beta_llm = 0.3
        config.gamma_task = 0.4
        config.temperature = 3.0
        
        return config
    
    def load_and_process_data(self) -> None:
        """加载和处理数据"""
        logger.info("Loading and processing MovieLens data...")
        
        start_time = time.time()
        
        # 加载数据
        self.data_processor.load_data()
        
        # 创建用户和物品profile
        self.user_profiles = self.data_processor.create_user_profiles()
        self.item_profiles = self.data_processor.create_item_profiles()
        
        # 创建训练数据
        (self.train_users, self.train_items, self.train_ratings,
         self.test_users, self.test_items, self.test_ratings) = \
            self.data_processor.create_training_data(self.user_profiles, self.item_profiles)
        
        processing_time = time.time() - start_time
        
        # 更新配置
        self.config.num_users = len(self.user_profiles)
        self.config.num_items = len(self.item_profiles)
        
        # 记录数据统计
        data_stats = {
            'num_users': len(self.user_profiles),
            'num_items': len(self.item_profiles),
            'num_train_interactions': len(self.train_users),
            'num_test_interactions': len(self.test_users),
            'processing_time_s': processing_time,
            'data_sparsity': len(self.train_users) / (len(self.user_profiles) * len(self.item_profiles))
        }
        
        self.experiment_results['data_stats'] = data_stats
        
        logger.info(f"Data processed in {processing_time:.2f}s")
        logger.info(f"Train: {len(self.train_users)} interactions")
        logger.info(f"Test: {len(self.test_users)} interactions")
        logger.info(f"Sparsity: {data_stats['data_sparsity']:.6f}")
    
    def run_baseline_experiment(self) -> Dict:
        """运行基线实验（无Teacher蒸馏）"""
        logger.info("Running baseline experiment (no teacher distillation)...")
        
        # 创建数据集
        train_dataset = RecommendationDataset(
            self.train_users, self.train_items, self.train_ratings
        )
        test_dataset = RecommendationDataset(
            self.test_users, self.test_items, self.test_ratings
        )
        
        # 创建trainer (无Teacher)
        trainer = create_trainer(self.config, use_teachers=False)
        trainer.setup_experiment_logging(
            f"{self.experiment_name}_baseline", 
            f"{self.results_dir}/baseline_logs"
        )
        
        # 训练
        start_time = time.time()
        history = trainer.train(
            train_dataset=train_dataset,
            val_dataset=test_dataset,
            save_dir=f"{self.results_dir}/baseline_checkpoints"
        )
        training_time = time.time() - start_time
        
        # 评估
        test_metrics = trainer.evaluate(test_dataset)
        
        baseline_results = {
            'training_time_s': training_time,
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1],
            'test_metrics': test_metrics,
            'training_history': history
        }
        
        logger.info(f"Baseline - Train Loss: {baseline_results['final_train_loss']:.4f}")
        logger.info(f"Baseline - Test RMSE: {test_metrics['rmse']:.4f}")
        logger.info(f"Baseline - Inference Time: {test_metrics['avg_inference_time_ms']:.2f}ms")
        
        return baseline_results
    
    def run_teacher_distillation_experiment(self) -> Dict:
        """运行Teacher蒸馏实验"""
        logger.info("Running teacher distillation experiment...")
        
        # 创建数据集
        train_dataset = RecommendationDataset(
            self.train_users, self.train_items, self.train_ratings
        )
        test_dataset = RecommendationDataset(
            self.test_users, self.test_items, self.test_ratings
        )
        
        # 创建trainer (使用Teacher)
        trainer = create_trainer(self.config, use_teachers=True)
        trainer.setup_experiment_logging(
            f"{self.experiment_name}_distillation", 
            f"{self.results_dir}/distillation_logs"
        )
        
        # 训练
        start_time = time.time()
        history = trainer.train(
            train_dataset=train_dataset,
            val_dataset=test_dataset,
            save_dir=f"{self.results_dir}/distillation_checkpoints"
        )
        training_time = time.time() - start_time
        
        # 评估
        test_metrics = trainer.evaluate(test_dataset)
        
        distillation_results = {
            'training_time_s': training_time,
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1],
            'test_metrics': test_metrics,
            'training_history': history
        }
        
        logger.info(f"Distillation - Train Loss: {distillation_results['final_train_loss']:.4f}")
        logger.info(f"Distillation - Test RMSE: {test_metrics['rmse']:.4f}")
        logger.info(f"Distillation - Inference Time: {test_metrics['avg_inference_time_ms']:.2f}ms")
        
        return distillation_results
    
    def run_model_size_ablation(self) -> Dict:
        """运行模型大小消融研究"""
        logger.info("Running model size ablation study...")
        
        size_configs = {
            'tiny': {'embedding_dim': 32, 'hidden_dim': 64, 'num_heads': 4},
            'small': {'embedding_dim': 64, 'hidden_dim': 128, 'num_heads': 8},
            'medium': {'embedding_dim': 128, 'hidden_dim': 256, 'num_heads': 8},
            'large': {'embedding_dim': 256, 'hidden_dim': 512, 'num_heads': 16}
        }
        
        ablation_results = {}
        
        for size_name, size_params in size_configs.items():
            logger.info(f"Testing {size_name} configuration...")
            
            # 创建配置
            config = ModelConfig()
            for key, value in vars(self.config).items():
                setattr(config, key, value)
            
            for key, value in size_params.items():
                setattr(config, key, value)
            
            config.num_users = self.config.num_users
            config.num_items = self.config.num_items
            config.max_epochs = 5  # 快速测试
            
            # 创建model
            model = LayerwiseAdapter(config)
            size_info = model.get_model_size()
            
            # 快速训练测试
            try:
                # 创建小数据集进行快速测试
                n_samples = min(1000, len(self.train_users))
                mini_dataset = RecommendationDataset(
                    self.train_users[:n_samples],
                    self.train_items[:n_samples],
                    self.train_ratings[:n_samples]
                )
                
                trainer = LayerwiseTrainer(config, model, device='cuda' if torch.cuda.is_available() else 'cpu')
                
                start_time = time.time()
                history = trainer.train(mini_dataset, save_dir=f"{self.results_dir}/ablation_{size_name}")
                training_time = time.time() - start_time
                
                # 推理速度测试
                model.eval()
                inference_times = []
                with torch.no_grad():
                    for _ in range(10):
                        start = time.time()
                        _ = model.predict([self.train_users[0]], [self.train_items[0]])
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        inference_times.append((time.time() - start) * 1000)
                
                ablation_results[size_name] = {
                    'config': size_params,
                    'model_size': size_info,
                    'training_time_s': training_time,
                    'final_loss': history['train_loss'][-1],
                    'avg_inference_time_ms': np.mean(inference_times)
                }
                
            except Exception as e:
                logger.error(f"Error in {size_name} configuration: {e}")
                ablation_results[size_name] = {
                    'config': size_params,
                    'model_size': size_info,
                    'error': str(e)
                }
        
        return ablation_results
    
    def analyze_user_behavior_patterns(self) -> Dict:
        """分析用户行为模式"""
        logger.info("Analyzing user behavior patterns...")
        
        # 用户活跃度分析
        user_activity = {}
        for user_id, profile in self.user_profiles.items():
            metadata = profile.metadata if profile.metadata else {}
            activity_level = metadata.get('activity_level', 'medium')
            if activity_level not in user_activity:
                user_activity[activity_level] = 0
            user_activity[activity_level] += 1
        
        # 评分分布分析
        all_ratings = [float(r) for r in self.train_ratings]
        rating_dist = {
            'mean': np.mean(all_ratings),
            'std': np.std(all_ratings),
            'min': np.min(all_ratings),
            'max': np.max(all_ratings),
            'unique_ratings': len(set(all_ratings))
        }
        
        # 物品流行度分析
        item_popularity = {}
        for item in self.train_items:
            item_id = item.item_id
            if item_id not in item_popularity:
                item_popularity[item_id] = 0
            item_popularity[item_id] += 1
        
        popularity_stats = {
            'most_popular_items': sorted(item_popularity.items(), key=lambda x: x[1], reverse=True)[:10],
            'avg_interactions_per_item': np.mean(list(item_popularity.values())),
            'items_with_single_interaction': sum(1 for count in item_popularity.values() if count == 1)
        }
        
        behavior_analysis = {
            'user_activity_distribution': user_activity,
            'rating_distribution': rating_dist,
            'item_popularity': popularity_stats,
            'total_interactions': len(self.train_ratings),
            'average_ratings_per_user': len(self.train_ratings) / len(self.user_profiles)
        }
        
        return behavior_analysis
    
    def visualize_results(self) -> None:
        """可视化实验结果"""
        logger.info("Creating visualizations...")
        
        # 设置绘图风格
        plt.style.use('seaborn-v0_8')
        
        # 1. 训练曲线对比
        if 'baseline_results' in self.experiment_results and 'distillation_results' in self.experiment_results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 训练损失
            baseline_loss = self.experiment_results['baseline_results']['training_history']['train_loss']
            distill_loss = self.experiment_results['distillation_results']['training_history']['train_loss']
            
            ax1.plot(baseline_loss, label='Baseline', linewidth=2)
            ax1.plot(distill_loss, label='Teacher Distillation', linewidth=2)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Training Loss')
            ax1.set_title('Training Loss Comparison')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 验证损失
            baseline_val = self.experiment_results['baseline_results']['training_history']['val_loss']
            distill_val = self.experiment_results['distillation_results']['training_history']['val_loss']
            
            ax2.plot(baseline_val, label='Baseline', linewidth=2)
            ax2.plot(distill_val, label='Teacher Distillation', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Validation Loss')
            ax2.set_title('Validation Loss Comparison')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{self.results_dir}/training_curves.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. 模型大小对比
        if 'ablation_results' in self.experiment_results:
            ablation = self.experiment_results['ablation_results']
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 参数量 vs 性能
            sizes = []
            params = []
            inference_times = []
            
            for size_name, results in ablation.items():
                if 'error' not in results:
                    sizes.append(size_name)
                    params.append(results['model_size']['total_parameters'])
                    inference_times.append(results['avg_inference_time_ms'])
            
            if sizes:
                ax1.bar(sizes, params, alpha=0.7)
                ax1.set_ylabel('Parameters Count')
                ax1.set_title('Model Parameters by Size')
                ax1.tick_params(axis='x', rotation=45)
                
                ax2.bar(sizes, inference_times, alpha=0.7, color='orange')
                ax2.set_ylabel('Inference Time (ms)')
                ax2.set_title('Inference Time by Model Size')
                ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(f"{self.results_dir}/model_size_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. 用户行为分析
        if 'behavior_analysis' in self.experiment_results:
            behavior = self.experiment_results['behavior_analysis']
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # 用户活跃度分布
            activity_dist = behavior['user_activity_distribution']
            ax1.pie(activity_dist.values(), labels=activity_dist.keys(), autopct='%1.1f%%')
            ax1.set_title('User Activity Distribution')
            
            # 评分分布
            ratings = [float(r) for r in self.train_ratings]
            ax2.hist(ratings, bins=20, alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Rating')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Rating Distribution')
            
            # 物品流行度
            popularity = [count for _, count in behavior['item_popularity']['most_popular_items']]
            items = [f"Item {item_id}" for item_id, _ in behavior['item_popularity']['most_popular_items']]
            
            ax3.barh(items, popularity)
            ax3.set_xlabel('Interaction Count')
            ax3.set_title('Top 10 Most Popular Items')
            
            # 数据稀疏性
            sparsity_data = {
                'Dense': self.experiment_results['data_stats']['data_sparsity'] * 100,
                'Sparse': (1 - self.experiment_results['data_stats']['data_sparsity']) * 100
            }
            ax4.pie(sparsity_data.values(), labels=sparsity_data.keys(), autopct='%1.4f%%')
            ax4.set_title('Data Sparsity')
            
            plt.tight_layout()
            plt.savefig(f"{self.results_dir}/behavior_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Visualizations saved to {self.results_dir}")
    
    def run_complete_experiment(self) -> Dict:
        """运行完整实验"""
        logger.info("="*50)
        logger.info("STARTING COMPLETE REAL DATA EXPERIMENT")
        logger.info("="*50)
        
        experiment_start_time = time.time()
        
        # 1. 数据加载和处理
        self.load_and_process_data()
        
        # 2. 用户行为分析
        self.experiment_results['behavior_analysis'] = self.analyze_user_behavior_patterns()
        
        # 3. 基线实验
        self.experiment_results['baseline_results'] = self.run_baseline_experiment()
        
        # 4. Teacher蒸馏实验
        try:
            self.experiment_results['distillation_results'] = self.run_teacher_distillation_experiment()
        except Exception as e:
            logger.error(f"Teacher distillation experiment failed: {e}")
            self.experiment_results['distillation_error'] = str(e)
        
        # 5. 模型大小消融研究
        self.experiment_results['ablation_results'] = self.run_model_size_ablation()
        
        # 6. 实验总结
        total_time = time.time() - experiment_start_time
        self.experiment_results['experiment_summary'] = {
            'total_experiment_time_s': total_time,
            'experiment_date': datetime.now().isoformat(),
            'config': self.config.to_dict()
        }
        
        # 7. 可视化
        self.visualize_results()
        
        # 8. 保存结果
        with open(f"{self.results_dir}/complete_results.json", "w") as f:
            # 处理不可序列化的对象
            serializable_results = self._make_serializable(self.experiment_results)
            json.dump(serializable_results, f, indent=2)
        
        # 9. 生成报告
        self._generate_experiment_report()
        
        logger.info("="*50)
        logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
        logger.info(f"Results saved to: {self.results_dir}")
        logger.info("="*50)
        
        return self.experiment_results
    
    def _make_serializable(self, obj):
        """将对象转换为JSON可序列化格式"""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj
    
    def _generate_experiment_report(self) -> None:
        """生成实验报告"""
        report_path = f"{self.results_dir}/EXPERIMENT_REPORT.md"
        
        with open(report_path, "w") as f:
            f.write("# MovieLens Real Data Experiment Report\n\n")
            f.write(f"**Experiment Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 数据统计
            if 'data_stats' in self.experiment_results:
                stats = self.experiment_results['data_stats']
                f.write("## 📊 Dataset Statistics\n\n")
                f.write(f"- **Users**: {stats['num_users']:,}\n")
                f.write(f"- **Items**: {stats['num_items']:,}\n")
                f.write(f"- **Training Interactions**: {stats['num_train_interactions']:,}\n")
                f.write(f"- **Test Interactions**: {stats['num_test_interactions']:,}\n")
                f.write(f"- **Data Sparsity**: {stats['data_sparsity']:.6f}\n")
                f.write(f"- **Processing Time**: {stats['processing_time_s']:.2f}s\n\n")
            
            # 实验结果对比
            f.write("## 🔬 Experimental Results\n\n")
            
            if 'baseline_results' in self.experiment_results:
                baseline = self.experiment_results['baseline_results']
                f.write("### Baseline Model (No Teacher Distillation)\n")
                f.write(f"- **Final Train Loss**: {baseline['final_train_loss']:.4f}\n")
                f.write(f"- **Final Val Loss**: {baseline['final_val_loss']:.4f}\n")
                f.write(f"- **Test RMSE**: {baseline['test_metrics']['rmse']:.4f}\n")
                f.write(f"- **Inference Time**: {baseline['test_metrics']['avg_inference_time_ms']:.2f}ms\n\n")
            
            if 'distillation_results' in self.experiment_results:
                distill = self.experiment_results['distillation_results']
                f.write("### Teacher Distillation Model\n")
                f.write(f"- **Final Train Loss**: {distill['final_train_loss']:.4f}\n")
                f.write(f"- **Final Val Loss**: {distill['final_val_loss']:.4f}\n")
                f.write(f"- **Test RMSE**: {distill['test_metrics']['rmse']:.4f}\n")
                f.write(f"- **Inference Time**: {distill['test_metrics']['avg_inference_time_ms']:.2f}ms\n\n")
            
            # 模型大小分析
            if 'ablation_results' in self.experiment_results:
                f.write("## 🏗️ Model Size Ablation\n\n")
                ablation = self.experiment_results['ablation_results']
                for size_name, results in ablation.items():
                    if 'error' not in results:
                        f.write(f"### {size_name.upper()} Configuration\n")
                        f.write(f"- **Parameters**: {results['model_size']['total_parameters']:,}\n")
                        f.write(f"- **Model Size**: {results['model_size']['model_size_mb']:.2f}MB\n")
                        f.write(f"- **Inference Time**: {results['avg_inference_time_ms']:.2f}ms\n\n")
            
            f.write("## 📈 Visualizations\n\n")
            f.write("- `training_curves.png`: Training and validation loss comparison\n")
            f.write("- `model_size_analysis.png`: Model size vs performance analysis\n")
            f.write("- `behavior_analysis.png`: User behavior and data distribution analysis\n\n")
            
            f.write("## 📁 Files Generated\n\n")
            f.write("- `complete_results.json`: Detailed experimental results\n")
            f.write("- `baseline_checkpoints/`: Baseline model checkpoints\n")
            f.write("- `distillation_checkpoints/`: Teacher distillation model checkpoints\n")
            f.write("- `*_logs/`: Training logs and metrics\n")
        
        logger.info(f"Experiment report generated: {report_path}")


def main():
    """运行完整的真实数据实验"""
    # 使用MoveLens small数据集
    data_path = "/home/coder-gw/7Projects_in_7Days/online-inference-system/data/movielens/small"
    
    experiment = RealDataExperiment(
        data_path=data_path,
        experiment_name=f"movielens_small_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    results = experiment.run_complete_experiment()
    
    # 打印关键结果
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    if 'data_stats' in results:
        stats = results['data_stats']
        print(f"📊 Dataset: {stats['num_users']:,} users, {stats['num_items']:,} items")
        print(f"📊 Interactions: {stats['num_train_interactions']:,} train, {stats['num_test_interactions']:,} test")
    
    if 'baseline_results' in results:
        baseline = results['baseline_results']
        print(f"🎯 Baseline RMSE: {baseline['test_metrics']['rmse']:.4f}")
        print(f"⚡ Baseline Inference: {baseline['test_metrics']['avg_inference_time_ms']:.2f}ms")
    
    if 'distillation_results' in results:
        distill = results['distillation_results']
        print(f"🎓 Distillation RMSE: {distill['test_metrics']['rmse']:.4f}")
        print(f"⚡ Distillation Inference: {distill['test_metrics']['avg_inference_time_ms']:.2f}ms")
    
    print(f"📁 Results saved to: {experiment.results_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
