"""
Simplified real data experiment using MovieLens small dataset with fixed CUDA indexing.
"""

import torch
import numpy as np
import pandas as pd
import logging
import os
import sys
import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import json
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.base import ModelConfig, UserProfile, ItemProfile
from models.layerwise_adapter import LayerwiseAdapter
from utils.trainer import RecommendationDataset, LayerwiseTrainer, create_trainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleMovieLensProcessor:
    """简化的MovieLens数据处理器，专门处理small数据集"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.user_id_map = {}  # 原始ID -> 连续ID
        self.movie_id_map = {}  # 原始ID -> 连续ID
        
    def load_and_process(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """加载并处理数据"""
        logger.info(f"Loading MovieLens small dataset from {self.data_path}")
        
        # 加载数据
        ratings_df = pd.read_csv(self.data_path / "ratings.csv")
        movies_df = pd.read_csv(self.data_path / "movies.csv")
        
        logger.info(f"Loaded {len(ratings_df)} ratings and {len(movies_df)} movies")
        
        # 创建连续的用户和电影ID映射 (解决CUDA索引问题)
        unique_users = sorted(ratings_df['userId'].unique())
        unique_movies = sorted(ratings_df['movieId'].unique())
        
        self.user_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_users)}
        self.movie_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_movies)}
        
        # 应用映射
        ratings_df['userId'] = ratings_df['userId'].map(self.user_id_map)
        ratings_df['movieId'] = ratings_df['movieId'].map(self.movie_id_map)
        movies_df['movieId'] = movies_df['movieId'].map(self.movie_id_map).fillna(-1).astype(int)
        
        # 过滤无效电影
        movies_df = movies_df[movies_df['movieId'] >= 0]
        
        # 创建用户信息
        users_df = pd.DataFrame({
            'userId': list(range(len(unique_users))),
            'age': np.random.randint(18, 65, len(unique_users)),
            'gender': np.random.choice(['M', 'F'], len(unique_users))
        })
        
        self.num_users = len(unique_users)
        self.num_movies = len(unique_movies)
        
        logger.info(f"Mapped to {self.num_users} users (0-{self.num_users-1}) and "
                   f"{self.num_movies} movies (0-{self.num_movies-1})")
        
        return ratings_df, movies_df, users_df
    
    def create_profiles(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame, 
                       users_df: pd.DataFrame, embedding_dim: int = 64) -> Tuple[List, List]:
        """创建用户和物品profile"""
        logger.info("Creating user and item profiles...")
        
        # 创建用户profiles
        user_profiles = []
        for _, user in users_df.iterrows():
            user_ratings = ratings_df[ratings_df['userId'] == user['userId']]
            avg_rating = user_ratings['rating'].mean() if len(user_ratings) > 0 else 3.0
            
            features = torch.randn(embedding_dim).float()
            features[0] = user['age'] / 65.0  # 归一化年龄
            features[1] = 1.0 if user['gender'] == 'M' else 0.0
            features[2] = avg_rating / 5.0  # 归一化评分
            
            profile = UserProfile(
                user_id=int(user['userId']),
                feature_vector=features,
                interaction_history=[],
                metadata={'age': user['age'], 'gender': user['gender']}
            )
            user_profiles.append(profile)
        
        # 创建物品profiles
        item_profiles = []
        for _, movie in movies_df.iterrows():
            movie_ratings = ratings_df[ratings_df['movieId'] == movie['movieId']]
            avg_rating = movie_ratings['rating'].mean() if len(movie_ratings) > 0 else 3.0
            
            features = torch.randn(embedding_dim).float()
            features[0] = avg_rating / 5.0  # 归一化评分
            features[1] = len(movie_ratings) / 100.0  # 流行度
            
            profile = ItemProfile(
                item_id=int(movie['movieId']),
                feature_vector=features,
                category=movie['genres'],
                metadata={'title': movie['title'], 'genres': movie['genres']}
            )
            item_profiles.append(profile)
        
        logger.info(f"Created {len(user_profiles)} user profiles and {len(item_profiles)} item profiles")
        return user_profiles, item_profiles
    
    def create_training_data(self, ratings_df: pd.DataFrame, user_profiles: List, 
                           item_profiles: List) -> Tuple[List, List, List, List, List, List]:
        """创建训练数据"""
        logger.info("Creating training data...")
        
        # 创建profile字典用于快速查找
        user_dict = {p.user_id: p for p in user_profiles}
        item_dict = {p.item_id: p for p in item_profiles}
        
        # 准备数据
        users, items, ratings = [], [], []
        
        for _, row in ratings_df.iterrows():
            user_id = int(row['userId'])
            item_id = int(row['movieId'])
            rating = float(row['rating'])
            
            if user_id in user_dict and item_id in item_dict:
                users.append(user_dict[user_id])
                items.append(item_dict[item_id])
                ratings.append(rating)
        
        # 训练/测试分割
        from sklearn.model_selection import train_test_split
        train_users, test_users, train_items, test_items, train_ratings, test_ratings = \
            train_test_split(users, items, ratings, test_size=0.2, random_state=42)
        
        logger.info(f"Created {len(train_users)} training and {len(test_users)} test samples")
        
        return train_users, train_items, train_ratings, test_users, test_items, test_ratings


class MovieLensSmallExperiment:
    """MovieLens Small数据集实验"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.experiment_name = f"movielens_small_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results_dir = f"./small_experiment_results/{self.experiment_name}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 数据处理器
        self.processor = SimpleMovieLensProcessor(data_path)
        
        # 结果存储
        self.results = {}
        
        logger.info(f"MovieLensSmallExperiment initialized: {self.experiment_name}")
    
    def create_config(self, num_users: int, num_items: int) -> ModelConfig:
        """创建模型配置"""
        config = ModelConfig()
        
        # 基础配置
        config.num_users = num_users
        config.num_items = num_items
        config.embedding_dim = 64
        config.hidden_dim = 128
        config.num_heads = 4
        config.num_layers = 2
        config.dropout = 0.1
        
        # 训练配置
        config.batch_size = 256
        config.learning_rate = 1e-3
        config.max_epochs = 10
        config.patience = 3
        config.warmup_steps = 100
        
        return config
    
    def run_baseline_experiment(self) -> Dict:
        """运行基线实验"""
        logger.info("Starting baseline experiment...")
        
        # 1. 数据加载和处理
        start_time = time.time()
        ratings_df, movies_df, users_df = self.processor.load_and_process()
        user_profiles, item_profiles = self.processor.create_profiles(ratings_df, movies_df, users_df)
        train_users, train_items, train_ratings, test_users, test_items, test_ratings = \
            self.processor.create_training_data(ratings_df, user_profiles, item_profiles)
        data_time = time.time() - start_time
        
        # 2. 创建配置和模型
        config = self.create_config(self.processor.num_users, self.processor.num_movies)
        model = LayerwiseAdapter(config)
        
        # 3. 创建数据集
        train_dataset = RecommendationDataset(train_users, train_items, train_ratings)
        test_dataset = RecommendationDataset(test_users, test_items, test_ratings)
        
        # 4. 训练
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        trainer = LayerwiseTrainer(config, model, device=device)
        trainer.setup_experiment_logging(
            f"{self.experiment_name}_baseline", 
            f"{self.results_dir}/baseline_logs"
        )
        
        logger.info(f"Training on {device}")
        
        start_time = time.time()
        history = trainer.train(
            train_dataset=train_dataset,
            val_dataset=test_dataset,
            save_dir=f"{self.results_dir}/baseline_checkpoints"
        )
        training_time = time.time() - start_time
        
        # 5. 评估
        test_metrics = trainer.evaluate(test_dataset)
        
        # 6. 收集结果
        results = {
            'data_processing_time_s': data_time,
            'training_time_s': training_time,
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1],
            'test_metrics': test_metrics,
            'training_history': history,
            'dataset_stats': {
                'num_users': self.processor.num_users,
                'num_movies': self.processor.num_movies,
                'num_train_samples': len(train_users),
                'num_test_samples': len(test_users),
                'total_ratings': len(ratings_df)
            },
            'model_stats': model.get_model_size(),
            'device': device
        }
        
        logger.info(f"Baseline completed - RMSE: {test_metrics['rmse']:.4f}, "
                   f"Training time: {training_time:.1f}s")
        
        return results
    
    def run_size_comparison(self) -> Dict:
        """运行模型大小对比实验"""
        logger.info("Running model size comparison...")
        
        # 加载数据（重用）
        ratings_df, movies_df, users_df = self.processor.load_and_process()
        user_profiles, item_profiles = self.processor.create_profiles(ratings_df, movies_df, users_df)
        
        # 创建小样本用于快速测试
        sample_size = min(1000, len(user_profiles))
        train_users = user_profiles[:sample_size]
        train_items = item_profiles[:sample_size]
        train_ratings = [3.5] * sample_size  # 模拟评分
        
        mini_dataset = RecommendationDataset(train_users, train_items, train_ratings)
        
        size_configs = {
            'tiny': {'embedding_dim': 32, 'hidden_dim': 64, 'num_heads': 2, 'num_layers': 1},
            'small': {'embedding_dim': 64, 'hidden_dim': 128, 'num_heads': 4, 'num_layers': 2},
            'medium': {'embedding_dim': 128, 'hidden_dim': 256, 'num_heads': 8, 'num_layers': 3}
        }
        
        comparison_results = {}
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        for size_name, size_params in size_configs.items():
            logger.info(f"Testing {size_name} configuration...")
            
            try:
                # 创建配置
                config = self.create_config(self.processor.num_users, self.processor.num_movies)
                for key, value in size_params.items():
                    setattr(config, key, value)
                config.max_epochs = 2  # 快速测试
                
                # 创建模型
                model = LayerwiseAdapter(config)
                trainer = LayerwiseTrainer(config, model, device=device)
                
                # 快速训练
                start_time = time.time()
                history = trainer.train(mini_dataset, save_dir=f"{self.results_dir}/size_{size_name}")
                training_time = time.time() - start_time
                
                # 推理测试
                model.eval()
                inference_times = []
                with torch.no_grad():
                    for _ in range(10):
                        start = time.time()
                        _ = model.predict([train_users[0]], [train_items[0]])
                        if device == 'cuda':
                            torch.cuda.synchronize()
                        inference_times.append((time.time() - start) * 1000)
                
                comparison_results[size_name] = {
                    'config': size_params,
                    'model_size': model.get_model_size(),
                    'training_time_s': training_time,
                    'final_loss': history['train_loss'][-1],
                    'avg_inference_time_ms': np.mean(inference_times)
                }
                
                logger.info(f"{size_name}: {model.get_model_size()['total_parameters']:,} params, "
                           f"{training_time:.1f}s training, {np.mean(inference_times):.2f}ms inference")
                
            except Exception as e:
                logger.error(f"Error in {size_name} configuration: {e}")
                comparison_results[size_name] = {'error': str(e)}
        
        return comparison_results
    
    def visualize_results(self) -> None:
        """创建可视化"""
        logger.info("Creating visualizations...")
        
        # 训练曲线
        if 'baseline_results' in self.results:
            history = self.results['baseline_results']['training_history']
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 训练损失
            ax1.plot(history['train_loss'], label='Training Loss', linewidth=2)
            ax1.plot(history['val_loss'], label='Validation Loss', linewidth=2)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training Progress')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 模型大小对比
            if 'size_comparison' in self.results:
                sizes = []
                params = []
                inference_times = []
                
                for size_name, result in self.results['size_comparison'].items():
                    if 'error' not in result:
                        sizes.append(size_name)
                        params.append(result['model_size']['total_parameters'])
                        inference_times.append(result['avg_inference_time_ms'])
                
                if sizes:
                    ax2_twin = ax2.twinx()
                    
                    bars1 = ax2.bar([s + ' (Params)' for s in sizes], params, alpha=0.7, color='skyblue')
                    bars2 = ax2_twin.bar([s + ' (Time)' for s in sizes], inference_times, alpha=0.7, color='orange')
                    
                    ax2.set_ylabel('Parameters', color='skyblue')
                    ax2_twin.set_ylabel('Inference Time (ms)', color='orange')
                    ax2.set_title('Model Size vs Performance')
                    ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(f"{self.results_dir}/experiment_results.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Visualizations saved to {self.results_dir}")
    
    def run_complete_experiment(self) -> Dict:
        """运行完整实验"""
        logger.info("="*60)
        logger.info("STARTING MOVIELENS SMALL EXPERIMENT")
        logger.info("="*60)
        
        start_time = time.time()
        
        # 1. 基线实验
        self.results['baseline_results'] = self.run_baseline_experiment()
        
        # 2. 模型大小对比
        self.results['size_comparison'] = self.run_size_comparison()
        
        # 3. 实验总结
        total_time = time.time() - start_time
        self.results['experiment_summary'] = {
            'total_time_s': total_time,
            'experiment_date': datetime.now().isoformat(),
            'dataset': 'MovieLens Small',
            'experiment_name': self.experiment_name
        }
        
        # 4. 可视化
        self.visualize_results()
        
        # 5. 保存结果
        with open(f"{self.results_dir}/results.json", "w") as f:
            serializable_results = self._make_serializable(self.results)
            json.dump(serializable_results, f, indent=2)
        
        # 6. 生成报告
        self._generate_report()
        
        logger.info("="*60)
        logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
        logger.info(f"Results saved to: {self.results_dir}")
        logger.info("="*60)
        
        return self.results
    
    def _make_serializable(self, obj):
        """转换为JSON可序列化格式"""
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
    
    def _generate_report(self) -> None:
        """生成实验报告"""
        report_path = f"{self.results_dir}/EXPERIMENT_REPORT.md"
        
        with open(report_path, "w") as f:
            f.write("# MovieLens Small Dataset Experiment Report\n\n")
            f.write(f"**Experiment Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Dataset**: MovieLens Small (100K ratings)\n\n")
            
            if 'baseline_results' in self.results:
                baseline = self.results['baseline_results']
                
                f.write("## 📊 Dataset Statistics\n\n")
                stats = baseline['dataset_stats']
                f.write(f"- **Users**: {stats['num_users']:,}\n")
                f.write(f"- **Movies**: {stats['num_movies']:,}\n")
                f.write(f"- **Total Ratings**: {stats['total_ratings']:,}\n")
                f.write(f"- **Training Samples**: {stats['num_train_samples']:,}\n")
                f.write(f"- **Test Samples**: {stats['num_test_samples']:,}\n\n")
                
                f.write("## 🎯 Baseline Results\n\n")
                f.write(f"- **Device**: {baseline['device']}\n")
                f.write(f"- **Model Parameters**: {baseline['model_stats']['total_parameters']:,}\n")
                f.write(f"- **Model Size**: {baseline['model_stats']['model_size_mb']:.2f} MB\n")
                f.write(f"- **Data Processing Time**: {baseline['data_processing_time_s']:.1f}s\n")
                f.write(f"- **Training Time**: {baseline['training_time_s']:.1f}s\n")
                f.write(f"- **Final Train Loss**: {baseline['final_train_loss']:.4f}\n")
                f.write(f"- **Final Val Loss**: {baseline['final_val_loss']:.4f}\n")
                f.write(f"- **Test RMSE**: {baseline['test_metrics']['rmse']:.4f}\n")
                f.write(f"- **Inference Time**: {baseline['test_metrics']['avg_inference_time_ms']:.2f}ms\n\n")
            
            if 'size_comparison' in self.results:
                f.write("## ⚖️ Model Size Comparison\n\n")
                for size_name, result in self.results['size_comparison'].items():
                    if 'error' not in result:
                        f.write(f"### {size_name.upper()}\n")
                        f.write(f"- **Parameters**: {result['model_size']['total_parameters']:,}\n")
                        f.write(f"- **Size**: {result['model_size']['model_size_mb']:.2f} MB\n")
                        f.write(f"- **Training Time**: {result['training_time_s']:.1f}s\n")
                        f.write(f"- **Inference Time**: {result['avg_inference_time_ms']:.2f}ms\n\n")
            
            f.write("## 📁 Generated Files\n\n")
            f.write("- `results.json`: Complete experimental results\n")
            f.write("- `experiment_results.png`: Training curves and model comparison\n")
            f.write("- `baseline_checkpoints/`: Model checkpoints\n")
            f.write("- `baseline_logs/`: Training logs\n")
        
        logger.info(f"Report generated: {report_path}")


def main():
    """运行MovieLens Small实验"""
    data_path = "/home/coder-gw/7Projects_in_7Days/online-inference-system/data/movielens/small"
    
    experiment = MovieLensSmallExperiment(data_path)
    results = experiment.run_complete_experiment()
    
    # 打印总结
    print("\n" + "="*70)
    print("MOVIELENS SMALL EXPERIMENT SUMMARY")
    print("="*70)
    
    if 'baseline_results' in results:
        baseline = results['baseline_results']
        stats = baseline['dataset_stats']
        
        print(f"📊 Dataset: {stats['num_users']:,} users, {stats['num_movies']:,} movies")
        print(f"📊 Ratings: {stats['total_ratings']:,} total, "
              f"{stats['num_train_samples']:,} train, {stats['num_test_samples']:,} test")
        print(f"🏗️ Model: {baseline['model_stats']['total_parameters']:,} parameters, "
              f"{baseline['model_stats']['model_size_mb']:.1f}MB")
        print(f"🎯 Performance: RMSE {baseline['test_metrics']['rmse']:.4f}")
        print(f"⚡ Speed: {baseline['test_metrics']['avg_inference_time_ms']:.2f}ms inference")
        print(f"⏱️ Training: {baseline['training_time_s']:.1f}s on {baseline['device']}")
    
    if 'size_comparison' in results:
        print(f"\n🏗️ Model Comparison:")
        for size, result in results['size_comparison'].items():
            if 'error' not in result:
                params = result['model_size']['total_parameters']
                inference = result['avg_inference_time_ms']
                print(f"   {size}: {params:,} params, {inference:.2f}ms")
    
    print(f"\n📁 Results: {experiment.results_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
