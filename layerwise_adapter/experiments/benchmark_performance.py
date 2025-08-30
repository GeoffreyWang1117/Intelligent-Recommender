"""
Performance benchmark for the LayerwiseAdapter system.
"""

import torch
import time
import logging
import os
import sys
import numpy as np
from typing import Dict, List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.base import ModelConfig, UserProfile, ItemProfile
from models.layerwise_adapter import LayerwiseAdapter
from utils.trainer import create_trainer, RecommendationDataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def benchmark_inference_speed(model: LayerwiseAdapter, 
                             user_profiles: List[UserProfile],
                             item_profiles: List[ItemProfile],
                             num_runs: int = 100) -> Dict[str, float]:
    """Benchmark model inference speed."""
    logger.info(f"Benchmarking inference speed with {num_runs} runs...")
    
    model.eval()
    device = next(model.parameters()).device
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(10):
            _ = model.predict(user_profiles[:1], item_profiles[:1])
    
    # Benchmark single prediction
    single_times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model.predict(user_profiles[:1], item_profiles[:1])
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            single_times.append((end_time - start_time) * 1000)  # ms
    
    # Benchmark batch prediction
    batch_sizes = [1, 5, 10, 20, 50]
    batch_times = {}
    
    for batch_size in batch_sizes:
        if batch_size <= len(user_profiles):
            times = []
            with torch.no_grad():
                for _ in range(20):
                    start_time = time.time()
                    _ = model.predict(
                        user_profiles[:batch_size], 
                        item_profiles[:batch_size]
                    )
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    end_time = time.time()
                    times.append((end_time - start_time) * 1000)  # ms
            
            batch_times[f'batch_{batch_size}'] = np.mean(times)
    
    results = {
        'single_prediction_avg_ms': np.mean(single_times),
        'single_prediction_std_ms': np.std(single_times),
        'single_prediction_min_ms': np.min(single_times),
        'single_prediction_max_ms': np.max(single_times),
        **batch_times
    }
    
    return results


def benchmark_memory_usage(config: ModelConfig) -> Dict[str, float]:
    """Benchmark memory usage of different model components."""
    logger.info("Benchmarking memory usage...")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Measure model memory
        model = LayerwiseAdapter(config)
        model = model.cuda()
        
        model_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        # Measure forward pass memory
        user_profiles = []
        item_profiles = []
        
        for i in range(32):  # Batch size 32
            user_profile = UserProfile(
                user_id=i,
                feature_vector=torch.randn(config.embedding_dim),
                interaction_history=[1, 2, 3]
            )
            user_profiles.append(user_profile)
            
            item_profile = ItemProfile(
                item_id=i,
                feature_vector=torch.randn(config.embedding_dim),
                category=f"cat_{i%3}"
            )
            item_profiles.append(item_profile)
        
        # Forward pass
        with torch.no_grad():
            _ = model(user_profiles, item_profiles)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        current_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        return {
            'model_memory_mb': model_memory,
            'peak_memory_mb': peak_memory,
            'current_memory_mb': current_memory,
            'forward_pass_memory_mb': peak_memory - model_memory
        }
    
    else:
        logger.warning("CUDA not available, skipping memory benchmark")
        return {}


def benchmark_model_sizes() -> Dict[str, Dict[str, float]]:
    """Benchmark different model configurations."""
    logger.info("Benchmarking different model sizes...")
    
    configs = {
        'tiny': {
            'embedding_dim': 32,
            'hidden_dim': 64,
            'num_heads': 4,
            'num_users': 1000,
            'num_items': 500
        },
        'small': {
            'embedding_dim': 64,
            'hidden_dim': 128,
            'num_heads': 8,
            'num_users': 5000,
            'num_items': 2000
        },
        'medium': {
            'embedding_dim': 128,
            'hidden_dim': 256,
            'num_heads': 8,
            'num_users': 10000,
            'num_items': 5000
        },
        'large': {
            'embedding_dim': 256,
            'hidden_dim': 512,
            'num_heads': 16,
            'num_users': 50000,
            'num_items': 20000
        }
    }
    
    results = {}
    
    for size_name, params in configs.items():
        config = ModelConfig()
        for key, value in params.items():
            setattr(config, key, value)
        
        model = LayerwiseAdapter(config)
        size_info = model.get_model_size()
        
        # Create sample data for speed test
        user_profiles = []
        item_profiles = []
        
        for i in range(10):
            user_profile = UserProfile(
                user_id=i,
                feature_vector=torch.randn(config.embedding_dim),
                interaction_history=[1, 2, 3]
            )
            user_profiles.append(user_profile)
            
            item_profile = ItemProfile(
                item_id=i,
                feature_vector=torch.randn(config.embedding_dim),
                category=f"cat_{i%3}"
            )
            item_profiles.append(item_profile)
        
        # Quick speed test
        model.eval()
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = model.predict(user_profiles[:1], item_profiles[:1])
        end_time = time.time()
        
        avg_inference_time = (end_time - start_time) / 10 * 1000  # ms
        
        results[size_name] = {
            **size_info,
            'config_params': params,
            'avg_inference_time_ms': avg_inference_time
        }
    
    return results


def benchmark_training_efficiency():
    """Benchmark training efficiency."""
    logger.info("Benchmarking training efficiency...")
    
    config = ModelConfig()
    config.num_users = 1000
    config.num_items = 500
    config.embedding_dim = 64
    config.hidden_dim = 128
    config.batch_size = 32
    config.max_epochs = 3
    config.learning_rate = 1e-3
    
    # Create synthetic data
    user_profiles = []
    item_profiles = []
    ratings = []
    
    for i in range(config.num_users):
        user_profile = UserProfile(
            user_id=i,
            feature_vector=torch.randn(config.embedding_dim),
            interaction_history=np.random.choice(config.num_items, size=5, replace=False).tolist()
        )
        user_profiles.append(user_profile)
    
    for i in range(config.num_items):
        item_profile = ItemProfile(
            item_id=i,
            feature_vector=torch.randn(config.embedding_dim),
            category=f"category_{i % 10}"
        )
        item_profiles.append(item_profile)
    
    # Create training data
    train_user_profiles = []
    train_item_profiles = []
    train_ratings = []
    
    for _ in range(3000):
        user_idx = np.random.randint(0, len(user_profiles))
        item_idx = np.random.randint(0, len(item_profiles))
        
        # Simple rating generation
        user_feat = user_profiles[user_idx].feature_vector
        item_feat = item_profiles[item_idx].feature_vector
        similarity = torch.cosine_similarity(user_feat, item_feat, dim=0)
        rating = 3.0 + 2.0 * similarity.item() + 0.2 * torch.randn(1).item()
        rating = max(1.0, min(5.0, rating))
        
        train_user_profiles.append(user_profiles[user_idx])
        train_item_profiles.append(item_profiles[item_idx])
        train_ratings.append(rating)
    
    # Create dataset
    dataset = RecommendationDataset(train_user_profiles, train_item_profiles, train_ratings)
    
    # Training benchmark
    trainer = create_trainer(config, use_teachers=False)
    
    start_time = time.time()
    history = trainer.train(dataset, save_dir="./benchmark_checkpoints")
    end_time = time.time()
    
    training_time = end_time - start_time
    final_loss = history['train_loss'][-1]
    
    return {
        'total_training_time_s': training_time,
        'training_time_per_epoch_s': training_time / config.max_epochs,
        'final_train_loss': final_loss,
        'loss_convergence': history['train_loss'][0] - final_loss,
        'dataset_size': len(dataset)
    }


def main():
    """Run comprehensive benchmarks."""
    logger.info("=== LayerwiseAdapter Performance Benchmark ===")
    
    # Standard configuration
    config = ModelConfig()
    config.num_users = 1000
    config.num_items = 500
    config.embedding_dim = 128
    config.hidden_dim = 256
    
    # Create model and sample data
    model = LayerwiseAdapter(config)
    
    user_profiles = []
    item_profiles = []
    
    for i in range(100):
        user_profile = UserProfile(
            user_id=i,
            feature_vector=torch.randn(config.embedding_dim),
            interaction_history=[1, 2, 3]
        )
        user_profiles.append(user_profile)
        
        item_profile = ItemProfile(
            item_id=i,
            feature_vector=torch.randn(config.embedding_dim),
            category=f"cat_{i%3}"
        )
        item_profiles.append(item_profile)
    
    # 1. Model size analysis
    logger.info("\n1. Model Size Analysis")
    size_results = benchmark_model_sizes()
    for size_name, results in size_results.items():
        logger.info(f"{size_name.upper()}:")
        logger.info(f"  Parameters: {results['total_parameters']:,}")
        logger.info(f"  Model Size: {results['model_size_mb']:.2f} MB")
        logger.info(f"  Inference Time: {results['avg_inference_time_ms']:.2f} ms")
    
    # 2. Inference speed benchmark
    logger.info("\n2. Inference Speed Benchmark")
    speed_results = benchmark_inference_speed(model, user_profiles, item_profiles)
    logger.info(f"Single Prediction: {speed_results['single_prediction_avg_ms']:.2f} ± {speed_results['single_prediction_std_ms']:.2f} ms")
    logger.info(f"Min/Max: {speed_results['single_prediction_min_ms']:.2f}/{speed_results['single_prediction_max_ms']:.2f} ms")
    
    for key, value in speed_results.items():
        if key.startswith('batch_'):
            batch_size = key.split('_')[1]
            logger.info(f"Batch {batch_size}: {value:.2f} ms")
    
    # 3. Memory usage benchmark
    logger.info("\n3. Memory Usage Benchmark")
    memory_results = benchmark_memory_usage(config)
    if memory_results:
        logger.info(f"Model Memory: {memory_results['model_memory_mb']:.2f} MB")
        logger.info(f"Peak Memory: {memory_results['peak_memory_mb']:.2f} MB")
        logger.info(f"Forward Pass Memory: {memory_results['forward_pass_memory_mb']:.2f} MB")
    
    # 4. Training efficiency benchmark
    logger.info("\n4. Training Efficiency Benchmark")
    training_results = benchmark_training_efficiency()
    logger.info(f"Training Time: {training_results['total_training_time_s']:.2f} s")
    logger.info(f"Time per Epoch: {training_results['training_time_per_epoch_s']:.2f} s")
    logger.info(f"Final Loss: {training_results['final_train_loss']:.4f}")
    logger.info(f"Loss Improvement: {training_results['loss_convergence']:.4f}")
    
    # 5. Performance summary
    logger.info("\n=== Performance Summary ===")
    
    # Check if meets target requirements
    target_inference_time = 100.0  # ms
    target_model_size = 10.0  # MB
    
    meets_speed = speed_results['single_prediction_avg_ms'] < target_inference_time
    meets_size = size_results['medium']['model_size_mb'] < target_model_size
    
    logger.info(f"Target: <{target_inference_time}ms inference, <{target_model_size}MB size")
    logger.info(f"✅ Speed requirement: {'PASS' if meets_speed else 'FAIL'}")
    logger.info(f"✅ Size requirement: {'PASS' if meets_size else 'FAIL'}")
    
    if meets_speed and meets_size:
        logger.info("🎉 All performance targets met!")
    else:
        logger.info("⚠️  Some performance targets not met - consider optimization")
    
    # Save results
    import json
    os.makedirs("./benchmark_results", exist_ok=True)
    
    all_results = {
        'model_sizes': size_results,
        'inference_speed': speed_results,
        'memory_usage': memory_results,
        'training_efficiency': training_results,
        'performance_summary': {
            'meets_speed_target': bool(meets_speed),
            'meets_size_target': bool(meets_size),
            'target_inference_time_ms': target_inference_time,
            'target_model_size_mb': target_model_size
        }
    }
    
    with open('./benchmark_results/performance_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info("Results saved to ./benchmark_results/performance_results.json")


if __name__ == "__main__":
    main()
